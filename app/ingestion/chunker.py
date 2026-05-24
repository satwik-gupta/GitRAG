"""
app/ingestion/chunker.py
─────────────────────────
ChunkingEngine: walks a cloned repository, detects languages,
dispatches to AST parsers, and returns a flat list of CodeChunk objects.

CodeChunk carries rich metadata consumed by the embedding worker,
Qdrant upserter, and the temporal knowledge-graph builder.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiofiles

from app.ingestion.ast_parsers import RawEntity, parse_file

logger = logging.getLogger(__name__)

# ── File extension → language mapping ─────────────────────────────────────

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".go": "golang",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
}

# Maximum bytes to read per file (4 MB — avoids loading generated giant files)
MAX_FILE_BYTES = 4 * 1024 * 1024


# ── CodeChunk ──────────────────────────────────────────────────────────────


@dataclass
class CodeChunk:
    """
    A single indexable unit of code.  Every chunk maps 1-to-1 with a
    Qdrant point and (optionally) a graph node.
    """

    # Stable deterministic ID — used as Qdrant point ID
    chunk_id: str

    # Content
    content: str
    docstring: Optional[str]

    # Code metadata
    language: str                   # python | java | golang | cpp
    doc_type: str                   # function | class | method
    section_name: str               # entity name (may include ClassName.method)
    parent_name: Optional[str]      # enclosing class, if a method
    annotations: list[str]

    # Location
    repo_url: str
    file_path: str                  # relative path from repo root
    start_line: int
    end_line: int

    # Provenance
    commit_sha: Optional[str] = None

    # Derived at embedding time
    embedding: Optional[list[float]] = field(default=None, repr=False)
    sparse_indices: Optional[list[int]] = field(default=None, repr=False)
    sparse_values: Optional[list[float]] = field(default=None, repr=False)

    @property
    def qualified_name(self) -> str:
        if self.parent_name:
            return f"{self.parent_name}.{self.section_name}"
        return self.section_name

    @property
    def citation_label(self) -> str:
        return f"{self.file_path}#{self.qualified_name} L{self.start_line}-{self.end_line}"

    def to_qdrant_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "docstring": self.docstring or "",
            "language": self.language,
            "doc_type": self.doc_type,
            "section_name": self.section_name,
            "qualified_name": self.qualified_name,
            "parent_name": self.parent_name or "",
            "annotations": self.annotations,
            "repo_url": self.repo_url,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "commit_sha": self.commit_sha or "",
        }


# ── ChunkID helper ────────────────────────────────────────────────────────


def _make_chunk_id(repo_url: str, file_path: str, section_name: str, start_line: int) -> str:
    """Return a deterministic UUIDv5-style hex string for a chunk."""
    key = f"{repo_url}::{file_path}::{section_name}::{start_line}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    # Convert first 32 hex chars to a valid UUID
    return str(uuid.UUID(digest[:32]))


# ── Fallback: line-based chunking for files with no AST hits ──────────────

_CHUNK_LINES = 60
_OVERLAP_LINES = 10


def _line_chunk(
    source: str,
    file_path: str,
    repo_url: str,
    language: str,
    commit_sha: Optional[str],
) -> list[CodeChunk]:
    """Split *source* into fixed-size overlapping line windows."""
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    step = _CHUNK_LINES - _OVERLAP_LINES
    idx = 0
    block_num = 0

    while idx < len(lines):
        window = lines[idx: idx + _CHUNK_LINES]
        content = "\n".join(window)
        start_line = idx + 1
        end_line = idx + len(window)
        section_name = f"block_{block_num}"
        chunk_id = _make_chunk_id(repo_url, file_path, section_name, start_line)

        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                content=content,
                docstring=None,
                language=language,
                doc_type="block",
                section_name=section_name,
                parent_name=None,
                annotations=[],
                repo_url=repo_url,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                commit_sha=commit_sha,
            )
        )
        idx += step
        block_num += 1

    return chunks


# ── ChunkingEngine ─────────────────────────────────────────────────────────


class ChunkingEngine:
    """
    Walks a local clone directory and produces CodeChunk objects.

    Parameters
    ----------
    repo_url:   Original GitHub URL (stored in chunk metadata).
    commit_sha: HEAD commit of the clone (stored in chunk metadata).
    concurrency:
        Maximum number of files parsed simultaneously.
    """

    def __init__(
        self,
        repo_url: str,
        commit_sha: Optional[str] = None,
        concurrency: int = 8,
    ) -> None:
        self.repo_url = repo_url
        self.commit_sha = commit_sha
        self._semaphore = asyncio.Semaphore(concurrency)

    # ── Public ─────────────────────────────────────────────────────────────

    async def chunk_repository(self, local_root: Path) -> list[CodeChunk]:
        """
        Discover all supported source files under *local_root* and
        return a flat list of CodeChunk objects.
        """
        files = self._discover_files(local_root)
        logger.info(
            "Discovered %d source files in %s", len(files), local_root
        )

        tasks = [
            self._process_file(fp, local_root)
            for fp in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunks: list[CodeChunk] = []
        for fp, result in zip(files, results):
            if isinstance(result, Exception):
                logger.error("Failed to chunk %s: %s", fp, result)
            else:
                chunks.extend(result)

        logger.info("Total chunks extracted: %d", len(chunks))
        return chunks

    # ── Private ────────────────────────────────────────────────────────────

    def _discover_files(self, root: Path) -> list[Path]:
        """Walk the directory tree and return paths with known extensions."""
        found: list[Path] = []
        # Common dirs to skip
        skip_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "vendor", "target", "build", "dist", ".idea", ".vscode",
        }
        for fp in root.rglob("*"):
            if fp.is_file():
                if any(part in skip_dirs for part in fp.parts):
                    continue
                if fp.suffix.lower() in EXTENSION_MAP:
                    found.append(fp)
        return found

    async def _process_file(self, fp: Path, root: Path) -> list[CodeChunk]:
        async with self._semaphore:
            rel_path = str(fp.relative_to(root)).replace("\\", "/")
            language = EXTENSION_MAP[fp.suffix.lower()]

            try:
                async with aiofiles.open(fp, "rb") as fh:
                    raw = await fh.read(MAX_FILE_BYTES)
            except OSError as exc:
                raise RuntimeError(f"Cannot read {fp}: {exc}") from exc

            source = raw.decode("utf-8", errors="replace")
            entities: list[RawEntity] = await asyncio.get_running_loop().run_in_executor(
                None, parse_file, source, language
            )

            if entities:
                return [
                    self._entity_to_chunk(e, rel_path, language)
                    for e in entities
                ]
            # Fall back to line-window chunking
            return _line_chunk(
                source, rel_path, self.repo_url, language, self.commit_sha
            )

    def _entity_to_chunk(
        self,
        entity: RawEntity,
        rel_path: str,
        language: str,
    ) -> CodeChunk:
        section_name = entity.name
        chunk_id = _make_chunk_id(
            self.repo_url, rel_path, section_name, entity.start_line
        )
        return CodeChunk(
            chunk_id=chunk_id,
            content=entity.content,
            docstring=entity.docstring,
            language=language,
            doc_type=entity.entity_type,
            section_name=section_name,
            parent_name=entity.parent_name,
            annotations=entity.annotations,
            repo_url=self.repo_url,
            file_path=rel_path,
            start_line=entity.start_line,
            end_line=entity.end_line,
            commit_sha=self.commit_sha,
        )
