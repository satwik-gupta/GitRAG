"""
app/graph/knowledge_graph.py
─────────────────────────────
Temporal Knowledge Graph builder.

For each repository ingestion run the graph builder:
  1. Creates / updates GraphNode rows for every code entity (function, class,
     method) found in the chunks.
  2. Extracts relationships between entities by scanning chunk content for
     call expressions, import statements, inheritance, and FFI patterns.
  3. Creates / updates GraphEdge rows, tagging each edge with the commit SHA
     so the temporal evolution of the graph is fully tracked.

Relationship types
──────────────────
  calls         – function / method invocation
  imports       – import or include statement
  inherits      – class inheritance
  implements    – interface implementation (Java)
  ffi_calls     – cross-language FFI call (ctypes, CGo, JNI)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import GraphRepository
from app.ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


# ── Relationship extractors ────────────────────────────────────────────────


def _extract_python_relations(content: str) -> list[tuple[str, str]]:
    """
    Return list of (relation_type, target_name) for Python source.
    We look for:
      - import / from-import → ("imports", module)
      - class Foo(Bar)       → ("inherits", Bar)
      - ctypes / cffi usage  → ("ffi_calls", lib)
      - function_name(...)   → ("calls", function_name)  [heuristic]
    """
    relations: list[tuple[str, str]] = []

    # Imports
    for m in re.finditer(r"^\s*import\s+([\w.]+)", content, re.MULTILINE):
        relations.append(("imports", m.group(1)))
    for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", content, re.MULTILINE):
        relations.append(("imports", m.group(1)))

    # Inheritance
    for m in re.finditer(r"class\s+\w+\s*\(([^)]+)\)", content):
        for base in m.group(1).split(","):
            base = base.strip()
            if base and base not in ("object", "Exception", "BaseException"):
                relations.append(("inherits", base))

    # FFI
    for m in re.finditer(r"ctypes\.(?:CDLL|cdll\.LoadLibrary|WinDLL)\(['\"]([^'\"]+)['\"]", content):
        relations.append(("ffi_calls", m.group(1)))
    for m in re.finditer(r"ffi\.(?:dlopen|cdef)\(['\"]([^'\"]+)['\"]", content):
        relations.append(("ffi_calls", m.group(1)))

    # Function calls (top-level only — avoids noise from chained calls)
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]+|[a-z_][a-z_0-9]+)\s*\(", content):
        name = m.group(1)
        if name not in ("if", "while", "for", "with", "print", "range", "len",
                        "str", "int", "dict", "list", "tuple", "set", "type",
                        "isinstance", "hasattr", "getattr", "super", "self"):
            relations.append(("calls", name))

    return relations


def _extract_java_relations(content: str) -> list[tuple[str, str]]:
    relations: list[tuple[str, str]] = []

    # Imports
    for m in re.finditer(r"^\s*import\s+([\w.]+);", content, re.MULTILINE):
        relations.append(("imports", m.group(1)))

    # Inheritance
    for m in re.finditer(r"\bextends\s+([\w.]+)", content):
        relations.append(("inherits", m.group(1)))
    for m in re.finditer(r"\bimplements\s+([\w.,\s]+)", content):
        for iface in m.group(1).split(","):
            relations.append(("implements", iface.strip()))

    # JNI (native methods)
    for m in re.finditer(r"\bnative\b.*\b(\w+)\s*\(", content):
        relations.append(("ffi_calls", m.group(1)))

    return relations


def _extract_go_relations(content: str) -> list[tuple[str, str]]:
    relations: list[tuple[str, str]] = []

    # Imports
    for m in re.finditer(r'"([^"]+)"', content):
        path = m.group(1)
        if "/" in path or "." in path:
            relations.append(("imports", path))

    # CGo FFI
    if 'import "C"' in content:
        for m in re.finditer(r"\bC\.(\w+)\s*\(", content):
            relations.append(("ffi_calls", f"C.{m.group(1)}"))

    # Interface implementation is implicit in Go — skip

    return relations


def _extract_cpp_relations(content: str) -> list[tuple[str, str]]:
    relations: list[tuple[str, str]] = []

    # Includes
    for m in re.finditer(r'#include\s+[<"]([^>"]+)[>"]', content):
        relations.append(("imports", m.group(1)))

    # Inheritance
    for m in re.finditer(r":\s*(?:public|protected|private)\s+([\w:]+)", content):
        relations.append(("inherits", m.group(1)))

    return relations


_LANG_EXTRACTORS = {
    "python": _extract_python_relations,
    "java": _extract_java_relations,
    "golang": _extract_go_relations,
    "cpp": _extract_cpp_relations,
}


# ── TemporalKnowledgeGraph ────────────────────────────────────────────────


class TemporalKnowledgeGraph:
    """
    Builds and persists the temporal knowledge graph for a single ingestion run.

    Parameters
    ----------
    session:    AsyncSession scoped to the caller's transaction.
    repo_url:   GitHub URL of the repository.
    commit_sha: HEAD SHA of the cloned commit.
    concurrency:
        Semaphore limit on concurrent DB operations.
    """

    def __init__(
        self,
        session: AsyncSession,
        repo_url: str,
        commit_sha: Optional[str] = None,
        concurrency: int = 16,
    ) -> None:
        self._repo = GraphRepository(session)
        self._repo_url = repo_url
        self._commit_sha = commit_sha
        self._semaphore = asyncio.Semaphore(concurrency)
        self._now = datetime.now(timezone.utc)

    # ── Public ─────────────────────────────────────────────────────────────

    async def build_from_chunks(self, chunks: list[CodeChunk]) -> None:
        """
        Process all *chunks*, upsert their graph nodes and edges.
        """
        logger.info(
            "Building temporal knowledge graph for %s (%d chunks)",
            self._repo_url,
            len(chunks),
        )
        tasks = [self._process_chunk(c) for c in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning(
                "Graph build: %d chunks raised errors (first: %s)",
                len(errors),
                errors[0],
            )
        logger.info("Graph build complete for %s", self._repo_url)

    # ── Private ────────────────────────────────────────────────────────────

    async def _process_chunk(self, chunk: CodeChunk) -> None:
        async with self._semaphore:
            node_key = self._node_key(chunk)
            source_node = await self._repo.get_or_create_node(
                node_key=node_key,
                entity_type=chunk.doc_type,
                name=chunk.qualified_name,
                language=chunk.language,
                repo_url=self._repo_url,
                file_path=chunk.file_path,
                commit_sha=self._commit_sha,
            )

            extractor = _LANG_EXTRACTORS.get(chunk.language)
            if extractor is None:
                return

            relations = extractor(chunk.content)
            for relation_type, target_name in relations:
                await self._upsert_relation(source_node.id, target_name, relation_type)

    async def _upsert_relation(
        self,
        source_id: int,
        target_name: str,
        relation_type: str,
    ) -> None:
        """
        Resolve *target_name* to a node (or create a placeholder) and
        upsert the directed edge.
        """
        target_key = f"{self._repo_url}::__ref__::{target_name}"
        target_node = await self._repo.get_or_create_node(
            node_key=target_key,
            entity_type="reference",
            name=target_name,
            language=None,
            repo_url=self._repo_url,
            file_path="",
            commit_sha=self._commit_sha,
        )
        await self._repo.upsert_edge(
            source_id=source_id,
            target_id=target_node.id,
            relation_type=relation_type,
            commit_sha=self._commit_sha,
            valid_from=self._now,
        )

    @staticmethod
    def _node_key(chunk: CodeChunk) -> str:
        return f"{chunk.repo_url}::{chunk.file_path}::{chunk.qualified_name}"
