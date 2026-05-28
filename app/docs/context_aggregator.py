"""
app/docs/context_aggregator.py
────────────────────────────────
Macro-Context Aggregator: builds a holistic, structured view of an ingested
repository by fusing data from three sources:

  1. PostgreSQL  — IngestionJob records (file tree, language distribution)
                  GraphNode + GraphEdge records (call graph, imports, FFI)
  2. Qdrant      — Class and function signatures with docstrings
                  (metadata-filtered scroll, no vector search needed)

The result is serialised into a :class:`StructuredContext` whose
`to_context_string(keys)` method produces the LLM-ready text block
for any given set of section context keys.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GraphEdge, GraphNode, IngestionJob
from app.db.repositories import IngestionJobRepository
from app.vector.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class ModuleSummary:
    name: str
    entity_type: str       # function | class | method
    language: str
    file_path: str
    docstring: str
    start_line: int


@dataclass
class CallEdge:
    source: str            # qualified_name of source
    relation: str          # calls | imports | inherits | implements | ffi_calls
    target: str            # name of target


@dataclass
class StructuredContext:
    """
    All aggregated repository metadata serialisable to LLM context strings.
    """

    repo_url: str
    repo_name: str
    commit_sha: str
    language_distribution: dict[str, int]
    total_files: int
    entry_points: list[str]
    file_tree: str
    call_graph: list[CallEdge]
    external_dependencies: list[str]
    module_summaries: list[ModuleSummary]

    # ── Serialisation ──────────────────────────────────────────────────────

    _KEY_HANDLERS = None  # populated lazily below

    def to_context_string(self, keys: list[str]) -> str:
        """Render only the context blocks requested by a template section."""
        parts: list[str] = []
        handlers = self._get_key_handlers()
        for key in keys:
            k = key.strip()
            fn = handlers.get(k)
            if fn:
                block = fn(self)
                if block:
                    parts.append(block)
            else:
                logger.debug("Unknown context key %r — skipping", k)
        return "\n\n".join(parts) if parts else "(No context available)"

    def full_context_string(self) -> str:
        """Render ALL context components into a single block."""
        return self.to_context_string(list(self._get_key_handlers().keys()))

    # ── Per-key renderers ──────────────────────────────────────────────────

    def _render_repo_name(self) -> str:
        return f"=== REPOSITORY ===\nName: {self.repo_name}\nURL: {self.repo_url}\nCommit: {self.commit_sha}"

    def _render_repo_url(self) -> str:
        return f"Repository URL: {self.repo_url}"

    def _render_commit_sha(self) -> str:
        return f"Commit SHA: {self.commit_sha}"

    def _render_language_distribution(self) -> str:
        if not self.language_distribution:
            return "=== LANGUAGE DISTRIBUTION ===\n(none)"
        lines = [f"  {lang}: {count} files" for lang, count in
                 sorted(self.language_distribution.items(), key=lambda x: -x[1])]
        return "=== LANGUAGE DISTRIBUTION ===\n" + "\n".join(lines)

    def _render_total_files(self) -> str:
        return f"Total source files indexed: {self.total_files}"

    def _render_entry_points(self) -> str:
        if not self.entry_points:
            return "=== ENTRY POINTS ===\n(none detected)"
        return "=== ENTRY POINTS ===\n" + "\n".join(f"  • {ep}" for ep in self.entry_points[:20])

    def _render_file_tree(self) -> str:
        return f"=== FILE TREE (abridged) ===\n{self.file_tree}"

    def _render_call_graph(self) -> str:
        if not self.call_graph:
            return "=== CALL GRAPH ===\n(no relationships extracted)"
        lines = [f"  {e.source}  --[{e.relation}]-->  {e.target}"
                 for e in self.call_graph[:100]]
        suffix = f"\n  … ({len(self.call_graph) - 100} more)" if len(self.call_graph) > 100 else ""
        return "=== CALL GRAPH ===\n" + "\n".join(lines) + suffix

    def _render_external_dependencies(self) -> str:
        if not self.external_dependencies:
            return "=== EXTERNAL DEPENDENCIES ===\n(none detected)"
        return "=== EXTERNAL DEPENDENCIES ===\n" + "\n".join(
            f"  • {dep}" for dep in sorted(self.external_dependencies)[:60]
        )

    def _render_module_structure(self) -> str:
        by_lang: dict[str, list[ModuleSummary]] = defaultdict(list)
        for ms in self.module_summaries:
            by_lang[ms.language].append(ms)
        parts: list[str] = ["=== MODULE STRUCTURE ==="]
        for lang, summaries in sorted(by_lang.items()):
            parts.append(f"\n[{lang.upper()}]")
            for ms in summaries[:30]:
                doc = f' — "{ms.docstring[:80]}"' if ms.docstring else ""
                parts.append(f"  [{ms.entity_type}] {ms.name} ({ms.file_path}:{ms.start_line}){doc}")
        return "\n".join(parts)

    def _render_module_summaries(self) -> str:
        return self._render_module_structure()

    def _render_class_signatures(self) -> str:
        classes = [m for m in self.module_summaries if m.entity_type == "class"]
        if not classes:
            return "=== CLASS SIGNATURES ===\n(none)"
        lines = ["=== CLASS SIGNATURES ==="]
        for c in classes[:40]:
            doc = f'\n    """{c.docstring[:120]}"""' if c.docstring else ""
            lines.append(f"  class {c.name} ({c.language}) — {c.file_path}{doc}")
        return "\n".join(lines)

    def _render_function_signatures(self) -> str:
        funcs = [m for m in self.module_summaries if m.entity_type in ("function", "method")]
        if not funcs:
            return "=== FUNCTION SIGNATURES ===\n(none)"
        lines = ["=== FUNCTION SIGNATURES ==="]
        for f in funcs[:60]:
            doc = f' — "{f.docstring[:80]}"' if f.docstring else ""
            lines.append(f"  {f.entity_type} {f.name} ({f.language}) — {f.file_path}{doc}")
        return "\n".join(lines)

    @classmethod
    def _get_key_handlers(cls) -> dict:
        if cls._KEY_HANDLERS is None:
            cls._KEY_HANDLERS = {
                "repo_name": cls._render_repo_name,
                "repo_url": cls._render_repo_url,
                "commit_sha": cls._render_commit_sha,
                "language_distribution": cls._render_language_distribution,
                "total_files": cls._render_total_files,
                "entry_points": cls._render_entry_points,
                "file_tree": cls._render_file_tree,
                "call_graph": cls._render_call_graph,
                "external_dependencies": cls._render_external_dependencies,
                "module_structure": cls._render_module_structure,
                "module_summaries": cls._render_module_summaries,
                "class_signatures": cls._render_class_signatures,
                "function_signatures": cls._render_function_signatures,
                "dependencies": cls._render_external_dependencies,
            }
        return cls._KEY_HANDLERS


# ── File tree builder ─────────────────────────────────────────────────────


def _build_file_tree(file_paths: list[str], max_files: int = 80) -> str:
    """Render a sorted list of relative file paths as an indented ASCII tree."""
    paths = sorted(file_paths)[:max_files]
    tree_lines: list[str] = []
    prev_parts: list[str] = []

    for path in paths:
        parts = PurePosixPath(path).parts
        for depth, part in enumerate(parts):
            if depth >= len(prev_parts) or prev_parts[depth] != part:
                indent = "  " * depth
                is_file = depth == len(parts) - 1
                prefix = "├── " if is_file else "├── "
                tree_lines.append(f"{indent}{prefix}{part}")
        prev_parts = list(parts)

    if len(file_paths) > max_files:
        tree_lines.append(f"  … ({len(file_paths) - max_files} more files)")

    return "\n".join(tree_lines) if tree_lines else "(empty)"


# ── Entry-point heuristic ─────────────────────────────────────────────────

_ENTRY_PATTERNS = {
    "main", "run", "start", "create_app", "app", "server",
    "cli", "entrypoint", "entry_point", "handler", "index",
}


def _is_entry_point(name: str, entity_type: str) -> bool:
    return entity_type in ("function",) and name.lower() in _ENTRY_PATTERNS


# ── MacroContextAggregator ────────────────────────────────────────────────


class MacroContextAggregator:
    """
    Assembles a :class:`StructuredContext` for a given repository from
    PostgreSQL graph tables and Qdrant payloads.

    Parameters
    ----------
    session:  Active AsyncSession (caller manages transaction).
    qdrant:   Connected QdrantManager.
    """

    def __init__(self, session: AsyncSession, qdrant: QdrantManager) -> None:
        self._session = session
        self._qdrant = qdrant

    async def aggregate(
        self,
        repo_url: str,
        workflow_id: Optional[int] = None,
    ) -> StructuredContext:
        """
        Fetch all context data and return a :class:`StructuredContext`.

        Runs graph queries and Qdrant scroll concurrently for speed.
        """
        logger.info("Aggregating macro context for %s", repo_url)

        # Run Qdrant calls concurrently, but execute all DB queries sequentially
        # inside a single coroutine to avoid concurrent operations on the
        # same AsyncSession (which raises "provisioning a new connection").
        async def _fetch_db():
            nodes, edges = await self._fetch_graph(repo_url)
            jobs = await self._fetch_ingestion_jobs(workflow_id, repo_url)
            return nodes, edges, jobs

        db_task = asyncio.create_task(_fetch_db())
        qdrant_task = asyncio.create_task(self._fetch_qdrant_summaries(repo_url))

        (nodes, edges, jobs), summaries = await asyncio.gather(db_task, qdrant_task)

        # ── Language distribution from ingestion jobs ──────────────────────
        lang_dist: Counter = Counter()
        file_paths: list[str] = []
        commit_sha = "unknown"
        for job in jobs:
            if job.language:
                lang_dist[job.language] += 1
            file_paths.append(job.file_path)

        # ── Call graph from graph edges ────────────────────────────────────
        node_map = {n.id: n for n in nodes}
        call_graph: list[CallEdge] = []
        external_deps: set[str] = set()

        for edge in edges:
            src_node = node_map.get(edge.source_node_id)
            tgt_node = node_map.get(edge.target_node_id)
            if src_node and tgt_node:
                call_graph.append(CallEdge(
                    source=src_node.name,
                    relation=edge.relation_type,
                    target=tgt_node.name,
                ))
                if edge.relation_type == "imports" and tgt_node.entity_type == "reference":
                    # External import: target name is the module path
                    top_level = tgt_node.name.split(".")[0].split("/")[0]
                    if top_level:
                        external_deps.add(top_level)

        # ── Entry points ───────────────────────────────────────────────────
        entry_points = [
            f"{ms.file_path}::{ms.name}"
            for ms in summaries
            if _is_entry_point(ms.name, ms.entity_type)
        ]

        # Pull workflow commit_sha if available
        if workflow_id:
            from app.db.repositories import WorkflowRepository  # noqa: PLC0415
            wf_repo = WorkflowRepository(self._session)
            wf = await wf_repo.get(workflow_id)
            if wf and wf.commit_sha:
                commit_sha = wf.commit_sha

        repo_name = repo_url.rstrip("/").split("/")[-1]

        ctx = StructuredContext(
            repo_url=repo_url,
            repo_name=repo_name,
            commit_sha=commit_sha,
            language_distribution=dict(lang_dist),
            total_files=len(file_paths),
            entry_points=entry_points,
            file_tree=_build_file_tree(file_paths),
            call_graph=call_graph,
            external_dependencies=sorted(external_deps),
            module_summaries=summaries,
        )

        logger.info(
            "Context aggregated: %d files, %d graph edges, %d summaries",
            len(file_paths),
            len(edges),
            len(summaries),
        )
        return ctx

    # ── Private: graph queries ─────────────────────────────────────────────

    async def _fetch_graph(
        self, repo_url: str
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes_result = await self._session.execute(
            select(GraphNode)
            .where(GraphNode.repo_url == repo_url)
            .limit(2000)
        )
        nodes = list(nodes_result.scalars().all())

        if not nodes:
            return [], []

        node_ids = [n.id for n in nodes]
        # Fetch edges where either endpoint is in this repo
        edges_result = await self._session.execute(
            select(GraphEdge)
            .where(
                GraphEdge.source_node_id.in_(node_ids),
                GraphEdge.is_active == True,  # noqa: E712
            )
            .limit(5000)
        )
        edges = list(edges_result.scalars().all())
        return nodes, edges

    # ── Private: Qdrant payload scroll ────────────────────────────────────

    async def _fetch_qdrant_summaries(self, repo_url: str) -> list[ModuleSummary]:
        """
        Scroll Qdrant for class-level and function entry-point payloads.
        We fetch two batches (classes + functions) concurrently.
        """
        class_filter = self._qdrant.build_filter(
            repo_url=repo_url, doc_type="class"
        )
        func_filter = self._qdrant.build_filter(
            repo_url=repo_url, doc_type="function"
        )

        class_payloads, func_payloads = await asyncio.gather(
            self._qdrant.scroll_chunks_by_filter(class_filter, limit=200),
            self._qdrant.scroll_chunks_by_filter(func_filter, limit=300),
        )

        summaries: list[ModuleSummary] = []
        for payload in class_payloads + func_payloads:
            summaries.append(ModuleSummary(
                name=payload.get("section_name", ""),
                entity_type=payload.get("doc_type", "function"),
                language=payload.get("language", ""),
                file_path=payload.get("file_path", ""),
                docstring=payload.get("docstring", ""),
                start_line=payload.get("start_line", 0),
            ))

        return summaries

    # ── Private: ingestion job query ───────────────────────────────────────

    async def _fetch_ingestion_jobs(
        self,
        workflow_id: Optional[int],
        repo_url: str,
    ) -> list[IngestionJob]:
        if workflow_id:
            repo = IngestionJobRepository(self._session)
            return list(await repo.list_by_workflow(workflow_id))

        # Fallback: find the most recent workflow for this repo and use its jobs
        from app.db.repositories import WorkflowRepository  # noqa: PLC0415
        wf_repo = WorkflowRepository(self._session)
        wf = await wf_repo.get_by_repo(repo_url)
        if wf:
            repo = IngestionJobRepository(self._session)
            return list(await repo.list_by_workflow(wf.id))
        return []
