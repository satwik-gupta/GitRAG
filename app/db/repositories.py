"""
app/db/repositories.py
──────────────────────
Thin async repository classes that wrap SQLAlchemy queries.
Each repository owns one or two related models and exposes only the
operations used by higher-level services — no raw SQL in service code.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    CagCache,
    EmbeddingJob,
    GraphEdge,
    GraphNode,
    IngestionJob,
    JobStatus,
    Workflow,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WorkflowRepository
# ---------------------------------------------------------------------------


class WorkflowRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, repo_url: str, branch: str = "HEAD") -> Workflow:
        wf = Workflow(repo_url=repo_url, branch=branch)
        self._s.add(wf)
        await self._s.flush()
        await self._s.refresh(wf)
        logger.info("Created workflow id=%s for %s", wf.id, repo_url)
        return wf

    async def get(self, workflow_id: int) -> Optional[Workflow]:
        result = await self._s.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        return result.scalar_one_or_none()

    async def get_by_repo(self, repo_url: str) -> Optional[Workflow]:
        result = await self._s.execute(
            select(Workflow)
            .where(Workflow.repo_url == repo_url)
            .order_by(Workflow.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        workflow_id: int,
        status: WorkflowStatus,
        error_message: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> None:
        values: dict = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if error_message is not None:
            values["error_message"] = error_message
        if commit_sha is not None:
            values["commit_sha"] = commit_sha
        await self._s.execute(
            update(Workflow).where(Workflow.id == workflow_id).values(**values)
        )

    async def increment_processed(self, workflow_id: int, chunks: int = 0) -> None:
        result = await self._s.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        wf = result.scalar_one_or_none()
        if wf:
            wf.processed_files += 1
            wf.total_chunks += chunks

    async def list_recent(self, limit: int = 20) -> Sequence[Workflow]:
        result = await self._s.execute(
            select(Workflow).order_by(Workflow.created_at.desc()).limit(limit)
        )
        return result.scalars().all()


# ---------------------------------------------------------------------------
# IngestionJobRepository
# ---------------------------------------------------------------------------


class IngestionJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create_bulk(
        self, workflow_id: int, file_paths: list[tuple[str, Optional[str]]]
    ) -> list[IngestionJob]:
        """Create one IngestionJob per (file_path, language) pair."""
        jobs = [
            IngestionJob(workflow_id=workflow_id, file_path=fp, language=lang)
            for fp, lang in file_paths
        ]
        self._s.add_all(jobs)
        await self._s.flush()
        return jobs

    async def get(self, job_id: int) -> Optional[IngestionJob]:
        result = await self._s.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def list_by_workflow(self, workflow_id: int) -> Sequence[IngestionJob]:
        result = await self._s.execute(
            select(IngestionJob).where(IngestionJob.workflow_id == workflow_id)
        )
        return result.scalars().all()

    async def update_status(
        self,
        job_id: int,
        status: JobStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        values: dict = {
            "status": status,
            "chunk_count": chunk_count,
            "updated_at": datetime.now(timezone.utc),
        }
        if status == JobStatus.COMPLETED:
            values["parsed_at"] = datetime.now(timezone.utc)
        if error_message is not None:
            values["error_message"] = error_message
        await self._s.execute(
            update(IngestionJob).where(IngestionJob.id == job_id).values(**values)
        )


# ---------------------------------------------------------------------------
# EmbeddingJobRepository
# ---------------------------------------------------------------------------


class EmbeddingJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self, ingestion_job_id: int, batch_size: int, batch_index: int = 0
    ) -> EmbeddingJob:
        job = EmbeddingJob(
            job_id=ingestion_job_id,
            batch_size=batch_size,
            batch_index=batch_index,
        )
        self._s.add(job)
        await self._s.flush()
        await self._s.refresh(job)
        return job

    async def complete(
        self,
        embed_job_id: int,
        qdrant_point_ids: list[str],
        device_used: str,
        duration_ms: int,
    ) -> None:
        await self._s.execute(
            update(EmbeddingJob)
            .where(EmbeddingJob.id == embed_job_id)
            .values(
                status=JobStatus.COMPLETED,
                qdrant_point_ids=qdrant_point_ids,
                device_used=device_used,
                duration_ms=duration_ms,
                updated_at=datetime.now(timezone.utc),
            )
        )

    async def fail(self, embed_job_id: int, error_message: str) -> None:
        await self._s.execute(
            update(EmbeddingJob)
            .where(EmbeddingJob.id == embed_job_id)
            .values(
                status=JobStatus.FAILED,
                error_message=error_message,
                updated_at=datetime.now(timezone.utc),
            )
        )


# ---------------------------------------------------------------------------
# CagCacheRepository
# ---------------------------------------------------------------------------


class CagCacheRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_by_query_hash(self, query_hash: str) -> Optional[CagCache]:
        result = await self._s.execute(
            select(CagCache).where(CagCache.query_hash == query_hash)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        query_hash: str,
        canonical_query: str,
        context_fingerprint: str,
        answer: str,
        source_citations: Optional[list],
        expires_at: Optional[datetime],
    ) -> CagCache:
        existing = await self.get_by_query_hash(query_hash)
        if existing:
            existing.context_fingerprint = context_fingerprint
            existing.answer = answer
            existing.source_citations = source_citations
            existing.expires_at = expires_at
            existing.is_stale = False
            await self._s.flush()
            return existing

        entry = CagCache(
            query_hash=query_hash,
            canonical_query=canonical_query,
            context_fingerprint=context_fingerprint,
            answer=answer,
            source_citations=source_citations,
            expires_at=expires_at,
        )
        self._s.add(entry)
        await self._s.flush()
        await self._s.refresh(entry)
        return entry

    async def record_hit(self, query_hash: str) -> None:
        now = datetime.now(timezone.utc)
        await self._s.execute(
            update(CagCache)
            .where(CagCache.query_hash == query_hash)
            .values(
                hit_count=CagCache.hit_count + 1,
                last_accessed_at=now,
            )
        )

    async def mark_stale_for_repo(self, repo_url: str) -> int:
        """Invalidate all cache entries that reference chunks from `repo_url`."""
        result = await self._s.execute(
            update(CagCache)
            .where(
                CagCache.source_citations.contains([{"repo_url": repo_url}])
            )
            .values(is_stale=True)
        )
        return result.rowcount  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# GraphRepository
# ---------------------------------------------------------------------------


class GraphRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_or_create_node(
        self,
        node_key: str,
        entity_type: str,
        name: str,
        language: Optional[str],
        repo_url: str,
        file_path: str,
        commit_sha: Optional[str] = None,
    ) -> GraphNode:
        result = await self._s.execute(
            select(GraphNode).where(GraphNode.node_key == node_key)
        )
        node = result.scalar_one_or_none()
        if node:
            node.last_seen_commit = commit_sha
            return node

        node = GraphNode(
            node_key=node_key,
            entity_type=entity_type,
            name=name,
            language=language,
            repo_url=repo_url,
            file_path=file_path,
            first_seen_commit=commit_sha,
            last_seen_commit=commit_sha,
        )
        self._s.add(node)
        await self._s.flush()
        await self._s.refresh(node)
        return node

    async def upsert_edge(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        commit_sha: Optional[str],
        valid_from: Optional[datetime],
        weight: float = 1.0,
    ) -> GraphEdge:
        result = await self._s.execute(
            select(GraphEdge).where(
                GraphEdge.source_node_id == source_id,
                GraphEdge.target_node_id == target_id,
                GraphEdge.relation_type == relation_type,
                GraphEdge.commit_sha == commit_sha,
            )
        )
        edge = result.scalar_one_or_none()
        if edge:
            edge.is_active = True
            edge.weight = weight
            return edge

        edge = GraphEdge(
            source_node_id=source_id,
            target_node_id=target_id,
            relation_type=relation_type,
            commit_sha=commit_sha,
            valid_from=valid_from,
            weight=weight,
        )
        self._s.add(edge)
        await self._s.flush()
        await self._s.refresh(edge)
        return edge

    async def get_neighbors(
        self, node_id: int, relation_type: Optional[str] = None
    ) -> Sequence[GraphEdge]:
        q = select(GraphEdge).where(
            GraphEdge.source_node_id == node_id,
            GraphEdge.is_active == True,  # noqa: E712
        )
        if relation_type:
            q = q.where(GraphEdge.relation_type == relation_type)
        result = await self._s.execute(q)
        return result.scalars().all()