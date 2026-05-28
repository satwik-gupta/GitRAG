"""
app/api/endpoints.py
─────────────────────
FastAPI router: all HTTP endpoints.

Endpoints
─────────
  POST  /ingest                 – Trigger async repository ingestion
  GET   /workflows              – List recent workflows
  GET   /workflows/{id}         – Workflow status
  POST  /query                  – RAG + CAG query
  POST  /cache/invalidate       – Invalidate cache for a repo
  GET   /health                 – Dependency health check
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    WorkflowListResponse,
    WorkflowStatusResponse,
)
from app.cache.cag import CAGCache
from app.db.models import WorkflowStatus
from app.db.repositories import WorkflowRepository
from app.db.session import get_db
from app.embedding.worker import EmbeddingWorker
from app.ingestion.chunker import ChunkingEngine
from app.ingestion.cloner import AsyncGitHubCloner
from app.retrieval.pipeline import RetrievalPipeline
from app.vector.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency injectors ──────────────────────────────────────────────────


def get_qdrant() -> QdrantManager:
    from app.main import qdrant_manager  # noqa: PLC0415

    return qdrant_manager


def get_embedder() -> EmbeddingWorker:
    from app.main import embedding_worker  # noqa: PLC0415

    return embedding_worker


def get_pipeline(
    qdrant: QdrantManager = Depends(get_qdrant),
    embedder: EmbeddingWorker = Depends(get_embedder),
) -> RetrievalPipeline:
    return RetrievalPipeline(qdrant=qdrant, embedder=embedder)


# ── Background ingestion task ─────────────────────────────────────────────


async def _run_ingestion(workflow_id: int, repo_url: str, branch: str) -> None:
    """
    Full async ingestion pipeline: clone → chunk → embed → upsert → graph.
    Runs as a FastAPI background task.
    """
    from app.db.models import JobStatus  # noqa: PLC0415
    from app.db.repositories import (  # noqa: PLC0415
        EmbeddingJobRepository,
        IngestionJobRepository,
        WorkflowRepository,
    )
    from app.db.session import get_session  # noqa: PLC0415
    from app.graph.knowledge_graph import TemporalKnowledgeGraph  # noqa: PLC0415
    from app.vector.qdrant_manager import QdrantManager  # noqa: PLC0415
    from app.main import qdrant_manager, embedding_worker  # noqa: PLC0415

    cloner = AsyncGitHubCloner()
    local_path = None

    async with get_session() as session:
        wf_repo = WorkflowRepository(session)
        ingest_repo = IngestionJobRepository(session)
        embed_repo = EmbeddingJobRepository(session)

        try:
            # ── Clone ──────────────────────────────────────────────────────
            await wf_repo.update_status(workflow_id, WorkflowStatus.CLONING)
            local_path, commit_sha = await cloner.clone(repo_url, branch)
            await wf_repo.update_status(
                workflow_id, WorkflowStatus.PARSING, commit_sha=commit_sha
            )

            # ── Chunk ──────────────────────────────────────────────────────
            engine = ChunkingEngine(repo_url=repo_url, commit_sha=commit_sha)
            chunks = await engine.chunk_repository(local_path)

            # Register ingestion jobs (one per unique file)
            file_map: dict[str, str] = {}
            for c in chunks:
                file_map[c.file_path] = c.language

            file_pairs = [(fp, lang) for fp, lang in file_map.items()]
            jobs = await ingest_repo.create_bulk(workflow_id, file_pairs)
            job_by_path = {j.file_path: j for j in jobs}

            # Update total counts
            wf = await wf_repo.get(workflow_id)
            if wf:
                wf.total_files = len(file_pairs)
                wf.total_chunks = len(chunks)

            # ── Embed ──────────────────────────────────────────────────────
            await wf_repo.update_status(workflow_id, WorkflowStatus.EMBEDDING)

            batch_size = 64
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                embed_job = await embed_repo.create(
                    ingestion_job_id=list(job_by_path.values())[0].id,
                    batch_size=len(batch),
                    batch_index=i // batch_size,
                )

                import time  # noqa: PLC0415
                t0 = time.monotonic()

                await embedding_worker.embed_chunks(batch)

                duration_ms = int((time.monotonic() - t0) * 1000)

                # Build Qdrant points
                points = []
                for chunk in batch:
                    if chunk.embedding is None:
                        continue
                    point = qdrant_manager.build_point(
                        point_id=chunk.chunk_id,
                        dense_vector=chunk.embedding,
                        sparse_indices=chunk.sparse_indices or [],
                        sparse_values=chunk.sparse_values or [],
                        payload=chunk.to_qdrant_payload(),
                    )
                    points.append(point)

                await qdrant_manager.upsert_points(points)

                await embed_repo.complete(
                    embed_job_id=embed_job.id,
                    qdrant_point_ids=[c.chunk_id for c in batch],
                    device_used="auto",
                    duration_ms=duration_ms,
                )

                # Update per-file status
                for chunk in batch:
                    if chunk.file_path in job_by_path:
                        job = job_by_path[chunk.file_path]
                        await ingest_repo.update_status(
                            job.id,
                            JobStatus.COMPLETED,
                            chunk_count=job.chunk_count + 1,
                        )
                await wf_repo.increment_processed(workflow_id)

            # ── Graph ──────────────────────────────────────────────────────
            await wf_repo.update_status(workflow_id, WorkflowStatus.GRAPHING)
            tkg = TemporalKnowledgeGraph(
                session=session,
                repo_url=repo_url,
                commit_sha=commit_sha,
                concurrency=1,
            )
            await tkg.build_from_chunks(chunks)

            # ── Done ───────────────────────────────────────────────────────
            await wf_repo.update_status(workflow_id, WorkflowStatus.COMPLETED)
            logger.info("Ingestion workflow %d completed.", workflow_id)

        except Exception as exc:
            logger.exception("Ingestion workflow %d FAILED: %s", workflow_id, exc)
            await wf_repo.update_status(
                workflow_id,
                WorkflowStatus.FAILED,
                error_message=str(exc)[:1000],
            )
        finally:
            if local_path:
                await cloner.cleanup(local_path)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger repository ingestion",
)
async def ingest_repository(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Enqueue an async ingestion workflow for the given GitHub repository.
    Returns immediately with a workflow ID to track progress.
    """
    wf_repo = WorkflowRepository(session)
    wf = await wf_repo.create(repo_url=body.repo_url, branch=body.branch)

    # Ensure the workflow row is committed and visible to other sessions
    await session.commit()

    background_tasks.add_task(_run_ingestion, wf.id, body.repo_url, body.branch)
    #background_tasks.add_task(_run_ingestion, wf.id, body.repo_url, body.branch)

    return IngestResponse(
        workflow_id=wf.id,
        repo_url=body.repo_url,
        branch=body.branch,
        status=wf.status.value,
        message=f"Ingestion workflow {wf.id} enqueued.",
    )


@router.get(
    "/workflows",
    response_model=WorkflowListResponse,
    summary="List recent ingestion workflows",
)
async def list_workflows(
    limit: int = 20,
    session: AsyncSession = Depends(get_db),
) -> WorkflowListResponse:
    wf_repo = WorkflowRepository(session)
    workflows = await wf_repo.list_recent(limit=min(limit, 100))
    items = [
        WorkflowStatusResponse(
            workflow_id=wf.id,
            repo_url=wf.repo_url,
            branch=wf.branch,
            commit_sha=wf.commit_sha,
            status=wf.status.value,
            total_files=wf.total_files,
            processed_files=wf.processed_files,
            total_chunks=wf.total_chunks,
            error_message=wf.error_message,
            created_at=wf.created_at,
            updated_at=wf.updated_at,
        )
        for wf in workflows
    ]
    return WorkflowListResponse(workflows=items, total=len(items))


@router.get(
    "/workflows/{workflow_id}",
    response_model=WorkflowStatusResponse,
    summary="Get workflow status",
)
async def get_workflow(
    workflow_id: int,
    session: AsyncSession = Depends(get_db),
) -> WorkflowStatusResponse:
    wf_repo = WorkflowRepository(session)
    wf = await wf_repo.get(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found.")
    return WorkflowStatusResponse(
        workflow_id=wf.id,
        repo_url=wf.repo_url,
        branch=wf.branch,
        commit_sha=wf.commit_sha,
        status=wf.status.value,
        total_files=wf.total_files,
        processed_files=wf.processed_files,
        total_chunks=wf.total_chunks,
        error_message=wf.error_message,
        created_at=wf.created_at,
        updated_at=wf.updated_at,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG + CAG query against ingested repositories",
)
async def query_codebase(
    body: QueryRequest,
    session: AsyncSession = Depends(get_db),
    pipeline: RetrievalPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Execute the full retrieval pipeline:
    normalise → metadata filter → CAG cache → hybrid ANN → rerank → LLM.
    """
    cag = CAGCache(session)
    result = await pipeline.query(
        raw_query=body.query,
        cag_cache=cag,
        repo_url=body.repo_url,
        top_k=body.top_k,
    )

    from app.api.schemas import CitationSchema  # noqa: PLC0415

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationSchema(
                file_path=c.file_path,
                section_name=c.section_name,
                repo_url=c.repo_url,
                start_line=c.start_line,
                end_line=c.end_line,
                commit_sha=c.commit_sha,
            )
            for c in result.citations
        ],
        canonical_query=result.canonical_query,
        query_hash=result.query_hash,
        cache_hit=result.cache_hit,
    )


@router.post(
    "/cache/invalidate",
    response_model=CacheInvalidateResponse,
    summary="Invalidate CAG cache entries for a repository",
)
async def invalidate_cache(
    body: CacheInvalidateRequest,
    session: AsyncSession = Depends(get_db),
) -> CacheInvalidateResponse:
    cag = CAGCache(session)
    count = await cag.invalidate_for_repo(body.repo_url)
    return CacheInvalidateResponse(invalidated_count=count, repo_url=body.repo_url)


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(
    session: AsyncSession = Depends(get_db),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> HealthResponse:
    db_status = "ok"
    qdrant_status = "ok"

    try:
        await session.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

    try:
        await qdrant.client.get_collections()
    except Exception as exc:
        qdrant_status = f"error: {exc}"

    overall = "ok" if db_status == "ok" and qdrant_status == "ok" else "degraded"
    return HealthResponse(status=overall, db=db_status, qdrant=qdrant_status)
