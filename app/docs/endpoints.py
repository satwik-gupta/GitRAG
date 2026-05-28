"""
app/docs/endpoints.py
──────────────────────
FastAPI router for the documentation-generation subsystem.

Endpoints
─────────
  POST  /api/v1/generate-docs          Enqueue a documentation generation job
  GET   /api/v1/doc-jobs               List recent doc-gen jobs
  GET   /api/v1/doc-jobs/{job_id}      Status + output path of one job
  GET   /api/v1/templates              List available built-in templates
  GET   /api/v1/doc-jobs/{job_id}/download
                                       Stream the generated markdown file
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.docs.models import DocJobStatus, DocumentType, TemplateSource
from app.docs.orchestrator import DocGenerationOrchestrator
from app.docs.repository import DocGenerationJobRepository
from app.docs.schemas import (
    DocJobListResponse,
    DocJobStatusResponse,
    GenerateDocsRequest,
    GenerateDocsResponse,
    TemplateInfo,
    TemplateListResponse,
)
from app.docs.template_manager import TemplateManager
from app.db.session import get_db
from app.vector.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)
docs_router = APIRouter()


# ── Dependency injectors ──────────────────────────────────────────────────


def get_qdrant() -> QdrantManager:
    from app.main import qdrant_manager  # noqa: PLC0415

    return qdrant_manager


# ── Background task ───────────────────────────────────────────────────────


async def _run_doc_generation(
    request: GenerateDocsRequest,
    job_id: uuid.UUID,
) -> None:
    """
    Full async documentation pipeline executed as a FastAPI BackgroundTask.
    Opens its own DB session so the HTTP response can return immediately.
    """
    from app.db.session import get_session  # noqa: PLC0415
    from app.main import qdrant_manager  # noqa: PLC0415

    async with get_session() as session:
        orchestrator = DocGenerationOrchestrator(
            session=session,
            qdrant=qdrant_manager,
        )
        try:
            await orchestrator.run(request, job_id)
        except Exception as exc:
            # Orchestrator already wrote FAILED state; just log here.
            logger.error("Background doc-gen job %s raised: %s", job_id, exc)


# ── Endpoints ─────────────────────────────────────────────────────────────


@docs_router.post(
    "/generate-docs",
    response_model=GenerateDocsResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger documentation generation for an ingested repository",
)
async def generate_docs(
    body: GenerateDocsRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> GenerateDocsResponse:
    """
    Enqueue a documentation generation job and return immediately with a job ID.

    The job runs asynchronously. Poll ``GET /api/v1/doc-jobs/{job_id}`` to
    track progress and retrieve the ``output_path`` when complete.
    """
    # Validate template exists before accepting the job
    manager = TemplateManager()
    path_or_hash: str

    if body.template_source == "LOCAL":
        name = body.template_name or body.document_type.lower()
        try:
            await manager.load_local(name)
            path_or_hash = name
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
    else:
        # CUSTOM — validate the template content
        if not body.template_content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="template_content is required for CUSTOM template_source.",
            )
        try:
            _, path_or_hash = manager.load_custom(body.template_content)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid template: {exc}",
            )

    # Create the DB record
    job_repo = DocGenerationJobRepository(session)
    job = await job_repo.create(
        repo_url=body.repo_url,
        document_type=DocumentType(body.document_type),
        template_source=TemplateSource(body.template_source),
        workflow_id=body.workflow_id,
        template_path_or_hash=path_or_hash,
    )

    # Enqueue background processing
    background_tasks.add_task(_run_doc_generation, body, job.id)

    return GenerateDocsResponse(
        job_id=job.id,
        repo_url=body.repo_url,
        document_type=body.document_type,
        status=job.status.value,
        message=(
            f"Documentation generation job {job.id} enqueued. "
            f"Poll /api/v1/doc-jobs/{job.id} for status."
        ),
    )


@docs_router.get(
    "/doc-jobs",
    response_model=DocJobListResponse,
    summary="List recent documentation generation jobs",
)
async def list_doc_jobs(
    limit: int = 20,
    repo_url: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> DocJobListResponse:
    job_repo = DocGenerationJobRepository(session)
    if repo_url:
        jobs = await job_repo.list_by_repo(repo_url, limit=min(limit, 100))
    else:
        jobs = await job_repo.list_recent(limit=min(limit, 100))

    items = [
        DocJobStatusResponse(
            job_id=job.id,
            repo_url=job.repo_url,
            document_type=job.document_type.value,
            template_source=job.template_source.value,
            template_path_or_hash=job.template_path_or_hash,
            current_section=job.current_section,
            output_path=job.output_path,
            status=job.status.value,
            error_log=job.error_log,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        for job in jobs
    ]
    return DocJobListResponse(jobs=items, total=len(items))


@docs_router.get(
    "/doc-jobs/{job_id}",
    response_model=DocJobStatusResponse,
    summary="Status and output path of a documentation job",
)
async def get_doc_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> DocJobStatusResponse:
    job_repo = DocGenerationJobRepository(session)
    job = await job_repo.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doc generation job {job_id} not found.",
        )
    return DocJobStatusResponse(
        job_id=job.id,
        repo_url=job.repo_url,
        document_type=job.document_type.value,
        template_source=job.template_source.value,
        template_path_or_hash=job.template_path_or_hash,
        current_section=job.current_section,
        output_path=job.output_path,
        status=job.status.value,
        error_log=job.error_log,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@docs_router.get(
    "/doc-jobs/{job_id}/download",
    summary="Download the generated markdown document",
    response_class=FileResponse,
)
async def download_doc(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> FileResponse:
    """
    Stream the generated `.md` file for a completed job.
    Returns 404 if the job is not yet complete or the file is missing.
    """
    job_repo = DocGenerationJobRepository(session)
    job = await job_repo.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status != DocJobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not complete yet (status: {job.status.value}).",
        )

    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found on disk.",
        )

    return FileResponse(
        path=job.output_path,
        media_type="text/markdown",
        filename=Path(job.output_path).name,
    )


@docs_router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="List all available built-in documentation templates",
)
async def list_templates() -> TemplateListResponse:
    manager = TemplateManager()
    local_templates = await manager.list_local()
    items = [
        TemplateInfo(
            name=t.name,
            document_type=t.document_type,
            description=t.description,
            version=t.version,
            section_count=len(t.sections),
        )
        for t in local_templates
    ]
    return TemplateListResponse(templates=items)
