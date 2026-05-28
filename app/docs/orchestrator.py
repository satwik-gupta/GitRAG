"""
app/docs/orchestrator.py
─────────────────────────
DocGenerationOrchestrator: the top-level async controller that drives the
full documentation pipeline through its state machine.

State transitions
─────────────────
  PENDING
    ↓  (template resolved, job accepted)
  AGGREGATING_CONTEXT
    ↓  (StructuredContext built from graph + Qdrant)
  GENERATING
    ↓  (each section generated; current_section updated per section)
  VALIDATING
    ↓  (full-document Mermaid validation + repair pass complete)
  COMPLETED  ←─── or ─── FAILED (on any unhandled exception)

The orchestrator does NOT own any DB session itself — the caller provides
a session so that state updates are flushed within the caller's transaction
boundary.  The FastAPI background task owns the session lifecycle.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.docs.context_aggregator import MacroContextAggregator, StructuredContext
from app.docs.generation_engine import IterativeDocumentConstructor
from app.docs.models import DocJobStatus, DocumentType, TemplateSource
from app.docs.repository import DocGenerationJobRepository
from app.docs.schemas import GenerateDocsRequest
from app.docs.template_manager import ParsedTemplate, TemplateManager
from app.vector.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)


class DocGenerationOrchestrator:
    """
    Coordinates template resolution → context aggregation →
    iterative section generation → Mermaid validation → persistence.

    Parameters
    ----------
    session:  AsyncSession for all DB writes.
    qdrant:   Connected QdrantManager for Qdrant queries.
    """

    def __init__(self, session: AsyncSession, qdrant: QdrantManager) -> None:
        self._session = session
        self._qdrant = qdrant
        self._template_manager = TemplateManager()
        self._job_repo = DocGenerationJobRepository(session)

    # ── Public entrypoint ─────────────────────────────────────────────────

    async def run(self, request: GenerateDocsRequest, job_id: uuid.UUID) -> Path:
        """
        Execute the full documentation-generation pipeline for *job_id*.

        Handles all state transitions and persists errors into `error_log`.
        Returns the Path of the generated markdown file on success.

        Raises
        ------
        Exception — only if the failure should also propagate to the caller
                    (the background task handler catches and logs it).
        """
        try:
            return await self._run_pipeline(request, job_id)
        except Exception as exc:
            logger.exception("DocGenerationJob %s FAILED: %s", job_id, exc)
            await self._fail(job_id, str(exc)[:2000])
            raise

    # ── Pipeline stages ───────────────────────────────────────────────────

    async def _run_pipeline(
        self, request: GenerateDocsRequest, job_id: uuid.UUID
    ) -> Path:

        # ── Stage 1: resolve template ──────────────────────────────────────
        template: ParsedTemplate
        path_or_hash: str

        template, path_or_hash = await self._template_manager.resolve(
            template_source=request.template_source,
            document_type=request.document_type,
            template_name=request.template_name,
            template_content=request.template_content,
        )
        logger.info(
            "Resolved template '%s' for job %s (%d sections)",
            template.name,
            job_id,
            len(template.sections),
        )

        # ── Stage 2: aggregate context ─────────────────────────────────────
        await self._transition(job_id, DocJobStatus.AGGREGATING_CONTEXT)

        aggregator = MacroContextAggregator(
            session=self._session,
            qdrant=self._qdrant,
        )
        context: StructuredContext = await aggregator.aggregate(
            repo_url=request.repo_url,
            workflow_id=request.workflow_id,
        )

        # ── Stage 3: iterative generation ─────────────────────────────────
        await self._transition(job_id, DocJobStatus.GENERATING)

        async def _on_section(section_id: str, title: str) -> None:
            await self._transition(
                job_id,
                DocJobStatus.GENERATING,
                current_section=title,
            )

        constructor = IterativeDocumentConstructor(
            template=template,
            context=context,
            job_id=job_id,
            on_section=_on_section,
        )
        _document_text, output_path = await constructor.construct()

        # ── Stage 4: validate (Mermaid pass already done inside construct) ─
        await self._transition(job_id, DocJobStatus.VALIDATING)
        await self._job_repo.set_output_path(job_id, str(output_path))

        # ── Stage 5: complete ──────────────────────────────────────────────
        await self._transition(job_id, DocJobStatus.COMPLETED)
        logger.info(
            "DocGenerationJob %s completed → %s", job_id, output_path
        )
        return output_path

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _transition(
        self,
        job_id: uuid.UUID,
        status: DocJobStatus,
        current_section: Optional[str] = None,
    ) -> None:
        await self._job_repo.update_status(
            job_id=job_id,
            status=status,
            current_section=current_section,
        )
        await self._session.commit()

    async def _fail(self, job_id: uuid.UUID, error_log: str) -> None:
        try:
            await self._job_repo.update_status(
                job_id=job_id,
                status=DocJobStatus.FAILED,
                error_log=error_log,
            )
            await self._session.commit()
        except Exception as inner:
            logger.error(
                "Could not persist FAILED state for job %s: %s", job_id, inner
            )
