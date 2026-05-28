"""
app/docs/repository.py
───────────────────────
Async repository for doc_generation_jobs state management.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.docs.models import DocGenerationJob, DocJobStatus, DocumentType, TemplateSource

logger = logging.getLogger(__name__)


class DocGenerationJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(
        self,
        repo_url: str,
        document_type: DocumentType,
        template_source: TemplateSource,
        workflow_id: Optional[int] = None,
        template_path_or_hash: Optional[str] = None,
    ) -> DocGenerationJob:
        job = DocGenerationJob(
            id=uuid.uuid4(),
            repo_url=repo_url,
            document_type=document_type,
            template_source=template_source,
            workflow_id=workflow_id,
            template_path_or_hash=template_path_or_hash,
            status=DocJobStatus.PENDING,
        )
        self._s.add(job)
        await self._s.flush()
        await self._s.refresh(job)
        logger.info(
            "Created DocGenerationJob id=%s type=%s repo=%s",
            job.id,
            document_type,
            repo_url,
        )
        return job

    async def get(self, job_id: uuid.UUID) -> Optional[DocGenerationJob]:
        result = await self._s.execute(
            select(DocGenerationJob).where(DocGenerationJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 20) -> Sequence[DocGenerationJob]:
        result = await self._s.execute(
            select(DocGenerationJob)
            .order_by(DocGenerationJob.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def list_by_repo(self, repo_url: str, limit: int = 20) -> Sequence[DocGenerationJob]:
        result = await self._s.execute(
            select(DocGenerationJob)
            .where(DocGenerationJob.repo_url == repo_url)
            .order_by(DocGenerationJob.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: DocJobStatus,
        current_section: Optional[str] = None,
        error_log: Optional[str] = None,
    ) -> None:
        values: dict = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
        }
        if current_section is not None:
            values["current_section"] = current_section
        if error_log is not None:
            values["error_log"] = error_log
        await self._s.execute(
            update(DocGenerationJob)
            .where(DocGenerationJob.id == job_id)
            .values(**values)
        )
        logger.debug("DocGenerationJob %s → status=%s", job_id, status)

    async def set_output_path(self, job_id: uuid.UUID, output_path: str) -> None:
        await self._s.execute(
            update(DocGenerationJob)
            .where(DocGenerationJob.id == job_id)
            .values(output_path=output_path, updated_at=datetime.now(timezone.utc))
        )
