"""
app/docs/models.py
───────────────────
SQLAlchemy model for the documentation-generation state machine table.

Registered against the shared `Base` so that `Base.metadata.create_all`
in app/main.py picks it up automatically.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base


# ── Status / type enumerations ────────────────────────────────────────────


class DocJobStatus(str, enum.Enum):
    PENDING = "pending"
    AGGREGATING_CONTEXT = "aggregating_context"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, enum.Enum):
    HLD = "HLD"
    LLD = "LLD"
    SOP = "SOP"
    DIAGRAM = "DIAGRAM"


class TemplateSource(str, enum.Enum):
    LOCAL = "LOCAL"
    CUSTOM = "CUSTOM"


# ── ORM model ─────────────────────────────────────────────────────────────


class DocGenerationJob(Base):
    """
    Tracks a single documentation-generation run.

    One row per API call to /generate-docs.  The `status` column drives the
    state machine; `current_section` records which template section is being
    actively generated so that crashes can be diagnosed.
    """

    __tablename__ = "doc_generation_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Optional FK to the source ingestion workflow (may be NULL for custom repos)
    workflow_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("workflows.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    repo_url: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)

    document_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType, name="document_type"),
        nullable=False,
    )

    template_source: Mapped[TemplateSource] = mapped_column(
        Enum(TemplateSource, name="template_source"),
        nullable=False,
    )

    # For LOCAL: relative template name (e.g. "hld").
    # For CUSTOM: SHA-256 of the uploaded content.
    template_path_or_hash: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    current_section: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )

    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[DocJobStatus] = mapped_column(
        Enum(DocJobStatus, name="doc_job_status"),
        nullable=False,
        default=DocJobStatus.PENDING,
        index=True,
    )

    error_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<DocGenerationJob id={self.id} type={self.document_type} "
            f"status={self.status} repo={self.repo_url!r}>"
        )
