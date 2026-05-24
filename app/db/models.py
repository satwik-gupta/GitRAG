"""
app/db/models.py
────────────────
SQLAlchemy 2.x mapped dataclasses for all state-management tables.

Tables
──────
  workflows       – one row per ingested GitHub repository
  ingestion_jobs  – one row per source file processed inside a workflow
  embedding_jobs  – one row per embedding batch
  cag_cache       – semantic-cache entries (query hash → context → answer)
  graph_nodes     – code-entity nodes for the temporal knowledge graph
  graph_edges     – directed relationships between nodes
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Status enumerations
# ---------------------------------------------------------------------------


class WorkflowStatus(str, enum.Enum):
    PENDING = "pending"
    CLONING = "cloning"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    GRAPHING = "graphing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------


class Workflow(Base):
    """Tracks a full repository ingestion lifecycle."""

    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    repo_url: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    branch: Mapped[str] = mapped_column(String(256), nullable=False, default="HEAD")
    commit_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[WorkflowStatus] = mapped_column(
        Enum(WorkflowStatus, name="workflow_status"),
        nullable=False,
        default=WorkflowStatus.PENDING,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="workflow", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Workflow id={self.id} repo={self.repo_url!r} status={self.status}>"


# ---------------------------------------------------------------------------
# ingestion_jobs
# ---------------------------------------------------------------------------


class IngestionJob(Base):
    """Tracks file-level processing within a workflow."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workflow_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    workflow: Mapped["Workflow"] = relationship(back_populates="ingestion_jobs")
    embedding_jobs: Mapped[list["EmbeddingJob"]] = relationship(
        back_populates="ingestion_job", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("workflow_id", "file_path", name="uq_job_file"),)

    def __repr__(self) -> str:
        return f"<IngestionJob id={self.id} file={self.file_path!r} status={self.status}>"


# ---------------------------------------------------------------------------
# embedding_jobs
# ---------------------------------------------------------------------------


class EmbeddingJob(Base):
    """Tracks a single embedding batch generated from an ingestion job."""

    __tablename__ = "embedding_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("ingestion_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    qdrant_point_ids: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    device_used: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    ingestion_job: Mapped["IngestionJob"] = relationship(back_populates="embedding_jobs")

    def __repr__(self) -> str:
        return (
            f"<EmbeddingJob id={self.id} job_id={self.job_id} "
            f"batch={self.batch_index} status={self.status}>"
        )


# ---------------------------------------------------------------------------
# cag_cache
# ---------------------------------------------------------------------------


class CagCache(Base):
    """
    Semantic cache: maps a normalised query hash to a frozen context
    fingerprint and the LLM-generated answer.
    """

    __tablename__ = "cag_cache"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    canonical_query: Mapped[str] = mapped_column(Text, nullable=False)
    context_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    source_citations: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<CagCache query_hash={self.query_hash!r} "
            f"hits={self.hit_count} stale={self.is_stale}>"
        )


# ---------------------------------------------------------------------------
# graph_nodes  (for the Temporal Knowledge Graph)
# ---------------------------------------------------------------------------


class GraphNode(Base):
    """A code entity (class, function, module, package)."""

    __tablename__ = "graph_nodes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    node_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)  # class|function|module
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    repo_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    first_seen_commit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_seen_commit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    outgoing_edges: Mapped[list["GraphEdge"]] = relationship(
        foreign_keys="GraphEdge.source_node_id",
        back_populates="source_node",
        cascade="all, delete-orphan",
    )
    incoming_edges: Mapped[list["GraphEdge"]] = relationship(
        foreign_keys="GraphEdge.target_node_id",
        back_populates="target_node",
        cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# graph_edges
# ---------------------------------------------------------------------------


class GraphEdge(Base):
    """
    A directed, temporal relationship between two code entities.
    The `commit_sha` + `timestamp` fields track when this edge was valid.
    """

    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_node_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_node_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # calls|imports|inherits|implements|ffi_calls
    commit_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    valid_from: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    weight: Mapped[float] = mapped_column(nullable=False, default=1.0)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    source_node: Mapped["GraphNode"] = relationship(
        foreign_keys=[source_node_id], back_populates="outgoing_edges"
    )
    target_node: Mapped["GraphNode"] = relationship(
        foreign_keys=[target_node_id], back_populates="incoming_edges"
    )

    __table_args__ = (
        UniqueConstraint(
            "source_node_id", "target_node_id", "relation_type", "commit_sha",
            name="uq_edge_temporal",
        ),
    )