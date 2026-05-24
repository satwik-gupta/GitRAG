"""001_initial_schema

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── enums ─────────────────────────────────────────────────────────────
    workflow_status = postgresql.ENUM(
        "pending", "cloning", "parsing", "embedding", "graphing",
        "completed", "failed",
        name="workflow_status",
    )
    job_status = postgresql.ENUM(
        "pending", "running", "completed", "failed", "skipped",
        name="job_status",
    )
    workflow_status.create(op.get_bind())
    job_status.create(op.get_bind())

    # ── workflows ─────────────────────────────────────────────────────────
    op.create_table(
        "workflows",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("repo_url", sa.String(1024), nullable=False),
        sa.Column("branch", sa.String(256), nullable=False, server_default="HEAD"),
        sa.Column("commit_sha", sa.String(64), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "cloning", "parsing", "embedding", "graphing",
                "completed", "failed",
                name="workflow_status",
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("total_files", sa.Integer, server_default="0"),
        sa.Column("processed_files", sa.Integer, server_default="0"),
        sa.Column("total_chunks", sa.Integer, server_default="0"),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_workflows_repo_url", "workflows", ["repo_url"])
    op.create_index("ix_workflows_status", "workflows", ["status"])

    # ── ingestion_jobs ────────────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("workflow_id", sa.BigInteger, sa.ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.String(2048), nullable=False),
        sa.Column("language", sa.String(64), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "completed", "failed", "skipped", name="job_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("parsed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.UniqueConstraint("workflow_id", "file_path", name="uq_job_file"),
    )
    op.create_index("ix_ingestion_jobs_workflow_id", "ingestion_jobs", ["workflow_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])

    # ── embedding_jobs ────────────────────────────────────────────────────
    op.create_table(
        "embedding_jobs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.BigInteger, sa.ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "completed", "failed", "skipped", name="job_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("batch_size", sa.Integer, nullable=False),
        sa.Column("batch_index", sa.Integer, server_default="0"),
        sa.Column("qdrant_point_ids", postgresql.JSONB, nullable=True),
        sa.Column("device_used", sa.String(32), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
    )
    op.create_index("ix_embedding_jobs_job_id", "embedding_jobs", ["job_id"])
    op.create_index("ix_embedding_jobs_status", "embedding_jobs", ["status"])

    # ── cag_cache ─────────────────────────────────────────────────────────
    op.create_table(
        "cag_cache",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("query_hash", sa.String(128), nullable=False, unique=True),
        sa.Column("canonical_query", sa.Text, nullable=False),
        sa.Column("context_fingerprint", sa.String(128), nullable=False),
        sa.Column("answer", sa.Text, nullable=False),
        sa.Column("source_citations", postgresql.JSONB, nullable=True),
        sa.Column("hit_count", sa.Integer, server_default="0"),
        sa.Column("is_stale", sa.Boolean, server_default="false"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_cag_cache_query_hash", "cag_cache", ["query_hash"])
    op.create_index("ix_cag_cache_context_fingerprint", "cag_cache", ["context_fingerprint"])

    # ── graph_nodes ───────────────────────────────────────────────────────
    op.create_table(
        "graph_nodes",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("node_key", sa.String(512), nullable=False, unique=True),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("language", sa.String(64), nullable=True),
        sa.Column("repo_url", sa.String(1024), nullable=False),
        sa.Column("file_path", sa.String(2048), nullable=False),
        sa.Column("first_seen_commit", sa.String(64), nullable=True),
        sa.Column("last_seen_commit", sa.String(64), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
    )
    op.create_index("ix_graph_nodes_node_key", "graph_nodes", ["node_key"])

    # ── graph_edges ───────────────────────────────────────────────────────
    op.create_table(
        "graph_edges",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("source_node_id", sa.BigInteger, sa.ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_node_id", sa.BigInteger, sa.ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(64), nullable=False),
        sa.Column("commit_sha", sa.String(64), nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("weight", sa.Float, server_default="1.0"),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.UniqueConstraint(
            "source_node_id", "target_node_id", "relation_type", "commit_sha",
            name="uq_edge_temporal",
        ),
    )
    op.create_index("ix_graph_edges_source", "graph_edges", ["source_node_id"])
    op.create_index("ix_graph_edges_target", "graph_edges", ["target_node_id"])
    op.create_index("ix_graph_edges_commit", "graph_edges", ["commit_sha"])
    op.create_index("ix_graph_edges_active", "graph_edges", ["is_active"])


def downgrade() -> None:
    op.drop_table("graph_edges")
    op.drop_table("graph_nodes")
    op.drop_table("cag_cache")
    op.drop_table("embedding_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_table("workflows")
    op.execute("DROP TYPE IF EXISTS workflow_status")
    op.execute("DROP TYPE IF EXISTS job_status")