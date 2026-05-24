"""002_doc_generation_jobs

Revision ID: 002
Revises: 001
Create Date: 2026-01-01 00:00:00.000000

Adds the `doc_generation_jobs` table used by the documentation-generation
state machine.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── enums ─────────────────────────────────────────────────────────────
    document_type = postgresql.ENUM(
        "HLD", "LLD", "SOP", "DIAGRAM",
        name="document_type",
    )
    template_source = postgresql.ENUM(
        "LOCAL", "CUSTOM",
        name="template_source",
    )
    doc_job_status = postgresql.ENUM(
        "pending",
        "aggregating_context",
        "generating",
        "validating",
        "completed",
        "failed",
        name="doc_job_status",
    )
    document_type.create(op.get_bind(), checkfirst=True)
    template_source.create(op.get_bind(), checkfirst=True)
    doc_job_status.create(op.get_bind(), checkfirst=True)

    # ── table ─────────────────────────────────────────────────────────────
    op.create_table(
        "doc_generation_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "workflow_id",
            sa.BigInteger,
            sa.ForeignKey("workflows.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("repo_url", sa.String(1024), nullable=False),
        sa.Column(
            "document_type",
            sa.Enum("HLD", "LLD", "SOP", "DIAGRAM", name="document_type"),
            nullable=False,
        ),
        sa.Column(
            "template_source",
            sa.Enum("LOCAL", "CUSTOM", name="template_source"),
            nullable=False,
        ),
        sa.Column("template_path_or_hash", sa.String(255), nullable=True),
        sa.Column("current_section", sa.String(100), nullable=True),
        sa.Column("output_path", sa.Text, nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "aggregating_context", "generating",
                "validating", "completed", "failed",
                name="doc_job_status",
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_log", sa.Text, nullable=True),
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
    op.create_index("ix_doc_jobs_workflow_id", "doc_generation_jobs", ["workflow_id"])
    op.create_index("ix_doc_jobs_repo_url", "doc_generation_jobs", ["repo_url"])
    op.create_index("ix_doc_jobs_status", "doc_generation_jobs", ["status"])


def downgrade() -> None:
    op.drop_table("doc_generation_jobs")
    op.execute("DROP TYPE IF EXISTS doc_job_status")
    op.execute("DROP TYPE IF EXISTS template_source")
    op.execute("DROP TYPE IF EXISTS document_type")
