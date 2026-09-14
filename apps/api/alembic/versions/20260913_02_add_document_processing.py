"""add durable document processing data

Revision ID: 20260913_02
Revises: 20260913_01
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260913_02"
down_revision: str | Sequence[str] | None = "20260913_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("processing_started_at", sa.DateTime(timezone=True))
    )
    op.add_column(
        "documents", sa.Column("processing_completed_at", sa.DateTime(timezone=True))
    )
    op.add_column("documents", sa.Column("failure_code", sa.String(length=64)))
    op.add_column("documents", sa.Column("failure_message", sa.String(length=255)))
    op.add_column(
        "documents",
        sa.Column(
            "processing_attempt_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_documents_status",
        "documents",
        "status IN ('Uploaded', 'Processing', 'Ready', 'Failed')",
    )

    op.create_table(
        "document_processing_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255)),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("failure_code", sa.String(length=64)),
        sa.Column("failure_message", sa.String(length=255)),
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state IN ('Queued', 'Processing', 'Completed', 'CleanupPending')",
            name="ck_document_processing_jobs_state",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("celery_task_id", name="uq_document_processing_jobs_task_id"),
        sa.UniqueConstraint("document_id", name="uq_document_processing_jobs_document_id"),
    )
    op.create_index(
        "ix_document_processing_jobs_recovery",
        "document_processing_jobs",
        ["state", "lease_expires_at"],
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("qdrant_point_id", sa.Uuid(), nullable=False),
        sa.CheckConstraint("length(content) > 0", name="ck_document_chunks_content"),
        sa.CheckConstraint("ordinal >= 0", name="ck_document_chunks_ordinal"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "content_hash", name="uq_document_chunks_hash"),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_document_chunks_ordinal"),
        sa.UniqueConstraint("qdrant_point_id", name="uq_document_chunks_qdrant_point_id"),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index(
        "ix_document_processing_jobs_recovery",
        table_name="document_processing_jobs",
    )
    op.drop_table("document_processing_jobs")
    op.drop_constraint("ck_documents_status", "documents", type_="check")
    op.drop_column("documents", "processing_attempt_count")
    op.drop_column("documents", "failure_message")
    op.drop_column("documents", "failure_code")
    op.drop_column("documents", "processing_completed_at")
    op.drop_column("documents", "processing_started_at")
