"""Retain unresolved external vector-write obligations across worker loss.

Revision ID: 20261108_01
Revises: 20261107_01
"""

import sqlalchemy as sa

from alembic import op

revision = "20261108_01"
down_revision = "20261107_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document_processing_jobs", sa.Column("vector_write_token", sa.Uuid()))
    op.add_column(
        "document_processing_jobs",
        sa.Column("vector_write_started_at", sa.DateTime(timezone=True)),
    )
    # Pre-upgrade in-flight work and vector-write failures have no trustworthy ACK
    # journal. Conservatively retain their obligations rather than treating NULL as
    # proof of settlement. Reuse the job UUID as the legacy write token.
    op.execute(sa.text(
        "UPDATE document_processing_jobs SET vector_write_token = id, "
        "vector_write_started_at = COALESCE(started_at, updated_at, queued_at) "
        "WHERE state IN ('Processing', 'CleanupPending') "
        "OR failure_code IN ('VECTOR_PERSISTENCE_FAILED', 'CLEANUP_FAILED')"
    ))


def downgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        connection.execute(sa.text(
            "LOCK TABLE document_processing_jobs IN ACCESS EXCLUSIVE MODE"
        ))
    if connection.scalar(sa.text(
        "SELECT EXISTS(SELECT 1 FROM document_processing_jobs "
        "WHERE vector_write_token IS NOT NULL)"
    )):
        raise RuntimeError("Resolve pending vector writes before downgrading.")
    op.drop_column("document_processing_jobs", "vector_write_started_at")
    op.drop_column("document_processing_jobs", "vector_write_token")
