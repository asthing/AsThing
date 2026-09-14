"""create documents

Revision ID: 20260913_01
Revises: 20260912_01
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260913_01"
down_revision: str | Sequence[str] | None = "20260912_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("knowledge_base_id", sa.Uuid(), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("file_type", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("storage_object_key", sa.String(length=512), nullable=False),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_object_key", name="uq_documents_storage_object_key"),
    )
    op.create_index("ix_documents_knowledge_base_id", "documents", ["knowledge_base_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_knowledge_base_id", table_name="documents")
    op.drop_table("documents")
