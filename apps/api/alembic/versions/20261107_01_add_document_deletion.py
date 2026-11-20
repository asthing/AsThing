"""Add durable document deletion intent.

Revision ID: 20261107_01
Revises: 20260913_02
"""

import sqlalchemy as sa

from alembic import op

revision = "20261107_01"
down_revision = "20260913_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("deletion_requested_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    connection = op.get_bind()
    # In PostgreSQL Alembic runs this lock, check, and DDL in one transaction.
    # A concurrent marker UPDATE must block until the column has been dropped.
    if connection.dialect.name == "postgresql":
        connection.execute(sa.text("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE"))
    if connection.scalar(sa.text(
        "SELECT EXISTS(SELECT 1 FROM documents WHERE deletion_requested_at IS NOT NULL)"
    )):
        raise RuntimeError("Finish pending document deletions before downgrading.")
    op.drop_column("documents", "deletion_requested_at")
