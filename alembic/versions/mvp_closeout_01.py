"""Add local MVP durable aggregate without changing existing template data.

Revision ID: mvp_closeout_01
Revises: b25d38b0cd7c
"""

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "mvp_closeout_01"
down_revision: str | None = "b25d38b0cd7c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add an isolated additive table; no external checkpoint table changes."""
    op.create_table(
        "local_mvp_records",
        sa.Column("change_id", sa.String(128), primary_key=True),
        sa.Column("owner_user_id", sa.Integer(), nullable=False),
        sa.Column("approver_user_id", sa.Integer(), nullable=False),
        sa.Column("creation_hash", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["approver_user_id"], ["user.id"]),
        sa.CheckConstraint("owner_user_id <> approver_user_id", name="local_mvp_separate_approver"),
    )
    op.create_index("ix_local_mvp_records_owner_user_id", "local_mvp_records", ["owner_user_id"])


def downgrade() -> None:
    """Destructive rollback is deliberately not a local runtime operation."""
    raise RuntimeError("local_mvp_destructive_downgrade_requires_separate_review")
