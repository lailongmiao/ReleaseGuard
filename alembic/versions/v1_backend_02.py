"""Persist source/model command audit references without modifying old facts."""

from typing import Sequence

from alembic import op

revision: str = "v1_backend_02"
down_revision: str | None = "v1_backend_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add append-only application events; hashes contain no source body."""
    op.execute("""CREATE TABLE v1_audit (
        event_id varchar(128) PRIMARY KEY, owner_id integer NOT NULL,
        command_id varchar(128) NOT NULL, resource varchar(128) NOT NULL,
        action varchar(32) NOT NULL, digest varchar(64) NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        FOREIGN KEY(owner_id,command_id) REFERENCES v1_commands(owner_id,command_id))""")
    op.execute("CREATE INDEX v1_audit_resource ON v1_audit(resource,created_at,event_id)")


def downgrade() -> None:
    """Retain all existing audit and business history on migration failure."""
    raise RuntimeError("v1_destructive_downgrade_requires_separate_review")
