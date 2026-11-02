"""Add readonly source, exact-vector index, command and budget storage.

Revision ID: v1_backend_01
Revises: mvp_closeout_01
"""

from typing import Sequence

from alembic import op

revision: str = "v1_backend_01"
down_revision: str | None = "mvp_closeout_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add isolated tables, retaining all previous aggregates and checkpoints."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""CREATE TABLE v1_sources (
        source_id varchar(128) PRIMARY KEY, owner_id integer NOT NULL REFERENCES "user"(id),
        version integer NOT NULL DEFAULT 0, acl jsonb NOT NULL DEFAULT '[]',
        generation varchar(128), required_checks jsonb NOT NULL DEFAULT '[]',
        deleted boolean NOT NULL DEFAULT false)""")
    op.execute("""CREATE TABLE v1_generations (
        generation varchar(128) PRIMARY KEY, source_id varchar(128) NOT NULL REFERENCES v1_sources(source_id),
        config jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())""")
    op.execute("""CREATE TABLE v1_chunks (
        generation varchar(128) NOT NULL REFERENCES v1_generations(generation),
        chunk_id varchar(128) NOT NULL, document_id varchar(128) NOT NULL,
        resource varchar(128) NOT NULL, content text NOT NULL, embedding vector NOT NULL,
        dimensions integer NOT NULL, metadata jsonb NOT NULL,
        PRIMARY KEY(generation,chunk_id), CHECK(vector_dims(embedding)=dimensions))""")
    op.execute("CREATE INDEX v1_chunk_scope ON v1_chunks(generation, resource)")
    op.execute("""CREATE TABLE v1_commands (
        owner_id integer NOT NULL REFERENCES "user"(id), command_id varchar(128) NOT NULL,
        digest varchar(64) NOT NULL, resource varchar(128) NOT NULL,
        status varchar(32) NOT NULL, result jsonb NOT NULL DEFAULT '{}',
        created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(owner_id,command_id))""")
    op.execute("""CREATE TABLE v1_budget (
        owner_id integer NOT NULL, command_id varchar(128) NOT NULL, attempt varchar(128) NOT NULL,
        reserved numeric(14,8) NOT NULL CHECK(reserved>0), charge numeric(14,8) NOT NULL CHECK(charge>=0),
        usage jsonb NOT NULL DEFAULT '{}', PRIMARY KEY(owner_id,command_id,attempt),
        FOREIGN KEY(owner_id,command_id) REFERENCES v1_commands(owner_id,command_id))""")


def downgrade() -> None:
    """Never drop existing local data as an automatic failure response."""
    raise RuntimeError("v1_destructive_downgrade_requires_separate_review")
