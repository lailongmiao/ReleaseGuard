"""Minimal local durable aggregate; unrelated template tables are preserved."""

from typing import Any

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class LocalMVPRecord(SQLModel, table=True):
    """Versioned validated facts; no tokens, credentials or raw tool responses."""

    __tablename__: Any = "local_mvp_records"
    change_id: str = Field(primary_key=True, max_length=128)
    owner_user_id: int = Field(index=True, foreign_key="user.id")
    approver_user_id: int = Field(foreign_key="user.id")
    creation_hash: str = Field(max_length=64)
    version: int = Field(default=0)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
