"""Typed SQLModel records for the additive, explicitly migrated review tables.

DDL stays in v11_backend_01; these row models deliberately do not ask SQLModel
to recreate or autogenerate changes to the existing raw SQL vector tables.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from sqlmodel import Field, SQLModel


class ReviewRepositoryRecord(SQLModel):
    """Public identity is verified independently of local source authorization."""

    repository_id: str
    source_id: str
    name: str
    github_id: int | None = None
    github_owner_id: int | None = None
    verification: str = "unverified"
    policy_version: str
    verified_at: datetime | None = None


class ReviewObjectRecord(SQLModel):
    """The existing change ID is also the sole review object ID."""

    object_id: str
    repository_id: str
    kind: Literal["pr", "commit", "branch_compare"]
    selector: dict[str, Any]
    selector_key: str
    object_version: int = Field(ge=0)
    active_snapshot_id: str | None = None
    active_generation_id: str | None = None
    collection_status: str
    sync_claim: str | None = None
    created_at: datetime


class ReviewSnapshotRecord(SQLModel):
    """Immutable code identity and coverage; CI is stored separately."""

    snapshot_id: str
    object_id: str
    sequence: int = Field(ge=1, le=20)
    content_hash: str
    facts: dict[str, Any]
    captured_at: datetime


class ReviewFileRecord(SQLModel):
    """A file ID is scoped to its immutable snapshot."""

    snapshot_id: str
    file_id: str
    path: str
    facts: dict[str, Any]


class ReviewObservationRecord(SQLModel):
    """CI/source observation cannot mutate the code snapshot."""

    observation_id: str
    object_id: str
    snapshot_id: str
    facts: dict[str, Any]
    captured_at: datetime


class ReviewRuleSetRecord(SQLModel):
    """Only applicable rule generations invalidate a review dependency."""

    rule_set_id: str
    source_id: str
    applicability: dict[str, Any]
    content_hash: str
    manifest: list[dict[str, Any]]
    generation: str | None = None
    version: int = Field(ge=0)


class ReviewAnalysisRecord(SQLModel):
    """One durable slot per revision retains even failed or uncertain attempts."""

    analysis_id: str
    object_id: str
    revision: int = Field(ge=1, le=20)
    snapshot_id: str
    generation: str
    observation_id: str
    owner_id: int
    command_id: str
    input_hash: str
    status: str
    report: dict[str, Any]
    dependencies: dict[str, Any]
    created_at: datetime


class ReviewBudgetCarryoverRecord(SQLModel):
    """A reconciliation proof carries historical and unknown charges without reset."""

    evidence_id: str
    evidence_hash: str
    total_charge: Decimal = Field(ge=0)
    smoke_charge: Decimal = Field(ge=0)
    unknown_charge: Decimal = Field(ge=0)
    reconciled: bool
    created_at: datetime
