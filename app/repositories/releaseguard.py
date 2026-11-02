"""Asynchronous ReleaseGuard storage protocol and typed repository messages."""

from typing import Literal, Protocol

from pydantic import Field

from app.schemas.releaseguard import (
    CONTRACT_VERSION,
    ChangeRequest,
    ChangeRunState,
    DomainModel,
    DomainRecord,
    Identifier,
    RunCommand,
    TransitionRecord,
    PositiveInt,
)


class ChangeLookup(DomainModel):
    """Lookup one immutable revision, or the latest revision if omitted."""

    change_id: Identifier
    revision: PositiveInt | None = None


class IdempotencyLookup(DomainModel):
    """Requester-scoped creation key, never a global cross-requester key."""

    requester_id: Identifier
    idempotency_key: Identifier


class RevisionAppend(DomainModel):
    """Append a new request revision with an optimistic revision expectation."""

    request: ChangeRequest
    expected_revision: PositiveInt


class RunLookup(DomainModel):
    """Lookup an immutable run identity."""

    run_id: Identifier


class RecordLookup(RunLookup):
    """Lookup a type-scoped record key within one run."""

    record_key: str = Field(min_length=1, max_length=256)


class RunHistory(DomainModel):
    """Copies of append-only facts and transition attribution."""

    records: list[DomainRecord]
    transitions: list[TransitionRecord]


class StoredCommandResult(DomainModel):
    """Original command result for exact replay even after subsequent commands."""

    command: RunCommand
    state: ChangeRunState


class RepositorySnapshot(DomainModel):
    """Versioned domain snapshot; no Graph checkpoints or execution recovery."""

    contract_version: Literal["releaseguard.v1"] = CONTRACT_VERSION
    changes: list[ChangeRequest]
    runs: list[ChangeRunState]
    command_results: list[StoredCommandResult]


class SnapshotPayload(DomainModel):
    """JSON transfer payload, not a persisted production backup."""

    data: str


class RepositorySummary(DomainModel):
    """Non-sensitive restore result reporting object counts only."""

    changes: int
    runs: int
    commands: int


class ReleaseGuardRepository(Protocol):
    """Async domain-only boundary; all writes use validated typed objects."""

    async def create_change(self, request: ChangeRequest) -> ChangeRequest:
        """Create once by requester/key or return a stable idempotency conflict."""
        ...

    async def get_change(self, query: ChangeLookup) -> ChangeRequest:
        """Read a request revision copy; missing identities return not_found."""
        ...

    async def find_by_idempotency(self, query: IdempotencyLookup) -> ChangeRequest:
        """Read the original creation result for a requester-scoped key."""
        ...

    async def append_revision(self, command: RevisionAppend) -> ChangeRequest:
        """Append rather than overwrite a request revision."""
        ...

    async def create_run(self, state: ChangeRunState) -> ChangeRunState:
        """Create an empty run against a stored current request revision."""
        ...

    async def get_run(self, query: RunLookup) -> ChangeRunState:
        """Read a deeply isolated run copy."""
        ...

    async def commit_run_update(self, command: RunCommand) -> ChangeRunState:
        """Atomically apply a domain command and its records, with version CAS."""
        ...

    async def get_run_history(self, query: RunLookup) -> RunHistory:
        """Read append-only history copies."""
        ...

    async def get_record(self, query: RecordLookup) -> DomainRecord:
        """Read a record without accepting mutable updates to its identity."""
        ...

    async def export_snapshot(self) -> SnapshotPayload:
        """Export versioned domain objects with original command replay results."""
        ...

    async def restore_snapshot(self, payload: SnapshotPayload) -> RepositorySummary:
        """Validate and replay all commands before installing an empty repository."""
        ...
