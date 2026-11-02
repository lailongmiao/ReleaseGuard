"""Single-process execution ledger with atomic candidate validation on restore."""

import asyncio
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.execution import StepClaim
from app.schemas.releaseguard import EventStatus, ExecutionEvent
from app.services.releaseguard.state_machine import DomainError


class LedgerEntry(BaseModel):
    """Serializable control value using the existing domain event model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: StepClaim
    events: tuple[ExecutionEvent, ...] = ()


class LedgerSnapshot(BaseModel):
    """Unknown versions are rejected instead of inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["execution-ledger.v1"] = "execution-ledger.v1"
    entries: tuple[LedgerEntry, ...]


def validate_entry(entry: LedgerEntry) -> None:
    """Reject foreign pins, duplicate ids, impossible chronology and lost UNKNOWN."""
    previous: ExecutionEvent | None = None
    seen: set[str] = set()
    for event in entry.events:
        claim = entry.claim
        if (
            (event.claim_id, event.run_id, event.revision, event.plan_hash, event.step_id, event.direction)
            != (claim.claim_id, claim.run_id, claim.revision, claim.plan_hash, claim.step_id, claim.direction)
            or event.mode != "offline_sandbox"
            or not event.audit_ids
            or event.event_id in seen
        ):
            raise DomainError("ledger_reference_conflict")
        if event.occurred_at < (previous.occurred_at if previous else claim.claimed_at):
            raise DomainError("ledger_chronology_conflict")
        allowed = {
            None: {EventStatus.STARTED},
            EventStatus.STARTED: {EventStatus.SUCCEEDED, EventStatus.FAILED, EventStatus.UNKNOWN},
            EventStatus.UNKNOWN: {EventStatus.SUCCEEDED, EventStatus.FAILED},
        }
        if event.status not in allowed.get(previous.status if previous else None, set()):
            raise DomainError("ledger_outcome_conflict")
        if previous and previous.status is EventStatus.UNKNOWN and event.reconciles_event_id != previous.event_id:
            raise DomainError("ledger_reconciliation_conflict")
        if event.actor_id != event.audit.actor_id or event.audit.occurred_at != event.occurred_at:
            raise DomainError("ledger_audit_conflict")
        seen.add(event.event_id)
        previous = event


class InMemoryExecutionRepository:
    """Locks do not provide multi-worker or crash durability."""

    def __init__(self) -> None:
        """Maintain one explicitly isolated effect ledger."""
        self._entries: dict[str, LedgerEntry] = {}
        self._lock = asyncio.Lock()

    async def claim(self, value: StepClaim) -> tuple[StepClaim, bool]:
        """The first owner persists; a later owner never gains a second send."""
        value = StepClaim.model_validate_json(value.model_dump_json())
        async with self._lock:
            existing = self._entries.get(value.key)
            if existing:
                if existing.claim.payload_hash != value.payload_hash:
                    raise DomainError("execution_idempotency_conflict")
                return existing.claim, False
            self._entries[value.key] = LedgerEntry(claim=value)
            return value, True

    async def load(self, key: str) -> tuple[StepClaim, tuple[ExecutionEvent, ...]]:
        """Return frozen values, without leaking mutable internal containers."""
        async with self._lock:
            if key not in self._entries:
                raise DomainError("execution_claim_missing")
            entry = self._entries[key]
            return entry.claim, entry.events

    async def append_fact(self, claim: StepClaim, event: ExecutionEvent) -> None:
        """Reconcile and complete through one immutable append boundary."""
        event = ExecutionEvent.model_validate_json(event.model_dump_json())
        async with self._lock:
            entry = self._entries.get(claim.key)
            if entry is None or entry.claim != claim:
                raise DomainError("execution_owner_conflict")
            existing = next((fact for fact in entry.events if fact.event_id == event.event_id), None)
            if existing:
                if existing != event:
                    raise DomainError("execution_fact_conflict")
                return
            candidate = LedgerEntry(claim=claim, events=(*entry.events, event))
            validate_entry(candidate)
            self._entries[claim.key] = candidate

    async def entries(self, run_id: str) -> tuple[tuple[StepClaim, tuple[ExecutionEvent, ...]], ...]:
        """Return a deterministic run-scoped ledger view."""
        async with self._lock:
            return tuple(
                (entry.claim, entry.events)
                for key, entry in sorted(self._entries.items())
                if entry.claim.run_id == run_id
            )

    async def export_snapshot(self) -> str:
        """Only an offline transfer snapshot."""
        async with self._lock:
            return LedgerSnapshot(entries=tuple(self._entries[key] for key in sorted(self._entries))).model_dump_json()

    async def restore_snapshot(self, snapshot: str) -> None:
        """Reject corrupt snapshots with no partial installation."""
        try:
            candidate = LedgerSnapshot.model_validate_json(snapshot)
            entries = {entry.claim.key: entry for entry in candidate.entries}
            if len(entries) != len(candidate.entries):
                raise DomainError("ledger_duplicate_claim")
            for entry in candidate.entries:
                validate_entry(entry)
        except (ValueError, DomainError):
            raise DomainError("execution_snapshot_invalid") from None
        async with self._lock:
            if self._entries:
                raise DomainError("execution_restore_nonempty")
            self._entries = entries
