"""Execution ledger boundary; production durability needs a separate implementation."""

from typing import Protocol

from app.schemas.execution import StepClaim
from app.schemas.releaseguard import ExecutionEvent


class ExecutionRepository(Protocol):
    """One owner and append-only outcomes, distinct from read-only tool caching."""

    async def claim(self, value: StepClaim) -> tuple[StepClaim, bool]:
        """Return the original claim and whether this caller owns a new send."""
        ...

    async def load(self, key: str) -> tuple[StepClaim, tuple[ExecutionEvent, ...]]:
        """Read immutable original and reconciled facts."""
        ...

    async def append_fact(self, claim: StepClaim, event: ExecutionEvent) -> None:
        """Append with owner/fence and strict event chronology checks."""
        ...

    async def entries(self, run_id: str) -> tuple[tuple[StepClaim, tuple[ExecutionEvent, ...]], ...]:
        """Read all confirmed or uncertain effects for reconciliation."""
        ...

    async def export_snapshot(self) -> str:
        """Export an offline transfer, never a production recovery certificate."""
        ...

    async def restore_snapshot(self, snapshot: str) -> None:
        """Validate everything before atomically installing into an empty store."""
        ...
