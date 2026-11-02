"""Finite executor boundary; there is no network, shell or live fallback."""

from datetime import datetime
from typing import Annotated, Protocol

from pydantic import AwareDatetime, Field

from app.schemas.execution import ControlValue, ManifestStep, Name, SandboxInput


class TargetObservation(ControlValue):
    """Independent target facts, never an executor response body."""

    resource_id: Name
    value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]
    version: Annotated[int, Field(strict=True, ge=0)]
    observed_at: AwareDatetime


class ExecutorAdapter(Protocol):
    """Only an injected isolated target supports this Phase 6 contract."""

    async def inspect_target(self, step: ManifestStep, *, at: datetime) -> TargetObservation:
        """Read an allowlisted synthetic target after permission checks."""
        ...

    async def execute_step(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
        """Perform a finite compare-and-set; false proves no change."""
        ...

    async def query_outcome(self, claim_id: str) -> bool | None:
        """Return a separately retained effect fact, or unknown."""
        ...
