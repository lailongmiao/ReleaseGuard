"""Isolated integer targets; explicitly synthetic and never a host resource."""

import asyncio
from datetime import datetime

from app.schemas.execution import ManifestStep, SandboxInput
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.execution_adapter import TargetObservation


class SyntheticTargetStore:
    """No filesystem, subprocess, network, database or external credentials."""

    def __init__(self, resources: dict[str, int]) -> None:
        """Only explicit resource identifiers are allowlisted."""
        self._values = dict(resources)
        self._versions = {key: 0 for key in resources}
        self._outcomes: dict[str, bool] = {}
        self._lock = asyncio.Lock()
        self.reads, self.calls = 0, 0

    async def read(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Independent finite observation, not an executor's reported result."""
        async with self._lock:
            if resource_id not in self._values:
                raise DomainError("target_not_allowlisted")
            self.reads += 1
            return TargetObservation(
                resource_id=resource_id,
                value=self._values[resource_id],
                version=self._versions[resource_id],
                observed_at=at,
            )

    async def compare_set(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
        """One finite atomic effect with separately observable completion fact."""
        async with self._lock:
            if step.resource_id not in self._values or step.environment not in {"development", "staging"}:
                raise DomainError("target_not_allowlisted")
            if claim_id in self._outcomes:
                raise DomainError("target_duplicate_send")
            self.calls += 1
            applied = self._values[step.resource_id] == value.expected_value
            if applied:
                self._values[step.resource_id] = value.desired_value
                self._versions[step.resource_id] += 1
            self._outcomes[claim_id] = applied
            return applied

    async def outcome(self, claim_id: str) -> bool | None:
        """A missing completion is unknown; absence never proves no send."""
        async with self._lock:
            return self._outcomes.get(claim_id)


class SandboxExecutor:
    """Finite adapter injected only into the protected execution service."""

    def __init__(self, targets: SyntheticTargetStore) -> None:
        """Retain an isolated store; never accept URLs or commands."""
        self.targets = targets

    async def inspect_target(self, step: ManifestStep, *, at: datetime) -> TargetObservation:
        """Inspect an allowlisted synthetic resource."""
        return await self.targets.read(step.resource_id, at=at)

    async def execute_step(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
        """No automatic retry exists around a side effect."""
        return await self.targets.compare_set(step, value, claim_id=claim_id)

    async def query_outcome(self, claim_id: str) -> bool | None:
        """Read the target effect journal without changing the target."""
        return await self.targets.outcome(claim_id)
