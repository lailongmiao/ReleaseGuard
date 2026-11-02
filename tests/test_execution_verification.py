"""Verification cannot borrow success from an executor's return value."""

import asyncio
from datetime import datetime, timedelta

import pytest

from app.schemas.releaseguard import ChangeStatus
from app.services.releaseguard.execution_commands import execution_command
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.verification import VerificationService
from app.services.tools.verification_adapter import IndependentRegisterReader
from app.services.tools.sandbox_executor import SyntheticTargetStore
from app.services.tools.execution_adapter import TargetObservation
from app.services.tools.sandbox_executor import SandboxExecutor
from app.schemas.execution import ManifestStep, SandboxInput
from tests.execution_fixtures import request_for
from tests.test_change_request_contract import NOW
from tests.test_controlled_execution import runtime_setup


async def verifying_setup() -> tuple:
    """Persist one completed effect before independent observation."""
    runtime, target = await runtime_setup(count=1)
    service = runtime.preflight.service
    run = await runtime.step(await request_for(service), at=NOW)
    run = await service.domain.commit_run_update(
        execution_command(
            run,
            runtime.preflight.resolver.context,
            command_id="verify-ready",
            at=NOW,
            code="all_steps_confirmed",
            target=ChangeStatus.VERIFYING,
        )
    )
    return runtime, target, run


@pytest.mark.parametrize("changed", [False, True])
def test_independent_read_controls_result_and_preserves_input_evidence(changed: bool) -> None:
    """Saved successful sends cannot conceal a target-state mismatch."""

    async def scenario() -> None:
        runtime, target, prior = await verifying_setup()
        if changed:
            target._values["synthetic-resource"] = 0
        run = await VerificationService(runtime, IndependentRegisterReader(target)).verify(
            await request_for(runtime.preflight.service), at=NOW
        )
        assert run.status is (ChangeStatus.RECOVERY_REQUIRED if changed else ChangeStatus.SUCCEEDED)
        assert run.evidence_items == prior.evidence_items and len(run.evidence_history) > len(prior.evidence_history)
        result = run.verification_result
        assert result is not None and result.citation_ids and result.observation_hash
        for identifier in result.citation_ids:
            citation = await runtime.preflight.service.evidence.get_citation(identifier)
            snapshot = await runtime.preflight.service.evidence.get_snapshot(citation.snapshot_id)
            assert snapshot.item in run.evidence_history and citation.run_id == run.run_id

    asyncio.run(scenario())


class InvalidReader(IndependentRegisterReader):
    """A source attempting to falsify its pin, time or target."""

    def __init__(self, targets: SyntheticTargetStore, fault: str) -> None:
        """Select one independently testable boundary failure."""
        super().__init__(targets)
        self.fault = fault
        if fault == "pin":
            self.source_pin = "executor-response.v1"

    async def observe(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Return an invalid observation, never an external response."""
        observation = await super().observe(resource_id, at=at)
        if self.fault == "future":
            return observation.model_copy(update={"observed_at": at + timedelta(seconds=1)})
        if self.fault == "old":
            return observation.model_copy(update={"observed_at": at - timedelta(seconds=1)})
        if self.fault == "bool":
            return observation.model_copy(update={"value": True})
        if self.fault == "range":
            return observation.model_copy(update={"value": 1000001})
        if self.fault == "naive":
            return observation.model_copy(update={"observed_at": at.replace(tzinfo=None)})
        return observation.model_copy(update={"resource_id": "foreign-target"})


@pytest.mark.parametrize("fault", ["pin", "future", "old", "foreign", "bool", "range", "naive"])
def test_invalid_independent_source_stops_without_success(fault: str) -> None:
    """Source labels and clock drift are not accepted as independent evidence."""

    async def scenario() -> None:
        runtime, target, _ = await verifying_setup()
        with pytest.raises(DomainError):
            await VerificationService(runtime, InvalidReader(target, fault)).verify(
                await request_for(runtime.preflight.service), at=NOW
            )
        assert target.calls == 1

    asyncio.run(scenario())


def test_executor_success_without_effect_fails_independent_verification() -> None:
    """A lying response cannot make unchanged target state pass verification."""

    async def scenario() -> None:
        runtime, target = await runtime_setup(count=1)
        service = runtime.preflight.service

        class LyingExecutor(SandboxExecutor):
            async def execute_step(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
                return True

        runtime.adapter = LyingExecutor(target)
        run = await runtime.step(await request_for(service), at=NOW)
        run = await service.domain.commit_run_update(
            execution_command(
                run,
                runtime.preflight.resolver.context,
                command_id="lying-verify",
                at=NOW,
                code="all_steps_confirmed",
                target=ChangeStatus.VERIFYING,
            )
        )
        run = await VerificationService(runtime, IndependentRegisterReader(target)).verify(
            await request_for(service), at=NOW
        )
        assert run.status is ChangeStatus.RECOVERY_REQUIRED and run.verification_result.status == "FAILED"
        assert run.receipt is None and (await target.read("synthetic-resource", at=NOW)).value == 0

    asyncio.run(scenario())
