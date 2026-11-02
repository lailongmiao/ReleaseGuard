"""Dry Run never sends an effect, including rejected and prediction-failure paths."""

import asyncio
from datetime import datetime, timedelta

import pytest

from app.schemas.releaseguard import ChangeStatus
from app.services.releaseguard.dry_run import DryRunService
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.sandbox_executor import SandboxExecutor, SyntheticTargetStore
from app.services.tools.execution_adapter import TargetObservation
from app.schemas.execution import ManifestStep
from tests.execution_fixtures import TrustedResolver, execution_setup, request_for
from tests.test_change_request_contract import NOW


@pytest.mark.parametrize(
    "simulation,initial,expected",
    [
        (True, 0, ChangeStatus.DRY_RUN_PASSED),
        (False, 0, ChangeStatus.EXECUTING_SANDBOX),
        (False, 9, ChangeStatus.DRY_RUN_FAILED),
    ],
)
def test_dry_run_only_predicts(simulation: bool, initial: int, expected: ChangeStatus) -> None:
    """Read target values, predict on copies and retain an explicit simulation fact."""

    async def scenario() -> None:
        service, context, descriptor = await execution_setup(simulation=simulation)
        targets = SyntheticTargetStore({"synthetic-resource": initial})
        dry = DryRunService(
            ExecutionPreflight(service, TrustedResolver(context), descriptor), SandboxExecutor(targets)
        )
        result = await dry.run(await request_for(service), at=NOW)
        assert result.status is expected and result.simulation_result is not None
        assert result.simulation_result.mode == "offline_simulation"
        assert (await targets.read("synthetic-resource", at=NOW)).value == initial
        assert targets.calls == 0 and not result.execution_events
        assert type(result).model_validate_json(result.model_dump_json()) == result
        with pytest.raises(DomainError):
            await dry.run(await request_for(service), at=NOW)
        assert targets.calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["bool", "foreign", "old", "naive"])
def test_invalid_inspection_never_sends_or_saves_success(fault: str) -> None:
    """An invalid adapter model cannot gain validity through JSON coercion."""

    async def scenario() -> None:
        service, context, descriptor = await execution_setup(simulation=True)
        target = SyntheticTargetStore({"synthetic-resource": 0})

        class InvalidExecutor(SandboxExecutor):
            async def inspect_target(self, step: ManifestStep, *, at: datetime) -> TargetObservation:
                observation = await super().inspect_target(step, at=at)
                values = {
                    "bool": {"value": True},
                    "foreign": {"resource_id": "foreign"},
                    "old": {"observed_at": at - timedelta(seconds=1)},
                    "naive": {"observed_at": at.replace(tzinfo=None)},
                }
                return observation.model_copy(update=values[fault])

        dry = DryRunService(ExecutionPreflight(service, TrustedResolver(context), descriptor), InvalidExecutor(target))
        with pytest.raises(DomainError, match="simulation_observation_invalid"):
            await dry.run(await request_for(service), at=NOW)
        assert target.calls == 0 and (await target.read("synthetic-resource", at=NOW)).value == 0

    asyncio.run(scenario())
