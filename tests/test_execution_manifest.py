"""Plan compatibility, scope and journal contracts."""

import asyncio

import pytest
from pydantic import ValidationError

from app.repositories.releaseguard import RunLookup
from app.schemas.execution import ExecutionManifest
from app.schemas.releaseguard import ChangeStatus, CommandAction, ExecutionPlan, StateInputs
from app.services.releaseguard.state_machine import DomainError
from tests.execution_fixtures import execution_setup, manifest_fixture
from tests.releaseguard_fixtures import command_fixture, plan_fixture


def test_legacy_plan_hash_and_json_are_stable() -> None:
    """Historical plans do not silently acquire executable manifests."""
    plan = plan_fixture()
    data = plan.model_dump(exclude={"manifest"})
    assert ExecutionPlan.model_validate(data).plan_hash == plan.plan_hash
    assert ExecutionPlan.model_validate_json(plan.model_dump_json()) == plan


@pytest.mark.parametrize("mutation", ["version", "input", "descriptor", "dependency", "reversal"])
def test_manifest_tamper_is_rejected(mutation: str) -> None:
    """Pins, hash, versions and dependency seals cannot be substituted."""
    data = manifest_fixture().model_dump(mode="json")
    if mutation == "version":
        data["version"] = "unknown.v9"
    elif mutation == "input":
        data["steps"][0]["input_hash"] = "a" * 64
    elif mutation == "descriptor":
        data["steps"][0]["descriptor_hash"] = "a" * 64
    elif mutation == "dependency":
        data["steps"][1]["dependencies"] = ["missing"]
    else:
        data["steps"][-1]["desired_value"] = 20
    with pytest.raises(ValidationError):
        ExecutionManifest.model_validate(data)


def test_scoped_plan_approval_and_fact_cas() -> None:
    """A scoped approval cannot be altered by the execution fact journal."""

    async def scenario() -> None:
        service, context, _ = await execution_setup()
        run = await service.domain.get_run(RunLookup(run_id=context.invocation.run_id))
        assert run.approval is not None and run.approval.scope == "plan"
        run = await service.domain.commit_run_update(command_fixture(run, ChangeStatus.DRY_RUN))
        # Non-execution facts cannot use this journal to alter risk or approval.
        command = command_fixture(
            run, action=CommandAction.APPEND_EXECUTION_FACTS, inputs=StateInputs(approval=run.approval)
        )
        with pytest.raises(DomainError, match="execution_fact_scope_denied"):
            await service.domain.commit_run_update(command)

    asyncio.run(scenario())
