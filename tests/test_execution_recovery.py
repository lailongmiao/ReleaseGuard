"""Exact partial effects, independent recovery review and safe diagnostics."""

import asyncio
from datetime import timedelta

import pytest

from app.repositories.releaseguard import RunLookup
from app.schemas.execution import Direction
from app.schemas.releaseguard import ApprovalDecision, ChangeStatus
from app.services.releaseguard.execution_commands import execution_command
from app.services.releaseguard.recovery import RecoveryService
from app.services.releaseguard.state_machine import DomainError
from tests.execution_fixtures import request_for
from tests.test_approval_contract import approver_context, decision_command
from tests.test_change_request_contract import NOW
from tests.test_controlled_execution import Fault, runtime_setup


async def partial_setup() -> tuple:
    """One successful effect, then a failed compare-and-set."""
    runtime, target = await runtime_setup()
    service = runtime.preflight.service
    await runtime.step(await request_for(service), at=NOW)
    target._values["synthetic-resource"] = 99
    run = await runtime.step(await request_for(service, step="step-1"), at=NOW)
    assert run.status is ChangeStatus.EXECUTION_FAILED
    context = runtime.preflight.resolver.context
    run = await service.domain.commit_run_update(
        execution_command(
            run,
            context,
            command_id="require-recovery",
            at=NOW,
            code="partial_failure",
            target=ChangeStatus.RECOVERY_REQUIRED,
        )
    )
    target._values["synthetic-resource"] = 1
    recovery = RecoveryService(runtime)
    review = await recovery.prepare(await request_for(service), at=NOW, request_id="recovery-review")
    actor = approver_context(context.model_copy(update={"invocation": review.binding.invocation}))
    await service.decide(
        decision_command(review, command_id="recovery-decision", expected_state_version=review.expected_state_version),
        actor,
        runtime.preflight.descriptor,
        at=NOW,
    )
    return runtime, target, recovery


def test_partial_compensation_only_confirmed_effects() -> None:
    """A never-successful step has no invented successful rollback."""

    async def scenario() -> None:
        runtime, target, recovery = await partial_setup()
        service = runtime.preflight.service
        request = await request_for(
            service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"
        )
        run = await recovery.start(request, at=NOW)
        assert run.recovery_authorization.effect_ids == ("step-0",)
        assert run.recovery_authorization.compensation_step_ids == ("undo-0",)
        before = target.calls
        with pytest.raises(DomainError):
            await recovery.compensate(
                await request_for(
                    service, step="undo-1", direction=Direction.COMPENSATION, approval="recovery-review"
                ),
                at=NOW,
            )
        assert target.calls == before
        run = await recovery.compensate(
            await request_for(service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"),
            at=NOW,
        )
        assert target._values["synthetic-resource"] == 0
        assert not any(event.step_id == "undo-1" for event in run.execution_events)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["after_recovery_approval", "after_recovery_start"])
def test_recovery_start_commit_fault_replays_without_new_effect(fault: str) -> None:
    """A saved recovery approval/authority is read back, never decided again."""

    async def scenario() -> None:
        runtime, target, recovery = await partial_setup()
        service = runtime.preflight.service
        runtime.faults = Fault(fault)
        request = await request_for(
            service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"
        )
        count = target.calls
        with pytest.raises(RuntimeError, match="synthetic_fault"):
            await recovery.start(request, at=NOW)
        run = await recovery.start(
            await request_for(service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"),
            at=NOW,
        )
        assert run.status is ChangeStatus.RECOVERING and target.calls == count
        assert len([record for record in run.record_history if getattr(record, "scope", None) == "recovery"]) == 1
        run = await recovery.compensate(
            await request_for(service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"),
            at=NOW,
        )
        assert target.calls == count + 1 and run.execution_events[-1].status == "SUCCEEDED"

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["revoked", "expired", "failed"])
def test_compensation_denial_or_failure_preserves_forward_facts(fault: str) -> None:
    """Invalid review sends nothing; a known failed compensation is retained."""

    async def scenario() -> None:
        runtime, target, recovery = await partial_setup()
        service = runtime.preflight.service
        run = await recovery.start(
            await request_for(service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"),
            at=NOW,
        )
        before = target.calls
        at = NOW
        if fault == "revoked":
            entry = await service.approvals.get("recovery-review")
            await service.invalidate(
                "recovery-review",
                runtime.preflight.resolver.context.model_copy(update={"invocation": entry.request.binding.invocation}),
                ApprovalDecision.REVOKED,
                "review_revoked",
                at=NOW,
            )
        elif fault == "expired":
            at = NOW + timedelta(minutes=11)
        else:
            target._values["synthetic-resource"] = 99
        request = await request_for(
            service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"
        )
        if fault == "failed":
            run = await recovery.compensate(request, at=at)
            assert run.status is ChangeStatus.FAILED and run.execution_events[-1].status == "FAILED"
        else:
            with pytest.raises(DomainError):
                await recovery.compensate(request, at=at)
            assert target.calls == before
        assert any(event.step_id == "step-0" and event.status == "SUCCEEDED" for event in run.execution_events)

    asyncio.run(scenario())


def test_unknown_can_only_be_reconciled_without_resend() -> None:
    """A lost response is queried read-only even after forward approval revocation."""

    async def scenario() -> None:
        runtime, target = await runtime_setup(count=1, faults=Fault("after_send"))
        service = runtime.preflight.service
        run = await runtime.step(await request_for(service), at=NOW)
        assert run.status is ChangeStatus.EXECUTION_UNKNOWN and target.calls == 1
        await service.invalidate(
            "synthetic-approval",
            runtime.preflight.resolver.context,
            ApprovalDecision.REVOKED,
            "review_revoked",
            at=NOW,
        )
        run = await RecoveryService(runtime).reconcile(await request_for(service), at=NOW)
        assert run.status is ChangeStatus.VERIFYING and target.calls == 1
        assert run.execution_events[-1].reconciles_event_id == run.execution_events[-2].event_id

    asyncio.run(scenario())


def test_unknown_without_observable_outcome_stays_quarantined() -> None:
    """A missing retained outcome cannot be inferred from target appearance."""

    async def scenario() -> None:
        runtime, target = await runtime_setup(count=1, faults=Fault("after_send"))
        service = runtime.preflight.service
        await runtime.step(await request_for(service), at=NOW)
        target._outcomes.clear()
        with pytest.raises(DomainError, match="execution_outcome_unknown"):
            await RecoveryService(runtime).reconcile(await request_for(service), at=NOW)
        run = await service.domain.get_run(RunLookup(run_id="synthetic-run"))
        assert run.status is ChangeStatus.EXECUTION_UNKNOWN and target.calls == 1

    asyncio.run(scenario())
