"""Failure boundaries retain facts and require new typed recovery authority."""

import asyncio

import pytest

from app.core.langgraph.releaseguard_execution import resume_execution
from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ToolAuditEvent
from app.schemas.execution import Direction, ExecutionIntent
from app.schemas.execution import ManifestStep, SandboxInput
from app.schemas.releaseguard import ChangeStatus, EventStatus, StateInputs
from app.services.releaseguard.recovery import RecoveryService
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard import execution as execution_module
from app.services.tools.sandbox_executor import SandboxExecutor, SyntheticTargetStore
from app.services.releaseguard.execution_commands import execution_command
from tests.execution_fixtures import request_for
from tests.test_approval_contract import approver_context, decision_command
from tests.test_change_request_contract import NOW
from tests.test_controlled_execution import Fault, runtime_setup
from tests.test_execution_recovery import partial_setup
from tests.test_releaseguard_execution_graph import graph_execution_setup


class LostResultExecutor(SandboxExecutor):
    """Produce an effect and then an ambiguous response, never a false failure."""

    def __init__(self, targets: SyntheticTargetStore, failure: str) -> None:
        """Inject a deterministic cancellation/timeout/invalid response."""
        super().__init__(targets)
        self.failure = failure

    async def execute_step(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
        """Retain the synthetic outcome before losing its response."""
        await super().execute_step(step, value, claim_id=claim_id)
        if self.failure == "cancel":
            raise asyncio.CancelledError()
        if self.failure == "timeout":
            raise TimeoutError("synthetic_timeout")
        return "success"  # type: ignore[return-value]


@pytest.mark.parametrize("failure", ["cancel", "timeout", "invalid_response"])
def test_sent_ambiguous_response_is_unknown_and_never_retried(failure: str) -> None:
    """A timeout/cancel is not proof of failure or authorization to resend."""

    async def scenario() -> None:
        execution, target = await runtime_setup(count=1)
        execution.adapter = LostResultExecutor(target, failure)
        service = execution.preflight.service
        run = await execution.step(await request_for(service), at=NOW)
        assert run.status is ChangeStatus.EXECUTION_UNKNOWN
        assert run.execution_events[-1].status is EventStatus.UNKNOWN and target.calls == 1
        with pytest.raises(DomainError):
            await execution.step(await request_for(service), at=NOW)
        run = await RecoveryService(execution).reconcile(await request_for(service), at=NOW)
        assert run.status is ChangeStatus.VERIFYING and target.calls == 1

    asyncio.run(scenario())


def test_adapter_exception_body_is_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep traceback location while replacing an untrusted adapter error body."""

    class PrivateErrorExecutor(SandboxExecutor):
        async def execute_step(self, step: ManifestStep, value: SandboxInput, *, claim_id: str) -> bool:
            await super().execute_step(step, value, claim_id=claim_id)
            raise RuntimeError("synthetic_private_adapter_payload")

    class SafeLog:
        def exception(self, event: str, **fields: object) -> None:
            assert event == "execution_outcome_unknown"
            exception_info = fields["exc_info"]
            assert isinstance(exception_info, tuple)
            assert str(exception_info[1]) == "execution_outcome_unknown"
            assert exception_info[2] is not None
            assert "synthetic_private_adapter_payload" not in str(fields)

    async def scenario() -> None:
        execution, target = await runtime_setup(count=1)
        execution.adapter = PrivateErrorExecutor(target)
        monkeypatch.setattr(execution_module, "logger", SafeLog())
        run = await execution.step(await request_for(execution.preflight.service), at=NOW)
        assert run.status is ChangeStatus.EXECUTION_UNKNOWN and target.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["execution_started", "execution_completed"])
def test_audit_failure_stops_before_send_or_holds_after_send(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing durable facts are never converted into cached success or another send."""

    async def scenario() -> None:
        execution, target = await runtime_setup(count=1)
        service = execution.preflight.service
        original = service.approvals.append_audit

        async def fail(event: ToolAuditEvent) -> ToolAuditEvent:
            if event.event_type == stage:
                raise RuntimeError("synthetic_audit_failure")
            return await original(event)

        monkeypatch.setattr(service.approvals, "append_audit", fail)
        with pytest.raises(RuntimeError, match="audit_failure"):
            await execution.step(await request_for(service), at=NOW)
        count = target.calls
        assert count == (0 if stage == "execution_started" else 1)
        monkeypatch.setattr(service.approvals, "append_audit", original)
        with pytest.raises(DomainError, match="inflight_unknown"):
            await execution.step(await request_for(service), at=NOW)
        assert target.calls == count

    asyncio.run(scenario())


def test_concurrent_foreign_domain_progress_is_not_adopted() -> None:
    """Readback before checkpoint must not silently accept another command's CAS."""

    async def scenario() -> None:
        _, runtime, target, state, _ = await graph_execution_setup()
        service = runtime.execution.preflight.service

        class ForeignProgress:
            async def hit(self, stage: str) -> None:
                if stage != "after_result":
                    return
                run = await service.domain.get_run(RunLookup(run_id="synthetic-run"))
                await service.domain.commit_run_update(
                    execution_command(
                        run,
                        runtime.execution.preflight.resolver.context,
                        command_id="foreign-progress",
                        at=NOW,
                        code="foreign_progress",
                        inputs=StateInputs(execution_events=run.execution_events),
                    )
                )

        runtime.execution.faults = ForeignProgress()
        with pytest.raises(DomainError, match="store_fork"):
            await runtime.advance(state.releaseguard.execution_intent)
        with pytest.raises(DomainError, match="store_fork"):
            await runtime.advance(state.releaseguard.execution_intent)
        assert target.calls == 1

    asyncio.run(scenario())


def test_compensation_lost_response_keeps_both_directions() -> None:
    """An unknown rollback cannot be retried or erase the original failed step."""

    async def scenario() -> None:
        execution, target, recovery = await partial_setup()
        service = execution.preflight.service
        await recovery.start(
            await request_for(service, approval="recovery-review", step="undo-0", direction=Direction.COMPENSATION),
            at=NOW,
        )
        execution.faults = Fault("after_send")
        run = await recovery.compensate(
            await request_for(service, approval="recovery-review", step="undo-0", direction=Direction.COMPENSATION),
            at=NOW,
        )
        assert run.status is ChangeStatus.EXECUTION_UNKNOWN
        count = target.calls
        with pytest.raises(DomainError):
            await recovery.compensate(
                await request_for(
                    service, approval="recovery-review", step="undo-0", direction=Direction.COMPENSATION
                ),
                at=NOW,
            )
        run = await recovery.reconcile(
            await request_for(service, approval="recovery-review", step="undo-0", direction=Direction.COMPENSATION),
            at=NOW,
        )
        assert run.status is ChangeStatus.VERIFYING and target.calls == count
        assert any(e.step_id == "step-1" and e.status is EventStatus.FAILED for e in run.execution_events)

    asyncio.run(scenario())


def test_graph_two_waits_require_separate_recovery_approval() -> None:
    """Reconcile UNKNOWN then wait for an independent exact partial-effect review."""

    async def scenario() -> None:
        graph, runtime, target, state, config = await graph_execution_setup(count=2, faults=Fault("after_send"))
        service = runtime.execution.preflight.service
        first = await graph.ainvoke(state, config)
        assert first["__interrupt__"][0].value["status"] == "EXECUTION_UNKNOWN"
        second = await resume_execution(
            graph,
            ExecutionIntent(action="reconcile", request=await request_for(service, command="graph-reconcile")),
            config,
        )
        payload = second["__interrupt__"][0].value
        assert payload["status"] == "RECOVERY_REQUIRED" and target.calls == 1
        review = await service.approvals.get(payload["recovery_request"]["approval_request_id"])
        actor = approver_context(
            runtime.execution.preflight.resolver.context.model_copy(
                update={"invocation": review.request.binding.invocation}
            )
        )
        await service.decide(
            decision_command(
                review.request,
                command_id="graph-recovery-decision",
                expected_state_version=review.request.expected_state_version,
            ),
            actor,
            runtime.execution.preflight.descriptor,
            at=NOW,
        )
        request = await request_for(
            service,
            command="graph-recover",
            approval=review.request.approval_request_id,
            step="undo-0",
            direction=Direction.COMPENSATION,
        )
        result = await resume_execution(graph, ExecutionIntent(action="recover", request=request), config)
        run = result["releaseguard"].domain_state
        assert run.status is ChangeStatus.RECEIPTED and run.receipt.final_status == "RECOVERED"
        assert target.calls == 2 and (await target.read("synthetic-resource", at=NOW)).value == 0

    asyncio.run(scenario())
