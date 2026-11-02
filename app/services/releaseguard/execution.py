"""Protected finite sandbox effects, with a single send ledger and fail-closed replay."""

import asyncio
from datetime import datetime
from typing import Protocol

from app.repositories.execution import ExecutionRepository
from app.core.logging import logger
from app.schemas.execution import Direction, ExecutionMode, ExecutionRequest, StepClaim
from app.schemas.releaseguard import ChangeRunState, ChangeStatus, EventStatus, ExecutionEvent, StateInputs
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.execution_commands import audit_fields, execution_command, identity
from app.services.releaseguard.execution_preflight import ExecutionPreflight, PreflightGrant
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.execution_adapter import ExecutorAdapter


class ExecutionFaultHook(Protocol):
    """Test-only crash windows; injected outside checkpoint and never serialized."""

    async def hit(self, stage: str) -> None:
        """Raise to simulate interruption at a specific boundary."""
        ...


def claim_for(grant: PreflightGrant, *, owner: str, at: datetime) -> StepClaim:
    """Key identity is stable across clocks, commands, checkpoints and owners."""
    pin, run = grant.step, grant.run
    assert run.execution_plan is not None
    fields = dict(
        tenant_id=grant.context.principal.tenant_id,
        run_id=run.run_id,
        revision=run.revision,
        plan_hash=run.execution_plan.plan_hash,
        step_id=pin.step_id,
        direction=pin.direction,
    )
    key, payload = canonical_hash(fields), canonical_hash(pin)
    return StepClaim.model_validate(
        {
            **fields,
            "key": key,
            "claim_id": canonical_hash({"key": key, "payload": payload}),
            "payload_hash": payload,
            "owner": owner,
            "claimed_at": at,
        }
    )


class ExecutionService:
    """No live transport or retry of a possibly sent side effect."""

    def __init__(
        self,
        preflight: ExecutionPreflight,
        ledger: ExecutionRepository,
        adapter: ExecutorAdapter,
        *,
        faults: ExecutionFaultHook | None = None,
    ) -> None:
        """Explicit protected injection, separate from Chat's tool registry."""
        self.preflight, self.ledger, self.adapter, self.faults = preflight, ledger, adapter, faults

    async def hit(self, stage: str) -> None:
        """Invoke an optional deterministic fault hook."""
        if self.faults is not None:
            await self.faults.hit(stage)

    async def fact(
        self,
        grant: PreflightGrant,
        claim: StepClaim,
        status: EventStatus,
        *,
        at: datetime,
        reconciles: str | None = None,
    ) -> ExecutionEvent:
        """Write an audit reference before exposing an execution fact."""
        code = "execution_" + status.value.lower()
        audit = await record_audit(
            self.preflight.service.approvals,
            grant.context.invocation,
            "execution_reconciled"
            if reconciles
            else "execution_started"
            if status is EventStatus.STARTED
            else "execution_completed",
            code,
            at=at,
            authorization_hash=grant.authorization.authorization_hash,
            approval_id=grant.authorization.approval_id,
            discriminator=claim.claim_id + status.value,
        )
        assert grant.run.execution_plan is not None
        return ExecutionEvent.model_validate(
            {
                **identity(grant.run),
                "event_id": claim.claim_id + ":" + status.value,
                "plan_hash": grant.run.execution_plan.plan_hash,
                "step_id": grant.step.step_id,
                "status": status,
                "occurred_at": at,
                "actor_id": grant.context.principal.principal_id,
                "reason": code,
                "mode": ExecutionMode.SANDBOX,
                "direction": grant.step.direction,
                "claim_id": claim.claim_id,
                "audit_ids": (*grant.authorization.audit_ids, audit.event_id),
                "reconciles_event_id": reconciles,
                "input_hash": grant.step.input_hash,
                "audit": audit_fields(grant.context, at=at, code=code, input_hash=grant.step.input_hash),
            }
        )

    async def publish(self, grant: PreflightGrant, claim: StepClaim, *, at: datetime) -> ChangeRunState:
        """Replay saved results into Domain without ever sending again."""
        _, facts = await self.ledger.load(claim.key)
        run = grant.run
        for event in facts:
            if any(item.event_id == event.event_id for item in run.execution_events):
                continue
            run = await self.preflight.service.domain.commit_run_update(
                execution_command(
                    run,
                    grant.context,
                    command_id="publish:" + event.event_id,
                    at=at,
                    code="execution_fact_saved",
                    inputs=StateInputs(execution_events=[event]),
                )
            )
        await self.hit("after_domain")
        return run

    async def step(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Reauthorize before claim, then make at most one protected send."""
        stage = "recover" if request.direction is Direction.COMPENSATION else "execute"
        grant = await self.preflight.check(request, at=at, stage=stage)
        if request.mode is not ExecutionMode.SANDBOX:
            raise DomainError("execution_mode_denied")
        latest = {event.step_id: event.status for event in grant.run.execution_events}
        dependencies = grant.step.dependencies
        if request.direction is Direction.COMPENSATION:
            auth = grant.run.recovery_authorization
            if auth is None or request.step_id not in auth.compensation_step_ids:
                raise DomainError("recovery_authorization_required")
            dependencies = tuple(identifier for identifier in dependencies if identifier in auth.compensation_step_ids)
        if any(latest.get(identifier) is not EventStatus.SUCCEEDED for identifier in dependencies):
            raise DomainError("execution_dependency_blocked")
        await self.hit("before_claim")
        claim, own = await self.ledger.claim(claim_for(grant, owner=request.command_id, at=at))
        await self.hit("after_claim")
        _, facts = await self.ledger.load(claim.key)
        if not own:
            if not facts or facts[-1].status is EventStatus.STARTED:
                # An overlapping or vanished owner is never proof of no effect.
                raise DomainError("execution_inflight_unknown")
            run = await self.publish(grant, claim, at=at)
            if facts[-1].status is EventStatus.UNKNOWN and run.status is not ChangeStatus.EXECUTION_UNKNOWN:
                return await self.preflight.service.domain.commit_run_update(
                    execution_command(
                        run,
                        grant.context,
                        command_id="unknown:" + claim.claim_id,
                        at=at,
                        code="execution_outcome_unknown",
                        target=ChangeStatus.EXECUTION_UNKNOWN,
                    )
                )
            if facts[-1].status is EventStatus.FAILED:
                target = (
                    ChangeStatus.FAILED
                    if request.direction is Direction.COMPENSATION
                    else ChangeStatus.EXECUTION_FAILED
                )
                return await self.preflight.service.domain.commit_run_update(
                    execution_command(
                        run,
                        grant.context,
                        command_id="outcome:" + claim.claim_id,
                        at=at,
                        code="execution_failed",
                        target=target,
                    )
                )
            return run
        started = await self.fact(grant, claim, EventStatus.STARTED, at=at)
        await self.ledger.append_fact(claim, started)
        run = await self.publish(grant, claim, at=at)
        await self.hit("after_started")
        fresh = request.model_copy(update={"state_version": run.state_version})
        grant = await self.preflight.check(fresh, at=at, stage=stage)
        await self.hit("before_send")
        try:
            plan = grant.run.execution_plan
            assert plan is not None
            definition = next(s for s in plan.steps + plan.rollback_steps if s.step_id == grant.step.step_id)
            remaining_window = (grant.run.change_request.change_window.end - at).total_seconds()
            timeout = min(self.preflight.descriptor.timeout_ms / 1000, definition.timeout_seconds, remaining_window)
            async with asyncio.timeout(timeout):
                applied = await self.adapter.execute_step(grant.step, grant.value, claim_id=claim.claim_id)
                if type(applied) is not bool:
                    raise DomainError("execution_response_invalid")
                await self.hit("after_send")
            status = EventStatus.SUCCEEDED if applied else EventStatus.FAILED
        except (Exception, asyncio.CancelledError) as error:
            logger.exception(
                "execution_outcome_unknown",
                run_id=grant.run.run_id,
                step_id=grant.step.step_id,
                direction=grant.step.direction.value,
                exc_info=(DomainError, DomainError("execution_outcome_unknown"), error.__traceback__),
            )
            status = EventStatus.UNKNOWN
        outcome = await self.fact(grant, claim, status, at=at)
        await self.ledger.append_fact(claim, outcome)
        await self.hit("after_result")
        run = await self.publish(grant, claim, at=at)
        if status in {EventStatus.FAILED, EventStatus.UNKNOWN}:
            target = (
                ChangeStatus.EXECUTION_UNKNOWN
                if status is EventStatus.UNKNOWN
                else (
                    ChangeStatus.FAILED
                    if request.direction is Direction.COMPENSATION
                    else ChangeStatus.EXECUTION_FAILED
                )
            )
            return await self.preflight.service.domain.commit_run_update(
                execution_command(
                    run,
                    grant.context,
                    command_id="outcome:" + claim.claim_id,
                    at=at,
                    code="execution_" + status.value.lower(),
                    target=target,
                )
            )
        return run
