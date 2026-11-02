"""Separately approved partial compensation and authorized UNKNOWN reconciliation."""

from datetime import datetime, timedelta

from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalRequest, ToolAuthorizationDecision
from app.schemas.authorization import PermissionEffect
from app.schemas.execution import Direction, ExecutionRequest, SandboxInput
from app.schemas.releaseguard import ChangeRunState, ChangeStatus, EventStatus, RecoveryAuthorization, StateInputs
from app.schemas.tooling import ToolInvocation
from app.services.authorization import evaluate_permission
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_commands import audit_fields, execution_command, identity
from app.services.releaseguard.execution_preflight import PreflightGrant
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.verification_adapter import observation_descriptor


class RecoveryService:
    """Recovery decisions never follow a serialized boolean alone."""

    def __init__(self, execution: ExecutionService) -> None:
        """Reuse the single protected sender and effects ledger."""
        self.execution = execution

    async def effects(self, run: ChangeRunState) -> tuple[str, ...]:
        """Select confirmed forward effects; any uncertain claim stops compensation."""
        entries = await self.execution.ledger.entries(run.run_id)
        effects = []
        for claim, events in entries:
            if (
                claim.revision != run.revision
                or run.execution_plan is None
                or claim.plan_hash != run.execution_plan.plan_hash
            ):
                raise DomainError("recovery_ledger_fork")
            if not events or events[-1].status in {EventStatus.UNKNOWN, EventStatus.STARTED}:
                raise DomainError("execution_outcome_unknown")
            if claim.direction is Direction.FORWARD and events[-1].status is EventStatus.SUCCEEDED:
                if events[-1] not in run.execution_events:
                    raise DomainError("recovery_ledger_fork")
                effects.append(claim.step_id)
        return tuple(sorted(effects))

    async def prepare(self, request: ExecutionRequest, *, at: datetime, request_id: str) -> ApprovalRequest:
        """Create a recovery review; this method cannot approve or perform effects."""
        preflight = self.execution.preflight
        forward = await preflight.service.approvals.get(request.approval_request_id)
        context = await preflight.resolver.resolve(forward.request.binding.invocation, at=at)
        permission = evaluate_permission(context, preflight.descriptor, preflight.service.permission_policy, at=at)
        if permission.effect not in {PermissionEffect.ALLOW, PermissionEffect.APPROVAL_REQUIRED}:
            raise DomainError(permission.reason_code)
        run = await preflight.service.domain.get_run(RunLookup(run_id=request.run_id))
        if run.status is not ChangeStatus.RECOVERY_REQUIRED or run.state_version != request.state_version:
            raise DomainError("recovery_status_conflict")
        plan = run.execution_plan
        if (
            plan is None
            or plan.manifest is None
            or (request.plan_hash, request.manifest_hash) != (plan.plan_hash, plan.manifest.manifest_hash)
        ):
            raise DomainError("recovery_binding_conflict")
        effects = await self.effects(run)
        pins = sorted((p for p in plan.manifest.steps if p.compensates in effects), key=lambda p: p.order)
        if not effects or {p.compensates for p in pins} != set(effects):
            raise DomainError("recovery_compensation_missing")
        pin = pins[0]
        call = ToolInvocation.model_validate(
            {
                **context.invocation.model_dump(),
                "invocation_id": request_id + ":anchor",
                "input_hash": pin.input_hash,
                "resource_id": pin.resource_id,
                "occurred_at": at,
            }
        )
        context = await preflight.resolver.resolve(call, at=at)
        _, binding = await load_authorization_binding(
            preflight.service.domain,
            preflight.service.evidence,
            context,
            preflight.descriptor,
            preflight.service.permission_policy,
            preflight.service.evidence_policy,
            at=at,
            scope="recovery",
            effect_ids=effects,
        )
        review = ApprovalRequest(
            approval_request_id=request_id,
            idempotency_key=request_id,
            binding=binding,
            authorization_hash=canonical_hash(binding),
            expected_state_version=run.state_version,
            requested_at=at,
            expires_at=at + timedelta(minutes=10),
            reason_code="partial_compensation_review",
        )
        await preflight.service.create(review, context, preflight.descriptor, at=at)
        return review

    async def start(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Seal the exact ledger subset after fresh recovery-scope authorization."""
        grant = await self.execution.preflight.check(request, at=at, stage="recover")
        run, service = grant.run, self.execution.preflight.service
        entry = await service.approvals.get(request.approval_request_id)
        effects = await self.effects(run)
        if effects != entry.request.binding.effect_ids or entry.decision is None:
            raise DomainError("recovery_effect_conflict")
        plan = run.execution_plan
        assert plan is not None and plan.manifest is not None
        selected = tuple(
            p.step_id for p in sorted(plan.manifest.steps, key=lambda p: p.order) if p.compensates in effects
        )
        if not selected:
            raise DomainError("recovery_compensation_missing")
        if run.status is ChangeStatus.RECOVERING:
            if (
                run.recovery_authorization is None
                or run.recovery_authorization.authorization_hash != entry.request.authorization_hash
            ):
                raise DomainError("recovery_authorization_conflict")
            return run
        prior = next((fact for fact in run.fact_history if fact.command_id == request.command_id + ":approval"), None)
        if prior is None:
            run = await service.domain.commit_run_update(
                execution_command(
                    run,
                    grant.context,
                    command_id=request.command_id + ":approval",
                    at=at,
                    code="recovery_approval_saved",
                    inputs=StateInputs(recovery_approval=entry.decision),
                )
            )
        elif entry.decision not in run.record_history:
            raise DomainError("recovery_approval_fact_missing")
        await self.execution.hit("after_recovery_approval")
        auth = RecoveryAuthorization.model_validate(
            {
                **identity(run),
                "authorization_id": request.approval_request_id + ":authority",
                "plan_hash": plan.plan_hash,
                "authorized": True,
                "approval_id": entry.decision.approval_id,
                "authorization_hash": entry.request.authorization_hash,
                "effect_ids": effects,
                "compensation_step_ids": selected,
                "input_hash": entry.request.binding.invocation.input_hash,
                "audit": audit_fields(
                    grant.context,
                    at=at,
                    code="partial_compensation_authorized",
                    input_hash=entry.request.binding.invocation.input_hash,
                ),
            }
        )
        run = await service.domain.commit_run_update(
            execution_command(
                run,
                grant.context,
                command_id=request.command_id + ":start",
                at=at,
                code="recovery_started",
                target=ChangeStatus.RECOVERING,
                inputs=StateInputs(recovery_authorization=auth),
            )
        )
        await self.execution.hit("after_recovery_start")
        return run

    async def compensate(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Require saved authority to match approved effects before every step."""
        grant = await self.execution.preflight.check(request, at=at, stage="recover")
        auth = grant.run.recovery_authorization
        entry = await self.execution.preflight.service.approvals.get(request.approval_request_id)
        if (
            auth is None
            or auth.authorization_hash != entry.request.authorization_hash
            or request.step_id not in auth.compensation_step_ids
            or auth.effect_ids != await self.effects(grant.run)
        ):
            raise DomainError("recovery_effect_conflict")
        return await self.execution.step(request, at=at)

    async def reconcile(self, request: ExecutionRequest, *, at: datetime) -> ChangeRunState:
        """Read-only diagnostic permission permits facts, never a new effect."""
        service = self.execution.preflight.service
        entry = await service.approvals.get(request.approval_request_id)
        original = entry.request.binding.invocation
        descriptor = observation_descriptor()
        read_call = ToolInvocation.model_validate(
            {
                **original.model_dump(),
                "tool_name": descriptor.tool_name,
                "version": descriptor.version,
                "descriptor_hash": descriptor.descriptor_hash,
                "input_schema": descriptor.input_schema,
                "action": "inspect",
                "occurred_at": at,
            }
        )
        context = await self.execution.preflight.resolver.resolve(read_call, at=at)
        decision = evaluate_permission(context, descriptor, service.permission_policy, at=at)
        if decision.effect is not PermissionEffect.ALLOW:
            raise DomainError(decision.reason_code)
        run = await service.domain.get_run(RunLookup(run_id=request.run_id))
        plan = run.execution_plan
        if (
            run.status is not ChangeStatus.EXECUTION_UNKNOWN
            or run.state_version != request.state_version
            or plan is None
            or plan.manifest is None
            or request.mode is not plan.manifest.mode
            or request.plan_hash != plan.plan_hash
            or request.manifest_hash != plan.manifest.manifest_hash
            or any(getattr(request, k) != getattr(run, k) for k in ("change_id", "run_id", "revision", "request_hash"))
            or original.run_id != run.run_id
            or request.thread_id != original.thread_id
        ):
            raise DomainError("reconciliation_binding_conflict")
        pin = next(
            (p for p in plan.manifest.steps if p.step_id == request.step_id and p.direction is request.direction), None
        )
        if pin is None:
            raise DomainError("reconciliation_scope_denied")
        read_call = ToolInvocation.model_validate(
            {
                **read_call.model_dump(),
                "resource_id": pin.resource_id,
                "resource_type": pin.resource_type,
                "environment": pin.environment,
                "input_hash": pin.input_hash,
            }
        )
        resource_context = await self.execution.preflight.resolver.resolve(read_call, at=at)
        resource_permission = evaluate_permission(resource_context, descriptor, service.permission_policy, at=at)
        if resource_permission.effect is not PermissionEffect.ALLOW or resource_context.principal != context.principal:
            raise DomainError("reconciliation_scope_denied")
        context = resource_context
        entries = await self.execution.ledger.entries(run.run_id)
        selected = next(((c, ev) for c, ev in entries if c.step_id == pin.step_id), None)
        if selected is None or selected[1][-1].status is not EventStatus.UNKNOWN:
            raise DomainError("reconciliation_fact_missing")
        claim, facts = selected
        if claim.plan_hash != plan.plan_hash or claim.payload_hash != canonical_hash(pin):
            raise DomainError("recovery_ledger_fork")
        outcome = await self.execution.adapter.query_outcome(claim.claim_id)
        if type(outcome) is not bool:
            raise DomainError("execution_outcome_unknown")
        auth = ToolAuthorizationDecision(
            decision="allow",
            reason_codes=("diagnostic_read_authorized",),
            policy_version=decision.policy_version,
            descriptor_hash=pin.descriptor_hash,
            input_hash=pin.input_hash,
            authorization_hash=entry.request.authorization_hash,
            evaluated_at=at,
            evidence_ids=read_call.evidence_ids,
            citation_ids=read_call.citation_ids,
            approval_id=None,
            audit_ids=(),
        )
        grant = PreflightGrant(
            run, context, pin, SandboxInput(expected_value=pin.expected_value, desired_value=pin.desired_value), auth
        )
        event = await self.execution.fact(
            grant,
            claim,
            EventStatus.SUCCEEDED if outcome else EventStatus.FAILED,
            at=at,
            reconciles=facts[-1].event_id,
        )
        await self.execution.ledger.append_fact(claim, event)
        run = await self.execution.publish(grant, claim, at=at)
        latest = {ev.step_id: ev.status for ev in run.execution_events}
        if EventStatus.UNKNOWN in latest.values():
            return run
        required = (
            run.recovery_authorization.compensation_step_ids
            if run.recovery_authorization
            else tuple(s.step_id for s in plan.steps)
        )
        target = (
            ChangeStatus.VERIFYING
            if all(latest.get(s) is EventStatus.SUCCEEDED for s in required)
            else (
                ChangeStatus.RECOVERY_REQUIRED
                if any(ev.status is EventStatus.SUCCEEDED for ev in run.execution_events)
                else ChangeStatus.FAILED
            )
        )
        return await service.domain.commit_run_update(
            execution_command(
                run,
                context,
                command_id=request.command_id + ":reconciled",
                at=at,
                code="outcome_reconciled",
                target=target,
            )
        )
