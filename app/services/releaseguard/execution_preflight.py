"""Fresh, typed authorization for each sealed execution or recovery step."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from app.schemas.approval import ToolAuthorizationDecision
from app.schemas.authorization import PermissionEffect, RequestContext
from app.schemas.execution import Direction, ExecutionMode, ExecutionRequest, ManifestStep, SandboxInput
from app.schemas.releaseguard import ApprovalDecision, ChangeRunState, ChangeStatus
from app.schemas.tooling import ToolDescriptor, ToolInvocation
from app.services.authorization import evaluate_permission
from app.services.releaseguard.approval import ApprovalService
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


class ExecutionContextResolver(Protocol):
    """Resolve authenticated identities at use time, outside the checkpoint."""

    async def resolve(self, call: ToolInvocation, *, at: datetime) -> RequestContext:
        """Bind a trusted principal, current session and resource tenant."""
        ...


@dataclass(frozen=True)
class PreflightGrant:
    """Internal authoritative readback; not a reusable bearer permission."""

    run: ChangeRunState
    context: RequestContext
    step: ManifestStep
    value: SandboxInput
    authorization: ToolAuthorizationDecision


class ExecutionPreflight:
    """Consume Phase 5 policies and scoped approvals without relaxing tool gates."""

    def __init__(
        self, service: ApprovalService, resolver: ExecutionContextResolver, descriptor: ToolDescriptor
    ) -> None:
        """Pin a local descriptor; a transport cannot register arbitrary capabilities."""
        self.service, self.resolver, self.descriptor = service, resolver, descriptor

    async def check(
        self,
        request: ExecutionRequest,
        *,
        at: datetime,
        stage: Literal["dry_run", "execute", "recover", "verify", "diagnose"] = "execute",
    ) -> PreflightGrant:
        """Check permission before Domain/Evidence/target reads and rebind authority."""
        request = ExecutionRequest.model_validate_json(request.model_dump_json())
        entry = await self.service.approvals.get(request.approval_request_id)
        original = entry.request.binding.invocation
        # The anchor is server-owned approval content. Resolve it freshly before
        # reading domain or Evidence. Client-provided identity is never accepted.
        anchor = await self.resolver.resolve(original, at=at)
        permission = evaluate_permission(anchor, self.descriptor, self.service.permission_policy, at=at)
        await record_audit(
            self.service.approvals,
            anchor.invocation,
            "execution_preflight",
            permission.reason_code,
            at=at,
            discriminator=canonical_hash(request),
        )
        if permission.effect not in {PermissionEffect.ALLOW, PermissionEffect.APPROVAL_REQUIRED}:
            raise DomainError(permission.reason_code)
        if request.mode not in {ExecutionMode.SIMULATION, ExecutionMode.SANDBOX}:
            raise DomainError("execution_mode_denied")
        binding_scope = entry.request.binding.scope
        if binding_scope != ("recovery" if request.direction is Direction.COMPENSATION else "plan"):
            raise DomainError("execution_scope_denied")
        try:
            run, binding = await load_authorization_binding(
                self.service.domain,
                self.service.evidence,
                anchor,
                self.descriptor,
                self.service.permission_policy,
                self.service.current_evidence_policy(),
                at=at,
                original_call=original,
                scope=binding_scope,
                effect_ids=entry.request.binding.effect_ids,
            )
        except DomainError as error:
            if error.code in {
                "authorization_identity_conflict",
                "evidence_or_risk_blocked",
                "evidence_gate_blocked",
                "evidence_time_conflict",
                "plan_tool_mismatch",
                "manifest_conflict",
            }:
                await self.service.invalidate(
                    request.approval_request_id,
                    anchor,
                    ApprovalDecision.INVALIDATED,
                    "approval_binding_changed",
                    at=at,
                )
            raise
        approval = entry.decision
        if entry.invalidations and stage != "verify":
            raise DomainError("approval_invalidated")
        if at >= entry.request.expires_at and stage != "verify":
            await self.service.invalidate(
                request.approval_request_id, anchor, ApprovalDecision.EXPIRED, "approval_expired", at=at
            )
            raise DomainError("approval_expired")
        if binding != entry.request.binding or canonical_hash(binding) != entry.request.authorization_hash:
            await self.service.invalidate(
                request.approval_request_id, anchor, ApprovalDecision.INVALIDATED, "approval_binding_changed", at=at
            )
            raise DomainError("approval_binding_changed")
        if approval is None or approval.decision is not ApprovalDecision.APPROVED:
            raise DomainError("approval_required")
        if stage == "verify" and (
            run.execution_started_at is None
            or approval.decided_at is None
            or not approval.decided_at <= run.execution_started_at < entry.request.expires_at
        ):
            raise DomainError("execution_start_authorization_missing")
        plan = run.execution_plan
        manifest = plan.manifest if plan else None
        if plan is None or manifest is None:
            raise DomainError("manifest_conflict")
        if any(
            getattr(request, key) != getattr(run, key) for key in ("change_id", "run_id", "revision", "request_hash")
        ) or (
            request.thread_id != original.thread_id
            or request.state_version != run.state_version
            or request.plan_hash != plan.plan_hash
            or request.manifest_hash != manifest.manifest_hash
            or request.mode is not manifest.mode
            or manifest.policy_version != self.service.permission_policy.policy_version
        ):
            raise DomainError("execution_binding_conflict")
        allowed = {
            "dry_run": {ChangeStatus.APPROVED, ChangeStatus.DRY_RUN},
            "execute": {ChangeStatus.EXECUTING_SANDBOX},
            "recover": {ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.RECOVERING},
            "verify": {ChangeStatus.VERIFYING},
            "diagnose": {ChangeStatus.EXECUTION_UNKNOWN},
        }[stage]
        if run.status not in allowed:
            raise DomainError("execution_status_blocked")
        if request.direction is Direction.FORWARD and run.approval != approval:
            raise DomainError("approval_domain_fork")
        if not run.change_request.change_window.start <= at < run.change_request.change_window.end:
            raise DomainError("execution_window_closed")
        pin = next((step for step in manifest.steps if step.step_id == request.step_id), None)
        if (
            pin is None
            or pin.direction is not request.direction
            or pin.descriptor_hash != self.descriptor.descriptor_hash
        ):
            raise DomainError("manifest_conflict")
        call = ToolInvocation.model_validate(
            {
                **original.model_dump(),
                "invocation_id": "execution:"
                + canonical_hash({"run": run.run_id, "plan": plan.plan_hash, "step": pin.step_id}),
                "tool_name": pin.tool_name,
                "version": pin.version,
                "descriptor_hash": pin.descriptor_hash,
                "input_schema": pin.input_schema,
                "input_hash": pin.input_hash,
                "resource_type": pin.resource_type,
                "resource_id": pin.resource_id,
                "action": pin.action,
                "environment": pin.environment,
                "occurred_at": at,
                "approval_id": approval.approval_id,
            }
        )
        context = await self.resolver.resolve(call, at=at)
        current = evaluate_permission(context, self.descriptor, self.service.permission_policy, at=at)
        if current.effect not in {PermissionEffect.ALLOW, PermissionEffect.APPROVAL_REQUIRED}:
            raise DomainError(current.reason_code)
        if context.invocation != call or context.principal != anchor.principal:
            raise DomainError("execution_identity_conflict")
        audit = await record_audit(
            self.service.approvals,
            call,
            "execution_preflight",
            "execution_authorized",
            at=at,
            authorization_hash=entry.request.authorization_hash,
            approval_id=approval.approval_id,
        )
        decision = ToolAuthorizationDecision(
            decision="allow",
            reason_codes=("execution_authorized",),
            policy_version=current.policy_version,
            evidence_ids=call.evidence_ids,
            citation_ids=call.citation_ids,
            approval_id=approval.approval_id,
            approval_expires_at=approval.expires_at,
            descriptor_hash=pin.descriptor_hash,
            input_hash=pin.input_hash,
            authorization_hash=entry.request.authorization_hash,
            evaluated_at=at,
            audit_ids=(audit.event_id,),
        )
        return PreflightGrant(
            run,
            context,
            pin,
            SandboxInput(expected_value=pin.expected_value, desired_value=pin.desired_value),
            decision,
        )
