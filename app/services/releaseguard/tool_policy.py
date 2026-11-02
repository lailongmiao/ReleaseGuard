"""Protected tool service combining authoritative Gate, policy and approvals."""

from datetime import datetime

from app.schemas.approval import ToolAuthorizationDecision, ToolCallEntry
from app.schemas.authorization import PermissionEffect, RequestContext
from app.schemas.releaseguard import ApprovalDecision, ChangeStatus
from app.schemas.tooling import SideEffect, ToolDescriptor, ToolResult, ToolStatus
from app.services.authorization import evaluate_permission
from app.services.releaseguard.approval import ApprovalService
from app.services.releaseguard.audit import record_audit
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.registry import ToolRegistry, tool_failure


class ToolAuthorizationPolicy:
    """Consume trusted policies and repositories, never checkpoint authorization."""

    def __init__(self, approvals: ApprovalService, registry: ToolRegistry) -> None:
        """Use the same authoritative inputs as the approval workflow."""
        self.service, self.registry = approvals, registry

    async def authorize(
        self, context: RequestContext, descriptor: ToolDescriptor, *, at: datetime
    ) -> ToolAuthorizationDecision:
        """Re-evaluate permission before accessing Evidence or any cached result."""
        call, service = context.invocation, self.service
        requested = await record_audit(
            service.approvals, call, "tool_invocation_requested", "authorization_requested", at=call.occurred_at
        )
        permission = evaluate_permission(context, descriptor, service.permission_policy, at=at)
        policy_event = await record_audit(
            service.approvals,
            call,
            "permission_evaluated",
            permission.reason_code,
            at=at,
            discriminator=permission.policy_hash,
        )
        decision, code = "deny", permission.reason_code
        binding_hash, expiry = None, None
        if permission.effect in {PermissionEffect.ALLOW, PermissionEffect.APPROVAL_REQUIRED}:
            try:
                run, binding = await load_authorization_binding(
                    service.domain,
                    service.evidence,
                    context,
                    descriptor,
                    service.permission_policy,
                    service.evidence_policy,
                    at=at,
                )
                binding_hash = canonical_hash(binding)
                if (
                    permission.effect is PermissionEffect.ALLOW
                    and descriptor.side_effect_level is SideEffect.READ_ONLY
                    and call.approval_id is None
                ):
                    if run.status not in {ChangeStatus.PLAN_READY, ChangeStatus.APPROVED}:
                        raise DomainError("tool_status_blocked")
                    decision, code = "allow", "read_only_authorized"
                elif not call.approval_id:
                    decision, code = "approval_required", "approval_required"
                else:
                    request_id = call.approval_id.removesuffix(":decision")
                    entry = await service.approvals.get(request_id)
                    approval = entry.decision
                    expiry = entry.request.expires_at
                    if entry.invalidations:
                        decision, code = "revoked", "approval_invalidated"
                    elif at >= expiry:
                        await service.invalidate(
                            request_id, context, ApprovalDecision.EXPIRED, "approval_expired", at=at
                        )
                        decision, code = "expired", "approval_expired"
                    elif binding_hash != entry.request.authorization_hash or binding != entry.request.binding:
                        await service.invalidate(
                            request_id, context, ApprovalDecision.INVALIDATED, "approval_binding_changed", at=at
                        )
                        decision, code = "conflicting", "approval_binding_changed"
                    elif approval is None:
                        decision, code = "approval_required", "approval_required"
                    elif (
                        approval.decision is not ApprovalDecision.APPROVED or approval.approval_id != call.approval_id
                    ):
                        decision, code = "deny", "approval_not_approved"
                    elif run.status is not ChangeStatus.APPROVED or run.approval != approval:
                        decision, code = "conflicting", "approval_domain_fork"
                    else:
                        decision, code = "allow", "tool_authorized"
            except DomainError as error:
                decision, code = "deny", error.code
        elif permission.effect is PermissionEffect.UNAVAILABLE:
            decision = "unavailable"
        elif permission.effect is PermissionEffect.POLICY_CONFLICT:
            decision = "conflicting"
        return ToolAuthorizationDecision.model_validate(
            dict(
                decision=decision,
                reason_codes=(code,),
                policy_version=service.permission_policy.policy_version,
                evidence_ids=call.evidence_ids,
                citation_ids=call.citation_ids,
                approval_id=call.approval_id,
                approval_expires_at=expiry,
                descriptor_hash=descriptor.descriptor_hash,
                input_hash=call.input_hash,
                authorization_hash=binding_hash,
                evaluated_at=at,
                audit_ids=(requested.event_id, policy_event.event_id),
            )
        )

    async def invoke(self, context: RequestContext, *, at: datetime) -> ToolResult:
        """Call only read-only adapters; a claim prevents duplicate/fault replay."""
        call, repository = context.invocation, self.service.approvals
        try:
            descriptor = self.registry.descriptor(call.tool_name, call.version)
        except DomainError:
            event = await record_audit(repository, call, "tool_call_rejected", "tool_unknown_version", at=at)
            return tool_failure(call, "unknown.v1", ToolStatus.REJECTED, "tool_unknown_version").model_copy(
                update={"audit_ids": (event.event_id,)}
            )
        authorization = await self.authorize(context, descriptor, at=at)
        if authorization.decision != "allow":
            status = {
                "deny": ToolStatus.PERMISSION_DENIED,
                "approval_required": ToolStatus.APPROVAL_REQUIRED,
                "expired": ToolStatus.EXPIRED,
                "revoked": ToolStatus.REVOKED,
                "conflicting": ToolStatus.CONFLICTING,
                "unavailable": ToolStatus.UNAVAILABLE,
            }[authorization.decision]
            event = await record_audit(
                repository,
                call,
                "tool_call_rejected",
                authorization.reason_codes[0],
                at=at,
                authorization_hash=authorization.authorization_hash,
                approval_id=call.approval_id,
            )
            return tool_failure(call, descriptor.output_schema, status, authorization.reason_codes[0]).model_copy(
                update={"audit_ids": (*authorization.audit_ids, event.event_id)}
            )
        if descriptor.side_effect_level is not SideEffect.READ_ONLY:
            event = await record_audit(
                repository,
                call,
                "tool_call_not_implemented",
                "execution_not_implemented",
                at=at,
                authorization_hash=authorization.authorization_hash,
                approval_id=call.approval_id,
            )
            return tool_failure(
                call, descriptor.output_schema, ToolStatus.NOT_IMPLEMENTED, "execution_not_implemented"
            ).model_copy(update={"audit_ids": (*authorization.audit_ids, event.event_id)})
        existing = await repository.get_call(call.invocation_id)
        if existing:
            if existing.invocation != call or existing.authorization_hash != authorization.authorization_hash:
                raise DomainError("tool_idempotency_conflict")
            if existing.result:
                if existing.result.status is not ToolStatus.SUCCEEDED:
                    # A failed invocation is an immutable audit fact, not an
                    # error cache or permission to repeat the transport call.
                    return tool_failure(
                        call, descriptor.output_schema, ToolStatus.REJECTED, "tool_prior_failure"
                    ).model_copy(update={"audit_ids": existing.result.audit_ids})
                return existing.result
            await record_audit(repository, call, "consistency_stopped", "tool_completion_unknown", at=at)
            raise DomainError("tool_completion_unknown")
        try:
            await repository.claim_call(
                ToolCallEntry(invocation=call, authorization_hash=str(authorization.authorization_hash))
            )
        except DomainError as error:
            if error.code == "tool_claim_conflict":
                await record_audit(repository, call, "consistency_stopped", "tool_call_in_progress", at=at)
            raise
        result = await self.registry.call_read_only(call)
        event = await record_audit(
            repository,
            call,
            "tool_call_completed",
            result.error_code or "tool_read_completed",
            at=at,
            authorization_hash=authorization.authorization_hash,
            approval_id=call.approval_id,
        )
        result = result.model_copy(update={"audit_ids": (*authorization.audit_ids, event.event_id)})
        await repository.complete_call(call.invocation_id, result)
        return result
