"""Identity-validated approval workflow; decisions do not execute any tool."""

from datetime import datetime
from collections.abc import Callable
from typing import Literal

from app.repositories.approval import ApprovalRepository
from app.repositories.evidence import EvidenceRepository
from app.repositories.releaseguard import ChangeLookup, ReleaseGuardRepository, RunLookup
from app.schemas.approval import ApprovalCommand, ApprovalEntry, ApprovalRequest
from app.schemas.authorization import PermissionPolicy, RequestContext
from app.schemas.evidence import EvidencePolicy
from app.schemas.releaseguard import (
    ActorType,
    ApprovalDecision,
    ApprovalInvalidation,
    ApprovalRecord,
    AuditFields,
    AuditContext,
    ChangeRunState,
    ChangeStatus,
    RunCommand,
    RunCommandInput,
    StateInputs,
    build_run_command,
)
from app.schemas.tooling import ToolDescriptor
from app.services.releaseguard.audit import build_audit
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


class ApprovalService:
    """Local policies and authoritative repositories are injected, not checkpointed."""

    def __init__(
        self,
        domain: ReleaseGuardRepository,
        evidence: EvidenceRepository,
        approvals: ApprovalRepository,
        permission_policy: PermissionPolicy,
        evidence_policy: EvidencePolicy,
        *,
        approver_roles: frozenset[str] = frozenset({"approver"}),
        evidence_policy_resolver: Callable[[], EvidencePolicy] | None = None,
    ) -> None:
        """Keep the trusted policy/identity boundary separate from request messages."""
        self.domain, self.evidence, self.approvals = domain, evidence, approvals
        self.permission_policy, self.evidence_policy = permission_policy, evidence_policy
        self.approver_roles = approver_roles
        self.evidence_policy_resolver = evidence_policy_resolver

    def current_evidence_policy(self) -> EvidencePolicy:
        """Resolve the trusted per-object product policy, retaining legacy injection."""
        return self.evidence_policy_resolver() if self.evidence_policy_resolver else self.evidence_policy

    async def create(
        self, request: ApprovalRequest, context: RequestContext, descriptor: ToolDescriptor, *, at: datetime
    ) -> ApprovalEntry:
        """Recompute all seals before creating a pending, scoped idempotent request."""
        request = ApprovalRequest.model_validate_json(request.model_dump_json())
        run, binding = await load_authorization_binding(
            self.domain,
            self.evidence,
            context,
            descriptor,
            self.permission_policy,
            self.current_evidence_policy(),
            at=at,
            scope=request.binding.scope,
            effect_ids=request.binding.effect_ids,
        )
        if (
            run.status
            not in (
                {ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.EXECUTION_UNKNOWN}
                if binding.scope == "recovery"
                else {ChangeStatus.PLAN_READY, ChangeStatus.WAITING_FOR_APPROVAL}
            )
            or binding != request.binding
            or request.authorization_hash != canonical_hash(binding)
            or context.principal.principal_id != run.change_request.requester_id
            or request.requested_at != context.invocation.occurred_at
            or not request.requested_at <= at < request.expires_at
            or run.state_version != request.expected_state_version + (run.status is ChangeStatus.WAITING_FOR_APPROVAL)
        ):
            raise DomainError("approval_request_conflict")
        pending = self._record(request, context, ApprovalDecision.PENDING, request.requested_at)
        event = build_audit(
            context.invocation,
            "approval_requested",
            request.reason_code,
            at=request.requested_at,
            authorization_hash=request.authorization_hash,
            approval_id=pending.approval_id,
        )
        entry = await self.approvals.create(ApprovalEntry(request=request, pending=pending), audit=event)
        if run.status is ChangeStatus.WAITING_FOR_APPROVAL and run.approval != entry.pending:
            raise DomainError("approval_domain_fork")
        return entry

    async def decide(
        self, command: ApprovalCommand, context: RequestContext, descriptor: ToolDescriptor, *, at: datetime
    ) -> ApprovalEntry:
        """Recheck actor, Gate and current domain version immediately before CAS."""
        command = ApprovalCommand.model_validate_json(command.model_dump_json())
        entry = await self.approvals.get(command.approval_request_id)
        request = entry.request
        self._check_actor(entry, context, at=at, approver=True)
        if command.authorization_hash != request.authorization_hash:
            raise DomainError("approval_command_binding_mismatch")
        recovery = request.binding.scope == "recovery"
        if command.expected_state_version != request.expected_state_version + (not recovery):
            raise DomainError("approval_version_conflict")
        if entry.invalidations:
            raise DomainError("approval_invalidated")
        if at >= request.expires_at:
            await self.invalidate(
                request.approval_request_id, context, ApprovalDecision.EXPIRED, "approval_expired", at=at
            )
            raise DomainError("approval_expired")
        if (
            descriptor.descriptor_hash != request.binding.invocation.descriptor_hash
            or canonical_hash(self.permission_policy) != request.binding.permission_policy_hash
            or canonical_hash(self.current_evidence_policy()) != request.binding.evidence_policy_hash
        ):
            await self.invalidate(
                request.approval_request_id, context, ApprovalDecision.INVALIDATED, "approval_binding_changed", at=at
            )
            raise DomainError("approval_binding_changed")
        actor = context.principal
        command_hash = canonical_hash(
            {"command": command, "actor": actor.model_dump(exclude={"authenticated_at", "expires_at"})}
        )
        if entry.decision and entry.decision_command_hash != command_hash:
            raise DomainError("approval_decision_conflict")
        if entry.decision and entry.decision.decision is ApprovalDecision.NEEDS_CHANGES:
            # NEEDS_CHANGES deliberately clears the current plan/risk/Gate.
            # Reconcile only an exact already-committed transition; this grants
            # no new approval or tool authority and never skips a fresh Gate.
            current = await self.domain.get_run(RunLookup(run_id=request.binding.invocation.run_id))
            if current.state_version == command.expected_state_version + 1:
                latest = await self.domain.get_change(ChangeLookup(change_id=current.change_id))
                replay = approval_domain_command(
                    current,
                    entry.decision,
                    command_id=command.command_id,
                    expected_state_version=command.expected_state_version,
                )
                transition = next(
                    (item for item in current.transition_history if item.command_id == command.command_id), None
                )
                if (
                    latest != current.change_request
                    or current.revision != request.binding.invocation.revision
                    or current.request_hash != request.binding.invocation.request_hash
                    or current.status is not ChangeStatus.WAITING_FOR_EVIDENCE
                    or entry.decision not in current.record_history
                    or transition is None
                    or transition.input_hash != replay.audit.input_hash
                    or transition.state_version != current.state_version
                ):
                    raise DomainError("approval_domain_fork")
                return entry
        try:
            run, binding = await load_authorization_binding(
                self.domain,
                self.evidence,
                context,
                descriptor,
                self.permission_policy,
                self.current_evidence_policy(),
                at=at,
                original_call=request.binding.invocation,
                scope=request.binding.scope,
                effect_ids=request.binding.effect_ids,
            )
        except DomainError as error:
            if error.code in {
                "authorization_identity_conflict",
                "evidence_or_risk_blocked",
                "plan_tool_mismatch",
                "evidence_gate_blocked",
                "evidence_time_conflict",
                "tool_descriptor_mismatch",
            }:
                await self.invalidate(
                    request.approval_request_id,
                    context,
                    ApprovalDecision.INVALIDATED,
                    "approval_binding_changed",
                    at=at,
                )
            raise
        if binding != request.binding or canonical_hash(binding) != command.authorization_hash:
            await self.invalidate(
                request.approval_request_id, context, ApprovalDecision.INVALIDATED, "approval_binding_changed", at=at
            )
            raise DomainError("approval_binding_changed")
        if run.risk_assessment is None or (
            run.risk_assessment.level.value == "HIGH" and actor.principal_id == request.binding.requester_id
        ):
            raise DomainError("approval_self_denied")
        if recovery:
            if run.state_version != command.expected_state_version or run.status not in {
                ChangeStatus.RECOVERY_REQUIRED,
                ChangeStatus.EXECUTION_UNKNOWN,
            }:
                raise DomainError("approval_domain_fork")
            if entry.decision:
                return entry
        elif entry.decision:
            if entry.decision_command_hash != command_hash:
                raise DomainError("approval_decision_conflict")
            if run.state_version == command.expected_state_version:
                if run.status is not ChangeStatus.WAITING_FOR_APPROVAL or run.approval != entry.pending:
                    raise DomainError("approval_domain_fork")
            elif run.state_version != command.expected_state_version + 1 or run.approval != entry.decision:
                raise DomainError("approval_domain_fork")
            return entry
        if not recovery and (
            run.state_version != command.expected_state_version
            or run.status is not ChangeStatus.WAITING_FOR_APPROVAL
            or run.approval != entry.pending
        ):
            raise DomainError("approval_domain_fork")
        decision = self._record(request, context, ApprovalDecision(command.decision), at, reason=command.reason_code)
        event = build_audit(
            context.invocation,
            "approval_decided",
            command.reason_code,
            at=at,
            discriminator=command.command_id,
            authorization_hash=request.authorization_hash,
            approval_id=decision.approval_id,
        )
        return await self.approvals.decide(command.approval_request_id, decision, command_hash, audit=event)

    async def invalidate(
        self,
        request_id: str,
        context: RequestContext,
        decision: Literal[ApprovalDecision.EXPIRED, ApprovalDecision.REVOKED, ApprovalDecision.INVALIDATED],
        code: str,
        *,
        at: datetime,
    ) -> ApprovalEntry:
        """Append expiry/revocation after identity checks; never rewrite the decision."""
        entry = await self.approvals.get(request_id)
        call, actor = entry.request.binding.invocation, context.principal
        self._check_actor(entry, context, at=at)
        if decision not in {ApprovalDecision.EXPIRED, ApprovalDecision.REVOKED, ApprovalDecision.INVALIDATED}:
            raise DomainError("approval_actor_denied")
        if decision is ApprovalDecision.EXPIRED and at < entry.request.expires_at:
            raise DomainError("approval_not_expired")
        if entry.invalidations:
            return entry
        original = entry.decision or entry.pending
        fact = ApprovalInvalidation(
            **{
                key: getattr(original, key) for key in ("change_id", "run_id", "revision", "request_hash", "plan_hash")
            },
            invalidation_id="invalidation:" + canonical_hash({"request_id": request_id, "decision": decision}),
            approval_id=original.approval_id,
            decision=decision,
            input_hash=call.input_hash,
            audit=AuditFields(
                actor_id=actor.principal_id,
                actor_type=ActorType.USER,
                reason=code,
                correlation_id=call.correlation_id,
                occurred_at=at,
                policy_version=call.policy_version,
                input_hash=call.input_hash,
            ),
        )
        event = build_audit(
            context.invocation,
            "approval_invalidated",
            code,
            at=at,
            authorization_hash=entry.request.authorization_hash,
            approval_id=original.approval_id,
        )
        return await self.approvals.invalidate(request_id, fact, audit=event)

    def _check_actor(
        self, entry: ApprovalEntry, context: RequestContext, *, at: datetime, approver: bool = False
    ) -> None:
        """Reject unrelated targets before a caller can invalidate any history."""
        actor, call = context.principal, context.invocation
        original = entry.request.binding.invocation
        if (
            at.tzinfo is None
            or at < entry.request.requested_at
            or not actor.authenticated_at <= at < actor.expires_at
            or (actor.principal_id, actor.session_id, actor.tenant_id)
            != (call.principal_id, call.session_id, call.tenant_id)
            or actor.tenant_id != original.tenant_id
            or context.resource_tenant_id != original.tenant_id
            or call.model_dump(exclude={"principal_id", "session_id", "occurred_at"})
            != original.model_dump(exclude={"principal_id", "session_id", "occurred_at"})
            or (
                not actor.roles & self.approver_roles
                and (approver or actor.principal_id != entry.request.binding.requester_id)
            )
        ):
            raise DomainError("approval_actor_denied")

    def _record(
        self,
        request: ApprovalRequest,
        context: RequestContext,
        decision: ApprovalDecision,
        at: datetime,
        *,
        reason: str | None = None,
    ) -> ApprovalRecord:
        """Create domain-compatible records whose identities remain append-only."""
        binding, call = request.binding, request.binding.invocation
        decided = decision is not ApprovalDecision.PENDING
        actor = context.principal
        code = reason or request.reason_code
        return ApprovalRecord(
            **{key: getattr(call, key) for key in ("change_id", "run_id", "revision", "request_hash")},
            approval_id=request.approval_request_id + (":decision" if decided else ":pending"),
            plan_hash=binding.plan_hash,
            input_hash=call.input_hash,
            decision=decision,
            requester_id=binding.requester_id,
            tenant_id=call.tenant_id,
            session_id=actor.session_id,
            descriptor_hash=call.descriptor_hash,
            authorization_hash=request.authorization_hash,
            permission_policy_hash=binding.permission_policy_hash,
            evidence_policy_hash=binding.evidence_policy_hash,
            risk_hash=binding.risk_hash,
            scope=binding.scope,
            manifest_hash=binding.manifest_hash,
            effect_ids=binding.effect_ids,
            evidence_ids=list(call.evidence_ids),
            citation_ids=list(call.citation_ids),
            approver_id=actor.principal_id if decided else None,
            approver_role=sorted(actor.roles & self.approver_roles)[0] if decided else None,
            approver_roles=sorted(actor.roles) if decided else [],
            idempotency_key=request.idempotency_key,
            reason=code,
            requested_at=request.requested_at,
            decided_at=at if decided else None,
            expires_at=request.expires_at,
            audit=AuditFields(
                actor_id=actor.principal_id,
                actor_type=ActorType.USER,
                reason=code,
                correlation_id=call.correlation_id,
                occurred_at=at,
                policy_version=call.policy_version,
                input_hash=call.input_hash,
            ),
        )


def approval_domain_command(
    run: ChangeRunState, record: ApprovalRecord, *, command_id: str, expected_state_version: int | None = None
) -> RunCommand:
    """Convert a verified immutable record into the existing domain CAS command."""
    target = {
        ApprovalDecision.PENDING: ChangeStatus.WAITING_FOR_APPROVAL,
        ApprovalDecision.APPROVED: ChangeStatus.APPROVED,
        ApprovalDecision.REJECTED: ChangeStatus.REJECTED,
        ApprovalDecision.NEEDS_CHANGES: ChangeStatus.WAITING_FOR_EVIDENCE,
    }[record.decision]
    return build_run_command(
        RunCommandInput(
            change_id=run.change_id,
            run_id=run.run_id,
            expected_revision=run.revision,
            expected_state_version=run.state_version if expected_state_version is None else expected_state_version,
            command_id=command_id,
            target_status=target,
            inputs=StateInputs(approval=record),
            audit=AuditContext.model_validate(record.audit.model_dump(exclude={"input_hash"})),
        )
    )
