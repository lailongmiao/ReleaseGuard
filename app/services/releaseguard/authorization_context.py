"""Authoritative Evidence/plan/risk validation shared by approval and tool gates."""

from datetime import datetime
from typing import Literal

from app.core.langgraph.releaseguard_identity import releaseguard_thread_id
from app.repositories.evidence import EvidenceRepository
from app.repositories.releaseguard import ChangeLookup, ReleaseGuardRepository, RunLookup
from app.schemas.approval import AuthorizationBinding
from app.schemas.authorization import PermissionEffect, PermissionPolicy, RequestContext
from app.schemas.evidence import EvidencePolicy
from app.schemas.releaseguard import ChangeRunState, PermissionLabel, RiskLevel, matches_binding
from app.schemas.tooling import SideEffect, ToolDescriptor, ToolInvocation
from app.services.authorization import evaluate_permission
from app.services.evidence_sources import EvidenceAccessContext
from app.services.releaseguard.evidence_policy import evaluate_evidence_sufficiency
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


async def load_authorization_binding(
    domain: ReleaseGuardRepository,
    evidence: EvidenceRepository,
    context: RequestContext,
    descriptor: ToolDescriptor,
    permission_policy: PermissionPolicy,
    evidence_policy: EvidencePolicy,
    *,
    at: datetime,
    original_call: ToolInvocation | None = None,
    scope: Literal["tool", "plan", "recovery"] = "tool",
    effect_ids: tuple[str, ...] = (),
) -> tuple[ChangeRunState, AuthorizationBinding]:
    """Permission precedes resource reads; stored sufficient flags are never trusted."""
    permission = evaluate_permission(context, descriptor, permission_policy, at=at)
    if permission.effect not in {PermissionEffect.ALLOW, PermissionEffect.APPROVAL_REQUIRED}:
        raise DomainError(permission.reason_code)
    call = original_call or context.invocation
    caller_values = context.invocation.model_dump(exclude={"principal_id", "session_id", "occurred_at"})
    if caller_values != call.model_dump(exclude={"principal_id", "session_id", "occurred_at"}):
        raise DomainError("request_context_mismatch")
    run = await domain.get_run(RunLookup(run_id=call.run_id))
    latest = await domain.get_change(ChangeLookup(change_id=call.change_id))
    if (
        not matches_binding(call, run)
        or latest != run.change_request
        or call.thread_id != releaseguard_thread_id(run.change_id, run.run_id)
        or call.environment != run.change_request.environment.value
    ):
        raise DomainError("authorization_identity_conflict")
    risk, assessment, plan = run.risk_assessment, run.evidence_assessment, run.execution_plan
    if (
        risk is None
        or risk.level is RiskLevel.UNKNOWN
        or risk.uncertainty
        or assessment is None
        or not assessment.sufficient
        or assessment.uncertainty
        or plan is None
        or risk.assessed_at > at
        or assessment.evaluated_at > at
        or risk.policy_version != evidence_policy.policy_version
        or assessment.policy_version != evidence_policy.policy_version
    ):
        raise DomainError("evidence_or_risk_blocked")
    labels = {
        SideEffect.READ_ONLY: PermissionLabel.READ_ONLY,
        SideEffect.SIMULATION: PermissionLabel.SIMULATED,
        SideEffect.MUTATING: PermissionLabel.MUTATING,
        SideEffect.IRREVERSIBLE: PermissionLabel.IRREVERSIBLE,
    }
    if not any(
        step.tool_name == descriptor.tool_name
        and step.arguments_hash == call.input_hash
        and step.permission is labels[descriptor.side_effect_level]
        for step in (plan.rollback_steps if scope == "recovery" else plan.steps)
    ):
        raise DomainError("plan_tool_mismatch")
    if scope != "tool":
        manifest = plan.manifest
        if manifest is None or not any(
            pin.tool_name == call.tool_name
            and pin.descriptor_hash == descriptor.descriptor_hash
            and pin.input_hash == call.input_hash
            and pin.version == call.version
            and pin.resource_id == call.resource_id
            and pin.resource_type == call.resource_type
            and pin.action == call.action
            and pin.environment == call.environment
            and (pin.compensates is not None) == (scope == "recovery")
            for pin in manifest.steps
        ):
            raise DomainError("manifest_conflict")
    if not call.citation_ids or len(set(call.citation_ids)) != len(call.citation_ids):
        raise DomainError("evidence_refs_missing")
    citations = [await evidence.get_citation(identifier) for identifier in call.citation_ids]
    snapshots = [
        await evidence.get_snapshot(identifier)
        for identifier in sorted({citation.snapshot_id for citation in citations})
    ]
    for snapshot in snapshots:
        access_scope = snapshot.item.access_scope
        if access_scope.tenant_id is not None and access_scope.tenant_id != context.principal.tenant_id:
            raise DomainError("evidence_tenant_denied")
        if snapshot.item.captured_at > at or snapshot.persisted_at > at:
            raise DomainError("evidence_time_conflict")
    if any(citation.cited_at > at for citation in citations):
        raise DomainError("evidence_time_conflict")
    gate = evaluate_evidence_sufficiency(
        run,
        snapshots=snapshots,
        citations=citations,
        policy=evidence_policy,
        evaluated_at=at,
        thread_id=call.thread_id,
        access=EvidenceAccessContext(
            principal_id=context.principal.principal_id,
            tenant_id=context.principal.tenant_id,
            roles=context.principal.roles,
        ),
    )
    if (
        not gate.sufficient
        or gate.uncertainty
        or set(gate.evidence_ids) != set(call.evidence_ids)
        or set(gate.evidence_ids) != set(assessment.evidence_ids)
        or set(gate.evidence_ids) != set(risk.evidence_ids)
        or any(snapshot.item not in run.evidence_items for snapshot in snapshots)
    ):
        raise DomainError("evidence_gate_blocked")
    return run, AuthorizationBinding(
        invocation=call,
        plan_hash=plan.plan_hash,
        permission_policy_hash=permission.policy_hash,
        evidence_policy_hash=canonical_hash(evidence_policy),
        risk_hash=canonical_hash(risk),
        evidence_hash=canonical_hash({"citations": citations, "snapshots": snapshots}),
        requester_id=run.change_request.requester_id,
        scope=scope,
        manifest_hash=plan.manifest.manifest_hash if scope != "tool" and plan.manifest else None,
        effect_ids=effect_ids,
        binding_version=context.product_binding_version if context.product_hash else "v1",
        product_hash=context.product_hash,
    )
