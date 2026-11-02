"""Offline approval state, identity, Evidence and concurrency contract tests."""

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from app.repositories.in_memory_approval import InMemoryApprovalRepository
from app.repositories.in_memory_evidence import InMemoryEvidenceRepository
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalCommand, ApprovalRequest
from app.schemas.authorization import RequestContext
from app.schemas.evidence import EvidenceCitation, EvidenceSnapshot
from app.schemas.releaseguard import ApprovalDecision, ChangeStatus, new_change_run
from app.schemas.tooling import ToolDescriptor
from app.services.releaseguard.approval import ApprovalService, approval_domain_command
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from tests.releaseguard_fixtures import command_fixture, edge_inputs, evidence_fixture, plan_fixture, risk_assessment
from tests.test_authorization import policy_fixture, principal_fixture, rule_fixture
from tests.test_change_request_contract import NOW, request_fixture
from tests.test_releaseguard_evidence_graph import policy_fixture as evidence_policy_fixture
from tests.test_tool_contract import descriptor_fixture, invocation_fixture


async def approval_setup(
    *, level: str = "LOW", effect: str = "simulation", stale: bool = False
) -> tuple[ApprovalService, ApprovalRequest, RequestContext, ToolDescriptor]:
    """Persist genuine domain transitions and one immutable synthetic Citation."""
    domain, evidence, approvals = (
        InMemoryReleaseGuardRepository(),
        InMemoryEvidenceRepository(),
        InMemoryApprovalRepository(),
    )
    request = request_fixture()
    item = evidence_fixture()
    if stale:
        item = item.model_copy(update={"freshness_until": NOW + timedelta(seconds=1)})
    await domain.create_change(request)
    run = await domain.create_run(new_change_run(request, "synthetic-run"))
    for status in (
        ChangeStatus.COLLECTING_EVIDENCE,
        ChangeStatus.EVIDENCE_SUFFICIENT,
        ChangeStatus.RISK_ASSESSED,
        ChangeStatus.PLAN_READY,
    ):
        inputs = edge_inputs(run, status)
        if status is ChangeStatus.EVIDENCE_SUFFICIENT:
            values = inputs["inputs"].model_dump()
            values["evidence_items"] = [item]
            inputs["inputs"] = type(inputs["inputs"]).model_validate(values)
        if status is ChangeStatus.RISK_ASSESSED:
            values = inputs["inputs"].model_dump()
            values["risk_assessment"] = risk_assessment(level=level)
            inputs["inputs"] = type(inputs["inputs"]).model_validate(values)
        if status is ChangeStatus.PLAN_READY and effect == "read_only":
            plan_data = plan_fixture().model_dump(exclude={"plan_hash", "record_type"})
            plan_data["steps"][0]["permission"] = "READ_ONLY"
            plan_data["permissions"] = ["READ_ONLY", "SIMULATED"]
            values = inputs["inputs"].model_dump()
            values["execution_plan"] = plan_fixture(**plan_data)
            inputs["inputs"] = type(inputs["inputs"]).model_validate(values)
        run = await domain.commit_run_update(command_fixture(run, status, **inputs))
    snapshot = EvidenceSnapshot(
        snapshot_id="sha256:"
        + canonical_hash(
            {
                "evidence_id": item.evidence_id,
                "source_uri": item.source_uri,
                "source_version": item.source_version,
                "content_hash": item.content_hash,
            }
        ),
        item=item,
        persisted_at=NOW,
    )
    await evidence.append_snapshot(snapshot)
    data = descriptor_fixture().model_dump(exclude={"descriptor_hash"})
    data.update(side_effect_level=effect, capabilities={effect}, idempotency_required=effect != "read_only")
    descriptor = ToolDescriptor.model_validate(data)
    call = invocation_fixture(
        descriptor,
        environment="staging",
        approval_id="synthetic-approval:decision",
        evidence_ids=(item.evidence_id,),
        citation_ids=("synthetic-citation",),
    )
    citation = EvidenceCitation(
        citation_id="synthetic-citation",
        snapshot_id=snapshot.snapshot_id,
        evidence_id=item.evidence_id,
        content_hash=item.content_hash,
        captured_at=NOW,
        required_item="ci",
        change_id=run.change_id,
        run_id=run.run_id,
        revision=run.revision,
        request_hash=run.request_hash,
        thread_id=call.thread_id,
        policy_version="synthetic-policy.v1",
        cited_at=NOW,
    )
    await evidence.append_citation(citation)
    context = RequestContext(principal=principal_fixture(), invocation=call, resource_tenant_id="synthetic-tenant")
    permissions = policy_fixture(
        rules=(rule_fixture(environments={"staging"}, side_effects={effect}, effect="approval_required"),)
    )
    service = ApprovalService(domain, evidence, approvals, permissions, evidence_policy_fixture())
    _, binding = await load_authorization_binding(
        domain, evidence, context, descriptor, permissions, service.evidence_policy, at=NOW
    )
    approval = ApprovalRequest(
        approval_request_id="synthetic-approval",
        idempotency_key="synthetic-approval-key",
        binding=binding,
        authorization_hash=canonical_hash(binding),
        expected_state_version=run.state_version,
        requested_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        reason_code="change_review_requested",
    )
    return service, approval, context, descriptor


def approver_context(context: RequestContext, **overrides: Any) -> RequestContext:
    """Simulate a fresh trusted identity resolution, never a resume-supplied actor."""
    actor = principal_fixture(principal_id="synthetic-approver", session_id="approver-session", **overrides)
    return RequestContext(
        principal=actor,
        invocation=context.invocation.model_copy(
            update={"principal_id": actor.principal_id, "session_id": actor.session_id}
        ),
        resource_tenant_id=context.resource_tenant_id,
    )


def decision_command(request: ApprovalRequest, **overrides: Any) -> ApprovalCommand:
    """Bind one deterministic intent command to the waiting state version."""
    return ApprovalCommand.model_validate(
        {
            "command_id": "synthetic-decision",
            "approval_request_id": request.approval_request_id,
            "authorization_hash": request.authorization_hash,
            "expected_state_version": request.expected_state_version + 1,
            "decision": "APPROVED",
            "reason_code": "change_review_approved",
            **overrides,
        }
    )


async def create_waiting(
    service: ApprovalService, request: ApprovalRequest, context: RequestContext, descriptor: ToolDescriptor
) -> None:
    """Save pending then commit its domain CAS, mirroring Graph write ordering."""
    entry = await service.create(request, context, descriptor, at=NOW)
    run = await service.domain.get_run(RunLookup(run_id=context.invocation.run_id))
    await service.domain.commit_run_update(approval_domain_command(run, entry.pending, command_id="approval-pending"))


@pytest.mark.parametrize(
    "decision,status", [("APPROVED", "APPROVED"), ("REJECTED", "REJECTED"), ("NEEDS_CHANGES", "WAITING_FOR_EVIDENCE")]
)
def test_approval_decisions_and_append_only_domain_records(decision: str, status: str) -> None:
    """All three decisions use existing domain transitions without execution."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        first = await service.create(request, context, descriptor, at=NOW)
        assert await service.create(request, context, descriptor, at=NOW) == first
        await create_waiting(service, request, context, descriptor)
        actor = approver_context(context)
        command = decision_command(request, decision=decision)
        entry = await service.decide(command, actor, descriptor, at=NOW)
        assert entry.pending.decision is ApprovalDecision.PENDING and entry.decision is not None
        assert await service.decide(command, actor, descriptor, at=NOW) == entry
        run = await service.domain.get_run(RunLookup(run_id=context.invocation.run_id))
        result = await service.domain.commit_run_update(
            approval_domain_command(run, entry.decision, command_id=command.command_id)
        )
        assert result.status == status and not result.execution_events
        assert len([record for record in result.record_history if record.record_type == "approval"]) == 2

    asyncio.run(scenario())


def test_high_risk_self_approval_denied() -> None:
    """Even a requester with an approver role cannot self-approve HIGH risk."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup(level="HIGH")
        await create_waiting(service, request, context, descriptor)
        with pytest.raises(DomainError, match="approval_self_denied"):
            await service.decide(decision_command(request), context, descriptor, at=NOW)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", [{"roles": {"reader"}}, {"tenant_id": "other-tenant"}, {"expires_at": NOW + timedelta(seconds=1)}]
)
def test_decision_rechecks_actor(change: dict[str, Any]) -> None:
    """Role/tenant/expiry failures do not create decisions."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        with pytest.raises(DomainError):
            await service.decide(
                decision_command(request),
                approver_context(context, **change),
                descriptor,
                at=NOW + timedelta(seconds=2),
            )
        assert (await service.approvals.get(request.approval_request_id)).decision is None

    asyncio.run(scenario())


def test_expiry_revocation_and_original_decision_preserved() -> None:
    """Terminal invalidation facts stop decisions without overwriting originals."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        actor = approver_context(context)
        with pytest.raises(DomainError, match="approval_expired"):
            await service.decide(decision_command(request), actor, descriptor, at=request.expires_at)
        entry = await service.approvals.get(request.approval_request_id)
        assert entry.invalidations[0].decision is ApprovalDecision.EXPIRED
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        entry = await service.decide(decision_command(request), approver_context(context), descriptor, at=NOW)
        revoked = await service.invalidate(
            request.approval_request_id, context, ApprovalDecision.REVOKED, "approval_revoked", at=NOW
        )
        assert revoked.decision == entry.decision
        assert revoked.invalidations[0].decision is ApprovalDecision.REVOKED
        with pytest.raises(DomainError, match="approval_invalidated"):
            await service.decide(decision_command(request), approver_context(context), descriptor, at=NOW)

    asyncio.run(scenario())


def test_concurrent_decisions_and_idempotency_conflict() -> None:
    """Identical concurrent intents converge; opposite intents cannot overwrite."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        first = await service.create(request, context, descriptor, at=NOW)
        changed = request.model_copy(update={"reason_code": "different_reason"})
        with pytest.raises(DomainError, match="idempotency_conflict"):
            await service.create(changed, context, descriptor, at=NOW)
        await create_waiting(service, request, context, descriptor)
        command, actor = decision_command(request), approver_context(context)
        results = await asyncio.gather(*(service.decide(command, actor, descriptor, at=NOW) for _ in range(8)))
        assert all(entry == results[0] for entry in results)
        with pytest.raises(DomainError, match="decision_conflict"):
            await service.decide(decision_command(request, decision="REJECTED"), actor, descriptor, at=NOW)
        assert first.pending == results[0].pending

    asyncio.run(scenario())


def test_policy_change_invalidates_old_binding() -> None:
    """A same-version policy content change cannot reuse an old approval seal."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        service.permission_policy = service.permission_policy.model_copy(
            update={
                "rules": (
                    *service.permission_policy.rules,
                    rule_fixture(
                        rule_id="another",
                        environments={"staging"},
                        side_effects={"simulation"},
                        effect="approval_required",
                    ),
                )
            }
        )
        with pytest.raises(DomainError, match="binding_changed"):
            await service.decide(decision_command(request), approver_context(context), descriptor, at=NOW)
        assert (await service.approvals.get(request.approval_request_id)).invalidations

    asyncio.run(scenario())


def test_missing_citations_unknown_risk_and_uncertainty_block_creation() -> None:
    """Gate results and structured risk cannot be weakened by request fields."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        bad = context.model_copy(update={"invocation": context.invocation.model_copy(update={"citation_ids": ()})})
        with pytest.raises(DomainError, match="evidence_refs_missing"):
            await service.create(request, bad, descriptor, at=NOW)
        service.evidence_policy = service.evidence_policy.model_copy(update={"requirements": []})
        with pytest.raises(DomainError, match="evidence_gate_blocked"):
            await service.create(request, context, descriptor, at=NOW)

    asyncio.run(scenario())


def test_decision_replay_with_new_clock_and_authentication_time() -> None:
    """Fresh authentication cannot duplicate a previously committed decision/audit."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        command = decision_command(request)
        first = await service.decide(command, approver_context(context), descriptor, at=NOW)
        actor = approver_context(context, authenticated_at=NOW + timedelta(seconds=1))
        replay = await service.decide(command, actor, descriptor, at=NOW + timedelta(seconds=2))
        assert first == replay
        events = (await service.approvals.export_snapshot()).audit_events
        assert len([event for event in events if event.event_type == "approval_decided"]) == 1
        assert first.decision.decided_at == NOW

    asyncio.run(scenario())


@pytest.mark.parametrize("drift", ["descriptor", "policy_version", "expired_evidence"])
def test_preflight_drift_preserves_invalidation_fact(drift: str) -> None:
    """Objective drift records invalidation even when the Gate cannot return a seal."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup(stale=drift == "expired_evidence")
        await create_waiting(service, request, context, descriptor)
        if drift == "descriptor":
            data = descriptor.model_dump(exclude={"descriptor_hash"})
            data["output_schema"] = "observation.v2"
            descriptor = ToolDescriptor.model_validate(data)
        elif drift == "policy_version":
            service.permission_policy = service.permission_policy.model_copy(update={"policy_version": "policy.v2"})
        with pytest.raises(DomainError):
            await service.decide(
                decision_command(request), approver_context(context), descriptor, at=NOW + timedelta(seconds=2)
            )
        entry = await service.approvals.get(request.approval_request_id)
        assert entry.decision is None and entry.invalidations[0].decision is ApprovalDecision.INVALIDATED
        assert "approval_invalidated" in {
            event.event_type for event in (await service.approvals.export_snapshot()).audit_events
        }

    asyncio.run(scenario())


def test_unrelated_context_cannot_revoke_approval() -> None:
    """A same-tenant requester cannot revoke a different target or forge its session."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        bad = context.model_copy(
            update={"invocation": context.invocation.model_copy(update={"resource_id": "another-resource"})}
        )
        with pytest.raises(DomainError, match="actor_denied"):
            await service.invalidate(
                request.approval_request_id, bad, ApprovalDecision.REVOKED, "approval_revoked", at=NOW
            )
        assert not (await service.approvals.get(request.approval_request_id)).invalidations

    asyncio.run(scenario())
