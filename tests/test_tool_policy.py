"""Protected-tool Gate, approval, audit and zero-side-effect tests."""

import asyncio
from datetime import timedelta

import pytest

from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalRequest
from app.schemas.authorization import RequestContext
from app.schemas.releaseguard import ApprovalDecision
from app.schemas.tooling import ToolDescriptor, ToolResult
from app.services.releaseguard.approval import approval_domain_command
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.tool_policy import ToolAuthorizationPolicy
from app.services.tools.registry import ToolRegistry
from tests.test_approval_contract import approval_setup, approver_context, create_waiting, decision_command
from tests.test_change_request_contract import NOW
from tests.test_tool_contract import FakeTool


async def approved_setup(
    *, effect: str = "simulation", stale: bool = False
) -> tuple[ToolAuthorizationPolicy, ApprovalRequest, RequestContext, ToolDescriptor, FakeTool]:
    """Build a fully bound decision and commit its genuine domain command."""
    service, request, context, descriptor = await approval_setup(effect=effect, stale=stale)
    await create_waiting(service, request, context, descriptor)
    command = decision_command(request)
    entry = await service.decide(command, approver_context(context), descriptor, at=NOW)
    run = await service.domain.get_run(RunLookup(run_id=context.invocation.run_id))
    await service.domain.commit_run_update(approval_domain_command(run, entry.decision, command_id=command.command_id))
    registry = ToolRegistry()
    adapter = FakeTool(descriptor)
    await registry.register(descriptor, adapter)
    return ToolAuthorizationPolicy(service, registry), request, context, descriptor, adapter


def test_authorization_is_not_execution() -> None:
    """A valid simulated authorization never invokes a Phase 6 adapter."""

    async def scenario() -> None:
        policy, _, context, descriptor, adapter = await approved_setup()
        authorization = await policy.authorize(context, descriptor, at=NOW)
        assert authorization.decision == "allow" and authorization.authorization_hash
        assert (await policy.invoke(context, at=NOW)).status == "not_implemented"
        assert adapter.calls == 0
        events = (await policy.service.approvals.export_snapshot()).audit_events
        assert {event.event_type for event in events} >= {
            "tool_invocation_requested",
            "permission_evaluated",
            "approval_requested",
            "approval_decided",
            "tool_call_not_implemented",
        }

    asyncio.run(scenario())


def test_read_only_call_and_exact_replay() -> None:
    """Permission and Gate run again before returning a previous safe result."""

    async def scenario() -> None:
        policy, _, context, _, adapter = await approved_setup(effect="read_only")
        first = await policy.invoke(context, at=NOW)
        assert first.status == "succeeded" and first.audit_ids
        assert await policy.invoke(context, at=NOW) == first
        assert adapter.calls == 1
        denied = context.model_copy(update={"resource_tenant_id": "other"})
        assert (await policy.invoke(denied, at=NOW)).status == "permission_denied"
        assert adapter.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("expired", "expired"),
        ("revoked", "revoked"),
        ("evidence_expired", "permission_denied"),
        ("policy_unavailable", "unavailable"),
        ("policy_changed", "conflicting"),
        ("missing", "permission_denied"),
    ],
)
def test_protected_call_blockers(mode: str, expected: str) -> None:
    """Invalid evidence/approvals/policy block before any adapter call."""

    async def scenario() -> None:
        policy, request, context, descriptor, adapter = await approved_setup(stale=mode == "evidence_expired")
        at = NOW
        if mode == "expired":
            at = request.expires_at
        elif mode == "revoked":
            await policy.service.invalidate(
                request.approval_request_id, context, ApprovalDecision.REVOKED, "approval_revoked", at=at
            )
        elif mode == "evidence_expired":
            at += timedelta(seconds=2)
        elif mode == "policy_unavailable":
            policy.service.permission_policy = policy.service.permission_policy.model_copy(update={"available": False})
        elif mode == "policy_changed":
            rule = policy.service.permission_policy.rules[0].model_copy(update={"rule_id": "changed"})
            policy.service.permission_policy = policy.service.permission_policy.model_copy(update={"rules": (rule,)})
        elif mode == "missing":
            context = context.model_copy(
                update={"invocation": context.invocation.model_copy(update={"citation_ids": ()})}
            )
        result = await policy.invoke(context, at=at)
        assert result.status == expected
        assert adapter.calls == 0
        assert await policy.service.approvals.export_snapshot()
        assert descriptor.descriptor_hash == context.invocation.descriptor_hash

    asyncio.run(scenario())


def test_gate_sufficient_does_not_approve() -> None:
    """A pending approval returns approval_required even with sufficient evidence."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        registry, adapter = ToolRegistry(), FakeTool(descriptor)
        await registry.register(descriptor, adapter)
        result = await ToolAuthorizationPolicy(service, registry).invoke(context, at=NOW)
        assert result.status == "approval_required" and adapter.calls == 0

    asyncio.run(scenario())


def test_unknown_completion_stops_retry() -> None:
    """A transport/audit failure leaves an unresolved claim, not another call."""

    async def scenario() -> None:
        policy, _, context, _, adapter = await approved_setup(effect="read_only")
        original = policy.service.approvals.complete_call

        async def fail_completion(invocation_id: str, result: ToolResult) -> None:
            raise DomainError("synthetic_completion_failure")

        policy.service.approvals.complete_call = fail_completion
        with pytest.raises(DomainError, match="synthetic_completion_failure"):
            await policy.invoke(context, at=NOW)
        policy.service.approvals.complete_call = original
        with pytest.raises(DomainError, match="tool_completion_unknown"):
            await policy.invoke(context, at=NOW)
        assert adapter.calls == 1
        assert "consistency_stopped" in {
            event.event_type for event in (await policy.service.approvals.export_snapshot()).audit_events
        }

    asyncio.run(scenario())
