"""Offline audit transfer, atomic rejection, races and unknown-completion checks."""

import asyncio
from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from app.repositories.in_memory_approval import InMemoryApprovalRepository
from app.repositories.in_memory_evidence import InMemoryEvidenceRepository
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.approval import ApprovalStoreSnapshot
from app.schemas.releaseguard import ApprovalDecision, ChangeStatus, new_change_run
from app.services.releaseguard.approval import ApprovalService
from app.services.releaseguard.authorization_context import load_authorization_binding
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.tool_policy import ToolAuthorizationPolicy
from tests.releaseguard_fixtures import command_fixture, edge_inputs, risk_assessment
from tests.test_approval_contract import approval_setup, approver_context, create_waiting, decision_command
from tests.test_change_request_contract import NOW, request_fixture
from tests.test_tool_policy import approved_setup


def test_three_repository_restore_and_exact_tool_replay() -> None:
    """Restored test stores retain first facts without rerunning safe transport."""

    async def scenario() -> None:
        policy, request, context, descriptor, adapter = await approved_setup(effect="read_only")
        result = await policy.invoke(context, at=NOW)
        original = policy.service
        approvals = InMemoryApprovalRepository()
        evidence = InMemoryEvidenceRepository()
        domain = InMemoryReleaseGuardRepository()
        snapshot = await original.approvals.export_snapshot()
        await approvals.restore_snapshot(ApprovalStoreSnapshot.model_validate_json(snapshot.model_dump_json()))
        # Evidence uses append/get Protocol methods, not a bulk restore API.
        for identifier in context.invocation.citation_ids:
            citation = await original.evidence.get_citation(identifier)
            source = await original.evidence.get_snapshot(citation.snapshot_id)
            await evidence.append_snapshot(type(source).model_validate_json(source.model_dump_json()))
            await evidence.append_citation(type(citation).model_validate_json(citation.model_dump_json()))
        await domain.restore_snapshot(await original.domain.export_snapshot())
        restored = ApprovalService(domain, evidence, approvals, original.permission_policy, original.evidence_policy)
        assert await ToolAuthorizationPolicy(restored, policy.registry).invoke(context, at=NOW) == result
        first = await approvals.get(request.approval_request_id)
        assert await restored.decide(decision_command(request), approver_context(context), descriptor, at=NOW) == first
        assert await approvals.export_snapshot() == snapshot
        assert adapter.calls == 1
        run = await domain.get_run(RunLookup(run_id=context.invocation.run_id))
        assert run.status is ChangeStatus.APPROVED and run.receipt is None and not run.execution_events

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode",
    [
        "missing_audit",
        "orphan_audit",
        "wrong_actor",
        "wrong_session",
        "wrong_time",
        "wrong_hash",
        "wrong_reason",
        "duplicate_entry",
        "duplicate_call",
        "wrong_result_ref",
        "wrong_call_actor",
        "wrong_call_hash",
        "wrong_call_authorization",
        "wrong_completion_reason",
        "early_completion",
    ],
)
def test_corrupted_restore_is_atomic(mode: str) -> None:
    """No partially validated approval, audit or call can reach the destination."""

    async def scenario() -> None:
        policy, _, context, _, _ = await approved_setup(effect="read_only")
        await policy.invoke(context, at=NOW)
        data: dict[str, Any] = (await policy.service.approvals.export_snapshot()).model_dump(mode="python")
        data["entries"] = list(data["entries"])
        data["calls"] = list(data["calls"])
        data["audit_events"] = list(data["audit_events"])
        pending = next(item for item in data["audit_events"] if item["event_type"] == "approval_requested")
        completion = next(item for item in data["audit_events"] if item["event_type"] == "tool_call_completed")
        if mode == "missing_audit":
            data["audit_events"].remove(pending)
        elif mode == "orphan_audit":
            pending["approval_id"] = "missing:pending"
        elif mode == "wrong_actor":
            pending["principal_id"] = "another-principal"
        elif mode == "wrong_session":
            pending["session_id"] = "another-session"
        elif mode == "wrong_time":
            pending["occurred_at"] = NOW + timedelta(seconds=1)
        elif mode == "wrong_hash":
            pending["input_hash"] = "e" * 64
        elif mode == "wrong_reason":
            pending["reason_code"] = "different_reason"
        elif mode == "duplicate_entry":
            data["entries"].append(data["entries"][0])
        elif mode == "duplicate_call":
            data["calls"].append(data["calls"][0])
        elif mode == "wrong_result_ref":
            data["calls"][0]["result"]["audit_ids"] = (pending["event_id"],) * 3
        elif mode == "wrong_call_actor":
            data["calls"][0]["invocation"]["principal_id"] = "another-principal"
        elif mode == "wrong_call_hash":
            data["calls"][0]["invocation"]["input_hash"] = "e" * 64
        elif mode == "wrong_call_authorization":
            data["calls"][0]["authorization_hash"] = "e" * 64
        elif mode == "wrong_completion_reason":
            completion["reason_code"] = "different_reason"
        else:
            completion["occurred_at"] = NOW - timedelta(seconds=1)
        target = InMemoryApprovalRepository()
        with pytest.raises((DomainError, ValidationError)):
            await target.restore_snapshot(ApprovalStoreSnapshot.model_validate(data))
        assert await target.export_snapshot() == ApprovalStoreSnapshot()

    asyncio.run(scenario())


def test_nonempty_restore_and_conflicting_audit_preserve_facts() -> None:
    """Existing work cannot be replaced even by a validated incoming transfer."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        await create_waiting(service, request, context, descriptor)
        snapshot = await service.approvals.export_snapshot()
        event = snapshot.audit_events[0]
        assert await service.approvals.append_audit(event) == event
        with pytest.raises(DomainError, match="audit_conflict"):
            await service.approvals.append_audit(event.model_copy(update={"reason_code": "other_reason"}))
        with pytest.raises(DomainError, match="restore_requires_empty"):
            await service.approvals.restore_snapshot(snapshot)
        assert await service.approvals.export_snapshot() == snapshot

    asyncio.run(scenario())


def test_concurrent_creation_opposite_decisions_and_revocation() -> None:
    """One local transaction wins; competing intents never overwrite history."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        entries = await asyncio.gather(*(service.create(request, context, descriptor, at=NOW) for _ in range(8)))
        assert all(item == entries[0] for item in entries)
        await create_waiting(service, request, context, descriptor)
        outcomes = await asyncio.gather(
            service.decide(decision_command(request), approver_context(context), descriptor, at=NOW),
            service.decide(
                decision_command(request, decision="REJECTED"), approver_context(context), descriptor, at=NOW
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(item, DomainError) for item in outcomes) == 1
        first = await service.approvals.get(request.approval_request_id)
        await asyncio.gather(
            *(
                service.invalidate(
                    request.approval_request_id, context, ApprovalDecision.REVOKED, "approval_revoked", at=NOW
                )
                for _ in range(8)
            )
        )
        after = await service.approvals.get(request.approval_request_id)
        assert after.decision == first.decision and len(after.invalidations) == 1
        snapshot = await service.approvals.export_snapshot()
        assert len(snapshot.audit_events) == 3
        clone = InMemoryApprovalRepository()
        await clone.restore_snapshot(snapshot)
        assert await clone.export_snapshot() == snapshot

    asyncio.run(scenario())


def test_concurrent_read_call_and_unresolved_transfer_stop() -> None:
    """The claim precedes the awaited adapter, so a race cannot double-call."""

    async def scenario() -> None:
        policy, _, context, _, adapter = await approved_setup(effect="read_only")
        original = adapter.invoke
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(call: Any) -> Any:
            entered.set()
            await release.wait()
            return await original(call)

        adapter.invoke = blocked
        task = asyncio.create_task(policy.invoke(context, at=NOW))
        await entered.wait()
        snapshot = await policy.service.approvals.export_snapshot()
        clone = InMemoryApprovalRepository()
        await clone.restore_snapshot(snapshot)
        service = policy.service
        restored = ApprovalService(
            service.domain, service.evidence, clone, service.permission_policy, service.evidence_policy
        )
        with pytest.raises(DomainError, match="tool_completion_unknown"):
            await ToolAuthorizationPolicy(restored, policy.registry).invoke(context, at=NOW)
        with pytest.raises(DomainError, match="tool_completion_unknown"):
            await policy.invoke(context, at=NOW)
        release.set()
        assert (await task).status == "succeeded"
        assert adapter.calls == 1
        assert "consistency_stopped" in {event.event_type for event in (await clone.export_snapshot()).audit_events}

    asyncio.run(scenario())


def test_failed_tool_is_a_fact_not_an_error_cache() -> None:
    """Prior failure neither returns a cached error response nor retries transport."""

    async def scenario() -> None:
        policy, _, context, _, adapter = await approved_setup(effect="read_only")
        adapter.mode = "exception"
        first = await policy.invoke(context, at=NOW)
        assert first.status == "unavailable"
        adapter.mode = "ok"
        retry = await policy.invoke(context, at=NOW)
        assert retry.status == "rejected" and retry.error_code == "tool_prior_failure"
        assert adapter.calls == 1
        assert "synthetic_sensitive_body" not in first.model_dump_json()

    asyncio.run(scenario())


@pytest.mark.parametrize("level,uncertainty", [("UNKNOWN", ["unresolved"]), ("LOW", ["unresolved"])])
def test_structured_unknown_and_uncertainty_block_authorization(level: str, uncertainty: list[str]) -> None:
    """Use real transitions to reach unsafe risk; no impossible PLAN_READY fixture."""

    async def scenario() -> None:
        service, _, context, descriptor = await approval_setup()
        domain = InMemoryReleaseGuardRepository()
        request = request_fixture()
        await domain.create_change(request)
        run = await domain.create_run(new_change_run(request, context.invocation.run_id))
        for status in (ChangeStatus.COLLECTING_EVIDENCE, ChangeStatus.EVIDENCE_SUFFICIENT, ChangeStatus.RISK_ASSESSED):
            inputs = edge_inputs(run, status)
            if status is ChangeStatus.RISK_ASSESSED:
                values = inputs["inputs"].model_dump()
                values["risk_assessment"] = risk_assessment(level=level, uncertainty=uncertainty)
                inputs["inputs"] = type(inputs["inputs"]).model_validate(values)
            run = await domain.commit_run_update(command_fixture(run, status, **inputs))
        if level == "UNKNOWN":
            with pytest.raises(DomainError):
                await domain.commit_run_update(
                    command_fixture(run, ChangeStatus.PLAN_READY, **edge_inputs(run, ChangeStatus.PLAN_READY))
                )
        else:
            await domain.commit_run_update(
                command_fixture(run, ChangeStatus.PLAN_READY, **edge_inputs(run, ChangeStatus.PLAN_READY))
            )
        with pytest.raises(DomainError, match="evidence_or_risk_blocked"):
            await load_authorization_binding(
                domain,
                service.evidence,
                context,
                descriptor,
                service.permission_policy,
                service.evidence_policy,
                at=NOW,
            )

    asyncio.run(scenario())
