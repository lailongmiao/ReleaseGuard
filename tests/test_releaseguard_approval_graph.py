"""Single-graph typed approval wait, authorization and fault/replay tests."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.core.langgraph.graph import build_agent_graph
from app.core.langgraph.releaseguard_approval import ApprovalRuntime, resume_approval
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.repositories.releaseguard import RunLookup
from app.schemas.authorization import RequestContext
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ApprovalDecision, ChangeRunState, RunCommand
from app.schemas.tooling import ToolInvocation
from app.services.releaseguard.approval import approval_domain_command
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.registry import ToolRegistry
from tests.test_approval_contract import approval_setup, approver_context, decision_command
from tests.releaseguard_fixtures import command_fixture, plan_fixture
from tests.test_change_request_contract import NOW
from tests.test_releaseguard_repository import variant
from tests.test_releaseguard_graph_integration import _chat_passthrough, _tool_passthrough
from tests.test_tool_contract import FakeTool


class TrustedResolver:
    """Synthetic trusted session boundary kept outside checkpoint and commands."""

    def __init__(self, context: RequestContext) -> None:
        """Keep synthetic identity outside all checkpointed state."""
        self.context = context

    async def resolve(self, invocation: ToolInvocation, config: RunnableConfig) -> RequestContext:
        """Return a freshly supplied trusted context without reading resume values."""
        return self.context


async def graph_setup() -> tuple[
    CompiledStateGraph, ApprovalRuntime, TrustedResolver, dict[str, datetime], GraphState, RunnableConfig, FakeTool
]:
    """Reuse genuine Phase 4/5 domain, Evidence and approval dependencies."""
    service, request, context, descriptor = await approval_setup()
    registry, adapter = ToolRegistry(), FakeTool(descriptor)
    await registry.register(descriptor, adapter)
    resolver = TrustedResolver(context)
    clock = {"at": NOW}
    runtime = ApprovalRuntime(service, registry, resolver, lambda: clock["at"])
    graph = build_agent_graph(
        _chat_passthrough,
        _tool_passthrough,
        InMemorySaver(),
        service.domain,
        service.evidence,
        approval_runtime=runtime,
    )
    run = await service.domain.get_run(RunLookup(run_id=context.invocation.run_id))
    state = GraphState(
        workflow_kind="releaseguard",
        releaseguard=ReleaseGuardGraphContext(
            change_request=run.change_request,
            run_id=run.run_id,
            expected_revision=run.revision,
            approval_request=request,
        ),
    )
    config = releaseguard_runnable_config(run.change_id, run.run_id, run.revision)
    return graph, runtime, resolver, clock, state, config, adapter


@pytest.mark.parametrize(
    "decision,status,outcome",
    [
        ("APPROVED", "APPROVED", "unsupported_execution"),
        ("REJECTED", "REJECTED", "terminal"),
        ("NEEDS_CHANGES", "WAITING_FOR_EVIDENCE", "awaiting_evidence"),
    ],
)
def test_graph_approval_decisions_and_chat_regression(decision: str, status: str, outcome: str) -> None:
    """The same Graph preserves Chat and routes typed decisions without an executor."""

    async def scenario() -> None:
        graph, runtime, resolver, _, state, config, adapter = await graph_setup()
        chat = await graph.ainvoke(
            GraphState(long_term_memory="synthetic memory"), {"configurable": {"thread_id": "synthetic-chat"}}
        )
        assert chat["workflow_kind"] == "chat"
        paused = await graph.ainvoke(state, config)
        payload = paused["__interrupt__"][0].value
        assert payload["kind"] == "releaseguard.approval.v1"
        request = state.releaseguard.approval_request
        assert payload["request"]["authorization_hash"] == request.authorization_hash
        resolver.context = approver_context(resolver.context)
        command = decision_command(request, decision=decision)
        result = await resume_approval(graph, command, config)
        domain = result["releaseguard"].domain_state
        assert domain.status == status and result["releaseguard"].routing_outcome == outcome
        assert domain.state_version == request.expected_state_version + 2
        assert len([record for record in domain.record_history if record.record_type == "approval"]) == 2
        assert adapter.calls == 0 and not domain.execution_events
        with pytest.raises(DomainError, match="checkpoint_missing_or_completed"):
            await resume_approval(graph, command, config)
        assert (await runtime.service.approvals.get(request.approval_request_id)).decision.decision == decision

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["wrong_thread", "wrong_revision", "wrong_hash", "wrong_version", "missing"])
def test_resume_entry_binding_and_missing_checkpoint(mode: str) -> None:
    """Reject absent, forked or stale checkpoint before a decision is persisted."""

    async def scenario() -> None:
        graph, runtime, resolver, _, state, config, adapter = await graph_setup()
        if mode != "missing":
            await graph.ainvoke(state, config)
        request = state.releaseguard.approval_request
        resolver.context = approver_context(resolver.context)
        command = decision_command(request)
        if mode == "wrong_thread":
            config = {**config, "configurable": {"thread_id": "another-thread"}}
        elif mode == "wrong_revision":
            config = {**config, "metadata": {**config["metadata"], "revision": 2}}
        elif mode == "wrong_hash":
            command = command.model_copy(update={"authorization_hash": "f" * 64})
        elif mode == "wrong_version":
            command = command.model_copy(update={"expected_state_version": command.expected_state_version + 1})
        with pytest.raises(DomainError):
            await resume_approval(graph, command, config)
        if mode != "missing":
            assert (await runtime.service.approvals.get(request.approval_request_id)).decision is None
        assert adapter.calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["tenant", "role", "session"])
def test_resolver_identity_is_rechecked_on_resume(mode: str) -> None:
    """Resume fields cannot provide trusted tenant/role/session attribution."""

    async def scenario() -> None:
        graph, runtime, resolver, _, state, config, adapter = await graph_setup()
        await graph.ainvoke(state, config)
        changes: dict[str, Any] = {"tenant_id": "other"} if mode == "tenant" else {"roles": {"reader"}}
        resolver.context = approver_context(resolver.context, **changes)
        if mode == "session":
            resolver.context = resolver.context.model_copy(
                update={"invocation": resolver.context.invocation.model_copy(update={"session_id": "forged-session"})}
            )
        with pytest.raises(DomainError):
            await resume_approval(graph, decision_command(state.releaseguard.approval_request), config)
        assert (
            await runtime.service.approvals.get(state.releaseguard.approval_request.approval_request_id)
        ).decision is None
        assert adapter.calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["expired", "revoked"])
def test_invalidated_approval_stops_without_domain_execution(mode: str) -> None:
    """Cancellation/expiry stop in a structured Graph outcome without Phase 6 nodes."""

    async def scenario() -> None:
        graph, runtime, resolver, clock, state, config, adapter = await graph_setup()
        await graph.ainvoke(state, config)
        request = state.releaseguard.approval_request
        if mode == "expired":
            clock["at"] = request.expires_at
        else:
            await runtime.service.invalidate(
                request.approval_request_id, resolver.context, ApprovalDecision.REVOKED, "approval_revoked", at=NOW
            )
        resolver.context = approver_context(resolver.context)
        result = await resume_approval(graph, decision_command(request), config)
        assert result["releaseguard"].approval_outcome == mode
        assert result["releaseguard"].domain_state.status == "WAITING_FOR_APPROVAL"
        assert adapter.calls == 0 and not result["releaseguard"].domain_state.execution_events

    asyncio.run(scenario())


@pytest.mark.parametrize("point", ["pending_before", "pending_after", "decision_before", "decision_after"])
@pytest.mark.parametrize("decision", ["APPROVED", "REJECTED", "NEEDS_CHANGES"])
def test_domain_commit_fault_windows_replay_once(point: str, decision: str) -> None:
    """Repository/checkpoint gaps reconcile exact facts rather than duplicating state."""

    async def scenario() -> None:
        graph, runtime, resolver, clock, state, config, adapter = await graph_setup()
        domain = runtime.service.domain
        original = domain.commit_run_update
        failed = False

        async def faulty(command: RunCommand) -> ChangeRunState:
            nonlocal failed
            target = (
                "WAITING_FOR_APPROVAL"
                if point.startswith("pending")
                else {"APPROVED": "APPROVED", "REJECTED": "REJECTED", "NEEDS_CHANGES": "WAITING_FOR_EVIDENCE"}[
                    decision
                ]
            )
            if not failed and command.target_status == target:
                failed = True
                if point.endswith("after"):
                    await original(command)
                raise RuntimeError("synthetic_checkpoint_gap")
            return await original(command)

        domain.commit_run_update = faulty
        request = state.releaseguard.approval_request
        if point.startswith("pending"):
            with pytest.raises(RuntimeError, match="synthetic_checkpoint_gap"):
                await graph.ainvoke(state, config)
            paused = await graph.ainvoke(None, config)
            assert paused["__interrupt__"]
        else:
            await graph.ainvoke(state, config)
        resolver.context = approver_context(resolver.context)
        command = decision_command(request, decision=decision)
        if point.startswith("decision"):
            with pytest.raises(RuntimeError, match="synthetic_checkpoint_gap"):
                await resume_approval(graph, command, config)
            clock["at"] += timedelta(seconds=1)
        result = await resume_approval(graph, command, config)
        run = result["releaseguard"].domain_state
        assert (
            run.status
            == {"APPROVED": "APPROVED", "REJECTED": "REJECTED", "NEEDS_CHANGES": "WAITING_FOR_EVIDENCE"}[decision]
            and run.state_version == request.expected_state_version + 2
        )
        assert (
            len(
                [
                    event
                    for event in (await runtime.service.approvals.export_snapshot()).audit_events
                    if event.event_type == "approval_decided"
                ]
            )
            == 1
        )
        assert adapter.calls == 0

    asyncio.run(scenario())


def test_failed_resume_cannot_substitute_a_different_decision() -> None:
    """A stored interrupt intent cannot silently replace a new opposite intent."""

    async def scenario() -> None:
        graph, runtime, resolver, _, state, config, adapter = await graph_setup()
        await graph.ainvoke(state, config)
        original = runtime.service.domain.commit_run_update
        failed = False

        async def faulty(command: RunCommand) -> ChangeRunState:
            nonlocal failed
            if command.target_status == "APPROVED" and not failed:
                failed = True
                raise RuntimeError("synthetic_checkpoint_gap")
            return await original(command)

        runtime.service.domain.commit_run_update = faulty
        resolver.context = approver_context(resolver.context)
        request = state.releaseguard.approval_request
        with pytest.raises(RuntimeError):
            await resume_approval(graph, decision_command(request), config)
        with pytest.raises(DomainError, match="intent_conflict|decision_conflict"):
            await resume_approval(graph, decision_command(request, decision="REJECTED"), config)
        assert (
            await runtime.service.domain.get_run(RunLookup(run_id=state.releaseguard.run_id))
        ).status == "WAITING_FOR_APPROVAL"
        assert adapter.calls == 0

    asyncio.run(scenario())


def test_resume_rejects_caller_selected_historical_checkpoint() -> None:
    """Approval recovery always reads the latest root thread, never a chosen fork."""

    async def scenario() -> None:
        graph, _, resolver, _, state, config, _ = await graph_setup()
        await graph.ainvoke(state, config)
        snapshot = await graph.aget_state(config)
        resolver.context = approver_context(resolver.context)
        config = {**config, "configurable": snapshot.config["configurable"]}
        with pytest.raises(DomainError, match="checkpoint_fork"):
            await resume_approval(graph, decision_command(state.releaseguard.approval_request), config)

    asyncio.run(scenario())


@pytest.mark.parametrize("drift", ["revision", "plan"])
def test_waiting_checkpoint_cannot_approve_replaced_domain_inputs(drift: str) -> None:
    """Authoritative revision/plan drift invalidates the original approval request."""

    async def scenario() -> None:
        graph, runtime, resolver, _, state, config, adapter = await graph_setup()
        await graph.ainvoke(state, config)
        run = await runtime.service.domain.get_run(RunLookup(run_id=state.releaseguard.run_id))
        if drift == "revision":
            command = command_fixture(
                run, action="revise", replacement_request=variant(revision=2, title="synthetic revision")
            )
        else:
            command = command_fixture(
                run,
                action="replace_plan",
                replacement_plan=plan_fixture(
                    plan_id="replacement-plan", verification_plan="synthetic revised verification"
                ),
            )
        await runtime.service.domain.commit_run_update(command)
        resolver.context = approver_context(resolver.context)
        request = state.releaseguard.approval_request
        with pytest.raises(DomainError):
            await resume_approval(graph, decision_command(request), config)
        entry = await runtime.service.approvals.get(request.approval_request_id)
        assert entry.decision is None and entry.invalidations
        assert adapter.calls == 0

    asyncio.run(scenario())


def test_prebuilt_approval_command_cannot_bypass_service() -> None:
    """Generic Phase 3 continuation cannot submit approval when Phase 5 is enabled."""

    async def scenario() -> None:
        graph, runtime, resolver, _, state, config, _ = await graph_setup()
        request = state.releaseguard.approval_request
        descriptor = runtime.registry.descriptor(
            request.binding.invocation.tool_name, request.binding.invocation.version
        )
        entry = await runtime.service.create(request, resolver.context, descriptor, at=NOW)
        run = await runtime.service.domain.get_run(RunLookup(run_id=state.releaseguard.run_id))
        command = approval_domain_command(run, entry.pending, command_id="bypass-command")
        context = ReleaseGuardGraphContext.model_validate(
            {
                **state.releaseguard.model_dump(),
                "domain_state": run,
                "state_version": run.state_version,
                "pending_command": command,
            }
        )
        with pytest.raises(DomainError, match="approval_service_required"):
            await graph.ainvoke(state.model_copy(update={"releaseguard": context}), config)
        assert (await runtime.service.domain.get_run(RunLookup(run_id=run.run_id))).status == "PLAN_READY"

    asyncio.run(scenario())
