"""Single Graph offline execution, waiting and typed root resume."""

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import build_agent_graph
from app.core.langgraph.releaseguard_execution import ExecutionRuntime, resume_execution
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.repositories.in_memory_execution import InMemoryExecutionRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.execution import ExecutionIntent
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ChangeStatus, StateInputs
from app.services.releaseguard.execution_commands import execution_command
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.verification import VerificationService
from app.services.tools.sandbox_executor import SandboxExecutor, SyntheticTargetStore
from app.services.tools.verification_adapter import IndependentRegisterReader
from tests.execution_fixtures import TrustedResolver, execution_setup, request_for
from tests.test_change_request_contract import NOW
from tests.test_controlled_execution import Fault
from tests.test_releaseguard_graph_integration import _chat_passthrough, _tool_passthrough


def test_client_fact_append_is_rejected_before_loading_execution() -> None:
    """Service-only action is rejected even when its fields contain no execution facts."""

    async def scenario() -> None:
        graph, runtime, target, state, config = await graph_execution_setup()
        service = runtime.execution.preflight.service
        run = await service.domain.get_run(RunLookup(run_id="synthetic-run"))
        command = execution_command(
            run,
            runtime.execution.preflight.resolver.context,
            command_id="client-facts",
            at=NOW,
            code="client_facts",
            inputs=StateInputs(evidence_assessment=run.evidence_assessment),
        )
        state = state.model_copy(
            update={"releaseguard": state.releaseguard.model_copy(update={"pending_command": command})}
        )
        with pytest.raises(DomainError, match="execution_service_required"):
            await graph.ainvoke(state, config)
        assert target.calls == 0 and target.reads == 0
        assert (await service.domain.get_run(RunLookup(run_id=run.run_id))).state_version == run.state_version

    asyncio.run(scenario())


async def graph_execution_setup(*, simulation: bool = False, faults: Fault | None = None, count: int = 1) -> tuple:
    """Inject only finite synthetic targets with genuine Phase 5 approval."""
    service, actor, descriptor = await execution_setup(simulation=simulation, count=count)
    target = SyntheticTargetStore({"synthetic-resource": 0})
    execution = ExecutionService(
        ExecutionPreflight(service, TrustedResolver(actor), descriptor),
        InMemoryExecutionRepository(),
        SandboxExecutor(target),
        faults=faults,
    )
    runtime = ExecutionRuntime(
        execution, VerificationService(execution, IndependentRegisterReader(target)), lambda: NOW
    )
    graph = build_agent_graph(
        _chat_passthrough,
        _tool_passthrough,
        InMemorySaver(),
        service.domain,
        service.evidence,
        execution_runtime=runtime,
    )
    run = await service.domain.get_run(RunLookup(run_id="synthetic-run"))
    intent = ExecutionIntent(request=await request_for(service, command="graph-execute"))
    state = GraphState(
        workflow_kind="releaseguard",
        releaseguard=ReleaseGuardGraphContext(
            change_request=run.change_request,
            run_id=run.run_id,
            expected_revision=run.revision,
            execution_intent=intent,
        ),
    )
    config = releaseguard_runnable_config(run.change_id, run.run_id, run.revision)
    return graph, runtime, target, state, config


@pytest.mark.parametrize("simulation", [False, True])
def test_single_graph_execution_and_chat(simulation: bool) -> None:
    """Simulation never sends; sandbox needs independent verification and receipt."""

    async def scenario() -> None:
        graph, runtime, target, state, config = await graph_execution_setup(simulation=simulation)
        result = await graph.ainvoke(state, config)
        run = result["releaseguard"].domain_state
        assert run.status is ChangeStatus.RECEIPTED
        assert run.receipt.final_status == ("SIMULATED" if simulation else "SUCCEEDED")
        assert target.calls == (0 if simulation else 1)
        assert run.simulation_result.predictions
        chat = await graph.ainvoke(
            GraphState(long_term_memory="synthetic"), {"configurable": {"thread_id": "chat-synthetic"}}
        )
        assert chat["workflow_kind"] == "chat" and target.calls == (0 if simulation else 1)
        with pytest.raises(DomainError, match="missing_or_completed"):
            await resume_execution(graph, state.releaseguard.execution_intent, config)

    asyncio.run(scenario())


def test_unknown_wait_requires_typed_readonly_reconciliation() -> None:
    """UNKNOWN stays isolated and diagnostic receipt survives a typed resume."""

    async def scenario() -> None:
        graph, runtime, target, state, config = await graph_execution_setup(faults=Fault("after_send"))
        paused = await graph.ainvoke(state, config)
        payload = paused["__interrupt__"][0].value
        assert payload["status"] == "EXECUTION_UNKNOWN" and target.calls == 1
        request = await request_for(runtime.execution.preflight.service, command="reconcile-graph")
        for updates in ({"plan_hash": "0" * 64}, {"thread_id": "other"}, {"state_version": request.state_version + 1}):
            with pytest.raises(DomainError):
                await resume_execution(
                    graph, ExecutionIntent(request=request.model_copy(update=updates), action="reconcile"), config
                )
        result = await resume_execution(graph, ExecutionIntent(request=request, action="reconcile"), config)
        run = result["releaseguard"].domain_state
        assert run.receipt.final_status == "SUCCEEDED" and run.receipt.prior_receipt_ids and target.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["before_claim", "after_result", "before_node_return"])
def test_node_fault_replay_uses_saved_facts(fault: str) -> None:
    """Domain commit before Graph checkpoint never leads to a second send."""

    async def scenario() -> None:
        graph, runtime, target, state, config = await graph_execution_setup(faults=Fault(fault))
        with pytest.raises(RuntimeError, match="synthetic_fault"):
            await graph.ainvoke(state, config)
        result = await graph.ainvoke(None, config)
        assert result["releaseguard"].domain_state.status is ChangeStatus.RECEIPTED and target.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["after_claim", "after_started", "before_send", "after_domain"])
def test_ambiguous_owner_stops_graph_without_send(fault: str) -> None:
    """An orphan or overlapping owner cannot be reacquired by Graph replay."""

    async def scenario() -> None:
        graph, _, target, state, config = await graph_execution_setup(faults=Fault(fault))
        with pytest.raises(RuntimeError):
            await graph.ainvoke(state, config)
        with pytest.raises(DomainError, match="inflight_unknown"):
            await graph.ainvoke(None, config)
        assert target.calls == 0

    asyncio.run(scenario())


def test_execution_intent_conflict_and_historical_root_rejected() -> None:
    """A reused command cannot acquire another action or historical checkpoint."""

    async def scenario() -> None:
        graph, runtime, target, state, config = await graph_execution_setup(faults=Fault("after_send"))
        await graph.ainvoke(state, config)
        intent = state.releaseguard.execution_intent
        with pytest.raises(DomainError, match="intent_conflict"):
            await runtime.advance(intent.model_copy(update={"action": "reconcile"}))
        snapshot = await graph.aget_state(config)
        historical = {**config, "configurable": snapshot.config["configurable"]}
        with pytest.raises(DomainError, match="checkpoint_fork"):
            await resume_execution(graph, intent, historical)
        assert target.calls == 1

    asyncio.run(scenario())
