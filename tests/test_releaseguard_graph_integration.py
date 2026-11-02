"""Offline integration tests for the single shared Chat/ReleaseGuard graph."""

import asyncio

import pytest
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import build_agent_graph
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ChangeRequest
from app.services.releaseguard.state_machine import DomainError
from tests.test_change_request_contract import request_fixture


def releaseguard_state(request: ChangeRequest, run_id: str = "integration-run") -> GraphState:
    """Build an unregistered ReleaseGuard invocation with stable identifiers."""
    return GraphState(
        workflow_kind="releaseguard",
        releaseguard=ReleaseGuardGraphContext(
            change_request=request,
            run_id=run_id,
            expected_revision=request.revision,
        ),
    )


async def _chat_passthrough(state: GraphState, config: RunnableConfig) -> dict[str, str]:
    """Stand in for the production chat node without invoking a provider."""
    return {"long_term_memory": state.long_term_memory}


async def _tool_passthrough(state: GraphState) -> dict[str, list[object]]:
    """Stand in for tool routing without executing any tool."""
    return {"messages": state.messages}


def test_one_compiled_graph_runs_chat_and_releaseguard_branches() -> None:
    """Both workflows compile and run through the same StateGraph builder."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        graph = build_agent_graph(
            _chat_passthrough,
            _tool_passthrough,
            InMemorySaver(),
            repository,
            name="offline-test-agent",
        )
        chat_result = await graph.ainvoke(
            GraphState(long_term_memory="synthetic memory"),
            {"configurable": {"thread_id": "chat-session"}},
        )
        assert chat_result["workflow_kind"] == "chat"
        assert chat_result["long_term_memory"] == "synthetic memory"

        request = request_fixture()
        state = releaseguard_state(request)
        config = releaseguard_runnable_config(request.change_id, "integration-run", request.revision)
        result = await graph.ainvoke(state, config)

        assert result["workflow_kind"] == "releaseguard"
        releaseguard = result["releaseguard"]
        assert releaseguard.domain_state is not None
        assert releaseguard.domain_state.status.value == "RECEIVED"
        assert releaseguard.routing_outcome == "awaiting_evidence"
        assert await repository.get_run(RunLookup(run_id="integration-run")) == releaseguard.domain_state

    asyncio.run(scenario())


def test_releaseguard_graph_without_repository_fails_closed_before_write() -> None:
    """Production graph construction does not silently substitute memory storage."""

    async def scenario() -> None:
        graph = build_agent_graph(_chat_passthrough, _tool_passthrough, InMemorySaver())
        request = request_fixture()
        state = releaseguard_state(request)
        config = releaseguard_runnable_config(request.change_id, "missing-repository-run", request.revision)

        with pytest.raises(DomainError, match="releaseguard_repository_unavailable"):
            await graph.ainvoke(state, config)

    asyncio.run(scenario())
