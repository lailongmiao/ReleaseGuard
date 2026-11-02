"""Typed resume and Repo-before-checkpoint replay contract tests."""

import asyncio

import pytest
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import Command

from app.core.langgraph.graph import build_agent_graph
from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.core.langgraph.releaseguard_resume import expected_releaseguard_resume
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import RunLookup
from app.schemas.graph import GraphState, ReleaseGuardGraphContext, ReleaseGuardResume
from app.schemas.releaseguard import ChangeRequest, ChangeRunState, ChangeStatus, RunCommand, new_change_run
from app.services.releaseguard.state_machine import DomainError
from tests.releaseguard_fixtures import command_fixture
from tests.test_change_request_contract import request_fixture


class CommitThenRaiseRepository(InMemoryReleaseGuardRepository):
    """Commit once, then simulate process failure before the Graph node returns."""

    def __init__(self) -> None:
        """Enable one deterministic failure after the base commit completes."""
        super().__init__()
        self.fail_after_commit = True

    async def commit_run_update(self, command: RunCommand) -> ChangeRunState:
        """Raise once after the base repository stores the command result."""
        result = await super().commit_run_update(command)
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise RuntimeError("synthetic_graph_checkpoint_gap")
        return result


class FailBeforeCommitRepository(InMemoryReleaseGuardRepository):
    """Raise once before a command mutates the in-memory Repository."""

    def __init__(self) -> None:
        """Enable one deterministic failure before the base commit runs."""
        super().__init__()
        self.fail_before_commit = True

    async def commit_run_update(self, command: RunCommand) -> ChangeRunState:
        """Raise once without committing, then delegate subsequent attempts."""
        if self.fail_before_commit:
            self.fail_before_commit = False
            raise RuntimeError("synthetic_repository_commit_failure")
        return await super().commit_run_update(command)


def releaseguard_state(request: ChangeRequest, run: ChangeRunState, command: RunCommand) -> GraphState:
    """Build an invocation with a stable pre-commit domain snapshot and command."""
    return GraphState(
        workflow_kind="releaseguard",
        releaseguard=ReleaseGuardGraphContext(
            change_request=request,
            run_id=run.run_id,
            expected_revision=request.revision,
            state_version=run.state_version,
            domain_state=run,
            pending_command=command,
        ),
    )


async def _chat_passthrough(state: GraphState, config: RunnableConfig) -> dict[str, str]:
    """Avoid Provider access in the offline graph fixture."""
    return {"long_term_memory": state.long_term_memory}


async def _tool_passthrough(state: GraphState) -> dict[str, list[object]]:
    """Avoid tool execution in the offline graph fixture."""
    return {"messages": state.messages}


def test_typed_resume_rejects_text_without_committing_command() -> None:
    """A text resume fails closed before the Repository receives a command."""

    async def scenario() -> None:
        request = request_fixture()
        run = new_change_run(request, "resume-run")
        command = command_fixture(run, ChangeStatus.COLLECTING_EVIDENCE)
        state = releaseguard_state(request, run, command)
        repository = InMemoryReleaseGuardRepository()
        await repository.create_change(request)
        await repository.create_run(run)
        checkpointer = InMemorySaver()
        graph = build_agent_graph(_chat_passthrough, _tool_passthrough, checkpointer, repository)
        config = releaseguard_runnable_config(request.change_id, run.run_id, request.revision)

        paused = await graph.ainvoke(state, config)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("releaseguard_validate_resume",)
        interrupt_id = paused["__interrupt__"][0].id

        with pytest.raises(DomainError, match="resume_payload_invalid"):
            await graph.ainvoke(Command(resume={interrupt_id: "please approve"}), config)

        stored = await repository.get_run(RunLookup(run_id=run.run_id))
        assert stored.state_version == 0
        assert not stored.transition_history

    asyncio.run(scenario())


def test_typed_resume_rejects_mismatched_command_binding() -> None:
    """A well-shaped envelope for another command fails before Repository mutation."""

    async def scenario() -> None:
        request = request_fixture()
        run = new_change_run(request, "resume-run")
        command = command_fixture(run, ChangeStatus.COLLECTING_EVIDENCE)
        state = releaseguard_state(request, run, command)
        repository = InMemoryReleaseGuardRepository()
        await repository.create_change(request)
        await repository.create_run(run)
        graph = build_agent_graph(_chat_passthrough, _tool_passthrough, InMemorySaver(), repository)
        config = releaseguard_runnable_config(request.change_id, run.run_id, request.revision)

        paused = await graph.ainvoke(state, config)
        interrupt_id = paused["__interrupt__"][0].id
        supplied = ReleaseGuardResume.model_validate(paused["__interrupt__"][0].value)
        mismatched = supplied.model_copy(update={"command_id": "another-command"}).model_dump(mode="json")

        with pytest.raises(DomainError, match="resume_binding_mismatch"):
            await graph.ainvoke(Command(resume={interrupt_id: mismatched}), config)

        stored = await repository.get_run(RunLookup(run_id=run.run_id))
        assert stored.state_version == 0
        assert not stored.transition_history

    asyncio.run(scenario())


def test_repository_commit_failure_before_mutation_can_resume_safely() -> None:
    """A pre-commit failure leaves state untouched and retries the same command."""

    async def scenario() -> None:
        request = request_fixture()
        run = new_change_run(request, "resume-run")
        command = command_fixture(run, ChangeStatus.COLLECTING_EVIDENCE)
        state = releaseguard_state(request, run, command)
        repository = FailBeforeCommitRepository()
        await repository.create_change(request)
        await repository.create_run(run)
        graph = build_agent_graph(_chat_passthrough, _tool_passthrough, InMemorySaver(), repository)
        config = releaseguard_runnable_config(request.change_id, run.run_id, request.revision)

        paused = await graph.ainvoke(state, config)
        interrupt_id = paused["__interrupt__"][0].id
        supplied = ReleaseGuardResume.model_validate(paused["__interrupt__"][0].value)

        with pytest.raises(RuntimeError, match="synthetic_repository_commit_failure"):
            await graph.ainvoke(
                Command(resume={interrupt_id: supplied.model_dump(mode="json")}),
                config,
            )

        stored = await repository.get_run(RunLookup(run_id=run.run_id))
        assert stored.state_version == 0
        assert not stored.transition_history
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("releaseguard_commit_command",)

        resumed = await graph.ainvoke(None, config)
        context = resumed["releaseguard"]
        assert context.domain_state is not None
        assert context.domain_state.state_version == 1
        assert context.pending_command is None
        assert len((await repository.get_run(RunLookup(run_id=run.run_id))).transition_history) == 1

    asyncio.run(scenario())


def test_typed_resume_replays_after_repository_commit_checkpoint_gap() -> None:
    """A bound resume commits once and exact replay restores Graph/Repository agreement."""

    async def scenario() -> None:
        request = request_fixture()
        run = new_change_run(request, "resume-run")
        command = command_fixture(run, ChangeStatus.COLLECTING_EVIDENCE)
        state = releaseguard_state(request, run, command)
        repository = CommitThenRaiseRepository()
        await repository.create_change(request)
        await repository.create_run(run)
        checkpointer = InMemorySaver()
        graph = build_agent_graph(_chat_passthrough, _tool_passthrough, checkpointer, repository)
        config = releaseguard_runnable_config(request.change_id, run.run_id, request.revision)

        paused = await graph.ainvoke(state, config)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("releaseguard_validate_resume",)
        checkpoint_state = GraphState.model_validate(snapshot.values)
        expected = expected_releaseguard_resume(checkpoint_state, config)
        assert paused["__interrupt__"][0].value == expected.model_dump(mode="json")
        interrupt_id = paused["__interrupt__"][0].id
        supplied = ReleaseGuardResume.model_validate(paused["__interrupt__"][0].value)

        with pytest.raises(RuntimeError, match="synthetic_graph_checkpoint_gap"):
            await graph.ainvoke(Command(resume={interrupt_id: supplied.model_dump(mode="json")}), config)

        committed = await repository.get_run(RunLookup(run_id=run.run_id))
        assert committed.state_version == 1
        assert len(committed.transition_history) == 1
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("releaseguard_commit_command",)

        resumed = await graph.ainvoke(None, config)
        context = resumed["releaseguard"]
        assert context.domain_state is not None
        assert context.domain_state.state_version == 1
        assert context.domain_state.status is ChangeStatus.COLLECTING_EVIDENCE
        assert context.pending_command is None
        assert context.routing_outcome == "awaiting_evidence"
        assert len((await repository.get_run(RunLookup(run_id=run.run_id))).transition_history) == 1

    asyncio.run(scenario())


def test_resume_envelope_is_bound_to_thread_and_command_identity() -> None:
    """A correctly shaped envelope changes when its run or command identity changes."""
    request = request_fixture()
    run = new_change_run(request, "resume-run")
    command = command_fixture(run, ChangeStatus.COLLECTING_EVIDENCE)
    state = releaseguard_state(request, run, command)
    config = releaseguard_runnable_config(request.change_id, run.run_id, request.revision)
    expected = expected_releaseguard_resume(state, config)

    assert expected.change_id == request.change_id
    assert expected.run_id == run.run_id
    assert expected.expected_state_version == run.state_version
    assert expected.input_hash == command.audit.input_hash
    with pytest.raises(DomainError, match="releaseguard_thread_binding_mismatch"):
        expected_releaseguard_resume(state, {**config, "configurable": {"thread_id": "other-thread"}})
