"""ReleaseGuard node boundaries against the Phase 2 repository protocol."""

import asyncio

import pytest
from pydantic import ValidationError
from langchain_core.runnables.config import RunnableConfig

from app.core.langgraph.releaseguard_identity import releaseguard_runnable_config
from app.core.langgraph.releaseguard_nodes import (
    commit_domain_command,
    load_domain_state,
    register_or_load_change,
    register_or_load_run,
)
from app.repositories.in_memory_releaseguard import InMemoryReleaseGuardRepository
from app.repositories.releaseguard import ChangeLookup, RevisionAppend, RunLookup
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import (
    ChangeRequest,
    ChangeRequestInput,
    ChangeRunState,
    ChangeStatus,
    RunCommand,
    new_change_run,
    request_input_hash,
)
from app.services.releaseguard.state_machine import DomainError
from tests.releaseguard_fixtures import command_fixture
from tests.test_change_request_contract import request_fixture


def request_variant(**overrides: object) -> ChangeRequest:
    """Build a variant whose normalized input hash matches its business fields."""
    values = {**request_fixture().model_dump(), **overrides}
    body = ChangeRequestInput.model_validate(
        {key: value for key, value in values.items() if key in ChangeRequestInput.model_fields}
    )
    values["request_hash"] = request_input_hash(body)
    return ChangeRequest.model_validate(values)


def graph_state(
    request: ChangeRequest | None = None,
    run_id: str = "synthetic-run",
    domain_state: ChangeRunState | None = None,
    pending_command: RunCommand | None = None,
) -> GraphState:
    """Build a ReleaseGuard state with optional repository snapshot/command."""
    bound_request = request or request_fixture()
    return GraphState(
        workflow_kind="releaseguard",
        releaseguard=ReleaseGuardGraphContext(
            change_request=bound_request,
            run_id=run_id,
            expected_revision=bound_request.revision,
            state_version=domain_state.state_version if domain_state is not None else None,
            domain_state=domain_state,
            pending_command=pending_command,
        ),
    )


def config_for(state: GraphState) -> RunnableConfig:
    """Build thread and metadata from the state identity used by the tests."""
    assert state.releaseguard is not None
    return releaseguard_runnable_config(
        state.releaseguard.change_request.change_id,
        state.releaseguard.run_id,
        state.releaseguard.expected_revision,
    )


def test_change_registration_is_idempotent_and_identity_checked() -> None:
    """A stable request can retry, but a regenerated change id cannot rebind."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        initial = graph_state()
        config = config_for(initial)
        first = await register_or_load_change(initial, repository, config)
        replay = await register_or_load_change(initial, repository, config)
        assert replay == first

        changed_identity = graph_state(request_variant(change_id="regenerated-change"))
        with pytest.raises(DomainError, match="identity_conflict"):
            await register_or_load_change(changed_identity, repository, config_for(changed_identity))

    asyncio.run(scenario())


def test_change_registration_rejects_thread_and_metadata_mismatch_before_write() -> None:
    """A caller cannot select a different thread or alter the traced identity."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        state = graph_state()
        wrong_thread = config_for(state)
        wrong_thread["configurable"]["thread_id"] = "user-selected-thread"
        with pytest.raises(DomainError, match="releaseguard_thread_binding_mismatch"):
            await register_or_load_change(state, repository, wrong_thread)

        wrong_metadata = config_for(state)
        wrong_metadata["metadata"]["revision"] = 99
        with pytest.raises(DomainError, match="releaseguard_metadata_binding_mismatch"):
            await register_or_load_change(state, repository, wrong_metadata)

        with pytest.raises(DomainError, match="not_found"):
            await repository.get_change(ChangeLookup(change_id="synthetic-change"))

    asyncio.run(scenario())


def test_run_registration_retries_by_reading_and_validating_existing_run() -> None:
    """Duplicate run IDs are read back; matching state is reused without writes."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        request = request_fixture()
        await repository.create_change(request)
        initial = graph_state(request)
        config = config_for(initial)
        first = await register_or_load_run(initial, repository, config)
        replay = await register_or_load_run(initial, repository, config)
        assert first == replay
        assert first.releaseguard is not None
        assert first.releaseguard.domain_state == await repository.get_run(RunLookup(run_id="synthetic-run"))

    asyncio.run(scenario())


def test_run_registration_rejects_existing_run_bound_to_another_change() -> None:
    """An identity conflict is never treated as success without full binding checks."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        first_request = request_fixture()
        second_request = request_variant(
            change_id="other-change",
            requester_id="other-requester",
            idempotency_key="other-key",
        )
        await repository.create_change(first_request)
        await repository.create_change(second_request)
        await repository.create_run(new_change_run(first_request, "shared-run-id"))

        state = graph_state(second_request, run_id="shared-run-id")
        with pytest.raises(DomainError, match="identity_conflict"):
            await register_or_load_run(state, repository, config_for(state))

    asyncio.run(scenario())


def test_load_domain_state_uses_repository_as_authority() -> None:
    """The node fills a missing cache from the repository's validated snapshot."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        request = request_fixture()
        await repository.create_change(request)
        expected = await repository.create_run(new_change_run(request, "synthetic-run"))
        state = graph_state(request)

        loaded = await load_domain_state(state, repository, config_for(state))

        assert loaded.releaseguard is not None
        assert loaded.releaseguard.domain_state == expected
        assert loaded.releaseguard.state_version == expected.state_version

    asyncio.run(scenario())


def test_load_domain_state_rejects_stale_revision() -> None:
    """A repository revision change invalidates the graph's pinned request."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        request = request_fixture()
        await repository.create_change(request)
        await repository.create_run(new_change_run(request, "synthetic-run"))
        revised = request_variant(revision=2, title="Synthetic revised title")
        await repository.append_revision(RevisionAppend(request=revised, expected_revision=1))
        state = graph_state(request)

        with pytest.raises(DomainError, match="revision_conflict"):
            await load_domain_state(state, repository, config_for(state))

    asyncio.run(scenario())


def test_command_commit_uses_repository_result_and_exact_replay() -> None:
    """An exact command replay returns the stored result without duplicate history."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        request = request_fixture()
        await repository.create_change(request)
        domain_state = await repository.create_run(new_change_run(request, "synthetic-run"))
        command = command_fixture(domain_state, ChangeStatus.COLLECTING_EVIDENCE)
        state = graph_state(request, domain_state=domain_state, pending_command=command)
        config = config_for(state)

        first = await commit_domain_command(state, repository, config)
        replay = await commit_domain_command(state, repository, config)

        assert first == replay
        assert first.releaseguard is not None
        assert first.releaseguard.domain_state is not None
        assert first.releaseguard.domain_state.status is ChangeStatus.COLLECTING_EVIDENCE
        assert len((await repository.get_run_history(RunLookup(run_id="synthetic-run"))).transitions) == 1

    asyncio.run(scenario())


def test_command_commit_requires_snapshot_and_prepared_command() -> None:
    """A node cannot mutate Repository state without a checked command envelope."""

    async def scenario() -> None:
        repository = InMemoryReleaseGuardRepository()
        state = graph_state()
        with pytest.raises(DomainError, match="command_precondition_failed"):
            await commit_domain_command(state, repository, config_for(state))

    asyncio.run(scenario())


def test_invalid_graph_context_is_rejected_before_node_invocation() -> None:
    """Invalid typed payloads fail at the schema boundary before repository access."""
    with pytest.raises(ValidationError):
        GraphState.model_validate({"workflow_kind": "releaseguard"})
