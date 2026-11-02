"""ReleaseGuard graph identity and state contract tests."""

import pytest
from pydantic import ValidationError

from app.core.langgraph.releaseguard_identity import (
    THREAD_MAPPING_VERSION,
    releaseguard_runnable_config,
    releaseguard_thread_id,
    validate_releaseguard_thread_id,
)
from app.schemas.graph import GraphState, ReleaseGuardGraphContext
from app.schemas.releaseguard import ChangeStatus, new_change_run
from tests.releaseguard_fixtures import command_fixture
from tests.test_change_request_contract import request_fixture


def graph_context() -> ReleaseGuardGraphContext:
    """Build a fully bound synthetic graph input and Phase 2 snapshot."""
    request = request_fixture()
    state = new_change_run(request, "synthetic-run")
    command = command_fixture(state, ChangeStatus.COLLECTING_EVIDENCE)
    return ReleaseGuardGraphContext(
        change_request=request,
        run_id=state.run_id,
        expected_revision=request.revision,
        state_version=state.state_version,
        domain_state=state,
        pending_command=command,
    )


def test_chat_state_remains_the_default_workflow() -> None:
    """Legacy Chat state construction does not require ReleaseGuard fields."""
    state = GraphState.model_validate({"messages": [], "long_term_memory": "memory"})

    assert state.workflow_kind == "chat"
    assert state.releaseguard is None
    assert GraphState.model_validate_json(state.model_dump_json()) == state


def test_releaseguard_state_roundtrips_typed_identity_and_command() -> None:
    """JSON restoration preserves domain snapshots and the pending command."""
    state = GraphState(workflow_kind="releaseguard", releaseguard=graph_context())

    restored = GraphState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.releaseguard is not None
    assert restored.releaseguard.pending_command is not None
    assert restored.releaseguard.pending_command.audit.input_hash


@pytest.mark.parametrize(
    "payload,expected_error",
    [
        ({"workflow_kind": "releaseguard"}, "releaseguard_payload_required"),
        (
            {"workflow_kind": "chat", "releaseguard": graph_context().model_dump(mode="json")},
            "releaseguard_payload_on_chat_workflow",
        ),
    ],
)
def test_workflow_discriminator_requires_matching_payload(payload: dict[str, object], expected_error: str) -> None:
    """Workflow tags cannot silently select an incomplete or mixed payload."""
    with pytest.raises(ValidationError, match=expected_error):
        GraphState.model_validate(payload)


def test_releaseguard_identity_mismatch_is_rejected() -> None:
    """Revision, state version, domain identity, and command identity are bound."""
    context = graph_context()
    values = context.model_dump(mode="python")
    values["expected_revision"] = 2
    with pytest.raises(ValidationError, match="releaseguard_revision_binding_mismatch"):
        ReleaseGuardGraphContext.model_validate(values)

    values = context.model_dump(mode="python")
    values["state_version"] = 1
    with pytest.raises(ValidationError, match="releaseguard_state_version_mismatch"):
        ReleaseGuardGraphContext.model_validate(values)

    values = context.model_dump(mode="python")
    values["run_id"] = "different-run"
    with pytest.raises(ValidationError, match="releaseguard_state_binding_mismatch"):
        ReleaseGuardGraphContext.model_validate(values)

    values = context.model_dump(mode="python")
    assert context.domain_state is not None
    values["pending_command"] = command_fixture(
        context.domain_state,
        ChangeStatus.COLLECTING_EVIDENCE,
        expected_state_version=1,
    )
    with pytest.raises(ValidationError, match="releaseguard_command_version_mismatch"):
        ReleaseGuardGraphContext.model_validate(values)


def test_thread_mapping_is_stable_by_change_and_run_not_revision() -> None:
    """One run keeps its thread across revisions, while separate runs do not collide."""
    original = releaseguard_thread_id("change-1", "run-1")
    assert releaseguard_thread_id("change-1", "run-1") == original
    assert releaseguard_thread_id("change-1", "run-2") != original
    assert releaseguard_thread_id("change-2", "run-1") != original


def test_thread_validation_and_runnable_config_bind_trace_metadata() -> None:
    """Runtime config uses the canonical thread and only identity metadata."""
    state = GraphState(workflow_kind="releaseguard", releaseguard=graph_context())
    assert state.releaseguard is not None
    expected = releaseguard_thread_id(state.releaseguard.change_request.change_id, state.releaseguard.run_id)
    validate_releaseguard_thread_id(state, expected)

    with pytest.raises(ValueError, match="releaseguard_thread_binding_mismatch"):
        validate_releaseguard_thread_id(state, "user-selected-thread")

    config = releaseguard_runnable_config(
        state.releaseguard.change_request.change_id,
        state.releaseguard.run_id,
        state.releaseguard.expected_revision,
    )
    assert config["configurable"]["thread_id"] == expected
    assert config["metadata"]["thread_mapping_version"] == THREAD_MAPPING_VERSION
    assert "requester_id" not in config["metadata"]


def test_thread_mapping_rejects_empty_identity() -> None:
    """Empty identities cannot collapse into a shared checkpoint thread."""
    with pytest.raises(ValueError, match="releaseguard_thread_identity_required"):
        releaseguard_thread_id(" ", "run-1")


def test_thread_mapping_normalizes_identity_and_requires_positive_revision() -> None:
    """Canonical ids do not split on boundary whitespace or accept invalid revisions."""
    assert releaseguard_thread_id(" change-1 ", "run-1") == releaseguard_thread_id("change-1", "run-1")
    with pytest.raises(ValidationError):
        releaseguard_runnable_config("change-1", "run-1", 0)
    with pytest.raises(ValidationError):
        releaseguard_runnable_config("change-1", "run-1", True)
