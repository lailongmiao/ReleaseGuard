"""Thread-bound typed continuation for already validated domain commands."""

from pydantic import ValidationError
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import interrupt

from app.core.langgraph.releaseguard_nodes import _context
from app.schemas.graph import GraphState, ReleaseGuardResume
from app.services.releaseguard.state_machine import DomainError


def expected_releaseguard_resume(state: GraphState, config: RunnableConfig) -> ReleaseGuardResume:
    """Build the only continuation value accepted for the checkpointed command."""
    context = _context(state, config)
    command = context.pending_command
    if command is None or context.domain_state is None:
        raise DomainError("command_precondition_failed")
    if command.expected_state_version != context.domain_state.state_version:
        raise DomainError("version_conflict")
    return ReleaseGuardResume(
        resume_id=f"releaseguard.resume.v1:{command.command_id}",
        change_id=command.change_id,
        run_id=command.run_id,
        revision=command.expected_revision,
        expected_state_version=command.expected_state_version,
        command_id=command.command_id,
        input_hash=command.audit.input_hash,
    )


async def await_typed_releaseguard_resume(state: GraphState, config: RunnableConfig) -> GraphState:
    """Pause before repository mutation and reject text or mismatched continuation data."""
    expected = expected_releaseguard_resume(state, config)
    supplied_value = interrupt(expected.model_dump(mode="json"))
    try:
        supplied = ReleaseGuardResume.model_validate(supplied_value)
    except ValidationError:
        raise DomainError("resume_payload_invalid") from None
    if supplied != expected:
        raise DomainError("resume_binding_mismatch")
    return state
