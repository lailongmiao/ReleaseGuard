"""Deterministic LangGraph thread identity for ReleaseGuard runs."""

from typing import Any

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ConfigDict

from app.schemas.graph import GraphState
from app.schemas.releaseguard import Identifier, PositiveInt
from app.services.releaseguard.normalization import canonical_hash


THREAD_MAPPING_VERSION = "releaseguard.thread.v1"


class _ThreadIdentity(BaseModel):
    """Normalize and constrain identity components before hashing."""

    model_config = ConfigDict(extra="forbid")

    change_id: Identifier
    run_id: Identifier


class _RunnableBinding(_ThreadIdentity):
    """Validate trace metadata identity and revision as one typed value."""

    revision: PositiveInt


def releaseguard_thread_id(change_id: str, run_id: str) -> str:
    """Derive one stable checkpoint thread from a change/run identity pair."""
    if not change_id.strip() or not run_id.strip():
        raise ValueError("releaseguard_thread_identity_required")
    identity = _ThreadIdentity.model_validate({"change_id": change_id, "run_id": run_id})
    identity_hash = canonical_hash(identity.model_dump(mode="json"))
    return f"releaseguard:v1:{identity_hash}"


def validate_releaseguard_thread_id(state: GraphState, thread_id: str) -> None:
    """Reject a checkpoint thread that is not bound to the typed graph identity."""
    if state.workflow_kind != "releaseguard" or state.releaseguard is None:
        raise ValueError("releaseguard_workflow_required")
    expected = releaseguard_thread_id(state.releaseguard.change_request.change_id, state.releaseguard.run_id)
    if thread_id != expected:
        raise ValueError("releaseguard_thread_binding_mismatch")


def releaseguard_runnable_config(
    change_id: str,
    run_id: str,
    revision: int,
) -> RunnableConfig:
    """Build a non-sensitive trace binding and the canonical checkpoint thread."""
    binding = _RunnableBinding.model_validate({"change_id": change_id, "run_id": run_id, "revision": revision})
    thread_id = releaseguard_thread_id(binding.change_id, binding.run_id)
    metadata: dict[str, Any] = {
        "workflow_kind": "releaseguard",
        "change_id": binding.change_id,
        # LangGraph reserves metadata.run_id for invocation replay detection.
        # A stable business id there silently discards subsequent fresh inputs.
        "domain_run_id": binding.run_id,
        "revision": binding.revision,
        "thread_mapping_version": THREAD_MAPPING_VERSION,
    }
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
    }
