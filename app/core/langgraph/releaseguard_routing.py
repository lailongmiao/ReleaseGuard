"""Deterministic ReleaseGuard routing based only on validated domain state."""

from typing import Literal

from app.schemas.graph import GraphState, ReleaseGuardRouteOutcome
from app.schemas.releaseguard import ChangeStatus, RiskLevel
from app.services.releaseguard.state_machine import DomainError


ReleaseGuardRoute = Literal[
    "awaiting_evidence",
    "awaiting_risk_assessment",
    "awaiting_plan",
    "awaiting_approval",
    "unsupported_execution",
    "blocked_unknown_risk",
    "terminal",
]
LoadedReleaseGuardRoute = Literal["resume_pending_command", "assess_evidence"] | ReleaseGuardRoute

STATUS_ROUTE: dict[ChangeStatus, ReleaseGuardRoute] = {
    ChangeStatus.RECEIVED: "awaiting_evidence",
    ChangeStatus.COLLECTING_EVIDENCE: "awaiting_evidence",
    ChangeStatus.WAITING_FOR_EVIDENCE: "awaiting_evidence",
    ChangeStatus.EVIDENCE_SUFFICIENT: "awaiting_risk_assessment",
    ChangeStatus.RISK_ASSESSED: "awaiting_plan",
    ChangeStatus.PLAN_READY: "awaiting_approval",
    ChangeStatus.WAITING_FOR_APPROVAL: "awaiting_approval",
    ChangeStatus.APPROVED: "unsupported_execution",
    ChangeStatus.DRY_RUN: "unsupported_execution",
    ChangeStatus.DRY_RUN_PASSED: "unsupported_execution",
    ChangeStatus.DRY_RUN_FAILED: "unsupported_execution",
    ChangeStatus.EXECUTING_SANDBOX: "unsupported_execution",
    ChangeStatus.EXECUTION_FAILED: "unsupported_execution",
    ChangeStatus.EXECUTION_UNKNOWN: "unsupported_execution",
    ChangeStatus.RECOVERY_REQUIRED: "unsupported_execution",
    ChangeStatus.RECOVERING: "unsupported_execution",
    ChangeStatus.VERIFYING: "unsupported_execution",
    ChangeStatus.SUCCEEDED: "terminal",
    ChangeStatus.REJECTED: "terminal",
    ChangeStatus.FAILED: "terminal",
    ChangeStatus.RECEIPTED: "terminal",
}

if set(STATUS_ROUTE) != set(ChangeStatus):
    raise RuntimeError("releaseguard_status_route_incomplete")


def route_workflow(state: GraphState) -> Literal["chat", "releaseguard"]:
    """Choose a top-level branch from the explicit workflow discriminator."""
    return state.workflow_kind


def route_by_domain_status(state: GraphState) -> ReleaseGuardRoute:
    """Route a loaded snapshot through Phase 3 capability boundaries."""
    context = state.releaseguard
    if state.workflow_kind != "releaseguard" or context is None or context.domain_state is None:
        raise DomainError("releaseguard_state_required")

    run = context.domain_state
    if run.status is ChangeStatus.RISK_ASSESSED and (
        run.risk_assessment is None or run.risk_assessment.level is RiskLevel.UNKNOWN
    ):
        return "blocked_unknown_risk"
    return STATUS_ROUTE[run.status]


def route_loaded_state(state: GraphState) -> LoadedReleaseGuardRoute:
    """Resume a checkpointed typed command before routing its current domain status."""
    context = state.releaseguard
    if state.workflow_kind != "releaseguard" or context is None or context.domain_state is None:
        raise DomainError("releaseguard_state_required")
    if context.pending_command is not None:
        return "resume_pending_command"
    if context.domain_state.status is ChangeStatus.COLLECTING_EVIDENCE and context.evidence_policy is not None:
        return "assess_evidence"
    return route_by_domain_status(state)


def set_routing_outcome(state: GraphState, outcome: ReleaseGuardRouteOutcome) -> GraphState:
    """Return a typed state copy exposing the deterministic capability result."""
    context = state.releaseguard
    if state.workflow_kind != "releaseguard" or context is None or context.domain_state is None:
        raise DomainError("releaseguard_state_required")
    updated_context = context.model_copy(update={"routing_outcome": outcome})
    return state.model_copy(update={"releaseguard": updated_context})
