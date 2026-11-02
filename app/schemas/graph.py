"""Serializable graph state for Chat and ReleaseGuard workflows."""

from datetime import datetime
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.evidence import (
    EvidenceCitation,
    EvidencePolicy,
    EvidenceSnapshot,
    EvidenceSourceResult,
)
from app.schemas.approval import ApprovalRequest
from app.schemas.execution import ExecutionIntent
from app.schemas.releaseguard import (
    ChangeRequest,
    ChangeRunState,
    EvidenceAssessment,
    Identifier,
    PositiveInt,
    RunCommand,
    Sha256,
)

ReleaseGuardRouteOutcome = Literal[
    "awaiting_evidence",
    "awaiting_risk_assessment",
    "awaiting_plan",
    "awaiting_approval",
    "unsupported_execution",
    "blocked_unknown_risk",
    "terminal",
    "execution_wait",
]


class ReleaseGuardResume(BaseModel):
    """Typed continuation envelope; it carries no approval decision or free text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_kind: Literal["releaseguard"] = "releaseguard"
    resume_id: Identifier
    change_id: Identifier
    run_id: Identifier
    revision: PositiveInt
    expected_state_version: Annotated[int, Field(strict=True, ge=0)]
    command_id: Identifier
    input_hash: Sha256
    action: Literal["continue"] = "continue"


class ReleaseGuardGraphContext(BaseModel):
    """Typed identity and replay material for one ReleaseGuard graph run."""

    change_request: ChangeRequest
    run_id: Identifier
    expected_revision: PositiveInt
    state_version: Annotated[int, Field(strict=True, ge=0)] | None = None
    domain_state: ChangeRunState | None = None
    pending_command: RunCommand | None = None
    routing_outcome: ReleaseGuardRouteOutcome | None = None
    evidence_policy: EvidencePolicy | None = None
    evidence_snapshots: list[EvidenceSnapshot] = Field(default_factory=list)
    evidence_citations: list[EvidenceCitation] = Field(default_factory=list)
    evidence_source_results: list[EvidenceSourceResult] = Field(default_factory=list)
    evidence_assessment: EvidenceAssessment | None = None
    evidence_evaluated_at: datetime | None = None
    analysis_command_id: Identifier | None = None
    analysis_result_id: Identifier | None = None
    approval_request: ApprovalRequest | None = None
    execution_intent: ExecutionIntent | None = None
    recovery_request: ApprovalRequest | None = None
    approval_outcome: (
        Literal["decided", "expired", "revoked", "denied", "conflicting", "consistency_stopped"] | None
    ) = None

    @model_validator(mode="after")
    def validate_bindings(self) -> "ReleaseGuardGraphContext":
        """Reject request, run, snapshot, and pending-command identity drift."""
        request = self.change_request
        if self.execution_intent is not None:
            call = self.execution_intent.request
            if (call.change_id, call.run_id, call.revision, call.request_hash) != (
                request.change_id,
                self.run_id,
                request.revision,
                request.request_hash,
            ):
                raise ValueError("execution_graph_identity_conflict")
        if self.expected_revision != request.revision:
            raise ValueError("releaseguard_revision_binding_mismatch")

        state = self.domain_state
        if state is not None:
            if (state.change_id, state.run_id, state.revision, state.request_hash) != (
                request.change_id,
                self.run_id,
                request.revision,
                request.request_hash,
            ):
                raise ValueError("releaseguard_state_binding_mismatch")
            if self.state_version != state.state_version:
                raise ValueError("releaseguard_state_version_mismatch")
        elif self.state_version is not None:
            raise ValueError("releaseguard_state_snapshot_required")

        command = self.pending_command
        if command is not None:
            if (command.change_id, command.run_id, command.expected_revision) != (
                request.change_id,
                self.run_id,
                request.revision,
            ):
                raise ValueError("releaseguard_command_binding_mismatch")
            if self.state_version is not None and command.expected_state_version != self.state_version:
                raise ValueError("releaseguard_command_version_mismatch")
        assessment = self.evidence_assessment
        if assessment is not None:
            if (assessment.change_id, assessment.run_id, assessment.revision, assessment.request_hash) != (
                request.change_id,
                self.run_id,
                request.revision,
                request.request_hash,
            ):
                raise ValueError("releaseguard_evidence_assessment_binding_mismatch")
            if self.evidence_policy is not None and assessment.policy_version != self.evidence_policy.policy_version:
                raise ValueError("releaseguard_evidence_policy_binding_mismatch")
        if self.evidence_evaluated_at is not None and self.evidence_evaluated_at.tzinfo is None:
            raise ValueError("releaseguard_evidence_time_requires_timezone")
        if self.approval_request is not None:
            call = self.approval_request.binding.invocation
            if (call.change_id, call.run_id, call.revision, call.request_hash) != (
                request.change_id,
                self.run_id,
                request.revision,
                request.request_hash,
            ):
                raise ValueError("releaseguard_approval_binding_mismatch")
        return self


class GraphState(BaseModel):
    """State definition shared by the existing Chat path and ReleaseGuard path."""

    workflow_kind: Literal["chat", "releaseguard"] = "chat"
    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    long_term_memory: str = Field(default="", description="The long term memory of the conversation")
    releaseguard: ReleaseGuardGraphContext | None = None

    @model_validator(mode="after")
    def validate_workflow_payload(self) -> "GraphState":
        """Keep Chat and ReleaseGuard payloads explicit and mutually exclusive."""
        if self.workflow_kind == "chat" and self.releaseguard is not None:
            raise ValueError("releaseguard_payload_on_chat_workflow")
        if self.workflow_kind == "releaseguard" and self.releaseguard is None:
            raise ValueError("releaseguard_payload_required")
        return self
