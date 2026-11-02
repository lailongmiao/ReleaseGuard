"""Strict evaluation envelopes; domain records remain the existing models."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.approval import ApprovalEntry, ToolAuditEvent
from app.schemas.evidence import EvidenceCitation, EvidencePolicy, EvidenceSnapshot, EvidenceSourceResult
from app.schemas.execution import StepClaim
from app.schemas.releaseguard import ChangeRunState, EvidenceAssessment, ExecutionEvent
from app.services.evidence_sources import EvidenceAccessContext

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]
Hash = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Count = Annotated[int, Field(strict=True, ge=0)]
SCENARIOS = (
    "risk_low",
    "risk_medium",
    "risk_high",
    "risk_unknown",
    "gate_sufficient",
    "gate_missing",
    "gate_conflicting",
    "gate_expired",
    "gate_unavailable",
    "gate_permission_denied",
    "gate_hash_drift",
    "gate_dangling",
    "gate_version",
    "gate_access",
    "approval_pending",
    "approval_rejected",
    "approval_expired",
    "approval_revoked",
    "approval_needs_changes",
    "approval_self",
    "deny_default",
    "deny_tenant",
    "deny_resource",
    "deny_action",
    "deny_environment",
    "drift_plan",
    "drift_manifest",
    "drift_input",
    "drift_descriptor",
    "drift_policy",
    "drift_evidence",
    "simulation",
    "repeat",
    "intent_conflict",
    "concurrent",
    "opposite",
    "partial_recovery",
    "compensation_failed",
    "timeout",
    "cancel",
    "orphan_after_claim",
    "orphan_after_started",
    "orphan_before_send",
    "graph_fault_before_claim",
    "graph_fault_after_result",
    "graph_fault_before_node_return",
    "graph_fault_after_domain",
    "false_success",
    "invalid_observation",
    "unknown_reconcile",
    "receipt_conflict",
    "typed_resume",
    "old_root",
    "two_waits",
    "chat_isolation",
)

METRICS = (
    "evidence_completeness",
    "citation_validity",
    "risk_calibration",
    "approval_compliance",
    "tool_safety",
    "recovery_correctness",
    "idempotency",
    "receipt_correctness",
)


class Envelope(BaseModel):
    """Reject unknown inputs and implicit mutation of manifests."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Status(StrEnum):
    """Unverified and not applicable never mean passed."""

    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


class Oracle(Envelope):
    """Reviewed expectations, distinct from measured facts."""

    states: tuple[Name, ...]
    calls: Count
    target_value: Annotated[int, Field(strict=True)]
    gate_sufficient: bool = Field(strict=True)
    risk: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "LOW"
    diagnostics: tuple[Name, ...] = ()
    receipt: Literal["SUCCEEDED", "SIMULATED", "RECOVERED", "FAILED", "UNKNOWN", "REJECTED"] | None = None
    evidence: Name = "domain-contract.v1"


class CaseSpec(Envelope):
    """Allowlisted scenario and fixed synthetic initial conditions."""

    case_id: Annotated[str, StringConstraints(pattern=r"^RG-[A-Z][0-9]{2}(?:-[a-z_]+)?$")]
    scenario: Name
    entrypoint: Literal["graph", "service", "gate"]
    initial_status: Literal["RECEIVED"] = "RECEIVED"
    initial_value: Literal[0] = 0
    input_risk: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    clock_start: AwareDatetime
    synthetic: Literal[True] = True
    required_metrics: tuple[Name, ...] = METRICS
    regression_tags: tuple[Name, ...] = ()
    oracle: Oracle

    @model_validator(mode="after")
    def check_metrics(self) -> "CaseSpec":
        """Do not permit empty, duplicate or arbitrary metric definitions."""
        if (
            not self.required_metrics
            or len(set(self.required_metrics)) != len(self.required_metrics)
            or not set(self.required_metrics) <= set(METRICS)
        ):
            raise ValueError("case_metrics_invalid")
        if self.scenario not in SCENARIOS:
            raise ValueError("scenario_not_registered")
        if self.clock_start.utcoffset() is None or self.clock_start.isoformat() != "2026-01-01T00:00:00+00:00":
            raise ValueError("unsupported_synthetic_clock")
        return self


class Dataset(Envelope):
    """Version and content identity checked before any run."""

    schema_version: Literal["dataset.v1"] = "dataset.v1"
    dataset_id: Literal["releaseguard"] = "releaseguard"
    version: Name
    dataset_hash: Hash
    oracle_version: Literal["domain-contract.v1"] = "domain-contract.v1"
    synthetic: Literal[True] = True
    cases: tuple[CaseSpec, ...]


class LedgerFacts(Envelope):
    """Saved claim and immutable events, not a synthesized success summary."""

    claim: StepClaim
    events: tuple[ExecutionEvent, ...]


class CaseFacts(Envelope):
    """Facts read from authority after an isolated actual service/Graph run."""

    case_id: Name
    provenance: Literal["actual_offline_services", "fault_injected_offline"]
    domain: ChangeRunState | None = None
    gate: EvidenceAssessment | None = None
    gate_policy: EvidencePolicy | None = None
    gate_access: EvidenceAccessContext | None = None
    thread_id: Name | None = None
    gate_snapshots: tuple[EvidenceSnapshot, ...] = ()
    gate_citations: tuple[EvidenceCitation, ...] = ()
    source_results: tuple[EvidenceSourceResult, ...] = ()
    snapshots: tuple[EvidenceSnapshot, ...] = ()
    citations: tuple[EvidenceCitation, ...] = ()
    approvals: tuple[ApprovalEntry, ...] = ()
    audits: tuple[ToolAuditEvent, ...] = ()
    ledger: tuple[LedgerFacts, ...] = ()
    calls: Count
    reads: Count
    target_value: Annotated[int, Field(strict=True)]
    target_version: Count
    diagnostics: tuple[Name, ...] = ()
    network_calls: Count = 0
    subprocess_calls: Count = 0
    blocked_network_attempts: Count = 0
    blocked_process_attempts: Count = 0
    blocked_write_attempts: Count = 0
    runtime_persistence: Literal["same_process_memory_only"] = "same_process_memory_only"
    completed: bool = Field(default=True, strict=True)


class MetricResult(Envelope):
    """A hard rule needs evidence and cannot be overridden by Judge."""

    metric_id: Name
    version: Literal["rules.v1"] = "rules.v1"
    status: Status
    hard_gate: Literal[True] = True
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason_codes: tuple[Name, ...]
    fact_hash: Hash


class CaseResult(Envelope):
    """Retain every planned attempt, including failure or missing facts."""

    case_id: Name
    attempt: Annotated[int, Field(strict=True, ge=1)]
    facts: CaseFacts
    semantic_hash: Hash
    metrics: tuple[MetricResult, ...] = ()
    wall_seconds: float = Field(ge=0, allow_inf_nan=False)


class RunConfig(Envelope):
    """Default strictly offline; live targets require separate integration scopes."""

    mode: Literal["offline"] = "offline"
    judge: Literal["off", "fake"] = "off"
    langfuse: Literal["off"] = "off"
    repeat: Annotated[int, Field(strict=True, ge=1, le=10)] = 2
    seed: Literal[0] = 0
    timeout_seconds: float = Field(default=20, gt=0, le=120)
