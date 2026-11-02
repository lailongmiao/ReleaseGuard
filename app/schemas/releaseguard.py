"""ReleaseGuard domain contracts, independent from Graphs and external systems."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

from app.schemas.evidence import EvidenceIdentifier, EvidenceItem
from app.schemas.execution import Direction, ExecutionManifest, ExecutionMode, SimulationPrediction
from app.services.releaseguard.normalization import canonical_hash


CONTRACT_VERSION = "releaseguard.v1"


def _text(value: Any) -> str:
    """Trim boundary whitespace and reject non-text and control characters."""
    if not isinstance(value, str):
        raise ValueError("text_required")
    value = value.strip()
    if not value or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in value):
        raise ValueError("empty_or_control_text")
    return value


def _optional_text(value: Any) -> str | None:
    """Normalize optional empty text to null."""
    if value is None or isinstance(value, str) and not value.strip():
        return None
    return _text(value)


def _business_text(value: Any) -> str:
    """Preserve multiline plans while rejecting non-printing control characters."""
    if not isinstance(value, str):
        raise ValueError("text_required")
    value = value.strip().replace("\r\n", "\n").replace("\r", "\n")
    if not value or any((ord(char) < 32 and char not in "\n\t") or 127 <= ord(char) < 160 for char in value):
        raise ValueError("empty_or_control_text")
    return value


def _hash(value: Any) -> str:
    """Accept the Phase 1 hash representation and emit lowercase bare SHA-256."""
    if not isinstance(value, str):
        raise ValueError("hash_required")
    return value.removeprefix("sha256:").lower()


def _time(value: datetime) -> datetime:
    """Reject naive timestamps and normalize aware timestamps to UTC."""
    if value.utcoffset() is None:
        raise ValueError("timestamp_requires_timezone")
    return value.astimezone(UTC)


Identifier = Annotated[str, BeforeValidator(_text), StringConstraints(min_length=1, max_length=128)]
Text = Annotated[str, BeforeValidator(_business_text), StringConstraints(min_length=1, max_length=8000)]
OptionalText = Annotated[str | None, BeforeValidator(_optional_text)]
Sha256 = Annotated[str, BeforeValidator(_hash), StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class DomainModel(BaseModel):
    """Strict, frozen domain values; collections are copied at storage boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Environment(StrEnum):
    """Supported target environment labels."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    OTHER = "other"


class ChangeIdentity(DomainModel):
    """Stable change identifier and monotonic business revision."""

    change_id: Identifier
    revision: PositiveInt


class RunIdentity(ChangeIdentity):
    """Run identity distinct from the business request revision."""

    run_id: Identifier


class ChangeWindow(DomainModel):
    """Timezone-aware window retaining a validated presentation timezone."""

    start: datetime
    end: datetime
    timezone: Identifier

    @model_validator(mode="after")
    def validate_window(self) -> "ChangeWindow":
        """Check IANA offsets and retain zone-aware values for nested validation."""
        try:
            zone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError:
            raise ValueError("invalid_iana_timezone") from None
        for value in (self.start, self.end):
            if value.utcoffset() is None or value.utcoffset() != value.astimezone(zone).utcoffset():
                raise ValueError("window_timezone_mismatch")
        if _time(self.end) <= _time(self.start):
            raise ValueError("invalid_change_window")
        object.__setattr__(self, "start", self.start.astimezone(zone))
        object.__setattr__(self, "end", self.end.astimezone(zone))
        return self


class ChangeRequestInput(DomainModel):
    """Business inputs included in the request hash whitelist."""

    requester_id: Identifier
    service: Text
    environment: Annotated[Environment, BeforeValidator(lambda value: _text(value).lower())]
    title: Text
    description: Text
    target_version: OptionalText = None
    change_diff_ref: OptionalText = None
    change_window: ChangeWindow
    rollback_plan: Text
    verification_plan: Text
    risk_acceptor_id: Annotated[Identifier | None, BeforeValidator(_optional_text)] = None

    @model_validator(mode="after")
    def validate_production_owner(self) -> "ChangeRequestInput":
        """Require an explicit risk owner without implying authorization."""
        if self.environment is Environment.PRODUCTION and not self.risk_acceptor_id:
            raise ValueError("production_requires_risk_acceptor")
        return self


def request_input_hash(request: ChangeRequestInput) -> str:
    """Hash only business inputs, excluding generated identity and audit fields."""
    payload = request.model_dump(include=set(ChangeRequestInput.model_fields))
    payload["change_window"] = {"start": request.change_window.start, "end": request.change_window.end}
    return canonical_hash(payload)


class ChangeRequest(ChangeRequestInput, ChangeIdentity):
    """Validated request revision with a recomputable business input hash."""

    idempotency_key: Identifier
    request_hash: Sha256
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_request(self) -> "ChangeRequest":
        """Reject forged hashes and invalid audit chronology."""
        if self.request_hash != request_input_hash(self):
            raise PydanticCustomError("hash_mismatch", "hash_mismatch")
        if _time(self.updated_at) < _time(self.created_at):
            raise ValueError("invalid_request_chronology")
        object.__setattr__(self, "created_at", _time(self.created_at))
        object.__setattr__(self, "updated_at", _time(self.updated_at))
        return self


class CreateChangeRequest(DomainModel):
    """Typed construction command with explicit identity and time."""

    identity: ChangeIdentity
    body: ChangeRequestInput
    idempotency_key: Identifier
    occurred_at: datetime


def build_change_request(command: CreateChangeRequest) -> ChangeRequest:
    """Construct and validate a request from normalized business inputs."""
    return ChangeRequest.model_validate(
        {
            **command.body.model_dump(),
            **command.identity.model_dump(),
            "idempotency_key": command.idempotency_key,
            "request_hash": request_input_hash(command.body),
            "created_at": command.occurred_at,
            "updated_at": command.occurred_at,
        }
    )


def _sorted_unique(value: Any) -> Any:
    """Normalize set-like string collections without silently accepting scalar text."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("collection_required")
    return sorted({_text(item) for item in value})


StringSet = Annotated[list[Identifier], BeforeValidator(_sorted_unique)]
EvidenceIds = Annotated[list[EvidenceIdentifier], BeforeValidator(_sorted_unique)]


class RunBound(RunIdentity):
    """Result bound to one run and the exact business input hash."""

    request_hash: Sha256


class EvidenceAssessment(RunBound):
    """Internal consistency of an explicit result, without source-policy evaluation."""

    record_type: Literal["evidence_assessment"] = "evidence_assessment"
    sufficient: bool
    required_items: StringSet
    present_items: StringSet
    missing_items: StringSet
    blocking_reasons: list[Text] = Field(default_factory=list)
    evidence_ids: EvidenceIds = Field(default_factory=list)
    uncertainty: list[Text] = Field(default_factory=list)
    policy_version: Identifier
    evaluated_at: datetime

    @model_validator(mode="after")
    def validate_assessment(self) -> "EvidenceAssessment":
        """Reject contradictory sufficiency results and normalize assessment time."""
        required, present = set(self.required_items), set(self.present_items)
        if not present <= required or set(self.missing_items) != required - present:
            raise ValueError("inconsistent_evidence_items")
        gaps = bool(self.missing_items or self.blocking_reasons or self.uncertainty)
        if self.sufficient and (not required or gaps or not self.evidence_ids):
            raise ValueError("inconsistent_evidence_sufficiency")
        if not self.sufficient and not gaps:
            raise ValueError("insufficient_requires_reason")
        object.__setattr__(self, "evaluated_at", _time(self.evaluated_at))
        return self


class RiskLevel(StrEnum):
    """Explicit risk outcomes; UNKNOWN has no implicit LOW conversion."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class RiskAssessment(RunBound):
    """Versioned assessment result without a production risk scoring algorithm."""

    record_type: Literal["risk_assessment"] = "risk_assessment"
    level: RiskLevel
    factors: list[Text] = Field(default_factory=list)
    blast_radius: Text
    uncertainty: list[Text] = Field(default_factory=list)
    controls: list[Text] = Field(default_factory=list)
    evidence_ids: EvidenceIds = Field(default_factory=list)
    policy_version: Identifier
    assessed_at: datetime

    @model_validator(mode="after")
    def validate_risk(self) -> "RiskAssessment":
        """Require explicit uncertainty and evidence or declared assumptions."""
        if self.level is RiskLevel.UNKNOWN and not self.uncertainty:
            raise ValueError("unknown_requires_uncertainty")
        if (
            self.factors
            and not self.evidence_ids
            and any(not factor.startswith("assumption:") for factor in self.factors)
        ):
            raise ValueError("risk_factor_requires_evidence_or_assumption")
        object.__setattr__(self, "assessed_at", _time(self.assessed_at))
        return self


class AssessmentContext(DomainModel):
    """Bind result evidence references to actual Phase 1 contract objects."""

    identity: RunBound
    evidence_items: list[EvidenceItem]
    evidence_assessment: EvidenceAssessment | None = None
    risk_assessment: RiskAssessment | None = None

    @model_validator(mode="after")
    def validate_references(self) -> "AssessmentContext":
        """Reject duplicate evidence ids, missing references and foreign results."""
        ids = {item.evidence_id for item in self.evidence_items}
        if len(ids) != len(self.evidence_items):
            raise ValueError("duplicate_evidence_id")
        for assessment in (self.evidence_assessment, self.risk_assessment):
            if assessment is None:
                continue
            if any(getattr(assessment, field) != getattr(self.identity, field) for field in RunBound.model_fields):
                raise ValueError("assessment_binding_mismatch")
            if not set(assessment.evidence_ids) <= ids:
                raise ValueError("unknown_evidence_reference")
        return self


Timestamp = Annotated[datetime, AfterValidator(_time)]


class ActorType(StrEnum):
    """Audit actor categories without authentication or authorization semantics."""

    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


class AuditContext(DomainModel):
    """Audit context supplied by callers, including a trusted explicit time."""

    actor_id: Identifier
    actor_type: ActorType
    reason: OptionalText = None
    correlation_id: Identifier
    occurred_at: Timestamp
    policy_version: Identifier
    contract_version: Literal["releaseguard.v1"] = CONTRACT_VERSION


class AuditFields(AuditContext):
    """Audit context bound to a canonical operation input hash."""

    input_hash: Sha256


class AuditedRecord(RunBound):
    """Record with one consistent audit representation."""

    input_hash: Sha256
    audit: AuditFields

    @model_validator(mode="after")
    def validate_audit(self) -> "AuditedRecord":
        """Reject contradictory duplicate hash, time, actor and reason fields."""
        if self.input_hash != self.audit.input_hash:
            raise ValueError("audit_hash_mismatch")
        for field in ("actor_id", "reason", "occurred_at"):
            if field in type(self).model_fields and getattr(self, field) != getattr(self.audit, field):
                raise ValueError("audit_field_mismatch")
        return self


class ApprovalDecision(StrEnum):
    """Recorded decisions; invalidation is append-only rather than an overwrite."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_CHANGES = "NEEDS_CHANGES"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    INVALIDATED = "INVALIDATED"


class ApprovalRecord(AuditedRecord):
    """Immutable decision bound to a request revision and execution plan."""

    record_type: Literal["approval"] = "approval"
    scope: Literal["tool", "plan", "recovery"] = "tool"
    manifest_hash: Sha256 | None = None
    effect_ids: tuple[Identifier, ...] = ()
    approval_id: Identifier
    plan_hash: Sha256
    decision: ApprovalDecision
    approver_id: Annotated[Identifier | None, BeforeValidator(_optional_text)] = None
    approver_role: Annotated[Identifier | None, BeforeValidator(_optional_text)] = None
    reason: OptionalText = None
    requested_at: Timestamp
    decided_at: Timestamp | None = None
    expires_at: Timestamp | None = None
    requester_id: Identifier | None = None
    tenant_id: Identifier | None = None
    session_id: Identifier | None = None
    descriptor_hash: Sha256 | None = None
    authorization_hash: Sha256 | None = None
    permission_policy_hash: Sha256 | None = None
    evidence_policy_hash: Sha256 | None = None
    risk_hash: Sha256 | None = None
    evidence_ids: StringSet = []
    citation_ids: StringSet = []
    approver_roles: StringSet = []
    idempotency_key: Identifier | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ApprovalRecord":
        """Require decision attribution and coherent expiry timestamps."""
        if (self.scope == "tool") != (self.manifest_hash is None) or (self.scope == "recovery") != bool(
            self.effect_ids
        ):
            raise ValueError("approval_scope_conflict")
        decided = self.decision in {
            ApprovalDecision.APPROVED,
            ApprovalDecision.REJECTED,
            ApprovalDecision.NEEDS_CHANGES,
        }
        if decided and (not self.approver_id or self.decided_at is None):
            raise ValueError("approval_decision_requires_actor_and_time")
        if self.decision is ApprovalDecision.PENDING and (self.approver_id is not None or self.decided_at is not None):
            raise ValueError("pending_approval_has_decision")
        if self.decided_at is not None and self.decided_at < self.requested_at:
            raise ValueError("invalid_approval_chronology")
        if self.expires_at is not None and self.expires_at <= self.requested_at:
            raise ValueError("invalid_approval_expiry")
        if self.audit.occurred_at != (self.decided_at or self.requested_at):
            raise ValueError("audit_field_mismatch")
        return self


class PermissionLabel(StrEnum):
    """Descriptive labels only; this module cannot grant or execute permissions."""

    READ_ONLY = "READ_ONLY"
    SIMULATED = "SIMULATED"
    MUTATING = "MUTATING"
    IRREVERSIBLE = "IRREVERSIBLE"


class ExecutionStep(DomainModel):
    """Plan step retaining parameter hashes instead of arbitrary tool arguments."""

    step_id: Identifier
    order: PositiveInt
    tool_name: Identifier
    permission: PermissionLabel
    arguments_hash: Sha256
    idempotency_key: Identifier
    expected_effect: Text
    timeout_seconds: PositiveInt
    compensation_step_id: Annotated[Identifier | None, BeforeValidator(_optional_text)] = None


class ExecutionPlanInput(AuditedRecord):
    """Typed plan input excluding its derived hash."""

    plan_id: Identifier
    steps: list[ExecutionStep] = Field(min_length=1)
    rollback_plan: Text
    verification_plan: Text
    dry_run: bool
    permissions: Annotated[list[PermissionLabel], BeforeValidator(_sorted_unique)]
    rollback_steps: list[ExecutionStep] = Field(default_factory=list)
    manifest: ExecutionManifest | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> "ExecutionPlanInput":
        """Check step identity, order, labels and compensation references."""
        all_steps = self.steps + self.rollback_steps
        if len({step.step_id for step in all_steps}) != len(all_steps):
            raise ValueError("duplicate_step_id")
        for steps in (self.steps, self.rollback_steps):
            orders = [step.order for step in steps]
            if orders != sorted(set(orders)):
                raise ValueError("invalid_step_order")
        if not {step.permission for step in all_steps} <= set(self.permissions):
            raise ValueError("undeclared_step_permission")
        rollback_ids = {step.step_id for step in self.rollback_steps}
        if any(step.compensation_step_id not in rollback_ids for step in self.steps if step.compensation_step_id):
            raise ValueError("unknown_compensation_step")
        if any(step.compensation_step_id for step in self.rollback_steps):
            raise ValueError("nested_compensation_not_supported")
        if self.manifest is not None:
            manifest = {step.step_id: step for step in self.manifest.steps}
            if set(manifest) != {step.step_id for step in all_steps}:
                raise ValueError("manifest_step_mismatch")
            if self.dry_run != (self.manifest.mode is ExecutionMode.SIMULATION):
                raise ValueError("manifest_mode_mismatch")
            if self.manifest.policy_version != self.audit.policy_version:
                raise ValueError("manifest_policy_mismatch")
            for step in all_steps:
                pin = manifest[step.step_id]
                direction = Direction.FORWARD if step in self.steps else Direction.COMPENSATION
                if (pin.tool_name, pin.input_hash, pin.order, pin.direction) != (
                    step.tool_name,
                    step.arguments_hash,
                    step.order,
                    direction,
                ) or step.permission is not PermissionLabel.MUTATING:
                    raise ValueError("manifest_step_mismatch")
                if direction is Direction.FORWARD and step.compensation_step_id:
                    if manifest[step.compensation_step_id].compensates != step.step_id:
                        raise ValueError("manifest_compensation_mismatch")
        return self


def plan_input_hash(plan: ExecutionPlanInput) -> str:
    """Hash executable meaning while excluding plan/run identity and audit time."""
    return canonical_hash(
        {
            field: getattr(plan, field)
            for field in (
                "change_id",
                "revision",
                "request_hash",
                "steps",
                "rollback_plan",
                "verification_plan",
                "dry_run",
                "permissions",
                "rollback_steps",
            )
        }
        | {"policy_version": plan.audit.policy_version, "contract_version": plan.audit.contract_version}
        | ({"manifest": plan.manifest} if plan.manifest is not None else {})
    )


class ExecutionPlan(ExecutionPlanInput):
    """Plan with a validated deterministic hash; not an executor."""

    record_type: Literal["plan"] = "plan"
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan_hash(self) -> "ExecutionPlan":
        """Reject serialized plans whose contents do not match their hash."""
        if self.plan_hash != plan_input_hash(self):
            raise PydanticCustomError("hash_mismatch", "hash_mismatch")
        return self


def build_execution_plan(plan: ExecutionPlanInput) -> ExecutionPlan:
    """Seal a typed plan with its derived canonical hash."""
    return ExecutionPlan.model_validate({**plan.model_dump(), "plan_hash": plan_input_hash(plan)})


class EventStatus(StrEnum):
    """Append-only observation status for a described plan step."""

    PLANNED = "PLANNED"
    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNKNOWN = "UNKNOWN"


class ExecutionEvent(AuditedRecord):
    """Event contract; does not perform the operation described by the event."""

    record_type: Literal["event"] = "event"
    event_id: Identifier
    plan_hash: Sha256
    step_id: Identifier
    status: EventStatus
    occurred_at: Timestamp
    actor_id: Identifier
    reason: OptionalText = None
    mode: ExecutionMode | None = None
    direction: Direction = Direction.FORWARD
    claim_id: Identifier | None = None
    audit_ids: tuple[Identifier, ...] = ()
    reconciles_event_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "ExecutionEvent":
        """Keep failures explicit rather than silently returning success."""
        if self.status in {EventStatus.FAILED, EventStatus.SKIPPED, EventStatus.UNKNOWN} and not self.reason:
            raise ValueError("event_requires_reason")
        return self


class VerificationStatus(StrEnum):
    """Independent verification outcomes without implicit UNKNOWN promotion."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class VerificationCheck(DomainModel):
    """One externally supplied observation with source and evidence references."""

    check_id: Identifier
    description: Text
    status: VerificationStatus
    evidence_ids: EvidenceIds
    source_type: Identifier
    observed_at: Timestamp
    failure_reason: OptionalText = None

    @model_validator(mode="after")
    def validate_check(self) -> "VerificationCheck":
        """Require evidence for success and reasons for failure or uncertainty."""
        if self.status is VerificationStatus.PASSED and not self.evidence_ids:
            raise ValueError("passed_check_requires_evidence")
        if self.status is not VerificationStatus.PASSED and not self.failure_reason:
            raise ValueError("verification_check_requires_reason")
        return self


class VerificationResult(AuditedRecord):
    """Supplied verification result; independent labels do not prove authenticity."""

    record_type: Literal["verification"] = "verification"
    verification_id: Identifier
    mode: ExecutionMode | None = None
    citation_ids: tuple[Identifier, ...] = ()
    audit_ids: tuple[Identifier, ...] = ()
    observation_hash: Sha256 | None = None
    plan_hash: Sha256
    status: VerificationStatus
    checks: list[VerificationCheck]
    independent_source_types: StringSet
    verified_at: Timestamp

    @model_validator(mode="after")
    def validate_verification(self) -> "VerificationResult":
        """Enforce aggregate consistency without verifying production sources."""
        if len({check.check_id for check in self.checks}) != len(self.checks):
            raise ValueError("duplicate_verification_check")
        sources = {check.source_type for check in self.checks}
        if not set(self.independent_source_types) <= sources:
            raise ValueError("unknown_independent_source")
        if self.status is VerificationStatus.PASSED:
            if (
                not self.checks
                or not self.independent_source_types
                or any(check.status is not VerificationStatus.PASSED for check in self.checks)
            ):
                raise ValueError("inconsistent_passed_verification")
        if self.status is VerificationStatus.FAILED and not any(
            check.status is VerificationStatus.FAILED for check in self.checks
        ):
            raise ValueError("inconsistent_failed_verification")
        if (
            any(check.observed_at > self.verified_at for check in self.checks)
            or self.audit.occurred_at != self.verified_at
        ):
            raise ValueError("invalid_verification_chronology")
        return self


class ReceiptStatus(StrEnum):
    """Terminal receipt outcome distinct from workflow status."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    RECOVERED = "RECOVERED"
    SIMULATED = "SIMULATED"
    UNKNOWN = "UNKNOWN"


class ExecutionReceipt(AuditedRecord):
    """Audit receipt shape; cross-record references are checked by run state."""

    record_type: Literal["receipt"] = "receipt"
    receipt_id: Identifier
    intent_hash: Sha256 | None = None
    mode: ExecutionMode | None = None
    manifest_hash: Sha256 | None = None
    simulation_id: Identifier | None = None
    citation_ids: tuple[Identifier, ...] = ()
    authorization_ids: tuple[Identifier, ...] = ()
    claim_ids: tuple[Identifier, ...] = ()
    compensation_ids: tuple[Identifier, ...] = ()
    audit_ids: tuple[Identifier, ...] = ()
    prior_receipt_ids: tuple[Identifier, ...] = ()
    terminal: bool = True
    plan_hash: Sha256 | None = None
    evidence_ids: EvidenceIds = Field(default_factory=list)
    evidence_assessment: EvidenceAssessment | None = None
    risk_assessment: RiskAssessment | None = None
    approval_id: Identifier | None = None
    verification_id: Identifier | None = None
    final_status: ReceiptStatus
    execution_event_ids: StringSet = Field(default_factory=list)
    issued_at: Timestamp

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExecutionReceipt":
        """Require a complete success shape and an explicit early failure reason."""
        if self.terminal != (self.final_status is not ReceiptStatus.UNKNOWN):
            raise ValueError("receipt_terminal_mismatch")
        if self.final_status is ReceiptStatus.SIMULATED and (
            self.mode is not ExecutionMode.SIMULATION or not self.simulation_id or self.execution_event_ids
        ):
            raise ValueError("simulation_receipt_mismatch")
        for result in (self.evidence_assessment, self.risk_assessment):
            if result is not None and any(
                getattr(result, field) != getattr(self, field) for field in RunBound.model_fields
            ):
                raise ValueError("receipt_assessment_binding_mismatch")
        if self.final_status in {ReceiptStatus.SUCCEEDED, ReceiptStatus.RECOVERED}:
            if not all(
                (self.plan_hash, self.approval_id, self.verification_id, self.execution_event_ids, self.evidence_ids)
            ):
                raise ValueError("success_receipt_requires_references")
            if self.evidence_assessment is None or not self.evidence_assessment.sufficient:
                raise ValueError("success_receipt_requires_evidence_assessment")
            if self.risk_assessment is None or self.risk_assessment.level is RiskLevel.UNKNOWN:
                raise ValueError("success_receipt_requires_known_risk")
        elif not self.audit.reason:
            raise ValueError("failure_receipt_requires_reason")
        if self.audit.occurred_at != self.issued_at:
            raise ValueError("audit_field_mismatch")
        return self


class ChangeStatus(StrEnum):
    """Domain states only; none of these values schedules a workflow or a tool."""

    RECEIVED = "RECEIVED"
    COLLECTING_EVIDENCE = "COLLECTING_EVIDENCE"
    WAITING_FOR_EVIDENCE = "WAITING_FOR_EVIDENCE"
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    RISK_ASSESSED = "RISK_ASSESSED"
    PLAN_READY = "PLAN_READY"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    APPROVED = "APPROVED"
    DRY_RUN = "DRY_RUN"
    DRY_RUN_PASSED = "DRY_RUN_PASSED"
    DRY_RUN_FAILED = "DRY_RUN_FAILED"
    EXECUTING_SANDBOX = "EXECUTING_SANDBOX"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    RECOVERING = "RECOVERING"
    VERIFYING = "VERIFYING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    RECEIPTED = "RECEIPTED"


ALLOWED_TRANSITIONS: dict[ChangeStatus, frozenset[ChangeStatus]] = {
    ChangeStatus.RECEIVED: frozenset({ChangeStatus.COLLECTING_EVIDENCE}),
    ChangeStatus.COLLECTING_EVIDENCE: frozenset({ChangeStatus.WAITING_FOR_EVIDENCE, ChangeStatus.EVIDENCE_SUFFICIENT}),
    ChangeStatus.WAITING_FOR_EVIDENCE: frozenset({ChangeStatus.COLLECTING_EVIDENCE}),
    ChangeStatus.EVIDENCE_SUFFICIENT: frozenset({ChangeStatus.RISK_ASSESSED}),
    ChangeStatus.RISK_ASSESSED: frozenset({ChangeStatus.PLAN_READY}),
    ChangeStatus.PLAN_READY: frozenset({ChangeStatus.WAITING_FOR_APPROVAL}),
    ChangeStatus.WAITING_FOR_APPROVAL: frozenset(
        {ChangeStatus.APPROVED, ChangeStatus.REJECTED, ChangeStatus.WAITING_FOR_EVIDENCE}
    ),
    ChangeStatus.APPROVED: frozenset({ChangeStatus.DRY_RUN}),
    ChangeStatus.DRY_RUN: frozenset(
        {ChangeStatus.EXECUTING_SANDBOX, ChangeStatus.DRY_RUN_FAILED, ChangeStatus.DRY_RUN_PASSED}
    ),
    ChangeStatus.DRY_RUN_PASSED: frozenset({ChangeStatus.RECEIPTED}),
    ChangeStatus.DRY_RUN_FAILED: frozenset({ChangeStatus.FAILED}),
    ChangeStatus.EXECUTING_SANDBOX: frozenset(
        {ChangeStatus.VERIFYING, ChangeStatus.EXECUTION_FAILED, ChangeStatus.EXECUTION_UNKNOWN}
    ),
    ChangeStatus.EXECUTION_UNKNOWN: frozenset(
        {ChangeStatus.VERIFYING, ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.FAILED}
    ),
    ChangeStatus.EXECUTION_FAILED: frozenset({ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.FAILED}),
    ChangeStatus.RECOVERY_REQUIRED: frozenset({ChangeStatus.RECOVERING}),
    ChangeStatus.RECOVERING: frozenset({ChangeStatus.VERIFYING, ChangeStatus.FAILED, ChangeStatus.EXECUTION_UNKNOWN}),
    ChangeStatus.VERIFYING: frozenset({ChangeStatus.SUCCEEDED, ChangeStatus.RECOVERY_REQUIRED, ChangeStatus.FAILED}),
    ChangeStatus.SUCCEEDED: frozenset({ChangeStatus.RECEIPTED}),
    ChangeStatus.REJECTED: frozenset({ChangeStatus.RECEIPTED}),
    ChangeStatus.FAILED: frozenset({ChangeStatus.RECEIPTED}),
    ChangeStatus.RECEIPTED: frozenset(),
}
TERMINAL_STATES = frozenset(
    {ChangeStatus.SUCCEEDED, ChangeStatus.REJECTED, ChangeStatus.FAILED, ChangeStatus.RECEIPTED}
)
REVISION_STATES = frozenset(
    {
        ChangeStatus.RECEIVED,
        ChangeStatus.COLLECTING_EVIDENCE,
        ChangeStatus.WAITING_FOR_EVIDENCE,
        ChangeStatus.EVIDENCE_SUFFICIENT,
        ChangeStatus.RISK_ASSESSED,
        ChangeStatus.PLAN_READY,
        ChangeStatus.WAITING_FOR_APPROVAL,
        ChangeStatus.APPROVED,
        ChangeStatus.DRY_RUN_FAILED,
    }
)
PLAN_REPLACEMENT_STATES = frozenset(
    {ChangeStatus.PLAN_READY, ChangeStatus.WAITING_FOR_APPROVAL, ChangeStatus.APPROVED, ChangeStatus.DRY_RUN_FAILED}
)


class CommandAction(StrEnum):
    """Explicit state change commands, including versioned replacement actions."""

    TRANSITION = "transition"
    REVISE = "revise"
    REPLACE_PLAN = "replace_plan"
    APPEND_EXECUTION_FACTS = "append_execution_facts"


class ApprovalInvalidation(AuditedRecord):
    """Append-only invalidation fact referencing an unchanged original decision."""

    record_type: Literal["approval_invalidation"] = "approval_invalidation"
    invalidation_id: Identifier
    approval_id: Identifier
    plan_hash: Sha256
    decision: Literal[ApprovalDecision.INVALIDATED, ApprovalDecision.EXPIRED, ApprovalDecision.REVOKED]


class SimulationResult(AuditedRecord):
    """Supplied simulation outcome, with no simulator implementation."""

    record_type: Literal["simulation"] = "simulation"
    simulation_id: Identifier
    plan_hash: Sha256
    passed: bool
    mode: ExecutionMode | None = None
    manifest_hash: Sha256 | None = None
    authorization_hash: Sha256 | None = None
    prediction_hash: Sha256 | None = None
    predictions: tuple[SimulationPrediction, ...] = ()
    audit_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_simulation(self) -> "SimulationResult":
        """Require an explanation for a supplied simulation failure."""
        if not self.passed and not self.audit.reason:
            raise ValueError("simulation_failure_requires_reason")
        return self


class RecoveryAuthorization(AuditedRecord):
    """Supplied recovery authorization shape, without authentication or execution."""

    record_type: Literal["recovery_authorization"] = "recovery_authorization"
    authorization_id: Identifier
    plan_hash: Sha256
    authorized: bool
    approval_id: Identifier | None = None
    authorization_hash: Sha256 | None = None
    effect_ids: tuple[Identifier, ...] = ()
    compensation_step_ids: tuple[Identifier, ...] = ()


DomainRecord = Annotated[
    EvidenceAssessment
    | RiskAssessment
    | ExecutionPlan
    | ApprovalRecord
    | ExecutionEvent
    | VerificationResult
    | ExecutionReceipt
    | ApprovalInvalidation
    | SimulationResult
    | RecoveryAuthorization,
    Field(discriminator="record_type"),
]


def record_plan_hash(record: DomainRecord) -> str | None:
    """Read the optional plan binding without introducing a duplicate result model."""
    if isinstance(record, (EvidenceAssessment, RiskAssessment)):
        return None
    return record.plan_hash


def record_key(record: DomainRecord) -> str:
    """Return an immutable, type-scoped record key for append-only storage."""
    own_identifier = {
        "plan": "plan_id",
        "approval": "approval_id",
        "event": "event_id",
        "verification": "verification_id",
        "receipt": "receipt_id",
        "approval_invalidation": "invalidation_id",
        "simulation": "simulation_id",
        "recovery_authorization": "authorization_id",
    }.get(record.record_type)
    if own_identifier is not None:
        return record.record_type + ":" + str(getattr(record, own_identifier))
    for field in (
        "plan_id",
        "approval_id",
        "event_id",
        "verification_id",
        "receipt_id",
        "invalidation_id",
        "simulation_id",
        "authorization_id",
    ):
        if field in type(record).model_fields:
            # Invalidation records also contain approval_id; use their own id.
            if isinstance(record, ApprovalInvalidation):
                return "approval_invalidation:" + record.invalidation_id
            return record.record_type + ":" + str(getattr(record, field))
    return record.record_type + ":" + canonical_hash(record)


def matches_binding(record: RunBound, identity: RunBound) -> bool:
    """Compare every change/run/revision/request binding, without coercion."""
    return all(getattr(record, field) == getattr(identity, field) for field in RunBound.model_fields)


class TransitionRecord(AuditedRecord):
    """One accepted command with before/after state and immutable attribution."""

    transition_id: Identifier
    command_id: Identifier
    action: CommandAction
    from_status: ChangeStatus
    to_status: ChangeStatus
    from_revision: PositiveInt
    plan_hash: Sha256 | None = None
    state_version: PositiveInt


class StateInputs(DomainModel):
    """Typed inputs to a transition; absence means retain the current value."""

    evidence_items: list[EvidenceItem] | None = None
    verification_evidence: list[EvidenceItem] = Field(default_factory=list)
    recovery_approval: ApprovalRecord | None = None
    evidence_assessment: EvidenceAssessment | None = None
    risk_assessment: RiskAssessment | None = None
    execution_plan: ExecutionPlan | None = None
    approval: ApprovalRecord | None = None
    execution_events: list[ExecutionEvent] = Field(default_factory=list)
    verification_result: VerificationResult | None = None
    receipt: ExecutionReceipt | None = None
    simulation_result: SimulationResult | None = None
    recovery_authorization: RecoveryAuthorization | None = None


class RunCommandFields(DomainModel):
    """Shared command payload excluding the draft or sealed audit representation."""

    change_id: Identifier
    run_id: Identifier
    command_id: Identifier
    expected_revision: PositiveInt
    expected_state_version: Annotated[int, Field(strict=True, ge=0)]
    action: CommandAction = CommandAction.TRANSITION
    target_status: ChangeStatus | None = None
    inputs: StateInputs = Field(default_factory=StateInputs)
    replacement_request: ChangeRequest | None = None
    replacement_plan: ExecutionPlan | None = None

    @model_validator(mode="after")
    def validate_action_shape(self) -> "RunCommandFields":
        """Reject irrelevant fields rather than ignoring hidden command payloads."""
        if self.action is CommandAction.TRANSITION:
            if self.target_status is None or self.replacement_request or self.replacement_plan:
                raise ValueError("invalid_transition_command")
        elif self.action is CommandAction.APPEND_EXECUTION_FACTS:
            if (
                self.target_status is not None
                or self.replacement_request
                or self.replacement_plan
                or self.inputs == StateInputs()
            ):
                raise ValueError("invalid_execution_facts_command")
        else:
            if self.target_status is not None or self.inputs != StateInputs():
                raise ValueError("invalid_replacement_command")
            if self.action is CommandAction.REVISE and (
                self.replacement_request is None or self.replacement_plan is not None
            ):
                raise ValueError("invalid_revision_command")
            if self.action is CommandAction.REPLACE_PLAN and (
                self.replacement_plan is None or self.replacement_request is not None
            ):
                raise ValueError("invalid_plan_command")
        return self


class RunCommandInput(RunCommandFields):
    """Typed command draft whose input hash is computed before validation."""

    audit: AuditContext


def command_input_hash(command: RunCommandFields) -> str:
    """Bind all typed command meaning and attribution except its derived hash."""
    payload = command.model_dump()
    payload["audit"].pop("input_hash", None)
    return canonical_hash(payload)


class RunCommand(RunCommandFields):
    """Command with a validated input hash and explicit concurrency expectations."""

    audit: AuditFields

    @model_validator(mode="after")
    def validate_command_hash(self) -> "RunCommand":
        """Reject mutated or forged command payloads."""
        if self.audit.input_hash != command_input_hash(self):
            raise PydanticCustomError("hash_mismatch", "hash_mismatch")
        return self


def build_run_command(command: RunCommandInput) -> RunCommand:
    """Seal a typed command draft without reading clock or generating ids."""
    payload = command.model_dump()
    payload["audit"]["input_hash"] = command_input_hash(command)
    return RunCommand.model_validate(payload)


class ChangeRunState(RunBound):
    """Serializable domain aggregate with append-only history and no Graph state."""

    contract_version: Literal["releaseguard.v1"] = CONTRACT_VERSION
    change_request: ChangeRequest
    status: ChangeStatus = ChangeStatus.RECEIVED
    state_version: Annotated[int, Field(strict=True, ge=0)] = 0
    request_history: list[ChangeRequest]
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_history: list[EvidenceItem] = Field(default_factory=list)
    evidence_assessment: EvidenceAssessment | None = None
    risk_assessment: RiskAssessment | None = None
    execution_plan: ExecutionPlan | None = None
    approval: ApprovalRecord | None = None
    execution_events: list[ExecutionEvent] = Field(default_factory=list)
    verification_result: VerificationResult | None = None
    receipt: ExecutionReceipt | None = None
    simulation_result: SimulationResult | None = None
    recovery_authorization: RecoveryAuthorization | None = None
    execution_started_at: Timestamp | None = None
    execution_approval_id: Identifier | None = None
    errors: list[Text] = Field(default_factory=list)
    record_history: list[DomainRecord] = Field(default_factory=list)
    transition_history: list[TransitionRecord] = Field(default_factory=list)
    fact_history: list[TransitionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_aggregate(self) -> "ChangeRunState":
        """Validate references and history on construction and JSON restoration."""
        request = self.change_request
        if (self.change_id, self.revision, self.request_hash) != (
            request.change_id,
            request.revision,
            request.request_hash,
        ):
            raise ValueError("request_binding_mismatch")
        if not self.request_history or self.request_history[-1] != request:
            raise ValueError("invalid_request_history")
        for previous, current in zip(self.request_history, self.request_history[1:], strict=False):
            if current.revision != previous.revision + 1 or current.change_id != previous.change_id:
                raise ValueError("invalid_request_history")
        if any(item.change_id != self.change_id for item in self.request_history):
            raise ValueError("foreign_request_history")
        revisions = {item.revision: item for item in self.request_history}
        history = {record_key(record): record for record in self.record_history}
        if len(history) != len(self.record_history):
            raise ValueError("duplicate_record_id")
        evidence = {item.evidence_id: item for item in self.evidence_history}
        if len(evidence) != len(self.evidence_history):
            raise ValueError("duplicate_evidence_id")
        for item in self.evidence_items:
            if evidence.get(item.evidence_id) != item:
                raise ValueError("missing_evidence_history")
        AssessmentContext(
            identity=RunBound.model_validate(self.model_dump(include=set(RunBound.model_fields))),
            evidence_items=self.evidence_items,
            evidence_assessment=self.evidence_assessment,
            risk_assessment=self.risk_assessment,
        )
        plans = {
            (record.revision, record.plan_hash): record
            for record in self.record_history
            if isinstance(record, ExecutionPlan)
        }
        for record in self.record_history:
            historical_request = revisions.get(record.revision)
            if (
                record.change_id != self.change_id
                or record.run_id != self.run_id
                or historical_request is None
                or record.request_hash != historical_request.request_hash
            ):
                raise ValueError("record_binding_mismatch")
            if "plan_hash" in type(record).model_fields and not isinstance(record, ExecutionPlan):
                plan_hash = record_plan_hash(record)
                if plan_hash is not None and (record.revision, plan_hash) not in plans:
                    raise ValueError("unknown_plan_reference")
            if isinstance(record, (EvidenceAssessment, RiskAssessment, ExecutionReceipt)) and not set(
                record.evidence_ids
            ) <= set(evidence):
                raise ValueError("unknown_evidence_reference")
            if isinstance(record, VerificationResult) and any(
                not set(check.evidence_ids) <= set(evidence) for check in record.checks
            ):
                raise ValueError("unknown_evidence_reference")
            if isinstance(record, ApprovalInvalidation):
                original = history.get("approval:" + record.approval_id)
                if (
                    not isinstance(original, ApprovalRecord)
                    or not matches_binding(original, record)
                    or original.plan_hash != record.plan_hash
                ):
                    raise ValueError("unknown_approval_reference")
        for record in self.current_records():
            if not matches_binding(record, self) or history.get(record_key(record)) != record:
                raise ValueError("current_record_binding_mismatch")
            if "plan_hash" in type(record).model_fields and not isinstance(record, ExecutionPlan):
                if record_plan_hash(record) is not None and (
                    self.execution_plan is None or record_plan_hash(record) != self.execution_plan.plan_hash
                ):
                    raise ValueError("current_plan_binding_mismatch")
        _validate_event_history(self.record_history, plans)
        for record in self.record_history:
            if isinstance(record, RecoveryAuthorization) and plans[(record.revision, record.plan_hash)].manifest:
                approval = history.get("approval:" + str(record.approval_id))
                manifest = plans[(record.revision, record.plan_hash)].manifest
                assert manifest is not None
                selected = tuple(
                    pin.step_id
                    for pin in sorted(manifest.steps, key=lambda pin: pin.order)
                    if pin.compensates in record.effect_ids
                )
                if (
                    not isinstance(approval, ApprovalRecord)
                    or approval.scope != "recovery"
                    or not approval_valid_at(approval, record.audit.occurred_at)
                    or approval.effect_ids != record.effect_ids
                    or approval.plan_hash != record.plan_hash
                    or not matches_binding(approval, record)
                    or not record.authorization_hash
                    or selected != record.compensation_step_ids
                    or not selected
                    or any(
                        not any(
                            isinstance(event, ExecutionEvent)
                            and event.step_id == effect
                            and event.status is EventStatus.SUCCEEDED
                            and event.direction is Direction.FORWARD
                            and event.plan_hash == record.plan_hash
                            and event.revision == record.revision
                            and event.occurred_at <= record.audit.occurred_at
                            for event in self.record_history
                        )
                        for effect in record.effect_ids
                    )
                ):
                    raise ValueError("recovery_authorization_reference_mismatch")
        self._validate_transitions(revisions)
        self._validate_status_shape()
        if self.receipt is not None:
            self._validate_receipt_references(history)
        return self

    def current_records(self) -> list[DomainRecord]:
        """Return current result records separately from preserved historical facts."""
        records: list[DomainRecord] = [
            record
            for record in (
                self.evidence_assessment,
                self.risk_assessment,
                self.execution_plan,
                self.approval,
                self.verification_result,
                self.receipt,
                self.simulation_result,
                self.recovery_authorization,
            )
            if record is not None
        ]
        records.extend(self.execution_events)
        return records

    def _validate_transitions(self, revisions: dict[int, ChangeRequest]) -> None:
        """Ensure a contiguous, attributed chain ending at the current state."""
        journal = sorted(self.transition_history + self.fact_history, key=lambda item: item.state_version)
        if len(journal) != self.state_version:
            raise ValueError("invalid_state_version")
        status, revision = ChangeStatus.RECEIVED, self.request_history[0].revision
        seen: set[str] = set()
        last_time = self.request_history[0].updated_at
        for index, record in enumerate(journal, 1):
            if (
                record.command_id in seen
                or record.state_version != index
                or record.from_status is not status
                or record.from_revision != revision
            ):
                raise ValueError("invalid_transition_history")
            if (
                record.run_id != self.run_id
                or record.change_id != self.change_id
                or record.revision not in revisions
                or record.request_hash != revisions[record.revision].request_hash
            ):
                raise ValueError("transition_binding_mismatch")
            if record.audit.occurred_at < last_time:
                raise ValueError("invalid_transition_chronology")
            expected_revision = revision + 1 if record.action is CommandAction.REVISE else revision
            if record.revision != expected_revision:
                raise ValueError("invalid_transition_revision")
            if record.action is CommandAction.TRANSITION:
                allowed = ALLOWED_TRANSITIONS[status] | (
                    frozenset({ChangeStatus.FAILED}) if status not in TERMINAL_STATES else frozenset()
                )
                if record.to_status not in allowed:
                    raise ValueError("invalid_state_transition")
            if record.action is CommandAction.APPEND_EXECUTION_FACTS and (
                record not in self.fact_history
                or record.to_status is not status
                or status
                not in {
                    ChangeStatus.DRY_RUN,
                    ChangeStatus.EXECUTING_SANDBOX,
                    ChangeStatus.EXECUTION_UNKNOWN,
                    ChangeStatus.RECOVERING,
                    ChangeStatus.VERIFYING,
                    ChangeStatus.RECOVERY_REQUIRED,
                }
            ):
                raise ValueError("invalid_fact_history")
            if record in self.transition_history and record.action is CommandAction.APPEND_EXECUTION_FACTS:
                raise ValueError("invalid_transition_history")
            if record.action is CommandAction.REVISE and (
                status not in REVISION_STATES or record.to_status is not ChangeStatus.RECEIVED
            ):
                raise ValueError("invalid_revision_transition")
            if record.action is CommandAction.REPLACE_PLAN and (
                status not in PLAN_REPLACEMENT_STATES or record.to_status is not ChangeStatus.PLAN_READY
            ):
                raise ValueError("invalid_plan_transition")
            seen.add(record.command_id)
            status, revision, last_time = record.to_status, record.revision, record.audit.occurred_at
        if status is not self.status or revision != self.revision:
            raise ValueError("invalid_transition_history")

    def _validate_status_shape(self) -> None:
        """Reject unsafe status shapes even when restored without a service call."""
        evidence_states = set(ChangeStatus) - {
            ChangeStatus.RECEIVED,
            ChangeStatus.COLLECTING_EVIDENCE,
            ChangeStatus.WAITING_FOR_EVIDENCE,
            ChangeStatus.FAILED,
            ChangeStatus.REJECTED,
            ChangeStatus.RECEIPTED,
        }
        if self.status in evidence_states and (
            self.evidence_assessment is None or not self.evidence_assessment.sufficient
        ):
            raise ValueError("status_requires_sufficient_assessment")
        risk_states = evidence_states - {ChangeStatus.EVIDENCE_SUFFICIENT}
        if self.status in risk_states and self.risk_assessment is None:
            raise ValueError("status_requires_risk")
        plan_states = risk_states - {ChangeStatus.RISK_ASSESSED}
        if self.status in plan_states and (
            self.risk_assessment is None
            or self.risk_assessment.level is RiskLevel.UNKNOWN
            or self.execution_plan is None
        ):
            raise ValueError("status_requires_known_risk_and_plan")
        approval_states = plan_states - {ChangeStatus.PLAN_READY, ChangeStatus.WAITING_FOR_APPROVAL}
        if self.status in approval_states and (
            self.approval is None or self.approval.decision is not ApprovalDecision.APPROVED
        ):
            raise ValueError("status_requires_approval")
        if self.status in approval_states and any(
            isinstance(record, ApprovalInvalidation)
            and self.approval is not None
            and record.approval_id == self.approval.approval_id
            for record in self.record_history
        ):
            raise ValueError("approval_invalidated")
        if self.status in {ChangeStatus.APPROVED, ChangeStatus.DRY_RUN}:
            if self.approval is None or not approval_valid_at(
                self.approval, self.transition_history[-1].audit.occurred_at
            ):
                raise ValueError("approval_expired")
        execution_states = {
            ChangeStatus.EXECUTING_SANDBOX,
            ChangeStatus.EXECUTION_FAILED,
            ChangeStatus.EXECUTION_UNKNOWN,
            ChangeStatus.RECOVERY_REQUIRED,
            ChangeStatus.RECOVERING,
            ChangeStatus.VERIFYING,
            ChangeStatus.SUCCEEDED,
        }
        if self.status in execution_states:
            if (
                self.execution_started_at is None
                or self.execution_approval_id is None
                or self.execution_plan is None
                or self.execution_plan.dry_run
            ):
                raise ValueError("status_requires_execution_start")
        if self.status is ChangeStatus.DRY_RUN_PASSED and (
            self.execution_plan is None
            or not self.execution_plan.dry_run
            or self.simulation_result is None
            or not self.simulation_result.passed
            or self.execution_events
        ):
            raise ValueError("simulation_state_mismatch")
        if self.status is ChangeStatus.SUCCEEDED and (
            self.verification_result is None or self.verification_result.status is not VerificationStatus.PASSED
        ):
            raise ValueError("status_requires_passed_verification")
        if (self.status is ChangeStatus.RECEIPTED) != (self.receipt is not None and self.receipt.terminal):
            raise ValueError("status_receipt_mismatch")
        if (
            self.receipt is not None
            and not self.receipt.terminal
            and self.status is not ChangeStatus.EXECUTION_UNKNOWN
        ):
            raise ValueError("unknown_receipt_state_mismatch")

    def _validate_receipt_references(self, history: dict[str, DomainRecord]) -> None:
        """Prevent fabricated success through mismatched audit record references."""
        receipt = self.receipt
        if receipt is None:
            return
        previous = self.transition_history[-1].from_status
        if receipt.final_status is ReceiptStatus.UNKNOWN and self.status is not ChangeStatus.EXECUTION_UNKNOWN:
            raise ValueError("unknown_receipt_state_mismatch")
        expected = {
            ChangeStatus.FAILED: {ReceiptStatus.FAILED},
            ChangeStatus.REJECTED: {ReceiptStatus.REJECTED},
            ChangeStatus.SUCCEEDED: {ReceiptStatus.SUCCEEDED, ReceiptStatus.RECOVERED},
            ChangeStatus.DRY_RUN_PASSED: {ReceiptStatus.SIMULATED},
        }
        if receipt.final_status is not ReceiptStatus.UNKNOWN and receipt.final_status not in expected.get(
            previous, set()
        ):
            raise ValueError("receipt_status_mismatch")
        event_ids = {event.event_id for event in self.execution_events}
        if self.execution_plan is not None and self.execution_plan.manifest is not None:
            plan = self.execution_plan
            assert plan.manifest is not None
            if (
                not receipt.intent_hash
                or receipt.manifest_hash != plan.manifest.manifest_hash
                or receipt.plan_hash != plan.plan_hash
                or receipt.input_hash != plan.input_hash
                or not receipt.audit_ids
                or not receipt.authorization_ids
                or receipt.simulation_id != (self.simulation_result.simulation_id if self.simulation_result else None)
                or set(receipt.execution_event_ids) != event_ids
                or set(receipt.claim_ids) != {event.claim_id for event in self.execution_events}
                or set(receipt.compensation_ids)
                != {event.event_id for event in self.execution_events if event.direction is Direction.COMPENSATION}
                or not {identifier for event in self.execution_events for identifier in event.audit_ids}
                <= set(receipt.audit_ids)
                or any(
                    not isinstance(history.get("receipt:" + identifier), ExecutionReceipt)
                    for identifier in receipt.prior_receipt_ids
                )
                or receipt.mode
                is not (
                    ExecutionMode.SIMULATION
                    if receipt.final_status is ReceiptStatus.SIMULATED
                    else ExecutionMode.SANDBOX
                )
            ):
                raise ValueError("receipt_execution_reference_mismatch")
            if self.verification_result and (
                self.verification_result.mode is not ExecutionMode.SANDBOX
                or not self.verification_result.citation_ids
                or not self.verification_result.audit_ids
                or not self.verification_result.observation_hash
                or not set(self.verification_result.citation_ids) <= set(receipt.citation_ids)
                or not set(self.verification_result.audit_ids) <= set(receipt.audit_ids)
            ):
                raise ValueError("receipt_verification_reference_mismatch")
        if not set(receipt.execution_event_ids) <= event_ids:
            raise ValueError("unknown_event_reference")
        for prefix, identifier, cls in (
            ("approval:", receipt.approval_id, ApprovalRecord),
            ("verification:", receipt.verification_id, VerificationResult),
        ):
            if identifier is not None:
                record = history.get(prefix + identifier)
                if (
                    not isinstance(record, cls)
                    or not matches_binding(record, self)
                    or record_plan_hash(record) != receipt.plan_hash
                ):
                    raise ValueError("receipt_reference_mismatch")
        if receipt.evidence_assessment != self.evidence_assessment or receipt.risk_assessment != self.risk_assessment:
            raise ValueError("receipt_snapshot_mismatch")
        if receipt.final_status in {ReceiptStatus.SUCCEEDED, ReceiptStatus.RECOVERED}:
            approval = history.get("approval:" + str(receipt.approval_id))
            verification = history.get("verification:" + str(receipt.verification_id))
            if (
                not isinstance(approval, ApprovalRecord)
                or self.execution_started_at is None
                or not approval_valid_at(approval, self.execution_started_at)
                or approval.approval_id != self.execution_approval_id
            ):
                raise ValueError("receipt_requires_execution_approval")
            if (
                not isinstance(verification, VerificationResult)
                or verification.status is not VerificationStatus.PASSED
                or verification != self.verification_result
            ):
                raise ValueError("receipt_requires_passed_verification")
            recovered = self.recovery_authorization is not None
            if (receipt.final_status is ReceiptStatus.RECOVERED) != recovered:
                raise ValueError("receipt_recovery_mismatch")
            if not completed_steps(self, rollback=recovered) or set(receipt.execution_event_ids) != event_ids:
                raise ValueError("receipt_requires_complete_events")


def approval_valid_at(approval: ApprovalRecord, occurred_at: datetime) -> bool:
    """Check a recorded approval at an explicit time, never against ambient clock."""
    return (
        approval.decision is ApprovalDecision.APPROVED
        and approval.decided_at is not None
        and approval.decided_at <= occurred_at
        and (approval.expires_at is None or occurred_at < approval.expires_at)
    )


def completed_steps(state: ChangeRunState, *, rollback: bool = False) -> bool:
    """Check supplied terminal step observations without executing those steps."""
    if state.execution_plan is None:
        return False
    steps = state.execution_plan.rollback_steps if rollback else state.execution_plan.steps
    if rollback and state.execution_plan.manifest is not None:
        selected = state.recovery_authorization.compensation_step_ids if state.recovery_authorization else ()
        steps = [step for step in steps if step.step_id in selected]
    latest = {event.step_id: event.status for event in state.execution_events}
    return bool(steps) and all(latest.get(step.step_id) is EventStatus.SUCCEEDED for step in steps)


def _validate_event_history(records: list[DomainRecord], plans: dict[tuple[int, str], ExecutionPlan]) -> None:
    """Reject event mutation, foreign steps and success without a started event."""
    latest: dict[tuple[int, str, str], ExecutionEvent] = {}
    allowed = {
        None: {EventStatus.PLANNED, EventStatus.STARTED},
        EventStatus.PLANNED: {EventStatus.STARTED, EventStatus.SKIPPED},
        EventStatus.STARTED: {EventStatus.SUCCEEDED, EventStatus.FAILED, EventStatus.UNKNOWN},
        EventStatus.UNKNOWN: {EventStatus.SUCCEEDED, EventStatus.FAILED},
    }
    for record in records:
        if not isinstance(record, ExecutionEvent):
            continue
        plan = plans[(record.revision, record.plan_hash)]
        if plan.manifest:
            pin = next((pin for pin in plan.manifest.steps if pin.step_id == record.step_id), None)
            if (
                pin is None
                or record.mode is not ExecutionMode.SANDBOX
                or record.direction is not pin.direction
                or record.input_hash != pin.input_hash
                or not record.claim_id
                or not record.audit_ids
            ):
                raise ValueError("execution_manifest_event_mismatch")
        if record.step_id not in {step.step_id for step in plan.steps + plan.rollback_steps}:
            raise ValueError("unknown_execution_step")
        key = (record.revision, record.plan_hash, record.step_id)
        previous = latest.get(key)
        if previous and previous.status is EventStatus.UNKNOWN and record.reconciles_event_id != previous.event_id:
            raise ValueError("reconciliation_reference_missing")
        if (
            plan.manifest
            and previous
            and (record.claim_id != previous.claim_id or record.input_hash != previous.input_hash)
        ):
            raise ValueError("execution_claim_reference_mismatch")
        if (
            record.status not in allowed.get(previous.status if previous else None, set())
            or previous
            and record.occurred_at < previous.occurred_at
        ):
            raise ValueError("invalid_event_sequence")
        latest[key] = record


def new_change_run(request: ChangeRequest, run_id: str) -> ChangeRunState:
    """Start an empty domain run, with no Graph or external initialization."""
    return ChangeRunState(
        change_id=request.change_id,
        run_id=run_id,
        revision=request.revision,
        request_hash=request.request_hash,
        change_request=request,
        request_history=[request],
    )
