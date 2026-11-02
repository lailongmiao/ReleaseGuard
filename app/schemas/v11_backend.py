"""Explicit readonly review contracts; model output never supplies authority."""

import re
import json
from datetime import datetime
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from app.schemas.local_mvp import LocalCommand, LocalPrepare, LocalRevision, LocalView
from app.schemas.v1_backend import Chunk
from app.schemas.approval import ApprovalCommand
from app.schemas.releaseguard import ChangeRequestInput, Identifier, RiskLevel, Sha256

CONTRACT = "releaseguard.review.v1.1"
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
StrictVersion = Annotated[int, Field(strict=True, ge=0)]
Kind = Literal["pr", "commit", "branch_compare"]
Completeness = Literal["complete", "partial", "unknown"]


class ReviewModel(BaseModel):
    """Reject extra input and validate defaults at every boundary."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class PRSelector(ReviewModel):
    """Public PR identity in the approved base repository."""

    kind: Literal["pr"] = "pr"
    pr_number: Annotated[int, Field(strict=True, ge=1)]


class CommitSelector(ReviewModel):
    """Only complete SHA values; merge commits need an explicit actual parent."""

    kind: Literal["commit"] = "commit"
    commit_sha: CommitSha
    parent_sha: CommitSha | None = None


class BranchSelector(ReviewModel):
    """Exact branch names, never arbitrary git revision expressions."""

    kind: Literal["branch_compare"] = "branch_compare"
    base_branch: Annotated[str, Field(min_length=1, max_length=255)]
    head_branch: Annotated[str, Field(min_length=1, max_length=255)]
    comparison: Literal["merge_base"] = "merge_base"

    @field_validator("base_branch", "head_branch")
    @classmethod
    def branch_name(cls, value: str) -> str:
        """Apply git ref safety without normalizing branch identity."""
        if (
            value == "@"
            or value.startswith(("-", "/", "refs/"))
            or value.endswith(("/", "."))
            or any(x in value for x in ("..", "@{", "//"))
            or re.search(r"[\x00-\x20\x7f~^:?*\[\\]", value)
            or any(p.startswith(".") or p.endswith(".lock") for p in value.split("/"))
        ):
            raise ValueError("branch_name_invalid")
        return value


Selector = Annotated[PRSelector | CommitSelector | BranchSelector, Field(discriminator="kind")]


class ReviewBinding(ReviewModel):
    """Server allowlist identity and a typed immutable selector."""

    contract_version: Literal["releaseguard.review.v1.1"] = CONTRACT
    repository_id: Literal["checkout", "setup-python"]
    selector: Selector


class ReviewCreate(ReviewModel):
    """Creation retains the existing business request and independent reviewer."""

    idempotency_key: Identifier
    change: ChangeRequestInput
    approver_user_id: Annotated[int, Field(strict=True, ge=1)]
    binding: ReviewBinding


class SyncInput(ReviewModel):
    """CAS applies to object index state, not domain state."""

    command_id: Identifier
    expected_object_version: StrictVersion


class ReviewAnalyze(SyncInput):
    """A paid intent fixes domain and source versions before sending."""

    expected_state_version: StrictVersion
    snapshot_id: Identifier
    generation_id: Identifier
    query: Annotated[str, Field(min_length=1, max_length=2000)]


class ReviewSearch(SyncInput):
    """Object-scoped retrieval is separately idempotent and budgeted."""

    snapshot_id: Identifier
    generation_id: Identifier
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    limit: Annotated[int, Field(strict=True, ge=1, le=5)] = 5


class ReviewRevision(LocalRevision):
    """Explicit revision selects a current immutable snapshot without replacing history."""

    expected_object_version: StrictVersion
    snapshot_id: Identifier
    generation_id: Identifier


class ReviewPrepare(LocalPrepare):
    """Approval preparation additionally fixes the review object version."""

    expected_object_version: StrictVersion


class ReviewCommand(LocalCommand):
    """Domain command retaining current object CAS."""

    expected_object_version: StrictVersion


class ReviewApproval(ApprovalCommand):
    """The original approval intent plus explicit review object CAS."""

    expected_object_version: StrictVersion


class Comparison(ReviewModel):
    """Resolved immutable comparison facts, never supplied as authority by callers."""

    kind: Kind
    base_kind: Literal["commit", "empty_tree"] = "commit"
    base_tip_sha: CommitSha | None = None
    head_sha: CommitSha
    diff_base_sha: CommitSha | None = None
    parent_sha: CommitSha | None = None
    merge_base_sha: CommitSha | None = None
    comparison: Literal["merge_base", "parent", "empty_tree"]

    @model_validator(mode="after")
    def consistent(self) -> "Comparison":
        """Disallow mixed endpoint, parent and logical empty-tree identities."""
        if self.kind == "commit":
            if self.merge_base_sha is not None:
                raise ValueError("comparison_identity_invalid")
            if self.parent_sha is None:
                if (
                    self.base_kind != "empty_tree"
                    or self.comparison != "empty_tree"
                    or self.diff_base_sha
                    or self.base_tip_sha
                ):
                    raise ValueError("comparison_identity_invalid")
            elif (
                self.comparison != "parent"
                or self.base_kind != "commit"
                or self.diff_base_sha != self.parent_sha
                or self.base_tip_sha != self.parent_sha
            ):
                raise ValueError("comparison_identity_invalid")
        elif (
            self.comparison != "merge_base"
            or self.base_kind != "commit"
            or not self.base_tip_sha
            or not self.merge_base_sha
            or self.diff_base_sha != self.merge_base_sha
            or self.parent_sha
        ):
            raise ValueError("comparison_identity_invalid")
        return self


class DiffLine(ReviewModel):
    """One exact decoded patch line with independently tracked old/new coordinates."""

    kind: Literal["add", "delete", "context"]
    text: str
    old_line: Annotated[int, Field(ge=1)] | None = None
    new_line: Annotated[int, Field(ge=1)] | None = None
    no_final_newline: bool = False


class DiffHunk(ReviewModel):
    """A validated hunk or a disclosed bounded prefix of that hunk."""

    hunk_id: Identifier
    header: str
    lines: list[DiffLine]
    complete: bool = True


class DiffFile(ReviewModel):
    """File metadata never claims that unavailable file contents were read."""

    file_id: Identifier
    path: str
    old_path: str | None = None
    status: str
    additions: Annotated[int, Field(ge=0)] = 0
    deletions: Annotated[int, Field(ge=0)] = 0
    patch: str = ""
    hunks: list[DiffHunk] = []
    content_hash: Sha256
    complete: bool = False
    reasons: list[str] = []


class Coverage(ReviewModel):
    """Collection and model coverage are different measured quantities."""

    manifest_completeness: Completeness
    content_completeness: Completeness
    analysis_coverage: Completeness = "unknown"
    total_files: int | None = None
    observed_files: int = 0
    read_files: int = 0
    read_changed_lines: int = 0
    model_changed_lines: int = 0
    model_file_ids: list[Identifier] = []
    reasons: list[str] = []


class CheckFact(ReviewModel):
    """Sanitized current-head check observation."""

    name: Annotated[str, Field(max_length=128)]
    app_id: int
    check_id: int
    status: str
    conclusion: str | None = None
    head_sha: CommitSha


class CIObservation(ReviewModel):
    """Missing CI and failed collection cannot be confused."""

    head_sha: CommitSha
    status: Literal["success", "failed", "pending", "missing", "unavailable", "conflicting"]
    complete: bool
    checks: list[CheckFact] = []
    checks_passed: bool = False
    reasons: list[str] = []
    captured_at: datetime


class CollectedReview(ReviewModel):
    """A fully identity-checked collection, possibly with disclosed content gaps."""

    repository_id: str
    github_id: int
    owner_id: int
    repository_name: str
    comparison: Comparison
    files: list[DiffFile]
    coverage: Coverage
    ci: CIObservation
    captured_at: datetime
    policy_version: str = "github-review-get.v1.1"


class ReviewCitation(ReviewModel):
    """Only server-validated source positions can be supplied to the model."""

    citation_id: Identifier
    source_role: Literal[
        "application_policy", "repository_policy", "usage_reference", "change_observation", "ci_observation"
    ]
    kind: Literal["diff", "rule", "file_metadata", "ci"]
    snapshot_id: Identifier
    generation_id: Identifier
    file_id: Identifier | None = None
    path: str | None = None
    old_path: str | None = None
    side: Literal["LEFT", "RIGHT"] | None = None
    sha: CommitSha | None = None
    hunk_id: Identifier | None = None
    start_line: int | None = None
    end_line: int | None = None
    text: str
    text_hash: Sha256
    locator: str

    @model_validator(mode="after")
    def exact_position(self) -> "ReviewCitation":
        """Reject incomplete or impossible diff coordinates."""
        if (
            sha256(json.dumps(self.text, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            != self.text_hash
        ):
            raise ValueError("citation_hash_invalid")
        if self.kind == "diff" and (
            not all((self.file_id, self.path, self.side, self.sha, self.hunk_id))
            or self.start_line is None
            or self.end_line is None
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ValueError("diff_citation_position_invalid")
        return self


class ReviewClaim(ReviewModel):
    """A bounded statement with explicit epistemic role and cited premises."""

    text: Annotated[str, Field(min_length=1, max_length=1000)]
    kind: Literal["observation", "rule", "inference"]
    citation_ids: Annotated[list[Identifier], Field(min_length=1, max_length=5)]


class ReviewRisk(ReviewClaim):
    """Risk is a model suggestion, not a deterministic authorization decision."""

    severity: RiskLevel


class ValidationSuggestion(ReviewModel):
    """Suggested validation is never represented as performed."""

    text: Annotated[str, Field(min_length=1, max_length=1000)]
    performed: Literal[False] = False


class ReviewOutput(ReviewModel):
    """Strict model-only output, excluding identity, coverage and permissions."""

    summary: Annotated[str, Field(min_length=1, max_length=4000)]
    observations: Annotated[list[ReviewClaim], Field(max_length=16)]
    risks: Annotated[list[ReviewRisk], Field(max_length=12)]
    validation_suggestions: Annotated[list[ValidationSuggestion], Field(max_length=8)]
    unknowns: Annotated[list[Annotated[str, Field(min_length=1, max_length=1000)]], Field(max_length=12)]
    candidate_steps: Annotated[list[Literal["validate_inputs", "simulate", "verify_predictions"]], Field(max_length=3)]
    model_risk: RiskLevel

    @model_validator(mode="after")
    def uncertainty(self) -> "ReviewOutput":
        """Unknowns cannot create an actionable model risk recommendation."""
        if self.unknowns and self.model_risk is not RiskLevel.UNKNOWN:
            raise ValueError("analysis_uncertainty_requires_unknown")
        if not self.observations and not self.unknowns:
            raise ValueError("analysis_requires_claims_or_unknowns")
        return self


class Eligibility(ReviewModel):
    """Deterministic qualification separate from report completion."""

    eligible: bool
    reason_codes: list[str]
    evaluated_at: datetime
    policy_version: str = "review-eligibility.v1.1"


class ReviewResult(ReviewModel):
    """Persisted command outcome; HTTP success alone proves no business success."""

    status: Literal["completed", "blocked", "failed", "unknown"]
    reason: str | None = None
    object_id: str | None = None
    snapshot_id: str | None = None
    generation_id: str | None = None
    analysis_id: str | None = None
    unchanged: bool = False
    observation_id: str | None = None
    object_version: int | None = None
    result: LocalView | None = None
    items: list[Chunk] | None = None
    citations: list[ReviewCitation] | None = None


class ReviewPage(ReviewModel):
    """Authorized bounded metadata pages without global counts."""

    items: list[dict[str, Any]]
    next_cursor: str | None = None
    consistency: Literal["live", "upstream_live"] = "live"


class ReviewCapabilities(ReviewModel):
    """Live phase policy, never a saved report or an authorization seal."""

    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True)

    version: Literal["review-capabilities.rehearsal.2"] = "review-capabilities.rehearsal.2"
    phase: Literal["backend_functional"] = "backend_functional"
    ai_explanation: Literal["experimental"] = "experimental"
    diff_verification_required: Literal[True] = True
    semantic_acceptance: Literal["deferred"] = "deferred"
    prepare: Literal["available_subject_to_authorization"] = "available_subject_to_authorization"
    approval: Literal["available_subject_to_authorization"] = "available_subject_to_authorization"
    run: Literal["available_subject_to_authorization"] = "available_subject_to_authorization"
    pause_reason: str | None = None
    revoke: Literal["available_subject_to_authorization"] = "available_subject_to_authorization"
    history: Literal["available_subject_to_authorization"] = "available_subject_to_authorization"


class ReviewView(ReviewModel):
    """Saved review with explicit domain and source version boundaries."""

    object_id: str
    object_version: int
    owner_user_id: int | None = None
    approver_user_id: int | None = None
    approval_request: dict[str, Any] | None = None
    binding: ReviewBinding
    state: dict[str, Any]
    active_analysis_id: str | None = None
    active_snapshot_id: str | None
    active_generation_id: str | None
    collection_status: str
    analysis_status: str
    freshness: str
    stale_reasons: list[str]
    coverage: Coverage | None = None
    rehearsal_eligibility: Eligibility
    execution_mode: Literal["offline_simulation"] = "offline_simulation"
    production_execution: Literal["disabled"] = "disabled"
    capabilities: ReviewCapabilities | None = None


class ReviewAnalysis(ReviewModel):
    """Immutable report body accompanied by live authorization and freshness."""

    analysis_id: str
    object_id: str
    revision: int
    snapshot_id: str
    generation_id: str
    status: str
    resolved_comparison: Comparison | None = None
    output: ReviewOutput | None = None
    coverage: Coverage | None = None
    citations: list[ReviewCitation] = []
    dependencies: dict[str, Any] = {}
    telemetry: dict[str, Any] = {}
    stale_reasons: list[str] = []
    rehearsal_eligibility: Eligibility
    ci_observation_id: str | None = None
    provenance: str | None = None
    evaluation_only: bool = False
    reason: str | None = None
    capabilities: ReviewCapabilities | None = None


class ReviewError(ReviewModel):
    """Stable safe failure details never include source or Provider envelopes."""

    code: str
    request_id: str
    retryable: bool = False
    details: dict[str, Any] = {}
