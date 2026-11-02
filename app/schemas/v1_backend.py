"""Bounded product-readonly contracts; no execution or identity authority."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.releaseguard import Identifier, RiskLevel, Sha256
from app.schemas.releaseguard import ChangeStatus
from app.schemas.evidence import EvidenceSnapshot, EvidenceCitation

SourceReason = Literal[
    "permission_denied",
    "not_found",
    "rate_limited",
    "truncated",
    "source_byte_limit",
    "head_changed",
    "check_sha_conflict",
    "check_context_conflict",
    "missing_checks",
    "source_timeout_or_unavailable",
    "source_invalid_response",
    "source_invalid_sha",
    "source_unavailable",
    "checks_failed_pending_or_policy_missing",
    "markdown_unavailable_or_unsafe",
]


class ProductBinding(BaseModel):
    """Only a server-allowlisted public PR and the reviewed rule source."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["product_readonly"] = "product_readonly"
    repository: Literal["actions/checkout"] = "actions/checkout"
    pr_number: Annotated[int, Field(strict=True, ge=1)]
    github_source: Literal["github-checkout"] = "github-checkout"
    markdown_source: Literal["markdown-rules"] = "markdown-rules"


class SourceCommand(BaseModel):
    """Idempotent local command; principal is resolved outside this body."""

    model_config = ConfigDict(extra="forbid")
    command_id: Identifier
    expected_version: Annotated[int, Field(strict=True, ge=0)]
    pr_number: Annotated[int | None, Field(strict=True, ge=1)] = None


class SourceGrant(SourceCommand):
    """Explicit source owner grant or revoke, not an automatic approver grant."""

    user_id: Annotated[int, Field(strict=True, ge=1)]


class SearchInput(BaseModel):
    """Resource and generation scope cannot be supplied by an LLM."""

    model_config = ConfigDict(extra="forbid")
    change_id: Identifier
    command_id: Identifier
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    limit: Annotated[int, Field(strict=True, ge=1, le=5)] = 5


class AnalyzeInput(BaseModel):
    """One bounded, persistent analysis command."""

    model_config = ConfigDict(extra="forbid")
    command_id: Identifier
    expected_state_version: Annotated[int, Field(strict=True, ge=0)]
    query: Annotated[str, Field(min_length=1, max_length=2000)]


class SourceDocument(BaseModel):
    """Sanitized immutable source data; excludes authors and provider envelopes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: Identifier
    source_id: Identifier
    resource: str
    version: str
    content: Annotated[str, Field(min_length=1, max_length=200000)]
    content_hash: Sha256
    locator: str
    captured_at: datetime


class Collection(BaseModel):
    """Collection status is distinct from the business check outcome."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["verified", "unavailable", "permission_denied", "conflicting"]
    reason: SourceReason | None = None
    documents: list[SourceDocument] = []
    head_sha: str | None = None
    checks_passed: bool = False
    check_names: tuple[str, ...] = ()


class Chunk(BaseModel):
    """Retrieval metadata extends source references, never replaces Evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: Identifier
    document: SourceDocument
    ordinal: int
    text: str
    text_hash: Sha256
    tokens: Annotated[int, Field(ge=1, le=512)]
    vector: list[float] = []
    generation: Identifier
    start_char: Annotated[int, Field(ge=0)] = 0
    end_char: Annotated[int, Field(ge=0)] = 0
    locator: str = ""


class Claim(BaseModel):
    """Claims require retrieved citation ids, without arbitrary tool calls."""

    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    citation_ids: Annotated[list[Identifier], Field(min_length=1, max_length=5)]


class AnalysisOutput(BaseModel):
    """Provider output is a suggestion and contains no approval authority."""

    model_config = ConfigDict(extra="forbid")
    explanation: Annotated[str, Field(min_length=1, max_length=4000)]
    claims: Annotated[list[Claim], Field(max_length=12)]
    risk: RiskLevel
    unknowns: Annotated[list[str], Field(max_length=12)]
    candidate_steps: Annotated[list[Literal["validate_inputs", "simulate", "verify_predictions"]], Field(max_length=3)]

    @model_validator(mode="after")
    def require_grounding(self) -> "AnalysisOutput":
        """Empty or uncertain answers cannot grant a usable plan."""
        if not self.claims and not self.unknowns:
            raise ValueError("analysis_requires_claims_or_unknowns")
        if self.unknowns and self.risk is not RiskLevel.UNKNOWN:
            raise ValueError("analysis_uncertainty_requires_unknown")
        return self


class SourceView(BaseModel):
    """Caller-visible source configuration; never a credential reference."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    owner_id: int
    version: int
    acl: list[int]
    generation: str | None
    required_checks: list[str]
    deleted: bool


class SourceList(BaseModel):
    """No global source count or identities outside the caller ACL."""

    items: list[SourceView]


class SourceResult(BaseModel):
    """Explicit synchronous command outcome and replayable diagnostics."""

    status: Literal["completed", "failed", "blocked"]
    reason: str | None = None
    generation: str | None = None
    chunks: int | None = None
    checks_passed: bool | None = None
    unchanged: bool = False
    analysis_id: str | None = None


class GrantResult(BaseModel):
    """An owner-authorized ACL version transition."""

    source_id: str
    version: int
    acl: list[int]


class SearchResult(BaseModel):
    """Scoped retrieval, with source CAS state and explicit failure status."""

    status: Literal["completed", "lexical_degraded", "failed"]
    items: list[Chunk]
    sources: dict[str, dict]
    reason: str | None = None


class AnalysisView(BaseModel):
    """Saved suggestions are distinct from authority and execution evidence."""

    status: Literal["not_applicable", "not_started", "started", "blocked", "failed", "unknown", "completed"]
    reason: str | None = None
    stale: bool = False
    analysis_id: str | None = None
    analysis: AnalysisOutput | None = None
    citations: list[Chunk] = []
    telemetry: dict | None = None
    binding_hash: str | None = None
    execution_mode: Literal["offline_simulation"] = "offline_simulation"


class ChangeSummary(BaseModel):
    """Object and source authorized metadata, without Evidence body or totals."""

    model_config = ConfigDict(extra="forbid")
    change_id: Identifier
    status: ChangeStatus
    revision: Annotated[int, Field(strict=True, ge=1)]


class ChangePage(BaseModel):
    """Pagination exposes only a cursor belonging to authorized objects."""

    model_config = ConfigDict(extra="forbid")
    items: list[ChangeSummary]
    next_cursor: Identifier | None = None


class EvidenceView(BaseModel):
    """Reuse the existing Evidence and Citation contracts in OpenAPI."""

    model_config = ConfigDict(extra="forbid")
    snapshots: list[EvidenceSnapshot]
    citations: list[EvidenceCitation]
