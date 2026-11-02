"""Evidence and read-only retrieval contracts for the runtime boundary.

These models deliberately describe evidence as a separate concern from Mem0
memory and LangGraph checkpoints.  They do not model a ReleaseGuard change,
approval, execution, or receipt.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from app.services.releaseguard.normalization import canonical_hash


EvidenceIdentifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
SourceUri = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
ContentHash = Annotated[str, StringConstraints(pattern=r"^(?:sha256:)?[0-9a-fA-F]{64}$")]
PolicyIdentifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class EvidenceSourceType(StrEnum):
    """Controlled source categories for read-only evidence."""

    CODE_REVIEW = "code_review"
    CI = "ci"
    RELEASE_RECORD = "release_record"
    DOCUMENT = "document"
    HISTORY = "history"
    INCIDENT = "incident"
    RUNTIME_READ = "runtime_read"


class EvidenceCollectionStatus(StrEnum):
    """Collection outcome that must not be rewritten by an LLM."""

    VERIFIED = "verified"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    CONFLICTING = "conflicting"


class RetrievalStatus(StrEnum):
    """Outcome of a read-only retrieval attempt."""

    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SOURCE_UNAVAILABLE = "source_unavailable"


class EvidenceAccessScope(BaseModel):
    """Explicit access scope used before retrieval ranking or truncation.

    A scope can be public, tenant-scoped, principal-scoped, or role-scoped.
    At least one restriction is required for non-public evidence so an item
    cannot accidentally become globally visible through an empty scope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    public: bool = Field(default=False, description="Whether the evidence is readable by every caller")
    tenant_id: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = Field(
        default=None, description="Tenant allowed to read this evidence"
    )
    principals: frozenset[Annotated[str, StringConstraints(min_length=1, max_length=128)]] = Field(
        default_factory=frozenset, description="User or service principals allowed to read this evidence"
    )
    roles: frozenset[Annotated[str, StringConstraints(min_length=1, max_length=64)]] = Field(
        default_factory=frozenset, description="Roles allowed to read this evidence"
    )

    @field_serializer("principals", "roles", when_used="json")
    def ordered_scope(self, value: frozenset[str]) -> list[str]:
        """Keep JSON facts stable across process hash seeds; retain set semantics."""
        return sorted(value)

    @model_validator(mode="after")
    def validate_scope(self) -> "EvidenceAccessScope":
        """Require an explicit restriction for non-public evidence."""
        if not self.public and self.tenant_id is None and not self.principals and not self.roles:
            raise ValueError("non-public evidence requires tenant_id, principals, or roles")
        if self.public and (self.tenant_id is not None or self.principals or self.roles):
            raise ValueError("public evidence cannot include restricted scope values")
        return self


class EvidenceItem(BaseModel):
    """A source-backed, immutable-at-read-time evidence item.

    ``snippet`` is bounded context for a caller.  It is intentionally not a
    logging field; callers should use :meth:`redacted_log_context` instead.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_id: EvidenceIdentifier = Field(..., description="Stable evidence identifier")
    source_type: EvidenceSourceType = Field(..., description="Controlled category of the source")
    source_uri: SourceUri = Field(..., description="Auditable source URI or controlled fixture URI")
    source_version: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = Field(
        default=None, description="Source revision, commit, or document version"
    )
    captured_at: datetime = Field(..., description="Timezone-aware capture time")
    content_hash: ContentHash = Field(..., description="SHA-256 content hash, optionally prefixed with sha256:")
    access_scope: EvidenceAccessScope = Field(..., description="Scope checked before ranking and truncation")
    snippet: Annotated[str, StringConstraints(min_length=1, max_length=2000)] = Field(
        ..., description="Bounded source excerpt for context and citation"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Deterministic source/retrieval confidence in [0, 1]")
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, description="Non-sensitive retrieval metadata; raw source content is excluded"
    )
    freshness_until: datetime | None = Field(default=None, description="Optional expiry for freshness checks")
    collection_status: EvidenceCollectionStatus = Field(
        default=EvidenceCollectionStatus.VERIFIED,
        description="Source collection outcome; unavailable states are never promoted to verified",
    )

    @field_validator("captured_at", "freshness_until")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Reject naive timestamps so citations have an unambiguous time basis."""
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamp must include timezone information")
        return value

    @field_validator("source_uri", "snippet")
    @classmethod
    def reject_control_values(cls, value: str) -> str:
        """Reject null bytes that could corrupt storage or downstream logs."""
        if "\0" in value:
            raise ValueError("value contains null bytes")
        return value

    @model_validator(mode="after")
    def validate_freshness(self) -> "EvidenceItem":
        """Ensure a freshness deadline cannot precede capture time."""
        if self.freshness_until is not None and self.freshness_until < self.captured_at:
            raise ValueError("freshness_until must be later than captured_at")
        return self

    def redacted_log_context(self) -> dict[str, str]:
        """Return safe identifiers for logs without source text or access data."""
        return {
            "evidence_id": self.evidence_id,
            "content_hash": self.content_hash,
            "source_type": self.source_type.value,
            "collection_status": self.collection_status.value,
        }


class EvidenceSnapshot(BaseModel):
    """Immutable stored reference to one Phase 1 evidence item version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: ContentHash
    item: EvidenceItem
    persisted_at: datetime

    @field_validator("persisted_at")
    @classmethod
    def require_persisted_timezone(cls, value: datetime) -> datetime:
        """Keep storage timestamps comparable across repositories."""
        if value.tzinfo is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> "EvidenceSnapshot":
        """Bind a snapshot id to the source identity and immutable content hash."""
        expected = canonical_hash(
            {
                "evidence_id": self.item.evidence_id,
                "source_uri": self.item.source_uri,
                "source_version": self.item.source_version,
                "content_hash": self.item.content_hash,
            }
        )
        if self.snapshot_id.removeprefix("sha256:").lower() != expected:
            raise ValueError("snapshot_id_mismatch")
        return self


class EvidenceCitation(BaseModel):
    """Auditable pointer from a required item to an immutable snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: EvidenceIdentifier
    snapshot_id: ContentHash
    evidence_id: EvidenceIdentifier
    source_version: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    content_hash: ContentHash
    captured_at: datetime
    required_item: EvidenceIdentifier
    change_id: EvidenceIdentifier
    run_id: EvidenceIdentifier
    revision: Annotated[int, Field(strict=True, ge=1)]
    request_hash: ContentHash
    thread_id: EvidenceIdentifier
    policy_version: PolicyIdentifier
    cited_at: datetime
    locator: Annotated[str | None, StringConstraints(min_length=1, max_length=512)] = None

    @field_validator("captured_at", "cited_at")
    @classmethod
    def require_citation_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous citation timestamps."""
        if value.tzinfo is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(UTC)


class EvidenceSourceResult(BaseModel):
    """Bounded read-only source collection result shared by adapters and Gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: EvidenceIdentifier
    source_type: EvidenceSourceType
    requirement_id: EvidenceIdentifier
    collection_status: EvidenceCollectionStatus
    items: list[EvidenceItem] = Field(default_factory=list, max_length=100)
    source_version: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = None
    captured_at: datetime
    audit_reference: EvidenceIdentifier
    degradation_reason: Annotated[str | None, StringConstraints(min_length=1, max_length=256)] = None

    @field_validator("captured_at")
    @classmethod
    def require_source_timezone(cls, value: datetime) -> datetime:
        """Normalize source capture time to UTC."""
        if value.tzinfo is None:
            raise ValueError("timestamp must include timezone information")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_source_state(self) -> "EvidenceSourceResult":
        """Prevent failed sources from carrying trusted snapshots."""
        if self.collection_status is not EvidenceCollectionStatus.VERIFIED and self.items:
            raise ValueError("untrusted_source_cannot_return_items")
        if self.collection_status is not EvidenceCollectionStatus.VERIFIED and self.degradation_reason is None:
            raise ValueError("untrusted_source_requires_degradation_reason")
        if self.collection_status is EvidenceCollectionStatus.VERIFIED and not self.items:
            raise ValueError("verified_source_requires_items")
        if any(item.source_type is not self.source_type for item in self.items):
            raise ValueError("source_item_type_mismatch")
        return self


class EvidenceRequirement(BaseModel):
    """One deterministic required evidence category."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: EvidenceIdentifier
    source_types: frozenset[EvidenceSourceType] = frozenset()
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidencePolicy(BaseModel):
    """Versioned required evidence policy consumed without an LLM or clock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: PolicyIdentifier
    requirements: list[EvidenceRequirement] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_unique_requirements(self) -> "EvidencePolicy":
        """Reject ambiguous policy keys before deterministic evaluation."""
        keys = [item.requirement_id for item in self.requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate_requirement_id")
        return self


class RetrievalMetadata(BaseModel):
    """Provider-neutral metadata for a retrieval result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: Annotated[str, StringConstraints(min_length=1, max_length=128)] = Field(...)
    model: Annotated[str | None, StringConstraints(min_length=1, max_length=128)] = Field(default=None)
    strategy: Annotated[str, StringConstraints(min_length=1, max_length=64)] = Field(...)
    query_hash: ContentHash = Field(..., description="Hash of the normalized query; raw query is excluded")
    limit: PositiveInt = Field(..., le=100)
    result_count: NonNegativeInt = Field(default=0)
    deduplicated_count: NonNegativeInt = Field(default=0)
    latency_ms: NonNegativeFloat | None = Field(default=None)


class RetrievalResult(BaseModel):
    """Structured result for a read-only evidence lookup.

    Empty and unavailable outcomes are valid results so chat callers can
    continue with an explicit degradation state instead of fabricated evidence.
    """

    model_config = ConfigDict(extra="forbid")

    status: RetrievalStatus = Field(...)
    items: list[EvidenceItem] = Field(default_factory=list, max_length=100)
    metadata: RetrievalMetadata = Field(...)
    degradation_reason: Annotated[str | None, StringConstraints(min_length=1, max_length=256)] = Field(default=None)

    @model_validator(mode="after")
    def validate_result_state(self) -> "RetrievalResult":
        """Keep status and item presence consistent at the contract boundary."""
        if self.status is RetrievalStatus.SUCCEEDED and not self.items:
            raise ValueError("succeeded retrieval must contain at least one item")
        if self.status is RetrievalStatus.EMPTY and self.items:
            raise ValueError("empty retrieval cannot contain items")
        if self.status in {RetrievalStatus.PROVIDER_UNAVAILABLE, RetrievalStatus.SOURCE_UNAVAILABLE}:
            if self.items:
                raise ValueError("unavailable retrieval cannot contain items")
            if self.degradation_reason is None:
                raise ValueError("unavailable retrieval requires degradation_reason")
        if self.metadata.result_count != len(self.items):
            raise ValueError("metadata.result_count must match items")
        return self


__all__ = [
    "ContentHash",
    "EvidenceAccessScope",
    "EvidenceCollectionStatus",
    "EvidenceIdentifier",
    "EvidenceItem",
    "EvidenceCitation",
    "EvidenceSnapshot",
    "EvidenceSourceResult",
    "EvidenceRequirement",
    "EvidencePolicy",
    "EvidenceSourceType",
    "RetrievalMetadata",
    "RetrievalResult",
    "RetrievalStatus",
    "SourceUri",
]
