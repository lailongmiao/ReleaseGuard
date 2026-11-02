"""Read-only Evidence source adapter contracts and offline implementations."""

from datetime import UTC, datetime
from typing import Protocol, override

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCollectionStatus,
    EvidenceIdentifier,
    EvidenceSourceType,
    EvidenceSourceResult,
)


class EvidenceSourceRequest(BaseModel):
    """Non-sensitive, bounded source query context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: EvidenceIdentifier | None = None
    tenant_id: EvidenceIdentifier | None = None
    roles: frozenset[EvidenceIdentifier] = frozenset()
    change_id: EvidenceIdentifier
    run_id: EvidenceIdentifier
    revision: int = Field(strict=True, ge=1)
    requirement_id: EvidenceIdentifier
    query_hash: str = Field(pattern=r"^(?:sha256:)?[0-9a-fA-F]{64}$")
    limit: int = Field(default=20, strict=True, ge=1, le=100)
    timeout_ms: int = Field(default=5000, strict=True, ge=1, le=60000)


class EvidenceAccessContext(BaseModel):
    """Principal and scope used while consuming stored Evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: EvidenceIdentifier | None = None
    tenant_id: EvidenceIdentifier | None = None
    roles: frozenset[EvidenceIdentifier] = frozenset()


class EvidenceSourceAdapter(Protocol):
    """Read-only source boundary; implementations cannot mutate source systems."""

    source_id: EvidenceIdentifier

    async def collect(self, request: EvidenceSourceRequest) -> EvidenceSourceResult:
        """Collect bounded source snapshots under an explicit access context."""
        ...


def scope_allows(scope: EvidenceAccessScope, request: EvidenceSourceRequest | EvidenceAccessContext) -> bool:
    """Apply access scope before any ordering, deduplication, or truncation."""
    if scope.public:
        return True
    if scope.tenant_id is not None and scope.tenant_id == request.tenant_id:
        return True
    if request.principal_id is not None and request.principal_id in scope.principals:
        return True
    return bool(scope.roles.intersection(request.roles))


class InMemoryEvidenceSourceAdapter(EvidenceSourceAdapter):
    """Deterministic fixture adapter with explicit read-only source outcomes."""

    def __init__(self, result: EvidenceSourceResult) -> None:
        """Copy a prevalidated source result without retaining caller mutation."""
        self._result = EvidenceSourceResult.model_validate_json(result.model_dump_json())
        self.source_id = self._result.source_id

    @override
    async def collect(self, request: EvidenceSourceRequest) -> EvidenceSourceResult:
        """Filter authorized items before stable ordering and bounded output."""
        result = self._result
        if result.collection_status is not EvidenceCollectionStatus.VERIFIED:
            return result.model_copy(deep=True)
        if any(item.source_type is not result.source_type for item in result.items):
            return _unavailable(result, "source_type_mismatch")
        authorized = [item for item in result.items if scope_allows(item.access_scope, request)]
        if len(authorized) != len(result.items):
            return EvidenceSourceResult(
                source_id=result.source_id,
                source_type=result.source_type,
                requirement_id=result.requirement_id,
                collection_status=EvidenceCollectionStatus.PERMISSION_DENIED,
                captured_at=result.captured_at,
                audit_reference=result.audit_reference,
                degradation_reason="source_access_denied",
            )
        ordered = sorted(authorized, key=lambda item: (item.evidence_id, item.source_version or "", item.source_uri))
        return result.model_copy(update={"items": ordered[: request.limit]}, deep=True)


class FailingEvidenceSourceAdapter(EvidenceSourceAdapter):
    """Fixture adapter for timeout and provider failure degradation tests."""

    source_id = "failing-source"

    def __init__(self, *, timeout: bool = False) -> None:
        """Configure a deterministic failure without contacting a provider."""
        self._timeout = timeout

    @override
    async def collect(self, request: EvidenceSourceRequest) -> EvidenceSourceResult:
        """Raise only an internal exception; callers receive a redacted result."""
        del request
        if self._timeout:
            raise TimeoutError
        raise RuntimeError("source fixture failure")


async def collect_source(
    adapter: EvidenceSourceAdapter,
    request: EvidenceSourceRequest,
    *,
    captured_at: datetime,
) -> EvidenceSourceResult:
    """Normalize adapter failures without exposing provider or query details."""
    if captured_at.tzinfo is None:
        raise ValueError("timestamp must include timezone information")
    try:
        result = await adapter.collect(request)
    except TimeoutError:
        return EvidenceSourceResult(
            source_id=adapter.source_id,
            source_type=EvidenceSourceType.DOCUMENT,
            requirement_id=request.requirement_id,
            collection_status=EvidenceCollectionStatus.UNAVAILABLE,
            captured_at=captured_at.astimezone(UTC),
            audit_reference="source-timeout",
            degradation_reason="source_timeout",
        )
    except Exception:
        return EvidenceSourceResult(
            source_id=adapter.source_id,
            source_type=EvidenceSourceType.DOCUMENT,
            requirement_id=request.requirement_id,
            collection_status=EvidenceCollectionStatus.UNAVAILABLE,
            captured_at=captured_at.astimezone(UTC),
            audit_reference="source-unavailable",
            degradation_reason="source_unavailable",
        )
    return EvidenceSourceResult.model_validate_json(result.model_dump_json())


def _unavailable(result: EvidenceSourceResult, reason: str) -> EvidenceSourceResult:
    """Convert malformed fixture bindings to a fail-closed source state."""
    return EvidenceSourceResult(
        source_id=result.source_id,
        source_type=result.source_type,
        requirement_id=result.requirement_id,
        collection_status=EvidenceCollectionStatus.UNAVAILABLE,
        captured_at=result.captured_at,
        audit_reference=result.audit_reference,
        degradation_reason=reason,
    )
