"""Read-only, provider-neutral Evidence Store boundary.

The fixture implementation is deliberately independent from Mem0.  It is
safe for tests and local demonstrations and has no write or external-tool
surface.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from app.core.logging import logger
from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceItem,
    RetrievalMetadata,
    RetrievalResult,
    RetrievalStatus,
)


class EvidenceAccessContext:
    """Caller identity used for access filtering before ranking."""

    def __init__(
        self,
        *,
        principal_id: str | None = None,
        tenant_id: str | None = None,
        roles: Iterable[str] = (),
    ) -> None:
        """Create a minimal access context without retaining query text."""
        self.principal_id = principal_id
        self.tenant_id = tenant_id
        self.roles = frozenset(roles)


class EvidenceStore(Protocol):
    """Read-only interface required by retrieval callers."""

    async def search(
        self,
        query: str,
        *,
        access: EvidenceAccessContext,
        limit: int = 5,
    ) -> RetrievalResult:
        """Search accessible evidence and return a structured result."""
        ...


class InMemoryEvidenceStore:
    """Deterministic fixture-backed implementation of :class:`EvidenceStore`."""

    def __init__(
        self,
        items: Sequence[EvidenceItem] = (),
        *,
        provider: str = "fixture",
        model: str | None = None,
        available: bool = True,
        unavailable_reason: str = "evidence_store_unavailable",
        snippet_max_chars: int = 600,
    ) -> None:
        """Initialize immutable read-only data and retrieval settings."""
        if not provider.strip():
            raise ValueError("provider must not be empty")
        if snippet_max_chars <= 0:
            raise ValueError("snippet_max_chars must be positive")
        self._items = tuple(items)
        self._provider = provider
        self._model = model
        self._available = available
        self._unavailable_reason = unavailable_reason
        self._snippet_max_chars = snippet_max_chars

    async def search(
        self,
        query: str,
        *,
        access: EvidenceAccessContext,
        limit: int = 5,
    ) -> RetrievalResult:
        """Filter, deduplicate, rank, and truncate accessible fixture evidence."""
        started = datetime.now(UTC)
        bounded_limit = _bounded_limit(limit)
        normalized_query = _normalize_query(query)
        query_hash = _hash_query(normalized_query)
        if not self._available:
            return self._result(
                RetrievalStatus.PROVIDER_UNAVAILABLE,
                query_hash=query_hash,
                limit=bounded_limit,
                started=started,
                degradation_reason=self._unavailable_reason,
            )
        if not normalized_query:
            return self._result(
                RetrievalStatus.EMPTY,
                query_hash=query_hash,
                limit=bounded_limit,
                started=started,
            )

        # Access filtering is intentionally the first data operation.  A
        # caller can never influence ranking of an item it cannot read.
        accessible = [item for item in self._items if _is_accessible(item.access_scope, access)]
        stable_candidates = sorted(accessible, key=lambda item: (item.evidence_id, item.source_uri))
        unique: list[EvidenceItem] = []
        seen_hashes: set[str] = set()
        for item in stable_candidates:
            if item.content_hash in seen_hashes:
                continue
            seen_hashes.add(item.content_hash)
            unique.append(item)

        terms = tuple(normalized_query.split())
        ranked = sorted(
            unique,
            key=lambda item: (-_score(item, terms), item.evidence_id, item.source_uri),
        )
        matches = [item for item in ranked if _score(item, terms) > 0][:bounded_limit]
        if not matches:
            return self._result(
                RetrievalStatus.EMPTY,
                query_hash=query_hash,
                limit=bounded_limit,
                started=started,
                deduplicated_count=len(accessible) - len(unique),
            )

        result_items = [self._citation_item(item, _score(item, terms)) for item in matches]
        return self._result(
            RetrievalStatus.SUCCEEDED,
            query_hash=query_hash,
            limit=bounded_limit,
            started=started,
            items=result_items,
            deduplicated_count=len(accessible) - len(unique),
        )

    def _citation_item(self, item: EvidenceItem, score: float) -> EvidenceItem:
        """Return a bounded citation with deterministic confidence."""
        confidence = min(1.0, max(0.0, item.confidence * score))
        snippet = item.snippet[: self._snippet_max_chars]
        if len(item.snippet) > self._snippet_max_chars:
            # Keep the complete citation within the configured bound,
            # including the truncation marker.
            marker = "…"
            snippet = item.snippet[: max(0, self._snippet_max_chars - len(marker))].rstrip() + marker
        return item.model_copy(update={"snippet": snippet, "confidence": round(confidence, 6)})

    def _result(
        self,
        status: RetrievalStatus,
        *,
        query_hash: str,
        limit: int,
        started: datetime,
        items: Sequence[EvidenceItem] = (),
        deduplicated_count: int = 0,
        degradation_reason: str | None = None,
    ) -> RetrievalResult:
        """Build a validated result and log only safe retrieval identifiers."""
        result_items = list(items)
        metadata = RetrievalMetadata(
            provider=self._provider,
            model=self._model,
            strategy="fixture_lexical",
            query_hash=query_hash,
            limit=limit,
            result_count=len(result_items),
            deduplicated_count=max(0, deduplicated_count),
            latency_ms=round((datetime.now(UTC) - started).total_seconds() * 1000, 3),
        )
        result = RetrievalResult(
            status=status,
            items=result_items,
            metadata=metadata,
            degradation_reason=degradation_reason,
        )
        logger.debug(
            "evidence_retrieval_completed",
            status=status.value,
            result_count=len(result_items),
            provider=self._provider,
        )
        return result


def _bounded_limit(limit: int) -> int:
    """Validate and cap caller-controlled result limits."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be positive")
    return min(limit, 100)


def _normalize_query(query: str) -> str:
    """Normalize query text for matching and hashing."""
    return " ".join(query.strip().lower().split())


def _hash_query(query: str) -> str:
    """Hash query text so raw user content never enters retrieval metadata."""
    return f"sha256:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"


def _is_accessible(scope: EvidenceAccessScope, access: EvidenceAccessContext) -> bool:
    """Evaluate explicit public, tenant, principal, and role access."""
    if scope.public:
        return True
    if scope.tenant_id is not None and scope.tenant_id == access.tenant_id:
        return True
    if access.principal_id is not None and access.principal_id in scope.principals:
        return True
    return bool(scope.roles.intersection(access.roles))


def _score(item: EvidenceItem, terms: tuple[str, ...]) -> float:
    """Compute a deterministic lexical relevance score in [0, 1]."""
    haystack = f"{item.snippet.lower()} {' '.join(str(value).lower() for value in item.metadata.values())}"
    matches = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", haystack))
    return matches / len(terms) if terms else 0.0
