"""Read-only source adapter state and access boundary tests."""

import asyncio
from datetime import UTC, datetime
from hashlib import sha256

from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCollectionStatus,
    EvidenceItem,
    EvidenceSourceResult,
    EvidenceSourceType,
)
from app.services.evidence_sources import (
    EvidenceSourceRequest,
    FailingEvidenceSourceAdapter,
    InMemoryEvidenceSourceAdapter,
    collect_source,
)


def source_item(evidence_id: str, scope: EvidenceAccessScope) -> EvidenceItem:
    """Build a bounded fixture item with the requested access scope."""
    snippet = f"verified {evidence_id}"
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.CI,
        source_uri=f"fixture://{evidence_id}",
        source_version="build-1",
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=f"sha256:{sha256(snippet.encode()).hexdigest()}",
        access_scope=scope,
        snippet=snippet,
        confidence=1.0,
    )


def request(**overrides: object) -> EvidenceSourceRequest:
    """Build a stable source query without raw user text."""
    return EvidenceSourceRequest(
        change_id="change-1",
        run_id="run-1",
        revision=1,
        requirement_id="ci_success",
        query_hash="sha256:" + "a" * 64,
        **overrides,
    )


def result(
    *items: EvidenceItem, status: EvidenceCollectionStatus = EvidenceCollectionStatus.VERIFIED
) -> EvidenceSourceResult:
    """Build a source result with safe audit metadata."""
    return EvidenceSourceResult(
        source_id="ci-fixture",
        source_type=EvidenceSourceType.CI,
        requirement_id="ci_success",
        collection_status=status,
        items=list(items),
        captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        audit_reference="audit-1",
        degradation_reason=None if status is EvidenceCollectionStatus.VERIFIED else "synthetic_failure",
    )


def test_access_is_filtered_before_ordering_and_limit() -> None:
    """A forbidden item cannot affect returned ordering or result count."""

    async def scenario() -> None:
        adapter = InMemoryEvidenceSourceAdapter(
            result(
                source_item("z-public", EvidenceAccessScope(public=True)),
                source_item("a-private", EvidenceAccessScope(tenant_id="other-tenant")),
            )
        )
        denied = await adapter.collect(request())
        assert denied.collection_status is EvidenceCollectionStatus.PERMISSION_DENIED
        assert not denied.items

        public = InMemoryEvidenceSourceAdapter(result(source_item("z-public", EvidenceAccessScope(public=True))))
        allowed = await public.collect(request(limit=1))
        assert [item.evidence_id for item in allowed.items] == ["z-public"]

    asyncio.run(scenario())


def test_source_states_are_preserved_without_fake_items() -> None:
    """Unavailable and conflicting outcomes remain distinct and empty."""

    async def scenario() -> None:
        for status in (EvidenceCollectionStatus.UNAVAILABLE, EvidenceCollectionStatus.CONFLICTING):
            adapter = InMemoryEvidenceSourceAdapter(result(status=status))
            collected = await adapter.collect(request())
            assert collected.collection_status is status
            assert not collected.items

    asyncio.run(scenario())


def test_adapter_failures_are_redacted_and_timezone_normalized() -> None:
    """Timeout and provider failures expose only stable degradation codes."""

    async def scenario() -> None:
        timestamp = datetime(2026, 1, 3, 8, tzinfo=UTC)
        timeout = await collect_source(FailingEvidenceSourceAdapter(timeout=True), request(), captured_at=timestamp)
        failed = await collect_source(FailingEvidenceSourceAdapter(), request(), captured_at=timestamp)
        assert timeout.degradation_reason == "source_timeout"
        assert failed.degradation_reason == "source_unavailable"
        assert timeout.items == failed.items == []

    asyncio.run(scenario())
