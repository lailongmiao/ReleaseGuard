"""Read-only evidence retrieval and memory degradation tests."""

from datetime import UTC, datetime
from hashlib import sha256
from asyncio import run

import pytest

from app.core import config as config_module
from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCollectionStatus,
    EvidenceItem,
    EvidenceSourceType,
    RetrievalStatus,
)
from app.services.evidence_store import EvidenceAccessContext, InMemoryEvidenceStore


def _item(evidence_id: str, snippet: str, *, scope: EvidenceAccessScope, content: str | None = None) -> EvidenceItem:
    """Create a fixture evidence item with a deterministic content hash."""
    value = content or snippet
    return EvidenceItem(
        evidence_id=evidence_id,
        source_type=EvidenceSourceType.DOCUMENT,
        source_uri=f"fixture://{evidence_id}",
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=f"sha256:{sha256(value.encode()).hexdigest()}",
        access_scope=scope,
        snippet=snippet,
        confidence=1.0,
        collection_status=EvidenceCollectionStatus.VERIFIED,
    )


def test_evidence_filters_access_before_dedup_and_stable_ranking() -> None:
    """Exclude restricted evidence, deduplicate hashes, and rank deterministically."""
    public = EvidenceAccessScope(public=True)
    tenant = EvidenceAccessScope(tenant_id="tenant-a")
    items = [
        _item("z-public", "deployment passed in production", scope=public, content="same"),
        _item("a-public-duplicate", "deployment passed in production", scope=public, content="same"),
        _item("private", "deployment passed in production", scope=tenant),
    ]
    store = InMemoryEvidenceStore(items, snippet_max_chars=20)

    result = run(store.search("deployment production", access=EvidenceAccessContext(), limit=5))

    assert result.status is RetrievalStatus.SUCCEEDED
    assert [item.evidence_id for item in result.items] == ["a-public-duplicate"]
    assert result.metadata.deduplicated_count == 1


def test_evidence_scope_allows_tenant_and_truncates_snippet() -> None:
    """Allow matching tenant evidence and return bounded citation snippets."""
    item = _item(
        "tenant-doc",
        "rollback plan verified and ready for production deployment",
        scope=EvidenceAccessScope(tenant_id="tenant-a"),
    )
    store = InMemoryEvidenceStore([item], snippet_max_chars=20)

    result = run(
        store.search(
            "rollback production",
            access=EvidenceAccessContext(tenant_id="tenant-a"),
        )
    )

    assert result.status is RetrievalStatus.SUCCEEDED
    assert len(result.items[0].snippet) <= 20
    assert result.items[0].snippet.endswith("…")


def test_evidence_empty_and_unavailable_are_structured() -> None:
    """Represent empty and unavailable sources without fabricated evidence."""
    empty = run(InMemoryEvidenceStore().search("missing", access=EvidenceAccessContext()))
    unavailable = run(InMemoryEvidenceStore(available=False).search("query", access=EvidenceAccessContext()))

    assert empty.status is RetrievalStatus.EMPTY
    assert unavailable.status is RetrievalStatus.PROVIDER_UNAVAILABLE
    assert unavailable.degradation_reason == "evidence_store_unavailable"


def test_evidence_settings_have_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep memory and fixture retrieval configurable without secrets."""
    monkeypatch.delenv("MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("LOCAL_MVP_ONLY", raising=False)
    monkeypatch.delenv("EVIDENCE_PROVIDER", raising=False)
    fresh = config_module.Settings()

    assert fresh.MEMORY_ENABLED is True
    assert fresh.EVIDENCE_PROVIDER == "fixture"
