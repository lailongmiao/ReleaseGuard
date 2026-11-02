"""Offline immutable Evidence snapshot and Citation repository contracts."""

import asyncio
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from app.repositories.in_memory_evidence import InMemoryEvidenceRepository, snapshot_id_for_item
from app.schemas.evidence import (
    EvidenceCitation,
    EvidenceCollectionStatus,
    EvidenceItem,
    EvidenceAccessScope,
    EvidenceSnapshot,
    EvidenceSourceType,
)
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError


def item(version: str = "v1", *, content: str = "verified deployment") -> EvidenceItem:
    """Build a bounded public fixture with a stable content hash."""
    return EvidenceItem(
        evidence_id="ci-deployment",
        source_type=EvidenceSourceType.CI,
        source_uri="fixture://ci/deployment",
        source_version=version,
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=f"sha256:{sha256(content.encode()).hexdigest()}",
        access_scope=EvidenceAccessScope(public=True),
        snippet=content,
        confidence=1.0,
        freshness_until=datetime(2026, 2, 1, tzinfo=UTC),
        collection_status=EvidenceCollectionStatus.VERIFIED,
    )


def snapshot(value: EvidenceItem) -> EvidenceSnapshot:
    """Wrap one item in its deterministic immutable snapshot envelope."""
    return EvidenceSnapshot(
        snapshot_id=snapshot_id_for_item(value),
        item=value,
        persisted_at=datetime(2026, 1, 2, tzinfo=UTC),
    )


def citation(value: EvidenceSnapshot, *, citation_id: str = "citation-1") -> EvidenceCitation:
    """Build a citation matching the stored snapshot exactly."""
    return EvidenceCitation(
        citation_id=citation_id,
        snapshot_id=value.snapshot_id,
        evidence_id=value.item.evidence_id,
        source_version=value.item.source_version,
        content_hash=value.item.content_hash,
        captured_at=value.item.captured_at,
        required_item="ci_success",
        change_id="change-1",
        run_id="run-1",
        revision=1,
        request_hash=canonical_hash({"request": "change-1"}),
        thread_id="thread-1",
        policy_version="policy.v1",
        cited_at=datetime(2026, 1, 2, 1, tzinfo=UTC),
    )


def test_snapshot_round_trip_and_same_version_replay() -> None:
    """Equivalent snapshots are idempotent and JSON-stable."""

    async def scenario() -> None:
        repository = InMemoryEvidenceRepository()
        stored = await repository.append_snapshot(snapshot(item()))
        replay = await repository.append_snapshot(stored)
        assert replay == stored
        assert EvidenceSnapshot.model_validate_json(stored.model_dump_json()) == stored
        assert await repository.get_snapshot(stored.snapshot_id) == stored

    asyncio.run(scenario())


def test_same_source_version_with_different_hash_is_rejected() -> None:
    """A new content hash cannot overwrite an existing source version."""

    async def scenario() -> None:
        repository = InMemoryEvidenceRepository()
        await repository.append_snapshot(snapshot(item()))
        with pytest.raises(DomainError, match="evidence_conflict"):
            await repository.append_snapshot(snapshot(item(content="changed deployment")))

    asyncio.run(scenario())


def test_distinct_versions_are_retained_and_citation_is_bound() -> None:
    """Version history remains addressable and citations cannot drift."""

    async def scenario() -> None:
        repository = InMemoryEvidenceRepository()
        first = snapshot(item("v1"))
        second = snapshot(item("v2", content="new verified deployment"))
        await repository.append_snapshot(first)
        await repository.append_snapshot(second)
        assert len(await repository.find_snapshots("ci-deployment")) == 2
        saved = await repository.append_citation(citation(second))
        assert await repository.get_citation(saved.citation_id) == saved
        with pytest.raises(DomainError, match="citation_snapshot_mismatch"):
            await repository.append_citation(
                citation(second).model_copy(update={"content_hash": first.item.content_hash})
            )

    asyncio.run(scenario())


def test_untrusted_snapshot_and_missing_citation_target_fail_closed() -> None:
    """Unavailable source states never become trusted persisted evidence."""

    async def scenario() -> None:
        repository = InMemoryEvidenceRepository()
        unavailable = item().model_copy(update={"collection_status": EvidenceCollectionStatus.UNAVAILABLE})
        with pytest.raises(DomainError, match="untrusted_evidence"):
            await repository.append_snapshot(snapshot(unavailable))
        missing = citation(snapshot(item()), citation_id="missing")
        with pytest.raises(DomainError, match="evidence_not_found"):
            await repository.append_citation(missing)

    asyncio.run(scenario())


def test_snapshot_identifier_changes_for_identity_or_version() -> None:
    """The snapshot identity includes source namespace and source version."""
    assert snapshot_id_for_item(item("v1")) != snapshot_id_for_item(item("v2"))
    assert snapshot_id_for_item(item()) == snapshot(item()).snapshot_id
