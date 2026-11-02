"""Citation binding and immutable snapshot contract tests for Phase 4C."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCitation,
    EvidenceItem,
    EvidenceSnapshot,
    EvidenceSourceType,
)
from app.services.releaseguard.normalization import canonical_hash


CAPTURED_AT = datetime(2026, 1, 1, tzinfo=UTC)
CONTENT_HASH = "a" * 64


def item_fixture(**overrides: object) -> EvidenceItem:
    """Build a bounded, public synthetic evidence item."""
    values: dict[str, object] = {
        "evidence_id": "synthetic:ci",
        "source_type": EvidenceSourceType.CI,
        "source_uri": "fixture://ci/build-1",
        "source_version": "build-1",
        "captured_at": CAPTURED_AT,
        "content_hash": CONTENT_HASH,
        "access_scope": EvidenceAccessScope(public=True),
        "snippet": "synthetic build passed",
        "confidence": 1.0,
    }
    values.update(overrides)
    return EvidenceItem.model_validate(values)


def snapshot_fixture(**overrides: object) -> EvidenceSnapshot:
    """Build a snapshot with its deterministic identity."""
    item = item_fixture(**overrides)
    snapshot_id = "sha256:" + canonical_hash(
        {
            "evidence_id": item.evidence_id,
            "source_uri": item.source_uri,
            "source_version": item.source_version,
            "content_hash": item.content_hash,
        }
    )
    return EvidenceSnapshot(snapshot_id=snapshot_id, item=item, persisted_at=CAPTURED_AT)


def citation_fixture(snapshot: EvidenceSnapshot, **overrides: object) -> EvidenceCitation:
    """Build a citation bound to the synthetic ReleaseGuard run."""
    item = snapshot.item
    values: dict[str, object] = {
        "citation_id": "synthetic:citation:1",
        "snapshot_id": snapshot.snapshot_id,
        "evidence_id": item.evidence_id,
        "source_version": item.source_version,
        "content_hash": item.content_hash,
        "captured_at": item.captured_at,
        "required_item": "ci",
        "change_id": "synthetic-change",
        "run_id": "synthetic-run",
        "revision": 1,
        "request_hash": "b" * 64,
        "thread_id": "synthetic-thread",
        "policy_version": "synthetic-policy.v1",
        "cited_at": CAPTURED_AT,
    }
    values.update(overrides)
    return EvidenceCitation.model_validate(values)


def test_snapshot_identity_is_deterministic_and_citation_round_trips() -> None:
    """A citation keeps only the immutable snapshot identity and audit binding."""
    snapshot = snapshot_fixture()
    citation = citation_fixture(snapshot)

    restored = EvidenceCitation.model_validate_json(citation.model_dump_json())

    assert restored == citation
    assert restored.snapshot_id == snapshot.snapshot_id
    assert "snippet" not in restored.model_dump()


def test_snapshot_rejects_forged_identity() -> None:
    """A snapshot id cannot be detached from its source/version/hash tuple."""
    snapshot = snapshot_fixture()

    with pytest.raises(ValidationError, match="snapshot_id_mismatch"):
        EvidenceSnapshot(
            snapshot_id="sha256:" + "f" * 64,
            item=snapshot.item,
            persisted_at=CAPTURED_AT,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("content_hash", "c" * 64),
        ("source_version", "build-2"),
        ("revision", 2),
        ("thread_id", "other-thread"),
    ],
)
def test_citation_can_be_constructed_but_gate_must_recheck_binding(field: str, value: object) -> None:
    """Schema shape alone cannot make a citation valid for another snapshot/run."""
    snapshot = snapshot_fixture()
    citation = citation_fixture(snapshot, **{field: value})

    assert citation.snapshot_id == snapshot.snapshot_id
