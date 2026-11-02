"""Contract tests for the Phase 1A evidence boundary."""

import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCollectionStatus,
    EvidenceItem,
    EvidenceSourceType,
    RetrievalMetadata,
    RetrievalResult,
    RetrievalStatus,
)


HASH = "a" * 64
CAPTURED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_item(**overrides: object) -> EvidenceItem:
    """Build a valid item for focused contract assertions."""
    values: dict[str, object] = {
        "evidence_id": "fixture:ci:1",
        "source_type": EvidenceSourceType.CI,
        "source_uri": "fixture://ci/1",
        "captured_at": CAPTURED_AT,
        "content_hash": HASH,
        "access_scope": EvidenceAccessScope(public=True),
        "snippet": "build passed",
        "confidence": 0.9,
    }
    values.update(overrides)
    return EvidenceItem(**values)


def build_metadata(**overrides: object) -> RetrievalMetadata:
    """Build valid provider-neutral retrieval metadata."""
    values: dict[str, object] = {
        "provider": "fixture",
        "strategy": "fixture",
        "query_hash": HASH,
        "limit": 10,
    }
    values.update(overrides)
    return RetrievalMetadata(**values)


def test_evidence_item_exposes_citable_fields_and_redacted_log_context() -> None:
    """A valid item contains citation fields while logs omit source content."""
    item = build_item(metadata={"score": 0.8})

    assert item.source_uri == "fixture://ci/1"
    assert item.captured_at.tzinfo is not None
    assert item.redacted_log_context() == {
        "evidence_id": "fixture:ci:1",
        "content_hash": HASH,
        "source_type": "ci",
        "collection_status": "verified",
    }
    assert "snippet" not in item.redacted_log_context()
    assert "source_uri" not in item.redacted_log_context()


@pytest.mark.parametrize(
    "overrides",
    [
        {"captured_at": datetime(2026, 1, 1)},
        {"content_hash": "not-a-hash"},
        {"confidence": 1.1},
        {"access_scope": {}},
    ],
)
def test_evidence_item_rejects_invalid_boundaries(overrides: dict[str, object]) -> None:
    """Missing integrity, time, confidence, or scope guarantees must fail."""
    with pytest.raises(ValidationError):
        build_item(**overrides)


def test_non_public_scope_requires_an_explicit_restriction() -> None:
    """An empty non-public scope must fail closed."""
    with pytest.raises(ValidationError):
        EvidenceAccessScope()


def test_access_scope_json_is_stable_across_process_hash_seeds() -> None:
    """Restarted processes must serialize equal ACL sets into identical facts."""
    command = (
        "from app.schemas.evidence import EvidenceAccessScope; "
        "print(EvidenceAccessScope(principals={'owner', 'approver'}, roles={'reader', 'reviewer'}).model_dump_json())"
    )
    results = [
        subprocess.check_output(
            [sys.executable, "-c", command], env={**os.environ, "PYTHONHASHSEED": str(seed)}
        ).splitlines()[-1]
        for seed in (1, 2, 3)
    ]
    assert results[0] == results[1] == results[2]


def test_retrieval_result_supports_empty_and_unavailable_states() -> None:
    """Callers receive structured degradation states without fabricated items."""
    empty = RetrievalResult(
        status=RetrievalStatus.EMPTY,
        metadata=build_metadata(result_count=0),
    )
    unavailable = RetrievalResult(
        status=RetrievalStatus.PROVIDER_UNAVAILABLE,
        metadata=build_metadata(result_count=0),
        degradation_reason="embedding_provider_timeout",
    )

    assert empty.items == []
    assert unavailable.degradation_reason == "embedding_provider_timeout"


def test_retrieval_result_rejects_inconsistent_success_metadata() -> None:
    """The result count and status must agree with returned items."""
    with pytest.raises(ValidationError):
        RetrievalResult(
            status=RetrievalStatus.SUCCEEDED,
            items=[build_item(collection_status=EvidenceCollectionStatus.VERIFIED)],
            metadata=build_metadata(result_count=0),
        )
