"""Deterministic, fail-closed Evidence Sufficiency Gate tests."""

from datetime import UTC, datetime

import pytest

from app.schemas.evidence import (
    EvidenceAccessScope,
    EvidenceCitation,
    EvidenceCollectionStatus,
    EvidenceItem,
    EvidencePolicy,
    EvidenceRequirement,
    EvidenceSnapshot,
    EvidenceSourceResult,
    EvidenceSourceType,
)
from app.schemas.releaseguard import RiskAssessment, RiskLevel, new_change_run
from app.services.evidence_sources import EvidenceAccessContext
from app.services.releaseguard.evidence_policy import evaluate_evidence_sufficiency
from app.services.releaseguard.normalization import canonical_hash
from tests.test_change_request_contract import request_fixture


EVALUATED_AT = datetime(2026, 1, 2, tzinfo=UTC)
CAPTURED_AT = datetime(2026, 1, 1, tzinfo=UTC)
REQUEST_HASH = request_fixture().request_hash


def item_fixture(**overrides: object) -> EvidenceItem:
    """Build one synthetic CI EvidenceItem."""
    values: dict[str, object] = {
        "evidence_id": "synthetic:ci",
        "source_type": EvidenceSourceType.CI,
        "source_uri": "fixture://ci/build-1",
        "source_version": "build-1",
        "captured_at": CAPTURED_AT,
        "content_hash": "a" * 64,
        "access_scope": EvidenceAccessScope(public=True),
        "snippet": "synthetic build passed",
        "confidence": 1.0,
    }
    values.update(overrides)
    return EvidenceItem.model_validate(values)


def snapshot_fixture(**overrides: object) -> EvidenceSnapshot:
    """Create an immutable snapshot with its canonical id."""
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


def policy_fixture(**overrides: object) -> EvidencePolicy:
    """Require one CI item with a deterministic minimum confidence."""
    values: dict[str, object] = {
        "policy_version": "synthetic-policy.v1",
        "requirements": [
            EvidenceRequirement(
                requirement_id="ci",
                source_types=frozenset({EvidenceSourceType.CI}),
                min_confidence=0.8,
            )
        ],
    }
    values.update(overrides)
    return EvidencePolicy.model_validate(values)


def citation_fixture(snapshot: EvidenceSnapshot, **overrides: object) -> EvidenceCitation:
    """Bind a citation to the fixture request and graph thread."""
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
        "request_hash": REQUEST_HASH,
        "thread_id": "synthetic-thread",
        "policy_version": "synthetic-policy.v1",
        "cited_at": EVALUATED_AT,
    }
    values.update(overrides)
    return EvidenceCitation.model_validate(values)


def state_fixture():
    """Build the current Phase 2 aggregate used by the gate."""
    return new_change_run(request_fixture(), "synthetic-run")


def evaluate(
    *,
    snapshots: list[EvidenceSnapshot] | None = None,
    citations: list[EvidenceCitation] | None = None,
    policy: EvidencePolicy | None = None,
    access: EvidenceAccessContext | None = None,
    source_results: list[EvidenceSourceResult] | None = None,
):
    """Evaluate with stable identity and explicit time."""
    snapshot_values = snapshots if snapshots is not None else [snapshot_fixture()]
    citation_values = citations if citations is not None else [citation_fixture(snapshot_values[0])]
    return evaluate_evidence_sufficiency(
        state_fixture(),
        snapshots=snapshot_values,
        citations=citation_values,
        policy=policy or policy_fixture(),
        evaluated_at=EVALUATED_AT,
        thread_id="synthetic-thread",
        access=access or EvidenceAccessContext(),
        source_results=source_results or [],
    )


def test_complete_current_evidence_is_sufficient_and_deterministic() -> None:
    """The same typed inputs produce the same sufficient assessment."""
    first = evaluate()
    second = evaluate()

    assert first == second
    assert first.sufficient is True
    assert first.required_items == ["ci"]
    assert first.present_items == ["ci"]
    assert first.missing_items == []
    assert first.evidence_ids == ["synthetic:ci"]


@pytest.mark.parametrize(
    "policy",
    [EvidencePolicy(policy_version="empty-policy", requirements=[])],
)
def test_empty_policy_fails_closed(policy: EvidencePolicy) -> None:
    """An empty policy cannot silently make an assessment sufficient."""
    assessment = evaluate(policy=policy, citations=[])

    assert assessment.sufficient is False
    assert "policy_missing_or_empty" in assessment.blocking_reasons


def test_missing_required_item_is_blocking() -> None:
    """A required item without a citation remains missing."""
    assessment = evaluate(citations=[])

    assert assessment.sufficient is False
    assert assessment.missing_items == ["ci"]
    assert "missing_required_item:ci" in assessment.blocking_reasons


@pytest.mark.parametrize(
    "status",
    [
        EvidenceCollectionStatus.UNAVAILABLE,
        EvidenceCollectionStatus.PERMISSION_DENIED,
        EvidenceCollectionStatus.CONFLICTING,
    ],
)
def test_untrusted_source_states_are_blocking(status: EvidenceCollectionStatus) -> None:
    """Source degradation is never rewritten as empty or verified success."""
    result = EvidenceSourceResult(
        source_id="synthetic-source",
        source_type=EvidenceSourceType.CI,
        requirement_id="ci",
        collection_status=status,
        captured_at=EVALUATED_AT,
        audit_reference="synthetic-audit",
        degradation_reason="synthetic_source_state",
    )

    assessment = evaluate(citations=[], source_results=[result])

    assert assessment.sufficient is False
    assert f"source_{status.value}:ci" in assessment.blocking_reasons


def test_expired_evidence_is_missing() -> None:
    """Freshness is evaluated against the explicit evaluation time."""
    snapshot = snapshot_fixture(freshness_until=EVALUATED_AT)
    assessment = evaluate(snapshots=[snapshot], citations=[citation_fixture(snapshot)])

    assert assessment.sufficient is False
    assert "evidence_expired:ci" in assessment.blocking_reasons


def test_hash_version_and_run_bindings_are_blocking() -> None:
    """A citation cannot authorize another content version or graph run."""
    snapshot = snapshot_fixture()
    hash_mismatch = citation_fixture(
        snapshot,
        content_hash="c" * 64,
        citation_id="synthetic:citation:hash",
    )
    binding_mismatch = citation_fixture(
        snapshot,
        run_id="other-run",
        citation_id="synthetic:citation:binding",
    )
    assessment = evaluate(snapshots=[snapshot], citations=[hash_mismatch, binding_mismatch])

    assert assessment.sufficient is False
    assert "citation_hash_or_version_mismatch" in assessment.blocking_reasons
    assert "citation_binding_mismatch" in assessment.blocking_reasons


def test_access_denied_is_blocking() -> None:
    """A caller without the item scope cannot satisfy the policy."""
    snapshot = snapshot_fixture(access_scope=EvidenceAccessScope(tenant_id="tenant-a"))
    assessment = evaluate(
        snapshots=[snapshot],
        citations=[citation_fixture(snapshot)],
        access=EvidenceAccessContext(tenant_id="tenant-b"),
    )

    assert assessment.sufficient is False
    assert "citation_access_denied" in assessment.blocking_reasons


def test_conflicting_hashes_for_one_requirement_are_blocking() -> None:
    """Different versions are retained as a conflict, never chosen by recency."""
    first = snapshot_fixture()
    second = snapshot_fixture(
        source_version="build-2",
        content_hash="b" * 64,
        source_uri="fixture://ci/build-2",
    )
    assessment = evaluate(
        snapshots=[first, second],
        citations=[citation_fixture(first), citation_fixture(second, citation_id="synthetic:citation:2")],
    )

    assert assessment.sufficient is False
    assert "conflicting_evidence:ci" in assessment.blocking_reasons


def test_unknown_risk_level_remains_explicitly_unknown() -> None:
    """The Evidence gate never maps an unknown risk result to LOW."""
    risk = RiskAssessment.model_validate(
        {
            "change_id": "synthetic-change",
            "run_id": "synthetic-run",
            "revision": 1,
            "request_hash": REQUEST_HASH,
            "level": RiskLevel.UNKNOWN,
            "blast_radius": "unknown",
            "uncertainty": ["synthetic uncertainty"],
            "policy_version": "synthetic-policy.v1",
            "assessed_at": EVALUATED_AT,
        }
    )

    assessment = evaluate()

    assert assessment.sufficient is True
    assert risk.level is RiskLevel.UNKNOWN
