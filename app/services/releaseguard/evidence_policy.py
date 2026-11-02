"""Pure deterministic Evidence Sufficiency Gate for ReleaseGuard."""

from collections.abc import Sequence
from datetime import UTC, datetime

from app.schemas.evidence import (
    EvidenceCitation,
    EvidenceCollectionStatus,
    EvidencePolicy,
    EvidenceSnapshot,
    EvidenceSourceResult,
)
from app.schemas.releaseguard import ChangeRunState, EvidenceAssessment
from app.services.evidence_sources import EvidenceAccessContext, scope_allows


def evaluate_evidence_sufficiency(
    state: ChangeRunState,
    *,
    snapshots: Sequence[EvidenceSnapshot],
    citations: Sequence[EvidenceCitation],
    policy: EvidencePolicy,
    evaluated_at: datetime,
    thread_id: str,
    access: EvidenceAccessContext,
    source_results: Sequence[EvidenceSourceResult] = (),
) -> EvidenceAssessment:
    """Return a fail-closed assessment from explicit, immutable inputs."""
    if evaluated_at.tzinfo is None:
        raise ValueError("timestamp must include timezone information")
    evaluated_at = evaluated_at.astimezone(UTC)
    requirements = {item.requirement_id: item for item in policy.requirements}
    required_ids = set(requirements)
    blocking: list[str] = []
    uncertainty: list[str] = []
    if not required_ids:
        required_ids = {"policy"}
        blocking.append("policy_missing_or_empty")

    snapshot_by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    if len(snapshot_by_id) != len(snapshots):
        blocking.append("duplicate_snapshot_id")
    valid_by_requirement: dict[str, set[str]] = {key: set() for key in required_ids}
    seen_hashes: dict[str, set[str]] = {key: set() for key in required_ids}

    for result in source_results:
        if result.requirement_id not in required_ids:
            blocking.append("unknown_source_requirement")
        elif result.collection_status is not EvidenceCollectionStatus.VERIFIED:
            blocking.append(f"source_{result.collection_status.value}:{result.requirement_id}")

    for citation in citations:
        requirement = requirements.get(citation.required_item)
        snapshot = snapshot_by_id.get(citation.snapshot_id)
        if requirement is None:
            blocking.append("unknown_required_item")
            continue
        if snapshot is None:
            blocking.append("citation_snapshot_missing")
            continue
        item = snapshot.item
        if (
            item.evidence_id != citation.evidence_id
            or item.source_version != citation.source_version
            or item.content_hash != citation.content_hash
            or item.captured_at != citation.captured_at
        ):
            blocking.append("citation_hash_or_version_mismatch")
            continue
        if (
            citation.change_id,
            citation.run_id,
            citation.revision,
            citation.request_hash,
            citation.thread_id,
            citation.policy_version,
        ) != (
            state.change_id,
            state.run_id,
            state.revision,
            state.request_hash,
            thread_id,
            policy.policy_version,
        ):
            blocking.append("citation_binding_mismatch")
            continue
        if item.collection_status is not EvidenceCollectionStatus.VERIFIED:
            blocking.append("untrusted_collection_status")
            continue
        if (
            policy.policy_version == "product-evidence.v1"
            and citation.required_item == "ci"
            and item.metadata.get("checks_passed") is not True
        ):
            blocking.append("ci_policy_blocked")
            continue
        if not scope_allows(item.access_scope, access):
            blocking.append("citation_access_denied")
            continue
        if item.freshness_until is not None and item.freshness_until <= evaluated_at:
            blocking.append(f"evidence_expired:{citation.required_item}")
            continue
        if requirement.source_types and item.source_type not in requirement.source_types:
            blocking.append(f"source_type_mismatch:{citation.required_item}")
            continue
        if item.confidence < requirement.min_confidence:
            blocking.append(f"confidence_below_policy:{citation.required_item}")
            continue
        seen_hashes[citation.required_item].add(item.content_hash)
        valid_by_requirement[citation.required_item].add(item.evidence_id)

    for requirement_id, hashes in seen_hashes.items():
        if len(hashes) > 1:
            blocking.append(f"conflicting_evidence:{requirement_id}")

    missing = {key for key in required_ids if not valid_by_requirement[key]}
    if missing:
        blocking.extend(f"missing_required_item:{key}" for key in sorted(missing))
    present = required_ids - missing
    evidence_ids = sorted({evidence_id for ids in valid_by_requirement.values() for evidence_id in ids})
    blocking = sorted(set(blocking))
    sufficient = bool(requirements) and not missing and not blocking and not uncertainty
    if not sufficient and not blocking:
        uncertainty.append("evidence_gate_not_sufficient")
    return EvidenceAssessment(
        change_id=state.change_id,
        run_id=state.run_id,
        revision=state.revision,
        request_hash=state.request_hash,
        sufficient=sufficient,
        required_items=sorted(required_ids),
        present_items=sorted(present),
        missing_items=sorted(missing),
        blocking_reasons=blocking,
        evidence_ids=evidence_ids,
        uncertainty=uncertainty,
        policy_version=policy.policy_version,
        evaluated_at=evaluated_at,
    )
