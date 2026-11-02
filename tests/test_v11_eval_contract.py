"""Frozen denominators, separate semantic evidence and no paid synthetic sampling."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.services.releaseguard.normalization import canonical_hash
from evals.v11_backend.contracts import Dataset, SemanticReview
from evals.v11_backend.development import dataset
from evals.v11_backend.real import Authorization
from evals.v11_backend.runner import evaluate
from tests.test_v11_policy import output


def batch() -> tuple[dict, SemanticReview]:
    """Perfect synthetic scoring fixture proves no real model quality."""
    data = dataset()
    positions = [
        {
            "case_id": case.case_id,
            "repeat": repeat,
            "status": "completed",
            "analysis": {"output": output().model_dump(mode="json")},
            "input": {"query": case.query},
            "hard_gates_passed": True,
            "rag_status": "completed",
            "elapsed_seconds": 1,
            "retrieved_documents": case.expected_documents,
        }
        for case in data.cases
        for repeat in (1, 2)
    ]
    report = {"dataset": data.model_dump(mode="json"), "positions": positions}
    review = SemanticReview.model_validate(
        {
            "review_method": "assistant_delegated",
            "independent_human_reviewed": False,
            "report_hash": canonical_hash(report),
            "positions": [
                {
                    "case_id": p["case_id"],
                    "repeat": p["repeat"],
                    "score": 5,
                    "claims_supported": 1,
                    "body_supported": 1,
                    "unknown_correct": True,
                    "critical_error": False,
                    "citation_identity_valid": True,
                    "rationale": "Synthetic evaluator counterexample fixture, not a real semantic review.",
                    "output_hash": canonical_hash(p["analysis"]),
                    "input_hash": canonical_hash(p["input"]),
                }
                for p in positions
            ],
        }
    )
    return report, review


def test_fixed_matrix_rejects_relabeling_or_missing_slots() -> None:
    """Removing failed positions cannot improve the frozen denominator."""
    value = dataset().model_dump(mode="json")
    value["cases"].pop()
    with pytest.raises(ValidationError, match="forty_unique"):
        Dataset.model_validate(value)
    report, review = batch()
    report["positions"].pop()
    review.report_hash = canonical_hash(report)
    assert evaluate(report, review)["reason"] == "frozen_denominator_incomplete"


def test_security_failure_cannot_be_averaged_into_success() -> None:
    """A single critical error defeats even otherwise perfect mean scores."""
    report, review = batch()
    assert evaluate(report, review)["passed"]
    review.positions[0].critical_error = True
    assert not evaluate(report, review)["passed"]


@pytest.mark.parametrize("field,value", [("score", 2), ("claims_supported", 0), ("body_supported", 0)])
def test_quality_is_separate_from_critical_identity(field: str, value: int) -> None:
    """Quality failures do not disappear behind passing identity checks."""
    report, review = batch()
    changed = deepcopy(review)
    for position in changed.positions[:10]:
        setattr(position, field, value)
    assert not evaluate(report, changed)["passed"]


def test_changed_output_requires_new_semantic_review() -> None:
    """A semantic judgment is bound to the actual output it reviewed."""
    report, review = batch()
    report["positions"][0]["analysis"]["output"]["summary"] = "changed"
    review.report_hash = canonical_hash(report)
    with pytest.raises(ValueError, match="semantic_position_hash_mismatch"):
        evaluate(report, review)


def test_synthetic_cases_never_authorize_real_sampling() -> None:
    """Synthetic test composition cannot be relabeled as frozen public evidence."""
    now = datetime.now(UTC)
    auth = Authorization.model_validate(
        {
            "authorization_reference": "synthetic-test-only",
            "authorized_at": now,
            "expires_at": now + timedelta(hours=1),
            "public_github_reads": True,
            "model_calls": True,
            "max_analysis_calls": 80,
            "dataset_hash": "placeholder",
            "source_fingerprint": "placeholder",
            "designated_ledger_id": "synthetic-ledger",
            "reconciled_budget_evidence_hash": "placeholder",
            "price_evidence_hash": "placeholder",
            "api_origin": "http://127.0.0.1:8000",
            "database_name": "v11_isolated",
            "database_port": 35442,
            "owner_user_id": 1,
            "approver_user_id": 2,
            "session_token": "synthetic-no-login",
            "budget_remaining_cny": 20,
            "rules_ready": True,
        }
    )
    with pytest.raises(ValueError, match="real_frozen_source_evidence_required"):
        auth.validate_preflight(dataset())
