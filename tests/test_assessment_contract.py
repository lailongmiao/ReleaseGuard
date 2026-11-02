"""Assessment consistency and Phase 1 evidence reuse tests."""

import pytest
from pydantic import ValidationError

from app.schemas.evidence import EvidenceItem
from app.schemas.releaseguard import AssessmentContext, EvidenceAssessment, RiskAssessment, RiskLevel
from releaseguard_fixtures import evidence_assessment, evidence_fixture, risk_assessment, run_binding


def test_assessments_reuse_evidence_and_roundtrip() -> None:
    """References bind directly to Phase 1 objects and survive serialization."""
    context = AssessmentContext(
        identity=run_binding(),
        evidence_items=[evidence_fixture()],
        evidence_assessment=evidence_assessment(required_items=["ci", "ci"]),
        risk_assessment=risk_assessment(),
    )
    assert type(context.evidence_items[0]) is EvidenceItem
    assert context.evidence_assessment.required_items == ["ci"]
    assert AssessmentContext.model_validate_json(context.model_dump_json()) == context


@pytest.mark.parametrize(
    "overrides",
    [
        {"present_items": ["other"]},
        {"missing_items": ["ci"]},
        {"required_items": [], "present_items": []},
        {"uncertainty": ["synthetic gap"]},
        {"evidence_ids": []},
        {"sufficient": False},
    ],
)
def test_contradictory_assessments_fail(overrides: dict[str, object]) -> None:
    """Internal inconsistency cannot be serialized as sufficient evidence."""
    with pytest.raises(ValidationError):
        evidence_assessment(**overrides)


def test_insufficient_result_retains_gaps() -> None:
    """Explicit absence remains observable without invoking a production gate."""
    result = evidence_assessment(sufficient=False, present_items=[], missing_items=["ci"], evidence_ids=[])
    assert EvidenceAssessment.model_validate_json(result.model_dump_json()).missing_items == ["ci"]
    empty = evidence_assessment(
        sufficient=False,
        required_items=[],
        present_items=[],
        evidence_ids=[],
        blocking_reasons=["synthetic rules absent"],
    )
    assert not empty.sufficient


@pytest.mark.parametrize("level", list(RiskLevel))
def test_risk_levels_remain_explicit(level: RiskLevel) -> None:
    """UNKNOWN survives roundtrip and never becomes LOW through a default."""
    result = risk_assessment(level=level, uncertainty=["synthetic unknown"] if level is RiskLevel.UNKNOWN else [])
    assert RiskAssessment.model_validate_json(result.model_dump_json()).level is level


def test_unknown_and_assumptions_require_explicit_context() -> None:
    """Uncertainty and unsupported factors cannot be silently discarded."""
    with pytest.raises(ValidationError):
        risk_assessment(level="UNKNOWN")
    with pytest.raises(ValidationError):
        risk_assessment(evidence_ids=[])
    assert risk_assessment(evidence_ids=[], factors=["assumption: synthetic estimate"]).level is RiskLevel.LOW


@pytest.mark.parametrize("foreign", ["missing_reference", "foreign_run", "duplicate_item"])
def test_assessment_context_rejects_foreign_references(foreign: str) -> None:
    """A supplied assessment cannot refer to unknown evidence or another run."""
    with pytest.raises(ValidationError):
        AssessmentContext(
            identity=run_binding(),
            evidence_items=[evidence_fixture()] * (2 if foreign == "duplicate_item" else 1),
            evidence_assessment=evidence_assessment(
                evidence_ids=["missing"] if foreign == "missing_reference" else ["synthetic:ci"],
                run_id="foreign-run" if foreign == "foreign_run" else "synthetic-run",
            ),
        )
