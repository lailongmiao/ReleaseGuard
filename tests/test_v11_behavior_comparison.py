"""Independent counterexamples distinguish evidence checks from semantic truth."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.schemas.v11_backend import ReviewOutput
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_comparison import ComparedReviewOutput, validate_comparisons
from app.services.v11_policy import citation_line_roles
from app.services.v11_provider import build_context
from evals.v11_backend.behavior_cases import cases, evidence


def wire(case: dict, citations: list) -> ComparedReviewOutput:
    """Gold statements come from independent before/after source, not the model."""
    before = [c.citation_id for c in citations if c.side == "LEFT"]
    after = [c.citation_id for c in citations if c.side == "RIGHT"]
    return ComparedReviewOutput.model_validate(
        {
            "summary": case["facts"][0],
            "observations": [{"text": case["facts"][0], "kind": "observation", "citation_ids": before + after}],
            "risks": [],
            "validation_suggestions": [],
            "unknowns": [],
            "candidate_steps": [],
            "model_risk": "MEDIUM",
            "comparisons": [
                {
                    "section": "observations",
                    "index": 0,
                    "classification": case["expected"],
                    "before": case["before"],
                    "after": case["after"],
                    "before_ids": before,
                    "after_ids": after,
                }
            ],
        }
    )


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_independent_evidence_and_expected_categories(case: dict) -> None:
    """True changes and unchanged cases all retain paired, precisely mapped evidence."""
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Compare this change", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    assert {c.citation_id for c in selected} == {c.citation_id for c in citations}
    validate_comparisons(wire(case, citations), selected, json.loads(user))
    if case["id"] == "context":
        for c in citations:
            for line, kind in zip(c.text.split("\n"), roles[c.citation_id], strict=True):
                if line in {"limit = 12", "retries = 2"}:
                    assert kind == "context"
    if case["id"] == "condition":
        assert "if enabled:" in case["before"] and "if enabled and admin:" in case["after"]


@pytest.mark.parametrize(
    "damage", ["right_only", "wrong_side", "unselected", "partial", "no_edit", "document", "missing", "duplicate"]
)
def test_behavior_claim_requires_actual_paired_code(damage: str) -> None:
    """A plausible claim cannot use new-side text, docs, omitted input or context alone."""
    case = cases()[1]
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Compare", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    sent = json.loads(user)
    report = wire(case, citations).model_dump()
    item = report["comparisons"][0]
    if damage == "right_only":
        item["before_ids"] = []
    elif damage == "wrong_side":
        item["before_ids"] = item["after_ids"]
    elif damage == "unselected":
        sent["sources"] = sent["sources"][1:]
    elif damage == "partial":
        sent["SERVER_OBSERVATIONS"]["model_coverage_complete"] = False
    elif damage == "no_edit":
        for source in sent["sources"]:
            source["line_roles"] = ["context"] * len(source["line_roles"])
    elif damage == "document":
        selected = [c.model_copy(update={"path": "README.md"}) for c in selected]
    elif damage == "missing":
        report["comparisons"] = []
    elif damage == "duplicate":
        report["comparisons"].append(deepcopy(item))
    with pytest.raises(DomainError):
        validate_comparisons(ComparedReviewOutput.model_validate(report), selected, sent)


@pytest.mark.parametrize(
    "field,value",
    [("sha", "c" * 40), ("path", "foreign.py"), ("citation_id", "diff:foreign"), ("start_line", 2), ("side", "RIGHT")],
)
def test_actual_snapshot_rejects_forged_identity(field: str, value: object) -> None:
    """Even valid text hashes cannot substitute another side, object, SHA or range."""
    facts, citations, _ = evidence([cases()[1]])
    wrong = citations[0].model_copy(update={field: value})
    with pytest.raises(DomainError):
        citation_line_roles([wrong], facts, "synthetic-behavior")


def test_hunk_pair_not_split_by_context_capacity() -> None:
    """A large old side cannot leave only its small new side presented as comparison."""
    case = {**cases()[1], "before": "\n".join("x = " + str(i) for i in range(600)), "after": "x = 1"}
    facts, citations, roles = evidence([case])
    with pytest.raises(DomainError, match="unsupported_content"):
        build_context("Compare", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles)


def test_shared_literal_is_not_a_semantic_oracle() -> None:
    """The same literal links a changed condition too; validator cannot prove equivalence."""
    case = cases()[4]
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Compare", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    sent = json.loads(user)
    assert any(x["literal"] == "fast" for x in sent["comparison_index"]["shared_literals_not_equivalence"])
    wrong = wire(case, selected).model_dump()
    wrong["comparisons"][0]["classification"] = "relocated"
    # Deliberately documents the boundary: correct citations do not detect a false classification.
    validate_comparisons(ComparedReviewOutput.model_validate(wrong), selected, sent)
    assert case["expected"] == "behavior_modified"


def test_existing_report_bytes_unchanged_and_old_failure_kept() -> None:
    """New wire fields do not enter old output serialization or reinterpret history."""
    path = Path("evals/reports/v11_repair1_real.json")
    if not path.exists():
        pytest.skip("local historical evidence is not distributed")
    report = json.loads(path.read_text(encoding="utf-8"))
    for position in report["positions"]:
        raw = position["analysis"]["output"]
        assert canonical_hash(ReviewOutput.model_validate(raw).model_dump(mode="json")) == canonical_hash(raw)
    assert "新增默认" in report["positions"][8]["analysis"]["output"]["risks"][0]["text"]
