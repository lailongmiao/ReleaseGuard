"""Retained real fragment only, plus independent evidence/semantic counterexamples."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.schemas.v11_backend import ReviewOutput
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_comparison import (
    ComparedReviewOutput,
    ComparedReviewOutputV2,
    resolve_comparisons_v2,
    validate_comparisons,
)
from app.services.v11_policy import citation_line_roles, validate_output
from app.services.v11_provider import build_context
from evals.v11_backend.behavior_cases import cases, evidence
from tests.test_v11_behavior_comparison import wire


def v2(report: dict) -> ComparedReviewOutputV2:
    """Explicit fixture translation for future wire, not historical repair."""
    raw = deepcopy(report)
    raw["comparison_contract"] = "v11-behavior-comparison.2"
    for item in raw["comparisons"]:
        item.pop("before_ids")
        item.pop("after_ids")
    return ComparedReviewOutputV2.model_validate(raw)


def test_synthetic_fragment_supports_new_document_contract() -> None:
    """The v2 contract resolves document evidence without private evaluation artifacts."""
    case = cases()[3]
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Describe wording",
        [],
        [],
        citations,
        {"coverage": facts["coverage"], "index_complete": True},
        roles,
    )
    sent = json.loads(user)
    raw = wire(case, selected).model_dump()
    raw["comparisons"][0]["before_ids"] = []
    raw["comparisons"][0]["after_ids"] = []
    with pytest.raises(DomainError, match="analysis_documentation_requires_document"):
        validate_comparisons(ComparedReviewOutput.model_validate(raw), selected, sent)
    future = v2(raw)
    validate_output(future, selected)
    resolved = resolve_comparisons_v2(future, selected, sent)
    assert len(resolved.comparisons[0].before_ids) == len(resolved.comparisons[0].after_ids) == 1
    assert "before_ids" not in future.model_dump_json()


@pytest.mark.parametrize("path", ["GUIDE.md", "inputs.yaml", "options.toml"])
@pytest.mark.parametrize("damage", [None, "none", "rule", "side", "unsent", "foreign", "snapshot", "context"])
def test_independent_document_evidence(path, damage):
    """Verify bounded evidence and authorization without external calls."""
    case = {
        "path": path,
        "before": "description: delay",
        "after": "description: delay in seconds",
        "facts": ["The description specifies seconds."],
        "expected": "documentation",
    }
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Describe wording", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    sent = json.loads(user)
    raw = wire(case, selected).model_dump()
    if damage == "none":
        raw["observations"][0]["citation_ids"] = []
    if damage == "rule":
        selected = [c.model_copy(update={"kind": "rule", "source_role": "application_policy"}) for c in selected]
        for source in sent["sources"]:
            source.update(kind="rule", source_role="application_policy")
    if damage == "side":
        sent["sources"][0]["side"] = "RIGHT"
    if damage == "unsent":
        sent["sources"] = sent["sources"][1:]
    if damage in {"foreign", "snapshot"}:
        selected[0] = selected[0].model_copy(
            update={"citation_id": "diff:foreign"} if damage == "foreign" else {"snapshot_id": "old-snapshot"}
        )
    if damage == "context":
        for source in sent["sources"]:
            source["line_roles"] = ["context"]

    def validate():
        citation_line_roles([c for c in selected if c.kind == "diff"], facts, "synthetic-behavior")
        result = v2(raw)
        validate_output(result, selected)
        resolve_comparisons_v2(result, selected, sent)

    if damage:
        with pytest.raises((DomainError, ValidationError)):
            validate()
    else:
        validate()


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["id"])
def test_new_contract_retains_independent_code_expectations(case):
    """Verify bounded evidence and authorization without external calls."""
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Compare", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    result = v2(wire(case, selected).model_dump())
    resolve_comparisons_v2(result, selected, json.loads(user))
    public = ReviewOutput.model_validate(result.model_dump(exclude={"comparisons", "comparison_contract"}))
    assert canonical_hash(public.model_dump()) == canonical_hash(
        wire(case, selected).model_dump(exclude={"comparisons"})
    )
    if case["expected"] in {"behavior_added", "behavior_modified", "relocated"}:
        result.observations[0].citation_ids = [c.citation_id for c in selected if c.side == "RIGHT"]
        with pytest.raises(DomainError, match="requires_both_sides"):
            resolve_comparisons_v2(result, selected, json.loads(user))


def test_relocation_is_not_semantic_equivalence_certificate():
    """Verify bounded evidence and authorization without external calls."""
    case = cases()[4]
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Compare", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    result = v2(wire(case, selected).model_dump())
    result.comparisons[0].classification = "relocated"
    resolve_comparisons_v2(result, selected, json.loads(user))
    # A deliberately wrong classification can pass structural gates: the independent semantic oracle rejects it.
    assert case["expected"] == "behavior_modified"
    assert "if enabled and admin:" in case["after"] and "if enabled:" in case["before"]
    assert result.comparisons[0].classification != case["expected"]


@pytest.mark.parametrize("damage", ["partial", "no_edit", "documents_only"])
def test_behavior_still_requires_code_edits_and_complete_coverage(damage):
    """Resolving sides does not waive behavioral acquisition or implementation gates."""
    case = cases()[1]
    facts, citations, roles = evidence([case])
    user, selected = build_context(
        "Compare", [], [], citations, {"coverage": facts["coverage"], "index_complete": True}, roles
    )
    result = v2(wire(case, selected).model_dump())
    sent = json.loads(user)
    if damage == "partial":
        sent["SERVER_OBSERVATIONS"]["model_coverage_complete"] = False
    if damage == "no_edit":
        for source in sent["sources"]:
            source["line_roles"] = ["context"]
    if damage == "documents_only":
        selected = [c.model_copy(update={"path": "GUIDE.md"}) for c in selected]
        for source in sent["sources"]:
            source["path"] = "GUIDE.md"
    with pytest.raises(DomainError):
        resolve_comparisons_v2(result, selected, sent)
