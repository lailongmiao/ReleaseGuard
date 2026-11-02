"""Evidence takes priority over optional references and redundant navigation metadata."""

import json
from pathlib import Path

import pytest

from app.schemas.v11_backend import ReviewCitation
from app.services.v11_policy import citation_line_roles, token_count
from app.services.v11_provider import SYSTEM, build_context
from evals.v11_backend.behavior_cases import evidence


def test_optional_reference_cannot_evict_paired_code() -> None:
    """Independent renamed code retains both sides with or without oversized optional prose."""
    facts, citations, roles = evidence([{"path": "z/normalizer.py", "before": "limit = 8", "after": "limit = 16"}])
    rule = citations[0].model_copy(
        update={
            "kind": "rule",
            "source_role": "usage_reference",
            "citation_id": "rule:large",
            "text": "Optional historical usage reference. " * 1200,
        }
    )
    user, selected = build_context("Compare defaults", [], [rule], citations, {"coverage": facts["coverage"]}, roles)
    assert {c.citation_id for c in selected} == {c.citation_id for c in citations}
    assert json.loads(user)["SERVER_OBSERVATIONS"]["model_coverage_complete"]
    assert len((SYSTEM + user).encode()) + 64 <= 32768


def test_archived_product_selection_preserves_all_evidence_under_original_limits() -> None:
    """Read-only archived product input proves selection behavior, never model semantics."""
    source = Path("evals/reports/v11_eval_candidate_archived_inputs.json")
    prior = Path("evals/reports/v11_eval_candidate_frozen_payload.json")
    if not source.exists() or not prior.exists():
        pytest.skip("local historical artifacts not distributed")
    raw = json.loads(source.read_text("utf-8"))
    previous = json.loads(prior.read_text("utf-8"))
    sent = json.loads(previous["user"])
    retained = [ReviewCitation.model_validate(c) for c in previous["citations"]]
    diffs = [ReviewCitation.model_validate(c) for c in raw["index_config"]["citations"]]
    mandatory = [c for c in retained if c.kind == "ci" or c.source_role == "application_policy"]
    references = [c for c in retained if c.kind == "rule" and c.source_role != "application_policy"]
    user, selected = build_context(
        sent["query"],
        mandatory,
        references,
        diffs,
        sent["SERVER_OBSERVATIONS"],
        citation_line_roles(diffs, raw["facts"], raw["object_id"]),
    )
    ids = {c.citation_id for c in selected}
    assert {c.citation_id for c in [*mandatory, *diffs]} <= ids
    assert json.loads(user)["SERVER_OBSERVATIONS"]["model_coverage_complete"]
    assert len((SYSTEM + user).encode()) + 64 <= 32768 and token_count(SYSTEM + user) + 64 <= 12288
    assert sum(token_count(c.text) for c in selected if c.kind == "diff") <= 2048
    assert sum(len(c.text.encode()) for c in selected if c.kind == "diff") <= 8192
