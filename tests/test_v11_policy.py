"""Coverage, rule-role isolation, citation integrity and deterministic eligibility."""

from datetime import UTC, datetime
import json

import pytest
from pydantic import ValidationError

from app.schemas.v11_backend import CIObservation, Coverage, ReviewCitation, ReviewOutput
from app.schemas.releaseguard import RiskLevel
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_diff import read_file
from app.services.v11_policy import (
    CAPABILITIES,
    citation_line_roles,
    diff_chunks,
    load_rules,
    qualification,
    validate_output,
)
from app.services.v11_provider import build_context
from test_v11_provider import citation


def output() -> ReviewOutput:
    """A supported suggestion is still not application authorization."""
    return ReviewOutput.model_validate(
        {
            "summary": "Line changed.",
            "observations": [{"text": "Line changed.", "kind": "observation", "citation_ids": ["diff:fixture"]}],
            "risks": [],
            "validation_suggestions": [],
            "unknowns": [],
            "candidate_steps": ["simulate"],
            "model_risk": "LOW",
        }
    )


def test_model_line_roles_preserve_unchanged_neighbor_and_changed_value() -> None:
    """Repeated text and different coordinates cannot turn an unchanged line into an edit."""
    file = read_file(
        {
            "filename": "配置.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": "@@ -7,3 +11,3 @@\n limit = 10\n-timeout = 10\n+timeout = 20\n retries = 10\n",
        },
        32768,
    )
    facts = {
        "comparison": {
            "kind": "commit",
            "base_kind": "commit",
            "head_sha": "b" * 40,
            "base_tip_sha": "a" * 40,
            "diff_base_sha": "a" * 40,
            "parent_sha": "a" * 40,
            "comparison": "parent",
        },
        "files": [file.model_dump(mode="json")],
    }
    _, citations, _ = diff_chunks("object:x", "snapshot:x", "generation:x", "source:x", facts, datetime.now(UTC))
    roles = citation_line_roles(citations, facts)
    user, _ = build_context("Which setting changed?", [], [], citations, line_roles=roles)
    sources = json.loads(user)["sources"]
    assert sources[0]["line_roles"] == ["context", "delete", "context"]
    assert sources[1]["line_roles"] == ["context", "add", "context"]
    assert sources[0]["start_line"] == 7 and sources[1]["start_line"] == 11
    corrupted = citations[1].model_copy(update={"text": "invented = 20"})
    with pytest.raises(DomainError, match="diff_citation_position_invalid"):
        citation_line_roles([corrupted], facts)


@pytest.mark.parametrize(
    "repository,kind,complete,ci,eligible",
    [
        ("checkout", "pr", True, True, True),
        ("setup-python", "pr", True, True, False),
        ("checkout", "commit", True, True, False),
        ("checkout", "branch_compare", True, True, False),
        ("checkout", "pr", False, True, False),
        ("checkout", "pr", True, False, False),
    ],
)
def test_qualification_orthogonal_to_output(
    repository: str, kind: str, complete: bool, ci: bool, eligible: bool
) -> None:
    """No successful report can turn partial coverage or failed CI into approval."""
    coverage = Coverage(
        manifest_completeness="complete",
        content_completeness="complete",
        analysis_coverage="complete" if complete else "partial",
    )
    observed = CIObservation(
        head_sha="a" * 40,
        status="success" if ci else "failed",
        complete=True,
        checks_passed=ci,
        captured_at=datetime.now(UTC),
    )
    result = qualification(repository, kind, coverage, observed, output(), [])
    assert result.eligible == eligible
    assert CAPABILITIES.pause_reason is None
    assert not qualification(repository, kind, coverage, observed, output(), ["snapshot_expired"]).eligible


def test_rules_never_cross_repository_or_claim_code() -> None:
    """Setup-python receives application rules but no checkout usage references."""
    _, checkout = load_rules("checkout")
    _, python = load_rules("setup-python")
    assert len(checkout) == 12 and len(python) == 3
    assert all(entry["role"] == "application_policy" for entry, _ in python)
    rule = citation().model_copy(update={"kind": "rule", "source_role": "application_policy"})
    with pytest.raises(DomainError, match="analysis_observation_requires_diff"):
        validate_output(output(), [rule])


def test_citation_hash_and_cross_input_rejection() -> None:
    """Valid-looking coordinates alone never validate absent or changed content."""
    value = citation().model_dump()
    value["text"] = "changed after capture"
    with pytest.raises(ValidationError, match="citation_hash_invalid"):
        ReviewCitation.model_validate(value)
    with pytest.raises(DomainError, match="analysis_citation_invalid"):
        validate_output(output(), [citation().model_copy(update={"citation_id": "other-object"})])


def test_unknown_candidates_rejected_at_adoption_without_rewriting_history() -> None:
    """Historical report parsing remains possible; newly actionable UNKNOWN is rejected."""
    report = output().model_copy(update={"model_risk": RiskLevel.UNKNOWN})
    report = ReviewOutput.model_validate(report.model_dump())
    with pytest.raises(DomainError, match="analysis_unknown_forbids_candidate_steps"):
        validate_output(report, [citation()])


def test_context_carries_known_identity_and_actual_model_coverage() -> None:
    """Model input must retain source SHA and distinguish acquired from selected coverage."""
    source = citation()
    text, _ = build_context("Explain", [], [], [source], {"ci": {"status": "failed"}})
    sent = json.loads(text)
    assert sent["SERVER_OBSERVATIONS"]["ci"]["status"] == "failed"
    assert sent["SERVER_OBSERVATIONS"]["model_coverage_complete"] is True
    assert sent["sources"][0]["sha"] == source.sha
    assert sent["sources"][0]["side"] == "RIGHT"


def test_diff_index_exact_sides_and_object_scope() -> None:
    """Same upstream content in separate review objects cannot share citation authority."""
    file = read_file(
        {"filename": "x.py", "status": "modified", "additions": 1, "deletions": 1, "patch": "@@ -3 +3 @@\n-old\n+new"},
        1000,
    )
    facts = {
        "comparison": {
            "kind": "commit",
            "comparison": "parent",
            "head_sha": "a" * 40,
            "base_tip_sha": "b" * 40,
            "parent_sha": "b" * 40,
            "diff_base_sha": "b" * 40,
        },
        "files": [file.model_dump()],
    }
    _, references, complete = diff_chunks(
        "object-a", "snapshot-a", "generation-a", "review-github-checkout", facts, datetime.now(UTC)
    )
    assert complete and [(c.side, c.sha, c.start_line, c.text) for c in references] == [
        ("LEFT", "b" * 40, 3, "old"),
        ("RIGHT", "a" * 40, 3, "new"),
    ]
    _, other, _ = diff_chunks(
        "object-b", "snapshot-b", "generation-b", "review-github-checkout", facts, datetime.now(UTC)
    )
    assert {c.citation_id for c in references}.isdisjoint(c.citation_id for c in other)


def test_only_blank_added_line_is_not_a_model_input() -> None:
    """Preserve the file fact while refusing an empty embedding/model fragment."""
    file = read_file(
        {"filename": "github", "status": "added", "additions": 1, "deletions": 0, "patch": "@@ -0,0 +1 @@\n+"}, 1000
    )
    facts = {
        "comparison": {"kind": "commit", "comparison": "empty_tree", "base_kind": "empty_tree", "head_sha": "a" * 40},
        "files": [file.model_dump()],
    }
    chunks, citations, complete = diff_chunks("o", "s", "g", "review-github-checkout", facts, datetime.now(UTC))
    assert not chunks and not citations and not complete


def test_model_context_never_drops_mandatory_rules() -> None:
    """Huge mandatory context blocks rather than silently dropping safety policy."""
    rule = ReviewCitation(
        citation_id="rule:test",
        kind="rule",
        source_role="application_policy",
        snapshot_id="snapshot:test",
        generation_id="generation:test",
        text="a" * 40000,
        text_hash=canonical_hash("a" * 40000),
        locator="knowledge:synthetic",
    )
    with pytest.raises(DomainError, match="mandatory_rule_context_limit"):
        build_context("query", [rule], [], [citation()])
    with pytest.raises(DomainError, match="unsupported_content"):
        build_context("query", [], [], [])
