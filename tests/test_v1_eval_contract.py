"""Dataset mutation and source/citation edge cases must fail independently."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.v1_backend.runner import Dataset
from app.services.releaseguard.normalization import canonical_hash
from app.services.v1_sources import document
from app.services.v1_rag import chunk_documents, fuse
from evals.v1_backend.review import HumanReview, aggregate, answer_contract
from scripts.v1_http_analysis_probe import load, save, summarize
from scripts.v1_review_packet import build


def test_frozen_dataset_rejects_duplicate_oracle_and_threshold_changes() -> None:
    """High averages cannot justify removing or altering a failing case."""
    original = json.loads(Path("evals/datasets/releaseguard_rag_v1.json").read_text(encoding="utf-8"))
    assert len(Dataset.model_validate(original).cases) == 66
    for change in ("duplicate", "threshold", "oracle"):
        data = json.loads(json.dumps(original))
        if change == "duplicate":
            data["cases"][1]["case_id"] = data["cases"][0]["case_id"]
        elif change == "threshold":
            data["thresholds"]["safety_each_case"] = 0.9
        else:
            data["cases"][0]["expected_analysis_calls"] = 0
        with pytest.raises(ValueError):
            Dataset.model_validate(data)


def test_long_unicode_paragraph_chunks_resolve_to_exact_source() -> None:
    """No replacement characters, rewritten whitespace or ambiguous text locations."""
    content = "# Heading\n\n" + ("证据和权限 🔐 不授予执行权。\n" * 120) + "\n\n# Next\n\n" + ("é审批\n" * 120)
    source = document(
        "markdown-rules", "synthetic.md", "version1", content, "knowledge:synthetic.md", datetime.now(UTC)
    )
    chunks = chunk_documents([source], "generation:test")
    assert len(chunks) > 2
    for chunk in chunks:
        assert content[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.tokens <= 512 and chunk.text_hash == canonical_hash(chunk.text)
        assert chunk.locator.endswith(f"#chars={chunk.start_char}-{chunk.end_char}")
    assert len(fuse(chunks, chunks)) == 1


def test_long_topic_heading_keeps_relevant_rule_in_one_chunk() -> None:
    """Metadata paragraphs may not separate a heading from its short rules."""
    content = (
        "# Upstream reference\n\n"
        + "Versions and runtime prerequisites. " * 75
        + "\n\n## Local simulation rule\n\n"
        + "Only validate_inputs, simulate, verify_predictions are candidate identifiers. " * 4
        + "\n\n"
        + "Bilingual reference context. " * 80
    )
    source = document("markdown-rules", "synthetic-heading.md", "v1", content, "synthetic", datetime.now(UTC))
    chunks = chunk_documents([source], "heading-generation")
    matching = [c for c in chunks if "## Local simulation rule" in c.text]
    assert matching and "verify_predictions" in matching[-1].text
    assert all(c.tokens <= 512 and c.text == content[c.start_char : c.end_char] for c in chunks)
    assert all(right.start_char < left.end_char for left, right in zip(chunks, chunks[1:], strict=False))


def test_human_review_cannot_pass_missing_ratings_or_changed_outputs() -> None:
    """No missing review, failed output or mismatched sample may average away."""
    sample = {"case_id": "normal", "attempt": 1, "status": "passed", "expected": "grounded", "output": {"claims": [1]}}
    provider = {"dataset_hash": "frozen", "rows": [sample]}
    values = {
        "schema": "v1-human-review.1",
        "dataset_hash": "frozen",
        "output_hash": canonical_hash(provider["rows"]),
        "status": "unverified",
        "rows": [{"case_id": "normal", "attempt": 1, "total_claims": 1}],
    }
    assert aggregate(provider, HumanReview.model_validate(values))["status"] == "unverified"
    values["status"] = "reviewed"
    values["rows"][0].update(human_rating=4, grounded_claim_count=1, reviewer_reference="synthetic-review-test")
    assert not aggregate(provider, HumanReview.model_validate(values))["passed"]
    assert aggregate(provider, HumanReview.model_validate(values))["human_rating_mean"] == 4
    values["rows"][0]["grounded_claim_count"] = 0
    assert not aggregate(provider, HumanReview.model_validate(values))["passed"]
    provider["rows"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="binding_mismatch"):
        aggregate(provider, HumanReview.model_validate(values))
    values["output_hash"] = canonical_hash(provider["rows"])
    assert aggregate(provider, HumanReview.model_validate(values))["human_rating_mean"] == 0
    values["rows"].append(values["rows"][0])
    with pytest.raises(ValueError, match="case_set"):
        aggregate(provider, HumanReview.model_validate(values))


def test_rule_question_unnecessary_unknown_is_not_correct_refusal() -> None:
    """Having supported claims alone does not satisfy a known policy answer."""
    sample = {
        "status": "passed",
        "expected": "grounded",
        "output": {"risk": "UNKNOWN", "unknowns": ["name absent"], "claims": [1]},
    }
    assert not answer_contract(sample)
    sample["output"].update(risk="HIGH", unknowns=[])
    assert answer_contract(sample)
    sample["output"]["risk"] = "UNKNOWN"
    assert answer_contract(sample)  # Domain risk is not an answer-quality label.
    sample["expected"] = "unknown"
    assert not answer_contract(sample)
    sample["output"].update(risk="UNKNOWN", unknowns=["instance absent"])
    assert answer_contract(sample)
    sample["status"] = "failed"
    assert not answer_contract(sample)


def test_performance_resume_binding_unknown_and_failure_denominator(tmp_path: Path) -> None:
    """A lost timing, duplicate paid slot or quota rejection cannot pass by average."""
    dataset = {"cases": [{"case_id": "case"}]}
    state = {"source_hash": "frozen"}
    path = tmp_path / "ledger.json"
    report = load(path, dataset, state)
    report["rows"] = [
        {"case_id": "case", "attempt": a, "latency_seconds": 1, "model_attempts": 1, "verified_provider_attempt": True}
        for a in (1, 2)
    ]
    summarize(report)
    assert report["status"] == "passed"
    report["rows"].append({"case_id": "case", "attempt": 1, "latency_seconds": 90, "model_attempts": 0})
    summarize(report)
    assert report["status"] == "failed" and report["p95_seconds_including_failures"] == 90
    save(path, report)
    assert len(load(path, dataset, state)["rows"]) == 3
    with pytest.raises(ValueError, match="binding_mismatch"):
        load(path, dataset, {"source_hash": "changed"})
    report["rows"][-1]["model_attempts"] = 1
    save(path, report)
    with pytest.raises(ValueError, match="duplicate_paid"):
        load(path, dataset, state)
    report["rows"] = report["rows"][:2]
    report["rows"][1]["latency_seconds"] = None
    summarize(report)
    assert report["status"] == "unverified"
    report["rows"][1]["latency_seconds"] = 1
    report["rows"][1]["verified_provider_attempt"] = False
    summarize(report)
    assert report["status"] == "unverified"


def test_human_form_is_exact_bound_unscored_and_escapes_source_scripts() -> None:
    """No source can execute scripts, and export begins with null human scores."""
    dataset = {"cases": [{"case_id": "case", "query": "rule"}]}
    provider = {
        "dataset_hash": canonical_hash(dataset),
        "rows": [
            {"case_id": "case", "attempt": 1, "status": "passed", "expected": "grounded", "output": {"claims": []}}
        ],
        "review_sources": [{"chunk_id": "chunk", "text": "</script><script>unsafe</script>"}],
    }
    template, form, _ = build(provider, dataset)
    assert template["output_hash"] == canonical_hash(provider["rows"])
    assert template["rows"][0]["human_rating"] is None and template["status"] == "unverified"
    assert "</script><script>unsafe" not in form and "connect-src 'none'" in form


def test_unknown_supporting_claims_cannot_hide_normal_grounding_failure() -> None:
    """Five correct refusals cannot average one wholly ungrounded normal case away."""
    samples = []
    ratings = []
    for case in range(20):
        for attempt in (1, 2):
            unknown = case >= 15
            samples.append(
                {
                    "case_id": str(case),
                    "attempt": attempt,
                    "status": "passed",
                    "expected": "unknown" if unknown else "grounded",
                    "citation_correct": True,
                    "output": {
                        "claims": [1],
                        "risk": "UNKNOWN" if unknown else "HIGH",
                        "unknowns": ["missing instance"] if unknown else [],
                    },
                }
            )
            ratings.append(
                {
                    "case_id": str(case),
                    "attempt": attempt,
                    "total_claims": 1,
                    "human_rating": 5,
                    "grounded_claim_count": 0 if case == 0 else 1,
                    "reviewer_reference": "synthetic-test",
                }
            )
    provider = {"dataset_hash": "frozen", "rows": samples}
    review = HumanReview.model_validate(
        {
            "schema": "v1-human-review.1",
            "dataset_hash": "frozen",
            "output_hash": canonical_hash(samples),
            "status": "reviewed",
            "rows": ratings,
        }
    )
    result = aggregate(provider, review)
    assert not result["passed"] and result["normal_cases"] == 15
    assert result["grounded_claim_rate_case_macro"] == pytest.approx(14 / 15)
    assert result["supporting_claim_rate_including_unknown_cases"] == 0.95


@pytest.mark.parametrize("unknown_correct", [False, True])
def test_unknown_subset_cannot_be_hidden_by_normal_answer_average(unknown_correct: bool) -> None:
    """A 38/40 aggregate cannot pass when the two unknown samples both fail."""
    samples, ratings = [], []
    for case in range(20):
        for attempt in (1, 2):
            unknown = case == 19
            samples.append(
                {
                    "case_id": str(case),
                    "attempt": attempt,
                    "status": "passed",
                    "expected": "unknown" if unknown else "grounded",
                    "citation_correct": True,
                    "output": {
                        "claims": [1],
                        "risk": "UNKNOWN" if unknown and unknown_correct else "HIGH",
                        "unknowns": ["missing empirical observation"] if unknown and unknown_correct else [],
                    },
                }
            )
            ratings.append(
                {
                    "case_id": str(case),
                    "attempt": attempt,
                    "total_claims": 1,
                    "human_rating": 5,
                    "grounded_claim_count": 1,
                    "reviewer_reference": "synthetic-test",
                }
            )
    provider = {"dataset_hash": "frozen", "rows": samples}
    review = HumanReview.model_validate(
        {
            "schema": "v1-human-review.1",
            "dataset_hash": "frozen",
            "output_hash": canonical_hash(samples),
            "status": "reviewed",
            "rows": ratings,
        }
    )
    result = aggregate(provider, review)
    assert result["answer_contract_rate"] == (1 if unknown_correct else 0.95)
    assert result["unknown_correctness_case_macro"] == int(unknown_correct)
    assert result["unknown_cases"] == 1
    assert result["passed"] is unknown_correct


def test_delegated_review_requires_provenance_and_explicit_acceptance() -> None:
    """Delegation never becomes independent human evidence or lowers thresholds."""
    samples, ratings = [], []
    for case in range(20):
        for attempt in (1, 2):
            unknown = case == 19
            samples.append(
                {
                    "case_id": str(case),
                    "attempt": attempt,
                    "status": "passed",
                    "expected": "unknown" if unknown else "grounded",
                    "citation_correct": True,
                    "output": {
                        "claims": [1],
                        "risk": "UNKNOWN" if unknown else "HIGH",
                        "unknowns": ["missing observation"] if unknown else [],
                    },
                }
            )
            ratings.append(
                {
                    "case_id": str(case),
                    "attempt": attempt,
                    "total_claims": 1,
                    "human_rating": 4,
                    "grounded_claim_count": 1,
                    "reviewer_reference": "delegated-test",
                }
            )
    provider = {"dataset_hash": "frozen", "rows": samples}
    values = {
        "schema": "v1-human-review.1",
        "review_method": "assistant_delegated",
        "dataset_hash": "frozen",
        "output_hash": canonical_hash(samples),
        "status": "reviewed",
        "rows": ratings,
    }
    with pytest.raises(ValueError, match="explicit_provenance"):
        HumanReview.model_validate(values)
    values.update(schema="v1-semantic-review.2", delegation_reference="explicit-user-request")
    review = HumanReview.model_validate(values)
    result = aggregate(provider, review)
    assert result["quality_thresholds_met"] and result["status"] == "unverified"
    assert not result["passed"] and not result["independent_human_reviewed"]
    result = aggregate(provider, review, allow_delegated_review=True)
    assert result["passed"] and result["review_method"] == "assistant_delegated"
    assert not result["independent_human_reviewed"] and result["explanation_rating_mean"] == 4
    review.rows[0].human_rating = 2
    assert not aggregate(provider, review, allow_delegated_review=True)["passed"]
    values["review_method"] = "independent_human"
    with pytest.raises(ValueError, match="cannot_have_delegation"):
        HumanReview.model_validate(values)
