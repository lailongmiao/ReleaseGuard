"""Fail closed on missing samples, forged review coverage and empty evidence."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from tenacity import wait_none

from app.services.releaseguard.normalization import canonical_hash
from evals.v1_backend.enhancement import AnalysisDataset, RetrievalDataset, summarize_retrieval
from evals.v1_backend.enhancement_review import REGRESSION_CASES, SemanticReview, summarize
from scripts.mvp_product_probe import ProductProbe
from scripts.v1_http_analysis_probe import quota, save


@pytest.mark.parametrize(
    "failure", [subprocess.CalledProcessError(1, "synthetic"), subprocess.TimeoutExpired("synthetic", 25)]
)
def test_quota_retry_is_bounded_read_only_and_never_repeats_analysis(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A transient quota read failure never re-enters the paid request path."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["timeout"] == 25 and kwargs["check"] is True
        assert "rate_limit" in command[-1] and "httpx.get" in command[-1]
        assert "analyze" not in command[-1] and "httpx.post" not in command[-1]
        if len(calls) == 1:
            raise failure
        return subprocess.CompletedProcess(command, 0, '{"status":200,"remaining":42}', "")

    monkeypatch.setattr(subprocess, "run", run)
    probe = cast(ProductProbe, SimpleNamespace(args=["docker", "compose"]))
    assert quota.retry_with(wait=wait_none())(probe) == {"status": 200, "remaining": 42}
    assert len(calls) == 2


def test_quota_persistent_failure_stops_after_three_read_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persistent subprocess failure has a finite retry boundary."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    probe = cast(ProductProbe, SimpleNamespace(args=["docker", "compose"]))
    with pytest.raises(subprocess.CalledProcessError):
        quota.retry_with(wait=wait_none())(probe)
    assert len(calls) == 3


def test_quota_invalid_json_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid response is not mistaken for a transient transport error."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "invalid-json", "")

    monkeypatch.setattr(subprocess, "run", run)
    probe = cast(ProductProbe, SimpleNamespace(args=["docker", "compose"]))
    with pytest.raises(ValueError):
        quota.retry_with(wait=wait_none())(probe)
    assert len(calls) == 1


def test_report_file_share_conflict_retries_only_local_atomic_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows readers can briefly deny replacement; no paid operation is repeated."""
    original = Path.replace
    attempts = []

    def replace(path: Path, target: Path) -> Path:
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("synthetic local file sharing conflict")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    destination = tmp_path / "report.json"
    report = {"status": "in_flight", "command_id": "synthetic-unknown-no-resend"}
    save(destination, report)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert len(attempts) == 2 and not destination.with_suffix(".json.tmp").exists()


def test_corpus_revision_keeps_frozen_questions_and_gold() -> None:
    """A post-observation corpus repair is regression, never a changed oracle."""
    for kind, schema in (("rag", RetrievalDataset), ("analysis", AnalysisDataset)):
        original = schema.model_validate_json(
            Path(f"evals/datasets/v1_enhancement_{kind}.json").read_text(encoding="utf-8")
        )
        revised = schema.model_validate_json(
            Path(f"evals/datasets/v1_enhancement_{kind}_corpus3.json").read_text(encoding="utf-8")
        )
        assert revised.cases == original.cases
        assert revised.corpus_manifest_hash != original.corpus_manifest_hash
        assert revised.version.endswith(".2")


def reviewed() -> tuple[dict, dict, dict]:
    """Synthetic evaluator inputs, explicitly not real quality evidence."""
    dataset = json.loads(Path("evals/datasets/v1_enhancement_analysis.json").read_text(encoding="utf-8"))
    rows, ratings = [], []
    for case in dataset["cases"]:
        for attempt in (1, 2):
            unknown = case["expected"] == "unknown"
            rows.append(
                {
                    **case,
                    "attempt": attempt,
                    "status": "passed",
                    "citation_correct": True,
                    "review_sources": [{"chunk_id": "synthetic-citation"}],
                    "output": {
                        "explanation": "Synthetic assertion.",
                        "claims": [{"text": "Synthetic assertion.", "citation_ids": ["synthetic-citation"]}],
                        "unknowns": ["synthetic missing observation"] if unknown else [],
                        "risk": "UNKNOWN" if unknown else "HIGH",
                        "candidate_steps": [],
                    },
                }
            )
            ratings.append(
                {
                    "case_id": case["case_id"],
                    "attempt": attempt,
                    "human_rating": 5,
                    "grounded_claim_count": 1,
                    "total_claims": 1,
                    "reviewer_reference": "synthetic-evaluator-fixture",
                    "body_assertions": [
                        {
                            "text": "Synthetic assertion.",
                            "start_char": 0,
                            "end_char": 20,
                            "supported": True,
                            "rationale": "synthetic supplied fact",
                            "citation_ids": ["synthetic-citation"],
                        }
                    ],
                    "safety_violation": False,
                    "explanation_note": "synthetic fixture only",
                }
            )
    provider = {"rows": rows, "planned": 80, "dataset_hash": canonical_hash(dataset)}
    review = {
        "schema": "v1-enhancement-semantic-review.1",
        "review_method": "assistant_delegated",
        "delegation_reference": "synthetic-test",
        "dataset_hash": provider["dataset_hash"],
        "output_hash": canonical_hash(rows),
        "status": "reviewed",
        "rows": ratings,
        "known_regressions": {
            key: [
                {"case_id": case, "attempt": attempt, "status": "passed", "rationale": "synthetic regression fixture"}
                for case in sorted(cases)
                for attempt in (1, 2)
            ]
            for key, cases in REGRESSION_CASES.items()
        },
    }
    return provider, review, dataset


def test_body_omission_or_forged_text_cannot_pass() -> None:
    """Perfect claims cannot conceal an omitted/altered explanation."""
    provider, review, dataset = reviewed()
    assert summarize(provider, SemanticReview.model_validate(review), dataset)["passed"]
    for replacement in ([], [{**review["rows"][0]["body_assertions"][0], "text": "Invented assertion."}]):
        changed = deepcopy(review)
        changed["rows"][0]["body_assertions"] = replacement
        with pytest.raises(ValueError, match="body_review"):
            summarize(provider, SemanticReview.model_validate(changed), dataset)


def test_incorrect_case_set_and_input_cannot_pass() -> None:
    """80 rows are insufficient without exact frozen cases and inputs."""
    provider, review, dataset = reviewed()
    provider["rows"][0]["query"] = "different easier query"
    review["output_hash"] = canonical_hash(provider["rows"])
    with pytest.raises(ValueError, match="input_mismatch"):
        summarize(provider, SemanticReview.model_validate(review), dataset)
    provider["rows"][0]["attempt"] = 4
    with pytest.raises(ValueError, match="case_set_mismatch"):
        summarize(provider, SemanticReview.model_validate(review), dataset)

    dataset["cases"][-1] = deepcopy(dataset["cases"][0])
    with pytest.raises(ValueError, match="dataset_counts_invalid"):
        AnalysisDataset.model_validate(dataset)


def test_safety_known_regression_and_missing_review_fail_closed() -> None:
    """A high aggregate never masks one violation or unreviewed regression."""
    provider, review, dataset = reviewed()
    review["rows"][0]["safety_violation"] = True
    assert not summarize(provider, SemanticReview.model_validate(review), dataset)["passed"]
    review["rows"][0]["safety_violation"] = False
    review["known_regressions"]["V1-IR-01"] = []
    assert not summarize(provider, SemanticReview.model_validate(review), dataset)["passed"]
    review["rows"][0]["human_rating"] = None
    assert summarize(provider, SemanticReview.model_validate(review), dataset)["status"] == "unverified"


def test_retrieval_empty_failed_and_duplicate_slots() -> None:
    """Neither no rows nor duplicate easy questions constitute valid recall."""
    dataset = RetrievalDataset.model_validate_json(
        Path("evals/datasets/v1_enhancement_rag.json").read_text(encoding="utf-8")
    )
    assert summarize_retrieval([], dataset, 2)["status"] == "unverified"
    rows = [
        {
            "case_id": c.case_id,
            "attempt": a,
            "status": "failed",
            "resources": c.gold_resources,
            "citations_valid": True,
        }
        for c in dataset.cases
        for a in (1, 2)
    ]
    assert summarize_retrieval(rows, dataset, 2)["recall_at_5_macro"] == 0
    with pytest.raises(ValueError, match="slots_invalid"):
        summarize_retrieval(rows + [rows[0]], dataset, 2)
