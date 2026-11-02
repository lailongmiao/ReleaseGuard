"""Evaluate frozen V1.1 positions without changing prior failures or filling missing slots."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from rich.console import Console

from app.schemas.v11_backend import ReviewOutput
from app.services.releaseguard.normalization import canonical_hash
from evals.v11_backend.contracts import Dataset, SemanticReview


def evaluate(report: dict, review: SemanticReview) -> dict:
    """Missing or duplicate samples fail the batch; security cannot be averaged away."""
    dataset = Dataset.model_validate(report["dataset"])
    if review.report_hash != canonical_hash(report):
        raise ValueError("review_report_hash_mismatch")
    expected = {(case.case_id, repeat) for case in dataset.cases for repeat in (1, 2)}
    samples = {(item["case_id"], item["repeat"]): item for item in report["positions"]}
    scores = {(item.case_id, item.repeat): item for item in review.positions}
    denominator_valid = (
        len(report["positions"]) == len(samples) == 80
        and set(samples) == expected
        and len(review.positions) == len(scores) == 80
        and set(scores) == expected
    )
    if not denominator_valid:
        return {
            "passed": False,
            "reason": "frozen_denominator_incomplete",
            "expected_positions": 80,
            "actual_positions": len(report["positions"]),
            "reviewed_positions": len(review.positions),
        }
    cases = {case.case_id: case for case in dataset.cases}
    structure = []
    hard = []
    recalls = []
    regular_times, large_times = [], []
    for key, sample in samples.items():
        judgment = scores[key]
        if judgment.output_hash != canonical_hash(sample.get("analysis")) or judgment.input_hash != canonical_hash(
            sample.get("input")
        ):
            raise ValueError("semantic_position_hash_mismatch")
        try:
            ReviewOutput.model_validate(sample["analysis"]["output"])
            valid = sample["status"] == "completed"
        except (ValueError, KeyError, TypeError):
            valid = False
        structure.append(valid)
        hard.append(
            not judgment.critical_error
            and judgment.citation_identity_valid
            and sample.get("hard_gates_passed") is True
        )
        gold = set(cases[key[0]].expected_documents)
        recalls.append(
            len(gold & set(sample.get("retrieved_documents", [])[:5])) / len(gold)
            if sample.get("rag_status") == "completed"
            else 0.0
        )
        elapsed = sample.get("elapsed_seconds", 120.0)
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid_latency")
        (regular_times if cases[key[0]].group == "regular" else large_times).append(elapsed)
    ordinary = [value for key, value in scores.items() if not cases[key[0]].expected_unknown]
    unknowns = [value for key, value in scores.items() if cases[key[0]].expected_unknown]
    p95 = sorted(regular_times)[math.ceil(0.95 * len(regular_times)) - 1]
    metrics = {
        "valid_structure": sum(structure),
        "hard_gates_passed": all(hard),
        "explanation_mean": mean(item.score for item in scores.values()),
        "score_at_least_three": mean(item.score >= 3 for item in scores.values()),
        "ordinary_claims_supported_macro": mean(item.claims_supported for item in ordinary),
        "ordinary_body_supported_macro": mean(item.body_supported for item in ordinary),
        "unknown_correct_macro": mean(item.unknown_correct for item in unknowns),
        "recall_at_5": mean(recalls),
        "regular_http_p95_seconds": p95,
        "regular_positions": len(regular_times),
        "large_positions": len(large_times),
        "large_max_seconds": max(large_times),
    }
    passed = (
        all(structure)
        and all(hard)
        and metrics["explanation_mean"] >= 4
        and metrics["score_at_least_three"] >= 0.95
        and metrics["ordinary_claims_supported_macro"] >= 0.95
        and metrics["ordinary_body_supported_macro"] >= 0.95
        and metrics["unknown_correct_macro"] >= 0.95
        and metrics["recall_at_5"] >= 0.85
        and p95 <= 30
        and max(large_times) <= 120
    )
    return {
        "version": "v11-quality.1",
        "passed": passed,
        "metrics": metrics,
        "denominators": {"all": 80, "ordinary": 60, "unknown": 20, "rag": 80},
        "review_method": review.review_method,
        "independent_human_reviewed": False,
        "legacy_quality_result": "unchanged",
    }


def main() -> int:
    """A separate review file is mandatory; the business model cannot grade itself."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["review", "validate-dataset"])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.report.read_text(encoding="utf-8"))
    if args.command == "validate-dataset":
        dataset = Dataset.model_validate(raw)
        Console().print({"cases": len(dataset.cases), "hash": canonical_hash(dataset), "model_calls": 0})
        return 0
    if args.review is None or args.output is None or args.output.exists():
        raise ValueError("new_output_and_independent_review_required")
    result = evaluate(raw, SemanticReview.model_validate_json(args.review.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Console().print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
