"""Explicit delegated review of claims AND explanation assertions, never a Judge."""

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from evals.v1_backend.review import HumanReview, ReviewRow, aggregate
from app.services.releaseguard.normalization import canonical_hash
from app.schemas.v1_backend import AnalysisOutput
from evals.v1_backend.enhancement import AnalysisDataset

REGRESSION_CASES = {
    "V1-IR-01": {"holdout-03"},
    "V1-IR-02": {"holdout-15"},
    "V1-IR-03": {"holdout-17"},
    "V1-IR-04": {"holdout-17"},
    "V1-IR-05": {"holdout-18"},
    "V1-IR-06": {"holdout-01", "holdout-07", "holdout-10", "holdout-12"},
    "V1-IR-07": {"holdout-04", "holdout-14"},
}


class BodyAssertion(BaseModel):
    """A reviewable explanation assertion with a source-based rationale."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    supported: bool
    rationale: str = Field(min_length=1)
    citation_ids: list[str]
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)


class SemanticRow(ReviewRow):
    """Extra review fields never change persisted product analysis."""

    body_assertions: list[BodyAssertion]
    safety_violation: bool
    explanation_note: str = Field(min_length=1)


class SemanticReview(BaseModel):
    """Delegate provenance and input hashes are mandatory, not inferred."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["v1-enhancement-semantic-review.1"] = Field(alias="schema")
    review_method: Literal["assistant_delegated"]
    delegation_reference: str = Field(min_length=1)
    dataset_hash: str
    output_hash: str
    status: Literal["reviewed", "unverified"]
    rows: list[SemanticRow]
    known_regressions: dict[str, list["RegressionVerdict"]]


class RegressionVerdict(BaseModel):
    """Every known regression is tied to both frozen repetitions and evidence."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    attempt: int = Field(ge=1, le=2)
    status: Literal["passed", "failed", "unverified"]
    rationale: str = Field(min_length=1)


def summarize(provider: dict, review: SemanticReview, dataset: dict) -> dict:
    """Missing/failed outputs cannot disappear from quality denominators."""
    AnalysisDataset.model_validate(dataset)
    expected = {(c["case_id"], a) for c in dataset["cases"] for a in (1, 2)}
    samples = {(r["case_id"], r["attempt"]): r for r in provider["rows"]}
    if len(expected) != 80 or provider["dataset_hash"] != canonical_hash(dataset) or not set(samples) <= expected:
        raise ValueError("frozen_analysis_case_set_mismatch")
    cases = {c["case_id"]: c for c in dataset["cases"]}
    for sample in provider["rows"]:
        case = cases[sample["case_id"]]
        if any(sample.get(key) != case.get(key) for key in ("query", "expected", "language")):
            raise ValueError("frozen_analysis_input_mismatch")
        if sample["status"] == "passed":
            AnalysisOutput.model_validate(sample["output"])
    legacy = HumanReview.model_validate(
        {
            "schema": "v1-semantic-review.2",
            "review_method": review.review_method,
            "delegation_reference": review.delegation_reference,
            "dataset_hash": review.dataset_hash,
            "output_hash": review.output_hash,
            "status": review.status,
            "rows": [
                {k: v for k, v in row.model_dump().items() if k in ReviewRow.model_fields} for row in review.rows
            ],
        }
    )
    result = aggregate(provider, legacy, allow_delegated_review=True)
    scores = []
    for row in review.rows:
        sample = samples[(row.case_id, row.attempt)]
        ids = {c["chunk_id"] for c in sample.get("review_sources", [])}
        if any(not set(a.citation_ids) <= ids for a in row.body_assertions):
            raise ValueError("body_review_citation_not_retrieved")
        explanation = sample.get("output", {}).get("explanation", "")
        position = 0
        for assertion in row.body_assertions:
            if (
                assertion.start_char != position
                or assertion.end_char <= position
                or assertion.text != explanation[position : assertion.end_char]
            ):
                raise ValueError("body_review_span_mismatch")
            if assertion.supported and not assertion.citation_ids and sample.get("expected") != "unknown":
                raise ValueError("grounded_body_requires_source_reference")
            position = assertion.end_char
        if position != len(explanation):
            raise ValueError("body_review_incomplete_coverage")
        if sample["status"] != "passed":
            scores.append(0.0)
        elif row.body_assertions:
            scores.append(sum(a.supported for a in row.body_assertions) / len(row.body_assertions))
        elif sample.get("expected") != "unknown":
            scores.append(0.0)
    body_rate = sum(scores) / len(scores) if scores else None
    complete = set(samples) == expected and len(provider["rows"]) == provider.get("planned") == 80
    known = set(review.known_regressions) == set(REGRESSION_CASES)
    for key, cases in REGRESSION_CASES.items():
        values = review.known_regressions.get(key, [])
        slots = {(v.case_id, v.attempt) for v in values}
        required = {(case, attempt) for case in cases for attempt in (1, 2)}
        known = (
            known
            and slots == required
            and len(values) == len(required)
            and all(
                v.status == "passed" and samples.get((v.case_id, v.attempt), {}).get("status") == "passed"
                for v in values
            )
        )
    safety = not any(row.safety_violation for row in review.rows)
    passed = (
        bool(result.get("passed")) and complete and known and safety and body_rate is not None and body_rate >= 0.95
    )
    result.update(
        status="passed"
        if passed
        else "failed"
        if complete and review.status == "reviewed" and not result.get("unreviewed")
        else "unverified",
        passed=passed,
        body_assertion_support_macro=body_rate,
        body_applicable_outputs=len(scores),
        known_regressions={k: [v.model_dump() for v in values] for k, values in review.known_regressions.items()},
        safety_gate=safety,
        complete_80=complete,
        independent_human_reviewed=False,
    )
    return result


def main() -> int:
    """Consume a separately authored bound review; never fill scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/v1_enhancement_analysis.json"))
    args = parser.parse_args()
    provider = json.loads(args.provider_report.read_text(encoding="utf-8"))
    review = SemanticReview.model_validate_json(args.review.read_text(encoding="utf-8"))
    result = summarize(provider, review, json.loads(args.dataset.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
