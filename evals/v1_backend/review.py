"""Validate bound semantic reviews and preserve explicit reviewer provenance."""

import argparse
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from app.services.releaseguard.normalization import canonical_hash


def answer_contract(sample: dict) -> bool:
    """Observable refusal contract only; semantic grounding still needs a human."""
    output = sample.get("output") or {}
    if sample.get("status") != "passed":
        return False
    if sample.get("expected") == "unknown":
        return output.get("risk") == "UNKNOWN" and bool(output.get("unknowns"))
    # UNKNOWN risk alone is a conservative domain recommendation, not a refusal.
    # Do not demand a higher-confidence risk label merely to improve an eval.
    return bool(output.get("claims")) and not output.get("unknowns")


class ReviewRow(BaseModel):
    """Null is unverified; the reviewer reference contains no personal data."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    attempt: Annotated[int, Field(strict=True, ge=1)]
    human_rating: Annotated[int | None, Field(strict=True, ge=0, le=5)] = None
    grounded_claim_count: Annotated[int | None, Field(strict=True, ge=0)] = None
    total_claims: Annotated[int, Field(strict=True, ge=0)]
    reviewer_reference: Annotated[str | None, Field(min_length=1, max_length=100)] = None

    @model_validator(mode="after")
    def counts(self) -> "ReviewRow":
        """A reviewer cannot ground more claims than the actual output contains."""
        if self.grounded_claim_count is not None and self.grounded_claim_count > self.total_claims:
            raise ValueError("grounded_claim_count_exceeds_actual")
        return self


class HumanReview(BaseModel):
    """Bind every rating to the exact frozen dataset and evaluated outputs."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["v1-human-review.1", "v1-semantic-review.2"] = Field(alias="schema")
    review_method: Literal["independent_human", "assistant_delegated"] = "independent_human"
    delegation_reference: Annotated[str | None, Field(min_length=1, max_length=200)] = None
    dataset_hash: str
    output_hash: str
    status: Literal["unverified", "reviewed"]
    rows: list[ReviewRow] = Field(min_length=1)

    @model_validator(mode="after")
    def provenance(self) -> "HumanReview":
        """Legacy human files cannot silently impersonate a delegated review."""
        if self.review_method == "assistant_delegated" and (
            self.schema_version != "v1-semantic-review.2" or not self.delegation_reference
        ):
            raise ValueError("delegated_review_requires_explicit_provenance")
        if self.review_method == "independent_human" and self.delegation_reference is not None:
            raise ValueError("human_review_cannot_have_delegation_reference")
        return self


def aggregate(provider: dict, review: HumanReview, *, allow_delegated_review: bool = False) -> dict:
    """Failed outputs score zero; missing reviews remain unverified, never pass."""
    samples = provider["rows"]
    if review.dataset_hash != provider["dataset_hash"] or review.output_hash != canonical_hash(samples):
        raise ValueError("review_output_binding_mismatch")
    expected = {(s["case_id"], s["attempt"]): s for s in samples}
    supplied = {(r.case_id, r.attempt): r for r in review.rows}
    if (
        not samples
        or len(expected) != len(samples)
        or len(supplied) != len(review.rows)
        or expected.keys() != supplied.keys()
    ):
        raise ValueError("review_case_set_mismatch")
    groups: dict[str, list[tuple[float, int]]] = {}
    all_supported: dict[str, list[float]] = {}
    missing = []
    for key, sample in expected.items():
        row = supplied[key]
        claims = sample.get("output", {}).get("claims", [])
        if row.total_claims != len(claims):
            raise ValueError("review_claim_count_mismatch")
        if sample["status"] != "passed":
            score, rating = 0.0, 0
        elif row.human_rating is None or row.grounded_claim_count is None or not row.reviewer_reference:
            missing.append({"case_id": key[0], "attempt": key[1]})
            continue
        else:
            # Normal questions require claims. Deliberate unknown cases have
            # their own refusal metric and still require explanation review.
            score = row.grounded_claim_count / row.total_claims if row.total_claims else 0.0
            rating = row.human_rating
        if sample.get("expected") != "unknown":
            groups.setdefault(key[0], []).append((score, rating))
        if sample.get("expected") != "unknown" or claims:
            all_supported.setdefault(key[0], []).append(score)
    if missing or review.status != "reviewed":
        return {
            "status": "unverified",
            "samples": len(samples),
            "unreviewed": len(missing),
            "passed": False,
            "review_method": review.review_method,
            "independent_human_reviewed": False,
        }
    ratings = [0 if s["status"] != "passed" else supplied[k].human_rating for k, s in expected.items()]
    grounded = (
        sum(sum(v[0] for v in values) / len(values) for values in groups.values()) / len(groups) if groups else None
    )
    average = sum(int(v or 0) for v in ratings) / len(ratings)
    supporting_claim_rate = (
        sum(sum(v) / len(v) for v in all_supported.values()) / len(all_supported) if all_supported else None
    )
    minimum = min(int(v or 0) for v in ratings)
    attempts_by_case = {case: {attempt for cid, attempt in expected if cid == case} for case, _ in expected}
    sample_requirement = len(attempts_by_case) >= 20 and all(len(v) >= 2 for v in attempts_by_case.values())
    refusal = sum(answer_contract(s) for s in samples) / len(samples)
    unknown_cases = {
        case: [s for s in samples if s["case_id"] == case and s.get("expected") == "unknown"]
        for case in {s["case_id"] for s in samples if s.get("expected") == "unknown"}
    }
    unknown_rate = (
        sum(sum(answer_contract(s) for s in rows) / len(rows) for rows in unknown_cases.values()) / len(unknown_cases)
        if unknown_cases
        else None
    )
    citations = all(s.get("citation_correct") is True for s in samples)
    quality_thresholds_met = (
        sample_requirement
        and grounded is not None
        and grounded >= 0.95
        and supporting_claim_rate is not None
        and supporting_claim_rate >= 0.95
        and average >= 4
        and minimum >= 3
        and refusal >= 0.95
        and unknown_rate is not None
        and unknown_rate >= 0.95
        and citations
    )
    method_accepted = review.review_method == "independent_human" or allow_delegated_review
    passed = quality_thresholds_met and method_accepted
    return {
        "status": "unverified" if not method_accepted else "passed" if passed else "failed",
        "passed": passed,
        "quality_thresholds_met": quality_thresholds_met,
        "review_method": review.review_method,
        "review_method_accepted": method_accepted,
        "delegation_reference": review.delegation_reference,
        "independent_human_reviewed": review.review_method == "independent_human",
        "samples": len(samples),
        "sample_requirement_met": sample_requirement,
        "normal_cases": len(groups),
        "grounded_claim_rate_case_macro": grounded,
        "human_rating_mean": average,
        "human_rating_min": minimum,
        "explanation_rating_mean": average,
        "explanation_rating_min": minimum,
        "answer_contract_rate": refusal,
        "unknown_correctness_case_macro": unknown_rate,
        "unknown_cases": len(unknown_cases),
        "citation_id_contract_each_sample": citations,
        "dataset_hash": review.dataset_hash,
        "output_hash": review.output_hash,
        "aggregation_version": "v1-semantic-review-aggregate.4",
        "supporting_claim_rate_including_unknown_cases": supporting_claim_rate,
    }


def main() -> int:
    """Report-only validation; never calls a model or alters business authority."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-delegated-review", action="store_true")
    args = parser.parse_args()
    provider = json.loads(args.provider_report.read_text(encoding="utf-8"))
    review = HumanReview.model_validate_json(args.review.read_text(encoding="utf-8"))
    result = aggregate(provider, review, allow_delegated_review=args.allow_delegated_review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
