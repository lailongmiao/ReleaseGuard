"""Frozen quality positions and independent semantic review contracts."""

from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.v11_backend import Comparison, ReviewBinding


class Case(BaseModel):
    """A target and its role must be frozen before any model samples are obtained."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    binding: ReviewBinding
    comparison: Comparison
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    retrieval_query: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
    language: Literal["zh", "en"]
    expected_unknown: bool
    group: Literal["regular", "large_partial"]
    expected_documents: Annotated[list[str], Field(min_length=1)]
    expected_facts: Annotated[list[str], Field(min_length=1)]
    prohibited_claims: list[str]
    related_issues: list[str]


class Dataset(BaseModel):
    """The acceptance denominator cannot be reduced after failures occur."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["releaseguard.v11.acceptance.1"]
    provenance: Literal["synthetic_development", "frozen_public_source"]
    source_evidence_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    cases: list[Case]

    @model_validator(mode="after")
    def fixed_matrix(self) -> "Dataset":
        """Require the approved 40-case composition and independent case identities."""
        if len(self.cases) != 40 or len({case.case_id for case in self.cases}) != 40:
            raise ValueError("forty_unique_cases_required")
        if Counter(case.binding.selector.kind for case in self.cases) != {
            "pr": 16,
            "commit": 12,
            "branch_compare": 12,
        }:
            raise ValueError("object_type_matrix_invalid")
        if Counter(case.language for case in self.cases) != {"zh": 20, "en": 20}:
            raise ValueError("language_matrix_invalid")
        if (
            sum(case.expected_unknown for case in self.cases) != 10
            or sum(case.group == "large_partial" for case in self.cases) != 10
        ):
            raise ValueError("unknown_or_partial_matrix_invalid")
        if {case.binding.repository_id for case in self.cases} != {"checkout", "setup-python"}:
            raise ValueError("two_repositories_required")
        if any(case.binding.selector.kind != case.comparison.kind for case in self.cases):
            raise ValueError("comparison_kind_conflict")
        return self


class SemanticPosition(BaseModel):
    """Scoring requires actual input/output review, with all denominators explicit."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    repeat: Annotated[int, Field(strict=True, ge=1, le=2)]
    score: Annotated[int, Field(strict=True, ge=1, le=5)]
    claims_supported: Annotated[float, Field(ge=0, le=1)]
    body_supported: Annotated[float, Field(ge=0, le=1)]
    unknown_correct: bool
    critical_error: bool
    citation_identity_valid: bool
    rationale: Annotated[str, Field(min_length=20)]
    output_hash: str
    input_hash: str


class SemanticReview(BaseModel):
    """Delegated evaluation never presents itself as independent human review."""

    model_config = ConfigDict(extra="forbid")
    review_method: Literal["assistant_delegated"]
    independent_human_reviewed: Literal[False]
    report_hash: str
    positions: list[SemanticPosition]
