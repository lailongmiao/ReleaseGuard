"""Frozen bilingual RAG cases and evidence-based document recall."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalCase(BaseModel):
    """Gold resources refer to reviewed documents, never model-picked answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    query: str = Field(min_length=1, max_length=2000)
    language: Literal["zh", "en"]
    expected: Literal["grounded", "unknown"]
    gold_resources: list[str] = Field(max_length=5)
    basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def gold(self) -> "RetrievalCase":
        """Empty Gold belongs only to explicit no-answer cases."""
        if bool(self.gold_resources) != (self.expected == "grounded") or len(set(self.gold_resources)) != len(
            self.gold_resources
        ):
            raise ValueError("retrieval_gold_invalid")
        return self


class RetrievalDataset(BaseModel):
    """Exact approved evaluation size and language/answer strata."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["v1-enhancement-rag.1", "v1-enhancement-rag.2"]
    corpus_manifest_hash: str
    cases: list[RetrievalCase] = Field(min_length=40, max_length=40)

    @model_validator(mode="after")
    def counts(self) -> "RetrievalDataset":
        """Do not replace failed cases or duplicate easier questions."""
        if (
            len({c.case_id for c in self.cases}),
            len({c.query for c in self.cases}),
            sum(c.language == "zh" for c in self.cases),
            sum(c.expected == "grounded" for c in self.cases),
        ) != (40, 40, 32, 30):
            raise ValueError("retrieval_dataset_counts_invalid")
        return self


class AnalysisDataset(BaseModel):
    """Frozen independent bilingual holdout, validated before any paid call."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["v1-enhancement-analysis.1", "v1-enhancement-analysis.2"]
    frozen_before_provider_calls: Literal[True]
    corpus_manifest_hash: str
    required_probe_pr: Literal[2557]
    cases: list[RetrievalCase] = Field(min_length=40, max_length=40)

    @model_validator(mode="after")
    def counts(self) -> "AnalysisDataset":
        """The approved English and Chinese strata cannot be silently changed."""
        expected = {f"holdout-{i:02}" for i in range(1, 21)} | {f"zh-holdout-{i:02}" for i in range(1, 21)}
        if (
            {c.case_id for c in self.cases} != expected
            or len({c.query for c in self.cases}) != 40
            or any((c.language == "zh") != c.case_id.startswith("zh-") for c in self.cases)
            or sum(c.language == "zh" and c.expected == "grounded" for c in self.cases) != 15
        ):
            raise ValueError("analysis_dataset_counts_invalid")
        return self


def summarize_retrieval(rows: list[dict], dataset: RetrievalDataset, repeat: int) -> dict:
    """Failures score zero; no-answer retrieval is not a refusal or zero recall."""
    expected = {(c.case_id, attempt) for c in dataset.cases for attempt in range(1, repeat + 1)}
    keys = [(r["case_id"], r["attempt"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("retrieval_report_slots_invalid")
    scores = []
    for case in dataset.cases:
        if case.expected == "unknown":
            continue
        selected = [r for r in rows if r["case_id"] == case.case_id]
        scores.append(
            sum(
                len(set(r.get("resources", [])) & set(case.gold_resources)) / len(case.gold_resources)
                if r.get("status") == "completed"
                else 0
                for r in selected
            )
            / repeat
        )
    recall = sum(scores) / len(scores) if scores else None
    complete = set(keys) == expected
    citations = complete and all(r.get("citations_valid") is True for r in rows)
    return {
        "status": "passed"
        if complete and citations and recall is not None and recall >= 0.85
        else "failed"
        if complete
        else "unverified",
        "samples": len(rows),
        "planned": len(expected),
        "recall_at_5_macro": recall,
        "answerable_cases": len(scores),
        "no_answer_cases": 10,
        "no_answer_refusal": "evaluated_in_analysis_not_inferred_from_nonempty_retrieval",
        "citation_gate": citations,
    }
