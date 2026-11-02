"""Model-wire comparisons with bounded evidence checks, not semantic certification."""

import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field

from app.schemas.v11_backend import ReviewCitation, ReviewModel, ReviewOutput
from app.services.v11_rejection import ReviewValidationError


class ClaimComparison(ReviewModel):
    """Every new claim declares its scope and explicit before/after premises."""

    section: Literal["observations", "risks"]
    index: Annotated[int, Field(strict=True, ge=0, le=15)]
    classification: Literal[
        "behavior_added",
        "behavior_modified",
        "relocated",
        "documentation",
        "literal_edit",
        "unchanged_context",
        "insufficient",
        "rule",
    ]
    before: Annotated[str, Field(max_length=500)]
    after: Annotated[str, Field(max_length=500)]
    before_ids: Annotated[list[str], Field(max_length=5)]
    after_ids: Annotated[list[str], Field(max_length=5)]


class ComparedReviewOutput(ReviewOutput):
    """Internal wire extension; never changes historical/public ReviewOutput bytes."""

    comparisons: Annotated[list[ClaimComparison], Field(max_length=28)]


COMPARISON_CONTRACT = "v11-behavior-comparison.2"


class ClaimComparisonV2(ReviewModel):
    """Claim citation_ids are the sole evidence relation; sides belong to the server."""

    section: Literal["observations", "risks"]
    index: Annotated[int, Field(strict=True, ge=0, le=15)]
    classification: Literal[
        "behavior_added",
        "behavior_modified",
        "relocated",
        "documentation",
        "literal_edit",
        "unchanged_context",
        "insufficient",
        "rule",
    ]
    before: Annotated[str, Field(max_length=500)]
    after: Annotated[str, Field(max_length=500)]


class ComparedReviewOutputV2(ReviewOutput):
    """Versioned internal wire only; public ReviewOutput retains its exact fields."""

    comparison_contract: Literal["v11-behavior-comparison.2"]
    comparisons: Annotated[list[ClaimComparisonV2], Field(max_length=28)]


def resolve_comparisons_v2(
    output: ComparedReviewOutputV2, citations: list[ReviewCitation], sent: dict[str, Any]
) -> ComparedReviewOutput:
    """Resolve only current supplied identities, then reuse the strict side/coverage checks.

    This is a v2 server interpretation, never migration or repair of a historical v1 wire.
    Passing these structural gates cannot prove behavior equivalence or semantic truth.
    """
    available = {c.citation_id: c for c in citations}
    sources = {c["citation_id"]: c for c in sent["sources"]}
    expected = {(section, i) for section in ("observations", "risks") for i in range(len(getattr(output, section)))}
    actual = [(item.section, item.index) for item in output.comparisons]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ReviewValidationError("analysis_comparison_coverage_invalid", "validate_comparisons")
    resolved = []
    for item in output.comparisons:
        claim = getattr(output, item.section)[item.index]
        sides: dict[str, list[str]] = {"LEFT": [], "RIGHT": []}
        for key in claim.citation_ids:
            citation, source = available.get(key), sources.get(key)
            fields = ("kind", "source_role", "text", "side", "sha", "path", "start_line", "end_line")
            if citation is None or source is None or any(source.get(f) != getattr(citation, f) for f in fields):
                raise ReviewValidationError(
                    "analysis_comparison_citation_invalid", "validate_comparisons", item.section, item.index
                )
            if citation.kind == "diff":
                if citation.side not in sides:
                    raise ReviewValidationError(
                        "analysis_comparison_side_invalid", "validate_comparisons", item.section, item.index
                    )
                sides[citation.side].append(key)
        selected = sides["LEFT"] + sides["RIGHT"]
        if item.classification == "documentation" and (
            not selected
            or not any({"add", "delete"}.intersection(sources[key].get("line_roles", [])) for key in selected)
        ):
            raise ReviewValidationError(
                "analysis_documentation_requires_document", "validate_comparisons", item.section, item.index
            )
        resolved.append({**item.model_dump(), "before_ids": sides["LEFT"], "after_ids": sides["RIGHT"]})
    result = ComparedReviewOutput.model_validate(
        {**output.model_dump(exclude={"comparison_contract", "comparisons"}), "comparisons": resolved}
    )
    validate_comparisons(result, citations, sent)
    return result


def documentation(path: str | None) -> bool:
    """Conservative file-level evidence label, not a language execution classifier."""
    return PurePosixPath(path or "").suffix.lower() in {".md", ".rst", ".txt"}


def comparison_groups(citations: list[ReviewCitation]) -> list[list[ReviewCitation]]:
    """Keep both sides of each captured hunk together, including split chunks."""
    groups: dict[tuple, list[ReviewCitation]] = {}
    for citation in citations:
        key = (citation.file_id, citation.hunk_id) if citation.kind == "diff" else (citation.citation_id,)
        groups.setdefault(key, []).append(citation)
    return list(groups.values())


def comparison_index(citations: list[ReviewCitation], roles: dict[str, list[str]]) -> dict:
    """Cross-hunk literal matches are navigation hints, never equivalence claims."""
    hunks = []
    anchors: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"LEFT": set(), "RIGHT": set()})
    for group in comparison_groups([c for c in citations if c.kind == "diff"]):
        hunks.append(
            {
                "path": group[0].path,
                "hunk": group[0].hunk_id,
                "before": [c.citation_id for c in group if c.side == "LEFT"],
                "after": [c.citation_id for c in group if c.side == "RIGHT"],
            }
        )
        for c in group:
            for line, role in zip(c.text.split("\n"), roles.get(c.citation_id, []), strict=False):
                if role not in {"add", "delete", "context"}:
                    continue
                for literal in re.findall(r"['\"]([^'\"\n]{3,64})['\"]", line):
                    anchors[literal][str(c.side)].add(c.citation_id)
    return {
        "hunks": hunks,
        "shared_literals_not_equivalence": [
            {"literal": value, "before": sorted(sides["LEFT"]), "after": sorted(sides["RIGHT"])}
            for value, sides in sorted(anchors.items())
            if sides["LEFT"] and sides["RIGHT"]
        ][:16],
    }


def validate_comparisons(output: ComparedReviewOutput, citations: list[ReviewCitation], sent: dict[str, Any]) -> None:
    """Reject absent sides/edits/coverage; valid evidence can still be misinterpreted."""
    available = {c.citation_id: c for c in citations}
    sources = {c["citation_id"]: c for c in sent["sources"]}
    expected = {(section, i) for section in ("observations", "risks") for i in range(len(getattr(output, section)))}
    actual = [(item.section, item.index) for item in output.comparisons]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ReviewValidationError("analysis_comparison_coverage_invalid", "validate_comparisons")
    observations = sent.get("SERVER_OBSERVATIONS", {})
    for item in output.comparisons:
        claim = getattr(output, item.section)[item.index]
        selected = item.before_ids + item.after_ids
        if not set(selected) <= set(claim.citation_ids) or not set(selected) <= available.keys():
            raise ReviewValidationError(
                "analysis_comparison_citation_invalid", "validate_comparisons", item.section, item.index
            )
        for side, ids in (("LEFT", item.before_ids), ("RIGHT", item.after_ids)):
            for key in ids:
                c = available[key]
                source = sources.get(key)
                if c.kind != "diff" or c.side != side or source is None or source["text"] != c.text:
                    raise ReviewValidationError(
                        "analysis_comparison_side_invalid", "validate_comparisons", item.section, item.index
                    )
        classification = item.classification
        if classification in {"behavior_added", "behavior_modified", "relocated"}:
            if not item.before.strip() or not item.after.strip() or not item.before_ids or not item.after_ids:
                raise ReviewValidationError(
                    "analysis_behavior_requires_both_sides", "validate_comparisons", item.section, item.index
                )
            coverage = observations.get("coverage", {})
            if (
                not observations.get("model_coverage_complete")
                or any(coverage.get(k) != "complete" for k in ("manifest_completeness", "content_completeness"))
                or observations.get("index_complete") is not True
            ):
                raise ReviewValidationError(
                    "analysis_behavior_coverage_incomplete", "validate_comparisons", item.section, item.index
                )
            for side, ids in (("LEFT", item.before_ids), ("RIGHT", item.after_ids)):
                code = [key for key in ids if not documentation(available[key].path)]
                if not code:
                    raise ReviewValidationError(
                        "analysis_behavior_requires_code", "validate_comparisons", item.section, item.index
                    )
                if not any(
                    ("delete" if side == "LEFT" else "add") in sources[key].get("line_roles", []) for key in code
                ):
                    raise ReviewValidationError(
                        "analysis_behavior_requires_edit", "validate_comparisons", item.section, item.index
                    )
        elif classification == "documentation":
            if not selected:
                raise ReviewValidationError(
                    "analysis_documentation_requires_document", "validate_comparisons", item.section, item.index
                )
        elif classification in {"literal_edit", "unchanged_context"}:
            wanted = {"add", "delete"} if classification == "literal_edit" else {"context"}
            if not selected or not any(wanted.intersection(sources[key].get("line_roles", [])) for key in selected):
                raise ReviewValidationError(
                    "analysis_comparison_line_role_invalid", "validate_comparisons", item.section, item.index
                )
        elif classification == "insufficient":
            if not output.unknowns or output.model_risk.value != "UNKNOWN":
                raise ReviewValidationError(
                    "analysis_insufficient_requires_unknown", "validate_comparisons", item.section, item.index
                )
        elif classification == "rule":
            if not any(available[key].kind in {"rule", "ci"} for key in claim.citation_ids):
                raise ReviewValidationError(
                    "analysis_comparison_requires_rule", "validate_comparisons", item.section, item.index
                )
