"""Deterministic review coverage, rule applicability and non-authorizing citations."""

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.schemas.v1_backend import Chunk, SourceDocument
from app.schemas.v11_backend import (
    CIObservation,
    Comparison,
    Coverage,
    DiffFile,
    Eligibility,
    ReviewCitation,
    ReviewCapabilities,
    ReviewOutput,
)
from app.services.llm.tokenizer import ProviderNeutralTokenizer
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_rejection import ReviewValidationError

PROMPT_VERSION = "v11-diff-grounded.6"
INDEX_POLICY = "v11-lines-512-max64-rrf60-two-per-document.1"
ROOT = Path(__file__).resolve().parents[2] / "resources" / "knowledge"
CAPABILITIES = ReviewCapabilities()
PAUSED_COMMANDS = frozenset({"prepare", "decide", "execute"})


def require_rehearsal() -> None:
    """Read the deployed capability policy, never a request or environment override."""
    if CAPABILITIES.pause_reason is not None:
        raise DomainError(CAPABILITIES.pause_reason)


def rehearsal_seal(report: dict) -> str:
    """Bind new authorizations to the capability epoch without rewriting saved reports."""
    return canonical_hash(
        {
            "capability_version": CAPABILITIES.version,
            "report": {
                key: value
                for key, value in report.items()
                if key not in {"stale_reasons", "rehearsal_eligibility", "capabilities"}
            },
        }
    )


def token_count(text: str) -> int:
    """Use the declared local tokenizer; unavailable counts are never zero."""
    value = ProviderNeutralTokenizer.from_model(
        "", provider="openai-compatible", tokenizer_name="cl100k_base"
    ).count_text(text)
    if value.value is None:
        raise DomainError("review_tokenizer_unavailable")
    return value.value


def load_rules(repository: str) -> tuple[dict, list[tuple[dict, str]]]:
    """Verify original bytes and restrict rules before reading returned content."""
    path = ROOT / "v11_corpus_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != "v11-reviewed-corpus.1":
        raise DomainError("rule_manifest_version_invalid")
    result = []
    for entry in manifest["documents"]:
        if repository not in entry["repositories"] and "*" not in entry["repositories"]:
            continue
        filename = entry["filename"]
        file = ROOT / filename
        if Path(filename).name != filename or file.is_symlink() or file.resolve().parent != ROOT.resolve():
            raise DomainError("rule_source_unsafe")
        data = file.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        if len(data) > 65536 or sha256(data).hexdigest() != entry["sha256"]:
            raise DomainError("rule_content_changed")
        result.append((entry, data.decode("utf-8")))
    if not any(entry["role"] == "application_policy" for entry, _ in result):
        raise DomainError("application_policy_missing")
    return manifest, result


def diff_chunks(
    object_id: str, snapshot: str, generation: str, source: str, facts: dict, captured_at: datetime
) -> tuple[list[Chunk], list[ReviewCitation], bool]:
    """Create exact contiguous side ranges; truncate only at known line boundaries."""
    comparison = Comparison.model_validate(facts["comparison"])
    chunks: list[Chunk] = []
    citations: list[ReviewCitation] = []
    complete = True
    for raw in sorted(facts["files"], key=lambda item: item["path"]):
        file = DiffFile.model_validate(raw)
        if not file.hunks:
            text = json.dumps(
                {
                    "path": file.path,
                    "old_path": file.old_path,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "content_read": False,
                    "reasons": file.reasons,
                },
                ensure_ascii=False,
            )
            citations.append(
                ReviewCitation(
                    citation_id="metadata:" + canonical_hash([snapshot, generation, file.file_id, text]),
                    source_role="change_observation",
                    kind="file_metadata",
                    snapshot_id=snapshot,
                    generation_id=generation,
                    file_id=file.file_id,
                    path=file.path,
                    old_path=file.old_path,
                    sha=comparison.head_sha,
                    text=text,
                    text_hash=canonical_hash(text),
                    locator=file.path + ":metadata",
                )
            )
        for hunk in file.hunks:
            for side in ("LEFT", "RIGHT"):
                lines = [
                    line for line in hunk.lines if (line.old_line if side == "LEFT" else line.new_line) is not None
                ]
                if not any(line.kind == ("delete" if side == "LEFT" else "add") for line in lines):
                    continue
                start = 0
                while start < len(lines):
                    end = start
                    while (
                        end < len(lines)
                        and token_count("\n".join(line.text for line in lines[start : end + 1])) <= 512
                    ):
                        end += 1
                    if end == start:
                        complete = False
                        start += 1
                        continue
                    selected = lines[start:end]
                    text = "\n".join(line.text for line in selected)
                    if not text.strip():
                        complete = False
                        start = end
                        continue
                    if len(chunks) >= 64:
                        return chunks, citations, False
                    start_line = selected[0].old_line if side == "LEFT" else selected[0].new_line
                    end_line = selected[-1].old_line if side == "LEFT" else selected[-1].new_line
                    identity = canonical_hash(
                        [object_id, snapshot, generation, file.file_id, hunk.hunk_id, side, start_line, end_line, text]
                    )
                    identifier = "diff:" + identity
                    citation = ReviewCitation.model_validate(
                        {
                            "citation_id": identifier,
                            "source_role": "change_observation",
                            "kind": "diff",
                            "snapshot_id": snapshot,
                            "generation_id": generation,
                            "file_id": file.file_id,
                            "path": file.path,
                            "old_path": file.old_path,
                            "side": side,
                            "sha": comparison.diff_base_sha if side == "LEFT" else comparison.head_sha,
                            "hunk_id": hunk.hunk_id,
                            "start_line": start_line,
                            "end_line": end_line,
                            "text": text,
                            "text_hash": canonical_hash(text),
                            "locator": f"{file.path}:{side}:{start_line}-{end_line}",
                        }
                    )
                    document = SourceDocument(
                        document_id=file.file_id,
                        source_id=source,
                        resource=object_id,
                        version=snapshot,
                        content=text or "\n",
                        content_hash=canonical_hash(text or "\n"),
                        locator=citation.locator,
                        captured_at=captured_at,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=identifier,
                            document=document,
                            ordinal=len(chunks),
                            text=text,
                            text_hash=citation.text_hash,
                            tokens=max(1, token_count(text)),
                            generation=generation,
                            start_char=0,
                            end_char=len(text),
                            locator=citation.locator,
                        )
                    )
                    citations.append(citation)
                    start = end
    return chunks, citations, complete


def fuse_review(lexical: list[Chunk], semantic: list[Chunk], limit: int = 5) -> list[Chunk]:
    """Preserve legacy RRF while allowing two useful fragments per document."""
    scores: dict[str, float] = {}
    values: dict[str, Chunk] = {}
    for ranking in (lexical[:20], semantic[:20]):
        for rank, chunk in enumerate(ranking, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (60 + rank)
            values[chunk.chunk_id] = chunk
    counts: Counter[str] = Counter()
    result = []
    for identifier in sorted(scores, key=lambda key: (-scores[key], key)):
        chunk = values[identifier]
        if counts[chunk.document.document_id] == 2:
            continue
        result.append(chunk)
        counts[chunk.document.document_id] += 1
        if len(result) == limit:
            break
    return result


def validate_output(output: ReviewOutput, citations: list[ReviewCitation]) -> None:
    """A rule cannot masquerade as a code observation or reference absent input."""
    if output.model_risk.value == "UNKNOWN" and output.candidate_steps:
        raise DomainError("analysis_unknown_forbids_candidate_steps")
    available = {citation.citation_id: citation for citation in citations}
    for section in ("observations", "risks"):
        for index, claim in enumerate(getattr(output, section)):
            if not set(claim.citation_ids) <= available.keys():
                raise ReviewValidationError("analysis_citation_invalid", "validate_output", section, index)
            if claim.kind == "observation" and not any(
                available[key].kind in {"diff", "ci"} for key in claim.citation_ids
            ):
                raise ReviewValidationError("analysis_observation_requires_diff", "validate_output", section, index)
            if claim.kind == "rule" and not any(
                available[key].source_role in {"application_policy", "repository_policy", "usage_reference"}
                for key in claim.citation_ids
            ):
                raise ReviewValidationError("analysis_rule_citation_invalid", "validate_output", section, index)


def qualification(
    repository: str,
    kind: str,
    coverage: Coverage | None,
    ci: CIObservation | None,
    output: ReviewOutput | None,
    stale: list[str],
    *,
    evaluation_readonly: bool = False,
) -> Eligibility:
    """Phase restriction supplements every original eligibility requirement."""
    reasons = list(stale)
    if CAPABILITIES.pause_reason is not None:
        reasons.append(CAPABILITIES.pause_reason)
    if evaluation_readonly:
        reasons.append("evaluation_permanently_readonly")
    if kind != "pr":
        reasons.append("object_type_readonly")
    if repository != "checkout":
        reasons.append("repository_rehearsal_disabled")
    if coverage is None or any(
        getattr(coverage, name) != "complete"
        for name in ("manifest_completeness", "content_completeness", "analysis_coverage")
    ):
        reasons.append("coverage_incomplete")
    if ci is None or not ci.complete or not ci.checks_passed:
        reasons.append("ci_policy_blocked")
    if output is None or output.model_risk.value == "UNKNOWN" or output.unknowns or not output.candidate_steps:
        reasons.append("analysis_not_actionable")
    return Eligibility(
        eligible=not reasons,
        reason_codes=sorted(set(reasons)),
        evaluated_at=datetime.now(UTC),
        policy_version=CAPABILITIES.version,
    )


def stale_reasons(snapshot: dict, expected: dict, current: dict) -> list[str]:
    """Freshness is an observation bound, never a cross-system atomic promise."""
    reasons = []
    if expected != current:
        reasons.append("review_dependencies_changed")
    if snapshot["captured_at"] + timedelta(minutes=15) <= datetime.now(UTC):
        reasons.append("snapshot_expired")
    return reasons


def model_sources(
    citations: list[ReviewCitation], line_roles: dict[str, list[str]] | None = None
) -> list[dict[str, Any]]:
    """Send only bounded source text and identity, never permission fields."""
    return [
        {
            "citation_id": c.citation_id,
            "source_role": c.source_role,
            "text": c.text,
            "locator": c.locator,
            "kind": c.kind,
            "sha": c.sha,
            "side": c.side,
            "path": c.path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            **({"line_roles": line_roles[c.citation_id]} if line_roles and c.citation_id in line_roles else {}),
        }
        for c in citations
    ]


def citation_line_roles(
    citations: list[ReviewCitation], facts: dict, object_id: str | None = None
) -> dict[str, list[str]]:
    """Preserve parsed edit/context roles without asking the model to infer a diff."""
    result = {}
    files = {file["file_id"]: file for file in facts["files"]}
    for citation in citations:
        if citation.kind != "diff":
            continue
        file = files.get(citation.file_id)
        hunk = next((h for h in file["hunks"] if h["hunk_id"] == citation.hunk_id), None) if file else None
        if file is None or hunk is None or citation.start_line is None or citation.end_line is None:
            raise DomainError("diff_citation_position_invalid")
        coordinate = "old_line" if citation.side == "LEFT" else "new_line"
        lines = [
            line
            for line in hunk["lines"]
            if line[coordinate] is not None and citation.start_line <= line[coordinate] <= citation.end_line
        ]
        if "\n".join(line["text"] for line in lines) != citation.text:
            raise DomainError("diff_citation_position_invalid")
        if (
            not lines
            or lines[0][coordinate] != citation.start_line
            or lines[-1][coordinate] != citation.end_line
            or citation.sha != facts["comparison"]["diff_base_sha" if citation.side == "LEFT" else "head_sha"]
            or citation.path != file["path"]
            or citation.old_path != file["old_path"]
            or citation.text_hash != canonical_hash(citation.text)
        ):
            raise DomainError("diff_citation_identity_invalid")
        if object_id is not None and citation.citation_id != "diff:" + canonical_hash(
            [
                object_id,
                citation.snapshot_id,
                citation.generation_id,
                citation.file_id,
                citation.hunk_id,
                citation.side,
                citation.start_line,
                citation.end_line,
                citation.text,
            ]
        ):
            raise DomainError("diff_citation_identity_invalid")
        result[citation.citation_id] = [line["kind"] for line in lines]
    return result
