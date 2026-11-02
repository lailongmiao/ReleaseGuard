"""Bounded unified diff parsing with exact old/new coordinates and honest gaps."""

import difflib
import re
from typing import Any

from app.schemas.v11_backend import DiffFile, DiffHunk, DiffLine
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError

FILE_BYTES = 32768
TOTAL_BYTES = 131072
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")


def safe_path(value: Any) -> str:
    """Keep path identity while rejecting traversal, control bytes and URL inputs."""
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4096
        or value.startswith(("/", "\\"))
        or "\\" in value
        or ":" in value
        or any(p in {".", "..", ""} for p in value.split("/"))
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise DomainError("source_path_invalid")
    return value


def parse_patch(patch: str) -> list[DiffHunk]:
    """Validate hunk lengths and preserve payload CR and missing-newline markers."""
    hunks: list[DiffHunk] = []
    lines: list[DiffLine] = []
    header = ""
    old = new = old_count = new_count = used_old = used_new = 0
    for raw in patch.split("\n"):
        match = HUNK.fullmatch(raw)
        if match:
            if header:
                if (used_old, used_new) != (old_count, new_count):
                    raise DomainError("patch_hunk_truncated")
                hunks.append(DiffHunk(hunk_id="hunk:" + canonical_hash([header, lines]), header=header, lines=lines))
            old, old_count, new, new_count = (
                int(match[1]),
                int(match[2]) if match[2] is not None else 1,
                int(match[3]),
                int(match[4]) if match[4] is not None else 1,
            )
            if (old_count and not old) or (new_count and not new):
                raise DomainError("patch_coordinates_invalid")
            if hunks:
                prior_old = [line.old_line for line in hunks[-1].lines if line.old_line is not None]
                prior_new = [line.new_line for line in hunks[-1].lines if line.new_line is not None]
                if (prior_old and old <= prior_old[-1]) or (prior_new and new <= prior_new[-1]):
                    raise DomainError("patch_coordinates_invalid")
            header, lines, used_old, used_new = raw, [], 0, 0
            continue
        if raw == "\\ No newline at end of file":
            if not lines:
                raise DomainError("patch_invalid")
            if lines[-1].no_final_newline:
                raise DomainError("patch_invalid")
            lines[-1] = lines[-1].model_copy(update={"no_final_newline": True})
            continue
        if raw == "" and (used_old, used_new) == (old_count, new_count):
            continue
        if not header or not raw or raw[0] not in {" ", "+", "-"}:
            raise DomainError("patch_invalid")
        kind = {" ": "context", "+": "add", "-": "delete"}[raw[0]]
        item = DiffLine.model_validate(
            {
                "kind": kind,
                "text": raw[1:],
                "old_line": old if kind != "add" else None,
                "new_line": new if kind != "delete" else None,
            }
        )
        lines.append(item)
        if kind != "add":
            old += 1
            used_old += 1
        if kind != "delete":
            new += 1
            used_new += 1
        if used_old > old_count or used_new > new_count:
            raise DomainError("patch_hunk_overflow")
    if not header or (used_old, used_new) != (old_count, new_count):
        raise DomainError("patch_hunk_truncated")
    hunks.append(DiffHunk(hunk_id="hunk:" + canonical_hash([header, lines]), header=header, lines=lines))
    return hunks


def bounded_hunks(hunks: list[DiffHunk], budget: int) -> tuple[list[DiffHunk], str, bool]:
    """Select deterministic line prefixes without inventing new original hunk lengths."""
    selected: list[DiffHunk] = []
    parts: list[str] = []
    size = 0
    for hunk in hunks:
        heading = hunk.header + "\n"
        if size + len(heading.encode()) > budget:
            return selected, "".join(parts), False
        kept = []
        hunk_parts = [heading]
        size += len(heading.encode())
        for line in hunk.lines:
            raw = {"add": "+", "delete": "-", "context": " "}[line.kind] + line.text + "\n"
            if line.no_final_newline:
                raw += "\\ No newline at end of file\n"
            if size + len(raw.encode()) > budget:
                if kept:
                    selected.append(hunk.model_copy(update={"lines": kept, "complete": False}))
                    parts.extend(hunk_parts)
                return selected, "".join(parts), False
            size += len(raw.encode())
            kept.append(line)
            hunk_parts.append(raw)
        selected.append(hunk)
        parts.extend(hunk_parts)
    return selected, "".join(parts), True


def read_file(raw: dict[str, Any], budget: int) -> DiffFile:
    """Validate metadata, detect absent/truncated patches, then apply local caps."""
    path = safe_path(raw.get("filename"))
    old_path = safe_path(raw["previous_filename"]) if raw.get("previous_filename") else None
    status = raw.get("status")
    if status not in {"added", "removed", "modified", "renamed", "copied", "changed", "unchanged"}:
        raise DomainError("source_file_status_invalid")
    additions, deletions = raw.get("additions"), raw.get("deletions")
    if type(additions) is not int or type(deletions) is not int or additions < 0 or deletions < 0:
        raise DomainError("source_file_stats_invalid")
    file = DiffFile(
        file_id="file:" + canonical_hash([path, old_path]),
        path=path,
        old_path=old_path,
        status=status,
        additions=additions,
        deletions=deletions,
        content_hash=canonical_hash(""),
    )
    patch = raw.get("patch")
    if not isinstance(patch, str) or not patch:
        if status == "renamed" and additions == deletions == 0:
            return file.model_copy(update={"complete": True, "reasons": ["rename_metadata_only"]})
        return file.model_copy(update={"reasons": ["patch_missing_unclassified"]})
    try:
        hunks = parse_patch(patch)
        added = sum(line.kind == "add" for h in hunks for line in h.lines)
        removed = sum(line.kind == "delete" for h in hunks for line in h.lines)
        if (added, removed) != (additions, deletions):
            raise DomainError("patch_stats_mismatch")
    except DomainError as error:
        return file.model_copy(update={"reasons": [error.code]})
    hunks, selected, complete = bounded_hunks(hunks, min(FILE_BYTES, max(0, budget)))
    return file.model_copy(
        update={
            "hunks": hunks,
            "patch": selected,
            "content_hash": canonical_hash(selected),
            "complete": complete,
            "reasons": [] if complete else ["patch_byte_limit"],
        }
    )


def rebuild_patch(before: str, after: str) -> str:
    """Reconstruct text differences off-thread; never execute or normalize code."""
    # Git lines are separated by LF, not Python's wider Unicode line boundaries.
    old_lines = before.split("\n")
    new_lines = after.split("\n")
    old_rows = [line + "\n" for line in old_lines[:-1]] + ([old_lines[-1]] if old_lines[-1] else [])
    new_rows = [line + "\n" for line in new_lines[:-1]] + ([new_lines[-1]] if new_lines[-1] else [])
    rows = list(difflib.unified_diff(old_rows, new_rows, n=3))
    result = []
    for row in rows[2:]:
        result.append(row if row.endswith("\n") else row + "\n\\ No newline at end of file\n")
    return "".join(result).rstrip("\n")
