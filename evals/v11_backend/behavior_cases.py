"""Independent bounded fixtures; never represented as observed GitHub code."""

import json
from datetime import UTC, datetime
from difflib import unified_diff
from pathlib import Path

from app.schemas.v11_backend import ReviewCitation
from app.services.v11_diff import read_file
from app.services.v11_policy import citation_line_roles, diff_chunks

ROOT = Path(__file__).resolve().parents[1]


def cases() -> list[dict]:
    """Load the development contract frozen before real sampling."""
    return json.loads((ROOT / "datasets/v11_repair2_development.json").read_text(encoding="utf-8"))["cases"]


def evidence(selected: list[dict]) -> tuple[dict, list[ReviewCitation], dict[str, list[str]]]:
    """Parse actual independent fixture hunks with the production diff parser."""
    files = []
    for case in selected:
        lines = list(unified_diff(case["before"].splitlines(), case["after"].splitlines(), n=3, lineterm=""))[2:]
        files.append(
            read_file(
                {
                    "filename": case["path"],
                    "status": "modified",
                    "patch": "\n".join(lines),
                    "additions": sum(line.startswith("+") for line in lines),
                    "deletions": sum(line.startswith("-") for line in lines),
                },
                32768,
            ).model_dump(mode="json")
        )
    facts = {
        "comparison": {
            "kind": "commit",
            "comparison": "parent",
            "head_sha": "a" * 40,
            "base_tip_sha": "b" * 40,
            "diff_base_sha": "b" * 40,
            "parent_sha": "b" * 40,
        },
        "files": files,
        "coverage": {"manifest_completeness": "complete", "content_completeness": "complete"},
    }
    _, citations, complete = diff_chunks(
        "synthetic-behavior",
        "synthetic-snapshot",
        "synthetic-generation",
        "synthetic-development",
        facts,
        datetime.now(UTC),
    )
    assert complete
    return facts, citations, citation_line_roles(citations, facts, "synthetic-behavior")
