"""Synthetic experience input produces truthful partial coverage, without sockets."""

import asyncio

from app.schemas.v11_backend import ReviewBinding
from scripts.v11_probe_fixtures import source


def test_experience_missing_content_is_not_reconstructed() -> None:
    """Keep a readable diff beside unavailable Unicode-path content."""
    binding = ReviewBinding.model_validate({"repository_id": "checkout", "selector": {"kind": "pr", "pr_number": 4}})
    collected = asyncio.run(
        source(complete_evidence=True).collect(
            binding, {"github_id": 11, "github_owner_id": 10, "name": "actions/checkout"}
        )
    )
    assert collected.coverage.content_completeness == "partial"
    assert any(file.hunks for file in collected.files)
    missing = next(file for file in collected.files if file.path == "docs/证据/缺失.md")
    assert not missing.complete and not missing.hunks and missing.reasons
