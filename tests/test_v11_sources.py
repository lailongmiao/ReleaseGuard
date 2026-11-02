"""MockTransport source matrix; these tests never contact GitHub."""

import asyncio
from copy import deepcopy

import httpx
import pytest

from app.schemas.v11_backend import ReviewBinding
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_sources import CHECKS, GitHubReadSession, ReviewGitHubSource, ci_result, verify_repository

HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40
PIN = {"github_id": 1, "github_owner_id": 2, "name": "actions/checkout"}
FILE = {
    "filename": "src/版本.py",
    "status": "modified",
    "additions": 1,
    "deletions": 1,
    "patch": "@@ -1 +1 @@\n-old = 1\n+new = 2",
}


def fixture_transport(
    *, drift: bool = False, ci: str = "success", parents: list[str] | None = None, files: list[dict] | None = None
) -> httpx.MockTransport:
    """Control all public GET responses and refuse any unexpected URL or method."""
    calls: dict[str, int] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.host == "api.github.com"
        path = request.url.path
        calls[path] = calls.get(path, 0) + 1
        items = deepcopy(files if files is not None else [FILE])
        if path == "/repositories/1" or path == "/repos/actions/checkout":
            data = {"id": 1, "owner": {"id": 2}, "full_name": "actions/checkout", "private": False}
        elif path.endswith("/pulls/1"):
            data = {
                "base": {"sha": BASE},
                "head": {"sha": "d" * 40 if drift and calls[path] > 1 else HEAD},
                "changed_files": len(items),
            }
        elif "/compare/" in path:
            data = {"merge_base_commit": {"sha": MERGE}, "files": items}
        elif path.endswith("/pulls/1/files"):
            data = items
        elif path.endswith("check-runs"):
            if ci == "unavailable":
                return httpx.Response(403)
            runs = (
                []
                if ci == "missing"
                else [
                    {
                        "id": index,
                        "app": {"id": 9},
                        "name": name,
                        "status": "in_progress" if ci == "pending" else "completed",
                        "conclusion": ci,
                        "head_sha": HEAD,
                    }
                    for index, name in enumerate(CHECKS, 1)
                ]
            )
            data = {"check_runs": runs, "total_count": len(runs)}
        elif "/commits/" in path:
            data = {
                "sha": HEAD,
                "parents": [{"sha": sha} for sha in (parents if parents is not None else [BASE])],
                "files": items,
            }
        elif "/git/ref/heads/" in path:
            branch = path.split("/heads/")[1]
            data = {"ref": "refs/heads/" + branch, "object": {"sha": BASE if branch == "main" else HEAD}}
        else:
            raise AssertionError(path)
        return httpx.Response(200, json=data)

    return httpx.MockTransport(handle)


@pytest.mark.parametrize("ci", ["success", "failure", "pending", "missing", "unavailable"])
def test_ci_outcome_does_not_discard_collected_diff(ci: str) -> None:
    """Failed/pending/missing checks remain orthogonal to readonly code input."""
    binding = ReviewBinding.model_validate({"repository_id": "checkout", "selector": {"kind": "pr", "pr_number": 1}})
    result = asyncio.run(ReviewGitHubSource(fixture_transport(ci=ci)).collect(binding, PIN))
    assert result.comparison.diff_base_sha == MERGE
    assert result.files[0].hunks[0].lines[1].text == "new = 2"
    assert result.ci.checks_passed == (ci == "success")
    assert result.coverage.content_completeness == "complete"


def test_collection_drift_aborts_publication() -> None:
    """A moving head cannot produce a mixed-time partial snapshot."""
    binding = ReviewBinding.model_validate({"repository_id": "checkout", "selector": {"kind": "pr", "pr_number": 1}})
    with pytest.raises(DomainError, match="source_changed_during_collection"):
        asyncio.run(ReviewGitHubSource(fixture_transport(drift=True)).collect(binding, PIN))


@pytest.mark.parametrize(
    "parents,parent,reason",
    [
        ([], None, None),
        ([BASE], None, None),
        ([BASE, MERGE], None, "parent_selection_required"),
        ([BASE, MERGE], "e" * 40, "parent_selection_invalid"),
    ],
)
def test_commit_parent_selection(parents: list[str], parent: str | None, reason: str | None) -> None:
    """Root, ordinary and merge commits never acquire an implicit wrong base."""
    binding = ReviewBinding.model_validate(
        {"repository_id": "checkout", "selector": {"kind": "commit", "commit_sha": HEAD, "parent_sha": parent}}
    )

    async def collect() -> object:
        return await ReviewGitHubSource(fixture_transport(parents=parents)).collect(binding, PIN)

    if reason:
        with pytest.raises(DomainError, match=reason):
            asyncio.run(collect())
    else:
        result = asyncio.run(collect())
        assert result.comparison.parent_sha == (parents[0] if parents else None)


def test_branch_merge_base_and_upstream_cap() -> None:
    """Compare API's file cap cannot be bypassed by guessing total pages."""
    binding = ReviewBinding.model_validate(
        {
            "repository_id": "checkout",
            "selector": {"kind": "branch_compare", "base_branch": "main", "head_branch": "topic"},
        }
    )
    files = [{**FILE, "filename": f"f{index:03}.py"} for index in range(300)]
    result = asyncio.run(ReviewGitHubSource(fixture_transport(files=files)).collect(binding, PIN))
    assert result.comparison.diff_base_sha == MERGE and result.comparison.base_tip_sha == BASE
    assert result.coverage.total_files is None and result.coverage.manifest_completeness == "partial"


def test_public_repository_identity_and_latest_checks() -> None:
    """An extra failed latest check or owner transfer defeats rehearsal qualification."""
    with pytest.raises(DomainError, match="repository_identity_changed"):
        verify_repository({"private": False, "id": 1, "owner": {"id": 8}, "full_name": "actions/checkout"}, PIN)
    rows = [
        {"name": name, "id": i, "app": {"id": 9}, "head_sha": HEAD, "status": "completed", "conclusion": "success"}
        for i, name in enumerate(CHECKS, 1)
    ]
    assert ci_result(HEAD, rows, True).checks_passed
    rows.append({**rows[0], "name": "additional_check", "id": 20, "conclusion": "failure"})
    assert not ci_result(HEAD, rows, True).checks_passed
    rows.append({**rows[0], "id": 21, "conclusion": "skipped"})
    assert not ci_result(HEAD, rows, True).checks_passed


def test_decoded_response_limit_and_redirect_refusal() -> None:
    """HTTP response bytes are bounded before parsing or following any URL."""

    async def read(transport: httpx.MockTransport) -> None:
        async with httpx.AsyncClient(
            base_url="https://api.github.com", transport=transport, follow_redirects=False
        ) as client:
            await GitHubReadSession(client).get("/repositories/1")

    with pytest.raises(DomainError, match="source_byte_limit"):
        asyncio.run(read(httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 1048577))))
    with pytest.raises(DomainError, match="source_unavailable"):
        asyncio.run(
            read(
                httpx.MockTransport(
                    lambda request: httpx.Response(302, headers={"location": "https://unapproved.invalid"})
                )
            )
        )
