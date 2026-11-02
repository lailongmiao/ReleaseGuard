"""Comparison and source-cap counterexamples over bounded local HTTP transports."""

import asyncio
import base64
import json

import httpx
import pytest

from app.schemas.v11_backend import ReviewBinding
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_sources import GitHubReadSession, ReviewGitHubSource
from tests.test_v11_sources import BASE, FILE, HEAD, MERGE, PIN, fixture_transport


@pytest.mark.parametrize("mode", ["reverse", "identical", "deleted", "unrelated", "moving", "merge_parent"])
def test_fixed_comparison_semantics(mode: str) -> None:
    """Do not silently turn endpoint comparisons or unstable refs into valid snapshots."""
    transport = fixture_transport(parents=[BASE, MERGE])
    seen = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        path = request.url.path
        if "/git/ref/heads/" in path:
            seen += 1
            if mode == "deleted":
                return httpx.Response(404)
            name = path.split("/heads/")[1]
            sha = HEAD if mode == "identical" else BASE if name == "main" else HEAD
            if mode == "moving" and seen > 2:
                sha = "d" * 40
            return httpx.Response(200, json={"ref": "refs/heads/" + name, "object": {"sha": sha}})
        if "/compare/" in path:
            if mode == "unrelated":
                return httpx.Response(409)
            if mode == "reverse":
                assert path.endswith(HEAD + "..." + BASE)
            return httpx.Response(
                200,
                json={
                    "merge_base_commit": {"sha": HEAD if mode == "identical" else MERGE},
                    "files": [] if mode == "identical" else [FILE],
                },
            )
        if mode == "reverse" and path.endswith("check-runs"):
            return httpx.Response(200, json={"check_runs": [], "total_count": 0})
        return transport.handle_request(request)

    selector = (
        {"kind": "commit", "commit_sha": HEAD, "parent_sha": MERGE}
        if mode == "merge_parent"
        else {
            "kind": "branch_compare",
            "base_branch": "topic" if mode == "reverse" else "main",
            "head_branch": "main" if mode == "reverse" else "topic",
        }
    )
    binding = ReviewBinding.model_validate({"repository_id": "checkout", "selector": selector})
    coroutine = ReviewGitHubSource(httpx.MockTransport(respond)).collect(binding, PIN)
    if mode in {"deleted", "unrelated", "moving"}:
        reason = {
            "deleted": "not_found",
            "unrelated": "unrelated_histories",
            "moving": "source_changed_during_collection",
        }[mode]
        with pytest.raises(DomainError, match=reason):
            asyncio.run(coroutine)
    else:
        result = asyncio.run(coroutine)
        assert result.comparison.head_sha == (BASE if mode == "reverse" else HEAD)
        if mode == "identical":
            assert result.files == [] and result.coverage.total_files == 0
        else:
            assert result.comparison.diff_base_sha == MERGE


@pytest.mark.parametrize("total", [499, 500, 501])
def test_file_page_boundary_preserves_known_incomplete_count(total: int) -> None:
    """N+1 files cannot be silently reported as a complete N-file manifest."""

    def respond(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        first = (page - 1) * 100
        rows = [{"filename": str(i)} for i in range(first, min(first + 100, total))]
        headers = {"link": '<https://api.github.com/ignored>; rel="next"'} if first + 100 < total else {}
        return httpx.Response(200, json={"files": rows, "total_count": total}, headers=headers)

    async def run() -> tuple:
        async with httpx.AsyncClient(
            base_url="https://api.github.com", transport=httpx.MockTransport(respond)
        ) as client:
            session = GitHubReadSession(client)
            rows, complete, count = await session.pages("/repos/actions/checkout/commits/" + HEAD, "files")
            return len(rows), complete, count, session.attempts

    assert asyncio.run(run()) == (min(total, 500), total <= 500, total, 5)


@pytest.mark.parametrize("size", [1048575, 1048576, 1048577])
def test_decoded_byte_boundary(size: int) -> None:
    """The exact decompressed byte count controls acceptance before JSON parsing."""

    async def run() -> None:
        data = json.dumps("x" * (size - 2)).encode()
        async with httpx.AsyncClient(
            base_url="https://api.github.com",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data)),
        ) as client:
            await GitHubReadSession(client).get("/repositories/1")

    if size > 1048576:
        with pytest.raises(DomainError, match="source_byte_limit"):
            asyncio.run(run())
    else:
        asyncio.run(run())


def test_discovery_rechecks_repository_after_page() -> None:
    """Public source discovery cannot return metadata after owner identity changed."""
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/repositories/1":
            calls += 1
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "owner": {"id": 2 if calls == 1 else 3},
                    "private": False,
                    "full_name": "actions/checkout",
                },
            )
        return httpx.Response(200, json=[{"sha": HEAD, "parents": [{"sha": BASE}]}])

    with pytest.raises(DomainError, match="repository_identity_changed"):
        asyncio.run(ReviewGitHubSource(httpx.MockTransport(respond)).discover("checkout", PIN, "commits", 1, 20))


@pytest.mark.parametrize("mode", ["100644", "100755", "120000", "160000"])
def test_blob_fallback_never_follows_symlinks_or_submodules(mode: str) -> None:
    """Mode is verified before a blob is read; contents URLs are never consulted."""
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        sent.append(path)
        if "/git/commits/" in path:
            data = {"sha": HEAD, "tree": {"sha": BASE}}
        elif "/git/trees/" in path:
            data = {
                "sha": BASE,
                "truncated": False,
                "tree": [
                    {
                        "path": "x.py",
                        "mode": mode,
                        "type": "commit" if mode == "160000" else "blob",
                        "sha": MERGE,
                        "size": 3,
                    }
                ],
            }
        else:
            assert "/git/blobs/" in path
            data = {"sha": MERGE, "size": 3, "encoding": "base64", "content": base64.b64encode(b"new").decode()}
        return httpx.Response(200, json=data)

    async def run() -> dict:
        async with httpx.AsyncClient(
            base_url="https://api.github.com", transport=httpx.MockTransport(respond)
        ) as client:
            return await ReviewGitHubSource().text_blob(
                GitHubReadSession(client), "/repos/actions/checkout", HEAD, "x.py"
            )

    if mode in {"120000", "160000"}:
        with pytest.raises(DomainError, match="unsupported_file_kind"):
            asyncio.run(run())
        assert not any("/git/blobs/" in path for path in sent)
    else:
        assert asyncio.run(run())["size"] == 3
