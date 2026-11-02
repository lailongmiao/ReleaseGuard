"""Approved GitHub reads with stream budgets, fixed SHA identity and honest partials."""

import asyncio
import base64
import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.schemas.v11_backend import (
    BranchSelector,
    CheckFact,
    CIObservation,
    CollectedReview,
    CommitSelector,
    Comparison,
    Coverage,
    PRSelector,
    ReviewBinding,
)
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_diff import TOTAL_BYTES, read_file, rebuild_patch

REPOSITORIES = {"checkout": "actions/checkout", "setup-python": "actions/setup-python"}
CHECKS = (
    "build",
    "test (ubuntu-latest)",
    "test (macos-latest)",
    "test (windows-latest)",
    "test-proxy",
    "test-bypass-proxy",
    "test-git-container",
    "test-output",
    "check-dist",
    "Check licenses",
    "Analyze (javascript)",
)


def checked_sha(value: Any) -> str:
    """Validate remote SHA values before constructing another API path."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise DomainError("source_invalid_sha")
    return value


class GitHubReadSession:
    """Per-operation resource counters, including unsuccessful transport attempts."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Counters are never shared between concurrent object collections."""
        self.client, self.attempts, self.bytes, self.content_gets = client, 0, 0, 0

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    async def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, bool]:
        """Stream decoded bytes before JSON parsing and reject redirects or errors."""
        if not path.startswith(("/repos/", "/repositories/")) or "?" in path or "://" in path:
            raise DomainError("source_path_invalid")
        if self.attempts >= 24:
            raise DomainError("source_request_limit")
        self.attempts += 1
        async with self.client.stream("GET", path, params=params) as response:
            if response.status_code != 200:
                limited = response.status_code == 429 or (
                    response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0"
                )
                raise DomainError(
                    "rate_limited"
                    if limited
                    else {
                        401: "permission_denied",
                        403: "permission_denied",
                        404: "not_found",
                        409: "unrelated_histories",
                    }.get(response.status_code, "source_unavailable")
                )
            body = bytearray()
            async for block in response.aiter_bytes():
                self.bytes += len(block)
                if len(body) + len(block) > 1048576 or self.bytes > 8388608:
                    raise DomainError("source_byte_limit")
                body.extend(block)
            try:
                return json.loads(body), 'rel="next"' in response.headers.get("link", "")
            except (ValueError, UnicodeError):
                raise DomainError("source_invalid_response") from None

    async def pages(self, path: str, key: str | None = None) -> tuple[list[dict], bool, int | None]:
        """Read at most five pages and explicitly preserve unknown totals."""
        rows: list[dict] = []
        expected: int | None = None
        for page in range(1, 6):
            raw, more = await self.get(path, {"per_page": 100, "page": page})
            if key:
                if not isinstance(raw, dict):
                    raise DomainError("source_invalid_response")
                items = raw.get(key)
                total = raw.get("total_count")
                if total is not None and (type(total) is not int or total < 0 or expected not in {None, total}):
                    raise DomainError("source_pagination_conflict")
                expected = total
            else:
                items = raw
            if not isinstance(items, list) or len(items) > 100 or any(not isinstance(i, dict) for i in items):
                raise DomainError("source_invalid_response")
            rows.extend(items)
            if not more:
                return rows, expected is None or expected == len(rows), expected if expected is not None else len(rows)
        return rows, False, expected


def verify_repository(raw: dict, pinned: dict | None) -> dict:
    """Stable identity, public visibility and owner are distinct checks."""
    if not isinstance(raw, dict) or not isinstance(raw.get("owner"), dict):
        raise DomainError("repository_identity_invalid")
    if raw.get("private") is not False or raw.get("visibility", "public") != "public":
        raise DomainError("repository_not_public")
    identity = raw.get("id"), raw.get("owner", {}).get("id")
    if any(type(n) is not int or n <= 0 for n in identity):
        raise DomainError("repository_identity_invalid")
    if pinned and identity != (pinned["github_id"], pinned["github_owner_id"]):
        raise DomainError("repository_identity_changed")
    name = raw.get("full_name")
    if not isinstance(name, str) or len(name.split("/")) != 2 or any(not p for p in name.split("/")):
        raise DomainError("repository_identity_invalid")
    return {"github_id": identity[0], "github_owner_id": identity[1], "name": name}


def ci_result(head: str, rows: list[dict], complete: bool) -> CIObservation:
    """Never let missing, ambiguous, neutral or skipped checks authorize preparation."""
    at = datetime.now(UTC)
    latest: dict[str, CheckFact] = {}
    for row in rows:
        fact = CheckFact(
            name=row["name"],
            app_id=row.get("app", {}).get("id", 0),
            check_id=row["id"],
            status=row["status"],
            conclusion=row.get("conclusion"),
            head_sha=row["head_sha"],
        )
        if any(ord(c) < 32 for c in fact.name) or fact.head_sha != head:
            raise DomainError("check_sha_conflict")
        if fact.name in latest and latest[fact.name].app_id != fact.app_id:
            raise DomainError("check_context_conflict")
        if fact.name not in latest or fact.check_id > latest[fact.name].check_id:
            latest[fact.name] = fact
    checks = [latest[n] for n in sorted(latest)]
    passed = (
        bool(checks)
        and complete
        and set(CHECKS) <= latest.keys()
        and all(c.status == "completed" and c.conclusion == "success" for c in checks)
    )
    status = (
        "missing"
        if not checks
        else "pending"
        if any(c.status != "completed" for c in checks)
        else ("success" if all(c.conclusion == "success" for c in checks) else "failed")
    )
    return CIObservation.model_validate(
        {
            "head_sha": head,
            "status": status,
            "checks": checks,
            "checks_passed": passed,
            "complete": complete,
            "captured_at": at,
            "reasons": [] if complete else ["checks_truncated"],
        }
    )


class ReviewGitHubSource:
    """Only the two server-approved public repository identities can be queried."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Tests explicitly inject a network-free transport."""
        self.transport = transport

    def client(self) -> httpx.AsyncClient:
        """Use the existing private server credential solely at api.github.com."""
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "releaseguard-readonly-v11"}
        token = settings.V1_GITHUB_READ_TOKEN.get_secret_value()
        if token:
            headers["Authorization"] = "Bearer " + token
        return httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=15,
            trust_env=False,
            follow_redirects=False,
            transport=self.transport,
        )

    async def identity(self, repository_id: str, pinned: dict | None = None) -> dict:
        """Initial binding is by fixed name; later reads use the pinned numeric ID."""
        if repository_id not in REPOSITORIES:
            raise DomainError("source_not_found")
        async with self.client() as client, asyncio.timeout(60):
            path = f"/repositories/{pinned['github_id']}" if pinned else "/repos/" + REPOSITORIES[repository_id]
            raw, _ = await GitHubReadSession(client).get(path)
            return verify_repository(raw, pinned)

    async def discover(
        self,
        repository_id: str,
        pinned: dict,
        kind: str,
        page: int,
        limit: int,
        state: str = "all",
        sha: str | None = None,
    ) -> tuple[list[dict], bool]:
        """Malformed or interrupted discovery never leaks raw upstream errors."""
        try:
            return await self._discover(repository_id, pinned, kind, page, limit, state, sha)
        except (httpx.HTTPError, TimeoutError):
            raise DomainError("source_timeout_or_unavailable") from None
        except DomainError:
            raise
        except (ValueError, KeyError, TypeError):
            raise DomainError("source_invalid_response") from None

    async def _discover(
        self,
        repository_id: str,
        pinned: dict,
        kind: str,
        page: int,
        limit: int,
        state: str = "all",
        sha: str | None = None,
    ) -> tuple[list[dict], bool]:
        """Live upstream pages have no cross-request snapshot guarantee."""
        if repository_id not in REPOSITORIES or not 1 <= page <= 10 or not 1 <= limit <= 50:
            raise DomainError("discovery_input_invalid")
        async with self.client() as client, asyncio.timeout(60):
            session = GitHubReadSession(client)
            raw, _ = await session.get(f"/repositories/{pinned['github_id']}")
            identity = verify_repository(raw, pinned)
            prefix = "/repos/" + "/".join(quote(p, safe="") for p in identity["name"].split("/"))
            endpoint = {"pulls": "pulls", "commits": "commits", "branches": "branches"}.get(kind)
            if endpoint is None:
                raise DomainError("discovery_input_invalid")
            params: dict[str, Any] = {"per_page": limit, "page": page}
            if kind == "pulls":
                params.update(state=state, sort="created", direction="desc")
            if sha:
                endpoint += "/" + checked_sha(sha)
            values, more = await session.get(prefix + "/" + endpoint, params)
            values = [values] if sha else values
            if not isinstance(values, list) or len(values) > limit:
                raise DomainError("source_invalid_response")
            items = []
            for value in values:
                if kind == "pulls":
                    items.append(
                        {
                            "pr_number": value["number"],
                            "state": value["state"],
                            "head_sha": checked_sha(value["head"]["sha"]),
                            "base_sha": checked_sha(value["base"]["sha"]),
                        }
                    )
                elif kind == "branches":
                    items.append({"name": value["name"], "sha": checked_sha(value["commit"]["sha"])})
                else:
                    items.append(
                        {
                            "sha": checked_sha(value["sha"]),
                            "parents": [checked_sha(p["sha"]) for p in value["parents"]],
                        }
                    )
            after, _ = await session.get(f"/repositories/{pinned['github_id']}")
            if verify_repository(after, pinned) != identity:
                raise DomainError("source_changed_during_collection")
            return items, more

    async def collect(self, binding: ReviewBinding, pinned: dict) -> CollectedReview:
        """Collect outside database locks; any identity drift aborts publication."""
        try:
            async with self.client() as client, asyncio.timeout(60):
                return await self._collect(GitHubReadSession(client), binding, pinned)
        except (httpx.HTTPError, TimeoutError):
            raise DomainError("source_timeout_or_unavailable") from None
        except DomainError:
            raise
        except (ValueError, KeyError, TypeError):
            raise DomainError("source_invalid_response") from None

    async def resolve(
        self, session: GitHubReadSession, prefix: str, binding: ReviewBinding
    ) -> tuple[Comparison, dict]:
        """Resolve selector SHA identities without silently choosing a merge parent."""
        selector = binding.selector
        if isinstance(selector, PRSelector):
            pr, _ = await session.get(prefix + f"/pulls/{selector.pr_number}")
            base, head = checked_sha(pr["base"]["sha"]), checked_sha(pr["head"]["sha"])
            comparison, _ = await session.get(prefix + f"/compare/{base}...{head}")
            merge = comparison["merge_base_commit"]["sha"]
            return Comparison(
                kind="pr",
                base_tip_sha=base,
                head_sha=head,
                diff_base_sha=merge,
                merge_base_sha=merge,
                comparison="merge_base",
            ), pr
        if isinstance(selector, BranchSelector):
            base, _ = await session.get(prefix + "/git/ref/heads/" + quote(selector.base_branch, safe=""))
            head, _ = await session.get(prefix + "/git/ref/heads/" + quote(selector.head_branch, safe=""))
            base_sha, head_sha = checked_sha(base["object"]["sha"]), checked_sha(head["object"]["sha"])
            if (
                base.get("ref") != "refs/heads/" + selector.base_branch
                or head.get("ref") != "refs/heads/" + selector.head_branch
            ):
                raise DomainError("source_ref_conflict")
            comparison, _ = await session.get(prefix + f"/compare/{base_sha}...{head_sha}")
            merge = comparison["merge_base_commit"]["sha"]
            return Comparison(
                kind="branch_compare",
                base_tip_sha=base_sha,
                head_sha=head_sha,
                diff_base_sha=merge,
                merge_base_sha=merge,
                comparison="merge_base",
            ), comparison
        commit, _ = await session.get(prefix + "/commits/" + selector.commit_sha)
        if commit["sha"] != selector.commit_sha:
            raise DomainError("source_ref_conflict")
        parents = [p["sha"] for p in commit["parents"]]
        if len(parents) > 1 and selector.parent_sha is None:
            raise DomainError("parent_selection_required")
        if selector.parent_sha and selector.parent_sha not in parents:
            raise DomainError("parent_selection_invalid")
        parent = selector.parent_sha or (parents[0] if parents else None)
        return Comparison(
            kind="commit",
            base_kind="commit" if parent else "empty_tree",
            head_sha=selector.commit_sha,
            base_tip_sha=parent,
            diff_base_sha=parent,
            parent_sha=parent,
            comparison="parent" if parent else "empty_tree",
        ), commit

    async def _collect(self, session: GitHubReadSession, binding: ReviewBinding, pinned: dict) -> CollectedReview:
        raw, _ = await session.get(f"/repositories/{pinned['github_id']}")
        identity = verify_repository(raw, pinned)
        prefix = "/repos/" + "/".join(quote(p, safe="") for p in identity["name"].split("/"))
        comparison, metadata = await self.resolve(session, prefix, binding)
        if isinstance(binding.selector, PRSelector):
            rows, complete, total = await session.pages(prefix + f"/pulls/{binding.selector.pr_number}/files")
            total = metadata.get("changed_files")
            complete = complete and type(total) is int and len(rows) == total
        elif isinstance(binding.selector, CommitSelector) and len(metadata["parents"]) <= 1:
            rows, complete, total = await session.pages(prefix + "/commits/" + comparison.head_sha, "files")
        else:
            if comparison.kind == "commit":
                metadata, _ = await session.get(
                    prefix + f"/compare/{comparison.diff_base_sha}...{comparison.head_sha}"
                )
                if metadata["merge_base_commit"]["sha"] != comparison.diff_base_sha:
                    raise DomainError("comparison_base_conflict")
            rows = metadata.get("files")
            if not isinstance(rows, list):
                raise DomainError("source_invalid_response")
            complete, total = len(rows) < 300, len(rows) if len(rows) < 300 else None
        if len(rows) > 500 or len({r.get("filename") for r in rows}) != len(rows):
            raise DomainError("source_file_manifest_invalid")
        files, remaining = [], TOTAL_BYTES
        for raw_file in sorted(rows, key=lambda r: r["filename"]):
            file = read_file(raw_file, remaining)
            if (
                not file.hunks
                and file.reasons
                and session.content_gets < 8
                and remaining > 0
                and session.attempts < 14
            ):
                file = await self.supplement(session, prefix, comparison, raw_file, remaining)
            remaining -= len(file.patch.encode())
            files.append(file)
        try:
            checks, ci_complete, _ = await session.pages(
                prefix + f"/commits/{comparison.head_sha}/check-runs", "check_runs"
            )
            ci = ci_result(comparison.head_sha, checks, ci_complete)
        except httpx.TransportError:
            ci = CIObservation(
                head_sha=comparison.head_sha,
                status="unavailable",
                complete=False,
                captured_at=datetime.now(UTC),
                reasons=["check_transport_unavailable"],
            )
        except DomainError as error:
            if error.code in {
                "check_sha_conflict",
                "check_context_conflict",
                "source_invalid_response",
                "source_pagination_conflict",
            }:
                raise
            ci = CIObservation(
                head_sha=comparison.head_sha,
                status="unavailable",
                complete=False,
                captured_at=datetime.now(UTC),
                reasons=[error.code],
            )
        after, _ = await self.resolve(session, prefix, binding)
        final_repo, _ = await session.get(f"/repositories/{pinned['github_id']}")
        if verify_repository(final_repo, pinned) != identity or after != comparison:
            raise DomainError("source_changed_during_collection")
        reasons = ([] if complete else ["upstream_file_cap"]) + sorted(
            {r for f in files for r in f.reasons if not f.complete}
        )
        return CollectedReview(
            repository_id=binding.repository_id,
            github_id=identity["github_id"],
            owner_id=identity["github_owner_id"],
            repository_name=identity["name"],
            comparison=comparison,
            files=files,
            ci=ci,
            captured_at=datetime.now(UTC),
            coverage=Coverage(
                manifest_completeness="complete" if complete else "partial",
                content_completeness="complete" if complete and all(f.complete for f in files) else "partial",
                total_files=total,
                observed_files=len(files),
                read_files=sum(bool(f.hunks) for f in files),
                read_changed_lines=sum(line.kind != "context" for f in files for h in f.hunks for line in h.lines),
                reasons=reasons,
            ),
        )

    async def supplement(
        self, session: GitHubReadSession, prefix: str, comparison: Comparison, raw_file: dict, remaining: int
    ) -> Any:
        """Read only bounded text at exact SHA; unknown kinds stay explicit."""
        file = read_file(raw_file, remaining)
        values = []
        for sha, path, absent in (
            (comparison.diff_base_sha, file.old_path or file.path, file.status == "added"),
            (comparison.head_sha, file.path, file.status == "removed"),
        ):
            if sha is None or absent:
                values.append("")
                continue
            if session.content_gets >= 8 or session.attempts >= 15:
                return file
            try:
                raw = await self.text_blob(session, prefix, sha, path)
                if raw.get("encoding") != "base64" or type(raw.get("size")) is not int or raw["size"] > 65536:
                    return file.model_copy(update={"reasons": ["content_byte_limit"]})
                data = base64.b64decode("".join(raw["content"].split()), validate=True)
                if len(data) != raw["size"] or len(data) > 65536:
                    raise DomainError("source_invalid_response")
                if b"\0" in data:
                    return file.model_copy(update={"reasons": ["binary_content"]})
                values.append(data.decode("utf-8"))
            except UnicodeError:
                return file.model_copy(update={"reasons": ["unsupported_encoding"]})
            except DomainError as error:
                if error.code in {"source_invalid_response", "repository_identity_changed"}:
                    raise
                return file.model_copy(update={"reasons": [error.code, "patch_missing_unclassified"]})
        patch = await asyncio.to_thread(rebuild_patch, *values)
        if not patch:
            return file
        return read_file({**raw_file, "patch": patch}, remaining)

    async def text_blob(self, session: GitHubReadSession, prefix: str, commit_sha: str, path: str) -> dict:
        """Resolve exact Git tree entries; never use the symlink-following contents API."""

        async def get(suffix: str) -> Any:
            if session.content_gets >= 8 or session.attempts >= 15:
                raise DomainError("content_read_limit")
            session.content_gets += 1
            value, _ = await session.get(prefix + suffix)
            return value

        commit = await get("/git/commits/" + checked_sha(commit_sha))
        if checked_sha(commit["sha"]) != commit_sha:
            raise DomainError("source_invalid_response")
        tree_sha = checked_sha(commit["tree"]["sha"])
        components = path.split("/")
        for index, component in enumerate(components):
            tree = await get("/git/trees/" + tree_sha)
            if (
                tree.get("truncated") is not False
                or tree.get("sha") != tree_sha
                or not isinstance(tree.get("tree"), list)
            ):
                raise DomainError("tree_incomplete_or_invalid")
            entries = [entry for entry in tree["tree"] if entry.get("path") == component]
            if len(entries) != 1:
                raise DomainError("source_invalid_response")
            entry = entries[0]
            tree_sha = checked_sha(entry["sha"])
            if index < len(components) - 1:
                if entry.get("type") != "tree" or entry.get("mode") != "040000":
                    raise DomainError("unsupported_file_kind")
            elif entry.get("type") != "blob" or entry.get("mode") not in {"100644", "100755"}:
                raise DomainError("unsupported_file_kind")
            elif type(entry.get("size")) is not int or not 0 <= entry["size"] <= 65536:
                raise DomainError("content_byte_limit")
        blob = await get("/git/blobs/" + tree_sha)
        if blob.get("sha") != tree_sha:
            raise DomainError("source_invalid_response")
        return blob
