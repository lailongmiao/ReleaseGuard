"""Server-pinned readonly adapters, with bounded and sanitized source facts."""

import asyncio
import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.schemas.v1_backend import Collection, SourceDocument
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError

REPOSITORY = "actions/checkout"
SOURCE_POLICY_VERSION = "github-get-sanitized.v1"
LEGACY_MARKDOWN_FILES = ("v1_release_policy.md", "v1_verification.md", "v1_source_security.md")
MARKDOWN_FILES = LEGACY_MARKDOWN_FILES + (
    "checkout_versions.md",
    "checkout_refs.md",
    "checkout_history.md",
    "checkout_credentials.md",
    "checkout_workspace.md",
    "checkout_sparse.md",
    "checkout_dependencies.md",
    "checkout_validation.md",
    "checkout_upgrade.md",
)
ROOT = Path(__file__).resolve().parents[2] / "resources" / "knowledge"
SHA = re.compile(r"[0-9a-f]{40}\Z")


class CorpusEntry(BaseModel):
    """Reviewed file metadata cannot grant business or tool permissions."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    filename: str
    role: str
    sha256: str
    origin: str
    commit: str
    section: str
    license: str
    review: str
    references: list[str] = []


class CorpusManifest(BaseModel):
    """Fixed server allowlist and immutable, normalized content digests."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    content_hash_algorithm: str
    documents: list[CorpusEntry]
    upstream: dict[str, Any]


def corpus_manifest(root: Path = ROOT) -> CorpusManifest:
    """Read a bounded local manifest; caller inputs never select these paths."""
    path = root / "v1_corpus_manifest.json"
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise DomainError("markdown_manifest_unsafe")
    raw = path.read_bytes()
    if len(raw) > 100000:
        raise DomainError("markdown_manifest_unsafe")
    manifest = CorpusManifest.model_validate_json(raw)
    if (
        manifest.version not in {"v1-reviewed-corpus.2", "v1-reviewed-corpus.3"}
        or manifest.content_hash_algorithm != "sha256_utf8_lf"
        or len(manifest.documents) != len(MARKDOWN_FILES)
        or {entry.filename for entry in manifest.documents} != set(MARKDOWN_FILES)
        or any(
            entry.role != ("project_policy" if entry.filename in LEGACY_MARKDOWN_FILES else "usage_reference")
            or not re.fullmatch(r"[0-9a-f]{64}", entry.sha256)
            or not SHA.fullmatch(entry.commit)
            or not entry.review
            for entry in manifest.documents
        )
    ):
        raise DomainError("markdown_manifest_invalid")
    return manifest


def document(source: str, resource: str, version: str, content: str, locator: str, at: datetime) -> SourceDocument:
    """Construct immutable sanitized references without an upstream envelope."""
    digest = canonical_hash(content)
    return SourceDocument(
        document_id="doc:" + canonical_hash({"source": source, "resource": resource, "hash": digest}),
        source_id=source,
        resource=resource,
        version=version,
        content=content,
        content_hash=digest,
        locator=locator,
        captured_at=at,
    )


class GitHubSource:
    """Fixed public GETs; optional server token never grants source ACL access."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Only tests may inject an explicitly isolated transport."""
        self.transport = transport

    async def collect(self, pr_number: int, *, required_checks: tuple[str, ...]) -> Collection:
        """Check PR head twice and collect complete, bounded current-head checks."""
        if pr_number < 1:
            raise DomainError("source_resource_invalid")
        try:
            headers = {"Accept": "application/vnd.github+json", "User-Agent": "releaseguard-readonly-v1"}
            token = settings.V1_GITHUB_READ_TOKEN.get_secret_value()
            if token:
                headers["Authorization"] = "Bearer " + token
            async with httpx.AsyncClient(
                base_url="https://api.github.com/repos/" + REPOSITORY,
                transport=self.transport,
                follow_redirects=False,
                timeout=15,
                trust_env=False,
                headers=headers,
            ) as client:
                async with asyncio.timeout(60):
                    return await self._collect(client, pr_number, required_checks)
        except DomainError as error:
            status = "permission_denied" if error.code == "permission_denied" else "unavailable"
            if error.code in {"head_changed", "check_sha_conflict", "check_context_conflict"}:
                status = "conflicting"
            return Collection.model_validate({"status": status, "reason": error.code})
        except (httpx.HTTPError, TimeoutError):
            return Collection(status="unavailable", reason="source_timeout_or_unavailable")
        except (ValueError, KeyError, TypeError):
            return Collection(status="unavailable", reason="source_invalid_response")

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    async def _get(self, client: httpx.AsyncClient, path: str, page: int = 1) -> tuple[Any, bool]:
        """Retry transport failures of a fixed GET, within the outer 60s bound."""
        response = await client.get(path, params={"per_page": 100, "page": page})
        if response.status_code != 200:
            if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
                raise DomainError("rate_limited")
            raise DomainError(
                {401: "permission_denied", 403: "permission_denied", 404: "not_found", 429: "rate_limited"}.get(
                    response.status_code, "source_unavailable"
                )
            )
        if len(response.content) > 1024 * 1024:
            raise DomainError("source_byte_limit")
        return response.json(), 'rel="next"' in response.headers.get("link", "")

    async def _pages(self, client: httpx.AsyncClient, path: str, key: str | None = None) -> list[dict]:
        values: list[dict] = []
        expected = None
        for page in range(1, 6):
            raw, more = await self._get(client, path, page)
            if key:
                expected = raw["total_count"]
                items = raw[key]
            else:
                items = raw
            if not isinstance(items, list):
                raise DomainError("source_invalid_response")
            values.extend(items)
            if not more:
                if expected is not None and expected != len(values):
                    raise DomainError("truncated")
                return values
        raise DomainError("truncated")

    async def _collect(self, client: httpx.AsyncClient, number: int, required: tuple[str, ...]) -> Collection:
        pr, _ = await self._get(client, f"pulls/{number}")
        head = pr["head"]["sha"]
        if not SHA.fullmatch(head):
            raise DomainError("source_invalid_sha")
        checks = await self._pages(client, f"commits/{head}/check-runs", "check_runs")
        files = await self._pages(client, f"pulls/{number}/files")
        # Only three release metadata rows, not descriptions, authors, or assets.
        releases, _ = await self._get(client, "releases")
        after, _ = await self._get(client, f"pulls/{number}")
        if after["head"]["sha"] != head:
            raise DomainError("head_changed")
        if not checks:
            return Collection(status="unavailable", reason="missing_checks", head_sha=head)
        if any(item["head_sha"] != head for item in checks):
            raise DomainError("check_sha_conflict")
        latest: dict[str, dict] = {}
        apps: dict[str, int] = {}
        for item in checks:
            name = str(item["name"])
            if len(name) > 128 or "\n" in name:
                raise DomainError("source_invalid_response")
            app = int(item.get("app", {}).get("id", 0))
            if name in apps and apps[name] != app:
                raise DomainError("check_context_conflict")
            apps[name] = app
            if name not in latest or int(item["id"]) > int(latest[name]["id"]):
                latest[name] = item
        check_facts = [
            {"name": name, "status": item["status"], "conclusion": item.get("conclusion"), "head_sha": head}
            for name, item in sorted(latest.items())
        ]
        checks_passed = (
            bool(required)
            and set(required) <= latest.keys()
            and all(item["status"] == "completed" and item.get("conclusion") == "success" for item in latest.values())
        )
        facts = {
            "repository": REPOSITORY,
            "source_policy_version": SOURCE_POLICY_VERSION,
            "check_policy_hash": canonical_hash({"required": list(required), "all_observed_latest_success": True}),
            "pr_number": number,
            "head_sha": head,
            "complete": True,
            "checks": check_facts,
            "required_checks": list(required),
            "checks_passed": checks_passed,
            "files": sorted(str(item["filename"]) for item in files),
            "release_tags": [str(item["tag_name"])[:128] for item in releases[:3]],
        }
        at = datetime.now(UTC)
        text = json.dumps(facts, sort_keys=True, ensure_ascii=False)
        return Collection(
            status="verified",
            documents=[
                document(
                    "github-checkout", f"pr:{number}", head, text, f"https://github.com/{REPOSITORY}/pull/{number}", at
                )
            ],
            head_sha=head,
            checks_passed=checks_passed,
            check_names=tuple(sorted(latest)),
            reason=None if checks_passed else "checks_failed_pending_or_policy_missing",
        )


class MarkdownSource:
    """Only reviewed, hash-bound files; no request-controlled filesystem paths."""

    def __init__(self, root: Path = ROOT) -> None:
        """An isolated root can be injected by tests or the local launcher."""
        self.root = root.resolve()

    async def collect(self) -> Collection:
        """Read bounded files off the event loop and record their actual hashes."""
        try:
            return await asyncio.to_thread(self._read)
        except (OSError, ValueError, DomainError):
            return Collection(status="unavailable", reason="markdown_unavailable_or_unsafe")

    async def manifest_digest(self) -> str:
        """Bind source metadata as well as content to each new generation."""
        try:
            manifest = await asyncio.to_thread(corpus_manifest, self.root)
        except (OSError, ValueError):
            raise DomainError("markdown_unavailable_or_unsafe") from None
        return canonical_hash(manifest)

    def _read(self) -> Collection:
        at, documents = datetime.now(UTC), []
        manifest = {entry.filename: entry for entry in corpus_manifest(self.root).documents}
        for filename in MARKDOWN_FILES:
            original = self.root / filename
            path = original.resolve()
            if path.parent != self.root or original.is_symlink():
                raise DomainError("markdown_path_denied")
            raw = path.read_bytes()
            if not raw or len(raw) > 100000:
                raise DomainError("markdown_byte_limit")
            text = raw.decode("utf-8").replace("\r\n", "\n")
            if sha256(text.encode("utf-8")).hexdigest() != manifest[filename].sha256:
                raise DomainError("markdown_manifest_content_changed")
            documents.append(
                document("markdown-rules", filename, canonical_hash(text), text, "knowledge:" + filename, at)
            )
        return Collection(status="verified", documents=documents, checks_passed=True)
