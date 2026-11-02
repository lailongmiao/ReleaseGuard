"""Deterministic synthetic source/Provider fixtures, never real quality evidence."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from typing import override

import httpx

from app.schemas.local_mvp import LocalActor
from app.services.llm.review_wait import BUSINESS_WAIT_POLICY, BusinessWaitPolicy
from app.schemas.v11_backend import ReviewCitation, ReviewOutput
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_sources import CHECKS, ReviewGitHubSource
from app.services.v11_storage import V11Store

HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


def source(*, complete_evidence: bool = False) -> ReviewGitHubSource:
    """Synthetic PR 2 has failing CI; PR 3 is empty; no socket is opened."""
    head_kind: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method != "GET" or request.url.host != "api.github.com":
            raise AssertionError("unexpected_source_request")
        path = request.url.path
        repository = "setup-python" if "setup-python" in path or path.startswith("/repositories/12") else "checkout"
        repo_id = 12 if repository == "setup-python" else 11
        files = [
            {
                "filename": "src/example.py",
                "status": "modified",
                "additions": 1,
                "deletions": 1,
                "patch": "@@ -1 +1 @@\n-limit = 1\n+limit = 2",
            }
        ]
        if complete_evidence:
            files = [
                {
                    "filename": "package-lock.json",
                    "status": "modified",
                    "additions": 2,
                    "deletions": 2,
                    "patch": '@@ -1,2 +1,2 @@\n-"resolved": "https://registry.npmjs.org/example/-/example-1.0.0.tgz",\n-"integrity": "sha512-'
                    + "A" * 86
                    + '=="\n+"resolved": "https://registry.npmjs.org/example/-/example-2.0.0.tgz",\n+"integrity": "sha512-'
                    + "B" * 86
                    + '=="',
                }
            ]
        if path in {"/repositories/11", "/repositories/12", "/repos/actions/checkout", "/repos/actions/setup-python"}:
            data: Any = {"id": repo_id, "owner": {"id": 10}, "private": False, "full_name": "actions/" + repository}
        elif path.endswith("/pulls"):
            data = [
                {
                    "number": n,
                    "title": f"Synthetic PR {n}",
                    "state": "open",
                    "base": {"sha": BASE},
                    "head": {"sha": HEAD},
                }
                for n in (1, 2, 3)
            ]
        elif path.endswith("/branches"):
            data = [
                {"name": name, "commit": {"sha": BASE if name == "main" else HEAD}} for name in ("main", "feature")
            ]
        elif path.endswith("/commits"):
            data = [
                {"sha": HEAD, "parents": [{"sha": BASE}]},
                {"sha": "f" * 40, "parents": [{"sha": BASE}, {"sha": MERGE}]},
            ]
        elif "/pulls/" in path:
            number = int(path.split("/pulls/")[1].split("/")[0])
            head = ("d" if number == 2 else "e" if number == 3 else "9" if number == 4 else "a") * 40
            head_kind[head] = number
            # Explicit experience-only object; discovery stays unchanged for existing fixtures.
            selected_files = (
                files + [{"filename": "docs/证据/缺失.md", "status": "modified", "additions": 1, "deletions": 1}]
                if number == 4
                else files
            )
            data = (
                ([] if number == 3 else selected_files)
                if path.endswith("/files")
                else {
                    "base": {"sha": BASE},
                    "head": {"sha": head},
                    "changed_files": 0 if number == 3 else len(selected_files),
                }
            )
        elif "/git/commits/" in path:
            return httpx.Response(404, json={"message": "Synthetic original content unavailable"})
        elif "/compare/" in path:
            data = {
                "merge_base_commit": {
                    "sha": path.split("/compare/")[1].split("...")[0] if path.endswith("f" * 40) else MERGE
                },
                "files": files,
            }
        elif path.endswith("check-runs"):
            head = path.split("/commits/")[1].split("/")[0]
            checks = [
                {
                    "name": name,
                    "id": index,
                    "app": {"id": 5},
                    "head_sha": head,
                    "status": "completed",
                    "conclusion": "failure" if head_kind.get(head) == 2 else "success",
                }
                for index, name in enumerate(CHECKS, 1)
            ]
            data = {"check_runs": checks, "total_count": len(checks)}
        elif "/commits/" in path:
            head = path.split("/commits/")[1]
            data = {
                "sha": head,
                "parents": [{"sha": BASE}, {"sha": MERGE}] if head == "f" * 40 else [{"sha": BASE}],
                "files": files,
            }
        elif "/git/ref/heads/" in path:
            branch = path.split("/heads/")[1]
            data = {"ref": "refs/heads/" + branch, "object": {"sha": BASE if branch == "main" else HEAD}}
        else:
            raise AssertionError("unexpected_source_path")
        return httpx.Response(200, json=data)

    return ReviewGitHubSource(httpx.MockTransport(handler))


class ProbeProvider:
    """Synthetic reservations exercise the actual PG ledger but send zero paid requests."""

    provenance = "unverified"

    def __init__(self, store: V11Store) -> None:
        """Inject the real isolated storage boundary, never production credentials."""
        self.store = store

    async def embed(self, texts: list[str], actor: LocalActor, command: str, attempt: str) -> list[list[float]]:
        """A deterministic exact-size embedding has explicitly synthetic provenance."""
        await self.store.reserve(actor, command, attempt + "-batch-0", Decimal("0.00000001"), Decimal("100"))
        await self.store.usage(
            actor, command, attempt + "-batch-0", {"provenance": "synthetic_no_provider_send"}, Decimal("0")
        )
        return [[1.0, *([0.0] * 1535)] for _ in texts]

    async def review(
        self,
        user: str,
        citations: list[ReviewCitation],
        actor: LocalActor,
        command: str,
        analysis: str,
    ) -> tuple[ReviewOutput, dict]:
        """Explicit fault query preserves the reservation and never returns a report."""
        await self.store.reserve(actor, command, "analysis-1", Decimal("0.00000001"), Decimal("100"))
        if '"query": "probe-unknown"' in user:
            raise DomainError("analysis_outcome_unknown")
        if '"query": "probe-slow"' in user:
            await asyncio.sleep(0.25)
        await self.store.usage(
            actor, command, "analysis-1", {"provenance": "synthetic_no_provider_send"}, Decimal("0")
        )
        citation = next(c for c in citations if c.kind == "diff" and c.side == "RIGHT")
        return ReviewOutput.model_validate(
            {
                "summary": "The added line sets limit to 2.",
                "observations": [
                    {
                        "text": "The added line sets limit to 2.",
                        "kind": "observation",
                        "citation_ids": [citation.citation_id],
                    }
                ],
                "risks": [],
                "validation_suggestions": [],
                "unknowns": [],
                "candidate_steps": ["validate_inputs", "simulate", "verify_predictions"],
                "model_risk": "LOW",
            }
        ), {"provenance": "synthetic_no_provider_send", "cost_cny": "not_applicable"}


class BusinessProbeProvider(ProbeProvider):
    """New business signature without rewriting archived fixture subclasses."""

    @override
    async def review(
        self,
        user: str,
        citations: list[ReviewCitation],
        actor: LocalActor,
        command: str,
        analysis: str,
        *,
        business_wait_policy: BusinessWaitPolicy | None = None,
    ) -> tuple[ReviewOutput, dict]:
        """Verify current business dispatch before the existing synthetic operation."""
        if business_wait_policy != BUSINESS_WAIT_POLICY:
            raise DomainError("synthetic_business_wait_policy_invalid")
        return await super().review(user, citations, actor, command, analysis)


def create_body(key: str, repository: str = "checkout", selector: dict | None = None) -> dict:
    """Synthetic business request still exercises all original schema validation."""
    at = datetime.now(UTC)
    return {
        "idempotency_key": key,
        "approver_user_id": 2,
        "binding": {"repository_id": repository, "selector": selector or {"kind": "pr", "pr_number": 1}},
        "change": {
            "requester_id": "local-user:1",
            "service": "synthetic-v11",
            "environment": "staging",
            "title": "Synthetic review",
            "description": "Offline fixture only",
            "change_window": {
                "start": at.isoformat(),
                "end": (at + timedelta(hours=1)).isoformat(),
                "timezone": "UTC",
            },
            "rollback_plan": "No target writes; retain prior facts",
            "verification_plan": "Check simulated receipt",
        },
    }


async def seed(store: V11Store) -> None:
    """Only a verified isolated probe database may call this synthetic initializer."""
    async with await store.connection.connect() as db, db.transaction():
        for number in (1, 2, 3):
            await db.execute(
                'INSERT INTO "user"(id,email,hashed_password,created_at) VALUES(%s,%s,%s,now()) ON CONFLICT DO NOTHING',
                (number, f"v11-synthetic-{number}@example.invalid", "synthetic-no-password-login"),
            )
            await db.execute(
                "INSERT INTO session(id,user_id,name,created_at) VALUES(%s,%s,'',now()) ON CONFLICT DO NOTHING",
                ("v11-synthetic-session-" + str(number), number),
            )
        await db.execute(
            "INSERT INTO v11_budget_carryovers(evidence_id,evidence_hash,total_charge,smoke_charge,unknown_charge,reconciled) VALUES('synthetic-offline-only',%s,0,0,0,true) ON CONFLICT DO NOTHING",
            (canonical_hash("synthetic offline ledger; NOT a real budget reconciliation"),),
        )
