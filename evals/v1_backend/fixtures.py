"""Explicit isolated in-memory support for real services/Graph, not PG evidence."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast, override
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from jose import jwt
from langgraph.checkpoint.memory import InMemorySaver

from evals.releaseguard.boundary import OfflineBoundary
from app.api.v1.evidence import router as evidence_router
from app.api.v1.releaseguard import router
from app.core.config import settings
from app.core.langgraph.graph import build_agent_graph
from app.core.limiter import limiter
from app.schemas.local_mvp import LocalActor
from app.schemas.releaseguard import RiskLevel
from app.schemas.v1_backend import AnalysisOutput, Claim, SourceCommand
from app.services.local_mvp import LocalMVPService
from app.services.local_mvp_storage import LocalPostgresStore, LocalUnit, active_unit
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_backend import V1Backend
from app.services.v1_sources import GitHubSource, MarkdownSource
from app.services.v1_storage import SOURCE_IDS

HEAD = "a" * 40


class FixedClock(datetime):
    """Stable domain clock; real Provider latency is measured elsewhere."""

    @classmethod
    @override
    def now(cls, tz: Any = None) -> datetime:
        """Return the dataset's fixed domain timestamp."""
        return cls(2026, 10, 9, 12, 0, tzinfo=UTC)


class Cursor:
    """Only the CAS return value is needed by the storage double."""

    def __init__(self, version: int) -> None:
        """Initialize the isolated synthetic test double."""
        self.version = version

    async def fetchone(self) -> dict[str, int]:
        """Return the emulated CAS result."""
        return {"version": self.version}


class Connection:
    """Emulate the one-row update; real database tested separately."""

    def __init__(self, row: dict[str, Any]) -> None:
        """Initialize the isolated synthetic test double."""
        self.row = row

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Emulate the one-row transaction boundary."""
        yield

    async def execute(self, sql: str, parameters: tuple) -> Cursor:
        """Validate and apply a single-row CAS."""
        assert sql.startswith("UPDATE local_mvp_records")
        payload, identity, version = parameters
        assert identity == self.row["change_id"] and version == self.row["version"]
        self.row["payload"] = deepcopy(payload.obj)
        self.row["version"] += 1
        return Cursor(self.row["version"])


class Store(LocalPostgresStore):
    """Explicit test double with object locks and persisted typed roundtrips."""

    def __init__(self) -> None:
        """Initialize the isolated synthetic test double."""
        self.rows: dict[str, dict[str, Any]] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.source_rows: dict[str, dict] = {}
        self.sessions = {"synthetic-session-" + str(i): i for i in (1, 2, 3)}

    @override
    async def authenticate(self, session_id: str, expires_at: float) -> LocalActor:
        """Validate a live synthetic session."""
        if session_id not in self.sessions or expires_at <= datetime.now(UTC).timestamp():
            raise DomainError("local_session_invalid")
        return LocalActor(user_id=self.sessions[session_id], session_id=session_id, expires_at=expires_at)

    @override
    async def list_records(self, actor: LocalActor, *, inbox: bool, cursor: str, limit: int) -> list[str]:
        """Apply the same membership-before-pagination contract as PostgreSQL."""
        return sorted(
            identity
            for identity, row in self.rows.items()
            if identity > cursor
            and (
                not row["payload"].get("facts", {}).get("product")
                or not self.source_rows
                or all(
                    actor.user_id in source["acl"] and not source.get("deleted")
                    for source in self.source_rows.values()
                )
            )
            and (
                not inbox
                or any(
                    run["run_id"] == row["payload"]["facts"]["creation"].get("run_id")
                    and run["status"] == "WAITING_FOR_APPROVAL"
                    for run in json.loads(row["payload"].get("facts", {}).get("domain") or "{}").get("runs", [])
                )
            )
            and (
                row["approver_user_id"] == actor.user_id
                if inbox
                else actor.user_id in {row["owner_user_id"], row["approver_user_id"]}
            )
        )[:limit]

    @asynccontextmanager
    async def open(
        self, change_id: str, actor: LocalActor, creation: tuple[int, str] | None = None
    ) -> AsyncIterator[LocalUnit]:
        """Authorize the object and hold its per-object test lock."""
        async with self.locks.setdefault(change_id, asyncio.Lock()):
            if creation:
                if creation[0] == actor.user_id:
                    raise DomainError("local_self_approval_denied")
                self.rows.setdefault(
                    change_id,
                    {
                        "change_id": change_id,
                        "owner_user_id": actor.user_id,
                        "approver_user_id": creation[0],
                        "creation_hash": creation[1],
                        "version": 0,
                        "payload": {},
                    },
                )
            row = self.rows.get(change_id)
            if row is None or actor.user_id not in {row["owner_user_id"], row["approver_user_id"]}:
                raise DomainError("local_object_not_found")
            if creation and creation[1] != row["creation_hash"]:
                raise DomainError("idempotency_conflict")
            current = LocalUnit(cast(Any, Connection(row)), deepcopy(row), actor)
            await current.load()
            token = active_unit.set(current)
            try:
                yield current
            finally:
                active_unit.reset(token)


async def no_chat(state: Any) -> dict:
    """No provider call in these product tests."""
    return {}


def body(key: str = "normal", scenario: str = "complete") -> dict:
    """Strict synthetic change data, with current UTC window."""
    now = FixedClock.now(UTC)
    return {
        "change": {
            "requester_id": "local-user:1",
            "service": "synthetic-service",
            "environment": "staging",
            "title": "Synthetic local change",
            "description": "Bounded register only",
            "change_window": {
                "start": now.isoformat(),
                "end": (now + timedelta(hours=1)).isoformat(),
                "timezone": "UTC",
            },
            "rollback_plan": "Restore synthetic integer",
            "verification_plan": "Independent integer observation",
        },
        "approver_user_id": 2,
        "idempotency_key": key,
        "evidence_scenario": scenario,
    }


async def call(
    client: httpx.AsyncClient, method: str, path: str, value: dict | None = None, user: int = 1
) -> httpx.Response:
    """Send an authenticated HTTP request without exposing the token."""
    return await client.request(
        method, "/releaseguard" + path, json=value, headers={"Authorization": "Bearer " + token(user)}
    )


def token(user: int) -> str:
    """Ephemeral signed synthetic session; no secret or token is reported."""
    return jwt.encode(
        {"sub": "synthetic-session-" + str(user), "exp": 4102444800},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


class SourceStore:
    """Explicit non-durable double; PostgreSQL behavior is separately probed."""

    def __init__(self) -> None:
        """A fresh source ACL and command/budget ledger for each test."""
        self.rows = {
            source: {
                "source_id": source,
                "owner_id": 1,
                "acl": [1, 2],
                "version": 0,
                "generation": None,
                "required_checks": ["test"],
                "deleted": False,
            }
            for source in SOURCE_IDS
        }
        self.generations: dict[str, dict] = {}
        self.chunks: dict[str, list] = {}
        self.commands: dict[tuple, dict] = {}
        self.reservations: dict[tuple, Decimal] = {}
        self.lock = asyncio.Lock()

    async def source(self, source_id: str, actor: LocalActor, *, owner: bool = False) -> dict:
        """The approver role does not bypass source rights."""
        row = self.rows[source_id]
        if actor.user_id not in row["acl"]:
            raise DomainError("source_not_found")
        if owner and row["owner_id"] != actor.user_id:
            raise DomainError("source_owner_required")
        return deepcopy(row)

    async def sources(self, actor: LocalActor) -> list:
        """Return only visible sources."""
        return [deepcopy(row) for row in self.rows.values() if actor.user_id in row["acl"]]

    async def generation(self, identifier: str) -> dict:
        """Only immutable configuration is returned."""
        return deepcopy(self.generations[identifier])

    async def audit(self, resource: str) -> list:
        """Offline command facts, explicitly not PG append-only evidence."""
        return [deepcopy(v) for v in self.commands.values() if v["resource"] == resource]

    async def lock_sources(self, db: Any, expected: dict, actor: LocalActor) -> None:
        """Offline CAS contract, never a persistent row-lock claim."""
        for key, value in expected.items():
            row = await self.source(key, actor)
            if (row["version"], row["generation"], row["acl"]) != (
                value["version"],
                value["generation"],
                value["acl"],
            ):
                raise DomainError("analysis_inputs_changed")

    async def claim(self, actor: LocalActor, command_id: str, digest: str, resource: str) -> dict | None:
        """Model replay never repeats calls."""
        key = (actor.user_id, command_id)
        async with self.lock:
            if key in self.commands:
                prior = self.commands[key]
                if prior["digest"] != digest:
                    raise DomainError("idempotency_conflict")
                if prior["status"] != "completed":
                    raise DomainError("provider_command_outcome_unknown")
                return deepcopy(prior["result"])
            self.commands[key] = {"digest": digest, "status": "started", "resource": resource}
        return None

    async def complete(self, actor: LocalActor, command_id: str, result: dict) -> None:
        """Keep exact original results."""
        self.commands[(actor.user_id, command_id)].update(status="completed", result=deepcopy(result))

    async def reserve(self, actor: LocalActor, command: str, attempt: str, amount: Decimal, limit: Decimal) -> None:
        """Serialize budget reservation without claiming a durable transaction."""
        async with self.lock:
            key = (actor.user_id, command, attempt)
            if key in self.reservations:
                raise DomainError("model_attempt_already_reserved")
            if sum(self.reservations.values(), Decimal(0)) + amount > limit:
                raise DomainError("model_budget_exhausted")
            self.reservations[key] = amount

    async def usage(self, *args: Any) -> None:
        """Unknown fake costs remain conservative reservations."""

    async def publish(self, actor: LocalActor, source: dict, generation: str, config: dict, chunks: list) -> None:
        """Failure leaves the previous published generation untouched."""
        row = await self.source(source["source_id"], actor, owner=True)
        if row["version"] != source["version"]:
            raise DomainError("source_changed_during_sync")
        self.generations[generation] = deepcopy(config)
        self.chunks[generation] = deepcopy(chunks)
        self.rows[source["source_id"]].update(version=row["version"] + 1, generation=generation)

    async def candidates(
        self, actor: LocalActor, pr_number: int, query: str, vector: list | None
    ) -> tuple[list, list]:
        """Permission precedes ranking in this clearly synthetic double."""
        values = []
        for row in self.rows.values():
            if actor.user_id not in row["acl"]:
                continue
            for chunk in self.chunks.get(row["generation"], []):
                if row["source_id"] == SOURCE_IDS[1] or chunk.document.resource == f"pr:{pr_number}":
                    values.append(chunk)
        values.sort(key=lambda c: (-sum(word.lower() in c.text.lower() for word in query.split()), c.chunk_id))
        lexical = [c for c in values if any(word.lower() in c.text.lower() for word in query.split())]
        return lexical[:20], values[:20] if vector else []


class Provider:
    """Synthetic Provider contract, never a real quality or cost measurement."""

    provenance = "unverified"

    def __init__(self) -> None:
        """Explicit call counters and server-only failure injection."""
        self.calls = 0
        self.embed_calls = 0
        self.before_return: Any = None
        self.failure: str | None = None
        self.risk = "LOW"

    def config(self, role: Any) -> Any:
        """Expose versioned synthetic dimensions."""
        return SimpleNamespace(
            model="synthetic", provider="fake", capabilities=SimpleNamespace(embedding_dimensions=3)
        )

    async def embed(self, texts: list, *args: Any) -> list:
        """No external side effects or charge."""
        self.embed_calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def analyze(self, query: str, chunks: list, *args: Any) -> tuple:
        """Use actual supplied identifiers, without arbitrary authority fields."""
        self.calls += 1
        if self.before_return:
            await self.before_return()
        if self.failure:
            raise DomainError(self.failure)
        output = AnalysisOutput(
            explanation="Synthetic rule explanation",
            claims=[Claim(text="Simulation only", citation_ids=[chunks[0].chunk_id])],
            risk=RiskLevel(self.risk),
            unknowns=["missing"] if self.risk == "UNKNOWN" else [],
            candidate_steps=["validate_inputs", "simulate", "verify_predictions"],
        )
        return output, {"model": "fake", "usage": "unknown", "cost": "unknown"}


def transport(
    *,
    status: int = 200,
    conclusion: str = "success",
    drift: bool = False,
    empty: bool = False,
    sha: str = HEAD,
    truncated: bool = False,
) -> httpx.MockTransport:
    """Simulate bounded public GETs and count every operation at the boundary."""
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.host == "api.github.com"
        calls.append(request.url.path)
        if status != 200:
            return httpx.Response(status)
        if request.url.path.endswith("/pulls/1"):
            value = {"head": {"sha": "b" * 40 if drift and calls.count(request.url.path) > 1 else HEAD}}
        elif request.url.path.endswith("check-runs"):
            items = (
                []
                if empty
                else [{"id": 1, "head_sha": sha, "name": "test", "status": "completed", "conclusion": conclusion}]
            )
            value = {"check_runs": items, "total_count": len(items) + int(truncated)}
        elif request.url.path.endswith("/files"):
            value = [{"filename": "src/synthetic.ts"}]
        else:
            value = [{"tag_name": "v1"}]
        return httpx.Response(200, json=value)

    return httpx.MockTransport(handle)


@asynccontextmanager
async def environment() -> AsyncIterator[tuple]:
    """Fresh routes, services, one Graph, ACL and counters for every case."""
    store, source_store, provider = Store(), SourceStore(), Provider()
    store.source_rows = source_store.rows
    graph: Any = None

    async def get_graph() -> Any:
        return graph

    local = LocalMVPService(get_graph, store)
    runtime = V1Backend(local, cast(Any, source_store), provider, GitHubSource(transport()), MarkdownSource())
    graph = build_agent_graph(no_chat, no_chat, InMemorySaver(), local.domain, **local.dependencies())
    app = FastAPI()
    app.state.local_mvp, app.state.limiter = local, limiter
    app.include_router(router, prefix="/releaseguard")
    app.include_router(evidence_router, prefix="/evidence")
    actor = await store.authenticate("synthetic-session-1", 4102444800)
    with patch("app.services.v1_sources.datetime", FixedClock):
        for source in SOURCE_IDS:
            result = await runtime.sync(
                source, actor, SourceCommand(command_id="sync-" + source, expected_version=0, pr_number=1)
            )
            if result["status"] != "completed":
                raise AssertionError("offline_fixture_sync_failed")
    previous = limiter.enabled
    limiter.enabled = False
    try:
        with (
            patch("app.services.local_mvp_storage.datetime", FixedClock),
            patch("app.services.local_mvp.datetime", FixedClock),
            patch("app.services.v1_sources.datetime", FixedClock),
            OfflineBoundary(),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://v1.invalid"
            ) as client:
                yield client, store, source_store, provider, runtime
    finally:
        limiter.enabled = previous
