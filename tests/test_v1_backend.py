"""Actual HTTP/single Graph product contracts over explicit offline doubles."""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from app.api.v1.evidence import router as evidence_router
from app.api.v1.releaseguard import router
from app.core.langgraph.graph import build_agent_graph
from app.core.limiter import limiter
from app.schemas.local_mvp import LocalActor
from app.schemas.v1_backend import AnalysisOutput, Claim, SourceCommand
from app.services.local_mvp import LocalMVPService
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_backend import V1Backend
from app.services.v1_sources import GitHubSource, MarkdownSource
from app.services.v1_storage import SOURCE_IDS
from test_local_mvp import Store, body, call, no_chat
from test_v1_sources import transport


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
        """Offline CAS contract only; PG row-lock contention has separate evidence."""
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
            risk=self.risk,
            unknowns=["missing"] if self.risk == "UNKNOWN" else [],
            candidate_steps=["validate_inputs", "simulate", "verify_predictions"],
        )
        return output, {"model": "fake", "usage": "unknown", "cost": "unknown"}


@pytest.fixture
def anyio_backend() -> str:
    """The backend and graph are asyncio only."""
    return "asyncio"


@pytest.fixture
async def product_v1():
    """Actual routers and Graph; all doubles clearly marked as offline."""
    store, source_store, provider = Store(), SourceStore(), Provider()
    store.source_rows = source_store.rows
    graph: Any = None

    async def get_graph() -> Any:
        return graph

    local = LocalMVPService(get_graph, store)
    runtime = V1Backend(local, source_store, provider, GitHubSource(transport()), MarkdownSource())
    graph = build_agent_graph(no_chat, no_chat, InMemorySaver(), local.domain, **local.dependencies())
    app = FastAPI()
    app.state.local_mvp, app.state.limiter = local, limiter
    app.include_router(router, prefix="/releaseguard")
    app.include_router(evidence_router, prefix="/evidence")
    actor = await store.authenticate("synthetic-session-1", datetime.now(UTC).timestamp() + 3600)
    for source in SOURCE_IDS:
        # No credentials or network requests occur in these fixture commands.
        result = await runtime.sync(
            source, actor, SourceCommand(command_id="sync-" + source, expected_version=0, pr_number=1)
        )
        assert result["status"] == "completed"
    limiter.enabled = False
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://v1.invalid") as client:
            yield client, store, source_store, provider, runtime
    finally:
        limiter.enabled = True


async def analyzed(client: httpx.AsyncClient, key: str = "product") -> tuple[dict, dict]:
    """Drive the actual authenticated product entry and analysis node."""
    value = body(key)
    value["product"] = {"pr_number": 1}
    created = await call(client, "POST", "/changes", value)
    assert created.status_code == 200, created.text
    view = created.json()
    command = {
        "command_id": "analyze-" + key,
        "expected_state_version": view["state"]["state_version"],
        "query": "simulation and approval",
    }
    result = await call(client, "POST", f"/changes/{view['state']['change_id']}/analyze", command)
    assert result.status_code == 200, result.text
    return view, result.json()


@pytest.mark.anyio
async def test_full_single_graph_dry_run_and_compatibility(product_v1) -> None:
    """A model LOW recommendation cannot lower the staging floor or approve."""
    client, _, _, provider, _ = product_v1
    view, analysis = await analyzed(client)
    assert analysis["status"] == "completed" and provider.calls == 1
    identity = view["state"]["change_id"]
    current = (await call(client, "GET", f"/changes/{identity}")).json()
    prepared = await call(
        client,
        "POST",
        f"/changes/{identity}/prepare",
        {
            "command_id": "prepare",
            "expected_state_version": current["state"]["state_version"],
            "analysis_id": analysis["analysis_id"],
        },
    )
    assert prepared.status_code == 200, prepared.text
    pending = prepared.json()
    assert pending["state"]["risk_assessment"]["level"] == "HIGH"
    assert pending["approval"]["request"]["binding"]["binding_version"] == "v2"
    request = pending["approval"]["request"]
    decision = {
        "command_id": "decide",
        "approval_request_id": request["approval_request_id"],
        "authorization_hash": request["authorization_hash"],
        "expected_state_version": pending["state"]["state_version"],
        "decision": "APPROVED",
        "reason_code": "synthetic_approval",
    }
    approved = await call(client, "POST", f"/changes/{identity}/approval", decision, user=2)
    assert approved.status_code == 200, approved.text
    executed = await call(
        client,
        "POST",
        f"/changes/{identity}/run",
        {"command_id": "run", "expected_state_version": approved.json()["state"]["state_version"]},
    )
    assert executed.status_code == 200, executed.text
    result = executed.json()
    assert result["state"]["status"] == "RECEIPTED" and result["target_write_calls"] == 0
    assert result["source_provenance"] == "readonly_github_markdown" and result["analysis_provenance"] == "unverified"
    assert result["execution_mode"] == "offline_simulation" and provider.calls == 1


@pytest.mark.anyio
async def test_model_replay_conflict_and_source_revocation(product_v1) -> None:
    """Replay is uncharged, changed commands conflict, revoked content is hidden."""
    client, _, sources, provider, _ = product_v1
    view, analysis = await analyzed(client)
    identity = view["state"]["change_id"]
    replay = {
        "command_id": "analyze-product",
        "expected_state_version": view["state"]["state_version"],
        "query": "simulation and approval",
    }
    assert (await call(client, "POST", f"/changes/{identity}/analyze", replay)).json() == analysis
    assert provider.calls == 1
    replay["query"] = "different"
    assert (await call(client, "POST", f"/changes/{identity}/analyze", replay)).status_code == 409
    sources.rows[SOURCE_IDS[0]]["acl"] = [1]
    assert (await call(client, "GET", f"/changes/{identity}/analysis", user=2)).status_code == 404
    for endpoint in ("/changes", "/approvals"):
        page = (await call(client, "GET", endpoint, user=2)).json()
        assert page == {"items": [], "next_cursor": None}
    assert (await call(client, "GET", f"/changes/{identity}", user=3)).status_code == 404


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure", ["analysis_invalid_response", "analysis_outcome_unknown", "analysis_citation_invalid"]
)
async def test_model_failure_never_prepares(product_v1, failure: str) -> None:
    """Failure remains visible and cannot generate approval or success Receipt."""
    client, _, _, provider, _ = product_v1
    provider.failure = failure
    view, result = await analyzed(client)
    assert result["status"] == "failed" and result["reason"] == failure
    current = (await call(client, "GET", f"/changes/{view['state']['change_id']}")).json()
    assert current["approval"] is None and current["state"]["receipt"] is None and current["target_write_calls"] == 0


@pytest.mark.anyio
async def test_slow_model_does_not_hold_object_and_stale_result_not_adopted(product_v1) -> None:
    """Actual Graph analysis can coexist with a version-changing revision."""
    client, _, _, provider, _ = product_v1

    async def during_call() -> None:
        listed = await call(client, "GET", "/changes")
        identity = listed.json()["items"][0]["change_id"]
        current = (await call(client, "GET", f"/changes/{identity}")).json()
        revised = await call(
            client,
            "POST",
            f"/changes/{identity}/revision",
            {
                "command_id": "revision-during-model",
                "expected_state_version": current["state"]["state_version"],
                "change": body()["change"],
            },
        )
        assert revised.status_code == 200, revised.text

    provider.before_return = during_call
    _, result = await analyzed(client)
    assert result["status"] == "failed" and result["reason"] == "analysis_inputs_changed"


@pytest.mark.anyio
async def test_unknown_outcome_new_key_cannot_resend(product_v1) -> None:
    """A new key cannot hide an unknown paid attempt on the same revision."""
    client, _, _, provider, _ = product_v1
    provider.failure = "analysis_outcome_unknown"
    view, result = await analyzed(client)
    identity = view["state"]["change_id"]
    current = (await call(client, "GET", f"/changes/{identity}")).json()
    second = await call(
        client,
        "POST",
        f"/changes/{identity}/analyze",
        {
            "command_id": "another-key",
            "expected_state_version": current["state"]["state_version"],
            "query": "simulation",
        },
    )
    assert result["reason"] == "analysis_outcome_unknown"
    assert second.json()["reason"] == "analysis_revision_required" and provider.calls == 1


@pytest.mark.anyio
async def test_slow_analysis_source_revoke_discards_result(product_v1) -> None:
    """Source version changes while model is pending invalidate its result."""
    client, _, sources, provider, _ = product_v1

    async def revoke() -> None:
        sources.rows[SOURCE_IDS[0]]["acl"] = [1]
        sources.rows[SOURCE_IDS[0]]["version"] += 1
        assert (await call(client, "GET", "/changes")).status_code == 200

    provider.before_return = revoke
    view, result = await analyzed(client)
    assert result["reason"] == "analysis_inputs_changed"
    identity = view["state"]["change_id"]
    assert (await call(client, "GET", f"/changes/{identity}/analysis", user=2)).status_code == 404


@pytest.mark.anyio
async def test_gate_failed_ci_has_zero_analysis_calls(product_v1) -> None:
    """Successful collection of failed CI does not allow a model or approval."""
    client, _, _, provider, runtime = product_v1
    runtime.github = GitHubSource(transport(conclusion="failure"))
    view, result = await analyzed(client)
    assert result["status"] == "failed" and result["reason"] == "index_source_mismatch"
    assert provider.calls == 0
    current = (await call(client, "GET", f"/changes/{view['state']['change_id']}")).json()
    assert current["approval"] is None and current["state"]["receipt"] is None


@pytest.mark.anyio
async def test_indexed_failed_ci_enters_evidence_wait_with_zero_model_calls(product_v1) -> None:
    """Even a current indexed failed-check fact must fail the authoritative Gate."""
    client, store, sources, provider, runtime = product_v1
    runtime.github = GitHubSource(transport(conclusion="failure"))
    actor = await store.authenticate("synthetic-session-1", datetime.now(UTC).timestamp() + 3600)
    synced = await runtime.sync(
        SOURCE_IDS[0],
        actor,
        SourceCommand(
            command_id="sync-failed-ci", expected_version=sources.rows[SOURCE_IDS[0]]["version"], pr_number=1
        ),
    )
    assert synced["status"] == "completed"
    embeddings_before = provider.embed_calls
    view, result = await analyzed(client, "indexed-failed-ci")
    assert result["status"] == "blocked" and result["reason"] == "evidence_or_ci_blocked"
    current = (await call(client, "GET", f"/changes/{view['state']['change_id']}")).json()
    assert current["state"]["status"] == "WAITING_FOR_EVIDENCE"
    assert provider.calls == 0 and provider.embed_calls == embeddings_before
    assert current["approval"] is None and current["state"]["receipt"] is None
