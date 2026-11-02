"""Product API/service tests with explicit storage double, not durability evidence."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from jose import jwt
from langgraph.checkpoint.memory import InMemorySaver
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.releaseguard import router
from app.core.config import settings
from app.core.langgraph.graph import build_agent_graph
from app.core.limiter import limiter
from app.schemas.local_mvp import LocalActor
from app.services.local_mvp import LocalMVPService
from app.services.local_mvp_storage import LocalPostgresStore, LocalUnit, active_unit
from app.services.releaseguard.state_machine import DomainError
from app.services.releaseguard.normalization import canonical_hash
from app.services.local_mvp_storage import unit
from app.services.tools.execution_adapter import TargetObservation


class WrongReader:
    """Fault-injected independent source, explicitly not a real target failure."""

    source_pin = "independent-register.v1"

    async def observe(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Return an intentionally inconsistent synthetic observation."""
        observation = await unit().target.read(resource_id, at=at)
        return observation.model_copy(update={"value": 0})


class Fault:
    """Server-injected crash window; never selectable from the user API."""

    def __init__(self, point: str) -> None:
        """Initialize the isolated synthetic test double."""
        self.point = point

    async def hit(self, stage: str) -> None:
        """Raise at the selected server-only fault window."""
        if stage == self.point:
            raise RuntimeError("synthetic_fault")


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

    async def authenticate(self, session_id: str, expires_at: float) -> LocalActor:
        """Validate a live synthetic session."""
        if session_id not in self.sessions or expires_at <= datetime.now(UTC).timestamp():
            raise DomainError("local_session_invalid")
        return LocalActor(user_id=self.sessions[session_id], session_id=session_id, expires_at=expires_at)

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


def token(user: int) -> str:
    """Synthetic session credential is generated in memory, never reported."""
    return jwt.encode(
        {"sub": "synthetic-session-" + str(user), "exp": datetime.now(UTC) + timedelta(hours=1)},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


@pytest.fixture
def anyio_backend() -> str:
    """Only asyncio; no optional trio dependency."""
    return "asyncio"


@pytest.fixture
async def product() -> AsyncIterator[tuple[httpx.AsyncClient, Store, LocalMVPService]]:
    """Use the actual router and graph factory, with explicit storage double."""
    store = Store()
    graph: Any = None

    async def get_graph() -> Any:
        """Return the existing graph for this isolated fixture."""
        return graph

    service = LocalMVPService(get_graph, store)
    graph = build_agent_graph(no_chat, no_chat, InMemorySaver(), service.domain, **service.dependencies())
    app = FastAPI()
    app.state.local_mvp = service
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router, prefix="/releaseguard")
    limiter.enabled = False
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local.invalid"
        ) as client:
            yield client, store, service
    finally:
        limiter.enabled = True


def body(key: str = "normal", scenario: str = "complete") -> dict:
    """Strict synthetic change data, with current UTC window."""
    now = datetime.now(UTC)
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


async def prepared(
    client: httpx.AsyncClient,
    key: str = "normal",
    mode: str = "offline_simulation",
    scenario: str = "complete",
    ttl: int = 600,
) -> tuple[str, dict]:
    """Create a synthetic change and prepare it through the Graph."""
    created = await call(client, "POST", "/changes", body(key, scenario))
    assert created.status_code == 200, created.text
    identity = created.json()["state"]["change_id"]
    response = await call(
        client,
        "POST",
        "/changes/" + identity + "/prepare",
        {"command_id": "prepare", "expected_state_version": 0, "mode": mode, "approval_ttl_seconds": ttl},
    )
    assert response.status_code == 200, response.text
    return identity, response.json()


def approval(view: dict, decision: str = "APPROVED") -> dict:
    """Build a typed decision from the server-issued binding."""
    request = view["approval"]["request"]
    return {
        "command_id": "decision",
        "approval_request_id": request["approval_request_id"],
        "authorization_hash": request["authorization_hash"],
        "expected_state_version": view["state"]["state_version"],
        "decision": decision,
        "reason_code": "synthetic_review",
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode,final,writes", [("offline_simulation", "SIMULATED", 0), ("offline_sandbox", "SUCCEEDED", 1)]
)
async def test_actual_product_chain(product: tuple, mode: str, final: str, writes: int) -> None:
    """Verify actual product chain."""
    client, _, _ = product
    identity, view = await prepared(client, mode=mode)
    assert view["state"]["status"] == "WAITING_FOR_APPROVAL"
    assert len(view["state"]["evidence_items"]) == 2
    decision = await call(client, "POST", "/changes/" + identity + "/approval", approval(view), 2)
    assert decision.status_code == 200, decision.text
    value = {"command_id": "run", "expected_state_version": decision.json()["state"]["state_version"]}
    result = await call(client, "POST", "/changes/" + identity + "/run", value)
    assert result.status_code == 200, result.text
    assert result.json()["state"]["receipt"]["final_status"] == final
    assert result.json()["target_write_calls"] == writes
    assert result.json()["production_execution"] == "disabled"
    duplicate = await call(client, "POST", "/changes/" + identity + "/run", value)
    assert duplicate.json() == result.json()
    assert (await call(client, "GET", "/changes/" + identity + "/audit")).json()["events"]
    assert (await call(client, "GET", "/changes/" + identity + "/receipt")).json()["receipt"]["final_status"] == final


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["missing", "expired", "conflicting", "unavailable"])
async def test_bad_evidence_cannot_approve_or_execute(product: tuple, scenario: str) -> None:
    """Verify bad evidence cannot approve or execute."""
    client, _, _ = product
    identity, view = await prepared(client, scenario=scenario)
    assert view["state"]["status"] == "WAITING_FOR_EVIDENCE"
    assert view["approval"] is None
    denied = await call(
        client,
        "POST",
        "/changes/" + identity + "/run",
        {"command_id": "run", "expected_state_version": view["state"]["state_version"]},
    )
    assert denied.status_code == 409
    assert (await call(client, "GET", "/changes/" + identity)).json()["target_write_calls"] == 0


@pytest.mark.anyio
async def test_scope_identity_and_pending_zero_call(product: tuple) -> None:
    """Verify scope identity and pending zero call."""
    client, store, _ = product
    identity, view = await prepared(client)
    path = "/changes/" + identity
    assert (await client.get("/releaseguard/identity")).status_code == 401
    assert (await call(client, "GET", path, user=3)).status_code == 404
    assert (await call(client, "GET", path + "/audit", user=3)).status_code == 404
    assert (await call(client, "POST", path + "/approval", approval(view), 1)).status_code == 403
    assert (
        await call(
            client,
            "POST",
            path + "/run",
            {"command_id": "pending", "expected_state_version": view["state"]["state_version"]},
        )
    ).status_code == 409
    assert (await call(client, "GET", path)).json()["target_write_calls"] == 0
    del store.sessions["synthetic-session-1"]
    assert (await call(client, "GET", path)).status_code == 401


@pytest.mark.anyio
@pytest.mark.parametrize("decision", ["REJECTED", "NEEDS_CHANGES"])
async def test_negative_approval(product: tuple, decision: str) -> None:
    """Verify negative approval."""
    client, _, _ = product
    identity, view = await prepared(client)
    path = "/changes/" + identity
    decided = await call(client, "POST", path + "/approval", approval(view, decision), 2)
    assert decided.status_code == 200, decided.text
    result = await call(
        client,
        "POST",
        path + "/run",
        {"command_id": "run", "expected_state_version": decided.json()["state"]["state_version"]},
    )
    assert result.status_code == 409
    assert (await call(client, "GET", path)).json()["target_write_calls"] == 0


@pytest.mark.anyio
async def test_revocation_and_binding_drift(product: tuple) -> None:
    """Verify revocation and binding drift."""
    client, _, _ = product
    identity, view = await prepared(client)
    path = "/changes/" + identity
    drift = approval(view)
    drift["authorization_hash"] = "0" * 64
    assert (await call(client, "POST", path + "/approval", drift, 2)).status_code == 409
    decided = await call(client, "POST", path + "/approval", {**approval(view), "command_id": "decision-valid"}, 2)
    assert decided.status_code == 200, decided.text
    version = decided.json()["state"]["state_version"]
    revoked = await call(client, "POST", path + "/revoke", {"command_id": "revoke", "expected_state_version": version})
    assert revoked.status_code == 200
    rejected = await call(client, "POST", path + "/run", {"command_id": "run", "expected_state_version": version})
    assert rejected.status_code == 409
    assert (await call(client, "GET", path)).json()["target_write_calls"] == 0


@pytest.mark.anyio
async def test_concurrent_create_and_conflicts(product: tuple) -> None:
    """Verify concurrent create and conflicts."""
    client, _, _ = product
    value = body()
    first, second = await asyncio.gather(
        call(client, "POST", "/changes", value), call(client, "POST", "/changes", value)
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    altered = deepcopy(value)
    altered["change"]["title"] = "Different body same key"
    assert (await call(client, "POST", "/changes", altered)).status_code == 409
    identity = first.json()["state"]["change_id"]
    prepare_value = {"command_id": "prepare", "expected_state_version": 0}
    responses = await asyncio.gather(
        *(call(client, "POST", "/changes/" + identity + "/prepare", prepare_value) for _ in range(2))
    )
    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    assert responses[0].json() == responses[1].json()


@pytest.mark.anyio
async def test_orphan_command_stops_without_replay(product: tuple) -> None:
    """Verify orphan command stops without replay."""
    client, store, _ = product
    identity, view = await prepared(client)
    row = store.rows[identity]
    # Inject an explicitly faulted durable root claim, not automatic recovery.
    facts = row["payload"]["facts"]
    facts["commands"]["orphan"] = {"hash": "0" * 64, "status": "started"}
    row["payload"]["hash"] = canonical_hash(facts)
    response = await call(
        client,
        "POST",
        "/changes/" + identity + "/run",
        {"command_id": "another", "expected_state_version": view["state"]["state_version"]},
    )
    assert response.status_code == 409
    saved = (await call(client, "GET", "/changes/" + identity)).json()
    assert saved["recovery"] == "safe_hold" and saved["target_write_calls"] == 0


@pytest.mark.anyio
async def test_approval_expiry_zero_call(product: tuple) -> None:
    """Verify approval expiry zero call."""
    client, _, _ = product
    identity, view = await prepared(client, ttl=1)
    await asyncio.sleep(1.1)
    response = await call(client, "POST", "/changes/" + identity + "/approval", approval(view), 2)
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["invalidations"][0]["decision"] == "EXPIRED"
    denied = await call(
        client,
        "POST",
        "/changes/" + identity + "/run",
        {"command_id": "run", "expected_state_version": view["state"]["state_version"]},
    )
    assert denied.status_code == 409
    assert (await call(client, "GET", "/changes/" + identity)).json()["target_write_calls"] == 0


@pytest.mark.anyio
async def test_independent_verification_failure_not_success(product: tuple) -> None:
    """Verify independent verification failure not success."""
    client, _, service = product
    identity, view = await prepared(client, mode="offline_sandbox")
    path = "/changes/" + identity
    decided = await call(client, "POST", path + "/approval", approval(view), 2)
    service.verification.reader = WrongReader()
    result = await call(
        client,
        "POST",
        path + "/run",
        {"command_id": "run", "expected_state_version": decided.json()["state"]["state_version"]},
    )
    assert result.status_code == 200, result.text
    saved = result.json()
    assert saved["state"]["verification_result"]["status"] == "FAILED"
    assert saved["state"]["status"] == "RECOVERY_REQUIRED" and saved["recovery"] == "safe_hold"
    assert saved["state"]["receipt"] is None or saved["state"]["receipt"]["final_status"] != "SUCCEEDED"


@pytest.mark.anyio
async def test_unknown_saved_receipt_and_zero_resend(product: tuple) -> None:
    """Verify unknown saved receipt and zero resend."""
    client, _, service = product
    identity, view = await prepared(client, mode="offline_sandbox")
    path = "/changes/" + identity
    decided = await call(client, "POST", path + "/approval", approval(view), 2)
    service.execution.faults = Fault("after_send")
    value = {"command_id": "run", "expected_state_version": decided.json()["state"]["state_version"]}
    result = await call(client, "POST", path + "/run", value)
    assert result.status_code == 200, result.text
    saved = result.json()
    assert saved["state"]["status"] == "EXECUTION_UNKNOWN" and saved["recovery"] == "safe_hold"
    assert saved["state"]["receipt"]["final_status"] == "UNKNOWN" and saved["target_write_calls"] == 1
    duplicate = await call(client, "POST", path + "/run", value)
    assert duplicate.json() == saved
    denied = await call(
        client,
        "POST",
        path + "/run",
        {"command_id": "new-run", "expected_state_version": saved["state"]["state_version"]},
    )
    assert denied.status_code == 409
    assert (await call(client, "GET", path)).json()["target_write_calls"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ["missing", "complete"])
async def test_revision_supplements_evidence_and_invalidates_old_approval(product: tuple, scenario: str) -> None:
    """Verify revision supplements evidence and invalidates old approval."""
    client, _, _ = product
    identity, view = await prepared(client, scenario=scenario)
    path = "/changes/" + identity
    old_command = approval(view) if view["approval"] else None
    revision = {
        "command_id": "revision",
        "expected_state_version": view["state"]["state_version"],
        "change": body()["change"],
        "evidence_scenario": "complete",
    }
    revised = await call(client, "POST", path + "/revision", revision)
    assert revised.status_code == 200, revised.text
    assert revised.json()["state"]["revision"] == 2
    if old_command:
        old_command["expected_state_version"] = revised.json()["state"]["state_version"]
        assert (await call(client, "POST", path + "/approval", old_command, 2)).status_code == 409
    prepared_again = await call(
        client,
        "POST",
        path + "/prepare",
        {"command_id": "prepare-r2", "expected_state_version": revised.json()["state"]["state_version"]},
    )
    assert prepared_again.status_code == 200, prepared_again.text
    assert prepared_again.json()["state"]["status"] == "WAITING_FOR_APPROVAL"
    command = {**approval(prepared_again.json()), "command_id": "decision-r2"}
    decided = await call(client, "POST", path + "/approval", command, 2)
    assert decided.status_code == 200, decided.text
    executed = await call(
        client,
        "POST",
        path + "/run",
        {"command_id": "run-r2", "expected_state_version": decided.json()["state"]["state_version"]},
    )
    assert executed.status_code == 200, executed.text
    assert executed.json()["state"]["receipt"]["revision"] == 2
