"""Limited-phase restrictions exercise real dispatch, including persisted approvals."""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from app.api.v1.releaseguard import actor
from app.api.v1.reviews import review_runtime, router
from app.core.limiter import limiter
from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import ReviewCommand
from app.services.local_mvp_storage import review_scope
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_backend import V1Backend
from app.services.v11_backend import V11Backend
from app.services.v11_policy import CAPABILITIES, PAUSED_COMMANDS
from test_local_mvp import approval, call, prepared, product as product_fixture

product = product_fixture

# Keep the historical paused-policy regression without deploying that policy.
CAPABILITIES = CAPABILITIES.model_copy(
    update={
        "version": "review-capabilities.limited.1",
        "prepare": "paused",
        "approval": "paused",
        "run": "paused",
        "pause_reason": "review_rehearsal_paused_this_phase",
    }
)


@pytest.fixture(autouse=True)
def historical_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise archived pause semantics only through explicit test injection."""
    for module in ("v11_policy", "v11_backend", "v11_queries"):
        monkeypatch.setattr("app.services." + module + ".CAPABILITIES", CAPABILITIES, raising=False)
    monkeypatch.setattr("app.api.v1.reviews.CAPABILITIES", CAPABILITIES, raising=False)


def backend() -> tuple[V11Backend, Any, Any]:
    """Spies fail if rejection touches claims, budgets, providers or local mutation."""
    store = SimpleNamespace(
        object_row=AsyncMock(return_value={"object_version": 0}),
        claim_review=AsyncMock(side_effect=AssertionError("claim_must_not_run")),
    )
    local = SimpleNamespace(command=AsyncMock(side_effect=AssertionError("domain_must_not_run")))
    runtime = V11Backend(SimpleNamespace(local=local), store, Mock(), Mock())
    return runtime, store, local


def principal() -> LocalActor:
    """Return only a synthetic authenticated identity."""
    return LocalActor(user_id=1, session_id="synthetic-session-1", expires_at=4102444800)


@pytest.mark.parametrize("kind", sorted(PAUSED_COMMANDS))
@pytest.mark.parametrize("key", ["old-completed", "old-unknown", "new-key"])
def test_pause_precedes_command_replay_and_claim(kind: str, key: str) -> None:
    """Neither old results nor new intent can bypass the phase restriction."""
    runtime, store, local = backend()
    result = asyncio.run(runtime.command("one", principal(), SimpleNamespace(command_id=key), kind))
    assert result == {"status": "blocked", "reason": CAPABILITIES.pause_reason}
    store.object_row.assert_awaited_once()
    store.claim_review.assert_not_awaited()
    local.command.assert_not_awaited()
    assert runtime.github.mock_calls == runtime.provider.mock_calls == []


def test_internal_guards_reject_before_external_reads_or_adoption() -> None:
    """Calling internal entry points cannot evade the HTTP dispatch restriction."""
    runtime, _, _ = backend()

    async def run() -> None:
        for invoke in (
            lambda: runtime.fresh("one", principal(), {}),
            lambda: runtime.guard({"object_id": "one"}, principal()),
            *(
                lambda kind=kind: runtime.check_command(
                    SimpleNamespace(row={"change_id": "one"}), principal(), None, kind
                )
                for kind in PAUSED_COMMANDS
            ),
        ):
            with pytest.raises(DomainError, match=CAPABILITIES.pause_reason):
                await invoke()
        assert runtime.github.mock_calls == runtime.provider.mock_calls == []

    asyncio.run(run())


def test_routes_and_capabilities_agree_and_preserve_acl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three public writes reject, unauthorized objects stay 404, capabilities do no I/O."""
    runtime, store, local = backend()
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router, prefix="/reviews")
    app.dependency_overrides[actor] = principal
    app.dependency_overrides[review_runtime] = lambda: runtime
    monkeypatch.setattr(limiter, "enabled", False)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local.invalid"
        ) as client:
            response = await client.get("/reviews/capabilities")
            assert response.json() == CAPABILITIES.model_dump(mode="json")
            store.object_row.assert_not_awaited()
            store.repositories = AsyncMock(
                return_value=[{"repository_id": "checkout"}, {"repository_id": "setup-python"}]
            )
            listing = (await client.get("/reviews/repositories")).json()["items"]
            assert len(listing) == 2
            assert all(row["capabilities"]["run"] == "paused" for row in listing)
            bodies = {
                "prepare": {"analysis_id": "analysis:old"},
                "approval": {
                    "approval_request_id": "approval:old",
                    "authorization_hash": "a" * 64,
                    "decision": "APPROVED",
                    "reason_code": "synthetic_reviewed",
                },
                "run": {},
            }
            for route, extra in bodies.items():
                body = {"command_id": "old", "expected_object_version": 0, "expected_state_version": 0, **extra}
                response = await client.post("/reviews/one/" + route, json=body)
                assert response.status_code == 200, response.text
                assert response.json() == {"status": "blocked", "reason": CAPABILITIES.pause_reason}
            store.object_row.side_effect = DomainError("local_object_not_found")
            response = await client.post("/reviews/foreign/run", json=body)
            assert response.status_code == 404
            assert "paused" not in response.text
            store.claim_review.assert_not_awaited()
            local.command.assert_not_awaited()

    asyncio.run(run())


def test_historical_analysis_adds_live_metadata_without_rewriting_saved_bytes() -> None:
    """Failed/unknown reports remain failed/unknown and stored hashes do not change."""
    runtime, store, _ = backend()
    saved = {
        "snapshot_id": "snapshot:old",
        "generation": "generation:old",
        "revision": 0,
        "dependencies": {},
        "status": "unknown",
        "report": {"reason": "analysis_outcome_unknown"},
    }
    before = canonical_hash(saved)
    store.analysis_row = AsyncMock(return_value=saved)
    store.snapshot = AsyncMock(return_value={"captured_at": datetime.now(UTC)})
    store.observation = AsyncMock(
        return_value={
            "captured_at": datetime.now(UTC),
            "facts": {
                "head_sha": "a" * 40,
                "status": "success",
                "complete": True,
                "checks_passed": True,
                "captured_at": datetime.now(UTC).isoformat(),
            },
        }
    )
    store.dependencies = AsyncMock(return_value={})
    store.object_row.return_value = {"repository_id": "checkout", "kind": "pr"}
    runtime.checked_dependencies = AsyncMock(return_value={})
    report = asyncio.run(runtime.analysis("one", "analysis:old", principal()))
    assert report["capabilities"] == CAPABILITIES.model_dump(mode="json")
    assert report["status"] == "unknown" and report["reason"] == "analysis_outcome_unknown"
    assert not report["rehearsal_eligibility"]["eligible"]
    assert CAPABILITIES.pause_reason in report["rehearsal_eligibility"]["reason_codes"]
    assert canonical_hash(saved) == before and "capabilities" not in saved["report"]


@pytest.fixture
def anyio_backend() -> str:
    """The graph fixture runs only on asyncio."""
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("receipted", [False, True])
async def test_saved_approval_pause_and_real_revoke_dispatch(product: tuple, receipted: bool) -> None:
    """Synthetic saved approvals/receipts survive pause; revocation still appends facts.

    Approval is built through the original graph, then wrapped in a synthetic v3
    record. This tests dispatch/history, not a new positive v3 approval authority.
    """
    client, local_store, local = product
    identity, pending = await prepared(client)
    approved = await call(client, "POST", "/changes/" + identity + "/approval", approval(pending), user=2)
    assert approved.status_code == 200
    if receipted:
        response = await call(
            client,
            "POST",
            "/changes/" + identity + "/run",
            {
                "command_id": "run",
                "expected_state_version": approved.json()["state"]["state_version"],
            },
        )
        assert response.status_code == 200 and response.json()["state"]["receipt"]
    row = local_store.rows[identity]
    facts = row["payload"]["facts"]
    facts.update(
        version="local-mvp.v3", product={"contract_version": "releaseguard.review.v1.1", "object_id": identity}
    )
    row["payload"]["hash"] = canonical_hash(facts)
    before = deepcopy(row["payload"])
    runtime, store, _ = backend()
    product_runtime = V1Backend(local, Mock(), Mock(), Mock(), Mock())
    runtime = V11Backend(product_runtime, store, Mock(), Mock())
    store.claim_review.side_effect = None
    store.claim_review.return_value = None
    store.complete = AsyncMock()
    store.dependencies = AsyncMock(return_value={"object_version": 0})
    token = review_scope.set({"save_guard": runtime.commit_guard})
    try:
        for kind in PAUSED_COMMANDS:
            result = await runtime.command(identity, principal(), SimpleNamespace(command_id="old"), kind)
            assert result["reason"] == CAPABILITIES.pause_reason
        assert row["payload"] == before
        store.claim_review.assert_not_awaited()
        with pytest.raises(DomainError, match=CAPABILITIES.pause_reason):
            await local.command(
                identity,
                principal(),
                ReviewCommand(command_id="internal", expected_state_version=0, expected_object_version=0),
                "execute",
            )
        # Read serialized facts; revocation must not require a new Graph or Provider call.
        async with local_store.open(identity, principal()):
            current = await local.view()
        result = await runtime.command(
            identity,
            principal(),
            ReviewCommand(
                command_id="revoke",
                expected_object_version=0,
                expected_state_version=current.state.state_version,
            ),
            "revoke",
        )
        assert result["status"] == "completed", result
        after = row["payload"]["facts"]
        assert after["approvals"]["entries"][0]["decision"] == before["facts"]["approvals"]["entries"][0]["decision"]
        assert after["approvals"]["entries"][0]["invalidations"]
        assert after["domain"] == before["facts"]["domain"]
        assert after["target_calls"] == 0
        assert runtime.github.mock_calls == runtime.provider.mock_calls == []
    finally:
        review_scope.reset(token)
