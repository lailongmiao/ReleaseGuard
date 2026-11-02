"""Read-only projections retain original hashes, actor scope and report bindings."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.schemas.approval import ApprovalStoreSnapshot
from app.schemas.local_mvp import LocalActor
from app.services.v11_backend import V11Backend
from app.services.inheritance_queries import approval_page, report_ci
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from tests.test_approval_contract import approval_setup
from tests.test_change_request_contract import NOW


def test_history_pagination_expiry_and_hashes() -> None:
    """Every entry is original; computed expiry cannot append an invalidation."""

    async def scenario() -> None:
        service, request, context, descriptor = await approval_setup()
        entry = await service.create(request, context, descriptor, at=NOW)
        entries = tuple(
            entry.model_copy(update={"request": request.model_copy(update={"approval_request_id": f"request-{i:03}"})})
            for i in range(101)
        )
        snapshot = ApprovalStoreSnapshot(entries=entries)
        original = canonical_hash(snapshot)
        actor = LocalActor(user_id=1, session_id="synthetic", expires_at=9999999999)
        first = approval_page(snapshot, actor, context.invocation.change_id, "local-mvp.v2", None, 50, NOW)
        second = approval_page(
            snapshot, actor, context.invocation.change_id, "local-mvp.v2", first.next_cursor, 50, NOW
        )
        last = approval_page(
            snapshot,
            actor,
            context.invocation.change_id,
            "local-mvp.v2",
            second.next_cursor,
            50,
            NOW + timedelta(days=1),
        )
        assert (len(first.items), len(second.items), len(last.items), last.next_cursor) == (50, 50, 1, None)
        assert not first.items[0].expired_now and last.items[0].expired_now
        assert not last.items[0].invalidations and canonical_hash(snapshot) == original
        assert "session_id" not in first.model_dump_json()
        assert first.items[0].authorization_hash == entry.request.authorization_hash
        with pytest.raises(DomainError):
            approval_page(
                snapshot,
                actor.model_copy(update={"user_id": 2}),
                context.invocation.change_id,
                "local-mvp.v2",
                first.next_cursor,
                50,
                NOW,
            )
        with pytest.raises(DomainError):
            approval_page(snapshot, actor, context.invocation.change_id, "local-mvp.v3", first.next_cursor, 50, NOW)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "case,expected",
    [
        ("available", "available"),
        ("unbound", "not_bound"),
        ("missing", "missing"),
        ("foreign", "missing"),
        ("conflict", "binding_conflict"),
        ("wrong_head", "binding_conflict"),
    ],
)
def test_report_observation_is_original_and_read_only(case: str, expected: str) -> None:
    """A DB double forbids writes and asserts every observation query is scoped."""

    class Database:
        @asynccontextmanager
        async def transaction(self):
            yield self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, sql, parameters):
            assert sql.startswith("SELECT")
            if "v11_analyses" in sql:
                assert parameters == ("object", "analysis")
                value = {
                    "snapshot_id": "original-snapshot",
                    "observation_id": "original-ci",
                    "report": {
                        "ci_observation_id": None
                        if case == "unbound"
                        else "wrong"
                        if case == "conflict"
                        else "original-ci",
                        "resolved_comparison": {"head_sha": "a" * 40},
                    },
                }
            else:
                assert parameters == ("object", "original-snapshot", "original-ci")
                value = (
                    None
                    if case in {"missing", "foreign"}
                    else {
                        "facts": {
                            "head_sha": ("b" if case == "wrong_head" else "a") * 40,
                            "status": "success",
                            "complete": True,
                            "checks_passed": True,
                            "captured_at": NOW,
                            "checks": [],
                        }
                    }
                )

            async def fetchone():
                return value

            return SimpleNamespace(fetchone=fetchone)

    async def scenario():
        db = Database()

        async def connect():
            return db

        async def authorize(identifier, actor, db, lock):
            assert identifier == "object" and actor.user_id == 1 and lock

        store = SimpleNamespace(connection=SimpleNamespace(connect=connect), object_row=authorize)
        result = await report_ci(
            store, "object", "analysis", LocalActor(user_id=1, session_id="synthetic", expires_at=9999999999)
        )
        assert result.availability == expected
        assert (result.observation is not None) == (expected == "available")

    asyncio.run(scenario())


def test_missing_ci_does_not_hide_saved_report() -> None:
    """A missing observation blocks current qualification, never rewrites the report."""
    store = SimpleNamespace(
        analysis_row=AsyncMock(
            return_value={
                "snapshot_id": "s",
                "generation": "g",
                "revision": 1,
                "dependencies": {},
                "status": "unknown",
                "report": {"reason": "original_unknown"},
            }
        ),
        snapshot=AsyncMock(return_value={"captured_at": NOW}),
        observation=AsyncMock(side_effect=DomainError("observation_missing")),
        dependencies=AsyncMock(return_value={}),
        object_row=AsyncMock(return_value={"repository_id": "checkout", "kind": "pr"}),
    )
    backend = V11Backend(SimpleNamespace(local=SimpleNamespace()), store, Mock(), Mock())
    backend.checked_dependencies = AsyncMock(return_value={})
    result = asyncio.run(
        backend.analysis("o", "a", LocalActor(user_id=1, session_id="synthetic", expires_at=9999999999))
    )
    assert result["reason"] == "original_unknown" and "observation_missing" in result["stale_reasons"]
    assert not result["rehearsal_eligibility"]["eligible"]
