"""New API contracts fail closed without configured runtime and reject loose input."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI

from app.api.v1.releaseguard import actor
from app.api.v1.reviews import review_runtime, router
from app.core.limiter import limiter
from app.schemas.local_mvp import LocalActor
from app.services.v11_queries import decode_cursor, encode_cursor
from app.services.releaseguard.state_machine import DomainError


def test_new_openapi_typed_isolated_contract() -> None:
    """Report, selector and command envelopes cannot silently become old AnalysisOutput."""
    app = FastAPI()
    app.include_router(router, prefix="/reviews")
    schema = app.openapi()
    assert "/reviews/{object_id}/analyze" in schema["paths"]
    assert schema["components"]["schemas"]["ReviewBinding"]["additionalProperties"] is False
    assert "ReviewAnalysis" in schema["components"]["schemas"]
    assert "AnalysisOutput" not in schema["components"]["schemas"]


def test_cursor_binds_actor_filters_and_acl() -> None:
    """Signed cursors cannot carry another user's accessible scan position."""
    current = LocalActor(user_id=1, session_id="synthetic-session", expires_at=4102444800)
    scope = {"kind": "snapshots", "object": "one"}
    cursor = encode_cursor(current, scope, "acl-version-1", [datetime.now(UTC).isoformat(), "object:one"])
    assert decode_cursor(cursor, current, scope, "acl-version-1", None)[1] == "object:one"
    for principal, filters, acl in [
        (current.model_copy(update={"user_id": 2}), scope, "acl-version-1"),
        (current, {"object": "two"}, "acl-version-1"),
        (current, scope, "acl-version-2"),
    ]:
        with pytest.raises(DomainError, match="cursor_invalid_or_expired"):
            decode_cursor(cursor, principal, filters, acl, None)


def test_invalid_payload_uses_safe_error_envelope() -> None:
    """Pydantic input details expose locations, never rejected secret-like values."""
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router, prefix="/reviews")
    app.dependency_overrides[actor] = lambda: LocalActor(
        user_id=1, session_id="synthetic-session", expires_at=4102444800
    )
    app.dependency_overrides[review_runtime] = lambda: object()

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local.invalid"
        ) as client:
            response = await client.post(
                "/reviews/one/sync",
                json={"command_id": "synthetic", "expected_object_version": "1", "untrusted": "private-value"},
            )
            assert response.status_code == 422
            assert response.json()["code"] == "review_input_invalid"
            assert "private-value" not in response.text

    asyncio.run(run())
