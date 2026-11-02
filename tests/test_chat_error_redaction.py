"""HTTP and SSE errors never expose raw Provider or database messages."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import chatbot
from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.limiter import limiter


def test_chat_and_stream_errors_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use actual routes with injected faults and synthetic identity, no Provider."""

    async def failed(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic-private-provider-payload")

    async def stream(*args: object, **kwargs: object):
        raise RuntimeError("synthetic-private-provider-payload")
        yield  # pragma: no cover

    monkeypatch.setattr(settings, "SESSION_NAMING_ENABLED", False)
    # Exercise the injected fault, not the separate local-only 503 capability gate.
    monkeypatch.setattr(settings, "LOCAL_MVP_ONLY", False)
    monkeypatch.setattr(chatbot.agent, "get_response", failed)
    monkeypatch.setattr(chatbot.agent, "get_stream_response", stream)
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chatbot.router)
    app.dependency_overrides[get_current_session] = lambda: SimpleNamespace(
        id="synthetic-session", user_id=1, username="synthetic-user", name="synthetic"
    )
    with TestClient(app) as client:
        body = {"messages": [{"role": "user", "content": "synthetic hello"}]}
        response = client.post("/chat", json=body)
        assert response.status_code == 500
        assert response.json()["detail"] == "chat_unavailable"
        response = client.post("/chat/stream", json=body)
        assert "synthetic-private" not in response.text
        assert "chat_stream_unavailable" in response.text
