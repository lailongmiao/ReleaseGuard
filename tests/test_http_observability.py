"""Actual ASGI route templates and context cleanup for requests and errors."""

import asyncio

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, Counter, Histogram

from app.core import middleware
from app.core.logging import get_context
from app.core.middleware import LoggingContextMiddleware, MetricsMiddleware


def test_http_metrics_are_bounded_and_context_cleared(monkeypatch) -> None:
    """Raw URL ids and unknown paths cannot create new label cardinality."""
    registry = CollectorRegistry()
    counter = Counter("test_http_total", "Requests", ["method", "endpoint", "status"], registry=registry)
    histogram = Histogram("test_duration_seconds", "Requests", ["method", "endpoint"], registry=registry)
    monkeypatch.setattr(middleware, "http_requests_total", counter)
    monkeypatch.setattr(middleware, "http_request_duration_seconds", histogram)
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingContextMiddleware)

    @app.get("/case/{case_id}")
    async def case(case_id: str) -> dict:
        return {"context": get_context(), "case_id": case_id}

    @app.get("/failure")
    async def failure() -> None:
        raise RuntimeError("synthetic failure")

    async def probe() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://local.invalid"
        ) as client:
            responses = await asyncio.gather(client.get("/case/one"), client.get("/case/two"))
            assert all(r.status_code == 200 for r in responses)
            assert (await client.get("/unknown-private-one")).status_code == 404
            assert (await client.get("/unknown-private-two")).status_code == 404
            assert (await client.get("/failure")).status_code == 500
        assert get_context() == {}

    asyncio.run(probe())
    assert (
        registry.get_sample_value("test_http_total", {"method": "GET", "endpoint": "/case/{case_id}", "status": "200"})
        == 2
    )
    assert (
        registry.get_sample_value("test_http_total", {"method": "GET", "endpoint": "unmatched", "status": "404"}) == 2
    )
