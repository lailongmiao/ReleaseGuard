"""Runtime HTTP probes tested without live identity, databases or Provider."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.runtime import get_readiness, router
from app.core.limiter import limiter
from app.services.runtime_readiness import DatabaseReadiness, RuntimeReadinessService


@pytest.mark.parametrize(
    "database,migration,graph,expected",
    [
        (False, False, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_required_dependencies_and_http_status(database: bool, migration: bool, graph: bool, expected: bool) -> None:
    """DB-only/liveness and degraded memory cannot imply Graph readiness."""

    async def probe() -> DatabaseReadiness:
        return DatabaseReadiness(database=database, migration=migration)

    service = RuntimeReadinessService(probe, lambda: graph, lambda: {"memory": "degraded", "cache": "disabled"})
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    app.dependency_overrides[get_readiness] = lambda: service
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        result = client.get("/ready")
        assert result.status_code == (200 if expected else 503)
        assert result.json()["ready"] is expected
        assert result.json()["provider"] == "unverified"
        assert result.json()["production_execution"] == "disabled"


def test_probe_error_is_safe_and_not_ready() -> None:
    """No raw connection exception is returned by the probe."""

    async def probe() -> DatabaseReadiness:
        raise RuntimeError("synthetic private DSN")

    result = asyncio.run(RuntimeReadinessService(probe, lambda: True, lambda: {}).check())
    assert not result.ready
    assert "synthetic private" not in result.model_dump_json()
