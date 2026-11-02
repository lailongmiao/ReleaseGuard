"""Dedicated synthetic HTTP probe app with real isolated PostgreSQL checkpoints."""

import os
import re
import argparse
import asyncio
import sys
import json
from pathlib import Path
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
import uvicorn
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.releaseguard import router as legacy_router
from app.api.v1.auth import router as auth_router
from app.api.v1.evidence import router as evidence_router
from app.models.user import User
from app.api.v1.reviews import router
from app.core.config import settings
from app.core.langgraph.graph import build_agent_graph
from app.core.limiter import limiter
from app.services.local_mvp import LocalMVPService
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.v1_backend import V1Backend
from app.services.v11_backend import V11Backend
from app.services.v11_storage import V11Store
from scripts.v11_probe_fixtures import BusinessProbeProvider as ProbeProvider
from scripts.v11_probe_fixtures import seed, source
from scripts.inheritance_fixtures import LegacyProbeProvider, legacy_source
from scripts.v11_native_probe import NativeProbeProvider, synthetic_call
from app.services.llm.service import llm_service
from evals.v11_backend.http_evidence import EvidencePermit, HTTPEvaluationEvidence, router as evaluation_router


async def no_chat(state: Any) -> dict:
    """This probe has no chat or paid Provider fallback."""
    raise AssertionError("probe_chat_denied")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Refuse nonisolated settings and wire only explicit synthetic boundary adapters."""
    if (
        not re.fullmatch(r"releaseguard-v11-[0-9a-f]{12}", os.environ.get("V11_PROBE_PROJECT", ""))
        or settings.POSTGRES_DB != "v11_isolated"
        or settings.POSTGRES_HOST != "127.0.0.1"
    ):
        raise RuntimeError("isolated_probe_required")
    if settings.V1_REAL_MODELS_ENABLED or settings.V11_REAL_MODELS_ENABLED or settings.V11_REAL_READS_ENABLED:
        raise RuntimeError("offline_probe_required")
    if os.environ.get("V11_SYNTHETIC_EVIDENCE_DIR") and not settings.LOCAL_MVP_ONLY:
        raise RuntimeError("synthetic_evidence_local_mode_required")
    connection = LocalPostgresStore()
    store = V11Store(connection)

    async def graph_factory() -> Any:
        return app.state.graph

    local = LocalMVPService(graph_factory, connection)
    product = V1Backend(local, provider=LegacyProbeProvider(), github=legacy_source())
    evaluation = None
    provider: Any = ProbeProvider(store)
    if os.environ.get("V11_SYNTHETIC_EVIDENCE_DIR"):
        directory = Path(os.environ["V11_SYNTHETIC_EVIDENCE_DIR"])
        raw = json.loads((directory / "permit.private.json").read_text(encoding="utf-8"))
        permit = EvidencePermit.model_validate(raw["permit"])
        if permit.mode != "synthetic":
            raise RuntimeError("synthetic_permit_required")
        evaluation = HTTPEvaluationEvidence(
            directory, permit, raw["signature"], (directory / "key.private").read_bytes()
        )
        provider = NativeProbeProvider(store, enabled=False)
        llm_service.call_once_with_usage = synthetic_call
    await seed(store)
    if os.environ.get("V11_BROWSER_PROBE") == "1":
        settings.LOG_FORMAT = "json"
        password = os.environ["V11_BROWSER_PASSWORD"]
        async with await connection.connect() as db, db.transaction():
            for number in (1, 2, 3):
                await db.execute(
                    'UPDATE "user" SET hashed_password=%s WHERE id=%s', (User.hash_password(password), number)
                )
            await db.execute("SELECT setval(pg_get_serial_sequence('\"user\"','id'), (SELECT MAX(id) FROM \"user\"))")
    backend = V11Backend(
        product,
        store,
        provider,
        source(complete_evidence=os.environ.get("V11_COMPLETE_EVIDENCE_PROBE") == "1"),
        evaluation=evaluation,
    )
    await product.store.bootstrap(1, ("test",))
    await backend.bootstrap(1)
    async with await connection.connect() as db:
        # Test fixtures grant the explicit designated reviewer, never runtime auto-grants.
        await db.execute("UPDATE v1_sources SET acl='[1,2]'::jsonb WHERE version=0 AND authz_version=0")
    async with await connection.connect() as checkpoints:
        saver = AsyncPostgresSaver(checkpoints)
        await saver.setup()
        app.state.graph = build_agent_graph(no_chat, no_chat, saver, local.domain, **local.dependencies())
        app.state.local_mvp = local
        app.state.limiter = limiter
        # Loopback probe dispatch still runs every limiter decorator; high-volume cases use different processes.
        yield


async def rate_limit_handler(request: Request, error: Exception) -> Response:
    """Adapt the typed limiter exception to Starlette's handler protocol."""
    if not isinstance(error, RateLimitExceeded):
        raise error
    return _rate_limit_exceeded_handler(request, error)


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
if os.environ.get("V11_BROWSER_PROBE") == "1":
    app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(router, prefix="/api/v1/releaseguard/reviews")
app.include_router(legacy_router, prefix="/api/v1/releaseguard")
app.include_router(evidence_router, prefix="/api/v1/evidence")
if os.environ.get("V11_SYNTHETIC_EVIDENCE_DIR"):
    app.include_router(evaluation_router, prefix="/_evaluation")


@app.get("/probe-ready")
@limiter.limit("120/minute")
async def ready(request: Request) -> dict:
    """Synthetic readiness never claims real source/model or production readiness."""
    return {"ready": hasattr(app.state, "graph"), "provenance": "synthetic_sources_models_real_pg"}


def main() -> None:
    """Use a psycopg-compatible Windows event loop without changing global settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port))
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(server.serve())


if __name__ == "__main__":
    main()
