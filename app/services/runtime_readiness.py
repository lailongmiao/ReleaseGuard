"""Async local readiness; model reachability and business durability are separate."""

import asyncio
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Literal

from psycopg import AsyncConnection
from pydantic import BaseModel, ConfigDict
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.core.config import settings
from app.core.logging import logger


def migration_head() -> str:
    """Use the installed migration graph as the single authoritative head."""
    root = Path(__file__).resolve().parents[2]
    config = Config()
    config.set_main_option("script_location", str(root / "alembic"))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        raise ValueError("migration_single_head_required")
    return heads[0]


MIGRATION_HEAD = migration_head()
Component = Literal["ready", "unavailable", "degraded", "disabled"]


class DatabaseReadiness(BaseModel):
    """Safe output contains no DSN or database rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    database: bool
    migration: bool


class RuntimeReadinessResult(BaseModel):
    """Readiness never asserts production, Provider or durable domain support."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ready: bool
    status: Literal["ready", "not_ready"]
    components: dict[str, Component]
    provider: Literal["unverified"] = "unverified"
    production_execution: Literal["disabled"] = "disabled"
    optional_scope: Literal["startup_initialization"] = "startup_initialization"


async def check_database() -> DatabaseReadiness:
    """Bounded read-only probe against the configured template database."""
    connected = False
    try:
        async with asyncio.timeout(4):
            async with await AsyncConnection.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                connect_timeout=3,
                options="-c default_transaction_read_only=on -c statement_timeout=2000",
            ) as connection:
                connected = True
                cursor = await connection.execute("SELECT version_num FROM alembic_version")
                rows = await cursor.fetchall()
                return DatabaseReadiness(database=True, migration=rows == [(MIGRATION_HEAD,)])
    except Exception:
        logger.exception("runtime_database_probe_failed")
        return DatabaseReadiness(database=connected, migration=False)


class RuntimeReadinessService:
    """Required DB/migration/Graph; optional memory/cache may degrade."""

    def __init__(
        self,
        database_probe: Callable[[], Awaitable[DatabaseReadiness]],
        graph_probe: Callable[[], bool],
        optional_probe: Callable[[], dict[str, Component]],
    ) -> None:
        """Inject probes so readiness tests require no live database or Provider."""
        self.database_probe, self.graph_probe, self.optional_probe = database_probe, graph_probe, optional_probe

    async def check(self) -> RuntimeReadinessResult:
        """Probe current DB state; do not trust a startup-only healthy latch."""
        try:
            database = await self.database_probe()
            graph = self.graph_probe()
            components: dict[str, Component] = {
                "database": "ready" if database.database else "unavailable",
                "migration": "ready" if database.migration else "unavailable",
                "graph": "ready" if graph else "unavailable",
                **self.optional_probe(),
            }
            ready = database.database and database.migration and graph
            return RuntimeReadinessResult(ready=ready, status="ready" if ready else "not_ready", components=components)
        except Exception:
            logger.exception("runtime_readiness_probe_failed")
            return RuntimeReadinessResult(ready=False, status="not_ready", components={"probe": "unavailable"})
