"""Readiness on fresh additive test databases; never mutate an existing schema."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from psycopg import sql
from rich.console import Console

from app.core.config import settings
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.runtime_readiness import check_database, migration_head
from evals.v1_backend.provenance import code_state
from scripts.v1_provider_probe import isolated_database


async def run(command: Path) -> dict:
    """Create only new random DBs, keep them for diagnostics; no DROP/downgrade."""
    isolated_database(command)
    original = settings.POSTGRES_DB
    rows = []
    created = []
    try:
        ready = await check_database()
        rows.append({"case": "current_unique_head_ready", "passed": ready.database and ready.migration})
        for kind in ("behind", "missing", "unknown"):
            name = "v1_readiness_" + kind + "_" + uuid4().hex[:12]
            async with await LocalPostgresStore().connect() as admin:
                await admin.set_autocommit(True)
                await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            created.append(name)
            settings.POSTGRES_DB = name
            if kind != "missing":
                async with await LocalPostgresStore().connect() as fresh:
                    await fresh.execute("CREATE TABLE alembic_version(version_num varchar(32) NOT NULL)")
                    await fresh.execute(
                        "INSERT INTO alembic_version VALUES(%s)",
                        ("mvp_closeout_01" if kind == "behind" else "unrecognized_head",),
                    )
                    await fresh.commit()
            result = await check_database()
            rows.append({"case": kind + "_not_ready", "passed": result.database and not result.migration})
            settings.POSTGRES_DB = original
    finally:
        settings.POSTGRES_DB = original
    return {
        "passed": all(r["passed"] for r in rows) and len(rows) == 4,
        "checks": rows,
        "head": migration_head(),
        "code": code_state(),
        "new_database_fixtures": created,
        "existing_database_mutations": 0,
        "destructive_operations": 0,
    }


def main() -> int:
    """Only the existing new-isolated command file may supply credentials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        report = runner.run(run(args.compose_command))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Console().print({"passed": report["passed"], "checks": len(report["checks"]), "head": report["head"]})
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
