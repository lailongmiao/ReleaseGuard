"""Upgrade synthetic legacy facts in a fresh isolated database without changing their bytes."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from psycopg.types.json import Jsonb
from rich.console import Console

from app.services.local_mvp_storage import LocalPostgresStore, SavedFacts
from app.services.releaseguard.normalization import canonical_hash
from scripts.v11_probe import configure


async def legacy_seed() -> dict:
    """Only the named isolated DB at the precise V1 parent revision is eligible."""
    connection = LocalPostgresStore()
    hashes = {}
    async with await connection.connect() as db, db.transaction():
        head = await (await db.execute("SELECT version_num FROM alembic_version")).fetchone()
        if head["version_num"] != "v1_backend_02":
            raise ValueError("v1_parent_required")
        for number in (1, 2, 3):
            await db.execute(
                'INSERT INTO "user"(id,email,hashed_password,created_at) VALUES(%s,%s,%s,now())',
                (number, f"v11-synthetic-{number}@example.invalid", "synthetic-no-password-login"),
            )
        for version in ("local-mvp.v1", "local-mvp.v2"):
            facts = SavedFacts.model_validate(
                {
                    "version": version,
                    "commands": {"unresolved": {"status": "started", "hash": canonical_hash("unknown")}},
                    **(
                        {
                            "product": {
                                "analysis_status": "unknown",
                                "binding": {"mode": "product_readonly", "pr_number": 1},
                            }
                        }
                        if version.endswith("v2")
                        else {}
                    ),
                }
            ).model_dump(mode="json")
            payload = {"facts": facts, "hash": canonical_hash(facts)}
            identity = "synthetic-legacy:" + version
            await db.execute(
                "INSERT INTO local_mvp_records(change_id,owner_user_id,approver_user_id,creation_hash,payload) VALUES(%s,1,2,%s,%s)",
                (identity, canonical_hash(version), Jsonb(payload)),
            )
            hashes[identity] = canonical_hash(payload)
        await db.execute(
            "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,result) VALUES(1,'legacy-unknown',%s,'synthetic-legacy','started',%s)",
            (canonical_hash("legacy-unknown"), Jsonb({"claimed": True})),
        )
        await db.execute(
            "INSERT INTO v1_budget(owner_id,command_id,attempt,reserved,charge) VALUES(1,'legacy-unknown','analysis-1',0.1,0.1)"
        )
    return hashes


async def verify(hashes: dict) -> dict:
    """Raw saved JSON hashes and unresolved charges remain identical after migration."""
    async with await LocalPostgresStore().connect() as db:
        rows = await (
            await db.execute(
                "SELECT change_id,payload FROM local_mvp_records WHERE change_id=ANY(%s)", (list(hashes),)
            )
        ).fetchall()
        budget = await (
            await db.execute("SELECT charge,reserved,usage FROM v1_budget WHERE command_id='legacy-unknown'")
        ).fetchone()
        command = await (
            await db.execute("SELECT status,result FROM v1_commands WHERE command_id='legacy-unknown'")
        ).fetchone()
    return {
        "provenance": "actual_new_isolated_pg_synthetic_legacy_records",
        "legacy_facts_unchanged": len(rows) == 2
        and all(canonical_hash(row["payload"]) == hashes[row["change_id"]] for row in rows),
        "unknown_preserved": command["status"] == "started"
        and command["result"] == {"claimed": True}
        and budget["charge"] == budget["reserved"]
        and str(budget["charge"]) == "0.10000000",
    }


def main() -> int:
    """No stamp, downgrade or destructive cleanup is available."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("report_already_exists")
    _, values = configure(args.compose_command)
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        hashes = runner.run(legacy_seed())
    migration = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env={**os.environ, **values},
        capture_output=True,
        timeout=60,
    )
    (args.compose_command.parent / "migration-upgrade.private.log").write_bytes(migration.stdout + migration.stderr)
    if migration.returncode:
        raise RuntimeError("migration_failed_no_automatic_repair")
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        result = runner.run(verify(hashes))
    result["passed"] = result["legacy_facts_unchanged"] and result["unknown_preserved"]
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
