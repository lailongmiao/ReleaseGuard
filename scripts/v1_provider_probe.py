"""Approved official-host V1 models; persistent budget and sanitized outputs only."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from math import ceil
from time import perf_counter
from uuid import uuid4

from dotenv import dotenv_values
from jose import jwt
from rich.console import Console

from app.core.config import settings
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_provider import BudgetedProvider, PRICE_DATE, PRICES
from app.services.v1_rag import chunk_documents
from app.services.v1_sources import LEGACY_MARKDOWN_FILES, MarkdownSource
from app.services.v1_storage import V1Store
from evals.v1_backend.provenance import code_state
from evals.v1_backend.review import answer_contract
from scripts.mvp_product_probe import ProductProbe


def isolated_database(command: Path) -> None:
    """Configure only the named new loopback DB; never print its credentials."""
    args = json.loads(command.read_text(encoding="utf-8"))
    if args[:3] != ["docker", "compose", "--project-name"] or not args[3].startswith("releaseguard-mvp-"):
        raise ValueError("isolated_command_required")
    env_path = Path(args[5]).resolve()
    if env_path.parent != command.resolve().parent:
        raise ValueError("private_config_required")
    values = dotenv_values(env_path)
    settings.POSTGRES_HOST = "127.0.0.1"
    settings.POSTGRES_PORT = int(values["LOCAL_DB_PORT"] or "0")
    settings.POSTGRES_DB = values["POSTGRES_DB"] or ""
    settings.POSTGRES_USER = values["POSTGRES_USER"] or ""
    settings.POSTGRES_PASSWORD = values["POSTGRES_PASSWORD"] or ""
    settings.V1_REAL_MODELS_ENABLED = True
    settings.V1_MODEL_BUDGET_CNY = float(values.get("V1_MODEL_BUDGET_CNY") or "100")
    settings.LANGFUSE_TRACING_ENABLED = False


async def reuse_sessions(probe: ProductProbe, command: Path) -> None:
    """Reuse only previously HTTP-registered synthetic users in the new test DB.

    The isolated operator signs ephemeral session credentials; the API still
    checks its signature and live DB session/object/source authority. This is
    reported separately from fresh HTTP registration, never as production auth.
    """
    isolated_database(command)
    config = dotenv_values(Path(probe.args[5]))
    db = LocalPostgresStore()
    async with await db.connect() as connection:
        source = await (
            await connection.execute("SELECT owner_id,acl FROM v1_sources WHERE source_id='github-checkout'")
        ).fetchone()
        if not source:
            raise ValueError("prior_synthetic_source_fixture_required")
        rows = await (
            await connection.execute(
                'SELECT u.id AS user_id,s.id AS session_id FROM "user" u JOIN session s ON s.user_id=u.id WHERE u.email LIKE %s ORDER BY u.id,s.id',
                ("synthetic-probe-%@example.com",),
            )
        ).fetchall()
    owner = source["owner_id"]
    users = {r["user_id"]: r["session_id"] for r in rows}
    approvers = [u for u in source["acl"] if u != owner and u in users]
    if not approvers:
        approvers = [u for u in users if u != owner][:1]
    unrelated = [u for u in users if u not in source["acl"]]
    unrelated = [u for u in unrelated if u != approvers[0]] if approvers else unrelated
    if owner not in users or not approvers or not unrelated:
        raise ValueError("three_prior_synthetic_fixture_users_required")
    for number, user in enumerate((owner, approvers[0], unrelated[0]), 1):
        session = users[user]
        probe.session_ids[number] = session
        probe.credentials[number] = jwt.encode(
            {"sub": session, "exp": datetime.now(UTC).timestamp() + 3600},
            config["JWT_SECRET_KEY"] or "",
            algorithm=settings.JWT_ALGORITHM,
        )
        probe.identities[number] = probe.request("GET", "/api/v1/releaseguard/identity", user=number)
    probe.check("prior_registered_synthetic_sessions_db_reauthorized", True)


async def run(args: argparse.Namespace, probe: ProductProbe) -> dict:
    """Freeze holdout before calling, keep failure samples, never retry unknown."""
    isolated_database(args.compose_command)
    db = LocalPostgresStore()
    actor = await db.authenticate(probe.session_ids[1], datetime.now(UTC).timestamp() + 3600)
    store = V1Store(db)
    provider = BudgetedProvider(store, enabled=True, limit=Decimal("30" if args.mode == "smoke" else "100"))
    collection = await MarkdownSource().collect()
    # Historical all-context component probe remains restricted to its three policies.
    documents = [d for d in collection.documents if d.resource in LEGACY_MARKDOWN_FILES]
    chunks = chunk_documents(documents, "probe-approved-markdown")
    command = "v1-model-probe:" + uuid4().hex
    await store.claim(actor, command, canonical_hash({"mode": args.mode}), "model-probe")
    started = perf_counter()
    vectors = await provider.embed([c.text for c in chunks], actor, command, "corpus")
    embedded = [c.model_copy(update={"vector": v}) for c, v in zip(chunks, vectors, strict=True)]
    queries = json.loads(Path("evals/datasets/v1_model_holdout.json").read_text(encoding="utf-8"))
    rows = []
    cases = (
        [c for c in queries["cases"] if c["case_id"] == args.case_id]
        if args.case_id
        else queries["cases"][:1]
        if args.mode == "smoke"
        else queries["cases"]
    )
    if not cases or args.case_id and args.mode != "smoke":
        raise ValueError("smoke_case_invalid")
    for case in cases:
        for attempt in range(1, (1 if args.mode == "smoke" else 2) + 1):
            identifier = command + ":" + case["case_id"] + ":" + str(attempt)
            await store.claim(actor, identifier, canonical_hash(case), "model-holdout")
            at = perf_counter()
            row = {
                "case_id": case["case_id"],
                "attempt": attempt,
                "status": "failed",
                "expected": case["expected"],
                "human_rating": "unverified",
            }
            try:
                output, telemetry = await provider.analyze(case["query"], embedded, actor, identifier, identifier)
                row.update(status="passed", output=output.model_dump(mode="json"), telemetry=telemetry)
                row["citation_correct"] = all(
                    set(c.citation_ids) <= {x.chunk_id for x in embedded} for c in output.claims
                )
                row["unknown_correct"] = answer_contract(row)
            except DomainError as error:
                row["reason"] = error.code
            row["latency_seconds"] = round(perf_counter() - at, 4)
            rows.append(row)
            await store.complete(actor, identifier, {"status": row["status"], "reason": row.get("reason")})
    async with await db.connect() as connection:
        budget = await (
            await connection.execute(
                "SELECT count(*) AS attempts,COALESCE(sum(charge),0) AS charged,COALESCE(sum(reserved),0) AS reserved FROM v1_budget"
            )
        ).fetchone()
    await store.complete(actor, command, {"status": "completed", "embedding_dimensions": len(vectors[0])})
    latency = sorted(x["latency_seconds"] for x in rows)
    return {
        "mode": args.mode,
        "code": code_state(),
        "dataset_hash": canonical_hash(queries),
        "evidence": "actual_official_provider_with_approved_markdown_only",
        "embedding": {
            "documents": len(documents),
            "chunks": len(chunks),
            "dimensions": len(vectors[0]),
            "latency_seconds": round(perf_counter() - started - sum(latency), 4),
            "cost": "unknown",
        },
        "model": settings.V1_CHAT_MODEL,
        "price_date": PRICE_DATE,
        "price_references": [v[2] for v in PRICES.values()],
        "actual_analyses": len(rows),
        "rows": rows,
        "review_sources": [c.model_dump(mode="json", exclude={"vector"}) for c in chunks],
        "p95_seconds": latency[max(0, ceil(len(latency) * 0.95) - 1)] if latency else None,
        "budget": {
            "attempts": budget["attempts"],
            "peak_price_charge_cny": str(budget["charged"]),
            "reserved_cny": str(budget["reserved"]),
            "limit_cny": str(provider.limit),
            "actual_cost": "unknown",
        },
        "human_quality": "unverified_requires_review",
        "judge": "unverified_not_called",
        "langfuse": "disabled_no_workspace_authorization",
        "production_execution": "disabled",
    }


def main() -> int:
    """Only an explicitly enabled, loopback isolated run can spend budget."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:33080")
    parser.add_argument("--mode", choices=["smoke", "holdout"], default="smoke")
    parser.add_argument("--case-id")
    parser.add_argument("--reuse-isolated-synthetic-sessions", action="store_true")
    parser.add_argument("--approved-v1-models", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    probe = ProductProbe(args.url, args.compose_command, "normal")
    report = {"status": "failed", "raw_response": "not_saved"}
    try:
        probe.restart("isolated_probe_initial_process")
        if args.reuse_isolated_synthetic_sessions:
            with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
                runner.run(reuse_sessions(probe, args.compose_command))
        else:
            probe.accounts()
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            report = runner.run(run(args, probe))
    except Exception as error:
        report["failure"] = error.code if isinstance(error, DomainError) else type(error).__name__
    finally:
        probe.client.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Console().print(
        {
            "report": str(args.report),
            "analyses": report.get("actual_analyses", 0),
            "failure": report.get("failure"),
            "budget": report.get("budget"),
        }
    )
    return 1 if report.get("failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
