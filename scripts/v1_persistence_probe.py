"""Isolated actual PG claims, budget contention, generation CAS and scoped rank."""

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from psycopg.types.json import Jsonb

from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.state_machine import DomainError
from app.services.v1_rag import fuse
from app.services.v1_storage import V1Store
from app.services.v1_provider import BudgetedProvider
from app.services.releaseguard.normalization import canonical_hash
from evals.v1_backend.provenance import code_state
from scripts.mvp_product_probe import ProductProbe
from scripts.v1_provider_probe import isolated_database, reuse_sessions


async def run(probe: ProductProbe, args: argparse.Namespace) -> dict:
    """Faults are confined to the named new DB and rolled back on failure."""
    isolated_database(args.compose_command)
    await reuse_sessions(probe, args.compose_command)
    connection = LocalPostgresStore()
    store = V1Store(connection)
    actor = await connection.authenticate(probe.session_ids[1], 4102444800)
    third = await connection.authenticate(probe.session_ids[3], 4102444800)
    approver = await connection.authenticate(probe.session_ids[2], 4102444800)
    checks = []

    def check(name: str, value: bool) -> None:
        checks.append({"case": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    nonce = uuid4().hex
    lexical, semantic = await store.candidates(actor, 2557, "unmatchable_lexical_probe_" + nonce, None)
    check("pg_zero_keyword_matches_do_not_enter_rrf", not lexical and not semantic)
    async with await connection.connect() as db:
        hidden = await (
            await db.execute(
                "SELECT change_id FROM local_mvp_records WHERE approver_user_id=%s "
                "AND COALESCE(payload->'facts'->'product','{}'::jsonb)<>'{}'::jsonb "
                "AND (SELECT count(*) FROM v1_sources s WHERE s.source_id IN ('github-checkout','markdown-rules') "
                "AND NOT s.deleted AND s.acl @> %s::jsonb)<2",
                (approver.user_id, Jsonb([approver.user_id])),
            )
        ).fetchall()
    if not hidden:
        raise AssertionError("actual_revoked_product_fixture_required")
    listed = await connection.list_records(approver, inbox=False, cursor="", limit=50)
    inbox = await connection.list_records(approver, inbox=True, cursor="", limit=50)
    check(
        "pg_revoked_source_filtered_before_list_and_inbox_pagination",
        not {row["change_id"] for row in hidden}.intersection(listed + inbox),
    )

    async def claim(index: int) -> str:
        try:
            result = await store.claim(actor, nonce + ":claim", "a" * 64, "persistence-probe")
            return "original" if result is None else "replayed"
        except DomainError as error:
            return error.code

    contenders = await asyncio.gather(*(claim(i) for i in range(8)))
    check(
        "pg_command_concurrent_single_owner",
        contenders.count("original") == 1 and contenders.count("provider_command_outcome_unknown") == 7,
    )
    check("pg_new_store_unknown_zero_resend", await claim(9) == "provider_command_outcome_unknown")
    await store.complete(actor, nonce + ":claim", {"status": "diagnostic", "reason": "unknown_no_resend"})
    check(
        "pg_completed_exact_replay",
        await store.claim(actor, nonce + ":claim", "a" * 64, "persistence-probe")
        == {"status": "diagnostic", "reason": "unknown_no_resend"},
    )
    try:
        await store.claim(actor, nonce + ":claim", "b" * 64, "persistence-probe")
        check("pg_payload_conflict", False)
    except DomainError as error:
        check("pg_payload_conflict", error.code == "idempotency_conflict")
    async with await connection.connect() as db:
        spent = (await (await db.execute("SELECT COALESCE(SUM(charge),0) AS total FROM v1_budget")).fetchone())[
            "total"
        ]
    cap = spent + Decimal("0.00000002")

    async def reserve(number: int) -> str:
        command = nonce + ":budget:" + str(number)
        await store.claim(actor, command, "c" * 64, "persistence-probe")
        try:
            await store.reserve(actor, command, "offline-no-provider", Decimal("0.00000001"), cap)
            await store.usage(
                actor, command, "offline-no-provider", {"provenance": "synthetic_budget_contention_no_network"}, None
            )
            return "reserved"
        except DomainError as error:
            return error.code

    values = await asyncio.gather(*(reserve(i) for i in range(6)))
    check(
        "pg_atomic_budget_at_most_two", values.count("reserved") == 2 and values.count("model_budget_exhausted") == 4
    )
    # Synthetic reservations are identified individually and remain traceable.
    async with await connection.connect() as db:
        await db.execute(
            "UPDATE v1_budget SET charge=0,usage=%s::jsonb WHERE command_id LIKE %s",
            (
                '{"provenance":"synthetic_no_provider_call_confirmed","actual_cost":"not_applicable"}',
                nonce + ":budget:%",
            ),
        )
    source = await store.source("markdown-rules", actor)
    source_seals = {
        name: {"version": row["version"], "generation": row["generation"], "acl": row["acl"]}
        for name in ("github-checkout", "markdown-rules")
        for row in [await store.source(name, actor)]
    }
    async with await connection.connect() as reader, await connection.connect() as writer:
        async with reader.transaction():
            await store.lock_sources(reader, source_seals, actor)
            await writer.execute("SET lock_timeout='100ms'")
            try:
                await writer.execute("UPDATE v1_sources SET version=version+1 WHERE source_id='markdown-rules'")
                check("pg_analysis_commit_lock_prevents_acl_write_race", False)
            except Exception as error:
                check("pg_analysis_commit_lock_prevents_acl_write_race", getattr(error, "sqlstate", None) == "55P03")
            finally:
                await writer.rollback()
    check("pg_source_lock_probe_preserves_fixture", await store.source("markdown-rules", actor) == source)
    generation = await store.generation(source["generation"])
    new_store = V1Store(LocalPostgresStore())
    check("pg_index_restored_new_store", await new_store.generation(source["generation"]) == generation)
    originals = (await store.candidates(actor, args.pr, "staging approval", None))[0]
    check("pg_nonempty_lexical", bool(originals))
    denied = await store.candidates(third, args.pr, "staging approval", [1.0] + [0.0] * 1535)
    check("pg_acl_before_both_rankings", not denied[0] and not denied[1])
    invalid = originals[0].model_copy(update={"vector": [1.0], "generation": "invalid-publication"})
    try:
        await store.publish(actor, source, "invalid-publication", generation, [invalid])
        check("pg_failed_publication_keeps_old", False)
    except DomainError:
        check(
            "pg_failed_publication_keeps_old",
            (await store.source("markdown-rules", actor))["generation"] == source["generation"],
        )
    source_changed = {**source, "version": source["version"] - 1}
    try:
        await store.publish(actor, source_changed, "invalid-publication", generation, [invalid])
    except DomainError:
        check(
            "pg_generation_still_active",
            (await store.source("markdown-rules", actor))["generation"] == source["generation"],
        )
    async with await connection.connect() as db:
        row = await (
            await db.execute(
                "SELECT embedding::text AS vector FROM v1_chunks WHERE generation=%s LIMIT 1", (source["generation"],)
            )
        ).fetchone()
    vector = json.loads(row["vector"])
    lexical, semantic = await new_store.candidates(actor, args.pr, "Independent verification", vector)
    check("pg_actual_exact_vector_and_rrf", bool(semantic) and bool(fuse(lexical, semantic)))
    retrieval = {"status": "unverified", "reason": "real_query_embedding_not_requested"}
    if args.real_query_embeddings:
        data = json.loads(Path("evals/datasets/releaseguard_rag_v1.json").read_text(encoding="utf-8"))
        cases = [c for c in data["cases"] if c.get("kind", "backend") == "backend"]
        queries = sorted({c["query"] for c in cases})
        identifier = nonce + ":rag-queries"
        await store.claim(actor, identifier, canonical_hash(queries), "retrieval-probe")
        vectors = await BudgetedProvider(store, enabled=True).embed(queries, actor, identifier, "real-query")
        by_query = dict(zip(queries, vectors, strict=True))
        results = []
        for attempt in (1, 2):
            for case in cases:
                lexical, semantic = await store.candidates(actor, args.pr, case["query"], by_query[case["query"]])
                found = {c.document.resource for c in fuse(lexical, semantic)}
                recall = len(found.intersection(case["gold"])) / len(set(case["gold"]))
                results.append(
                    {"case_id": case["case_id"], "attempt": attempt, "recall_at_5": recall, "documents": len(found)}
                )
        macro = sum(r["recall_at_5"] for r in results) / len(results)
        check("actual_pg_embedding_recall_macro", macro >= 0.85)
        await store.complete(actor, identifier, {"status": "completed", "queries": len(queries)})
        retrieval = {
            "status": "passed",
            "recall_at_5_macro": macro,
            "unique_queries": len(queries),
            "cases": len(cases),
            "attempts": len(results),
            "rows": results,
            "scope": "four authorized documents, three unique queries; no larger corpus claim",
            "dataset_hash": data["dataset_hash"],
        }
    return {
        "passed": True,
        "checks": checks,
        "code": code_state(),
        "transport": "actual_async_postgresql_pgvector",
        "provider_calls": 1 if args.real_query_embeddings else 0,
        "retrieval": retrieval,
        "production_execution": "disabled",
    }


def main() -> int:
    """No real external effects or old database targets are accepted."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:34080")
    parser.add_argument("--pr", type=int, default=2557)
    parser.add_argument("--real-query-embeddings", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    probe = ProductProbe(args.url, args.compose_command, "v1_pg")
    try:
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            report = runner.run(run(probe, args))
    except Exception as error:
        report = {
            "passed": False,
            "failure": str(error) if isinstance(error, AssertionError) else type(error).__name__,
        }
    finally:
        probe.client.close()
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Console().print(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
