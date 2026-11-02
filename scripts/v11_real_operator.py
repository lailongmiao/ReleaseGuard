"""Explicit local real-acceptance operations; never delete historical resources."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from jose import jwt
from rich.console import Console
import httpx
import uvicorn

from app.core.config import settings
from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import ReviewBinding
from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_sources import ReviewGitHubSource
from app.services.v11_storage import V11Store
from app.services.v1_provider import PRICE_EXPIRY, PRICES
from evals.v1_backend.provenance import code_state
from evals.v11_backend.contracts import Dataset
from evals.v11_backend.real import Authorization
from scripts.v11_stack import configuration, initialize

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = (
    "releaseguard-mvp-ee9ac120007b",
    "releaseguard-mvp-350e0584bb76",
    "releaseguard-mvp-10a1d6686966",
)


def save(path: Path, value: object) -> None:
    """Never replace previous evidence."""
    with path.open("x", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def inventory(export: Path, report: Path) -> None:
    """Read only aggregate historical ledgers and exclude credentials/account IDs."""
    ledgers = []
    for project in HISTORICAL:
        name = project + "-db-1"
        descriptors = (
            Path(os.environ["LOCALAPPDATA"]).joinpath("Temp").glob("releaseguard-mvp-private-*/runtime-command.json")
        )
        matches = [json.loads(p.read_text(encoding="utf-8")) for p in descriptors]
        selected = [command for command in matches if command[3] == project]
        if len(selected) != 1:
            raise ValueError("unique_historical_descriptor_required")
        env = {key: value for key, value in dotenv_values(selected[0][5]).items() if value is not None}
        query = (
            "BEGIN READ ONLY; SELECT json_build_object('records',count(*),'charge',sum(charge),"
            "'reserved',sum(reserved),'usage_empty',count(*) FILTER(WHERE usage='{}'::jsonb)) "
            "FROM v1_budget; COMMIT;"
        )
        result = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "psql",
                "-X",
                "-q",
                "-A",
                "-t",
                "-U",
                env["POSTGRES_USER"],
                "-d",
                env["POSTGRES_DB"],
                "-c",
                query,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        ledgers.append({"project": project, **json.loads(result.stdout.strip())})
    amount_path = export / "amount-2026-09-12_2026-10-11.csv"
    cost_path = export / "cost-2026-09-12_2026-10-11.csv"
    with amount_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    with cost_path.open(encoding="utf-8-sig", newline="") as file:
        costs = list(csv.DictReader(file))
    totals = {}
    for row in rows:
        entry = totals.setdefault(row["type"], {"amount": 0, "cost_cny": Decimal("0")})
        entry["amount"] += int(row["amount"])
        entry["cost_cny"] += Decimal(row["price"] or "0") * int(row["amount"])
    billed = sum((Decimal(row["cost"]) for row in costs), Decimal("0"))
    if billed != sum(entry["cost_cny"] for entry in totals.values()):
        raise ValueError("export_reconciliation_failed")
    evidence = {
        "at": datetime.now(UTC).isoformat(),
        "historical_ledgers_readonly": ledgers,
        "deepseek_export": {
            "cost_cny": str(billed),
            "totals": totals,
            "exclusive_end": "2026-10-11T00:00:00+08:00",
            "file_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (amount_path, cost_path)},
        },
        "invoice_attribution": "account export; does not cover embedding or identify every ledger attempt",
        "unknown_released": False,
    }
    save(report, evidence)
    Console().print(evidence)


def setup(token_env: Path, evidence: Path, port: int) -> None:
    """New environment copies only approved provider fields, never legacy DB credentials."""
    report = json.loads(evidence.read_text(encoding="utf-8"))
    # Reconcile latest charge against historical carryover; old reserve totals are not additive.
    latest = report["historical_ledgers_readonly"][-1]
    prior = sum(
        (
            max(Decimal(str(row["reserved"])), Decimal(str(row["charge"])))
            for row in report["historical_ledgers_readonly"][:-1]
        ),
        Decimal("0"),
    )
    carry = prior + Decimal(str(latest["charge"]))
    if carry < Decimal("22.60836278"):
        raise ValueError("historical_charge_regression")
    descriptor = initialize(port)
    _, values = configuration(descriptor)
    old = dotenv_values(token_env)
    credential = old.get("V1_GITHUB_READ_TOKEN")
    if not credential:
        raise ValueError("github_read_credential_missing")
    for key in (
        "CHAT_LLM_PROVIDER",
        "CHAT_LLM_MODEL",
        "CHAT_LLM_ENDPOINT",
        "CHAT_LLM_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_API_KEY",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_CAPABILITIES",
    ):
        value = getattr(settings, key, None)
        if value is not None:
            values[key] = json.dumps(value) if isinstance(value, dict) else str(value)
    ledger = "v11-real-" + uuid4().hex
    values.update(
        {
            "V1_GITHUB_READ_TOKEN": credential,
            "V11_REAL_READS_ENABLED": "true",
            "V11_REAL_MODELS_ENABLED": "true",
            "V1_REAL_MODELS_ENABLED": "false",
            "V1_MODEL_BUDGET_CNY": "80",
            "V11_BUDGET_LEDGER_ID": ledger,
            "V11_BUDGET_CATEGORY": "smoke",
            "V1_SOURCE_OWNER_ID": "1",
            "LOCAL_MVP_ONLY": "true",
            "VALKEY_HOST": "",
            "LOG_LEVEL": "WARNING",
            "DEBUG": "false",
            "V1_CHAT_MODEL": "deepseek-flash",
        }
    )
    (descriptor.parent / "runtime.env").write_text(
        "\n".join(key + "=" + json.dumps(value) for key, value in values.items()) + "\n", encoding="utf-8"
    )
    save(
        descriptor.parent / "reconciliation.json",
        {
            "evidence_id": "v1-closeout-and-export-20261011",
            "evidence_hash": canonical_hash(report),
            "ledger_id": ledger,
            "total_charge": str(carry),
            "smoke_charge": str(carry),
            "unknown_charge": str(carry),
            "reconciled": True,
            "note": "Entire historical occupancy conservatively retained, including unknowns; unknown is a subset, not added twice.",
        },
    )
    Console().print({"compose_command": str(descriptor), "carryover": str(carry), "ledger_id": ledger})


async def bootstrap(private: Path) -> None:
    """Seed explicit local operator/reviewer identities in the empty isolated database."""
    connection = LocalPostgresStore()
    store = V11Store(connection)
    receipt = json.loads((private / "reconciliation.json").read_text(encoding="utf-8"))
    tokens = {}
    async with await connection.connect() as db, db.transaction():
        count = await (await db.execute('SELECT count(*) AS n FROM "user"')).fetchone()
        if not count or count["n"]:
            raise ValueError("empty_real_acceptance_database_required")
        for number in (1, 2, 3):
            session = "v11-local-operator-" + uuid4().hex
            await db.execute(
                'INSERT INTO "user"(id,email,hashed_password,created_at) VALUES(%s,%s,%s,now())',
                (number, f"v11-local-role-{number}@example.invalid", "no-password-login"),
            )
            await db.execute(
                "INSERT INTO session(id,user_id,name,created_at) VALUES(%s,%s,'',now())", (session, number)
            )
            tokens[str(number)] = jwt.encode(
                {"sub": session, "exp": datetime.now(UTC) + timedelta(hours=12)},
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )
        await db.execute(
            "INSERT INTO v11_budget_carryovers(evidence_id,evidence_hash,ledger_id,total_charge,smoke_charge,unknown_charge,reconciled) VALUES(%s,%s,%s,%s,%s,%s,%s)",
            tuple(
                receipt[key]
                for key in (
                    "evidence_id",
                    "evidence_hash",
                    "ledger_id",
                    "total_charge",
                    "smoke_charge",
                    "unknown_charge",
                    "reconciled",
                )
            ),
        )
    await store.bootstrap(1, ())
    await store.bootstrap_reviews(1)
    save(private / "sessions.json", tokens)
    actor = LocalActor(
        user_id=1,
        session_id=jwt.get_unverified_claims(tokens["1"])["sub"],
        expires_at=int((datetime.now(UTC) + timedelta(hours=11)).timestamp()),
    )
    Console().print(await store.real_budget_identity(actor))


async def discover(output: Path) -> None:
    """Read bounded public discovery once and retain its provenance before sampling."""
    result = {}
    source = ReviewGitHubSource()
    for repository in ("checkout", "setup-python"):
        identity = await source.identity(repository)
        values = {"identity": identity}
        for kind in ("pulls", "commits", "branches"):
            items, more = await source.discover(repository, identity, kind, 1, 20)
            values[kind] = {"items": items, "more": more}
        result[repository] = values
    save(output, result)
    Console().print({"saved": str(output), "repositories": list(result)})


async def collect(output: Path, input_file: Path | None = None) -> None:
    """Freeze bounded real observations without spending model budget or selecting by CI."""
    discovery = json.loads((ROOT / "evals/reports/v11_source_discovery.json").read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("source_evidence_exists")
    source = ReviewGitHubSource()
    result: dict = {"at": datetime.now(UTC).isoformat(), "provenance": "real_public_github", "positions": []}
    for repository, discovered in discovery.items():
        explicit = json.loads(input_file.read_text(encoding="utf-8")) if input_file else None
        if explicit and explicit["repository_id"] != repository:
            continue
        selectors = [{"kind": "pr", "pr_number": row["pr_number"]} for row in discovered["pulls"]["items"][:6]]
        selectors += [
            {
                "kind": "commit",
                "commit_sha": row["sha"],
                "parent_sha": row["parents"][0] if len(row["parents"]) > 1 else None,
            }
            for row in discovered["commits"]["items"][:4]
        ]
        selectors += [
            {"kind": "branch_compare", "base_branch": "main", "head_branch": row["name"]}
            for row in discovered["branches"]["items"][:3]
        ]
        if explicit:
            selectors = [explicit["selector"]]
        for selector in selectors:
            binding = ReviewBinding.model_validate({"repository_id": repository, "selector": selector})
            slot: dict = {"binding": binding.model_dump(mode="json")}
            try:
                facts = await source.collect(binding, discovered["identity"])
                slot.update(status="collected", facts=facts.model_dump(mode="json"))
                Console().print(
                    {
                        "repository": repository,
                        "selector": selector,
                        "coverage": facts.coverage,
                        "ci": facts.ci.status,
                        "strict_checks": facts.ci.checks_passed,
                    }
                )
            except DomainError as error:
                slot.update(status="blocked", reason=error.code)
                Console().print({"repository": repository, "selector": selector, "reason": error.code})
            result["positions"].append(slot)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def http_setup(private: Path, output: Path, port: int) -> None:
    """Explicit source grants and real rule embeddings use authenticated product HTTP."""
    if output.exists():
        raise ValueError("setup_evidence_exists")
    token = json.loads((private / "sessions.json").read_text(encoding="utf-8"))["1"]
    prefix = "/api/v1/releaseguard/reviews"
    result = []
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=120, headers={"Authorization": "Bearer " + token}
    ) as client:
        response = await client.get(prefix + "/runtime")
        response.raise_for_status()
        result.append({"runtime": response.json()})
        for repository in ("checkout", "setup-python"):
            response = await client.post(prefix + "/repositories/" + repository + "/verify")
            response.raise_for_status()
            result.append({"repository": repository, "verification": response.json()})
        response = await client.get(prefix + "/sources")
        response.raise_for_status()
        for source in response.json()["items"]:
            response = await client.post(
                prefix + "/sources/" + source["source_id"] + "/grant",
                json={
                    "command_id": "real-grant-" + source["source_id"],
                    "expected_version": source["version"],
                    "user_id": 2,
                },
            )
            response.raise_for_status()
            result.append({"source": source["source_id"], "grant": response.json()})
        for rule in ("application-policy", "checkout-reference"):
            response = await client.post(
                prefix + "/rules/" + rule + "/sync",
                json={
                    "command_id": "real-rules-" + rule,
                    "expected_object_version": 0,
                },
            )
            result.append({"rule": rule, "status_code": response.status_code, "result": response.json()})
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            response.raise_for_status()
            if response.json().get("status") != "completed":
                raise ValueError("real_rule_index_incomplete_no_retry")
    Console().print({"saved": str(output), "operations": len(result)})


async def smoke(private: Path, output: Path, port: int, input_file: Path | None = None) -> None:
    """One explicit real PR analysis; a saved result is never silently replaced."""
    if output.exists():
        raise ValueError("smoke_evidence_exists")
    tokens = json.loads((private / "sessions.json").read_text(encoding="utf-8"))
    prefix = "/api/v1/releaseguard/reviews"
    at = datetime.now(UTC)
    nonce = canonical_hash(output.name)[:24]
    selected = (
        json.loads(input_file.read_text(encoding="utf-8"))
        if input_file
        else {
            "binding": {"repository_id": "checkout", "selector": {"kind": "pr", "pr_number": 2590}},
            "query": "请解释实际 diff 中依赖版本与完整性字段的变化，引用读到的行。区分事实与推断，说明尚未验证的运行影响；不要把只读分析当成审批通过。",
        }
    )
    report: dict = {
        "at": at.isoformat(),
        "category": settings.V11_BUDGET_CATEGORY,
        "intent": selected,
        "steps": [],
        "completed": False,
    }
    save(output, report)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        trust_env=False,
        timeout=120,
        headers={"Authorization": "Bearer " + tokens["1"]},
    ) as client:

        async def call(method: str, path: str, body: dict | None = None) -> dict:
            started = time.perf_counter()
            response = await client.request(method, prefix + path, json=body)
            result = response.json()
            report["steps"].append(
                {
                    "path": path,
                    "http_status": response.status_code,
                    "body": result,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            response.raise_for_status()
            return result

        created = await call(
            "POST",
            "",
            {
                "idempotency_key": "real-smoke-" + nonce,
                "approver_user_id": 2,
                "binding": selected["binding"],
                "change": {
                    "requester_id": "local-user:1",
                    "service": "releaseguard-acceptance",
                    "environment": "development",
                    "title": "Real readonly review capability check",
                    "description": "Explain the selected object's actual diff; no target writes",
                    "change_window": {
                        "start": at.isoformat(),
                        "end": (at + timedelta(hours=1)).isoformat(),
                        "timezone": "UTC",
                    },
                    "rollback_plan": "No target writes; preserve prior immutable records",
                    "verification_plan": "Read actual diff and structured report",
                },
            },
        )
        identifier = created["object_id"]
        synced = await call(
            "POST",
            "/" + identifier + "/sync",
            {
                "command_id": "smoke-sync-" + nonce,
                "expected_object_version": created["object_version"],
            },
        )
        if synced["status"] != "completed":
            return
        current = await call("GET", "/" + identifier)
        analyzed = await call(
            "POST",
            "/" + identifier + "/analyze",
            {
                "command_id": "smoke-analysis-" + nonce,
                "expected_object_version": current["object_version"],
                "expected_state_version": current["state"]["state_version"],
                "snapshot_id": synced["snapshot_id"],
                "generation_id": synced["generation_id"],
                "query": selected["query"],
            },
        )
        if analyzed["status"] == "completed":
            await call("GET", "/" + identifier + "/analyses/" + analyzed["analysis_id"])
            report["completed"] = True
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Console().print({"saved": str(output), "completed": report["completed"]})


async def authorize(private: Path, output: Path, port: int, reference: str) -> None:
    """Materialize the user's explicit real acceptance approval and verify the sole live ledger."""
    for project in HISTORICAL:
        info = json.loads(subprocess.check_output(["docker", "inspect", project + "-app-1"]))[0]
        if info["State"]["Running"]:
            raise ValueError("historical_paid_api_still_running")
    dataset = Dataset.model_validate_json((ROOT / "evals/datasets/v11_acceptance.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "evals/reports/v11_source_candidates.json").read_text(encoding="utf-8"))
    if dataset.source_evidence_hash != canonical_hash(source):
        raise ValueError("source_evidence_hash_mismatch")
    tokens = json.loads((private / "sessions.json").read_text(encoding="utf-8"))
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        trust_env=False,
        timeout=10,
        headers={"Authorization": "Bearer " + tokens["1"]},
    ) as client:
        response = await client.get("/api/v1/releaseguard/reviews/runtime")
        response.raise_for_status()
        runtime = response.json()
    now = datetime.now(UTC)
    async with await LocalPostgresStore().connect() as db, db.transaction():
        await db.execute("SET TRANSACTION READ ONLY")
        charges = await (
            await db.execute(
                "SELECT c.budget_category,sum(b.charge) AS charge,"
                "count(*) FILTER(WHERE b.attempt='analysis-1') AS model_calls "
                "FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id) GROUP BY c.budget_category"
            )
        ).fetchall()
    categories = {row["budget_category"]: row for row in charges}
    remaining = Decimal("25") - categories.get("acceptance", {}).get("charge", Decimal("0"))
    auth = Authorization(
        authorization_reference=reference,
        authorized_at=now,
        expires_at=now + timedelta(hours=8),
        public_github_reads=True,
        model_calls=True,
        max_analysis_calls=80,
        dataset_hash=canonical_hash(dataset),
        source_fingerprint=code_state()["source_hash"],
        designated_ledger_id=runtime["ledger_id"],
        reconciled_budget_evidence_hash=runtime["budget_evidence_hash"],
        price_evidence_hash=canonical_hash(
            {"expiry": PRICE_EXPIRY, "prices": {key: [str(p[0]), str(p[1]), p[2]] for key, p in PRICES.items()}}
        ),
        api_origin=f"http://127.0.0.1:{port}",
        database_name="v11_isolated",
        database_port=settings.POSTGRES_PORT,
        owner_user_id=1,
        approver_user_id=2,
        session_token=tokens["1"],
        budget_remaining_cny=float(remaining),
        rules_ready=True,
    )
    auth.validate_preflight(dataset)
    value = auth.model_dump(mode="json", exclude={"session_token"})
    save(
        output,
        {
            **value,
            "source_evidence_hash": dataset.source_evidence_hash,
            "runtime": runtime,
            "historical_paid_apis_paused": list(HISTORICAL),
            "category_ledger": charges,
            "development_model_calls": categories.get("development", {}).get("model_calls", 0),
            "smoke_model_calls": categories.get("smoke", {}).get("model_calls", 0),
            "ordinary_quality_and_latency_risks": "unnecessary UNKNOWN and full-chain latency remain; thresholds unchanged",
        },
    )
    save(private / "authorization.json", {**value, "session_token": tokens["1"]})
    Console().print({"freeze": str(output), "source_fingerprint": auth.source_fingerprint})


async def audit(output: Path) -> None:
    """Aggregate the designated ledger without releasing unresolved charges."""
    connection = LocalPostgresStore()
    async with await connection.connect() as db, db.transaction():
        await db.execute("SET TRANSACTION READ ONLY")
        carry = await (
            await db.execute(
                "SELECT evidence_id,evidence_hash,ledger_id,total_charge,smoke_charge,unknown_charge,reconciled FROM v11_budget_carryovers"
            )
        ).fetchall()
        budget = await (
            await db.execute(
                "SELECT c.budget_category,count(*) AS attempts,sum(b.charge) AS charge,sum(b.reserved) AS reserved,"
                "count(*) FILTER(WHERE b.attempt='analysis-1') AS model_attempts "
                "FROM v1_budget b JOIN v1_commands c USING(owner_id,command_id) GROUP BY c.budget_category ORDER BY c.budget_category"
            )
        ).fetchall()
        commands = await (
            await db.execute("SELECT status,count(*) AS count FROM v1_commands GROUP BY status ORDER BY status")
        ).fetchall()
        analyses = await (
            await db.execute("SELECT status,count(*) AS count FROM v11_analyses GROUP BY status ORDER BY status")
        ).fetchall()
        objects = await (await db.execute("SELECT count(*) AS count FROM v11_objects")).fetchone()
    save(
        output,
        {
            "at": datetime.now(UTC).isoformat(),
            "carryovers": carry,
            "categories": budget,
            "commands": commands,
            "analyses": analyses,
            "objects": objects,
            "cumulative_conservative_cny": sum(row["total_charge"] for row in carry)
            + sum(row["charge"] for row in budget),
            "actual_provider_invoice": "not yet exported for this run",
            "unknown_released": False,
        },
    )
    Console().print({"saved": str(output), "budget": budget, "analyses": analyses})


async def product_checks(private: Path, output: Path, port: int, input_file: Path) -> None:
    """Use saved real reports for role isolation and deterministic denial, without another model call."""
    if output.exists():
        raise ValueError("product_evidence_exists")
    source = json.loads(input_file.read_text(encoding="utf-8"))
    positions = source.get("positions", [])
    if "steps" in source and source["completed"]:
        positions = [{"analysis": source["steps"][-1]["body"], "object_id": source["steps"][-1]["body"]["object_id"]}]
    tokens = json.loads((private / "sessions.json").read_text(encoding="utf-8"))
    checks = []
    seen = set()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=90) as client:

        async def call(method: str, path: str, actor: str = "1", body: dict | None = None) -> httpx.Response:
            return await client.request(
                method,
                "/api/v1/releaseguard/reviews" + path,
                json=body,
                headers={"Authorization": "Bearer " + tokens[actor]},
            )

        for position in positions:
            analysis = position.get("analysis")
            if not analysis:
                continue
            identifier = position["object_id"]
            response = await call("GET", "/" + identifier)
            response.raise_for_status()
            current = response.json()
            key = (current["binding"]["repository_id"], current["binding"]["selector"]["kind"])
            if key in seen:
                continue
            seen.add(key)
            for actor, expected in (("2", 200), ("3", 404)):
                read = await call("GET", "/" + identifier + "/analyses/" + analysis["analysis_id"], actor)
                checks.append(
                    {
                        "case": str(key) + "_report_role_" + actor,
                        "http_status": read.status_code,
                        "passed": read.status_code == expected,
                    }
                )
            if not current["rehearsal_eligibility"]["eligible"]:
                prepared = await call(
                    "POST",
                    "/" + identifier + "/prepare",
                    body={
                        "command_id": "real-deny-" + canonical_hash([output.name, identifier])[:28],
                        "expected_object_version": current["object_version"],
                        "expected_state_version": current["state"]["state_version"],
                        "analysis_id": analysis["analysis_id"],
                    },
                )
                receipt = await call("GET", "/" + identifier + "/receipt")
                checks.append(
                    {
                        "case": str(key) + "_prepare_denied",
                        "result": prepared.json(),
                        "eligibility": current["rehearsal_eligibility"],
                        "receipt": receipt.json(),
                        "passed": prepared.status_code == 200
                        and prepared.json().get("status") == "blocked"
                        and receipt.json().get("target_write_calls") == 0
                        and receipt.json().get("receipt") is None,
                    }
                )
            else:
                nonce = canonical_hash([output.name, identifier])[:28]
                prepared = await call(
                    "POST",
                    "/" + identifier + "/prepare",
                    body={
                        "command_id": "real-prepare-" + nonce,
                        "expected_object_version": current["object_version"],
                        "expected_state_version": current["state"]["state_version"],
                        "analysis_id": analysis["analysis_id"],
                    },
                )
                prepared.raise_for_status()
                result = prepared.json()
                checks.append({"case": "real_pr_prepare", "result": result, "passed": result["status"] == "completed"})
                if result["status"] == "completed":
                    approval = result["result"]["approval"]["request"]
                    current = (await call("GET", "/" + identifier)).json()
                    approved = await call(
                        "POST",
                        "/" + identifier + "/approval",
                        "2",
                        {
                            "command_id": "real-approve-" + nonce,
                            "expected_object_version": current["object_version"],
                            "expected_state_version": current["state"]["state_version"],
                            "approval_request_id": approval["approval_request_id"],
                            "authorization_hash": approval["authorization_hash"],
                            "decision": "APPROVED",
                            "reason_code": "authorized_local_dry_run_review",
                        },
                    )
                    checks.append(
                        {
                            "case": "different_subject_approval",
                            "result": approved.json(),
                            "passed": approved.status_code == 200 and approved.json().get("status") == "completed",
                        }
                    )
                    if checks[-1]["passed"]:
                        current = (await call("GET", "/" + identifier)).json()
                        ran = await call(
                            "POST",
                            "/" + identifier + "/run",
                            body={
                                "command_id": "real-run-" + nonce,
                                "expected_object_version": current["object_version"],
                                "expected_state_version": current["state"]["state_version"],
                            },
                        )
                        receipt = await call("GET", "/" + identifier + "/receipt")
                        checks.append(
                            {
                                "case": "real_pr_dry_run",
                                "result": ran.json(),
                                "receipt": receipt.json(),
                                "passed": ran.status_code == 200
                                and ran.json().get("status") == "completed"
                                and receipt.json().get("target_write_calls") == 0
                                and receipt.json().get("receipt") is not None,
                            }
                        )
    save(
        output,
        {
            "provenance": "real_sources_provider_reports_http_pg",
            "checks": checks,
            "passed": bool(checks) and all(check["passed"] for check in checks),
            "model_calls": 0,
            "positive_pr_rehearsal": "passed"
            if any(c["case"] == "real_pr_dry_run" and c["passed"] for c in checks)
            else "unverified_no_current_eligible_report",
        },
    )
    Console().print({"saved": str(output), "checks": len(checks)})


def main() -> None:
    """Parent launch configures child environment before application modules initialize."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "inventory",
            "init",
            "bootstrap",
            "serve",
            "discover",
            "collect",
            "http-setup",
            "smoke",
            "authorize",
            "audit",
            "product-checks",
            "launch",
        ],
    )
    parser.add_argument("--export", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--token-env", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--compose-command", type=Path)
    parser.add_argument("--port", type=int, default=35452)
    parser.add_argument(
        "--authorization-reference",
        default="User instruction 2026-10-11: 那就完成真实层验收吧; approved V1.1 backend spec T10",
    )
    args, remaining = parser.parse_known_args()
    if remaining and args.action != "launch":
        parser.error("unexpected_arguments")
    if args.action == "inventory":
        inventory(args.export, args.report)
        return
    if args.action == "init":
        setup(args.token_env, args.report, args.port)
        return
    _, values = configuration(args.compose_command, offline=False)
    if args.action == "launch":
        command = remaining[1:] if remaining[:1] == ["--"] else remaining
        result = subprocess.run([sys.executable, *command], env={**os.environ, **values}, cwd=ROOT, check=False)
        raise SystemExit(result.returncode)
    if any(str(getattr(settings, key)) != values[key] for key in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB")):
        raise ValueError("launch_with_isolated_environment_required")
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        if args.action == "bootstrap":
            runner.run(bootstrap(args.compose_command.parent))
        elif args.action == "discover":
            runner.run(discover(args.report))
        elif args.action == "collect":
            runner.run(collect(args.report, args.input))
        elif args.action == "http-setup":
            runner.run(http_setup(args.compose_command.parent, args.report, args.port))
        elif args.action == "smoke":
            runner.run(smoke(args.compose_command.parent, args.report, args.port, args.input))
        elif args.action == "authorize":
            runner.run(authorize(args.compose_command.parent, args.report, args.port, args.authorization_reference))
        elif args.action == "audit":
            runner.run(audit(args.report))
        elif args.action == "product-checks":
            runner.run(product_checks(args.compose_command.parent, args.report, args.port, args.input))
        else:
            server = uvicorn.Server(
                uvicorn.Config("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")
            )
            runner.run(server.serve())


if __name__ == "__main__":
    main()
