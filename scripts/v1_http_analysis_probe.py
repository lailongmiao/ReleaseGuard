"""Resumable actual HTTP timings with PG attempt evidence, never blind resend."""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from time import perf_counter, sleep

from rich.console import Console
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.services.local_mvp_storage import LocalPostgresStore
from app.services.releaseguard.normalization import canonical_hash
from app.services.v1_sources import corpus_manifest
from evals.v1_backend.enhancement import AnalysisDataset
from evals.v1_backend.provenance import code_state
from scripts.mvp_product_probe import ProductProbe
from scripts.v1_provider_probe import reuse_sessions
from app.schemas.v1_backend import Chunk


@retry(
    retry=retry_if_exception_type(PermissionError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
    reraise=True,
)
def save(path: Path, report: dict) -> None:
    """Persist before and after every request with an atomic same-dir replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(report: dict) -> None:
    """Include observed failures; incomplete or in-flight samples cannot pass."""
    rows = report["rows"]
    measured = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    values = sorted(measured)
    report["requests"] = len(rows)
    report["actual_model_samples"] = sum(r.get("actual_model_sample", False) for r in rows)
    report["model_attempts_including_failures"] = sum(r.get("model_attempts", 0) for r in rows)
    slots = {(r["case_id"], r["attempt"]) for r in rows if r.get("verified_provider_attempt", False)}
    complete = len(slots) == report["planned"] and len(measured) == len(rows)
    report["p95_seconds_including_failures"] = values[ceil(len(values) * 0.95) - 1] if values else None
    report["languages"] = {}
    for language in ("zh", "en"):
        group = [r for r in rows if r.get("language", "en") == language]
        times = sorted(r["latency_seconds"] for r in group if r.get("latency_seconds") is not None)
        report["languages"][language] = {
            "requests": len(group),
            "measured": len(times),
            "p95_seconds_including_failures": times[ceil(len(times) * 0.95) - 1] if times else None,
        }
    p95 = report["p95_seconds_including_failures"]
    report["status"] = (
        "passed" if complete and p95 is not None and p95 <= 30 else "failed" if complete else "unverified"
    )


def load(path: Path, dataset: dict, state: dict, repeat: int = 2, pr: int = 2557) -> dict:
    """Reject stale datasets/code and duplicate paid slots on continuation."""
    if not path.exists():
        return {
            "schema": "v1-http-analysis-performance.2",
            "code": state,
            "dataset_hash": canonical_hash(dataset),
            "scope": "actual_http_source_gate_query_embedding_model_commit; excludes_initial_indexing",
            "planned": len(dataset["cases"]) * repeat,
            "repeat": repeat,
            "pr": pr,
            "rows": [],
            "quota_windows": [],
            "threshold_seconds": 30,
            "actual_cost_cny": "unknown",
            "target_write_calls": 0,
            "production_execution": "disabled",
            "status": "unverified",
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "v1-http-analysis-performance.2"
        or report["dataset_hash"] != canonical_hash(dataset)
        or report["code"]["source_hash"] != state["source_hash"]
        or report.get("repeat", 2) != repeat
        or report.get("pr", 2557) != pr
    ):
        raise ValueError("performance_resume_binding_mismatch")
    slots = [(r["case_id"], r["attempt"]) for r in report["rows"] if r.get("model_attempts", 0)]
    if len(slots) != len(set(slots)):
        raise ValueError("duplicate_paid_performance_slot")
    return report


@retry(
    retry=retry_if_exception_type((subprocess.CalledProcessError, subprocess.TimeoutExpired)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def quota(probe: ProductProbe) -> dict:
    """Read quota from the API network/credential without exposing the token."""
    code = (
        "import json,httpx; from app.core.config import settings; "
        "t=settings.V1_GITHUB_READ_TOKEN.get_secret_value(); "
        "h={'Authorization':'Bearer '+t} if t else {}; "
        "r=httpx.get('https://api.github.com/rate_limit',headers=h,trust_env=False,timeout=15); "
        "c=r.json().get('resources',{}).get('core',{}); "
        "print(json.dumps({'status':r.status_code,'remaining':c.get('remaining'),'reset':c.get('reset'),'authenticated':bool(t)}))"
    )
    result = subprocess.run(
        [*probe.args, "exec", "-T", "app", "/app/.venv/bin/python", "-c", code],
        capture_output=True,
        text=True,
        timeout=25,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


async def attempts(probe: ProductProbe, command: str) -> dict:
    """Persisted reservation identifies an actual Provider attempt, even on loss."""
    async with await LocalPostgresStore().connect() as db:
        row = await (
            await db.execute(
                "SELECT count(*) AS attempts,COALESCE(sum(charge),0) AS charge,count(*) FILTER (WHERE usage->>'model'='deepseek-flash') AS verified FROM v1_budget WHERE owner_id=%s AND command_id=%s AND attempt='analysis-1'",
                (probe.identities[1]["user_id"], command),
            )
        ).fetchone()
        command_row = await (
            await db.execute(
                "SELECT status FROM v1_commands WHERE owner_id=%s AND command_id=%s",
                (probe.identities[1]["user_id"], command),
            )
        ).fetchone()
    return {
        "model_attempts": row["attempts"],
        "conservative_charge_cny": str(row["charge"]),
        "verified_provider_attempt": row["verified"] == 1,
        "command_status": command_row["status"] if command_row else "not_claimed",
    }


async def corpus() -> dict:
    """Freeze active generation versions/digests without reading source text or ACL."""
    async with await LocalPostgresStore().connect() as db:
        rows = await (
            await db.execute(
                "SELECT s.source_id,s.version,s.generation,g.config->'fingerprint' AS fingerprint FROM v1_sources s JOIN v1_generations g ON g.generation=s.generation ORDER BY s.source_id"
            )
        ).fetchall()
    return {
        r["source_id"]: {
            "version": r["version"],
            "generation": r["generation"],
            "fingerprint_hash": canonical_hash(r["fingerprint"]),
        }
        for r in rows
    }


def run(probe: ProductProbe, report: dict, path: Path, batch: int, runner: asyncio.Runner, dataset: dict) -> None:
    """Keep rejected/unknown attempts and stop safely across quota windows."""
    if any(r.get("status") == "in_flight" for r in report["rows"]):
        report["stop_reason"] = "interrupted_http_request_diagnostic_hold_no_resend"
        return
    processed = {(r["case_id"], r["attempt"]) for r in report["rows"]}
    pending = [
        (c, a)
        for c in dataset["cases"]
        for a in range(1, report.get("repeat", 2) + 1)
        if (c["case_id"], a) not in processed
    ]
    report["stop_reason"] = None
    for case, attempt in pending[:batch]:
        capacity = quota(probe)
        report["quota_windows"].append({"at": datetime.now(UTC).isoformat(), **capacity})
        if capacity["status"] != 200 or not isinstance(capacity["remaining"], int) or capacity["remaining"] < 6:
            report["stop_reason"] = "github_quota_window_wait"
            break
        body = probe.body("http-latency-" + case["case_id"] + "-" + str(attempt))
        body["product"] = {"pr_number": report.get("pr", 2557)}
        view = probe.request("POST", "/api/v1/releaseguard/changes", body)
        endpoint = "/api/v1/releaseguard/changes/" + view["state"]["change_id"]
        command = probe.nonce + ":http:" + case["case_id"] + ":" + str(attempt)
        row = {
            "case_id": case["case_id"],
            "attempt": attempt,
            "change_id": view["state"]["change_id"],
            "command_id": command,
            "status": "in_flight",
            "latency_seconds": None,
            "model_attempts": 0,
            "expected": case["expected"],
            "query": case["query"],
            "language": case.get("language", "en"),
        }
        report["rows"].append(row)
        save(path, report)
        started = perf_counter()
        result: dict = {}
        try:
            result = probe.request(
                "POST",
                endpoint + "/analyze",
                {
                    "command_id": command,
                    "expected_state_version": view["state"]["state_version"],
                    "query": case["query"],
                },
            )
            row.update(status=result["status"], reason=result.get("reason"))
        except Exception as error:
            row.update(status="transport_unknown", reason=type(error).__name__)
        finally:
            row["latency_seconds"] = round(perf_counter() - started, 4)
            row.update(runner.run(attempts(probe, command)))
            row["actual_model_sample"] = result.get("status") == "completed" and row["verified_provider_attempt"]
            if result.get("status") == "completed":
                saved = probe.request("GET", endpoint + "/analysis")
                chunks = [Chunk.model_validate(c) for c in saved["citations"]]
                output = saved["analysis"]
                valid = all(
                    c.text == c.document.content[c.start_char : c.end_char]
                    and canonical_hash(c.text) == c.text_hash
                    and canonical_hash(c.document.content) == c.document.content_hash
                    for c in chunks
                )
                ids = {c.chunk_id for c in chunks}
                valid = valid and all(set(c["citation_ids"]) <= ids for c in output["claims"])
                row.update(
                    output=output,
                    review_sources=saved["citations"],
                    telemetry=saved["telemetry"],
                    citation_correct=valid,
                    analysis_id=saved["analysis_id"],
                )
            summarize(report)
            save(path, report)
        Console().print({"sample": len(report["rows"]), "status": row["status"], "seconds": row["latency_seconds"]})
        if row["status"] == "transport_unknown" or row["command_status"] == "started":
            report["stop_reason"] = "unknown_command_diagnostic_hold_no_resend"
            break
        if row["model_attempts"] != 1:
            report["stop_reason"] = row.get("reason") or "no_persisted_model_attempt"
            break
        sleep(max(0, 6.2 - row["latency_seconds"]))


def main() -> int:
    """Existing approved isolated DB/budget only; resume never repeats paid work."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:34080")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/v1_model_holdout.json"))
    parser.add_argument("--repeat", type=int, default=2, choices=range(1, 4))
    parser.add_argument("--pr", type=int, default=2557)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--batch", type=int, default=40, choices=range(1, 161))
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = dataset.get("cases", [])
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        parser.error("empty_or_duplicate_analysis_cases")
    if str(dataset.get("version", "")).startswith("v1-enhancement-analysis."):
        frozen = AnalysisDataset.model_validate(dataset)
        if frozen.corpus_manifest_hash != canonical_hash(corpus_manifest()):
            parser.error("frozen_corpus_mismatch")
        if args.repeat != 2:
            parser.error("enhancement_holdout_requires_two_repetitions")
    if dataset.get("required_probe_pr", args.pr) != args.pr:
        parser.error("frozen_dataset_pr_precondition_mismatch")
    report = load(args.report, dataset, code_state(), args.repeat, args.pr)
    probe = ProductProbe(args.url, args.compose_command, "v1_http_analysis")
    probe.client.timeout = 180
    try:
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            runner.run(reuse_sessions(probe, args.compose_command))
            current = runner.run(corpus())
            if report.get("corpus") is not None and report["corpus"] != current:
                raise ValueError("performance_corpus_changed")
            report["corpus"] = current
            run(probe, report, args.report, args.batch, runner, dataset)
    except Exception as error:
        report.update(failure=type(error).__name__)
    finally:
        probe.client.close()
        summarize(report)
        save(args.report, report)
        if args.quality_report:
            quality = {
                "schema": "v1-http-grounded-quality.1",
                "dataset_hash": report["dataset_hash"],
                "code": report["code"],
                "planned": report["planned"],
                "corpus": report.get("corpus"),
                "evidence": "actual_http_retrieved_context",
                "rows": [
                    {**r, "status": "passed" if r.get("actual_model_sample") else "failed"} for r in report["rows"]
                ],
            }
            save(args.quality_report, quality)
    Console().print({"status": report["status"], "requests": report["requests"], "reason": report.get("stop_reason")})
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
