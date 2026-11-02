"""Rich offline evaluation CLI; live commands need explicit reviewed scope files."""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from rich.console import Console

from app.core.config import settings
from app.services.llm.contracts import LLMRole
from app.services.llm.registry import LLMRegistry
from app.services.releaseguard.normalization import canonical_hash
from evals.releaseguard.dataset import load_dataset
from evals.releaseguard.judge import (
    JudgeAuthorization,
    JudgeTraceAuthorization,
    ServiceJudge,
    evaluate_gold,
    evaluate_judge,
    load_gold,
    mask_judge_trace,
)
from evals.releaseguard.langfuse_io import LangfuseIO, LangfuseScope, SDKLangfuseTransport, ScoreIntent
from evals.releaseguard.reports import Baseline, EvaluationReport, build_report, compare_baseline, save_report
from evals.releaseguard.schemas import RunConfig, Status

console = Console()


def parser() -> argparse.ArgumentParser:
    """No live defaults, no automatic baseline refresh or implicit write flag."""
    result = argparse.ArgumentParser(description="ReleaseGuard agent-specific evaluation")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", type=Path, default=Path("evals/datasets/releaseguard_v1.json"))
    run.add_argument("--mode", choices=["offline"], default="offline")
    run.add_argument("--judge", choices=["off", "fake"], default="off")
    run.add_argument("--langfuse", choices=["off"], default="off")
    run.add_argument("--repeat", type=int, default=2)
    run.add_argument("--report", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("--report", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    for command in ("probe-judge", "sync-langfuse"):
        live = commands.add_parser(command)
        live.add_argument("--environment", choices=["authorized-test"], required=True)
        live.add_argument("--authorization-ref", required=True)
        live.add_argument("--authorization-file", type=Path, required=True)
        if command == "probe-judge":
            live.add_argument("--dataset", type=Path, default=Path("evals/datasets/judge_gold_v1.json"))
            live.add_argument("--parse-mode", choices=["structured", "strict_json"], default="structured")
            live.add_argument("--report", type=Path)
            live.add_argument("--trace-authorization-file", type=Path, required=True)
        else:
            live.add_argument("--report", type=Path, required=True)
            live.add_argument("--write-scores", action="store_true")
    return result


async def dispatch(args: argparse.Namespace) -> int:
    """Exit 0 selected layer passed, 1 failed, 2 unverified/incomparable."""
    if args.command == "run":
        dataset = load_dataset(args.dataset)
        report = await build_report(dataset, RunConfig(judge=args.judge, repeat=args.repeat))
        if args.report:
            await asyncio.to_thread(save_report, report, args.report)
        console.print(report.summary.model_dump(mode="json"))
        console.print({"real_judge": "unverified", "langfuse": "unverified", "production": "disabled"})
        return exit_status(report.summary.status.value)
    if args.command == "compare":
        report = EvaluationReport.model_validate_json(args.report.read_text(encoding="utf-8"))
        baseline = Baseline.model_validate_json(args.baseline.read_text(encoding="utf-8"))
        comparison = compare_baseline(report, baseline)
        console.print(comparison.model_dump(mode="json"))
        return exit_status(comparison.status)
    if args.command == "probe-judge":
        authorization = JudgeAuthorization.model_validate_json(args.authorization_file.read_text(encoding="utf-8"))
        if authorization.authorization_ref != args.authorization_ref:
            raise PermissionError("judge_scope_denied")
        gold = load_gold(args.dataset)
        calls = len(gold.cases) * 2 + 1
        attempts = calls * (LLMRegistry.get_config("", LLMRole.JUDGE).max_retries + 1) ** 2
        if (
            authorization.dataset_hash != gold.dataset_hash
            or authorization.max_calls < calls
            or authorization.max_provider_attempts < attempts
        ):
            raise PermissionError("judge_budget_or_dataset_denied")
        tracing = JudgeTraceAuthorization.model_validate_json(
            args.trace_authorization_file.read_text(encoding="utf-8")
        )
        if (
            str(tracing.host).rstrip("/") != settings.LANGFUSE_HOST.rstrip("/")
            or tracing.dataset_hash != gold.dataset_hash
            or datetime.now(timezone.utc) >= tracing.expires_at
        ):
            raise PermissionError("judge_trace_scope_denied")
        target = args.report.resolve() if args.report else None
        allowed = (Path(__file__).resolve().parents[2] / "evals/reports").resolve()
        if target and (not target.is_relative_to(allowed) or target.suffix != ".json"):
            raise ValueError("report_path_denied")
        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            timeout=10,
            mask=mask_judge_trace,
        )
        try:
            # The SDK caches resources by public key. Never reuse an unmasked
            # instance or silently replace another workflow's mask.
            if getattr(client, "_mask", None) is not mask_judge_trace:
                raise PermissionError("judge_trace_mask_denied")
            if await SDKLangfuseTransport(client).workspaces() != {tracing.workspace_id}:
                raise PermissionError("judge_trace_workspace_denied")
            callback = CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY, update_trace=True)
            transport = ServiceJudge(authorization, callback, trace_authorization=tracing, parse_mode=args.parse_mode)
            probe = await evaluate_judge(gold.cases[0].request, transport)
            if probe.status is not Status.PASSED:
                console.print({"probe_status": probe.status.value, "reason_code": probe.reason_code})
                return 2
            result = await evaluate_gold(gold, transport)
            result["probe"] = probe.model_dump(mode="json")
            result["trace_authorization_ref"] = tracing.authorization_ref
            result["eval_run_id"] = tracing.eval_run_id
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(target.write_text, json.dumps(result, indent=2), encoding="utf-8")
            console.print({k: v for k, v in result.items() if k not in {"results", "probe"}})
            return exit_status(result["quality_status"])
        finally:
            await asyncio.to_thread(client.shutdown)
    scope = LangfuseScope.model_validate_json(args.authorization_file.read_text(encoding="utf-8"))
    expected_ref = scope.write_authorization_ref if args.write_scores else scope.authorization_ref
    report = EvaluationReport.model_validate_json(args.report.read_text(encoding="utf-8"))
    if (
        expected_ref != args.authorization_ref
        or (scope.eval_run_id, scope.dataset_hash) != (report.eval_run_id, report.dataset.dataset_hash)
        or datetime.now(timezone.utc) >= scope.expires_at
    ):
        raise PermissionError("langfuse_scope_denied")
    client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        timeout=10,
    )
    io = LangfuseIO(SDKLangfuseTransport(client), scope)
    try:
        read = await io.read(at=datetime.now(timezone.utc))
        console.print(read.model_dump(mode="json"))
        if read.status is not Status.PASSED or not args.write_scores:
            return exit_status(read.status.value)
        statuses = []
        for trace in read.traces:
            result = next(
                (r for r in report.results if (r.case_id, r.attempt) == (trace.case_id, trace.attempt)), None
            )
            if result is None:
                return 2
            for metric in result.metrics:
                if metric.status is not Status.PASSED:
                    statuses.append(Status.UNVERIFIED)
                    continue
                outcome = await io.publish(
                    ScoreIntent(
                        trace_id=trace.trace_id,
                        eval_run_id=report.eval_run_id,
                        case_id=trace.case_id,
                        attempt=trace.attempt,
                        dataset_hash=scope.dataset_hash,
                        metric_id=metric.metric_id,
                        metric_version=metric.version,
                        prompt_hash=canonical_hash(metric.version),
                        model="deterministic-rules.v1",
                        value=1.0,
                        provenance="offline_rules",
                    ),
                    at=datetime.now(timezone.utc),
                )
                statuses.append(outcome.status)
        console.print(
            {
                "confirmed": statuses.count(Status.PASSED),
                "planned": len(statuses),
                "real_judge_trace_acceptance": "unverified",
            }
        )
        return 0 if statuses and all(s is Status.PASSED for s in statuses) else 2
    finally:
        await asyncio.to_thread(client.shutdown)


def exit_status(status: str) -> int:
    """Unknown required validation retains an explicit nonzero status."""
    return 0 if status == "passed" else 1 if status == "failed" else 2


def main() -> int:
    """Errors print only safe classes, never sensitive exception messages."""
    args = parser().parse_args()
    try:
        return asyncio.run(dispatch(args))
    except (ValueError, PermissionError, OSError):
        console.print({"status": "unverified", "reason_code": "invalid_input_or_scope"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
