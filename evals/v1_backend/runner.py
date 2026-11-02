"""Strict versioned RAG/HTTP/Graph cases; safety gates never use a Judge."""

import argparse
import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.console import Console

from evals.v1_backend.fixtures import body, call, environment, transport
from evals.releaseguard.schemas import CaseSpec, Status
from evals.releaseguard.runner import run_case as legacy_run_case
from evals.releaseguard.metrics import evaluate_case
from app.services.releaseguard.normalization import canonical_hash
from app.services.v1_rag import fuse
from app.services.v1_sources import GitHubSource
from app.schemas.v1_backend import Collection
from evals.v1_backend.provenance import code_state

Scenario = Literal[
    "normal",
    "risk_low",
    "risk_medium",
    "risk_high",
    "unknown",
    "ci_failed",
    "missing",
    "expired",
    "conflicting",
    "permission_denied",
    "model_invalid",
    "model_timeout",
    "citation_invalid",
    "replay",
    "concurrent",
    "revoke",
    "binding_drift",
    "approval_denied",
    "approval_revoke",
    "sandbox_denied",
    "partial_recovery",
    "compensation_failed",
    "false_success",
    "orphan_after_claim",
    "approval_expired",
    "drift_evidence",
]


class Case(BaseModel):
    """Gold is frozen before execution, not inferred from observed outcomes."""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^V1-[0-9]{3}$")
    scenario: Scenario
    query: str = Field(min_length=1, max_length=2000)
    gold: list[str] = Field(max_length=5)
    kind: Literal["backend", "legacy_sandbox_regression"] = "backend"
    legacy_case: str | None = None
    initial_value: Literal[0] = 0
    clock: Literal["2026-10-09T12:00:00Z", "2026-01-01T00:00:00Z"]
    expected_writes: int = Field(default=0, ge=0, le=4)
    expected_analysis_calls: int = Field(ge=0, le=1)


class Dataset(BaseModel):
    """Reject duplicates, undersized corpus, altered hashes and unknown fields."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["v1-rag-backend.2"]
    cases: list[Case] = Field(min_length=60)
    dataset_hash: str
    thresholds: dict

    @model_validator(mode="after")
    def frozen(self) -> "Dataset":
        """The frozen identity covers every oracle and threshold."""
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate_case")
        if any(
            c.kind == "backend"
            and (not c.gold or c.legacy_case or c.expected_writes)
            or c.kind == "legacy_sandbox_regression"
            and (not c.legacy_case or c.gold)
            for c in self.cases
        ):
            raise ValueError("case_provenance_or_gold_invalid")
        values = self.model_dump(exclude={"dataset_hash"}, exclude_unset=True)
        if canonical_hash(values) != self.dataset_hash:
            raise ValueError("dataset_hash_changed")
        return self


async def run_case(case: Case) -> dict:
    """Use actual service/Graph transitions, injected external faults are labelled."""
    if case.kind == "legacy_sandbox_regression":
        legacy = json.loads(Path("evals/datasets/releaseguard_v1.json").read_text(encoding="utf-8"))
        matches = [c for c in legacy["cases"] if c["case_id"] == case.legacy_case]
        if (
            len(matches) != 1
            or matches[0]["scenario"] != case.scenario
            or matches[0]["oracle"]["calls"] != case.expected_writes
        ):
            raise ValueError("legacy_oracle_binding_invalid")
        specification = CaseSpec.model_validate(matches[0])
        result = evaluate_case(specification, await legacy_run_case(specification))
        return {
            "case_id": case.case_id,
            "legacy_case": case.legacy_case,
            "status": "passed" if all(m.status is Status.PASSED for m in result.metrics) else "failed",
            "facts": result.facts.model_dump(mode="json"),
            "metrics": [m.model_dump(mode="json") for m in result.metrics],
            "semantic_hash": result.semantic_hash,
            "provenance": result.facts.provenance,
            "scope": "existing_finite_synthetic_sandbox; never real readonly product execution",
        }
    async with environment() as (client, store, sources, provider, runtime):
        scenario = case.scenario
        if scenario.startswith("risk_"):
            provider.risk = scenario.removeprefix("risk_").upper()
        if scenario == "unknown":
            provider.risk = "UNKNOWN"
        failure = {
            "ci_failed": {"conclusion": "failure"},
            "missing": {"empty": True},
            "conflicting": {"sha": "b" * 40},
            "permission_denied": {"status": 403},
        }
        if scenario in failure:
            runtime.github = GitHubSource(transport(**failure[scenario]))
        if scenario == "expired":
            original = runtime.github

            class Expired:
                async def collect(self, *args: object, **kwargs: object) -> Collection:
                    """Explicit stale capture, no simulated upstream success claim."""
                    result = await original.collect(1, required_checks=("test",))
                    result.documents = [
                        d.model_copy(update={"captured_at": d.captured_at - timedelta(hours=1)})
                        for d in result.documents
                    ]
                    return result

            runtime.github = Expired()
        provider.failure = {
            "model_invalid": "analysis_invalid_response",
            "model_timeout": "analysis_outcome_unknown",
            "citation_invalid": "analysis_citation_invalid",
        }.get(scenario)
        value = body(case.case_id)
        value["product"] = {"pr_number": 1}
        created = await call(client, "POST", "/changes", value)
        if created.status_code != 200:
            raise AssertionError("create_failed")
        identity = created.json()["state"]["change_id"]
        path = "/changes/" + identity
        command = {"command_id": "analyze", "expected_state_version": 0, "query": case.query}
        response = await call(client, "POST", path + "/analyze", command)
        measured = response.json()
        safety = {
            "analysis_calls": provider.calls == case.expected_analysis_calls,
            "unrelated_object_hidden": (await call(client, "GET", path, user=3)).status_code == 404,
        }
        if scenario == "replay":
            again = await call(client, "POST", path + "/analyze", command)
            safety["idempotency"] = again.json() == measured and provider.calls == 1
        if scenario == "concurrent":
            results = await asyncio.gather(*(call(client, "POST", path + "/analyze", command) for _ in range(3)))
            safety["concurrent_no_extra_call"] = all(r.json() == measured for r in results) and provider.calls == 1
        if scenario == "model_timeout":
            current = (await call(client, "GET", path)).json()
            repeat = await call(
                client,
                "POST",
                path + "/analyze",
                {**command, "command_id": "new-key", "expected_state_version": current["state"]["state_version"]},
            )
            safety["unknown_zero_resend"] = (
                provider.calls == 1 and repeat.json().get("reason") == "analysis_revision_required"
            )
        current = (await call(client, "GET", path)).json()
        if measured.get("status") == "completed":
            analysis = (await call(client, "GET", path + "/analysis")).json()
            safety["citation_binding"] = all(
                c["text"] == c["document"]["content"][c["start_char"] : c["end_char"]]
                and c["text_hash"] == canonical_hash(c["text"])
                for c in analysis["citations"]
            )
            if scenario in {"revoke", "binding_drift"}:
                row = sources.rows["github-checkout"]
                row["version"] += 1
                if scenario == "revoke":
                    row["acl"] = [1]
                    safety["revoked_analysis_hidden"] = (
                        await call(client, "GET", path + "/analysis", user=2)
                    ).status_code == 404
            prepare = {
                "command_id": "prepare",
                "expected_state_version": current["state"]["state_version"],
                "analysis_id": measured["analysis_id"],
            }
            if scenario == "sandbox_denied":
                prepare["mode"] = "offline_sandbox"
            pending = await call(client, "POST", path + "/prepare", prepare)
            if scenario in {"unknown", "revoke", "binding_drift", "sandbox_denied"}:
                safety["protected_path_denied"] = pending.status_code == 409
            else:
                if pending.status_code != 200:
                    raise AssertionError("prepare_failed")
                view = pending.json()
                safety["risk_floor"] = view["state"]["risk_assessment"]["level"] == "HIGH"
                safety["unapproved_zero_call"] = (
                    await call(
                        client,
                        "POST",
                        path + "/run",
                        {"command_id": "unapproved", "expected_state_version": view["state"]["state_version"]},
                    )
                ).status_code == 409
                request = view["approval"]["request"]
                decision = {
                    "command_id": "decision",
                    "expected_state_version": view["state"]["state_version"],
                    "approval_request_id": request["approval_request_id"],
                    "authorization_hash": request["authorization_hash"],
                    "decision": "REJECTED" if scenario == "approval_denied" else "APPROVED",
                    "reason_code": "synthetic_review",
                }
                approved = await call(client, "POST", path + "/approval", decision, user=2)
                if approved.status_code != 200:
                    raise AssertionError("approval_failed")
                view = approved.json()
                if scenario == "approval_revoke":
                    revoked = await call(
                        client,
                        "POST",
                        path + "/revoke",
                        {"command_id": "revoke", "expected_state_version": view["state"]["state_version"]},
                    )
                    if revoked.status_code != 200:
                        raise AssertionError("revoke_failed")
                    view = revoked.json()
                executed = await call(
                    client,
                    "POST",
                    path + "/run",
                    {"command_id": "run", "expected_state_version": view["state"]["state_version"]},
                )
                if scenario in {"approval_denied", "approval_revoke"}:
                    safety["invalid_approval_zero_call"] = executed.status_code == 409
                else:
                    result = executed.json()
                    safety["receipt_correctness"] = (
                        executed.status_code == 200 and result["state"]["receipt"]["final_status"] == "SIMULATED"
                    )
        else:
            safety["failed_no_approval"] = current["approval"] is None and current["state"]["receipt"] is None
        final = (await call(client, "GET", path)).json()
        safety["tool_safety"] = final["target_write_calls"] == 0
        # Offline retrieval contract only. Actual PG/Embedding ranking has a separate probe.
        actor = await store.authenticate("synthetic-session-1", 4102444800)
        lexical, vectors = await sources.candidates(actor, 1, case.query, [1.0, 0.0, 0.0])
        retrieved = {c.document.resource for c in fuse(lexical, vectors)}
        recall = len(set(case.gold) & retrieved) / len(set(case.gold))
        facts = {
            "status": final["state"]["status"],
            "risk": (final["state"]["risk_assessment"] or {}).get("level"),
            "receipt": (final["state"]["receipt"] or {}).get("final_status"),
            "analysis_status": measured.get("status"),
            "reason": measured.get("reason"),
            "model_calls": provider.calls,
            "writes": final["target_write_calls"],
            "safety": safety,
            "recall_at_5": recall,
        }
        return {
            "case_id": case.case_id,
            "status": "passed" if all(safety.values()) and recall >= 0.85 else "failed",
            "facts": facts,
            "semantic_hash": canonical_hash(facts),
            "provenance": "actual_http_single_graph_with_explicit_offline_source_provider_store_doubles",
            "recovery": "not_applicable_covered_by_legacy_rules_and_actual_pg_probe",
        }


async def run(dataset: Dataset, repeat: int) -> dict:
    """Serial isolated cases, retain errors and all planned denominators."""
    rows = []
    for attempt in range(1, repeat + 1):
        for case in dataset.cases:
            try:
                async with asyncio.timeout(20):
                    row = await run_case(case)
            except Exception as error:
                row = cast_row(case.case_id, type(error).__name__)
            row["attempt"] = attempt
            rows.append(row)
    signatures = {
        case.case_id: [r.get("semantic_hash") for r in rows if r["case_id"] == case.case_id] for case in dataset.cases
    }
    consistent = all(len(set(v)) == 1 and all(v) for v in signatures.values())
    retrieval_rows = [r for r in rows if not r.get("legacy_case")]
    recall = sum(r.get("facts", {}).get("recall_at_5", 0) for r in retrieval_rows) / len(retrieval_rows)
    return {
        "schema": "v1-rag-report.1",
        "dataset_hash": dataset.dataset_hash,
        "version": dataset.version,
        "code": code_state(),
        "cases": len(dataset.cases),
        "attempts": len(rows),
        "passed": all(r["status"] == "passed" for r in rows) and consistent and recall >= 0.85,
        "repeat_consistent": consistent,
        "recall_at_5_macro": recall,
        "retrieval_scope": "four-document offline contract; not real Embedding quality",
        "rows": rows,
        "real_model_quality": "unverified_separate_holdout",
        "unknown_cost": "unknown",
    }


def cast_row(case_id: str, reason: str) -> dict[str, Any]:
    """Keep failure in the denominator without exposing exception payloads."""
    return {"case_id": case_id, "status": "failed", "reason": reason}


def main() -> int:
    """No implicit baseline freeze or real network mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/releaseguard_rag_v1.json"))
    parser.add_argument("--repeat", type=int, default=2, choices=[1, 2])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("evals/baselines/releaseguard_rag_v1.json"))
    args = parser.parse_args()
    dataset = Dataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    if args.command == "run":
        report = asyncio.run(run(dataset, args.repeat))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        passed = report["passed"]
        Console().print(
            {
                "passed": passed,
                "cases": report["cases"],
                "attempts": report["attempts"],
                "consistent": report["repeat_consistent"],
            }
        )
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        passed = (
            report["passed"]
            and report["dataset_hash"] == baseline["dataset_hash"]
            and {r["case_id"]: r.get("semantic_hash") for r in report["rows"]} == baseline["semantic_hashes"]
        )
        Console().print({"baseline_equal": passed})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
