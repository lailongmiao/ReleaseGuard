"""Explicitly authorized single-batch HTTP sampling; no retries or replacement slots."""

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.schemas.v11_backend import Comparison, ReviewOutput
from app.services.releaseguard.normalization import canonical_hash
from app.services.v1_provider import PRICE_EXPIRY, PRICES
from app.services.v11_evaluation_policy import FORMAL_BATCH, LINK_CONTRACT
from evals.v1_backend.provenance import code_state
from evals.v11_backend.contracts import Dataset
from evals.v11_backend.http_evidence import verify_windows_acl
from evals.v11_backend.review_submission import ReviewSubmissionError, prepare, wait_review
from evals.v11_backend.recovery import Segment


class Authorization(BaseModel):
    """A private operation record pins approval, dataset, source code and billing ledger."""

    model_config = ConfigDict(extra="forbid")
    authorization_reference: str = Field(min_length=10)
    authorized_at: datetime
    expires_at: datetime
    public_github_reads: bool
    model_calls: bool
    max_analysis_calls: int = Field(ge=80, le=80)
    dataset_hash: str
    source_fingerprint: str
    designated_ledger_id: str = Field(min_length=10)
    reconciled_budget_evidence_hash: str
    price_evidence_hash: str
    api_origin: str
    database_name: Literal["v11_isolated"]
    database_port: int = Field(strict=True, ge=1024, le=65535)
    owner_user_id: int = Field(gt=0)
    approver_user_id: int = Field(gt=0)
    session_token: SecretStr
    budget_remaining_cny: float = Field(gt=0, le=25)
    rules_ready: bool
    acceptance_increment_baseline: Decimal = Decimal("24.60984678")
    cumulative_ceiling: Decimal = Decimal("43.60984678")
    required_batch_reservation: Decimal = Decimal("18.756992")

    def validate_preflight(self, dataset: Dataset, *, recovery: bool = False) -> None:
        """Reject stale prices, unreviewed snapshots, remote servers or implicit approval."""
        origin = urlparse(self.api_origin)
        now = datetime.now(UTC)
        if dataset.provenance != "frozen_public_source":
            raise ValueError("real_frozen_source_evidence_required")
        if (
            origin.scheme != "http"
            or origin.hostname not in {"127.0.0.1", "localhost"}
            or origin.username
            or origin.password
            or origin.query
            or origin.fragment
            or origin.path not in {"", "/"}
        ):
            raise ValueError("local_api_origin_required")
        if (
            self.authorized_at.tzinfo is None
            or self.expires_at.tzinfo is None
            or not self.authorized_at <= now < self.expires_at
        ):
            raise ValueError("current_explicit_authorization_required")
        if self.expires_at - self.authorized_at > timedelta(hours=8):
            raise ValueError("formal_batch_eight_hour_limit")
        if (
            self.authorization_reference != FORMAL_BATCH
            or self.acceptance_increment_baseline != Decimal("24.60984678")
            or self.cumulative_ceiling != Decimal("43.60984678")
            or self.required_batch_reservation != Decimal("18.29706720" if recovery else "18.756992")
        ):
            raise ValueError("formal_batch_fixed_authorization_required")
        if (
            not self.public_github_reads
            or not self.model_calls
            or not self.rules_ready
            or self.owner_user_id == self.approver_user_id
        ):
            raise ValueError("real_preconditions_missing")
        if now >= datetime.fromisoformat(PRICE_EXPIRY):
            raise ValueError("price_evidence_expired")
        if self.dataset_hash != canonical_hash(dataset) or self.source_fingerprint != code_state()["source_hash"]:
            raise ValueError("freeze_mismatch")
        expected_price = canonical_hash(
            {"expiry": PRICE_EXPIRY, "prices": {key: [str(p[0]), str(p[1]), p[2]] for key, p in PRICES.items()}}
        )
        if self.price_evidence_hash != expected_price:
            raise ValueError("price_evidence_mismatch")
        if (
            not self.reconciled_budget_evidence_hash
            or Decimal(str(self.budget_remaining_cny)) < self.required_batch_reservation
        ):
            raise ValueError("budget_preflight_insufficient")


def position_stamp(auth: Authorization, case_id: str, repeat: int) -> str:
    """The frozen position, not a restarted process, owns the command identity."""
    return canonical_hash([auth.authorization_reference, auth.dataset_hash, case_id, repeat])


def capture_positions(dataset: Dataset, auth: Authorization) -> dict:
    """Share exact analyze identities with the signed HTTP capture permit."""
    return {
        "analysis:" + position_stamp(auth, case.case_id, repeat): {
            "case_id": case.case_id,
            "repeat": repeat,
            "query": case.query,
            "binding": case.binding.model_dump(mode="json"),
            "comparison": case.comparison.model_dump(mode="json"),
            "sync_command": "sync:" + position_stamp(auth, case.case_id, repeat),
        }
        for case in dataset.cases
        for repeat in (1, 2)
    }


def checkpoint(path: Path, report: dict, *, create: bool = False) -> None:
    """Exclusive start plus fsync/replace: restart cannot treat an uncertain slot as unused."""
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    target = path if create else path.with_name(path.name + ".pending")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not create:
        os.replace(target, path)


async def await_review(
    slot: dict, directory: Path, auth: Authorization, reauthorize: Callable[[], Awaitable[None]]
) -> bool:
    """Freeze one packet; invalid metadata never permits a next Provider send."""
    digest = prepare(directory, slot, auth.authorization_reference, auth.dataset_hash, auth.expires_at)
    return await wait_review(directory, slot["stamp"], digest, reauthorize)


async def sample(
    dataset: Dataset,
    auth: Authorization,
    output: Path,
    *,
    reviewer: Callable[[dict], Awaitable[bool]] | None = None,
    recovery: Segment | None = None,
) -> dict:
    """Real entry keeps all original authorization checks; no restart or replacement slots."""
    auth.validate_preflight(dataset, recovery=recovery is not None)
    if recovery is not None and reviewer is not None:
        raise ValueError("recovery_private_review_channel_required")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("private_report_directory_required")
    if os.name != "nt" and output.parent.stat().st_mode & 0o077:
        raise ValueError("private_report_directory_required")
    verify_windows_acl(output.parent)
    if output.exists() or output.with_name(output.name + ".pending").exists():
        raise ValueError("final_batch_already_exists_no_resend")
    headers = {"Authorization": "Bearer " + auth.session_token.get_secret_value()}
    async with httpx.AsyncClient(
        base_url=auth.api_origin, headers=headers, trust_env=False, follow_redirects=False, timeout=120
    ) as client:
        readiness = await client.get("/api/v1/releaseguard/reviews/runtime")
        readiness.raise_for_status()
        runtime = readiness.json()
        if (
            runtime.get("ledger_id") != auth.designated_ledger_id
            or runtime.get("budget_evidence_hash") != auth.reconciled_budget_evidence_hash
            or runtime.get("provider") != "real"
            or runtime.get("sources") != "real"
            or runtime.get("database_port") != auth.database_port
            or runtime.get("database_name") != auth.database_name
        ):
            raise ValueError("designated_runtime_not_verified")
        capture = await client.get("/_evaluation/runtime")
        capture.raise_for_status()
        evidence = capture.json()
        if (
            evidence.get("mode") != "real"
            or evidence.get("dataset_hash") != auth.dataset_hash
            or evidence.get("source_hash") != auth.source_fingerprint
            or evidence.get("positions_hash") != canonical_hash(capture_positions(dataset, auth))
            or evidence.get("ledger_id") != auth.designated_ledger_id
            or evidence.get("maximum_batch_charge") != "19"
            or evidence.get("increment_baseline") != str(auth.acceptance_increment_baseline)
            or evidence.get("cumulative_ceiling") != str(auth.cumulative_ceiling)
            or evidence.get("overlap_exception") != LINK_CONTRACT
            or evidence.get("recovery_manifest_hash") != (canonical_hash(recovery.manifest) if recovery else None)
            or (
                recovery is not None and evidence.get("recovery_boot_id") != recovery.store.evaluation_recovery_boot_id
            )
            or datetime.fromisoformat(evidence["expires_at"]) < auth.expires_at
        ):
            raise ValueError("private_capture_runtime_not_verified")
        try:
            return await sample_positions(dataset, auth, output, client, reviewer=reviewer, recovery=recovery)
        finally:
            if recovery is not None:
                await recovery.stop()


async def sample_positions(
    dataset: Dataset,
    auth: Authorization,
    output: Path,
    client: httpx.AsyncClient,
    *,
    reviewer: Callable[[dict], Awaitable[bool]] | None = None,
    pace_seconds: float = 13,
    recovery: Segment | None = None,
) -> dict:
    """Transport-injected engine also drives synthetic tests; never manufactures authorization."""
    if output.exists() or output.with_name(output.name + ".pending").exists():
        raise ValueError("final_batch_already_exists_no_resend")
    report = {
        "version": "v11-real-batch.2",
        "dataset": dataset.model_dump(mode="json"),
        "authorization_reference": auth.authorization_reference,
        "source_fingerprint": auth.source_fingerprint,
        "price_evidence_hash": auth.price_evidence_hash,
        "ledger_id": auth.designated_ledger_id,
        "provenance": "transport_execution_requires_runtime_attestation",
        "positions": [],
        "expected_positions": 80,
        "expected_rag_positions": 80,
        "completed": False,
    }
    if recovery is not None:
        recovery.manifest.check()
        if auth.authorization_reference != FORMAL_BATCH or canonical_hash(dataset) != recovery.manifest.dataset_hash:
            raise ValueError("recovery_frozen_identity_mismatch")
        report["manifest_hash"] = canonical_hash(recovery.manifest)
        report["segment_id"] = recovery.manifest.recovery_id
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint(output, report, create=True)
    prefix = "/api/v1/releaseguard/reviews"

    async def call(method: str, path: str, body: dict | None = None) -> dict:
        started = time.perf_counter()
        status = None
        outcome = "interrupted"
        # Fixed labels only: no request path, query, identifiers, headers or body.
        stage = (
            "evidence_read"
            if path.startswith("/_evaluation/")
            else "source_sync"
            if path.endswith("/sync")
            else "analysis"
            if path.endswith("/analyze")
            else "rag"
            if path.endswith("/search")
            else "report_read"
            if "/analyses/" in path or "/commands/" in path
            else "object_preparation"
        )
        try:
            response = await client.request(method, path, json=body)
            status = response.status_code
            response.raise_for_status()
            result = response.json()
            outcome = "completed"
            return result
        finally:
            slot.setdefault("http_timings", []).append(
                {
                    "stage": stage,
                    "method": method,
                    "status": status,
                    "outcome": outcome,
                    "seconds": time.perf_counter() - started,
                }
            )

    for case in dataset.cases:
        if recovery is not None and case.case_id == "real-01-zh":
            continue
        identifier = None
        first_generation = None
        for repeat in (1, 2):
            if datetime.now(UTC) >= auth.expires_at:
                report["stop_reason"] = "authorization_expired"
                checkpoint(output, report)
                return report
            if report["positions"]:
                await asyncio.sleep(pace_seconds)
            stamp = position_stamp(auth, case.case_id, repeat)
            if recovery is not None:
                await recovery.begin(stamp)
            slot = {
                "case_id": case.case_id,
                "repeat": repeat,
                "planned_revision": repeat,
                "stamp": stamp,
                "status": "dispatching",
                "analysis": None,
                "input": None,
                "retrieved_documents": [],
                "hard_gates_passed": False,
                "review_wait_seconds": 0.0,
            }
            report["positions"].append(slot)
            # Failure here happens before HTTP; existing journal still forbids automatic restart.
            checkpoint(output, report)
            started = time.perf_counter()
            try:
                async with asyncio.timeout(120):
                    at = datetime.now(UTC)
                    change = {
                        "requester_id": "local-user:" + str(auth.owner_user_id),
                        "service": "releaseguard-acceptance",
                        "environment": "development",
                        "title": case.case_id,
                        "description": "Readonly frozen V1.1 acceptance",
                        "change_window": {
                            "start": at.isoformat(),
                            "end": (at + timedelta(hours=1)).isoformat(),
                            "timezone": "UTC",
                        },
                        "rollback_plan": "No real target writes; retain earlier immutable facts",
                        "verification_plan": "Read saved evidence and simulated receipt only",
                    }
                    if repeat == 1:
                        obj = await call(
                            "POST",
                            prefix,
                            {
                                "idempotency_key": "acceptance-object:" + stamp,
                                "approver_user_id": auth.approver_user_id,
                                "binding": case.binding.model_dump(mode="json"),
                                "change": change,
                            },
                        )
                        identifier = obj["object_id"]
                    current = await call("GET", prefix + "/" + str(identifier))
                    slot["object_id"] = identifier
                    synced = await call(
                        "POST",
                        prefix + "/" + str(identifier) + "/sync",
                        {"command_id": "sync:" + stamp, "expected_object_version": current["object_version"]},
                    )
                    if synced["status"] != "completed" or not synced.get("generation_id"):
                        raise ValueError("sync_not_completed_no_retry")
                    snapshot = await call(
                        "GET", prefix + "/" + str(identifier) + "/snapshots/" + synced["snapshot_id"]
                    )
                    if Comparison.model_validate(snapshot["facts"]["comparison"]) != case.comparison:
                        raise ValueError("frozen_source_comparison_changed")
                    current = await call("GET", prefix + "/" + str(identifier))
                    if repeat == 2:
                        if synced["generation_id"] != first_generation:
                            raise ValueError("same_source_generation_changed")
                        revised = await call(
                            "POST",
                            prefix + "/" + str(identifier) + "/revision",
                            {
                                "command_id": "revision:" + stamp,
                                "expected_object_version": current["object_version"],
                                "expected_state_version": current["state"]["state_version"],
                                "snapshot_id": synced["snapshot_id"],
                                "generation_id": synced["generation_id"],
                                "change": change,
                            },
                        )
                        if revised["status"] != "completed":
                            raise ValueError("revision_not_completed")
                        current = await call("GET", prefix + "/" + str(identifier))
                    else:
                        first_generation = synced["generation_id"]
                    if current["state"]["revision"] != repeat:
                        raise ValueError("revision_mapping_mismatch")
                    intent = {
                        "command_id": "analysis:" + stamp,
                        "expected_object_version": current["object_version"],
                        "expected_state_version": current["state"]["state_version"],
                        "snapshot_id": synced["snapshot_id"],
                        "generation_id": synced["generation_id"],
                        "query": case.query,
                    }
                    result = await call("POST", prefix + "/" + str(identifier) + "/analyze", intent)
                    slot.update(status=result["status"], reason=result.get("reason"))
                    evidence = {}
                    for artifact in ("input", "answer", "validation", "sync_timing", "analyze_timing"):
                        evidence[artifact] = await call(
                            "GET", "/_evaluation/" + intent["command_id"] + "/evidence/" + artifact
                        )
                    slot["evaluation_evidence"] = evidence
                    slot["input"] = {"query": case.query, "snapshot": snapshot, "captured": evidence["input"]}
                    slot["command_evidence"] = {
                        stage: await call("GET", prefix + "/" + str(identifier) + "/commands/" + command)
                        for stage, command in (("sync", "sync:" + stamp), ("analysis", intent["command_id"]))
                    }
                    if result["status"] == "completed":
                        analysis = await call(
                            "GET", prefix + "/" + str(identifier) + "/analyses/" + result["analysis_id"]
                        )
                        ReviewOutput.model_validate(analysis["output"])
                        slot.update(
                            analysis=analysis,
                            hard_gates_passed=not analysis["stale_reasons"] and analysis["provenance"] == "provider",
                        )
                        if not evidence["answer"].get("review_complete") or evidence["input"].get("input_redacted"):
                            raise ValueError("complete_review_evidence_required")
                slot["elapsed_seconds"] = time.perf_counter() - started
                checkpoint(output, report)
                if slot["status"] != "completed":
                    report["stop_reason"] = "analysis_not_completed_no_resend"
                    checkpoint(output, report)
                    return report
                rag_started = time.perf_counter()
                async with asyncio.timeout(120):
                    search = {k: v for k, v in intent.items() if k != "expected_state_version"}
                    search.update(command_id="rag:" + stamp, query=case.retrieval_query or case.query)
                    found = await call("POST", prefix + "/" + str(identifier) + "/search", search)
                    slot.update(
                        rag_status=found["status"],
                        rag_query=search["query"],
                        rag_input=search,
                        rag_result=found,
                        rag_command_evidence=await call(
                            "GET", prefix + "/" + str(identifier) + "/commands/" + search["command_id"]
                        ),
                        retrieved_documents=[item["document"]["resource"] for item in found.get("items", [])],
                    )
                slot["rag_elapsed_seconds"] = time.perf_counter() - rag_started
                checkpoint(output, report)
                if slot["rag_status"] != "completed":
                    report["stop_reason"] = "rag_not_completed_no_retry"
                    checkpoint(output, report)
                    return report
                wait_started = time.perf_counter()

                async def reauthorize(review_stamp: str = stamp, review_slot: dict = slot) -> None:
                    # Existing private routes recheck session, object and live source ACL.
                    # These are review-time reads, not deducted from the saved request clock.
                    for artifact in ("input", "answer", "validation"):
                        permission_started = time.perf_counter()
                        status = None
                        try:
                            response = await client.get(f"/_evaluation/analysis:{review_stamp}/evidence/{artifact}")
                            status = response.status_code
                        except httpx.HTTPError as error:
                            raise ReviewSubmissionError("review_live_permission_unavailable") from error
                        finally:
                            review_slot.setdefault("review_permission_checks", []).append(
                                {
                                    "artifact": artifact,
                                    "status": status,
                                    "seconds": time.perf_counter() - permission_started,
                                }
                            )
                        if response.status_code != 200:
                            raise ReviewSubmissionError("review_live_permission_denied")
                        try:
                            refreshed = response.json()
                        except ValueError as error:
                            raise ReviewSubmissionError("review_live_evidence_invalid") from error
                        if refreshed != review_slot["evaluation_evidence"][artifact]:
                            raise ReviewSubmissionError("review_live_evidence_changed")

                approved = await (reviewer(slot) if reviewer else await_review(slot, output.parent, auth, reauthorize))
                slot["review_wait_seconds"] = time.perf_counter() - wait_started
                slot["review_approved"] = approved
                checkpoint(output, report)
                if not approved:
                    report["stop_reason"] = "semantic_review_stopped"
                    checkpoint(output, report)
                    return report
                if recovery is not None:
                    await recovery.finish(slot)
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError) as error:
                slot.update(
                    status="uncertain" if isinstance(error, (httpx.HTTPError, TimeoutError)) else "failed",
                    reason=type(error).__name__,
                    failure_code=(
                        str(error)
                        if isinstance(error, ReviewSubmissionError)
                        else "complete_review_evidence_required"
                        if isinstance(error, ValueError) and str(error) == "complete_review_evidence_required"
                        else "request_outcome_requires_review"
                    ),
                )
                slot.setdefault("elapsed_seconds", time.perf_counter() - started)
                report["stop_reason"] = "position_failed_no_retry"
                checkpoint(output, report)
                return report
    report["completed"] = len(report["positions"]) == (78 if recovery is not None else 80)
    checkpoint(output, report)
    return report
