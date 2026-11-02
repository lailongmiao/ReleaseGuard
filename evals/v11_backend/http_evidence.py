"""Signed, private evaluation evidence on the existing HTTP/Graph analysis path."""

import hashlib
import hmac
import json
import os
import subprocess
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.releaseguard import Actor
from app.api.v1.reviews import Backend, ReviewRoute
from app.core.config import settings
from app.core.limiter import limiter
from app.schemas.local_mvp import LocalActor
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_storage import V11Store
from app.services.v11_timing import span
from app.services.v11_provider import SYSTEM
from app.services.v11_policy import PROMPT_VERSION
from app.services.v11_evaluation_policy import (
    DATASET_HASH,
    FORMAL_BATCH,
    LINK_CONTRACT,
    OVERLAP_CASES,
    RETIRED_BATCHES,
)
from evals.v11_backend.contracts import Dataset
from evals.v1_backend.provenance import code_state
from evals.v11_backend.controlled import save_new
from evals.v11_backend.structured_evidence import STRUCTURED_LIMIT, structured_evidence
from evals.v11_backend.transport import FormalGitHubTransport
from evals.v11_backend.complete_evidence import input_evidence


class EvidencePermit(BaseModel):
    """An operator signature grants capture, never model sends or UNKNOWN recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["v11-http-evaluation.1"] = "v11-http-evaluation.1"
    batch_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    owner_id: int = Field(gt=0)
    expires_at: datetime
    mode: Literal["synthetic", "real"]
    designated_ledger_id: str = Field(min_length=10)
    ledger_charge_at_start: Decimal = Field(ge=0)
    maximum_batch_charge: Decimal = Field(gt=0, le=19)
    # command -> frozen query, binding, comparison, case_id, repeat
    positions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=80)
    overlap_exception: Literal["v11-formal-overlap.1"] | None = None
    rule_batches: dict[Literal["application-policy", "checkout-reference"], int] = Field(default_factory=dict)
    recovery_manifest_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class HTTPEvaluationEvidence:
    """Mounted only by an explicit operator server, never by ordinary API startup."""

    def __init__(self, directory: Path, permit: EvidencePermit, signature: str, key: bytes) -> None:
        """Verify the operator's bounded capture authorization before mounting routes."""
        if len(key) < 32 or not hmac.compare_digest(
            signature, hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
        ):
            raise DomainError("evaluation_approval_invalid")
        if not directory.is_dir() or directory.is_symlink():
            raise DomainError("evaluation_private_directory_required")
        if os.name != "nt" and directory.stat().st_mode & 0o077:
            raise DomainError("evaluation_private_directory_required")
        verify_windows_acl(directory)
        if code_state()["source_hash"] != permit.source_hash:
            raise DomainError("evaluation_frozen_identity_mismatch")
        self.directory, self.permit = directory, permit
        self.recovery_boot_id = uuid4().hex if permit.recovery_manifest_hash else None
        self.github_transport: FormalGitHubTransport | None = None
        self.validate_overlap_permit()
        if permit.mode == "real" and permit.designated_ledger_id != settings.V11_BUDGET_LEDGER_ID:
            raise DomainError("evaluation_ledger_mismatch")
        self.check_time()
        if permit.mode == "real":
            self.github_transport = FormalGitHubTransport(
                directory, check_time=self.check_time, carried_attempts=18 if permit.recovery_manifest_hash else 0
            )

    def validate_overlap_permit(self) -> None:
        """Only the approved batch and original four positions can claim this exception."""
        permit = self.permit
        if permit.recovery_manifest_hash and (
            permit.batch_id != FORMAL_BATCH or permit.overlap_exception != LINK_CONTRACT
        ):
            raise DomainError("recovery_fixed_manifest_required")
        if permit.batch_id in RETIRED_BATCHES:
            raise DomainError("evaluation_batch_retired")
        overlap = [p for p in permit.positions.values() if p["case_id"] in OVERLAP_CASES]
        if not permit.overlap_exception:
            if permit.mode == "real" and overlap:
                raise DomainError("evaluation_overlap_not_authorized")
            return
        if permit.batch_id != FORMAL_BATCH or sorted((p["case_id"], p["repeat"]) for p in overlap) != [
            (case, repeat) for case in sorted(OVERLAP_CASES) for repeat in (1, 2)
        ]:
            raise DomainError("evaluation_overlap_not_authorized")
        if permit.mode == "real":
            if permit.ledger_charge_at_start != Decimal("24.60984678") or permit.maximum_batch_charge != Decimal("19"):
                raise DomainError("evaluation_fixed_cumulative_ceiling_required")
            if any(type(n) is not int or not 1 <= n <= 3 for n in permit.rule_batches.values()):
                raise DomainError("evaluation_rule_batch_limit_invalid")
            dataset = Dataset.model_validate_json(
                Path("evals/datasets/v11_acceptance.json").read_text(encoding="utf-8")
            )
            if canonical_hash(dataset) != DATASET_HASH or permit.dataset_hash != DATASET_HASH:
                raise DomainError("evaluation_frozen_input_mismatch")
            for command, position in permit.positions.items():
                case = next((c for c in dataset.cases if c.case_id == position["case_id"]), None)
                stamp = canonical_hash([FORMAL_BATCH, DATASET_HASH, position["case_id"], position["repeat"]])
                if (
                    case is None
                    or position["repeat"] not in (1, 2)
                    or command != "analysis:" + stamp
                    or position.get("sync_command") != "sync:" + stamp
                    or position["query"] != case.query
                    or position["binding"] != case.binding.model_dump(mode="json")
                    or position["comparison"] != case.comparison.model_dump(mode="json")
                ):
                    raise DomainError("evaluation_frozen_input_mismatch")
            if len(permit.positions) != 80:
                raise DomainError("evaluation_fixed_denominator_required")

    def allowed_attempts(self) -> dict[str, set[str]]:
        """Fixed identities bound 80 models, 160 queries, 120 object index batches."""
        result: dict[str, set[str]] = {}
        for command, position in self.permit.positions.items():
            result[command] = {"query-1-batch-0", "analysis-1"}
            result["rag:" + command.removeprefix("analysis:")] = {"query-1-batch-0"}
            result[position["sync_command"]] = (
                {f"index-batch-{i}" for i in range(3)} if position["repeat"] == 1 else set()
            )
        for rule, count in self.permit.rule_batches.items():
            result[f"rules:{self.permit.batch_id}:{rule}"] = {f"index-batch-{i}" for i in range(count)}
        return result

    def validate_revision(self, command: str, revision: int) -> None:
        """A signed position cannot be reused against a different revision."""
        if self.permit.positions[command]["repeat"] != revision:
            raise DomainError("evaluation_revision_mismatch")

    def sync_position(self, command: str, binding: dict) -> dict:
        """Freeze sync identity and stop drift before any paid indexing."""
        self.check_time()
        position = next((p for p in self.permit.positions.values() if p.get("sync_command") == command), None)
        if position is None or any(binding[k] != position["binding"][k] for k in ("repository_id", "selector")):
            raise DomainError("evaluation_position_not_authorized")
        return position

    async def check_overlap(self, store: V11Store, actor: LocalActor, identifier: str, position: dict) -> None:
        """Check before collection/indexing and again before model claims."""
        if self.permit.mode == "real" or self.permit.overlap_exception:
            await store.check_evaluation_overlap(
                identifier,
                actor,
                permitted=bool(self.permit.overlap_exception and position["case_id"] in OVERLAP_CASES),
            )

    def check_time(self) -> None:
        """New captures require an unexpired permit and matching execution mode."""
        if self.permit.expires_at.tzinfo is None or datetime.now(UTC) >= self.permit.expires_at:
            raise DomainError("evaluation_approval_expired")
        if self.permit.mode == "synthetic" and (
            settings.V11_REAL_MODELS_ENABLED or settings.V11_REAL_READS_ENABLED or settings.V1_REAL_MODELS_ENABLED
        ):
            raise DomainError("evaluation_synthetic_only")

    def path(self, command: str, artifact: str) -> Path:
        """Use hashed filenames; commands can never select an arbitrary path."""
        if command not in self.permit.positions or artifact not in {
            "intent",
            "input",
            "answer",
            "validation",
            "sync_timing",
            "analyze_timing",
        }:
            raise DomainError("evaluation_position_not_authorized")
        return self.directory / (canonical_hash([self.permit.batch_id, command]) + "." + artifact + ".json")

    def save_timing(self, command: str, operation: str, report: dict, *, failed: bool) -> None:
        """Private exclusive sidecar; a replay never overwrites prior timing evidence."""
        target = (
            command
            if command in self.permit.positions
            else next(
                (key for key, value in self.permit.positions.items() if value.get("sync_command") == command), None
            )
        )
        if target is None:
            return
        path = self.path(target, operation + "_timing")
        if path.exists():
            return
        try:
            save_new(path, report)
        except OSError:
            if not failed:
                raise DomainError("evaluation_timing_save_failed_no_resend") from None

    async def authorize(self, store: V11Store, actor: LocalActor, identifier: str, body: Any) -> None:
        """Check frozen scope before claim/embedding; exclusive intent survives crashes."""
        self.check_time()
        position = self.permit.positions.get(body.command_id)
        obj = await store.object_row(identifier, actor)
        if not position or actor.user_id != self.permit.owner_id or obj["owner_user_id"] != actor.user_id:
            raise DomainError("evaluation_position_not_authorized")
        binding = {"repository_id": obj["repository_id"], "selector": obj["selector"]}
        expected = position["binding"]
        if body.query != position["query"] or binding != {
            "repository_id": expected["repository_id"],
            "selector": expected["selector"],
        }:
            raise DomainError("evaluation_frozen_input_mismatch")
        snapshot = await store.snapshot(identifier, body.snapshot_id, actor)
        if snapshot["facts"]["comparison"] != position["comparison"]:
            raise DomainError("evaluation_frozen_source_mismatch")
        await store.connection.authenticate(actor.session_id, actor.expires_at)
        await self.check_overlap(store, actor, identifier, position)
        overlap = self.permit.overlap_exception and position["case_id"] in OVERLAP_CASES
        if obj.get("evaluation_readonly") and not overlap:
            raise DomainError("evaluation_overlap_not_authorized")
        for other_command, other in self.permit.positions.items():
            if other_command == body.command_id or other["case_id"] != position["case_id"]:
                continue
            other_path = self.path(other_command, "intent")
            if other_path.exists():
                previous = await self.read(store, actor, other_command, "intent")
                if previous["object_id"] != identifier:
                    raise DomainError("evaluation_case_object_mismatch")
        if overlap:
            await store.claim_evaluation_position(
                identifier,
                actor,
                {
                    "batch_id": self.permit.batch_id,
                    "case_id": position["case_id"],
                    "repeat": position["repeat"],
                    "analysis_command": body.command_id,
                    "source_hash": self.permit.source_hash,
                    "dataset_hash": self.permit.dataset_hash,
                    "position_input_hash": canonical_hash(position),
                    "snapshot_id": body.snapshot_id,
                    "contract": LINK_CONTRACT,
                },
            )
        try:
            save_new(
                self.path(body.command_id, "intent"),
                {
                    "owner_id": actor.user_id,
                    "object_id": identifier,
                    "command_id": body.command_id,
                    "batch_id": self.permit.batch_id,
                    "dataset_hash": self.permit.dataset_hash,
                    "position": {k: position[k] for k in ("case_id", "repeat")},
                    "status": "claimed_no_automatic_resend",
                    "dependencies": await store.dependencies(identifier, actor),
                },
            )
        except FileExistsError:
            raise DomainError("evaluation_position_consumed_no_resend") from None

    async def read(self, store: V11Store, actor: LocalActor, command: str, artifact: str) -> dict:
        """Reads recheck owner, current sources and the captured dependency grants."""
        if actor.user_id != self.permit.owner_id:
            raise DomainError("local_object_not_found")
        await store.connection.authenticate(actor.session_id, actor.expires_at)
        intent_path = self.path(command, "intent")
        if intent_path.is_symlink() or not intent_path.is_file() or intent_path.stat().st_size > STRUCTURED_LIMIT:
            raise DomainError("local_object_not_found")
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        obj = await store.object_row(intent["object_id"], actor)
        dependencies = await store.dependencies(intent["object_id"], actor)
        if obj["owner_user_id"] != actor.user_id or dependencies["sources"] != intent["dependencies"]["sources"]:
            raise DomainError("source_not_found")
        path = self.path(command, artifact)
        if not path.exists():
            return {"status": "not_retained", "artifact": artifact}
        if path.is_symlink() or path.stat().st_size > STRUCTURED_LIMIT:
            raise DomainError("evaluation_artifact_invalid")
        return json.loads(path.read_text(encoding="utf-8"))

    async def begin(
        self,
        store: V11Store,
        actor: LocalActor,
        identifier: str,
        command: str,
        user: str,
        citations: list,
        observations: dict,
    ) -> Callable[[Any], Awaitable[None]]:
        """Save the exact bounded input before dispatch and return a private answer sink."""
        self.check_time()
        await self.read(store, actor, command, "intent")
        if self.permit.overlap_exception and self.permit.positions[command]["case_id"] in OVERLAP_CASES:
            await store.save_evaluation_input(identifier, actor, command, canonical_hash([SYSTEM, user]))
        retained, public_tokens = input_evidence(user, citations, observations)
        if len(json.dumps(retained, ensure_ascii=False).encode()) > STRUCTURED_LIMIT - len(SYSTEM.encode()) - 8192:
            retained.update(
                user=None,
                review_complete=False,
                input_truncated=True,
                failure_code="evaluation_input_capacity_exceeded",
                review_impact="incomplete",
            )
        with span("input_evidence_save"):
            save_new(
                self.path(command, "input"),
                {
                    **retained,
                    "system": SYSTEM,
                    "prompt_version": PROMPT_VERSION,
                    "citation_ids": [c.citation_id for c in citations],
                    "object_id": identifier,
                },
            )
        if not retained["review_complete"]:
            raise DomainError(retained["failure_code"])

        async def capture(parsed: Any) -> None:
            await self.read(store, actor, command, "intent")
            try:
                answer = structured_evidence(parsed, {c.citation_id for c in citations}, public_tokens=public_tokens)
                with span("answer_evidence_save"):
                    save_new(self.path(command, "answer"), answer)
            except OSError:
                raise DomainError("evaluation_answer_save_failed_no_resend") from None
            if not answer["review_complete"]:
                raise DomainError("evaluation_answer_incomplete_no_resend")

        return capture

    async def finish(
        self, store: V11Store, actor: LocalActor, command: str, status: str, diagnostic: dict | None = None
    ) -> None:
        """Persist validation independently from any business report adoption."""
        await self.read(store, actor, command, "intent")
        async with await store.connection.connect() as db:
            row = await (
                await db.execute(
                    "SELECT usage,reserved,charge FROM v1_budget WHERE owner_id=%s AND command_id=%s AND attempt='analysis-1'",
                    (actor.user_id, command),
                )
            ).fetchone()
        save_new(
            self.path(command, "validation"),
            {
                "status": status,
                "not_semantic_certification": True,
                "diagnostic": diagnostic,
                "budget": json.loads(json.dumps(row, default=str)) if row else None,
            },
        )


router = APIRouter(route_class=ReviewRoute)


@router.get("/runtime")
@limiter.limit("120/minute")
async def capture_runtime(request: Request, current: Actor, runtime: Backend) -> dict:
    """Attest a frozen capture permit before a sampler can send its first request."""
    if (
        not settings.LOCAL_MVP_ONLY
        or not request.client
        or request.client.host not in {"127.0.0.1", "::1", "testclient"}
        or runtime.evaluation is None
        or current.user_id != runtime.evaluation.permit.owner_id
    ):
        raise DomainError("local_object_not_found")
    await runtime.store.connection.authenticate(current.session_id, current.expires_at)
    runtime.evaluation.check_time()
    permit = runtime.evaluation.permit
    return {
        "mode": permit.mode,
        "dataset_hash": permit.dataset_hash,
        "source_hash": permit.source_hash,
        "positions_hash": canonical_hash(permit.positions),
        "ledger_id": permit.designated_ledger_id,
        "maximum_batch_charge": str(permit.maximum_batch_charge),
        "increment_baseline": str(permit.ledger_charge_at_start),
        "cumulative_ceiling": str(permit.ledger_charge_at_start + permit.maximum_batch_charge),
        "overlap_exception": permit.overlap_exception,
        "recovery_manifest_hash": permit.recovery_manifest_hash,
        "recovery_boot_id": runtime.evaluation.recovery_boot_id,
        "expires_at": permit.expires_at.isoformat(),
    }


def verify_windows_acl(directory: Path) -> None:
    """Reject any Windows allow ACE outside the current identity and LocalSystem."""
    if os.name != "nt":
        return
    script = (
        "$ErrorActionPreference='Stop';"
        "$me=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
        "$acl=Get-Acl -LiteralPath $env:V11_CAPTURE_ACL_PATH;"
        "$bad=@($acl.Access | Where-Object {"
        "$_.AccessControlType -eq 'Allow' -and "
        "$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -notin @($me,'S-1-5-18')});"
        "if($bad.Count -ne 0){exit 2}"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **{k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"},
            "V11_CAPTURE_ACL_PATH": str(directory.resolve()),
        },
        capture_output=True,
        timeout=15,
    )
    if result.returncode:
        raise DomainError("evaluation_private_directory_required")


@router.get("/{command}/evidence/{artifact}")
@limiter.limit("120/minute")
async def evidence(
    request: Request,
    command: str,
    artifact: Literal["intent", "input", "answer", "validation", "sync_timing", "analyze_timing"],
    current: Actor,
    runtime: Backend,
) -> dict:
    """This private router is never mounted on the ordinary business application."""
    if (
        not settings.LOCAL_MVP_ONLY
        or not request.client
        or request.client.host not in {"127.0.0.1", "::1", "testclient"}
    ):
        raise DomainError("local_object_not_found")
    if runtime.evaluation is None:
        raise DomainError("local_object_not_found")
    return await runtime.evaluation.read(runtime.store, current, command, artifact)
