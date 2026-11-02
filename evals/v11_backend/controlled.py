"""One explicitly permitted paid evaluation, never a business analysis or recovery."""

import asyncio
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.logging import logger
from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import ReviewCitation
from app.services.llm.contracts import LLMRole
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_policy import citation_line_roles
from app.services.v11_provider import SYSTEM, ReviewProvider
from app.services.v11_rejection import ReviewProviderFailure, ReviewValidationError, redact_text
from app.services.v11_storage import V11Store
from evals.v1_backend.provenance import code_state
from evals.v11_backend import supplement
from evals.v11_backend.structured_evidence import STRUCTURED_LIMIT, structured_evidence

EVALUATION_ID = "v11-d1-evidence-eval-01"
COMMAND_ID = "eval:v11-d1-evidence-01"
LEDGER = "v11-real-d8aef22d29674479a1e66d1f06d08f89"
TIMEOUT_EVALUATION_ID = "v11-d1-timeout60-eval-01"
TIMEOUT_COMMAND_ID = "eval:v11-d1-timeout60-01"
FINAL_EVALUATION_ID = "v11-d1-contract2-eval-01"
FINAL_COMMAND_ID = "eval:v11-d1-contract2-01"
BEHAVIOR_EVALUATION_ID = "v11-d1-contract2-behavior-eval-01"
BEHAVIOR_COMMAND_ID = "eval:v11-d1-contract2-behavior-01"
BEHAVIOR_PAYLOAD_HASH = "1a6d01fbab6c28898471963ee6de7634f819f04c2f21f87c46410c21e678f6f5"


class EvaluationPermit(BaseModel):
    """External operator approval is required; this module never issues signatures."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evaluation_id: Literal[
        "v11-d1-evidence-eval-01",
        "v11-d1-timeout60-eval-01",
        "v11-d1-contract2-eval-01",
        "v11-d1-contract2-behavior-eval-01",
        "v11-contract2-B-acceptance60-01",
    ]
    acceptance_increment_baseline: Decimal | None = None
    retain_unvalidated: bool = False
    call_timeout_seconds: Literal[30, 60] = 30
    command_timeout_seconds: Literal[120] = 120
    case: Literal["A", "B"]
    ledger_id: Literal["v11-real-d8aef22d29674479a1e66d1f06d08f89"]
    actor_id: int
    source_object: str
    payload_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    source_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    approval_reference: Annotated[str, Field(min_length=1, max_length=200)]
    expires_at: datetime
    max_calls: Literal[1]
    max_cny: Annotated[Decimal, Field(gt=0, le=Decimal("0.131072"))]
    preserve_old_unknown: Literal[True]
    no_business_adoption: Literal[True]

    @model_validator(mode="after")
    def timeout_scope(self) -> "EvaluationPermit":
        """The exceptional wait belongs only to the explicitly approved B experiment."""
        if (self.evaluation_id in {TIMEOUT_EVALUATION_ID, supplement.EVALUATION_ID}) != (
            self.call_timeout_seconds == 60
        ):
            raise ValueError("evaluation_timeout_identity_mismatch")
        if self.call_timeout_seconds == 60 and (self.case != "B" or self.max_cny > Decimal("0.130936")):
            raise ValueError("evaluation_timeout_scope_invalid")
        if self.retain_unvalidated != (
            self.evaluation_id in {FINAL_EVALUATION_ID, BEHAVIOR_EVALUATION_ID, supplement.EVALUATION_ID}
        ):
            raise ValueError("evaluation_evidence_scope_invalid")
        if self.evaluation_id in {BEHAVIOR_EVALUATION_ID, supplement.EVALUATION_ID} and (
            self.case != "B" or self.payload_hash != BEHAVIOR_PAYLOAD_HASH or self.max_cny > Decimal("0.130804")
        ):
            raise ValueError("evaluation_behavior_scope_invalid")
        if (self.evaluation_id == supplement.EVALUATION_ID) != (self.acceptance_increment_baseline is not None):
            raise ValueError("evaluation_acceptance_baseline_required")
        if self.acceptance_increment_baseline is not None and not (
            self.acceptance_increment_baseline.is_finite() and self.acceptance_increment_baseline > 0
        ):
            raise ValueError("evaluation_acceptance_baseline_invalid")
        return self


def verify_permit(permit: EvaluationPermit, signature: str, approval_key: bytes) -> None:
    """Fail closed without a separate operator key, valid signature and unexpired approval."""
    if len(approval_key) < 32:
        raise DomainError("evaluation_explicit_approval_required")
    expected = hmac.new(approval_key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise DomainError("evaluation_approval_invalid")
    if permit.expires_at.tzinfo is None or permit.expires_at <= datetime.now(UTC):
        raise DomainError("evaluation_approval_expired")


def save_new(path: Path, report: dict) -> None:
    """Create a new owner-readable artifact; never overwrite an earlier attempt."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


async def evaluate(
    payload: dict,
    permit: EvaluationPermit,
    signature: str,
    approval_key: bytes,
    actor: LocalActor,
    store: V11Store,
    provider: ReviewProvider,
    private_directory: Path,
) -> dict:
    """Consume one durable command; only evaluation command/budget/audit rows may change."""
    trusted_key = os.environ.get("V11_EVAL_OPERATOR_APPROVAL_KEY", "").encode()
    if len(trusted_key) < 32 or not hmac.compare_digest(trusted_key, approval_key):
        raise DomainError("evaluation_explicit_approval_required")
    verify_permit(permit, signature, approval_key)
    if (
        actor.user_id != permit.actor_id
        or canonical_hash(payload) != permit.payload_hash
        or code_state()["source_hash"] != permit.source_hash
        or payload["system"] != SYSTEM
        or payload["case"] != permit.case
        or payload["object_id"] != permit.source_object
        or provider.store is not store
    ):
        raise DomainError("evaluation_frozen_identity_mismatch")
    if not private_directory.is_dir() or private_directory.is_symlink():
        raise DomainError("evaluation_private_directory_required")
    # Windows permissions come from the operator-created private directory ACL.
    # No HTTP endpoint exposes these files. Raw headers/reasoning are never written.
    citations = [ReviewCitation.model_validate(c) for c in payload["citations"]]
    citation_line_roles([c for c in citations if c.kind == "diff"], payload["facts"], permit.source_object)
    amount = (Decimal(len((SYSTEM + payload["user"]).encode()) + 64) * 2 + Decimal(8192) * 8) / 1000000
    if amount > permit.max_cny:
        raise DomainError("evaluation_reservation_exceeds_approval")
    if provider.config(LLMRole.CHAT).model != "deepseek-flash":
        raise DomainError("evaluation_model_mismatch")
    await store.require_real_budget()
    # The configured ledger must match the signed identity, not merely be reconciled.
    identity = await store.real_budget_identity(actor)
    if identity["ledger_id"] != permit.ledger_id:
        raise DomainError("evaluation_ledger_mismatch")
    await store.connection.authenticate(actor.session_id, actor.expires_at)
    obj = await store.object_row(permit.source_object, actor)
    if obj["owner_user_id"] != actor.user_id:
        raise DomainError("local_owner_required")
    evaluation_id = permit.evaluation_id
    command_id = {
        supplement.EVALUATION_ID: supplement.COMMAND_ID,
        TIMEOUT_EVALUATION_ID: TIMEOUT_COMMAND_ID,
        FINAL_EVALUATION_ID: FINAL_COMMAND_ID,
        BEHAVIOR_EVALUATION_ID: BEHAVIOR_COMMAND_ID,
    }.get(evaluation_id, COMMAND_ID)
    digest = canonical_hash([evaluation_id, permit.payload_hash, permit.actor_id])
    binding = {}
    if evaluation_id == supplement.EVALUATION_ID:
        assert permit.acceptance_increment_baseline is not None
        binding = await supplement.claim(
            store, actor, permit.source_object, digest, permit.acceptance_increment_baseline
        )
    else:
        prior = await store.claim_review(actor, command_id, digest, permit.source_object, "development", digest)
        if prior is not None:
            raise DomainError("evaluation_consumed_no_resend")
    # If persistence fails here, the durable claim remains held and nothing is sent.
    save_new(
        private_directory / "started.json",
        {"evaluation_id": evaluation_id, "payload_hash": permit.payload_hash, **binding},
    )
    verify_permit(permit, signature, approval_key)
    report: dict = {
        **binding,
        "evaluation_id": evaluation_id,
        "case": permit.case,
        "not_business_analysis": True,
        "call_timeout_seconds": permit.call_timeout_seconds,
        "command_timeout_seconds": 120,
    }
    captured = False

    async def capture(parsed: object) -> None:
        """Only the verified signed evaluation can opt into this private pre-validation sink."""
        nonlocal captured
        await store.connection.authenticate(actor.session_id, actor.expires_at)
        current = await store.object_row(permit.source_object, actor)
        if current["owner_user_id"] != actor.user_id:
            raise DomainError("local_owner_required")
        artifact = structured_evidence(parsed, {c.citation_id for c in citations})
        artifact.update(
            owner_id=actor.user_id,
            source_object=permit.source_object,
            evaluation_id=evaluation_id,
            payload_hash=permit.payload_hash,
            source_hash=permit.source_hash,
        )
        try:
            save_new(private_directory / "unvalidated.json", artifact)
        except Exception:
            raise DomainError("evaluation_unvalidated_not_saved_no_resend") from None
        captured = True

    try:
        async with asyncio.timeout(permit.command_timeout_seconds):
            output, telemetry = await provider.review(
                payload["user"],
                citations,
                actor,
                command_id,
                evaluation_id,
                evaluation_timeout_seconds=60 if permit.call_timeout_seconds == 60 else None,
                capture_unvalidated=capture if permit.retain_unvalidated else None,
            )
        if permit.retain_unvalidated and not captured:
            raise DomainError("evaluation_unvalidated_missing_no_resend")
        report.update(status="completed", output=output.model_dump(mode="json"), telemetry=telemetry)
    except TimeoutError:
        report.update(
            status="unknown",
            reason="evaluation_command_timeout",
            diagnostic={
                "stage": "command",
                "timeout_source": "command_deadline",
                "http_status": None,
                "provider_request_id": None,
                "provider_processing": "unknown",
                "usage_known": False,
            },
        )
    except DomainError as error:
        report.update(
            status="unknown"
            if any(word in error.code for word in ("unknown", "timeout", "unavailable"))
            else "failed",
            reason=error.code,
        )
        if isinstance(error, (ReviewProviderFailure, ReviewValidationError)):
            report["diagnostic"] = error.diagnostic
    # Arbitrary exceptions/cancellation leave the claim reserved: never mark as safely retryable.
    result = {k: report[k] for k in ("evaluation_id", "case", "status", "not_business_analysis")}
    result.update(binding)
    await store.complete(actor, command_id, result)
    try:
        await store.connection.authenticate(actor.session_id, actor.expires_at)
        current = await store.object_row(permit.source_object, actor)
        if current["owner_user_id"] != actor.user_id:
            raise DomainError("local_owner_required")
    except DomainError:
        report = {**result, "body_withheld_authorization": True}
    report.update(source_object=permit.source_object, owner_id=actor.user_id)
    report["unvalidated_evidence_saved"] = captured
    try:
        save_new(private_directory / "result.json", redact_report(report))
    except Exception:
        logger.exception("evaluation_result_artifact_not_saved")
        raise DomainError("evaluation_result_artifact_not_saved_no_resend") from None
    return result


async def read_evaluation(
    directory: Path,
    source_object: str,
    actor: LocalActor,
    store: V11Store,
    artifact: Literal["result", "unvalidated"] = "result",
) -> dict:
    """Reauthorize local result reads; an exported file itself cannot be remotely revoked."""
    await store.connection.authenticate(actor.session_id, actor.expires_at)
    obj = await store.object_row(source_object, actor)
    if obj["owner_user_id"] != actor.user_id:
        raise DomainError("local_object_not_found")
    path = directory / (artifact + ".json")
    if (
        artifact not in {"result", "unvalidated"}
        or directory.is_symlink()
        or path.is_symlink()
        or path.stat().st_size > STRUCTURED_LIMIT
    ):
        raise DomainError("evaluation_artifact_invalid")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("owner_id") != actor.user_id or report.get("source_object") != source_object:
        raise DomainError("local_object_not_found")
    return report


def redact_report(report: dict) -> dict:
    """Redact model prose in the private artifact without changing validation or adoption."""
    result = json.loads(json.dumps(report))
    output = result.get("output")
    if output:
        result["original_output_hash"] = canonical_hash(output)
        output["summary"] = redact_text(output["summary"], limit=4000)
        for section in ("observations", "risks", "validation_suggestions"):
            for item in output[section]:
                item["text"] = redact_text(item["text"])
        output["unknowns"] = [redact_text(value) for value in output["unknowns"]]
        for item in result.get("telemetry", {}).get("comparisons", []):
            for key in ("before", "after"):
                item[key] = redact_text(item[key])
    result["body_redacted_not_original_bytes"] = True
    return result
