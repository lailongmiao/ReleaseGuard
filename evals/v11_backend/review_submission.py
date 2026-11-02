"""Private, immutable review packets and atomic operator submissions; no Provider calls."""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from evals.v11_backend.http_evidence import verify_windows_acl

LIMIT = 8 * 1024 * 1024


class ReviewSubmissionError(ValueError):
    """Only fixed safe error codes escape the review channel."""


class Decision(BaseModel):
    """The operator supplies judgments, never a hash or a replacement answer."""

    model_config = ConfigDict(extra="forbid", strict=True)
    approved: bool
    critical_error: bool
    score: int = Field(ge=1, le=5)
    claims_supported: bool
    body_supported: bool
    unknown_correct: bool
    citation_identity_valid: bool
    notes: str = Field(min_length=20, max_length=2000)
    evidence_read: bool
    reviewer: str = Field(pattern=r"^[a-z0-9_-]{3,80}$")


def private(path: Path) -> None:
    """Check the current operator ACL on every read/publication, including files."""
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ReviewSubmissionError("review_private_path_invalid")
    target = path if path.exists() else path.parent
    if not target.exists():
        raise ReviewSubmissionError("review_private_path_missing")
    if os.name == "nt":
        try:
            verify_windows_acl(target)
        except DomainError as error:
            raise ReviewSubmissionError("review_private_path_permissions") from error
    elif target.stat().st_mode & 0o077:
        raise ReviewSubmissionError("review_private_path_permissions")


def read(path: Path) -> dict:
    """Read bounded JSON without echoing failed input or private contents."""
    private(path)
    try:
        if path.stat().st_size > LIMIT:
            raise ReviewSubmissionError("review_file_too_large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReviewSubmissionError("review_json_invalid") from error
    except OSError as error:
        raise ReviewSubmissionError("review_file_unreadable") from error
    if not isinstance(value, dict):
        raise ReviewSubmissionError("review_object_required")
    return value


def publish(path: Path, payload: dict) -> None:
    """Fsync and validate a private temporary file, then atomically link without overwrite."""
    private(path.parent)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(data) > LIMIT:
        raise ReviewSubmissionError("review_file_too_large")
    fd, temporary = tempfile.mkstemp(prefix=".review-", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if read(Path(temporary)) != payload:
            raise ReviewSubmissionError("review_temporary_validation_failed")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReviewSubmissionError("review_record_exists") from error
    finally:
        Path(temporary).unlink(missing_ok=True)


def signature(payload: dict, key: bytes) -> str:
    """Bind private review metadata, not a license to send any model request."""
    return hmac.new(key, canonical_hash(payload).encode(), hashlib.sha256).hexdigest()


def packet_path(directory: Path, stamp: str) -> Path:
    """A stamp is only a path-safe locator; never the content hash."""
    if not re.fullmatch(r"[a-f0-9]{64}", stamp):
        raise ReviewSubmissionError("review_stamp_invalid")
    return directory / (stamp + ".review.snapshot.json")


def key_bytes(directory: Path) -> bytes:
    """The local operator key inherits the private directory ACL."""
    path = directory / "review-key.private.json"
    try:
        return bytes.fromhex(read(path)["key"])
    except (KeyError, ValueError) as error:
        raise ReviewSubmissionError("review_key_invalid") from error


def verify_evidence(slot: dict) -> None:
    """Bind retained input, output, citations and completed commands without new reads."""
    try:
        evidence = slot["evaluation_evidence"]
        captured, answer = evidence["input"], evidence["answer"]
        analysis = slot["analysis"]
        user = json.loads(captured["user"])
        commands = slot["command_evidence"]
        if (
            not captured["review_complete"]
            or not answer["review_complete"]
            or captured["transformations"]
            or answer["transformations"]
            or captured["input_hash"] != canonical_hash(captured["user"])
            or answer["original_parsed_hash"] != canonical_hash(answer["answer"])
            or slot["input"]["captured"] != captured
            or user["query"] != slot["input"]["query"]
            or evidence["validation"]["status"] != "contract_passed"
        ):
            raise ReviewSubmissionError("review_evidence_incomplete_or_changed")
        wire = answer["answer"]
        if analysis["output"] != {k: v for k, v in wire.items() if k not in {"comparisons", "comparison_contract"}}:
            raise ReviewSubmissionError("review_answer_binding_mismatch")
        if analysis["telemetry"]["comparisons"] != wire["comparisons"]:
            raise ReviewSubmissionError("review_comparison_binding_mismatch")
        citations = {c["citation_id"]: c for c in analysis["citations"]}
        if (
            set(captured["citation_ids"]) != set(citations)
            or len(user["sources"]) != len(citations)
            or {s["citation_id"] for s in user["sources"]} != set(citations)
        ):
            raise ReviewSubmissionError("review_citation_binding_mismatch")
        for source in user["sources"]:
            citation = citations[source["citation_id"]]
            if any(source[k] != citation.get(k) for k in source if k != "line_roles"):
                raise ReviewSubmissionError("review_citation_binding_mismatch")
            if canonical_hash(citation["text"]) != citation["text_hash"]:
                raise ReviewSubmissionError("review_citation_binding_mismatch")
            if citation["snapshot_id"] != analysis["snapshot_id"]:
                raise ReviewSubmissionError("review_citation_binding_mismatch")
        if (
            analysis["object_id"] != slot["object_id"]
            or captured["object_id"] != slot["object_id"]
            or analysis["revision"] != slot["repeat"]
            or analysis["snapshot_id"] != slot["input"]["snapshot"]["snapshot_id"]
            or slot["rag_input"]["snapshot_id"] != analysis["snapshot_id"]
            or slot["rag_input"]["generation_id"] != analysis["generation_id"]
            or slot["rag_input"]["command_id"] != "rag:" + slot["stamp"]
            or commands["analysis"]["result"]["analysis_id"] != analysis["analysis_id"]
            or any(commands[k]["status"] != "completed" for k in ("analysis", "sync"))
            or slot["rag_command_evidence"]["status"] != "completed"
            or slot["rag_command_evidence"]["result"] != slot["rag_result"]
            or sum(a["attempt"] == "analysis-1" for a in commands["analysis"]["attempts"]) != 1
        ):
            raise ReviewSubmissionError("review_command_binding_mismatch")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReviewSubmissionError("review_evidence_fields_missing") from error


def prepare(directory: Path, slot: dict, batch: str, dataset_hash: str, expires_at: datetime) -> str:
    """Freeze the full completed slot before wait/status updates; never replace a packet."""
    if slot.get("status") != "completed" or slot.get("rag_status") != "completed":
        raise ReviewSubmissionError("review_completed_evidence_required")
    if not slot.get("hard_gates_passed") or not slot.get("evaluation_evidence"):
        raise ReviewSubmissionError("review_complete_evidence_required")
    verify_evidence(slot)
    stamp = slot["stamp"]
    if stamp != canonical_hash([batch, dataset_hash, slot["case_id"], slot["repeat"]]):
        raise ReviewSubmissionError("review_position_binding_mismatch")
    if not (directory / "review-key.private.json").exists():
        publish(directory / "review-key.private.json", {"key": secrets.token_hex(32)})
    payload = {
        "version": "v11-review-snapshot.2",
        "batch": batch,
        "dataset_hash": dataset_hash,
        "expires_at": expires_at.isoformat(),
        "position": slot,
    }
    digest = canonical_hash(payload)
    publish(
        packet_path(directory, stamp),
        {"payload": payload, "hash": digest, "signature": signature(payload, key_bytes(directory))},
    )
    return digest


def load_packet(directory: Path, stamp: str, expected: str | None = None) -> dict:
    """Authenticate the immutable packet and explicit local revocation/expiry."""
    private(directory)
    if (directory / "review.revoked").exists() or (directory / (stamp + ".review.revoked")).exists():
        raise ReviewSubmissionError("review_authorization_revoked")
    path = packet_path(directory, stamp)
    if not path.exists():
        raise ReviewSubmissionError("review_position_not_found")
    value = read(path)
    try:
        payload = value["payload"]
        digest = canonical_hash(payload)
        valid = hmac.compare_digest(value["signature"], signature(payload, key_bytes(directory)))
        position = payload["position"]
        if not valid or digest != value["hash"] or (expected is not None and digest != expected):
            raise ReviewSubmissionError("review_snapshot_changed")
        if (
            payload["version"] != "v11-review-snapshot.2"
            or stamp
            != canonical_hash([payload["batch"], payload["dataset_hash"], position["case_id"], position["repeat"]])
            or position["stamp"] != stamp
        ):
            raise ReviewSubmissionError("review_position_binding_mismatch")
        if datetime.now(UTC) >= datetime.fromisoformat(payload["expires_at"]):
            raise ReviewSubmissionError("review_authorization_expired")
    except (KeyError, TypeError) as error:
        raise ReviewSubmissionError("review_packet_invalid") from error
    return value


def validate_decision(value: dict) -> Decision:
    """Reject missing acknowledgments and contradictory approvals."""
    try:
        decision = Decision.model_validate(value)
    except ValueError as error:
        raise ReviewSubmissionError("review_decision_invalid") from error
    if not decision.evidence_read:
        raise ReviewSubmissionError("review_evidence_read_required")
    if decision.approved and (decision.critical_error or not decision.citation_identity_valid):
        raise ReviewSubmissionError("review_decision_conflict")
    return decision


def validate_submission(value: dict, packet: dict, key: bytes) -> Decision:
    """Validate a proposal against the exact immutable packet, without auto-approving."""
    try:
        payload = value["payload"]
        if not hmac.compare_digest(value["signature"], signature(payload, key)):
            raise ReviewSubmissionError("review_submission_tampered")
        if payload["snapshot_hash"] != packet["hash"] or payload["batch"] != packet["payload"]["batch"]:
            raise ReviewSubmissionError("review_position_hash_mismatch")
        return validate_decision(payload["decision"])
    except (KeyError, TypeError) as error:
        raise ReviewSubmissionError("review_submission_invalid") from error


def submit(directory: Path, stamp: str, decision: dict) -> Path:
    """Stage an operator proposal; live object/source ACL is rechecked before consumption."""
    packet = load_packet(directory, stamp)
    validate_decision(decision)
    lock = directory / (stamp + ".review.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ReviewSubmissionError("review_submission_busy") from error
    try:
        os.close(descriptor)
        if (directory / (stamp + ".review.accepted.json")).exists():
            raise ReviewSubmissionError("review_already_consumed")
        key = key_bytes(directory)
        for path in directory.glob(stamp + ".review.proposal.*.json"):
            if path.name.endswith(".rejected.json"):
                continue
            try:
                validate_submission(read(path), packet, key)
            except ReviewSubmissionError:
                continue
            raise ReviewSubmissionError("review_duplicate_or_conflict")
        payload = {"batch": packet["payload"]["batch"], "snapshot_hash": packet["hash"], "decision": decision}
        value = {"payload": payload, "signature": signature(payload, key)}
        validate_submission(value, packet, key)
        load_packet(directory, stamp, packet["hash"])
        path = directory / (stamp + ".review.proposal." + secrets.token_hex(8) + ".json")
        publish(path, value)
        return path
    finally:
        lock.unlink()


async def wait_review(
    directory: Path, stamp: str, expected: str, reauthorize: Callable[[], Awaitable[None]], *, interval: float = 1
) -> bool:
    """Invalid proposals block, never trigger the next send; expiry/revocation fail closed."""
    while True:
        packet = load_packet(directory, stamp, expected)
        if (directory / (stamp + ".review.accepted.json")).exists():
            raise ReviewSubmissionError("review_already_consumed")
        legacy = directory / (stamp + ".review.json")
        legacy_error = directory / (stamp + ".review.legacy-rejected.json")
        if legacy.exists() and not legacy_error.exists():
            publish(legacy_error, {"code": "review_legacy_submission_not_accepted", "snapshot_hash": expected})
        valid = []
        for path in sorted(directory.glob(stamp + ".review.proposal.*.json")):
            if path.name.endswith(".rejected.json"):
                continue
            try:
                proposal = read(path)
                decision = validate_submission(proposal, packet, key_bytes(directory))
                valid.append((path, decision, canonical_hash(proposal)))
            except ReviewSubmissionError as error:
                error_path = directory / (path.name + ".rejected.json")
                if not error_path.exists():
                    publish(error_path, {"code": str(error), "proposal": path.name, "snapshot_hash": expected})
        if len(valid) > 1:
            raise ReviewSubmissionError("review_duplicate_or_conflict")
        if valid:
            remaining = (datetime.fromisoformat(packet["payload"]["expires_at"]) - datetime.now(UTC)).total_seconds()
            try:
                async with asyncio.timeout(max(0, remaining)):
                    await reauthorize()
            except TimeoutError as error:
                raise ReviewSubmissionError("review_permission_check_timeout") from error
            load_packet(directory, stamp, expected)
            path, decision, proposal_hash = valid[0]
            if canonical_hash(read(path)) != proposal_hash:
                raise ReviewSubmissionError("review_submission_changed_during_permission_check")
            publish(
                directory / (stamp + ".review.accepted.json"),
                {
                    "snapshot_hash": expected,
                    "proposal_hash": proposal_hash,
                    "approved": decision.approved,
                    "critical_error": decision.critical_error,
                    "permission_rechecked": True,
                },
            )
            return decision.approved and not decision.critical_error
        await asyncio.sleep(interval)


def main() -> int:
    """Submit a reviewed decision locally; this entry cannot send or resume a batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = submit(args.directory, args.stamp, read(args.decision))
        Console().print({"proposal": path.name, "status": "pending_live_permission_check"})
        return 0
    except ReviewSubmissionError as error:
        Console().print({"error": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
