"""One fixed evidence3 segment; immutable provenance and fail-closed PostgreSQL CAS."""

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.local_mvp import LocalActor
from app.schemas.v11_backend import Comparison, ReviewOutput
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_evaluation_policy import DATASET_HASH, FORMAL_BATCH, LINK_CONTRACT
from app.services.v11_recovery_guard import MARKER, POSITION_KEYS, RECOVERY_ID, STAMPS, VERSION
from app.services.v11_storage import V11Store
from evals.v11_backend.contracts import Dataset
from evals.v11_backend.review_submission import ReviewSubmissionError, publish, read, verify_evidence


class Manifest(BaseModel):
    """A content address is not a sending permit; live activation is separately required."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = VERSION
    recovery_id: str = RECOVERY_ID
    batch_id: str = FORMAL_BATCH
    dataset_hash: str = DATASET_HASH
    old_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    supplemental_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    supplemental_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    retained_hashes: tuple[str, str]
    retained_result_hashes: dict[str, str] = Field(default_factory=dict)
    remaining: tuple[str, ...]
    compatibility_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    def check(self) -> None:
        """No caller-defined batch, subset, repeat, replacement or third retained sample."""
        if (
            (self.version, self.recovery_id, self.batch_id, self.dataset_hash)
            != (VERSION, RECOVERY_ID, FORMAL_BATCH, DATASET_HASH)
            or self.remaining != STAMPS[2:]
            or len(set(self.retained_hashes)) != 2
        ):
            raise DomainError("recovery_fixed_manifest_required")


def build_manifest(old_path: Path, supplemental_path: Path, compatibility: dict) -> Manifest:
    """Read only individually authorized bounded files; never scan a private directory."""
    old = read(old_path)
    # The supplemental report is public evidence, not a private answer directory.
    supplemental_bytes = supplemental_path.read_bytes()
    supplemental = json.loads(supplemental_bytes)
    if (
        not compatibility.get("comparable")
        or old.get("authorization_reference") != FORMAL_BATCH
        or old.get("completed") is not False
    ):
        raise DomainError("recovery_old_batch_or_comparability_invalid")
    slots = old["positions"]
    if len(slots) != 2 or [(s["case_id"], s["repeat"]) for s in slots] != list(POSITION_KEYS[:2]):
        raise DomainError("recovery_exact_two_retained_required")
    if canonical_hash(Dataset.model_validate(old["dataset"])) != DATASET_HASH:
        raise DomainError("recovery_dataset_drift")
    for slot, stamp in zip(slots, STAMPS[:2], strict=True):
        if slot["stamp"] != stamp:
            raise DomainError("recovery_retained_identity_mismatch")
        verify_evidence(slot)
    if supplemental.get("files_sha256", {}).get("batch.json") != hashlib.sha256(old_path.read_bytes()).hexdigest():
        raise DomainError("recovery_supplemental_binding_mismatch")
    if [p["archived_position_hash"] for p in supplemental["positions"]] != [canonical_hash(s) for s in slots]:
        raise DomainError("recovery_supplemental_binding_mismatch")
    if any(p["semantic_review"]["critical_error"] or not p["evidence_complete"] for p in supplemental["positions"]):
        raise DomainError("recovery_retained_review_failed")
    result = Manifest(
        old_report_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        supplemental_sha256=hashlib.sha256(supplemental_bytes).hexdigest(),
        supplemental_hash=canonical_hash(supplemental),
        retained_hashes=(canonical_hash(slots[0]), canonical_hash(slots[1])),
        retained_result_hashes={
            prefix + s["stamp"]: canonical_hash(record["result"])
            for s in slots
            for prefix, record in (
                ("sync:", s["command_evidence"]["sync"]),
                ("analysis:", s["command_evidence"]["analysis"]),
                ("rag:", s["rag_command_evidence"]),
            )
        },
        remaining=STAMPS[2:],
        compatibility_hash=canonical_hash(compatibility),
    )
    result.check()
    return result


async def activate(store: V11Store, actor: LocalActor, manifest: Manifest, expires_at: datetime) -> None:
    """Atomic retirement plus unique activation; failures never delete an earlier claim.

    Call only after a separately signed permission and live runtime/budget preflight.
    This function neither creates sessions nor issues any Provider authorization.
    """
    manifest.check()
    now = datetime.now(UTC)
    if not store.evaluation_recovery_boot_id:
        raise DomainError("recovery_server_boot_binding_required")
    if expires_at.tzinfo is None or not now < expires_at <= now + timedelta(hours=8):
        raise DomainError("recovery_eight_hour_limit")
    await store.connection.authenticate(actor.session_id, actor.expires_at)
    for source in ("markdown-rules", "review-github-checkout", "review-github-setup-python"):
        await store.source(source, actor, owner=True)
    async with await store.connection.connect() as db, db.transaction():
        await db.execute("SELECT pg_advisory_xact_lock(71800801)")
        await db.execute("SELECT pg_advisory_xact_lock(71801110)")
        if await (await db.execute("SELECT 1 FROM v1_commands WHERE command_id=%s", (MARKER,))).fetchone():
            raise DomainError("recovery_already_claimed_no_restart")
        if await (
            await db.execute(
                "SELECT 1 FROM v1_commands WHERE contract_version=%s AND result->>'batch_id'=%s LIMIT 1",
                (LINK_CONTRACT, FORMAL_BATCH),
            )
        ).fetchone():
            raise DomainError("recovery_overlap_link_already_consumed")
        remaining_commands = [prefix + s for s in manifest.remaining for prefix in ("sync:", "analysis:", "rag:")]
        for table in ("v1_commands", "v1_budget", "v11_analyses"):
            if await (
                await db.execute(f"SELECT 1 FROM {table} WHERE command_id=ANY(%s) LIMIT 1", (remaining_commands,))
            ).fetchone():
                raise DomainError("recovery_remaining_position_consumed_or_uncertain")
        # Presence of a budget row counts even with zero charge; status alone cannot establish zero send.
        for stamp in STAMPS[:2]:
            for prefix in ("sync:", "analysis:", "rag:"):
                row = await (
                    await db.execute(
                        "SELECT owner_id,status,result FROM v1_commands WHERE command_id=%s", (prefix + stamp,)
                    )
                ).fetchone()
                if (
                    not row
                    or row["owner_id"] != actor.user_id
                    or row["status"] != "completed"
                    or row["result"].get("status") != "completed"
                ):
                    raise DomainError("recovery_retained_command_not_completed")
                if manifest.retained_result_hashes.get(prefix + stamp) != canonical_hash(row["result"]):
                    raise DomainError("recovery_retained_result_changed")
            attempts = await (
                await db.execute(
                    "SELECT attempt,usage FROM v1_budget WHERE owner_id=%s AND command_id=%s ORDER BY attempt",
                    (actor.user_id, "analysis:" + stamp),
                )
            ).fetchall()
            if [r["attempt"] for r in attempts] != ["analysis-1", "query-1-batch-0"] or any(
                not r["usage"] for r in attempts
            ):
                raise DomainError("recovery_retained_attempt_mismatch")
        state = {
            "state": "running",
            "manifest_hash": canonical_hash(manifest),
            "manifest": manifest.model_dump(mode="json"),
            "expires_at": expires_at.isoformat(),
            "allowed_commands": remaining_commands,
            "next_index": 0,
            "active_stamp": None,
            "old_permit_retired": True,
            "server_boot_id": store.evaluation_recovery_boot_id,
        }
        await db.execute(
            "INSERT INTO v1_commands(owner_id,command_id,digest,resource,status,contract_version,budget_category,result) "
            "VALUES(%s,%s,%s,%s,'completed',%s,'acceptance',%s)",
            (actor.user_id, MARKER, canonical_hash(manifest), FORMAL_BATCH, VERSION, Jsonb(state)),
        )


async def transition(
    store: V11Store,
    actor: LocalActor,
    manifest: Manifest,
    stamp: str,
    *,
    completed_hash: str | None = None,
) -> None:
    """CAS before each position and after its immutable reviewed output; orphan means stop."""
    manifest.check()
    await store.connection.authenticate(actor.session_id, actor.expires_at)
    for source in ("markdown-rules", "review-github-checkout", "review-github-setup-python"):
        await store.source(source, actor, owner=True)
    async with await store.connection.connect() as db, db.transaction():
        await db.execute("SELECT pg_advisory_xact_lock(71800801)")
        await db.execute("SELECT pg_advisory_xact_lock(71801110)")
        row = await (
            await db.execute("SELECT owner_id,result FROM v1_commands WHERE command_id=%s FOR UPDATE", (MARKER,))
        ).fetchone()
        if not row or row["owner_id"] != actor.user_id:
            raise DomainError("recovery_not_activated")
        state = row["result"]
        index = state["next_index"]
        if (
            state["state"] != "running"
            or state["manifest_hash"] != canonical_hash(manifest)
            or datetime.now(UTC) >= datetime.fromisoformat(state["expires_at"])
        ):
            raise DomainError("recovery_stopped_or_expired")
        if index >= 78 or manifest.remaining[index] != stamp:
            raise DomainError("recovery_position_out_of_order")
        if completed_hash is None:
            if state["active_stamp"] is not None:
                raise DomainError("recovery_orphan_position_no_restart")
            commands = [prefix + stamp for prefix in ("sync:", "analysis:", "rag:")]
            for table in ("v1_commands", "v1_budget", "v11_analyses"):
                if await (
                    await db.execute(f"SELECT 1 FROM {table} WHERE command_id=ANY(%s) LIMIT 1", (commands,))
                ).fetchone():
                    raise DomainError("recovery_position_consumed_or_uncertain")
            state["active_stamp"] = stamp
        else:
            if state["active_stamp"] != stamp or len(completed_hash) != 64:
                raise DomainError("recovery_completion_mismatch")
            for prefix in ("sync:", "analysis:", "rag:"):
                command = await (
                    await db.execute(
                        "SELECT status,result FROM v1_commands WHERE owner_id=%s AND command_id=%s",
                        (actor.user_id, prefix + stamp),
                    )
                ).fetchone()
                if not command or command["status"] != "completed" or command["result"].get("status") != "completed":
                    raise DomainError("recovery_position_not_completed")
            for prefix, expected in (("analysis:", ["analysis-1", "query-1-batch-0"]), ("rag:", ["query-1-batch-0"])):
                attempts = await (
                    await db.execute(
                        "SELECT attempt,usage FROM v1_budget WHERE owner_id=%s AND command_id=%s ORDER BY attempt",
                        (actor.user_id, prefix + stamp),
                    )
                ).fetchall()
                if [r["attempt"] for r in attempts] != expected or any(not r["usage"] for r in attempts):
                    raise DomainError("recovery_attempts_incomplete")
            state.setdefault("completed", {})[stamp] = completed_hash
            state["active_stamp"] = None
            state["next_index"] += 1
            if state["next_index"] == 78:
                state["state"] = "completed"
        await db.execute("UPDATE v1_commands SET result=%s WHERE command_id=%s", (Jsonb(state), MARKER))


async def stop(store: V11Store, actor: LocalActor, manifest: Manifest) -> None:
    """Never release an active slot, claim or reservation; stopping cannot grant authority."""
    async with await store.connection.connect() as db, db.transaction():
        await db.execute("SELECT pg_advisory_xact_lock(71801110)")
        await db.execute(
            "UPDATE v1_commands SET result=jsonb_set(result,'{state}','\"stopped\"') "
            "WHERE command_id=%s AND owner_id=%s AND digest=%s AND result->>'state'='running'",
            (MARKER, actor.user_id, canonical_hash(manifest)),
        )


def aggregate(manifest: Manifest, old: dict, supplemental: dict, segment: dict) -> dict:
    """A separate provenance view, never a rewrite or a claim of quality passage."""
    manifest.check()
    if tuple(canonical_hash(s) for s in old["positions"]) != manifest.retained_hashes:
        raise DomainError("recovery_retained_snapshot_changed")
    if segment.get("manifest_hash") != canonical_hash(manifest):
        raise DomainError("recovery_segment_binding_mismatch")
    if [p["archived_position_hash"] for p in supplemental.get("positions", [])] != list(manifest.retained_hashes):
        raise DomainError("recovery_supplemental_binding_mismatch")
    if canonical_hash(supplemental) != manifest.supplemental_hash:
        raise DomainError("recovery_supplemental_changed")
    slots = segment["positions"]
    if [s["stamp"] for s in slots] != list(manifest.remaining[: len(slots)]):
        raise DomainError("recovery_missing_conflicting_or_duplicate_position")
    sources = [{"segment": "historical", "hash": canonical_hash(s)} for s in old["positions"]]
    sources += [{"segment": RECOVERY_ID, "hash": canonical_hash(s)} for s in slots]
    complete_evidence = True
    dataset = Dataset.model_validate_json(Path("evals/datasets/v11_acceptance.json").read_text(encoding="utf-8"))
    if canonical_hash(dataset) != manifest.dataset_hash:
        raise DomainError("recovery_dataset_drift")
    for slot, expected in zip(slots, POSITION_KEYS[2:], strict=False):
        try:
            verify_position(slot, dataset)
            if (slot.get("case_id"), slot.get("repeat")) != expected:
                complete_evidence = False
        except (ReviewSubmissionError, DomainError):
            complete_evidence = False
    return {
        "version": "v11-segmented-provenance.1",
        "manifest_hash": canonical_hash(manifest),
        "expected_positions": 80,
        "expected_rag_positions": 80,
        "actual_positions": len(sources),
        "sources": sources,
        "historical_batch_unchanged": True,
        "supplemental_review_hash": canonical_hash(supplemental),
        "complete": len(sources) == 80
        and complete_evidence
        and segment.get("completed") is True
        and all(
            s.get("status") == "completed"
            and s.get("rag_status") == "completed"
            and s.get("hard_gates_passed") is True
            and s.get("review_approved") is True
            for s in slots
        ),
        "quality_passed": False,
        "requires_original_scorer_and_all_semantic_reviews": True,
    }


def save_manifest(path: Path, manifest: Manifest) -> None:
    """Atomic create-only publication using the existing private review channel."""
    manifest.check()
    publish(path, manifest.model_dump(mode="json"))


class Segment:
    """Sampler lifecycle, using its original HTTP/Graph implementation and commands."""

    def __init__(self, store: V11Store, actor: LocalActor, manifest: Manifest, directory: Path) -> None:
        """Bind one manifest to an already authenticated local store and private directory."""
        manifest.check()
        self.store, self.actor, self.manifest, self.directory = store, actor, manifest, directory

    async def begin(self, stamp: str) -> None:
        """The database, not a local skip list, authorizes the next unused position."""
        await transition(self.store, self.actor, self.manifest, stamp)

    async def finish(self, slot: dict) -> None:
        """Only complete reviewed evidence can advance the CAS cursor."""
        dataset = Dataset.model_validate_json(Path("evals/datasets/v11_acceptance.json").read_text(encoding="utf-8"))
        if canonical_hash(dataset) != self.manifest.dataset_hash:
            raise DomainError("recovery_dataset_drift")
        verify_position(slot, dataset)
        if (
            slot.get("review_approved") is not True
            or slot.get("status") != "completed"
            or slot.get("rag_status") != "completed"
        ):
            raise DomainError("recovery_review_required")
        publish(self.directory / (slot["stamp"] + ".recovery-position.json"), slot)
        await transition(self.store, self.actor, self.manifest, slot["stamp"], completed_hash=canonical_hash(slot))

    async def stop(self) -> None:
        """Cancellation, failed review or disk failure cannot reset the persistent claim."""
        await stop(self.store, self.actor, self.manifest)


def verify_position(slot: dict, dataset: Dataset) -> None:
    """Bind saved evidence back to the original question and source comparison, not a new input."""
    verify_evidence(slot)
    ReviewOutput.model_validate(slot["analysis"]["output"])
    case = next((c for c in dataset.cases if c.case_id == slot.get("case_id")), None)
    if case is None or slot.get("repeat") not in (1, 2):
        raise DomainError("recovery_position_identity_mismatch")
    if slot["stamp"] != canonical_hash([FORMAL_BATCH, DATASET_HASH, case.case_id, slot["repeat"]]):
        raise DomainError("recovery_position_identity_mismatch")
    if (
        slot["input"]["query"] != case.query
        or Comparison.model_validate(slot["input"]["snapshot"]["facts"]["comparison"]) != case.comparison
    ):
        raise DomainError("recovery_position_input_changed")


def scoring_view(manifest: Manifest, old: dict, supplemental: dict, segment: dict) -> dict:
    """New derived report for the unchanged scorer; preserve original failure provenance."""
    provenance = aggregate(manifest, old, supplemental, segment)
    retained = deepcopy(old["positions"])
    for slot, judgment in zip(retained, supplemental["positions"], strict=True):
        verify_evidence(slot)
        if judgment["semantic_review"]["critical_error"] or not judgment["evidence_complete"]:
            raise DomainError("recovery_retained_review_failed")
        slot["historical_status"] = slot["status"]
        slot["status"] = "completed"
        slot["review_approved"] = True
        slot["supplemental_review"] = judgment["semantic_review"]
        if slot["repeat"] == 2:
            slot["review_wait_seconds"] = None
    for slot in segment["positions"]:
        verify_evidence(slot)
    return {
        "version": "v11-segmented-scoring-view.1",
        "dataset": old["dataset"],
        "positions": retained + deepcopy(segment["positions"]),
        "segmented_provenance": provenance,
        "completed": provenance["complete"],
        "expected_positions": 80,
        "expected_rag_positions": 80,
        "original_report_sha256": manifest.old_report_sha256,
        "original_failure_unchanged": True,
    }
