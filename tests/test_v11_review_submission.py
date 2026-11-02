"""Offline review submission regressions; no database or Provider is required."""

import asyncio
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import httpx
from test_v11_acceptance_engineering import dataset_and_auth

from app.services.releaseguard.normalization import canonical_hash
from evals.v11_backend import review_submission as review
from evals.v11_backend import real
from scripts.v11_process_environment import minimal_environment


def slot_fixture():
    """Independent literal edit with fully bound synthetic input/output/RAG records."""
    stamp = canonical_hash(["synthetic-review", "dataset", "case", 1])
    citation = {
        "citation_id": "diff:synthetic",
        "text": "old -> new",
        "text_hash": canonical_hash("old -> new"),
        "snapshot_id": "snapshot",
        "kind": "diff",
        "side": "RIGHT",
    }
    source = {k: citation[k] for k in ("citation_id", "text", "kind", "side")}
    user = json.dumps({"query": "literal change?", "sources": [source]})
    output = {
        "summary": "Literal edit only",
        "observations": [{"text": "old becomes new", "kind": "observation", "citation_ids": ["diff:synthetic"]}],
        "risks": [],
    }
    wire = {**output, "comparisons": [], "comparison_contract": "v11-behavior-comparison.2"}
    captured = {
        "user": user,
        "input_hash": canonical_hash(user),
        "review_complete": True,
        "transformations": [],
        "citation_ids": ["diff:synthetic"],
        "object_id": "object",
    }
    rag = {"status": "completed", "items": [{"document": {"resource": "rule"}}]}
    return {
        "stamp": stamp,
        "case_id": "case",
        "repeat": 1,
        "status": "completed",
        "rag_status": "completed",
        "hard_gates_passed": True,
        "object_id": "object",
        "review_wait_seconds": 0,
        "input": {"query": "literal change?", "captured": captured, "snapshot": {"snapshot_id": "snapshot"}},
        "analysis": {
            "analysis_id": "analysis",
            "object_id": "object",
            "revision": 1,
            "snapshot_id": "snapshot",
            "generation_id": "generation",
            "output": output,
            "citations": [citation],
            "telemetry": {"comparisons": []},
        },
        "evaluation_evidence": {
            "input": captured,
            "answer": {
                "answer": wire,
                "review_complete": True,
                "original_parsed_hash": canonical_hash(wire),
                "transformations": [],
            },
            "validation": {"status": "contract_passed"},
        },
        "command_evidence": {
            "analysis": {
                "status": "completed",
                "result": {"analysis_id": "analysis"},
                "attempts": [{"attempt": "analysis-1"}],
            },
            "sync": {"status": "completed"},
        },
        "rag_input": {"snapshot_id": "snapshot", "generation_id": "generation", "command_id": "rag:" + stamp},
        "rag_result": rag,
        "rag_command_evidence": {"status": "completed", "result": rag},
    }


def decision_fixture():
    """This is an explicit synthetic reviewer decision, not an automatic rating."""
    return {
        "approved": True,
        "critical_error": False,
        "score": 4,
        "claims_supported": True,
        "body_supported": True,
        "unknown_correct": True,
        "citation_identity_valid": True,
        "notes": "Synthetic reviewer read the complete literal evidence.",
        "evidence_read": True,
        "reviewer": "synthetic-reviewer",
    }


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """Unit filesystem is isolated; the separate process probe exercises real NTFS ACL."""
    tmp_path.chmod(0o700)
    monkeypatch.setattr(review, "verify_windows_acl", lambda path: None)
    slot = slot_fixture()
    digest = review.prepare(tmp_path, slot, "synthetic-review", "dataset", datetime.now(UTC) + timedelta(hours=1))
    return tmp_path, slot, digest


def test_hash_is_content_not_stamp_and_wait_mutation_cannot_change_it(queue):
    """Runtime updates cannot alter the evidence the operator reviewed."""
    directory, slot, digest = queue
    assert digest != slot["stamp"]
    slot.update(status="failed", review_wait_seconds=600, failure_code="later-runtime-only")
    path = review.submit(directory, slot["stamp"], decision_fixture())
    proposal = review.read(path)
    assert proposal["payload"]["snapshot_hash"] == digest
    assert review.load_packet(directory, slot["stamp"])["payload"]["position"]["status"] == "completed"
    permission = AsyncMock()
    assert asyncio.run(review.wait_review(directory, slot["stamp"], digest, permission))
    permission.assert_awaited_once()
    with pytest.raises(review.ReviewSubmissionError, match="consumed"):
        review.submit(directory, slot["stamp"], decision_fixture())


@pytest.mark.parametrize(
    "error", ["stamp", "hash", "fields", "ack", "conflict", "duplicate", "snapshot", "old", "expired", "revoked"]
)
def test_bad_submissions_never_approve(queue, error):
    """Invalid metadata or authority cannot publish an accepted review."""
    directory, slot, digest = queue
    stamp = slot["stamp"]
    decision = decision_fixture()
    if error == "stamp":
        with pytest.raises((FileNotFoundError, review.ReviewSubmissionError)):
            review.submit(directory, "f" * 64, decision)
        return
    if error in {"hash", "fields"}:
        decision["position_hash" if error == "hash" else "replacement_answer"] = stamp
    elif error == "ack":
        decision["evidence_read"] = False
    elif error == "conflict":
        decision["critical_error"] = True
    elif error == "duplicate":
        review.submit(directory, stamp, decision)
    elif error in {"snapshot", "expired"}:
        path = review.packet_path(directory, stamp)
        packet = review.read(path)
        packet["payload"]["position"]["analysis"]["output"]["summary"] = "tampered"
        if error == "expired":
            packet["payload"]["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
            packet["hash"] = canonical_hash(packet["payload"])
            packet["signature"] = review.signature(packet["payload"], review.key_bytes(directory))
        path.write_text(json.dumps(packet), encoding="utf-8")
    elif error == "revoked":
        (directory / "review.revoked").touch()
    elif error == "old":
        with pytest.raises(review.ReviewSubmissionError, match="snapshot_changed"):
            review.load_packet(directory, stamp, "a" * 64)
        return
    with pytest.raises(review.ReviewSubmissionError):
        review.submit(directory, stamp, decision)
    assert not (directory / (stamp + ".review.accepted.json")).exists()


@pytest.mark.parametrize("field", ["input", "answer", "citation", "object", "rag", "comparison"])
def test_binding_tamper_is_rejected_before_packet(queue, field):
    """Different evidence identities cannot be combined into one review packet."""
    directory, slot, _ = queue
    slot = copy.deepcopy(slot)
    if field == "input":
        slot["evaluation_evidence"]["input"]["user"] += " "
    elif field == "answer":
        slot["evaluation_evidence"]["answer"]["review_complete"] = False
    elif field == "citation":
        slot["analysis"]["citations"][0]["text"] = "invented"
    elif field == "object":
        slot["object_id"] = "other"
    elif field == "rag":
        slot["rag_command_evidence"]["status"] = "unknown"
    elif field == "comparison":
        slot["analysis"]["telemetry"]["comparisons"] = [{"classification": "invented"}]
    with pytest.raises(review.ReviewSubmissionError):
        review.verify_evidence(slot)


def test_atomic_publication_and_failure_never_expose_partial(queue, monkeypatch):
    """Readers only observe validated complete publications."""
    directory, slot, _ = queue
    original = os.link
    seen = []

    def checked_link(source, destination):
        assert not Path(destination).exists()
        seen.append(review.read(Path(source)))
        original(source, destination)

    monkeypatch.setattr(review.os, "link", checked_link)
    review.submit(directory, slot["stamp"], decision_fixture())
    assert seen and not list(directory.glob("*.pending"))


def test_save_failure_and_cancel_do_not_consume_or_resend(queue, monkeypatch):
    """Disk failure and cancellation leave no consumed review or sending path."""
    directory, slot, digest = queue
    monkeypatch.setattr(review.os, "link", lambda *args: (_ for _ in ()).throw(OSError("synthetic_disk_full")))
    with pytest.raises(OSError):
        review.submit(directory, slot["stamp"], decision_fixture())
    assert not list(directory.glob("*.review.proposal.*.json"))
    permission = AsyncMock()

    async def run():
        task = asyncio.create_task(review.wait_review(directory, slot["stamp"], digest, permission, interval=0.001))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    permission.assert_not_awaited()


def test_invalid_metadata_blocks_until_valid_submission_zero_provider(queue):
    """Corrected metadata unblocks only after an explicit unchanged judgment."""
    directory, slot, digest = queue
    stamp = slot["stamp"]
    # Actual evidence3 operator error: stamp placed in the content-hash field.
    review.publish(directory / (stamp + ".review.proposal.invalid.json"), {"position_hash": stamp, "approved": True})
    permission = AsyncMock()

    async def run():
        task = asyncio.create_task(review.wait_review(directory, stamp, digest, permission, interval=0.001))
        await asyncio.sleep(0.02)
        assert not task.done()
        permission.assert_not_awaited()
        review.submit(directory, stamp, decision_fixture())
        assert await task

    asyncio.run(run())
    errors = list(directory.glob("*.rejected.json"))
    assert len(errors) == 1 and review.read(errors[0])["code"] == "review_submission_invalid"
    permission.assert_awaited_once()


def test_live_revocation_blocks_consumption(queue):
    """A staged proposal cannot override current source or object revocation."""
    directory, slot, digest = queue
    review.submit(directory, slot["stamp"], decision_fixture())
    permission = AsyncMock(side_effect=review.ReviewSubmissionError("review_live_permission_denied"))
    with pytest.raises(review.ReviewSubmissionError, match="live_permission_denied"):
        asyncio.run(review.wait_review(directory, slot["stamp"], digest, permission))
    assert not list(directory.glob("*.accepted.json"))


def test_actual_archived_evidence_when_explicitly_selected(monkeypatch):
    """Optional read-only archive regression; never discovers or copies private directories."""
    archive = os.environ.get("V11_REVIEW_ARCHIVE")
    if not archive:
        pytest.skip("explicit private archive not selected")
    path = Path(archive)
    review.private(path)
    batch = review.read(path)
    assert len(batch["positions"]) == 2 and not batch["completed"]
    for slot in batch["positions"]:
        review.verify_evidence(slot)
    assert batch["positions"][1]["reason"] == "ValueError"


def test_concurrent_submit_has_one_winner(queue):
    """Concurrent operator submissions cannot overwrite or double-consume."""
    directory, slot, _ = queue

    def run_one():
        try:
            return str(review.submit(directory, slot["stamp"], decision_fixture()))
        except review.ReviewSubmissionError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_one(), range(2)))
    assert sum(value.endswith(".json") for value in results) == 1
    assert len(list(directory.glob("*.review.proposal.*.json"))) == 1


def test_permission_expiry_during_check_stops(queue):
    """The permit deadline also bounds a slow live permission check."""
    directory, slot, _ = queue
    stamp = slot["stamp"]
    path = review.packet_path(directory, stamp)
    packet = review.read(path)
    packet["payload"]["expires_at"] = (datetime.now(UTC) + timedelta(seconds=0.1)).isoformat()
    packet["hash"] = canonical_hash(packet["payload"])
    packet["signature"] = review.signature(packet["payload"], review.key_bytes(directory))
    path.write_text(json.dumps(packet), encoding="utf-8")
    review.submit(directory, stamp, decision_fixture())

    async def slow_permission():
        await asyncio.sleep(10)

    with pytest.raises(review.ReviewSubmissionError, match="permission_check_timeout"):
        asyncio.run(review.wait_review(directory, stamp, packet["hash"], slow_permission))
    assert not list(directory.glob("*.accepted.json"))


@pytest.mark.parametrize("outcome", ["approve", "revoked"])
def test_three_process_private_packet_submit_consume(tmp_path, outcome):
    """Real private filesystem survives independent processes; no database or network."""
    directory = tmp_path / "private"
    directory.mkdir()
    if os.name != "nt":
        directory.chmod(0o700)
    process_env = minimal_environment()
    if os.name == "nt":
        script = (
            "$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
            "& icacls.exe $env:REVIEW_SYNTHETIC_PATH /inheritance:r /grant:r "
            "('*'+$sid+':(OI)(CI)F') '*S-1-5-18:(OI)(CI)F';"
            "if($LASTEXITCODE -ne 0){exit $LASTEXITCODE};"
            "& icacls.exe $env:REVIEW_SYNTHETIC_PATH /remove:g '*S-1-3-4' '*S-1-5-32-544';exit $LASTEXITCODE"
        )
        acl_result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**process_env, "REVIEW_SYNTHETIC_PATH": str(directory)},
            capture_output=True,
        )
        assert acl_result.returncode == 0, acl_result.stderr.decode(errors="replace")
    review.private(directory)
    slot = slot_fixture()
    (directory / "slot.json").write_text(json.dumps(slot), encoding="utf-8")
    (directory / "decision.json").write_text(json.dumps(decision_fixture()), encoding="utf-8")
    env = {
        **process_env,
        "APP_ENV": "test",
        "LOCAL_MVP_ONLY": "true",
        "LOCAL_MVP_ENABLED": "true",
        "V11_REAL_MODELS_ENABLED": "false",
        "V11_REAL_READS_ENABLED": "false",
        "V1_REAL_MODELS_ENABLED": "false",
        "LANGFUSE_TRACING_ENABLED": "false",
        "UV_OFFLINE": "true",
        "REVIEW_SYNTHETIC_PATH": str(directory),
        "REVIEW_SYNTHETIC_STAMP": slot["stamp"],
        "REVIEW_SYNTHETIC_OUTCOME": outcome,
    }
    prepare_code = """
import os,json
from datetime import UTC,datetime,timedelta
from pathlib import Path
from evals.v11_backend import review_submission as r
p=Path(os.environ['REVIEW_SYNTHETIC_PATH'])
s=r.read(p/'slot.json')
r.prepare(p,s,'synthetic-review','dataset',datetime.now(UTC)+timedelta(minutes=10))
"""
    first = subprocess.run([sys.executable, "-c", prepare_code], env=env, capture_output=True, timeout=60)
    assert first.returncode == 0, first.stderr.decode(errors="replace")
    second = subprocess.run(
        [
            sys.executable,
            "scripts/v11_review_submit.py",
            "--directory",
            str(directory),
            "--stamp",
            slot["stamp"],
            "--decision",
            str(directory / "decision.json"),
        ],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert second.returncode == 0, second.stderr.decode(errors="replace")
    consume_code = """
import asyncio,os
from pathlib import Path
from evals.v11_backend import review_submission as r
p=Path(os.environ['REVIEW_SYNTHETIC_PATH']);s=os.environ['REVIEW_SYNTHETIC_STAMP']
packet=r.load_packet(p,s)
async def check():
 if os.environ['REVIEW_SYNTHETIC_OUTCOME']=='revoked':
  raise r.ReviewSubmissionError('review_live_permission_denied')
try:
 approved=asyncio.run(r.wait_review(p,s,packet['hash'],check))
 r.publish(p/'result.json',{'approved':approved,'provider_sends':0})
except r.ReviewSubmissionError as error:
 r.publish(p/'result.json',{'error':str(error),'provider_sends':0})
"""
    third = subprocess.run([sys.executable, "-c", consume_code], env=env, capture_output=True, timeout=60)
    assert third.returncode == 0, third.stderr.decode(errors="replace")
    result = review.read(directory / "result.json")
    assert result["provider_sends"] == 0
    if outcome == "approve":
        assert result["approved"]
    else:
        assert result["error"] == "review_live_permission_denied"
        assert not list(directory.glob("*.accepted.json"))


@pytest.mark.parametrize("revoke", [False, True])
def test_http_sampler_blocks_next_send_until_review_and_rechecks_acl(tmp_path, monkeypatch, revoke):
    """Drive the real sampler via HTTP transport; metadata repair never calls Provider twice."""
    dataset, auth = dataset_and_auth()
    fixture = slot_fixture()
    first = dataset.cases[0]
    auth = auth.model_copy(update={"authorization_reference": "synthetic-review"})
    stamp = real.position_stamp(auth, first.case_id, 1)
    counts = {"analyze": 0, "rag": 0}
    state = {"revoked": False}
    output = {
        **fixture["analysis"]["output"],
        "validation_suggestions": [],
        "unknowns": [],
        "candidate_steps": [],
        "model_risk": "MEDIUM",
    }
    fixture["analysis"].update(output=output, stale_reasons=[], provenance="provider")
    wire = {**output, "comparisons": [], "comparison_contract": "v11-behavior-comparison.2"}
    fixture["evaluation_evidence"]["answer"].update(answer=wire, original_parsed_hash=canonical_hash(wire))
    captured = fixture["evaluation_evidence"]["input"]
    decoded = json.loads(captured["user"])
    decoded["query"] = first.query
    captured.update(user=json.dumps(decoded), input_hash=canonical_hash(json.dumps(decoded)))
    monkeypatch.setattr(review, "verify_windows_acl", lambda path: None)
    tmp_path.chmod(0o700)

    def handler(request):
        path = request.url.path
        if path.startswith("/_evaluation/"):
            if state["revoked"]:
                return httpx.Response(403, json={"detail": "revoked"})
            value = fixture["evaluation_evidence"].get(path.rsplit("/", 1)[1], {})
        elif path.endswith("/sync"):
            value = {"status": "completed", "snapshot_id": "snapshot", "generation_id": "generation"}
        elif "/snapshots/" in path:
            value = {"snapshot_id": "snapshot", "facts": {"comparison": first.comparison.model_dump(mode="json")}}
        elif path.endswith("/analyze"):
            counts["analyze"] += 1
            value = {"status": "completed", "analysis_id": "analysis"}
        elif "/analyses/" in path:
            value = fixture["analysis"]
        elif path.endswith("/search"):
            counts["rag"] += 1
            value = fixture["rag_result"]
        elif "/commands/rag:" in path:
            value = fixture["rag_command_evidence"]
        elif "/commands/analysis:" in path:
            value = fixture["command_evidence"]["analysis"]
        elif "/commands/sync:" in path:
            value = fixture["command_evidence"]["sync"]
        else:
            value = {"object_id": "object", "object_version": 1, "state": {"state_version": 1, "revision": 1}}
        return httpx.Response(200, json=value)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://synthetic.invalid"
        ) as client:
            task = asyncio.create_task(
                real.sample_positions(dataset, auth, tmp_path / "report.json", client, pace_seconds=0)
            )
            for _ in range(100):
                if review.packet_path(tmp_path, stamp).exists() or task.done():
                    break
                await asyncio.sleep(0.01)
            assert not task.done(), task.result() if task.done() else None
            review.publish(tmp_path / (stamp + ".review.proposal.wrong.json"), {"position_hash": stamp})
            await asyncio.sleep(0.05)
            assert not task.done() and counts == {"analyze": 1, "rag": 1}
            decision = decision_fixture()
            decision["approved"] = False
            state["revoked"] = revoke
            review.submit(tmp_path, stamp, decision)
            result = await asyncio.wait_for(task, 5)
            assert counts == {"analyze": 1, "rag": 1}
            if revoke:
                assert result["positions"][0]["failure_code"] == "review_live_permission_denied"
            else:
                assert result["stop_reason"] == "semantic_review_stopped"

    asyncio.run(run())


def test_proposal_mutation_during_permission_check_is_not_consumed(queue):
    """Reauthorization cannot hide a replaced decision between validation and consume."""
    directory, slot, digest = queue
    path = review.submit(directory, slot["stamp"], decision_fixture())

    async def change_during_check():
        proposal = review.read(path)
        proposal["payload"]["decision"]["approved"] = False
        proposal["signature"] = review.signature(proposal["payload"], review.key_bytes(directory))
        path.write_text(json.dumps(proposal), encoding="utf-8")

    with pytest.raises(review.ReviewSubmissionError, match="changed_during_permission_check"):
        asyncio.run(review.wait_review(directory, slot["stamp"], digest, change_during_check))
    assert not list(directory.glob("*.accepted.json"))
