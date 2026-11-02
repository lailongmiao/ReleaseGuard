"""Frozen 80-position engine, explicit revision, private evidence and crash-stop tests."""

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.main import app
from app.schemas.local_mvp import LocalActor
from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from evals.v11_backend import http_evidence, real
from evals.v11_backend.contracts import Dataset
from app.services.v11_recovery_guard import STAMPS
from app.services.v11_evaluation_policy import FORMAL_BATCH
from evals.v11_backend.recovery import Manifest


def dataset_and_auth():
    """Original questions and Gold remain byte-identical; only transport is synthetic."""
    dataset = Dataset.model_validate_json(Path("evals/datasets/v11_acceptance.json").read_text(encoding="utf-8"))
    auth = real.Authorization(
        authorization_reference="synthetic-local-test-not-real-permission",
        authorized_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        public_github_reads=False,
        model_calls=False,
        max_analysis_calls=80,
        dataset_hash=canonical_hash(dataset),
        source_fingerprint="a" * 64,
        designated_ledger_id="synthetic-offline-only",
        reconciled_budget_evidence_hash="b" * 64,
        price_evidence_hash="c" * 64,
        api_origin="http://synthetic.invalid",
        database_name="v11_isolated",
        database_port=35682,
        owner_user_id=1,
        approver_user_id=2,
        session_token="synthetic",
        budget_remaining_cny=19,
        rules_ready=True,
    )
    return dataset, auth


@pytest.mark.parametrize("failure", [None, "unknown", "rejected", "drift", "write", "cancel"])
@pytest.mark.parametrize("segmented", [False, True])
def test_fixed_positions_revision_and_no_restart(monkeypatch, tmp_path, failure, segmented):
    """All 80 slots run synthetically; every stop leaves a durable non-replayable journal."""
    dataset, auth = dataset_and_auth()
    recovery = None
    if segmented:
        auth = auth.model_copy(update={"authorization_reference": FORMAL_BATCH})
        recovery = SimpleNamespace(
            manifest=Manifest(
                old_report_sha256="a" * 64,
                supplemental_sha256="b" * 64,
                supplemental_hash="b" * 64,
                retained_hashes=("c" * 64, "d" * 64),
                remaining=STAMPS[2:],
                compatibility_hash="e" * 64,
            ),
            begin=AsyncMock(),
            finish=AsyncMock(),
        )
    objects, calls = {}, []
    current_case = {}
    output = {
        "summary": "Synthetic",
        "observations": [{"kind": "observation", "text": "Synthetic change", "citation_ids": ["diff:one"]}],
        "risks": [],
        "validation_suggestions": [],
        "unknowns": [],
        "candidate_steps": [],
        "model_risk": "MEDIUM",
    }

    def handler(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        calls.append((request.method, path))
        if path.endswith("/reviews"):
            identifier = "o" + str(len(objects))
            current_case[identifier] = next(c for c in dataset.cases if c.case_id == body["change"]["title"])
            objects[identifier] = {
                "object_id": identifier,
                "object_version": 1,
                "state": {"state_version": 1, "revision": 1},
            }
            value = objects[identifier]
        elif "/_evaluation/" in path:
            artifact = path.rsplit("/", 1)[1]
            value = (
                {"status": "unvalidated", "answer": output, "review_complete": True} if artifact == "answer" else {}
            )
        else:
            identifier = path.split("/reviews/")[1].split("/")[0]
            if path.endswith("/sync"):
                value = {"status": "completed", "snapshot_id": "s", "generation_id": "g"}
            elif "/snapshots/" in path:
                comp = current_case[identifier].comparison.model_dump(mode="json")
                if failure == "drift" and sum(p.endswith("/sync") for _, p in calls) == 2:
                    comp["head_sha"] = "0" * 40
                value = {"facts": {"comparison": comp}}
            elif path.endswith("/revision"):
                objects[identifier]["state"]["revision"] += 1
                value = {"status": "completed"}
            elif path.endswith("/analyze"):
                if failure == "cancel":
                    raise asyncio.CancelledError()
                value = {
                    "status": "unknown"
                    if failure == "unknown"
                    else "failed"
                    if failure == "rejected"
                    else "completed",
                    "analysis_id": "a",
                }
            elif "/analyses/" in path:
                value = {"output": output, "stale_reasons": [], "provenance": "provider"}
            elif "/commands/" in path:
                value = {"attempts": [{"usage": {"provenance": "synthetic"}, "charge": "0.0"}]}
            elif path.endswith("/search"):
                value = {"status": "completed", "items": [{"document": {"resource": "synthetic"}}]}
            else:
                value = objects[identifier]
        return httpx.Response(200, json=value)

    if failure == "write":
        original = real.checkpoint

        def write(path, report, **kwargs):
            if report["positions"] and report["positions"][0]["status"] == "completed":
                raise OSError("synthetic_disk_full")
            original(path, report, **kwargs)

        monkeypatch.setattr(real, "checkpoint", write)

    async def review(slot):
        await asyncio.sleep(0.01)
        return True

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://synthetic.invalid"
        ) as client:
            path = tmp_path / "report.json"
            if failure in {"write", "cancel"}:
                with pytest.raises(OSError if failure == "write" else asyncio.CancelledError):
                    await real.sample_positions(
                        dataset, auth, path, client, reviewer=review, pace_seconds=0, recovery=recovery
                    )
            else:
                result = await real.sample_positions(
                    dataset, auth, path, client, reviewer=review, pace_seconds=0, recovery=recovery
                )
                if failure is None:
                    expected = 78 if segmented else 80
                    assert result["completed"] and len(result["positions"]) == expected
                    assert len(objects) == expected // 2 and all(o["state"]["revision"] == 2 for o in objects.values())
                    assert sum(p.endswith("/search") for _, p in calls) == expected
                    if recovery:
                        assert [c.args[0] for c in recovery.begin.await_args_list] == list(STAMPS[2:])
                        assert recovery.finish.await_count == 78
                    assert all(
                        s["rag_command_evidence"]["attempts"] and s["rag_result"]["items"] for s in result["positions"]
                    )
                    assert all(s["review_wait_seconds"] >= 0.009 for s in result["positions"])
                else:
                    assert not result["completed"]
                    assert len(result["positions"]) == (2 if failure == "drift" else 1)
            if failure not in {"write", "cancel"}:
                timings = result["positions"][0]["http_timings"]
                assert timings and all(t["seconds"] >= 0 for t in timings)
                assert all(set(t) == {"stage", "method", "status", "outcome", "seconds"} for t in timings)
                assert any(t["stage"] == "analysis" for t in timings)
            before = len(calls)
            with pytest.raises(ValueError, match="already_exists"):
                await real.sample_positions(dataset, auth, path, client, reviewer=review, pace_seconds=0)
            assert len(calls) == before
            if failure:
                assert sum(p.endswith("/analyze") for _, p in calls) == 1

    asyncio.run(run())


def test_private_evidence_scope_acl_and_revoke(monkeypatch, tmp_path):
    """Frozen capture is exclusive; reads follow owner and current source permission."""
    monkeypatch.setattr(http_evidence, "code_state", lambda: {"source_hash": "a" * 64})
    monkeypatch.setattr(http_evidence, "verify_windows_acl", lambda directory: None)
    tmp_path.chmod(0o700)
    permit = http_evidence.EvidencePermit(
        batch_id="synthetic",
        source_hash="a" * 64,
        dataset_hash="b" * 64,
        owner_id=1,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        mode="synthetic",
        designated_ledger_id="synthetic-offline-only",
        ledger_charge_at_start=0,
        maximum_batch_charge=19,
        positions={
            "one": {
                "query": "q",
                "binding": {"repository_id": "checkout", "selector": {"kind": "pr", "pr_number": 1}},
                "comparison": {"head": "old"},
                "case_id": "one",
                "repeat": 1,
            }
        },
    )
    key = b"synthetic-memory-only-signing-material"
    signature = hmac.new(key, permit.model_dump_json().encode(), hashlib.sha256).hexdigest()
    sink = http_evidence.HTTPEvaluationEvidence(tmp_path, permit, signature, key)
    with pytest.raises(DomainError, match="revision_mismatch"):
        sink.validate_revision("one", 2)
    with pytest.raises(DomainError, match="position_not_authorized"):
        sink.sync_position("unapproved-sync", {})
    store = SimpleNamespace(
        connection=SimpleNamespace(authenticate=AsyncMock()),
        object_row=AsyncMock(
            return_value={"owner_user_id": 1, "repository_id": "checkout", "selector": {"kind": "pr", "pr_number": 1}}
        ),
        snapshot=AsyncMock(return_value={"facts": {"comparison": {"head": "old"}}}),
        dependencies=AsyncMock(return_value={"sources": {"source": 1}}),
    )
    actor = LocalActor(user_id=1, session_id="synthetic", expires_at=4102444800)
    body = SimpleNamespace(command_id="one", query="q", snapshot_id="snapshot")

    async def run():
        await sink.authorize(store, actor, "object", body)
        with pytest.raises(DomainError, match="consumed"):
            await sink.authorize(store, actor, "object", body)
        capture = await sink.begin(store, actor, "object", "one", '{"query":"q","sources":[]}', [], {})
        with pytest.raises(DomainError, match="answer_incomplete"):
            await capture({"summary": "token=private-value", "observations": []})
        answer = await sink.read(store, actor, "one", "answer")
        assert answer["status"] == "unvalidated" and not answer["review_complete"]
        assert "private-value" not in json.dumps(answer)
        store.dependencies.return_value = {"sources": {"source": 2}}
        with pytest.raises(DomainError, match="source_not_found"):
            await sink.read(store, actor, "one", "answer")
        with pytest.raises(DomainError, match="local_object_not_found"):
            await sink.read(store, actor.model_copy(update={"user_id": 2}), "one", "answer")

    asyncio.run(run())


def test_private_router_not_on_normal_app():
    """Ordinary application startup exposes no complete unvalidated answer endpoint."""
    assert not any(getattr(route, "path", "").startswith("/_evaluation") for route in app.routes)
