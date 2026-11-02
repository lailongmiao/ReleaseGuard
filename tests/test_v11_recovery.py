"""Fixed-segment negative cases with no Provider capability."""

import copy
import asyncio
import json
from pathlib import Path

import pytest
import httpx
from test_v11_review_submission import slot_fixture

from app.services.releaseguard.normalization import canonical_hash
from app.services.releaseguard.state_machine import DomainError
from app.services.v11_recovery_guard import STAMPS, POSITION_KEYS
from evals.v11_backend.recovery import Manifest, aggregate, build_manifest
from evals.v11_backend.transport import FormalGitHubTransport
from evals.v11_backend.contracts import Dataset


def manifest_fixture():
    """Exercise the named invariant without external capabilities."""
    return Manifest(
        old_report_sha256="a" * 64,
        supplemental_sha256="b" * 64,
        supplemental_hash="b" * 64,
        retained_hashes=(canonical_hash({"n": 1}), canonical_hash({"n": 2})),
        remaining=STAMPS[2:],
        compatibility_hash="c" * 64,
    )


def test_fixed_exact_matrix():
    """Exercise the named invariant without external capabilities."""
    manifest_fixture().check()
    assert len(set(STAMPS)) == 80
    assert len(set(STAMPS[2:])) == 78
    assert POSITION_KEYS[:2] == (("real-01-zh", 1), ("real-01-zh", 2))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "consumed", "batch", "dataset", "version"])
def test_manifest_cannot_select_other_work(mutation):
    """Exercise the named invariant without external capabilities."""
    raw = manifest_fixture().model_dump()
    if mutation == "missing":
        raw["remaining"] = STAMPS[3:]
    elif mutation == "duplicate":
        raw["remaining"] = STAMPS[2:-1] + (STAMPS[2],)
    elif mutation == "consumed":
        raw["remaining"] = STAMPS[:78]
    else:
        raw[{"batch": "batch_id", "dataset": "dataset_hash", "version": "version"}[mutation]] = "other"
    with pytest.raises(DomainError):
        Manifest.model_validate(raw).check()


@pytest.mark.parametrize("count", [0, 1, 77, 78])
def test_aggregate_never_shrinks_denominator_or_grants_quality(count):
    """Exercise the named invariant without external capabilities."""
    manifest = manifest_fixture()
    old = {"positions": [{"n": 1}, {"n": 2}]}
    supplemental = {"positions": [{"archived_position_hash": h} for h in manifest.retained_hashes]}
    manifest = manifest.model_copy(update={"supplemental_hash": canonical_hash(supplemental)})
    segment = {
        "manifest_hash": canonical_hash(manifest),
        "completed": True,
        "positions": [
            {
                "stamp": s,
                "status": "completed",
                "rag_status": "completed",
                "hard_gates_passed": True,
                "review_approved": True,
            }
            for s in STAMPS[2 : 2 + count]
        ],
    }
    result = aggregate(manifest, old, supplemental, segment)
    assert result["expected_positions"] == 80
    assert result["complete"] is False  # A full list without full evidence is not completion.
    assert result["quality_passed"] is False
    if count:
        changed = copy.deepcopy(segment)
        changed["positions"][0]["stamp"] = STAMPS[0]
        with pytest.raises(DomainError):
            aggregate(manifest, old, supplemental, changed)
        segment["positions"][0]["review_approved"] = False
        assert not aggregate(manifest, old, supplemental, segment)["complete"]


def test_actual_archived_pair_binding():
    """Exercise the named invariant without external capabilities."""
    archive = Path("C:/private/releaseguard-evaluation/batch.json")
    if not archive.exists():
        pytest.skip("explicit archive unavailable")
    manifest = build_manifest(
        archive, Path("evals/reports/v11_evidence3_supplemental_review.json"), {"comparable": True}
    )
    assert len(manifest.remaining) == 78


def test_github_carryover_is_not_new_allowance(tmp_path):
    """The recovery segment retains eighteen consumed attempts and the same 2000 cap."""
    sends = []

    async def run():
        transport = FormalGitHubTransport(
            tmp_path, httpx.MockTransport(lambda req: sends.append(req) or httpx.Response(200)), carried_attempts=18
        )
        request = httpx.Request("GET", "https://api.github.com/repos/example/repo")
        await transport.handle_async_request(request)
        assert transport.attempts == 19
        reopened = FormalGitHubTransport(
            tmp_path, httpx.MockTransport(lambda req: httpx.Response(200)), carried_attempts=18
        )
        assert reopened.attempts == 19
        reopened.attempts = 2000
        with pytest.raises(DomainError, match="attempt_limit"):
            await reopened.handle_async_request(request)
        assert len(sends) == 1
        await transport.shutdown()
        await reopened.shutdown()

    asyncio.run(run())


def test_full_bound_segment_can_be_complete_but_not_auto_graded():
    """All 78 synthetic records bind to original questions; a changed input cannot pass."""
    dataset = Dataset.model_validate_json(Path("evals/datasets/v11_acceptance.json").read_text(encoding="utf-8"))
    manifest = manifest_fixture()
    old = {"positions": [{"n": 1}, {"n": 2}]}
    supplemental = {"positions": [{"archived_position_hash": h} for h in manifest.retained_hashes]}
    manifest = manifest.model_copy(update={"supplemental_hash": canonical_hash(supplemental)})
    slots = []
    for case_id, repeat in POSITION_KEYS[2:]:
        case = next(c for c in dataset.cases if c.case_id == case_id)
        slot = slot_fixture()
        slot["analysis"]["output"].update(validation_suggestions=[], unknowns=[], candidate_steps=[], model_risk="LOW")
        answer = slot["evaluation_evidence"]["answer"]
        answer["answer"].update(slot["analysis"]["output"])
        answer["original_parsed_hash"] = canonical_hash(answer["answer"])
        stamp = STAMPS[len(slots) + 2]
        slot.update(stamp=stamp, case_id=case_id, repeat=repeat, review_approved=True)
        slot["analysis"]["revision"] = repeat
        slot["rag_input"]["command_id"] = "rag:" + stamp
        slot["input"]["query"] = case.query
        slot["input"]["snapshot"]["facts"] = {"comparison": case.comparison.model_dump(mode="json")}
        captured = slot["evaluation_evidence"]["input"]
        user = json.loads(captured["user"])
        user["query"] = case.query
        captured.update(user=json.dumps(user), input_hash=canonical_hash(json.dumps(user)))
        slots.append(slot)
    segment = {"manifest_hash": canonical_hash(manifest), "completed": True, "positions": slots}
    result = aggregate(manifest, old, supplemental, segment)
    assert result["complete"] is True and result["quality_passed"] is False
    changed_review = copy.deepcopy(supplemental)
    changed_review["new_conclusion"] = "approve"
    with pytest.raises(DomainError, match="supplemental_changed"):
        aggregate(manifest, old, changed_review, segment)
    slots[0]["input"]["query"] = "different question"
    assert not aggregate(manifest, old, supplemental, segment)["complete"]
