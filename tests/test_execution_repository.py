"""Ledger ownership, conflicts and atomic offline snapshot restore."""

import asyncio
import json

import pytest

from app.repositories.in_memory_execution import InMemoryExecutionRepository
from app.services.releaseguard.state_machine import DomainError
from tests.test_controlled_execution import runtime_setup
from tests.execution_fixtures import request_for
from tests.test_change_request_contract import NOW


def test_snapshot_restore_and_owner_rejection() -> None:
    """A transferred ledger retains the first owner and rejects corrupted installs."""

    async def scenario() -> None:
        runtime, _ = await runtime_setup()
        await runtime.step(await request_for(runtime.preflight.service), at=NOW)
        snapshot = await runtime.ledger.export_snapshot()
        clone = InMemoryExecutionRepository()
        await clone.restore_snapshot(snapshot)
        assert await clone.export_snapshot() == snapshot
        claim, events = (await clone.entries("synthetic-run"))[0]
        with pytest.raises(DomainError):
            await clone.append_fact(claim.model_copy(update={"owner": "impostor"}), events[-1])
        data = json.loads(snapshot)
        data["entries"][0]["events"][0]["run_id"] = "foreign-run"
        empty = InMemoryExecutionRepository()
        with pytest.raises(DomainError):
            await empty.restore_snapshot(json.dumps(data))
        assert await empty.entries("synthetic-run") == ()
        with pytest.raises(DomainError):
            await clone.restore_snapshot(snapshot)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault", ["version", "duplicate", "hash", "foreign", "audit", "actor", "sequence", "chronology"]
)
def test_corrupt_ledger_restoration_is_atomic(fault: str) -> None:
    """Validate all candidate records before installing a transferred store."""

    async def scenario() -> None:
        runtime, _ = await runtime_setup()
        await runtime.step(await request_for(runtime.preflight.service), at=NOW)
        payload = json.loads(await runtime.ledger.export_snapshot())
        entry = payload["entries"][0]
        if fault == "version":
            payload["version"] = "execution-ledger.v2"
        elif fault == "duplicate":
            payload["entries"].append(entry)
        elif fault == "hash":
            entry["claim"]["payload_hash"] = "0" * 64
        elif fault == "foreign":
            entry["events"][0]["run_id"] = "foreign-run"
        elif fault == "audit":
            entry["events"][0]["audit_ids"] = []
        elif fault == "actor":
            entry["events"][0]["actor_id"] = "impostor"
        elif fault == "sequence":
            entry["events"].reverse()
        else:
            entry["events"][0]["occurred_at"] = "2020-01-01T00:00:00Z"
        clone = InMemoryExecutionRepository()
        with pytest.raises(DomainError, match="execution_snapshot_invalid"):
            await clone.restore_snapshot(json.dumps(payload))
        assert await clone.entries("synthetic-run") == ()
        await clone.restore_snapshot(await runtime.ledger.export_snapshot())
        assert await clone.entries("synthetic-run")

    asyncio.run(scenario())
