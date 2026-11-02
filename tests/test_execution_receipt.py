"""Receipts distinguish simulation, confirmed execution and unknown outcomes."""

import asyncio

import pytest
from pydantic import ValidationError

from app.schemas.releaseguard import ChangeRunState, ChangeStatus
from app.services.releaseguard.execution_commands import execution_command
from app.services.releaseguard.receipt import ReceiptService
from app.services.releaseguard.recovery import RecoveryService
from app.services.releaseguard.verification import VerificationService
from app.services.tools.verification_adapter import IndependentRegisterReader
from tests.execution_fixtures import request_for
from tests.test_change_request_contract import NOW
from tests.test_controlled_execution import Fault, runtime_setup
from tests.test_execution_recovery import partial_setup
from tests.test_execution_verification import verifying_setup
from app.schemas.execution import Direction
from app.services.releaseguard.state_machine import DomainError


def test_success_receipt_complete_and_tampering_rejected() -> None:
    """All saved events, claims, citations and independent facts must join."""

    async def scenario() -> None:
        runtime, target, _ = await verifying_setup()
        service = runtime.preflight.service
        await VerificationService(runtime, IndependentRegisterReader(target)).verify(
            await request_for(service), at=NOW
        )
        request = await request_for(service, command="receipt")
        run = await ReceiptService(runtime).issue(request, at=NOW)
        assert run.status is ChangeStatus.RECEIPTED and run.receipt.final_status == "SUCCEEDED"
        assert await ReceiptService(runtime).issue(request, at=NOW) == run
        with pytest.raises(DomainError, match="receipt_intent_conflict"):
            await ReceiptService(runtime).issue(request.model_copy(update={"step_id": "step-impostor"}), at=NOW)
        assert ChangeRunState.model_validate_json(run.model_dump_json()) == run
        for key, value in (
            ("claim_ids", []),
            ("audit_ids", []),
            ("manifest_hash", "0" * 64),
            ("mode", "offline_simulation"),
        ):
            payload = run.model_dump(mode="json")
            payload["receipt"][key] = value
            for record in payload["record_history"]:
                if record.get("receipt_id") == run.receipt.receipt_id:
                    record[key] = value
            with pytest.raises(ValidationError):
                ChangeRunState.model_validate(payload)

    asyncio.run(scenario())


def test_unknown_receipt_nonterminal_and_later_receipt_links_it() -> None:
    """Reconciliation appends facts and a new receipt; UNKNOWN never authorizes sends."""

    async def scenario() -> None:
        runtime, target = await runtime_setup(count=1, faults=Fault("after_send"))
        service = runtime.preflight.service
        await runtime.step(await request_for(service), at=NOW)
        run = await ReceiptService(runtime).issue(await request_for(service, command="unknown"), at=NOW)
        assert run.status is ChangeStatus.EXECUTION_UNKNOWN and not run.receipt.terminal
        diagnostic = run.receipt.receipt_id
        await RecoveryService(runtime).reconcile(await request_for(service), at=NOW)
        await VerificationService(runtime, IndependentRegisterReader(target)).verify(
            await request_for(service), at=NOW
        )
        run = await ReceiptService(runtime).issue(await request_for(service, command="final"), at=NOW)
        assert (
            diagnostic in run.receipt.prior_receipt_ids
            and run.receipt.final_status == "SUCCEEDED"
            and target.calls == 1
        )

    asyncio.run(scenario())


def test_recovered_receipt_retains_original_failed_step() -> None:
    """Successful partial compensation cannot erase original failure facts."""

    async def scenario() -> None:
        runtime, target, recovery = await partial_setup()
        service = runtime.preflight.service
        await recovery.start(
            await request_for(service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"),
            at=NOW,
        )
        run = await recovery.compensate(
            await request_for(service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"),
            at=NOW,
        )
        run = await service.domain.commit_run_update(
            execution_command(
                run,
                runtime.preflight.resolver.context,
                command_id="rollback-verify",
                at=NOW,
                code="compensations_confirmed",
                target=ChangeStatus.VERIFYING,
            )
        )
        request = await request_for(
            service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review"
        )
        await VerificationService(runtime, IndependentRegisterReader(target)).verify(request, at=NOW)
        request = await request_for(
            service, step="undo-0", direction=Direction.COMPENSATION, approval="recovery-review", command="recovered"
        )
        run = await ReceiptService(runtime).issue(request, at=NOW)
        assert run.receipt.final_status == "RECOVERED" and run.receipt.compensation_ids
        assert any(event.step_id == "step-1" and event.status == "FAILED" for event in run.execution_events)

    asyncio.run(scenario())
