"""Real synthetic effects with fresh Gate and single-send safety."""

import asyncio

import pytest

from app.repositories.in_memory_execution import InMemoryExecutionRepository
from app.services.releaseguard.dry_run import DryRunService
from app.services.releaseguard.execution import ExecutionService
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.state_machine import DomainError
from app.schemas.releaseguard import ApprovalDecision
from app.services.tools.sandbox_executor import SandboxExecutor, SyntheticTargetStore
from tests.execution_fixtures import TrustedResolver, execution_setup, request_for
from tests.test_change_request_contract import NOW


class Fault:
    """One-shot deterministic fault, without timing races."""

    def __init__(self, stage: str) -> None:
        """Arm exactly one crash point."""
        self.stage = stage

    async def hit(self, stage: str) -> None:
        """Fail at a known boundary."""
        if self.stage == stage:
            self.stage = "done"
            raise RuntimeError("synthetic_fault")


async def runtime_setup(
    *, count: int = 2, faults: Fault | None = None
) -> tuple[ExecutionService, SyntheticTargetStore]:
    """Run a genuine pure simulation before creating any target effect."""
    service, context, descriptor = await execution_setup(count=count)
    preflight = ExecutionPreflight(service, TrustedResolver(context), descriptor)
    target = SyntheticTargetStore({"synthetic-resource": 0})
    adapter = SandboxExecutor(target)
    await DryRunService(preflight, adapter).run(await request_for(service), at=NOW)
    return ExecutionService(preflight, InMemoryExecutionRepository(), adapter, faults=faults), target


def test_concurrent_claim_one_send_and_known_replay() -> None:
    """Two current intents cannot send a step twice, even with different owners."""

    async def scenario() -> None:
        runtime, target = await runtime_setup()
        request = await request_for(runtime.preflight.service)
        results = await asyncio.gather(
            runtime.step(request, at=NOW),
            runtime.step(request.model_copy(update={"command_id": "other-owner"}), at=NOW),
            return_exceptions=True,
        )
        assert target.calls == 1 and any(not isinstance(r, BaseException) for r in results)
        run = await runtime.step(await request_for(runtime.preflight.service), at=NOW)
        assert target.calls == 1 and run.execution_events[-1].status == "SUCCEEDED"
        await runtime.step(await request_for(runtime.preflight.service, step="step-1"), at=NOW)
        assert target.calls == 2 and (await target.read("synthetic-resource", at=NOW)).value == 2

    asyncio.run(scenario())


def test_revocation_between_started_and_send_stops_effect() -> None:
    """Claim/STARTED is never continuing permission when review has been revoked."""

    async def scenario() -> None:
        runtime, target = await runtime_setup()
        service = runtime.preflight.service

        class RevokeBeforeSend:
            async def hit(self, stage: str) -> None:
                if stage == "after_started":
                    await service.invalidate(
                        "synthetic-approval",
                        runtime.preflight.resolver.context,
                        ApprovalDecision.REVOKED,
                        "review_revoked",
                        at=NOW,
                    )

        runtime.faults = RevokeBeforeSend()
        with pytest.raises(DomainError, match="approval_invalidated"):
            await runtime.step(await request_for(service), at=NOW)
        assert target.calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "stage",
    ["before_claim", "after_claim", "after_started", "before_send", "after_send", "after_result", "after_domain"],
)
def test_fault_windows_never_blind_resend(stage: str) -> None:
    """Lost ownership or possibly sent outcome is stopped or replayed, never resent."""

    async def scenario() -> None:
        runtime, target = await runtime_setup(faults=Fault(stage))
        try:
            await runtime.step(await request_for(runtime.preflight.service), at=NOW)
        except (RuntimeError, DomainError):
            pass
        before = target.calls
        try:
            await runtime.step(await request_for(runtime.preflight.service), at=NOW)
        except DomainError:
            pass
        assert target.calls <= 1
        if stage != "before_claim":
            assert target.calls == before

    asyncio.run(scenario())
