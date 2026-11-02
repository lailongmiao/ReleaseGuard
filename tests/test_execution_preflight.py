"""Fresh authorization blocks every effect before target inspection."""

import asyncio
from datetime import timedelta

import pytest

from app.schemas.releaseguard import ApprovalDecision
from app.schemas.execution import ExecutionMode
from app.schemas.tooling import ToolDescriptor
from app.services.releaseguard.dry_run import DryRunService
from app.services.releaseguard.execution_preflight import ExecutionPreflight
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.sandbox_executor import SandboxExecutor, SyntheticTargetStore
from tests.execution_fixtures import TrustedResolver, execution_setup, request_for
from tests.test_authorization import policy_fixture
from tests.test_change_request_contract import NOW


@pytest.mark.parametrize(
    "fault",
    [
        "permission",
        "expired",
        "revoked",
        "version",
        "manifest",
        "live",
        "tenant",
        "session",
        "descriptor",
        "evidence_expired",
    ],
)
def test_preflight_denials_have_zero_target_reads_or_writes(fault: str) -> None:
    """Do not reuse an old APPROVED state as continuing permission."""

    async def scenario() -> None:
        service, context, descriptor = await execution_setup()
        resolver = TrustedResolver(context)
        request, at = await request_for(service), NOW
        if fault == "permission":
            service.permission_policy = policy_fixture(rules=())
        elif fault in {"expired", "evidence_expired"}:
            at += timedelta(minutes=11 if fault == "expired" else 90)
        elif fault == "revoked":
            await service.invalidate(
                request.approval_request_id, context, ApprovalDecision.REVOKED, "review_revoked", at=NOW
            )
        elif fault == "version":
            request = request.model_copy(update={"state_version": 0})
        elif fault == "manifest":
            request = request.model_copy(update={"manifest_hash": "b" * 64})
        elif fault == "live":
            request = request.model_copy(update={"mode": ExecutionMode.AUTHORIZED_TEST})
        elif fault == "tenant":
            resolver.context = context.model_copy(update={"resource_tenant_id": "other-tenant"})
        elif fault == "session":
            resolver.context = context.model_copy(
                update={
                    "principal": context.principal.model_copy(
                        update={
                            "authenticated_at": NOW - timedelta(minutes=2),
                            "expires_at": NOW - timedelta(minutes=1),
                        }
                    )
                }
            )
        elif fault == "descriptor":
            descriptor = ToolDescriptor.model_validate(
                {**descriptor.model_dump(exclude={"descriptor_hash"}), "timeout_ms": 2000}
            )
        target = SyntheticTargetStore({"synthetic-resource": 0})
        dry = DryRunService(ExecutionPreflight(service, resolver, descriptor), SandboxExecutor(target))
        with pytest.raises(DomainError):
            await dry.run(request, at=at)
        assert target.calls == target.reads == 0

    asyncio.run(scenario())
