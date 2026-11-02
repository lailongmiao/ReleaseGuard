"""Synthetic capability, hash, transport and redaction checks."""

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.langgraph.releaseguard_identity import releaseguard_thread_id
from app.schemas.tooling import ToolDescriptor, ToolInvocation, ToolResult, ToolStatus
from app.services.releaseguard.state_machine import DomainError
from app.services.tools.registry import ToolRegistry
from tests.releaseguard_fixtures import binding
from tests.test_change_request_contract import NOW


def descriptor_fixture(**overrides: Any) -> ToolDescriptor:
    """Pin a synthetic read-only capability."""
    return ToolDescriptor.model_validate(
        dict(
            tool_name="synthetic-tool",
            version="v1",
            capabilities={"read_only"},
            resource_types={"service"},
            action_types={"inspect"},
            input_schema="inspect.v1",
            output_schema="observation.v1",
            side_effect_level="read_only",
            timeout_ms=100,
            idempotency_required=False,
            adapter_kind="internal",
            **overrides,
        )
    )


def invocation_fixture(descriptor: ToolDescriptor | None = None, **overrides: Any) -> ToolInvocation:
    """Use only synthetic identifiers and input hashes."""
    descriptor = descriptor or descriptor_fixture()
    values = dict(
        **binding(),
        invocation_id="synthetic-invocation",
        tool_name=descriptor.tool_name,
        version=descriptor.version,
        descriptor_hash=descriptor.descriptor_hash,
        input_schema=descriptor.input_schema,
        input_hash="c" * 64,
        thread_id=releaseguard_thread_id(binding()["change_id"], binding()["run_id"]),
        principal_id="synthetic-requester",
        session_id="synthetic-session",
        tenant_id="synthetic-tenant",
        resource_type="service",
        resource_id="synthetic-service",
        action="inspect",
        environment="test",
        policy_version="synthetic-policy.v1",
        correlation_id="synthetic-correlation",
        occurred_at=NOW,
    )
    return ToolInvocation.model_validate({**values, **overrides})


class FakeTool:
    """Synthetic adapter with explicit fault modes and observable call count."""

    def __init__(self, descriptor: ToolDescriptor, mode: str = "ok") -> None:
        """Retain a local pin and test-only fault selector."""
        self.pin = descriptor
        self.mode = mode
        self.calls = 0

    async def discover(self) -> ToolDescriptor:
        """Return a pin without remote access."""
        return self.pin

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Return hashes or a synthetic transport failure."""
        self.calls += 1
        if self.mode == "exception":
            raise RuntimeError("synthetic_sensitive_body_must_not_escape")
        if self.mode == "timeout":
            await asyncio.Event().wait()
        return ToolResult(
            invocation_id=invocation.invocation_id,
            status=ToolStatus.SUCCEEDED,
            input_hash=invocation.input_hash if self.mode != "mismatch" else "d" * 64,
            descriptor_hash=invocation.descriptor_hash,
            output_schema=self.pin.output_schema,
            output_hash="a" * 64,
            item_count=1,
        )


@pytest.mark.parametrize(
    "field,value",
    [("timeout_ms", 0), ("unknown", True), ("descriptor_hash", "b" * 64), ("capabilities", {"mutating"})],
)
def test_descriptor_rejects_unsafe_declarations(field: str, value: Any) -> None:
    """Invalid declarations cannot become registry pins."""
    data = descriptor_fixture().model_dump(mode="python")
    data[field] = value
    with pytest.raises(ValidationError):
        ToolDescriptor.model_validate(data)


def test_descriptor_roundtrip_and_version_hash() -> None:
    """Canonical hashes survive serialization and change with version."""
    descriptor = descriptor_fixture()
    assert ToolDescriptor.model_validate_json(descriptor.model_dump_json()) == descriptor
    data = descriptor.model_dump(exclude={"descriptor_hash"})
    data["version"] = "v2"
    assert ToolDescriptor.model_validate(data).descriptor_hash != descriptor.descriptor_hash
    with pytest.raises(ValidationError):
        ToolInvocation.model_validate({**invocation_fixture().model_dump(), "raw_body": "synthetic"})


@pytest.mark.parametrize(
    "mode,status,code",
    [
        ("ok", "succeeded", None),
        ("exception", "unavailable", "tool_adapter_failed"),
        ("timeout", "unavailable", "tool_timeout"),
        ("mismatch", "conflicting", "adapter_result_mismatch"),
    ],
)
def test_transport_boundaries(mode: str, status: str, code: str | None) -> None:
    """Only safe matching responses pass the transport boundary."""

    async def scenario() -> None:
        pin = descriptor_fixture()
        registry = ToolRegistry()
        adapter = FakeTool(pin, mode)
        await registry.register(pin, adapter)
        result = await registry.call_read_only(invocation_fixture(pin))
        assert result.status == status and result.error_code == code
        assert "synthetic_sensitive_body" not in result.model_dump_json()
        with pytest.raises(DomainError, match="registration_conflict"):
            await registry.register(pin, adapter)

    asyncio.run(scenario())


@pytest.mark.parametrize("effect", ["simulation", "mutating", "irreversible"])
def test_side_effect_adapters_are_never_called(effect: str) -> None:
    """Declarations cannot activate a Phase 6 executor in Phase 5."""

    async def scenario() -> None:
        data = descriptor_fixture().model_dump(exclude={"descriptor_hash"})
        data.update(side_effect_level=effect, capabilities={effect}, idempotency_required=True)
        pin = ToolDescriptor.model_validate(data)
        adapter = FakeTool(pin)
        registry = ToolRegistry()
        await registry.register(pin, adapter)
        assert (await registry.call_read_only(invocation_fixture(pin))).status is ToolStatus.NOT_IMPLEMENTED
        assert adapter.calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", [{"version": "v2"}, {"descriptor_hash": "d" * 64}, {"input_schema": "unknown.v1"}, {"action": "delete"}]
)
def test_unknown_or_mismatched_invocation_never_calls_adapter(change: dict[str, str]) -> None:
    """Unknown versions, schemas, hashes and scope fail before transport."""

    async def scenario() -> None:
        pin = descriptor_fixture()
        adapter = FakeTool(pin)
        registry = ToolRegistry()
        await registry.register(pin, adapter)
        result = await registry.call_read_only(invocation_fixture(pin, **change))
        assert result.status is not ToolStatus.SUCCEEDED and adapter.calls == 0

    asyncio.run(scenario())


def test_discovery_cannot_replace_local_pin() -> None:
    """A remote MCP capability response is compared, never accepted as policy."""

    async def scenario() -> None:
        data = descriptor_fixture().model_dump(exclude={"descriptor_hash"})
        data["adapter_kind"] = "mcp"
        pin = ToolDescriptor.model_validate(data)
        with pytest.raises(DomainError, match="descriptor_conflict"):
            await ToolRegistry().register(pin, FakeTool(descriptor_fixture()))

    asyncio.run(scenario())
