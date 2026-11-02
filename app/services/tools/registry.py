"""Locally pinned tool registry; adapters have no authority over policy."""

import asyncio
from typing import Protocol

import structlog

from app.schemas.tooling import SideEffect, ToolDescriptor, ToolInvocation, ToolResult, ToolStatus
from app.services.releaseguard.state_machine import DomainError

logger = structlog.get_logger(__name__)


class ToolAdapter(Protocol):
    """Internal/MCP transport contract; no credentials in invocation objects."""

    async def discover(self) -> ToolDescriptor:
        """Return declared capabilities for comparison with the local pin."""
        ...

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Resolve safe input references externally and return bounded metadata."""
        ...


class MCPToolAdapter(ToolAdapter, Protocol):
    """MCP transports implement the same locally constrained contract."""


class ToolRegistry:
    """Versioned in-process registry; never silently replaces a descriptor."""

    def __init__(self) -> None:
        """Start without any tools or implicitly trusted discovery."""
        self._entries: dict[tuple[str, str], tuple[ToolDescriptor, ToolAdapter]] = {}

    async def register(self, descriptor: ToolDescriptor, adapter: ToolAdapter) -> None:
        """Verify discovery against a local pin before making a tool visible."""
        descriptor = ToolDescriptor.model_validate_json(descriptor.model_dump_json())
        key = (descriptor.tool_name, descriptor.version)
        if key in self._entries:
            raise DomainError("tool_registration_conflict")
        try:
            async with asyncio.timeout(descriptor.timeout_ms / 1000):
                discovered = await adapter.discover()
                discovered = ToolDescriptor.model_validate_json(discovered.model_dump_json())
        except Exception:
            raise DomainError("tool_discovery_unavailable") from None
        if discovered != descriptor:
            raise DomainError("tool_descriptor_conflict")
        if key in self._entries:
            raise DomainError("tool_registration_conflict")
        self._entries[key] = (descriptor, adapter)

    def descriptor(self, name: str, version: str) -> ToolDescriptor:
        """Return an isolated pin or a stable unknown-version error."""
        entry = self._entries.get((name, version))
        if entry is None:
            raise DomainError("tool_unknown_version")
        return ToolDescriptor.model_validate_json(entry[0].model_dump_json())

    async def call_read_only(self, invocation: ToolInvocation) -> ToolResult:
        """Transport boundary called only after the protected service authorizes it."""
        invocation = ToolInvocation.model_validate_json(invocation.model_dump_json())
        entry = self._entries.get((invocation.tool_name, invocation.version))
        if entry is None:
            return tool_failure(invocation, "unregistered.v1", ToolStatus.REJECTED, "tool_unknown_version")
        descriptor, adapter = entry
        if invocation.descriptor_hash != descriptor.descriptor_hash:
            return tool_failure(invocation, descriptor.output_schema, ToolStatus.CONFLICTING, "descriptor_mismatch")
        if (
            invocation.input_schema != descriptor.input_schema
            or invocation.resource_type not in descriptor.resource_types
            or invocation.action not in descriptor.action_types
        ):
            return tool_failure(invocation, descriptor.output_schema, ToolStatus.REJECTED, "schema_scope_mismatch")
        if descriptor.side_effect_level is not SideEffect.READ_ONLY:
            return tool_failure(
                invocation, descriptor.output_schema, ToolStatus.NOT_IMPLEMENTED, "execution_not_implemented"
            )
        try:
            async with asyncio.timeout(descriptor.timeout_ms / 1000):
                result = await adapter.invoke(invocation)
                result = ToolResult.model_validate_json(result.model_dump_json())
            if (
                result.invocation_id != invocation.invocation_id
                or result.input_hash != invocation.input_hash
                or result.descriptor_hash != invocation.descriptor_hash
                or result.output_schema != descriptor.output_schema
                or result.audit_ids
            ):
                return tool_failure(
                    invocation, descriptor.output_schema, ToolStatus.CONFLICTING, "adapter_result_mismatch"
                )
            return result
        except TimeoutError:
            return tool_failure(invocation, descriptor.output_schema, ToolStatus.UNAVAILABLE, "tool_timeout")
        except Exception:
            # Do not log the transport exception: it may contain response bodies.
            logger.info("tool_adapter_failed", descriptor_hash=descriptor.descriptor_hash)
            return tool_failure(invocation, descriptor.output_schema, ToolStatus.UNAVAILABLE, "tool_adapter_failed")


def tool_failure(invocation: ToolInvocation, schema: str, status: ToolStatus, code: str) -> ToolResult:
    """Build a result that cannot echo external messages or raw inputs."""
    return ToolResult(
        invocation_id=invocation.invocation_id,
        status=status,
        input_hash=invocation.input_hash,
        descriptor_hash=invocation.descriptor_hash,
        output_schema=schema,
        error_code=code,
    )
