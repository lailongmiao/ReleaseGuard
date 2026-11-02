"""Independent reads; executor response objects cannot serve as observations."""

from datetime import datetime
from typing import Protocol

from app.schemas.tooling import ToolDescriptor
from app.services.tools.execution_adapter import TargetObservation
from app.services.tools.sandbox_executor import SyntheticTargetStore


def observation_descriptor() -> ToolDescriptor:
    """Explicitly read-only capability for verification and uncertain diagnostics."""
    return ToolDescriptor.model_validate(
        dict(
            tool_name="independent-register",
            version="v1",
            capabilities={"read_only"},
            resource_types={"synthetic_register"},
            action_types={"inspect"},
            input_schema="register-observation.v1",
            output_schema="register-observation.v1",
            side_effect_level="read_only",
            timeout_ms=1000,
            idempotency_required=False,
            adapter_kind="internal",
        )
    )


class VerificationAdapter(Protocol):
    """A separately configured reader with a pinned source boundary."""

    source_pin: str

    async def observe(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Observe the target, without consulting an execution response."""
        ...


class IndependentRegisterReader:
    """Offline independence means a separate target read, not a production guarantee."""

    source_pin = "independent-register.v1"

    def __init__(self, targets: SyntheticTargetStore) -> None:
        """Inject the target read surface rather than the executor."""
        self.targets = targets

    async def observe(self, resource_id: str, *, at: datetime) -> TargetObservation:
        """Read authoritative synthetic target state."""
        return await self.targets.read(resource_id, at=at)
