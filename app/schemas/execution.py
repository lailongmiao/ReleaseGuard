"""Versioned control values for synthetic execution, never parallel domain models."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.releaseguard.normalization import canonical_hash

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class ControlValue(BaseModel):
    """Frozen bounded control data; arbitrary scripts and response bodies are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ExecutionMode(StrEnum):
    """Live modes are identifiable but unsupported by the offline runtime."""

    SIMULATION = "offline_simulation"
    SANDBOX = "offline_sandbox"
    AUTHORIZED_TEST = "authorized_test"


class Direction(StrEnum):
    """Compensation has its own authority, key and immutable facts."""

    FORWARD = "forward"
    COMPENSATION = "compensation"


class SandboxInput(ControlValue):
    """Only a finite synthetic integer register operation is executable."""

    schema_version: Literal["sandbox-register.v1"] = "sandbox-register.v1"
    expected_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]
    desired_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]


class ManifestStep(ControlValue):
    """Exact approved adapter/input/target pin, including dependency and reversal."""

    step_id: Name
    order: Annotated[int, Field(strict=True, ge=1)]
    direction: Direction = Direction.FORWARD
    tool_name: Name
    version: Name
    descriptor_hash: Digest
    input_schema: Literal["sandbox-register.v1"] = "sandbox-register.v1"
    input_hash: Digest
    input_ref: Name
    resource_type: Literal["synthetic_register"] = "synthetic_register"
    resource_id: Name
    action: Literal["compare_set"] = "compare_set"
    environment: Literal["development", "staging"]
    dependencies: tuple[Name, ...] = ()
    compensates: Name | None = None
    expected_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]
    desired_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]

    @model_validator(mode="after")
    def check_input(self) -> "ManifestStep":
        """The resolver input seal must also match the declared target prediction."""
        expected = SandboxInput(expected_value=self.expected_value, desired_value=self.desired_value)
        if canonical_hash(expected) != self.input_hash:
            raise ValueError("manifest_input_mismatch")
        if (self.direction is Direction.COMPENSATION) != (self.compensates is not None):
            raise ValueError("manifest_compensation_mismatch")
        if len(set(self.dependencies)) != len(self.dependencies) or self.step_id in self.dependencies:
            raise ValueError("manifest_dependency_conflict")
        return self


class ExecutionManifest(ControlValue):
    """Sealed executable meaning; unknown versions never infer legacy permission."""

    version: Literal["execution-manifest.v1"] = "execution-manifest.v1"
    mode: ExecutionMode
    steps: Annotated[tuple[ManifestStep, ...], Field(min_length=1, max_length=64)]
    verification_adapter: Literal["independent-register.v1"] = "independent-register.v1"
    policy_version: Name
    manifest_hash: Digest = "0" * 64

    @model_validator(mode="after")
    def seal(self) -> "ExecutionManifest":
        """Reject missing dependency/reversal pins and non-reversible substitutions."""
        by_id = {step.step_id: step for step in self.steps}
        if len(by_id) != len(self.steps):
            raise ValueError("manifest_duplicate_step")
        forward = [step for step in self.steps if step.direction is Direction.FORWARD]
        if not forward or len({step.order for step in forward}) != len(forward):
            raise ValueError("manifest_order_conflict")
        reversed_ids: set[str] = set()
        for direction in Direction:
            orders = [step.order for step in self.steps if step.direction is direction]
            if len(set(orders)) != len(orders):
                raise ValueError("manifest_order_conflict")
        for step in self.steps:
            for identifier in step.dependencies:
                dependency = by_id.get(identifier)
                if dependency is None or dependency.direction is not step.direction or dependency.order >= step.order:
                    raise ValueError("manifest_dependency_conflict")
            if step.compensates:
                original = by_id.get(step.compensates)
                if original is None or original.direction is not Direction.FORWARD or step.compensates in reversed_ids:
                    raise ValueError("manifest_compensation_mismatch")
                if (step.resource_id, step.environment, step.expected_value, step.desired_value) != (
                    original.resource_id,
                    original.environment,
                    original.desired_value,
                    original.expected_value,
                ):
                    raise ValueError("manifest_compensation_mismatch")
                reversed_ids.add(step.compensates)
        compensation = {step.compensates: step for step in self.steps if step.compensates}
        for original in forward:
            for dependency in original.dependencies:
                if original.step_id in compensation and dependency in compensation:
                    later, earlier = compensation[original.step_id], compensation[dependency]
                    if later.order >= earlier.order or later.step_id not in earlier.dependencies:
                        raise ValueError("manifest_reverse_dependency_conflict")

        digest = canonical_hash(self.model_dump(exclude={"manifest_hash"}))
        if self.manifest_hash not in {"0" * 64, digest}:
            raise ValueError("manifest_hash_mismatch")
        object.__setattr__(self, "manifest_hash", digest)
        return self


class ExecutionRequest(ControlValue):
    """A bounded intent; authentication and authority remain outside this value."""

    command_id: Name
    change_id: Name
    run_id: Name
    revision: Annotated[int, Field(strict=True, ge=1)]
    request_hash: Digest
    thread_id: Name
    state_version: Annotated[int, Field(strict=True, ge=0)]
    plan_hash: Digest
    manifest_hash: Digest
    mode: ExecutionMode
    approval_request_id: Name
    step_id: Name
    direction: Direction = Direction.FORWARD


class SimulationPrediction(ControlValue):
    """Bounded synthetic prediction without arbitrary tool payloads."""

    step_id: Name
    resource_id: Name
    observed_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]
    expected_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]
    desired_value: Annotated[int, Field(strict=True, ge=-1000000, le=1000000)]
    passed: bool


class ExecutionIntent(ControlValue):
    """Typed intent only; no actor, approval decision or execution fact."""

    request: ExecutionRequest
    action: Literal["continue", "reconcile", "recover"] = "continue"


class StepClaim(ControlValue):
    """A single-owner effect intent; expiry is never permission to resend."""

    claim_id: Digest
    key: Digest
    payload_hash: Digest
    owner: Name
    fence: Literal[1] = 1
    tenant_id: Name
    run_id: Name
    revision: Annotated[int, Field(strict=True, ge=1)]
    plan_hash: Digest
    step_id: Name
    direction: Direction
    claimed_at: datetime

    @model_validator(mode="after")
    def check_time(self) -> "StepClaim":
        """Claims retain an explicit aware clock."""
        if self.claimed_at.tzinfo is None:
            raise ValueError("claim_time_invalid")
        key = canonical_hash(
            self.model_dump(include={"tenant_id", "run_id", "revision", "plan_hash", "step_id", "direction"})
        )
        if self.key != key or self.claim_id != canonical_hash({"key": key, "payload": self.payload_hash}):
            raise ValueError("claim_hash_invalid")
        return self
