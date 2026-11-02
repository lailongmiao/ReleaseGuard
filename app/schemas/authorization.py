"""Trusted identity input and deterministic, tenant-scoped permission contracts."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from app.schemas.releaseguard import DomainModel, Identifier, Sha256, Timestamp
from app.schemas.tooling import SafeCode, SideEffect, ToolInvocation

Names = Annotated[frozenset[Identifier], Field(max_length=64)]


class VerifiedPrincipal(DomainModel):
    """Created by an injected authentication boundary, never by graph messages."""

    principal_id: Identifier
    session_id: Identifier
    tenant_id: Identifier
    roles: Names
    groups: Names = frozenset()
    authenticated_at: Timestamp
    expires_at: Timestamp

    @model_validator(mode="after")
    def check_times(self) -> "VerifiedPrincipal":
        """Reject an empty authentication interval."""
        if self.expires_at <= self.authenticated_at:
            raise ValueError("authentication_interval_invalid")
        return self


class RequestContext(DomainModel):
    """Trusted resolver joins identity and resource ownership with a safe request."""

    principal: VerifiedPrincipal
    invocation: ToolInvocation
    resource_tenant_id: Identifier
    product_hash: Sha256 | None = None
    product_binding_version: Literal["v2", "v3"] = "v2"

    @model_serializer(mode="wrap")
    def old_context_encoding(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Do not add a default field to historical context encodings."""
        result = handler(self)
        if self.product_binding_version == "v2":
            result.pop("product_binding_version", None)
        return result


class PermissionEffect(StrEnum):
    """No implicit allow or permissive fallback exists."""

    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"
    UNAVAILABLE = "unavailable"
    POLICY_CONFLICT = "policy_conflict"


class PermissionRule(DomainModel):
    """Exact object/action/environment scope; no wildcard resource expansion."""

    rule_id: Identifier
    tenant_id: Identifier
    principals: Names = frozenset()
    roles: Names = frozenset()
    groups: Names = frozenset()
    public: bool = Field(default=False, strict=True)
    resource_type: Identifier
    resource_ids: Annotated[Names, Field(min_length=1)]
    actions: Annotated[Names, Field(min_length=1)]
    environments: Annotated[Names, Field(min_length=1)]
    side_effects: Annotated[frozenset[SideEffect], Field(min_length=1)]
    effect: PermissionEffect

    @model_validator(mode="after")
    def check_effect(self) -> "PermissionRule":
        """Operational errors cannot be configured as grant rules."""
        if self.effect not in {PermissionEffect.ALLOW, PermissionEffect.DENY, PermissionEffect.APPROVAL_REQUIRED}:
            raise ValueError("permission_rule_invalid")
        return self


class PermissionPolicy(DomainModel):
    """A locally supplied policy snapshot; adapters cannot author it."""

    policy_version: Identifier
    rules: Annotated[tuple[PermissionRule, ...], Field(max_length=256)] = ()
    available: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def check_rules(self) -> "PermissionPolicy":
        """Rule identifiers must be unique within a version."""
        if len({rule.rule_id for rule in self.rules}) != len(self.rules):
            raise ValueError("permission_rule_conflict")
        return self


class PermissionDecision(DomainModel):
    """Sealed policy attribution and safe denial codes, without resource contents."""

    effect: PermissionEffect
    reason_code: SafeCode
    policy_version: Identifier
    policy_hash: Sha256
    descriptor_hash: Sha256
    input_hash: Sha256
    evaluated_at: Timestamp
