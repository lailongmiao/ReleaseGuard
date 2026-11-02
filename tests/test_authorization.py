"""Default-deny, object-scoped authorization using synthetic verified inputs."""

from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.authorization import PermissionPolicy, PermissionRule, RequestContext, VerifiedPrincipal
from app.services.authorization import evaluate_permission
from tests.test_change_request_contract import NOW
from tests.test_tool_contract import descriptor_fixture, invocation_fixture


def principal_fixture(**overrides: Any) -> VerifiedPrincipal:
    """Represent the output of a synthetic authentication resolver."""
    return VerifiedPrincipal.model_validate(
        {
            "principal_id": "synthetic-requester",
            "session_id": "synthetic-session",
            "tenant_id": "synthetic-tenant",
            "roles": {"requester", "approver"},
            "groups": {"synthetic-team"},
            "authenticated_at": NOW,
            "expires_at": NOW + timedelta(hours=1),
            **overrides,
        }
    )


def context_fixture(**overrides: Any) -> RequestContext:
    """Bind a verified actor and explicit resource ownership."""
    return RequestContext.model_validate(
        {
            "principal": principal_fixture(),
            "invocation": invocation_fixture(),
            "resource_tenant_id": "synthetic-tenant",
            **overrides,
        }
    )


def rule_fixture(**overrides: Any) -> PermissionRule:
    """Grant one exact synthetic object and action."""
    return PermissionRule.model_validate(
        {
            "rule_id": "synthetic-rule",
            "tenant_id": "synthetic-tenant",
            "roles": {"requester", "approver"},
            "resource_type": "service",
            "resource_ids": {"synthetic-service"},
            "actions": {"inspect"},
            "environments": {"test"},
            "side_effects": {"read_only"},
            "effect": "allow",
            **overrides,
        }
    )


def policy_fixture(**overrides: Any) -> PermissionPolicy:
    """Build a local policy independent of the adapter."""
    return PermissionPolicy.model_validate(
        {"policy_version": "synthetic-policy.v1", "rules": (rule_fixture(),), **overrides}
    )


@pytest.mark.parametrize(
    "selector",
    [
        {"public": True},
        {"principals": {"synthetic-requester"}},
        {"roles": {"requester"}},
        {"groups": {"synthetic-team"}},
    ],
)
def test_explicit_grants(selector: dict[str, Any]) -> None:
    """Every selector still requires a valid identity and same-tenant object."""
    rule = rule_fixture(roles=set(), **selector) if "roles" not in selector else rule_fixture(**selector)
    result = evaluate_permission(context_fixture(), descriptor_fixture(), policy_fixture(rules=(rule,)), at=NOW)
    assert result.effect == "allow"


@pytest.mark.parametrize(
    "principal_change",
    [
        {"roles": set()},
        {"tenant_id": "other-tenant"},
        {"session_id": "other-session"},
        {"principal_id": "other-actor"},
        {"expires_at": NOW + timedelta(seconds=1)},
    ],
)
def test_invalid_actor_is_denied(principal_change: dict[str, Any]) -> None:
    """Unverified bindings and expired sessions receive an opaque refusal."""
    at = NOW + timedelta(seconds=2)
    result = evaluate_permission(
        context_fixture(principal=principal_fixture(**principal_change)), descriptor_fixture(), policy_fixture(), at=at
    )
    assert result.effect == "deny" and result.reason_code == "permission_denied"


@pytest.mark.parametrize(
    "change",
    [
        {"resource_id": "other-service"},
        {"action": "delete"},
        {"environment": "production"},
        {"policy_version": "unknown.v1"},
        {"descriptor_hash": "d" * 64},
    ],
)
def test_object_and_capability_scope(change: dict[str, Any]) -> None:
    """An identity grant cannot expand the target scope."""
    context = context_fixture(invocation=invocation_fixture(**change))
    assert evaluate_permission(context, descriptor_fixture(), policy_fixture(), at=NOW).effect == "deny"


def test_default_deny_conflict_and_unavailability() -> None:
    """Missing policy, conflicting grants and unavailable policy never allow."""
    context, descriptor = context_fixture(), descriptor_fixture()
    assert evaluate_permission(context, descriptor, policy_fixture(rules=()), at=NOW).effect == "deny"
    conflict = policy_fixture(rules=(rule_fixture(), rule_fixture(rule_id="denial", effect="deny")))
    assert evaluate_permission(context, descriptor, conflict, at=NOW).effect == "policy_conflict"
    assert evaluate_permission(context, descriptor, policy_fixture(available=False), at=NOW).effect == "unavailable"
    assert (
        evaluate_permission(context_fixture(resource_tenant_id="other"), descriptor, policy_fixture(), at=NOW).effect
        == "deny"
    )
    assert (
        evaluate_permission(
            context, descriptor, policy_fixture(rules=(rule_fixture(effect="approval_required"),)), at=NOW
        ).effect
        == "approval_required"
    )


@pytest.mark.parametrize(
    "field", ["principal_id", "session_id", "tenant_id", "roles", "authenticated_at", "expires_at"]
)
def test_incomplete_identity_cannot_be_constructed(field: str) -> None:
    """No JWT subject-only or inferred identity fallback is permitted."""
    values = principal_fixture().model_dump()
    values.pop(field)
    with pytest.raises(ValidationError):
        VerifiedPrincipal.model_validate(values)
