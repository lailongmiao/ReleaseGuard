"""Pure default-deny authorization; identity is supplied by a trusted resolver."""

from datetime import datetime

from app.schemas.authorization import PermissionDecision, PermissionEffect, PermissionPolicy, RequestContext
from app.schemas.tooling import ToolDescriptor
from app.services.releaseguard.normalization import canonical_hash


def evaluate_permission(
    context: RequestContext, descriptor: ToolDescriptor, policy: PermissionPolicy, *, at: datetime
) -> PermissionDecision:
    """Check identity, tenant ownership and local capability before matching grants."""
    context = RequestContext.model_validate_json(context.model_dump_json())
    descriptor = ToolDescriptor.model_validate_json(descriptor.model_dump_json())
    policy = PermissionPolicy.model_validate_json(policy.model_dump_json())
    principal, call = context.principal, context.invocation
    effect, code = PermissionEffect.DENY, "permission_denied"
    if at.tzinfo is None:
        raise ValueError("authorization_time_requires_timezone")
    valid = (
        principal.authenticated_at <= at < principal.expires_at
        and call.occurred_at <= at
        and bool(principal.roles)
        and (call.principal_id, call.session_id, call.tenant_id)
        == (principal.principal_id, principal.session_id, principal.tenant_id)
        and call.tenant_id == context.resource_tenant_id
        and (call.tool_name, call.version, call.descriptor_hash, call.input_schema)
        == (descriptor.tool_name, descriptor.version, descriptor.descriptor_hash, descriptor.input_schema)
        and call.resource_type in descriptor.resource_types
        and call.action in descriptor.action_types
        and call.policy_version == policy.policy_version
    )
    if valid and not policy.available:
        effect, code = PermissionEffect.UNAVAILABLE, "policy_unavailable"
    elif valid:
        matches = {
            rule.effect
            for rule in policy.rules
            if rule.tenant_id == principal.tenant_id
            and (
                rule.public
                or principal.principal_id in rule.principals
                or bool(principal.roles & rule.roles)
                or bool(principal.groups & rule.groups)
            )
            and rule.resource_type == call.resource_type
            and call.resource_id in rule.resource_ids
            and call.action in rule.actions
            and call.environment in rule.environments
            and descriptor.side_effect_level in rule.side_effects
        }
        if len(matches) > 1:
            effect, code = PermissionEffect.POLICY_CONFLICT, "policy_conflict"
        elif matches:
            effect = next(iter(matches))
            code = "permission_" + effect.value
    return PermissionDecision(
        effect=effect,
        reason_code=code,
        policy_version=policy.policy_version,
        policy_hash=canonical_hash(policy),
        descriptor_hash=descriptor.descriptor_hash,
        input_hash=call.input_hash,
        evaluated_at=at,
    )
