# Verification and restart rules

This reviewed demonstration document uses only synthetic resources.

## Dry Run

Validate inputs, resource scope, environment, evidence, permissions, approval, and all hashes before simulation. Save predictions separately from observed effects. A simulated success is not a real execution success.

The only candidate step identifiers are `validate_inputs`, `simulate`, and `verify_predictions`. These suggestions do not authorize a call. Predictions describe what a simulation expects; observed effects are independently measured facts about a target. Explaining this distinction does not assert that any real target was executed or observed.

## Independent verification

An independent observation must verify the target state for the existing isolated synthetic sandbox. Tool success alone is insufficient. Verification failure must not produce a success Receipt.

## Recovery and rollback

After restart, restore saved approval, domain, idempotency, audit, and Receipt facts from PostgreSQL. An unknown outcome or orphan claim requires a diagnostic safe hold with zero resend. Never automatically release a claim, retry an uncertain effect, or infer cross-process recovery from an in-memory snapshot. Real rollback remains disabled.
