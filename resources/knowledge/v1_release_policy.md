# Release policy for the readonly demonstration

This reviewed document contains no credentials, private repository material, or personal data.

## CI evidence

A PR must be bound to its current head SHA. Required checks must be declared before evaluating outcomes. All declared checks must be present, completed, and successful. Empty, truncated, stale, conflicting, failed, cancelled, skipped, or neutral checks cannot authorize preparation. Collection success does not mean CI success.

## Risk and approval

Development risk has a MEDIUM floor. Staging risk has a HIGH floor. Missing facts or unresolved uncertainty require UNKNOWN and a safe stop. The requester cannot approve their own change. A different designated approver needs explicit access to both sources.

## Execution boundary

The readonly product allows simulation only. GitHub writes, target writes, production execution, and arbitrary commands are disabled. Approval never grants additional tool capabilities.
