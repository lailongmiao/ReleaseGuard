"""Bounded, non-authoritative telemetry; never serialize arbitrary business data."""

import hashlib
import re
import sys
import traceback
from typing import Any

SAFE_WORD = re.compile(r"[A-Za-z0-9_.:-]{1,80}\Z")
EVENT_WORD = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
IDS = frozenset(
    {
        "request_id",
        "session_id",
        "user_id",
        "principal_id",
        "tenant_id",
        "change_id",
        "run_id",
        "thread_id",
        "step_id",
        "approval_id",
        "execution_id",
        "receipt_id",
        "correlation_id",
        "event_id",
        "analysis_id",
        "source_id",
        "index_generation",
        "prompt_version",
        "source_hash",
    }
)
WORDS = frozenset(
    {
        "event",
        "level",
        "logger",
        "environment",
        "status",
        "outcome",
        "category",
        "role",
        "provider",
        "model",
        "usage_quality",
        "availability",
        "error_category",
        "error_code",
        "component",
        "backend",
        "reason_code",
        "version",
        "workflow_kind",
        "log_level",
        "log_format",
        "method",
    }
)
NUMBERS = frozenset(
    {
        "revision",
        "state_version",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "latency_ms",
        "duration_seconds",
        "count",
        "error_count",
        "message_count",
        "tool_count",
        "ttl",
        "lineno",
        "debug",
        "retryable",
        "enabled",
        "ready",
    }
)


def reference(value: object) -> str:
    """Pseudonymize opaque identifiers, including untrusted identity input."""
    return "ref:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def safe_payload(value: Any, depth: int = 0) -> Any:
    """Recursive allowlist. Unknown text/objects are discarded, never repr'd."""
    if depth >= 6:
        return "redacted"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            if not isinstance(key, str):
                continue
            if key in IDS and item is not None:
                result[key] = reference(item)
            elif key in WORDS and isinstance(item, str):
                rule = EVENT_WORD if key == "event" else SAFE_WORD
                result[key] = item if rule.fullmatch(item) else "redacted"
            elif key in NUMBERS and (item is None or isinstance(item, (int, float, bool))):
                result[key] = item
            elif key == "timestamp" and isinstance(item, str) and re.fullmatch(r"[0-9T:.+Z-]{10,40}", item):
                result[key] = item
            elif isinstance(item, (dict, list, tuple)):
                result[key if key in {"metadata", "components", "context"} else "redacted"] = safe_payload(
                    item, depth + 1
                )
        return result
    if isinstance(value, (list, tuple)):
        return [safe_payload(item, depth + 1) for item in value[:100]]
    return "redacted"


def sanitize_event(_: Any, __: str, event: dict[str, Any]) -> dict[str, Any]:
    """Keep exception type and frame names without messages, locals or paths."""
    info = event.pop("exc_info", None)
    if info:
        exc = sys.exc_info() if info is True else info
        if isinstance(exc, BaseException):
            exc = (type(exc), exc, exc.__traceback__)
        if isinstance(exc, tuple) and len(exc) == 3 and exc[0]:
            event["error_category"] = exc[0].__name__
            frames = traceback.extract_tb(exc[2]) if exc[2] else []
            event["exception_frames"] = [{"event": f.name, "lineno": f.lineno} for f in frames[-12:]]
    return safe_payload(event)


def mask_trace(*, data: Any, **_: Any) -> Any:
    """SDK mask boundary: no prompt, model content, credentials or arbitrary text."""
    return safe_payload(data)
