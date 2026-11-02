"""Bounded evaluation-only monotonic spans; no payload, identifiers or URLs."""

import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")
current: ContextVar[dict | None] = ContextVar("v11_evaluation_timing", default=None)


@contextmanager
def span(stage: str) -> Iterator[None]:
    """Record success, failure and cancellation without catching or retrying them."""
    record = current.get()
    start = time.perf_counter()
    outcome = "interrupted"
    try:
        yield
        outcome = "completed"
    finally:
        if record is not None:
            end = time.perf_counter()
            if len(record["spans"]) < 256:
                record["spans"].append(
                    {
                        "stage": stage,
                        "start": start - record["origin"],
                        "end": end - record["origin"],
                        "seconds": end - start,
                        "outcome": outcome,
                    }
                )
            else:
                record["truncated"] = True


async def measured(stage: str, operation: Awaitable[T]) -> T:
    """Measure an existing await exactly once, propagating cancellation unchanged."""
    with span(stage):
        return await operation


def snapshot(record: dict) -> dict:
    """Union intervals rather than adding nested or concurrent spans."""
    elapsed = time.perf_counter() - record["origin"]
    covered = end = 0.0
    for item in sorted(record["spans"], key=lambda item: item["start"]):
        covered += max(0.0, item["end"] - max(end, item["start"]))
        end = max(end, item["end"])
    return {
        "version": "v11-evaluation-timing.1",
        "elapsed_seconds": elapsed,
        "spans": record["spans"],
        "covered_union_seconds": covered,
        "unattributed_seconds": max(0.0, elapsed - covered),
        "truncated": record["truncated"],
        "scope": "server_command_not_client_http",
        "nested_spans_not_additive": True,
    }


def evaluation_command(function: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Activate only on signed evaluation commands; persist via the private sink."""

    @wraps(function)
    async def wrapped(self: Any, identifier: str, actor: Any, body: Any) -> Any:
        if self.evaluation is None:
            return await function(self, identifier, actor, body)
        record: dict = {"origin": time.perf_counter(), "spans": [], "truncated": False}
        token = current.set(record)
        failed = True
        try:
            result = await function(self, identifier, actor, body)
            failed = False
            return result
        finally:
            current.reset(token)
            self.evaluation.save_timing(body.command_id, function.__name__, snapshot(record), failed=failed)

    return wrapped
