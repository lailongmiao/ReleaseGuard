"""Telemetry boundaries use synthetic sentinels, never live credentials."""

import asyncio
import json
import logging

from app.core.logging import SafeRecordFilter, bind_context, clear_context, get_context, logger
from app.core.telemetry import mask_trace, reference, sanitize_event


def test_log_and_trace_redact_nested_text_and_exception() -> None:
    """Messages/locals/PII and nested keys do not cross either boundary."""
    sentinel = "synthetic private payload@example.invalid"
    try:
        raise ValueError(sentinel)
    except ValueError:
        payload = sanitize_event(
            None,
            "exception",
            {
                "event": "probe_failed",
                "exc_info": True,
                "prompt": sentinel,
                "user_id": sentinel,
                "metadata": {"raw_response": sentinel, "password": sentinel},
            },
        )
    assert sentinel not in json.dumps(payload)
    assert payload["error_category"] == "ValueError"
    assert payload["user_id"] == reference(sentinel)
    assert sentinel not in json.dumps(mask_trace(data={"input": sentinel, "metadata": {"token": sentinel}}))


def test_foreign_logging_does_not_render_raw_exception() -> None:
    """The stdlib handler boundary suppresses unsanitized SDK text."""
    record = logging.LogRecord("sdk", logging.ERROR, "private-path", 1, "synthetic private %s", ("payload",), None)
    assert SafeRecordFilter().filter(record)
    assert record.getMessage() == "external_log_record"


def test_actual_bound_logger_exception_cannot_reattach_raw_traceback(caplog) -> None:
    """stdlib.exception otherwise attaches exc_info after structlog processors."""
    try:
        raise ValueError("synthetic-private-sentinel")
    except ValueError:
        logger.exception("actual_exception_probe")
    assert "synthetic-private-sentinel" not in caplog.text
    assert all(not record.exc_info for record in caplog.records)


def test_cyclic_untrusted_payload_is_bounded() -> None:
    """A telemetry cycle cannot overflow the stack or expose arbitrary data."""
    value = {}
    value["metadata"] = value
    result = mask_trace(data=value)
    assert len(json.dumps(result)) < 300


def test_concurrent_context_and_cancel_cleanup() -> None:
    """Separate tasks and cancellation leave no request context behind."""

    async def case() -> None:
        entered = asyncio.Event()

        async def worker(name: str) -> str:
            clear_context()
            try:
                bind_context(run_id=name)
                entered.set()
                await asyncio.sleep(0)
                return get_context()["run_id"]
            finally:
                clear_context()
                assert get_context() == {}

        assert await asyncio.gather(worker("case-a"), worker("case-b")) == ["case-a", "case-b"]
        task = asyncio.create_task(worker("cancel-case"))
        await entered.wait()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert get_context() == {}

    asyncio.run(case())
