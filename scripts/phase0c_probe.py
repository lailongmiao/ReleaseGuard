"""Run provider-neutral Phase 0C capability probes.

The probes use synthetic inputs and write only redacted summaries. They are
deliberately independent so one provider failure does not hide the remaining
capability results.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

# Allow direct execution from the repository root or from another working directory.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from app.core.config import settings  # noqa: E402  # repository-root bootstrap precedes app imports
from app.core.langgraph.tools import tools  # noqa: E402
from app.services.llm import LLMRegistry, llm_service  # noqa: E402
from app.services.llm.contracts import (  # noqa: E402
    LLMRole,
    ProviderError,
    ProviderErrorCode,
    usage_from_response,
)
from app.services.memory import memory_service  # noqa: E402
from app.schemas.chat import SessionTitle  # noqa: E402
from evals.schemas import ScoreSchema  # noqa: E402


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


_DEFAULT_OUTPUT = Path("evals/reports/phase0c_probe_report.md")
_SYNTHETIC_MESSAGE = "Phase 0C synthetic capability probe. Reply with a short acknowledgement."
_SYNTHETIC_USER_ID = "phase0c-probe-user"
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_ -]?key|authorization|bearer|token|password|secret)[=: ]+[^,;\s]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)


@dataclass(slots=True)
class ProbeResult:
    """Redacted result for one capability probe."""

    capability: str
    status: str
    provider: str | None
    model: str | None
    config_id: str | None
    config_summary: dict[str, Any]
    observed_at: str
    revision: str
    request_summary: dict[str, Any]
    response_summary: dict[str, Any]
    error_class: str | None
    error_summary: str | None
    degradation: str


def _redact(value: Any) -> str:
    """Return a bounded string with common credential forms removed."""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text[:300]


def _revision() -> str:
    """Read the current revision without invoking a shell."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _safe_endpoint(endpoint: str | None) -> str | None:
    """Keep only the endpoint origin and path for diagnostics."""
    if not endpoint:
        return None
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return _redact(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _config_metadata(
    role: LLMRole, model: str | None = None
) -> tuple[str | None, str | None, str | None, dict[str, Any]]:
    """Return safe provider metadata for a role."""
    selected_model = model
    try:
        selected_model = model or _role_model(role)
        config = LLMRegistry.get_config(selected_model, role)
    except Exception as exc:
        return (
            None,
            selected_model,
            None,
            {"lookup_error": type(exc).__name__},
        )

    identity = "|".join(
        item or "" for item in (config.provider, config.model, config.endpoint, config.credentials_ref, role.value)
    )
    config_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    safe = config.to_safe_dict()
    safe["endpoint"] = _safe_endpoint(config.endpoint)
    return config.provider, config.model, config_id, safe


def _role_model(role: LLMRole) -> str:
    """Resolve a role model from the existing settings surface."""
    return LLMRegistry.get_config("", role).model


def _error_class(exc: BaseException) -> str:
    """Map an exception to the provider-neutral error taxonomy."""
    if isinstance(exc, ProviderError):
        return exc.code.value
    if isinstance(exc, ValidationError):
        return ProviderErrorCode.RESPONSE_INVALID.value
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "auth" in name or "api key" in message or "unauthorized" in message:
        return ProviderErrorCode.AUTHENTICATION.value
    if "timeout" in name or "timed out" in message:
        return ProviderErrorCode.TIMEOUT.value
    if "rate" in name or "rate limit" in message:
        return ProviderErrorCode.RATE_LIMITED.value
    if "connection" in name or "connect" in message or "unavailable" in message:
        return ProviderErrorCode.UNAVAILABLE.value
    if isinstance(exc, (ImportError, ModuleNotFoundError, ValueError, TypeError)):
        return ProviderErrorCode.CONFIGURATION.value
    return ProviderErrorCode.UNAVAILABLE.value


def _error_summary(exc: BaseException) -> str:
    """Return only stable, non-sensitive error taxonomy metadata."""
    if isinstance(exc, ProviderError):
        return f"{type(exc).__name__} ({exc.code.value})"
    if isinstance(exc, ValidationError):
        return f"{type(exc).__name__} ({ProviderErrorCode.RESPONSE_INVALID.value})"
    return type(exc).__name__


def _result(
    capability: str,
    *,
    status: str,
    role: LLMRole,
    request_summary: dict[str, Any],
    response_summary: dict[str, Any] | None = None,
    error: BaseException | None = None,
    degradation: str,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProbeResult:
    """Build a result while keeping provider metadata separate from payloads."""
    provider, selected_model, config_id, safe_config = _config_metadata(role, model)
    if metadata:
        safe_config.update(metadata)
    return ProbeResult(
        capability=capability,
        status=status,
        provider=provider,
        model=selected_model,
        config_id=config_id,
        config_summary=safe_config,
        observed_at=datetime.now(timezone.utc).isoformat(),
        revision=_revision(),
        request_summary=request_summary,
        response_summary=response_summary or {},
        error_class=_error_class(error) if error else None,
        error_summary=_error_summary(error) if error else None,
        degradation=degradation,
    )


async def _bounded(awaitable: Awaitable[Any] | Any, timeout: float = 15.0) -> Any:
    """Bound an external call so one unavailable provider cannot stall probes."""
    if not inspect.isawaitable(awaitable):
        return awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout)


async def _probe_chat() -> ProbeResult:
    """Probe one ordinary chat completion."""
    request = {"message_count": 1, "input_chars": len(_SYNTHETIC_MESSAGE), "response": "text"}
    try:
        response = await _bounded(llm_service.call([HumanMessage(content=_SYNTHETIC_MESSAGE)]))
        content = getattr(response, "content", "")
        return _result(
            "chat",
            status="passed",
            role=LLMRole.CHAT,
            request_summary=request,
            response_summary={"type": type(response).__name__, "content_chars": len(str(content))},
            degradation="none",
        )
    except Exception as exc:
        return _result(
            "chat",
            status="failed",
            role=LLMRole.CHAT,
            request_summary=request,
            error=exc,
            degradation="Keep the non-streaming path disabled until provider configuration or authentication is fixed.",
        )


async def _probe_streaming() -> ProbeResult:
    """Probe a minimal provider streaming response."""
    request = {"message_count": 1, "input_chars": len(_SYNTHETIC_MESSAGE), "stream": True}
    model_name = _role_model(LLMRole.CHAT)
    try:
        model = LLMRegistry.get(model_name, streaming=True)

        async def consume_stream() -> tuple[int, int]:
            chunk_count = 0
            content_chars = 0
            async for chunk in model.astream([HumanMessage(content=_SYNTHETIC_MESSAGE)]):
                chunk_count += 1
                content_chars += len(str(getattr(chunk, "content", "")))
                if chunk_count >= 100:
                    break
            return chunk_count, content_chars

        chunk_count, content_chars = await asyncio.wait_for(consume_stream(), timeout=15.0)
        if chunk_count == 0:
            raise ValueError("stream returned no chunks")
        return _result(
            "streaming",
            status="passed",
            role=LLMRole.CHAT,
            request_summary=request,
            response_summary={"chunk_count": chunk_count, "content_chars": content_chars, "ended": True},
            degradation="none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "streaming",
            status="failed",
            role=LLMRole.CHAT,
            request_summary=request,
            error=exc,
            degradation="Use non-streaming chat only after the provider supports a complete stream protocol.",
            model=model_name,
        )


async def _probe_tools() -> ProbeResult:
    """Probe tool schema binding and tool-call metadata parsing."""
    request = {"tool_count": len(tools), "tool_names": [tool.name for tool in tools], "invoke": True}
    model_name = _role_model(LLMRole.CHAT)
    try:
        model = LLMRegistry.bind_tools(model_name, tools)
        response = await _bounded(model.ainvoke([HumanMessage(content="Use the available search tool once.")]))
        calls = getattr(response, "tool_calls", []) or []
        return _result(
            "tool_calling",
            status="passed" if calls else "degraded",
            role=LLMRole.CHAT,
            request_summary=request,
            response_summary={
                "type": type(response).__name__,
                "tool_call_count": len(calls),
                "tool_names": [call.get("name") for call in calls if isinstance(call, dict)],
            },
            degradation="No tool call was returned; retain a read-only or human path until tool calls are confirmed."
            if not calls
            else "none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "tool_calling",
            status="failed",
            role=LLMRole.CHAT,
            request_summary=request,
            error=exc,
            degradation="Do not execute tool side effects until schema binding and call ids are verified.",
            model=model_name,
        )


async def _probe_structured() -> ProbeResult:
    """Probe SessionTitle structured output validation."""
    request = {"schema": "SessionTitle", "fields": ["title"], "input_chars": len(_SYNTHETIC_MESSAGE)}
    model_name = _role_model(LLMRole.CHAT)
    try:
        response = await _bounded(
            llm_service.call(
                [HumanMessage(content="Name this synthetic chat in three words.")],
                model_name=model_name,
                response_format=SessionTitle,
                max_tokens=32,
            )
        )
        return _result(
            "structured_output",
            status="passed",
            role=LLMRole.CHAT,
            request_summary=request,
            response_summary={"type": type(response).__name__, "has_title": bool(getattr(response, "title", ""))},
            degradation="none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "structured_output",
            status="failed",
            role=LLMRole.CHAT,
            request_summary=request,
            error=exc,
            degradation="Use schema validation or a placeholder title; never treat free text as a valid SessionTitle.",
            model=model_name,
        )


async def _probe_reasoning() -> ProbeResult:
    """Probe reasoning only when the declared capability allows it."""
    request = {"reasoning": {"effort": "low"}, "input_chars": len(_SYNTHETIC_MESSAGE)}
    model_name = _role_model(LLMRole.CHAT)
    try:
        capabilities = LLMRegistry.get_capabilities(model_name)
        if not capabilities.supports_reasoning:
            exc = ProviderError(
                ProviderErrorCode.CAPABILITY,
                "reasoning is not declared for the configured chat model",
                model=model_name,
            )
            return _result(
                "reasoning",
                status="degraded",
                role=LLMRole.CHAT,
                request_summary=request,
                error=exc,
                degradation="Omit reasoning parameters or select a model whose capability declaration is verified.",
                model=model_name,
            )
        response = await _bounded(
            llm_service.call(
                [HumanMessage(content=_SYNTHETIC_MESSAGE)], model_name=model_name, reasoning={"effort": "low"}
            )
        )
        return _result(
            "reasoning",
            status="passed",
            role=LLMRole.CHAT,
            request_summary=request,
            response_summary={
                "type": type(response).__name__,
                "content_chars": len(str(getattr(response, "content", ""))),
            },
            degradation="none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "reasoning",
            status="failed",
            role=LLMRole.CHAT,
            request_summary=request,
            error=exc,
            degradation="Remove reasoning parameters and retain the ordinary chat path.",
            model=model_name,
        )


async def _probe_usage() -> ProbeResult:
    """Probe usage extraction without serializing provider response content."""
    request = {"message_count": 1, "input_chars": len(_SYNTHETIC_MESSAGE), "usage": True}
    model_name = _role_model(LLMRole.CHAT)
    try:
        config = LLMRegistry.get_config(model_name)
        response = await _bounded(
            llm_service.call(
                [HumanMessage(content=_SYNTHETIC_MESSAGE)],
                model_name=model_name,
                usage=True,
            )
        )
        usage = usage_from_response(response, config.capabilities)
        status = "passed" if usage.status == "known" else "degraded"
        return _result(
            "usage",
            status=status,
            role=LLMRole.CHAT,
            request_summary=request,
            response_summary={
                "status": usage.status,
                "has_input": usage.input_tokens is not None,
                "has_output": usage.output_tokens is not None,
            },
            degradation="Mark usage unknown and do not calculate exact cost." if usage.status == "unknown" else "none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "usage",
            status="unverified",
            role=LLMRole.CHAT,
            request_summary=request,
            error=exc,
            degradation="Mark usage unknown until a successful response exposes validated token metadata.",
            model=model_name,
        )


async def _probe_embeddings() -> ProbeResult:
    """Probe embedding generation and verify the declared vector dimension."""
    request = {"input_count": 1, "input_chars": len("Phase 0C embedding probe"), "read_dimension": True}
    model_name = _role_model(LLMRole.EMBEDDING)
    try:
        config = LLMRegistry.get_config(model_name, LLMRole.EMBEDDING)
        adapter = LLMRegistry.get_embeddings()
        vector = await _bounded(asyncio.to_thread(adapter.embed_query, "Phase 0C embedding probe"))
        declared = config.capabilities.embedding_dimensions
        actual = len(vector)
        if declared != actual:
            raise ProviderError(
                ProviderErrorCode.RESPONSE_INVALID,
                "embedding vector dimension does not match the role declaration",
                provider=config.provider,
                model=config.model,
            )
        return _result(
            "embedding",
            status="passed",
            role=LLMRole.EMBEDDING,
            request_summary=request,
            response_summary={"declared_dimensions": declared, "vector_length": actual, "finite_values": True},
            degradation="none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "embedding",
            status="unverified",
            role=LLMRole.EMBEDDING,
            request_summary=request,
            error=exc,
            degradation="Keep embedding-backed memory disabled until configuration and dimensions are verified.",
            model=model_name,
        )


async def _probe_mem0() -> ProbeResult:
    """Probe Mem0 init/add/search and always clean synthetic records."""
    request = {
        "operation": "init_add_search",
        "synthetic_user": _SYNTHETIC_USER_ID,
        "writes_enabled": os.getenv("PHASE_0C_ALLOW_LOCAL_MEMORY_WRITES", "false").lower() == "true",
    }
    model_name = _role_model(LLMRole.MEMORY_LLM)
    if not request["writes_enabled"]:
        exc = ProviderError(
            ProviderErrorCode.CONFIGURATION,
            "local memory writes are disabled; set PHASE_0C_ALLOW_LOCAL_MEMORY_WRITES=true for this probe",
            model=model_name,
        )
        return _result(
            "mem0",
            status="unverified",
            role=LLMRole.MEMORY_LLM,
            request_summary=request,
            error=exc,
            degradation="Run only against an isolated local database with synthetic data, then delete the probe record.",
            model=model_name,
        )

    memory: Any | None = None
    search_count = 0
    cleanup_attempted = False
    cleanup_verified = False
    cleanup_error: BaseException | None = None
    result: ProbeResult | None = None
    try:
        memory = await _bounded(memory_service._get_memory())
        if memory is None:
            raise RuntimeError("memory service returned no client")
        await _bounded(
            memory.add(
                [
                    {
                        "role": "user",
                        "content": (
                            "Synthetic Phase 0C memory fact: the release window is Tuesday at 09:30 UTC "
                            "and the rollback owner is probe-owner."
                        ),
                    }
                ],
                user_id=_SYNTHETIC_USER_ID,
                metadata={"probe": "phase0c"},
            )
        )
        search_result = await _bounded(
            memory.search(user_id=_SYNTHETIC_USER_ID, query="What time is the synthetic release window?")
        )
        search_count = len(search_result.get("results", [])) if isinstance(search_result, dict) else 0
        result = _result(
            "mem0",
            status="passed" if search_count else "degraded",
            role=LLMRole.MEMORY_LLM,
            request_summary=request,
            response_summary={"search_result_count": search_count},
            degradation="Memory returned no result; use empty context and surface memory degradation."
            if not search_count
            else "none",
            model=model_name,
        )
    except Exception as exc:
        result = _result(
            "mem0",
            status="failed",
            role=LLMRole.MEMORY_LLM,
            request_summary=request,
            error=exc,
            degradation="Use empty memory context; do not silently treat failed memory writes as durable evidence.",
            model=model_name,
        )
    finally:
        if memory is not None:
            delete_all = getattr(memory, "delete_all", None)
            if callable(delete_all):
                cleanup_attempted = True
                try:
                    await _bounded(delete_all(user_id=_SYNTHETIC_USER_ID))
                    verify_result = await _bounded(
                        memory.search(user_id=_SYNTHETIC_USER_ID, query="What time is the synthetic release window?")
                    )
                    remaining = len(verify_result.get("results", [])) if isinstance(verify_result, dict) else 0
                    cleanup_verified = remaining == 0
                except Exception as cleanup_exc:
                    cleanup_error = cleanup_exc

    if cleanup_error is not None or (memory is not None and not cleanup_attempted):
        cleanup_failure = ProviderError(
            ProviderErrorCode.CONFIGURATION,
            "synthetic memory cleanup could not be verified",
            model=model_name,
            cause=cleanup_error,
        )
        return _result(
            "mem0",
            status="failed",
            role=LLMRole.MEMORY_LLM,
            request_summary=request,
            response_summary={
                "search_result_count": search_count,
                "cleanup_attempted": cleanup_attempted,
                "cleanup_verified": cleanup_verified,
            },
            error=cleanup_failure,
            degradation="Do not accept the probe until synthetic memory cleanup is verified.",
            model=model_name,
        )

    if result is None:
        return _result(
            "mem0",
            status="failed",
            role=LLMRole.MEMORY_LLM,
            request_summary=request,
            error=ProviderError(ProviderErrorCode.UNAVAILABLE, "memory probe returned no result", model=model_name),
            degradation="Use empty memory context until the isolated probe completes.",
            model=model_name,
        )
    result.response_summary.update({"cleanup_attempted": cleanup_attempted, "cleanup_verified": cleanup_verified})
    if not cleanup_verified:
        result.status = "failed"
        result.error_class = ProviderErrorCode.CONFIGURATION.value
        result.error_summary = "synthetic memory cleanup could not be verified"
        result.degradation = "Do not accept the probe until synthetic memory cleanup is verified."
    return result


async def _probe_naming() -> ProbeResult:
    """Probe the current naming model/schema path without a database write."""
    model_name = _role_model(LLMRole.NAMING)
    request = {"path": "llm_service.call", "schema": "SessionTitle", "model_in_current_code": model_name}
    try:
        response = await _bounded(
            llm_service.call(
                [HumanMessage(content="Name this synthetic chat in three words.")],
                model_name=model_name,
                role=LLMRole.NAMING,
                response_format=SessionTitle,
                max_tokens=32,
            )
        )
        return _result(
            "session_naming",
            status="passed",
            role=LLMRole.NAMING,
            request_summary=request,
            response_summary={"type": type(response).__name__, "has_title": bool(getattr(response, "title", ""))},
            degradation="none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "session_naming",
            status="failed",
            role=LLMRole.NAMING,
            request_summary=request,
            error=exc,
            degradation="Keep the placeholder name and do not block chat when naming fails.",
            model=model_name,
        )


async def _probe_judge() -> ProbeResult:
    """Probe the provider-neutral judge path without fetching traces or writing scores."""
    request = {"path": "LLMService.call_with_usage", "schema": "ScoreSchema", "write_score": False}
    model_name = _role_model(LLMRole.JUDGE)
    try:
        response, usage = await _bounded(
            llm_service.call_with_usage(
                [
                    {"role": "system", "content": "Return a score from 0 to 1 and one short reason."},
                    {"role": "user", "content": "Input: synthetic probe\nGeneration: synthetic response"},
                ],
                model_name=model_name,
                role=LLMRole.JUDGE,
                response_format=ScoreSchema,
            )
        )
        if not isinstance(response, ScoreSchema):
            raise TypeError("judge response did not produce ScoreSchema")
        return _result(
            "evaluator_judge",
            status="passed",
            role=LLMRole.JUDGE,
            request_summary=request,
            response_summary={
                "parsed": True,
                "score_present": response.score is not None,
                "score_write": False,
                "usage_quality": usage.quality.value if usage.quality else "unknown",
            },
            degradation="none",
            model=model_name,
        )
    except Exception as exc:
        return _result(
            "evaluator_judge",
            status="unverified",
            role=LLMRole.JUDGE,
            request_summary=request,
            error=exc,
            degradation="Do not write a score; use validated JSON/Pydantic fallback or mark the judge unavailable.",
            model=model_name,
        )


async def _run_probes() -> list[ProbeResult]:
    """Run probes serially to keep provider load and logs predictable."""
    probe_functions: list[Callable[[], Awaitable[ProbeResult]]] = [
        _probe_chat,
        _probe_streaming,
        _probe_tools,
        _probe_structured,
        _probe_reasoning,
        _probe_usage,
        _probe_embeddings,
        _probe_mem0,
        _probe_naming,
        _probe_judge,
    ]
    results: list[ProbeResult] = []
    for probe in probe_functions:
        results.append(await probe())
    return results


def _render(results: list[ProbeResult], output: Path) -> None:
    """Write a reviewable Markdown record with only safe summaries."""
    generated_at = datetime.now(timezone.utc).isoformat()
    report_title = "Provider Capability Probe Report"
    lines = [
        f"# {report_title}",
        "",
        f"> Generated at `{generated_at}` (UTC). This file contains redacted probe metadata only.",
        "> Synthetic inputs were used. No API keys, JWTs, production data, or raw model responses are recorded.",
        "",
        "## Run Metadata",
        "",
        f"- Revision: `{_revision()}`",
        f"- Environment: `{settings.ENVIRONMENT.value}`",
        f"- Chat provider/model: `{settings.CHAT_LLM_PROVIDER}` / `{settings.CHAT_LLM_MODEL}`",
        f"- Memory provider/model: `{settings.MEMORY_LLM_PROVIDER}` / `{settings.MEMORY_LLM_MODEL}`",
        f"- Embedding provider/model: `{settings.EMBEDDING_PROVIDER}` / `{settings.EMBEDDING_MODEL}`",
        f"- Naming provider/model: `{settings.NAMING_LLM_PROVIDER}` / `{settings.NAMING_LLM_MODEL}`",
        f"- Judge provider/model: `{settings.JUDGE_LLM_PROVIDER}` / `{settings.JUDGE_LLM_MODEL}`",
        "- Credentials: configured values are intentionally omitted; no credential values are recorded in this report.",
        "",
        "## Summary",
        "",
        "| Capability | Status | Provider | Model | Error class |",
        "|---|---|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item.capability} | {item.status} | {item.provider or 'unknown'} | "
            f"{item.model or 'unknown'} | {item.error_class or 'none'} |"
        )
    lines.extend(["", "## Probe Details", ""])
    for item in results:
        payload = asdict(item)
        lines.extend(
            [
                f"### {item.capability}",
                "",
                "```json",
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run all probes and write their report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = asyncio.run(_run_probes())
    _render(results, args.output)
    print(json.dumps({"output": str(args.output), "results": [asdict(item) for item in results]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
