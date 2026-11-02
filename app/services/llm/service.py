"""Provider-neutral LLM service with bounded retries and request-local fallback."""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any, Literal, TypeVar, overload

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, before_sleep_log, retry_if_exception
from tenacity import stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import observe_usage
from app.core.observability import get_langfuse_callback_handler
from app.core.telemetry import safe_payload
from app.services.llm.review_wait import BUSINESS_MODEL_WAIT_SECONDS, BUSINESS_WAIT_POLICY, BusinessWaitPolicy
from app.services.llm.contracts import (
    LLMRole,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
    Availability,
    UsageInfo,
    capability_error,
    usage_from_response,
)
from app.services.llm.registry import LLMRegistry

T = TypeVar("T", bound=BaseModel)


def pinned_failure_diagnostic(
    error: ProviderError, config: ProviderConfig, stage: str, started: float
) -> dict[str, Any]:
    """Copy only bounded transport facts; never stringify an exception or its body."""
    cause = error.cause
    request_id = cause.request_id if isinstance(cause, APIStatusError) else None
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        request_id = None
    if request_id and (
        request_id.startswith(("sk-", "ghp_", "github_pat_", "eyJ"))
        or (config.api_key and config.api_key in request_id)
    ):
        request_id = None
    return {
        "version": "v11-provider-failure.1",
        "stage": stage,
        "error_code": error.code.value,
        "elapsed_ms": min(86400000, max(0, round((perf_counter() - started) * 1000, 3))),
        "http_status": cause.status_code
        if isinstance(cause, APIStatusError) and 100 <= cause.status_code <= 599
        else None,
        "provider_request_id": request_id,
        "response_received": True if isinstance(cause, APIStatusError) else None,
        "provider_processing": "unknown",
        "usage_known": False,
        "timeout_source": "sdk"
        if isinstance(cause, APITimeoutError)
        else "outer_wait"
        if isinstance(cause, TimeoutError)
        else None,
    }


class LLMService:
    """Route chat requests through provider-neutral registry contracts.

    The default runnable is retained for the LangGraph tool path. Fallback
    candidates are constructed inside each request and therefore cannot alter
    another request's default model or tool bindings.
    """

    def __init__(self) -> None:
        """Initialize the configured default chat model or fail explicitly."""
        self._llm: Any = None
        self._bound_tools: list[Any] = []
        self._default_model = ""
        if settings.LOCAL_MVP_ONLY:
            logger.info("local_mvp_model_operations_disabled")
            return

        try:
            self._default_model = LLMRegistry.get_config("", LLMRole.CHAT).model
            all_names = LLMRegistry.get_all_names()
            if self._default_model not in all_names:
                available = ", ".join(all_names) or "none"
                raise ProviderError(
                    ProviderErrorCode.CONFIGURATION,
                    f"default model '{self._default_model}' is not registered; available models: {available}",
                    model=self._default_model,
                )
            self._llm = LLMRegistry.get(self._default_model)
            logger.info(
                "llm_service_initialized",
                default_model=self._default_model,
                total_models=len(all_names),
                environment=settings.ENVIRONMENT.value,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._configuration_error(self._default_model, exc) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: str | None = ...,
        response_format: None = ...,
        **model_kwargs: Any,
    ) -> BaseMessage: ...

    @overload
    async def call(
        self,
        messages: LanguageModelInput,
        model_name: str | None = ...,
        *,
        response_format: type[T],
        **model_kwargs: Any,
    ) -> T: ...

    async def call(
        self,
        messages: LanguageModelInput,
        model_name: str | None = None,
        response_format: type[BaseModel] | None = None,
        role: LLMRole = LLMRole.CHAT,
        **model_kwargs: Any,
    ) -> BaseMessage | BaseModel:
        """Invoke a model with bounded retries and request-local fallback."""
        if settings.LOCAL_MVP_ONLY:
            raise ProviderError(ProviderErrorCode.CAPABILITY, "local_mvp_model_operations_disabled", retryable=False)
        try:
            return await asyncio.wait_for(
                self._call_with_fallback(messages, model_name, response_format, role, model_kwargs),
                timeout=float(settings.LLM_TOTAL_TIMEOUT),
            )
        except asyncio.TimeoutError as exc:
            logger.exception(
                "llm_total_timeout_exceeded",
                timeout_seconds=settings.LLM_TOTAL_TIMEOUT,
            )
            raise ProviderError(
                ProviderErrorCode.TIMEOUT,
                f"llm call exceeded total timeout of {settings.LLM_TOTAL_TIMEOUT}s",
                model=model_name or self._default_model,
                retryable=False,
                cause=exc,
            ) from exc

    async def call_with_usage(
        self,
        messages: LanguageModelInput,
        model_name: str | None = None,
        response_format: type[BaseModel] | None = None,
        role: LLMRole = LLMRole.CHAT,
        **model_kwargs: Any,
    ) -> tuple[BaseMessage | BaseModel, UsageInfo]:
        """Call a model and return provider-neutral usage and latency telemetry."""
        started = perf_counter()
        try:
            response = await self.call(
                messages,
                model_name=model_name,
                response_format=response_format,
                role=role,
                **model_kwargs,
            )
            config = self._get_config(model_name or "", role)
            usage = usage_from_response(response, config.capabilities)
            usage = replace(
                usage,
                provider=config.provider,
                model=config.model,
                latency_ms=round((perf_counter() - started) * 1000, 3),
                availability=Availability.AVAILABLE,
            )
            logger.info(
                "llm_usage_recorded",
                provider=config.provider,
                model=config.model,
                usage_quality=usage.quality.value if usage.quality else "unknown",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=usage.latency_ms,
                availability=usage.availability.value,
            )
            observe_usage(role.value, usage)
            return response, usage
        except ProviderError as exc:
            config = self._get_config(model_name or "", role)
            usage = replace(
                UsageInfo.unknown(),
                provider=config.provider,
                model=config.model,
                latency_ms=round((perf_counter() - started) * 1000, 3),
                availability=Availability.UNAVAILABLE,
                error_category=exc.code.value,
            )
            logger.warning(
                "llm_usage_unavailable",
                provider=config.provider,
                model=config.model,
                availability=usage.availability.value,
                error_category=usage.error_category,
                latency_ms=usage.latency_ms,
            )
            observe_usage(role.value, usage)
            raise

    def get_llm(self) -> Any:
        """Return the shared default runnable used by the graph."""
        return self._llm

    async def call_once_with_usage(
        self,
        messages: LanguageModelInput,
        config: ProviderConfig,
        *,
        metadata: dict[str, Any],
        overrides: dict[str, Any],
        budget_scope: Literal["legacy", "review"] = "legacy",
        evaluation_timeout_seconds: Literal[60] | None = None,
        business_wait_policy: BusinessWaitPolicy | None = None,
    ) -> tuple[BaseMessage, UsageInfo]:
        """One pinned budgeted attempt, traced and classified without fallback."""
        enabled = settings.V11_REAL_MODELS_ENABLED if budget_scope == "review" else settings.V1_REAL_MODELS_ENABLED
        if settings.LOCAL_MVP_ONLY and not enabled:
            raise ProviderError(ProviderErrorCode.CAPABILITY, "local_mvp_model_operations_disabled")
        if evaluation_timeout_seconds is not None and (budget_scope != "review" or evaluation_timeout_seconds != 60):
            raise ProviderError(ProviderErrorCode.CONFIGURATION, "evaluation_timeout_scope_invalid")
        if business_wait_policy is not None and (
            budget_scope != "review"
            or business_wait_policy != BUSINESS_WAIT_POLICY
            or evaluation_timeout_seconds is not None
        ):
            raise ProviderError(ProviderErrorCode.CONFIGURATION, "business_wait_policy_scope_invalid")
        wait_seconds = 60 if evaluation_timeout_seconds == 60 else min(config.timeout_seconds, 30)
        if business_wait_policy is not None:
            wait_seconds = BUSINESS_MODEL_WAIT_SECONDS
        if evaluation_timeout_seconds == 60 or business_wait_policy is not None:
            config = replace(config, timeout_seconds=wait_seconds)
        started = perf_counter()
        try:
            model = LLMRegistry.get_pinned_once(config, **overrides)
        except ProviderError as error:
            if budget_scope == "review":
                error.call_diagnostic = pinned_failure_diagnostic(error, config, "adapter_setup", started)
            raise
        try:
            response = await asyncio.wait_for(
                model.ainvoke(
                    messages,
                    config={
                        "callbacks": [get_langfuse_callback_handler()],
                        "metadata": safe_payload(metadata),
                    },
                ),
                timeout=wait_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            classified = self._provider_error(error, config, response_format=None)
            if budget_scope == "review":
                classified.call_diagnostic = pinned_failure_diagnostic(classified, config, "invoke", started)
            raise classified from None
        usage = replace(
            usage_from_response(response, config.capabilities),
            provider=config.provider,
            model=config.model,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            availability=Availability.AVAILABLE,
        )
        observe_usage(config.role.value, usage)
        return response, usage

    def bind_tools(self, tools: Sequence[Any]) -> "LLMService":
        """Bind graph tools after enforcing the default model capability."""
        if settings.LOCAL_MVP_ONLY:
            return self
        config = self._get_config(self._default_model)
        if not config.capabilities.supports_tools:
            raise capability_error(f"model '{self._default_model}' does not support tools", config)
        try:
            self._bound_tools = list(tools)
            self._llm = self._llm.bind_tools(self._bound_tools)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                ProviderErrorCode.CAPABILITY,
                f"failed to bind tools for model '{self._default_model}'",
                provider=config.provider,
                model=config.model,
                cause=exc,
            ) from exc
        logger.debug("tools_bound_to_llm", tool_count=len(self._bound_tools), model=self._default_model)
        return self

    # ------------------------------------------------------------------
    # Request routing and retries
    # ------------------------------------------------------------------

    async def _invoke_with_retry(
        self,
        llm: Any,
        messages: LanguageModelInput,
        *,
        config: ProviderConfig,
        response_format: type[BaseModel] | None,
    ) -> Any:
        """Invoke one runnable, retrying only explicitly retryable errors."""
        retry_limit = max(0, int(config.max_retries))
        retrying = AsyncRetrying(
            stop=stop_after_attempt(retry_limit + 1),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(lambda exc: isinstance(exc, ProviderError) and exc.retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                try:
                    response = await llm.ainvoke(messages)
                except ProviderError:
                    raise
                except BaseException as exc:
                    raise self._provider_error(exc, config, response_format=response_format) from exc
                logger.debug("llm_call_successful", provider=config.provider, model=config.model)
                if response_format is not None:
                    return self._validate_structured_response(response, response_format, config)
                return response
        raise AssertionError("tenacity retry loop completed without a result")

    async def _call_with_fallback(
        self,
        messages: LanguageModelInput,
        model_name: str | None,
        response_format: type[BaseModel] | None,
        role: LLMRole,
        model_kwargs: dict[str, Any],
    ) -> BaseMessage | BaseModel:
        """Try the selected model and its configured fallbacks for this request."""
        role_config = self._get_config("", role) if role is not LLMRole.CHAT else None
        names = (
            LLMRegistry.get_all_names() if role_config is None else [role_config.model, *role_config.fallback_models]
        )
        if not names:
            raise ProviderError(ProviderErrorCode.CONFIGURATION, f"no {role.value} models are configured")

        requested_name = model_name or (role_config.model if role_config else self._default_model)
        if requested_name not in names:
            available = ", ".join(names)
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                f"model '{requested_name}' is not registered for role '{role.value}'; available models: {available}",
                model=requested_name,
            )

        is_default_path = role is LLMRole.CHAT and not model_name and response_format is None and not model_kwargs
        start = names.index(requested_name)
        candidates = [names[(start + offset) % len(names)] for offset in range(len(names))]
        last_error: ProviderError | None = None

        for candidate in candidates:
            try:
                if is_default_path and candidate == self._default_model:
                    llm = self._llm
                    config = self._get_config(candidate, role)
                else:
                    request_kwargs = dict(model_kwargs)
                    if self._bound_tools and is_default_path:
                        request_kwargs["tools"] = list(self._bound_tools)
                    if response_format is not None:
                        request_kwargs["response_format"] = response_format
                    request_kwargs = self._normalise_request_kwargs(request_kwargs)
                    self._validate_capabilities(candidate, request_kwargs, response_format, role)
                    llm = LLMRegistry.get(candidate, role=role, **request_kwargs)
                    config = self._get_config(candidate, role)

                return await self._invoke_with_retry(
                    llm,
                    messages,
                    config=config,
                    response_format=response_format,
                )
            except ProviderError as exc:
                last_error = exc
                logger.warning(
                    "llm_candidate_failed",
                    provider=exc.provider,
                    model=candidate,
                    error_code=exc.code.value,
                    retryable=exc.retryable,
                )
                if not exc.retryable:
                    raise
            except Exception as exc:
                error = self._configuration_error(candidate, exc)
                last_error = error
                logger.warning("llm_candidate_configuration_failed", model=candidate, error=str(error))
                raise error from exc

        raise last_error or ProviderError(
            ProviderErrorCode.UNAVAILABLE,
            "all configured chat models failed",
            model=requested_name,
            retryable=False,
        )

    # ------------------------------------------------------------------
    # Capability, response, and error contracts
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_request_kwargs(model_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize provider-neutral aliases before registry validation."""
        kwargs = dict(model_kwargs)
        usage = kwargs.pop("usage", None)
        if usage is not None and "stream_usage" not in kwargs:
            kwargs["stream_usage"] = bool(usage)
        return kwargs

    def _validate_capabilities(
        self,
        model_name: str,
        model_kwargs: dict[str, Any],
        response_format: type[BaseModel] | None,
        role: LLMRole,
    ) -> None:
        """Reject unsupported features before constructing or invoking a model."""
        config = self._get_config(model_name, role)
        capabilities = config.capabilities
        requested: list[tuple[str, str, Any]] = [
            ("tools", "tools", model_kwargs.get("tools")),
            ("streaming", "streaming", model_kwargs.get("streaming", model_kwargs.get("stream"))),
            ("usage", "usage", model_kwargs.get("stream_usage")),
            ("reasoning", "reasoning", model_kwargs.get("reasoning", model_kwargs.get("reasoning_effort"))),
        ]
        if response_format is not None:
            requested.append(("structured_output", "structured output", True))
        for capability, label, value in requested:
            if value is not None and value is not False and not capabilities.supports(capability):
                raise capability_error(
                    f"provider '{config.provider}' model '{config.model}' does not support {label}",
                    config,
                )

    def _get_config(self, model_name: str, role: LLMRole = LLMRole.CHAT) -> ProviderConfig:
        """Resolve provider config and expose malformed settings uniformly."""
        try:
            return LLMRegistry.get_config(model_name, role)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._configuration_error(model_name, exc) from exc

    @staticmethod
    def _configuration_error(model_name: str, cause: BaseException) -> ProviderError:
        """Create a provider-neutral configuration error."""
        return ProviderError(
            ProviderErrorCode.CONFIGURATION,
            f"invalid configuration for model '{model_name}': {cause}",
            model=model_name,
            retryable=False,
            cause=cause,
        )

    @staticmethod
    def _provider_error(
        error: BaseException,
        config: ProviderConfig,
        *,
        response_format: type[BaseModel] | None,
    ) -> ProviderError:
        """Map SDK and parsing failures to the stable provider error taxonomy."""
        if isinstance(error, ProviderError):
            return error
        if isinstance(error, AuthenticationError):
            code, retryable = ProviderErrorCode.AUTHENTICATION, False
        elif isinstance(error, RateLimitError):
            code, retryable = ProviderErrorCode.RATE_LIMITED, True
        elif isinstance(error, (APITimeoutError, asyncio.TimeoutError, TimeoutError)):
            code, retryable = ProviderErrorCode.TIMEOUT, True
        elif isinstance(error, APIConnectionError):
            code, retryable = ProviderErrorCode.UNAVAILABLE, True
        elif isinstance(error, BadRequestError):
            code, retryable = (
                (ProviderErrorCode.RESPONSE_INVALID, False)
                if response_format is not None
                else (ProviderErrorCode.CONFIGURATION, False)
            )
        elif isinstance(error, APIError):
            code, retryable = ProviderErrorCode.UNAVAILABLE, True
        elif response_format is not None and isinstance(error, (ValidationError, ValueError, TypeError, KeyError)):
            code, retryable = ProviderErrorCode.RESPONSE_INVALID, False
        elif isinstance(error, OpenAIError):
            code, retryable = ProviderErrorCode.UNAVAILABLE, True
        else:
            code, retryable = ProviderErrorCode.UNAVAILABLE, True
        return ProviderError(
            code,
            f"{type(error).__name__}: {error}",
            provider=config.provider,
            model=config.model,
            retryable=retryable,
            cause=error,
        )

    @staticmethod
    def _validate_structured_response(
        response: Any,
        response_format: type[BaseModel],
        config: ProviderConfig,
    ) -> BaseModel:
        """Accept only a target model or a dictionary valid for that model."""
        if isinstance(response, response_format):
            return response
        if isinstance(response, dict):
            try:
                return response_format.model_validate(response)
            except ValidationError as exc:
                raise ProviderError(
                    ProviderErrorCode.RESPONSE_INVALID,
                    f"structured response failed schema validation: {exc}",
                    provider=config.provider,
                    model=config.model,
                    retryable=False,
                    cause=exc,
                ) from exc
        raise ProviderError(
            ProviderErrorCode.RESPONSE_INVALID,
            f"structured response must be {response_format.__name__} or a valid object",
            provider=config.provider,
            model=config.model,
            retryable=False,
        )


llm_service = LLMService()
