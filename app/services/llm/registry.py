"""Provider-neutral registry for chat model configurations.

The registry keeps the legacy ``LLMS`` view for existing callers while making
provider, model and capability metadata explicit. Concrete construction is
currently implemented with LangChain's OpenAI-compatible adapter; adding
another provider must be an explicit adapter registration, never an implicit
fallback to the first model.
"""

import os
import re
from dataclasses import fields, replace
from typing import Any, Callable, Dict, List, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.core.config import settings
from app.core.logging import logger
from app.services.llm.contracts import (
    LLMRole,
    ProviderCapabilities,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
    capability_error,
)
from app.services.llm.embeddings import DimensionValidatedEmbeddings


_LEGACY_MODEL_NAMES = (
    "gpt-5.6-luna",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
)
_REASONING_EFFORTS = {
    "gpt-5.6-luna": "medium",
    "gpt-5.4": "medium",
    "gpt-5.4-mini": "low",
    "gpt-5.4-nano": "low",
}
_OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "openai-compatible", "openai_compatible"})


def _normalise_role(role: LLMRole | str) -> LLMRole:
    """Convert a role value to the contract enum with a useful error."""
    if isinstance(role, LLMRole):
        return role
    try:
        return LLMRole(str(role).strip().lower())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown llm role: {role!r}") from exc


def _setting(*names: str, default: Any = None) -> Any:
    """Read the first non-empty setting or environment value."""
    for name in names:
        value = getattr(settings, name, None)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _as_capabilities(value: Any, *, default: ProviderCapabilities) -> ProviderCapabilities:
    """Convert a mapping from settings while retaining explicit defaults."""
    if value is None:
        return default
    if isinstance(value, ProviderCapabilities):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("provider capabilities must be a ProviderCapabilities or mapping")
    allowed = {item.name for item in fields(ProviderCapabilities)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unsupported provider capability field(s): {', '.join(unknown)}")
    return ProviderCapabilities(**dict(value))


def _structured_role_value(role: LLMRole) -> Any:
    """Find the first structured configuration that defines this role."""
    known_roles = {item.value for item in LLMRole}
    for setting_name in ("LLM_ROLE_CONFIGS", "PROVIDER_CONFIGS"):
        role_configs = _setting(setting_name, default=None)
        if role_configs is None:
            continue
        if not isinstance(role_configs, Mapping):
            raise ValueError(f"{setting_name} must be a mapping of role names to provider configurations")
        unknown = sorted(str(name) for name in role_configs if str(name) not in known_roles)
        if unknown:
            raise ValueError(f"unknown role(s) in {setting_name}: {', '.join(unknown)}")
        value = role_configs.get(role.value)
        if value is not None:
            return value
    for attr in (f"{role.value.upper()}_CONFIG", f"{role.value.upper()}_PROVIDER_CONFIG"):
        value = getattr(settings, attr, None)
        if value is not None:
            return value
    return None


def _credential_env_name(reference: str) -> str:
    """Accept an environment variable name, optionally prefixed with ``env:``."""
    name = reference.strip()
    if name.startswith("env:"):
        name = name[4:]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("credential references must name an environment variable")
    return name


def _as_provider_config(value: Any, *, role: LLMRole, fallback_model: str) -> ProviderConfig | None:
    """Normalize a role config supplied by settings or an environment fallback."""
    if value is None:
        return None
    if isinstance(value, ProviderConfig):
        if value.role is not role:
            raise ValueError(f"provider config role '{value.role.value}' does not match '{role.value}'")
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"provider config for role '{role.value}' must be a ProviderConfig or mapping")

    default_capabilities = ProviderCapabilities(
        supports_chat=role is not LLMRole.EMBEDDING,
        supports_embeddings=role is LLMRole.EMBEDDING,
    )
    if "capabilities" in value and "capability" in value:
        raise ValueError("use only one of capabilities or capability")
    capability_value = value.get("capabilities", value.get("capability"))
    if "embedding_dimensions" in value:
        if isinstance(capability_value, ProviderCapabilities):
            capability_values = {
                item.name: getattr(capability_value, item.name) for item in fields(ProviderCapabilities)
            }
        elif capability_value is None:
            capability_values = {}
        elif isinstance(capability_value, Mapping):
            capability_values = dict(capability_value)
        else:
            raise ValueError("provider capabilities must be a ProviderCapabilities or mapping")
        if (
            "embedding_dimensions" in capability_values
            and capability_values["embedding_dimensions"] != value["embedding_dimensions"]
        ):
            raise ValueError("embedding_dimensions conflicts with capabilities.embedding_dimensions")
        capability_values["embedding_dimensions"] = value["embedding_dimensions"]
        capability_value = capability_values
    capabilities = _as_capabilities(capability_value, default=default_capabilities)
    allowed = {
        "role",
        "provider",
        "model",
        "endpoint",
        "credentials_ref",
        "api_key_env",
        "api_key",
        "capabilities",
        "capability",
        "fallback_models",
        "parse_mode",
        "timeout_seconds",
        "max_retries",
        "embedding_dimensions",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unsupported provider config field(s): {', '.join(unknown)}")
    declared_role = value.get("role")
    if declared_role is not None and _normalise_role(declared_role) is not role:
        raise ValueError(f"provider config role '{declared_role}' does not match '{role.value}'")
    config_values: dict[str, Any] = {
        "role": role,
        "provider": value.get("provider", "openai"),
        "model": value.get("model", fallback_model),
        "endpoint": value.get("endpoint"),
        "credentials_ref": value.get("credentials_ref"),
        "api_key_env": value.get("api_key_env"),
        "api_key": value.get("api_key"),
        "capabilities": capabilities,
        "fallback_models": tuple(value.get("fallback_models", ())),
        "parse_mode": value.get("parse_mode"),
        "timeout_seconds": value.get("timeout_seconds", 60.0),
        "max_retries": value.get("max_retries", 3),
        "allow_global_api_key_fallback": False,
    }
    api_key_env = config_values["api_key_env"]
    credentials_ref = config_values["credentials_ref"]
    if api_key_env and credentials_ref:
        if _credential_env_name(api_key_env) != _credential_env_name(credentials_ref):
            raise ValueError("api_key_env and credentials_ref must refer to the same environment variable")
    for reference in (api_key_env, credentials_ref):
        if reference:
            _credential_env_name(reference)
    return ProviderConfig(**config_values)


def _legacy_capabilities(model_name: str) -> ProviderCapabilities:
    """Capabilities used only when legacy settings have no role config."""
    return ProviderCapabilities(
        supports_chat=True,
        supports_streaming=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_reasoning=model_name.startswith("gpt-5"),
        supports_usage=True,
    )


def _role_config(role: LLMRole, model_name: str) -> ProviderConfig:
    """Resolve an explicit role configuration, with legacy compatibility."""
    role_name = role.value
    role_prefix = {
        LLMRole.CHAT: "CHAT_LLM",
        LLMRole.MEMORY_LLM: "MEMORY_LLM",
        LLMRole.EMBEDDING: "EMBEDDING",
        LLMRole.NAMING: "NAMING_LLM",
        LLMRole.JUDGE: "JUDGE_LLM",
    }[role]
    model_settings = {
        LLMRole.CHAT: ("CHAT_LLM_MODEL", "CHAT_MODEL", "DEFAULT_LLM_MODEL"),
        LLMRole.MEMORY_LLM: ("MEMORY_LLM_MODEL",),
        LLMRole.EMBEDDING: ("EMBEDDING_MODEL",),
        LLMRole.NAMING: ("NAMING_LLM_MODEL",),
        LLMRole.JUDGE: ("JUDGE_LLM_MODEL", "EVALUATION_LLM"),
    }[role]
    configured_model = str(_setting(*model_settings, default=_LEGACY_MODEL_NAMES[0]))
    requested_model = model_name.strip() if isinstance(model_name, str) else ""
    candidate_model = requested_model or configured_model

    value = _structured_role_value(role)
    config = _as_provider_config(value, role=role, fallback_model=candidate_model)
    if config is not None:
        role_prefix = {
            LLMRole.CHAT: "CHAT_LLM",
            LLMRole.MEMORY_LLM: "MEMORY_LLM",
            LLMRole.EMBEDDING: "EMBEDDING",
            LLMRole.NAMING: "NAMING_LLM",
            LLMRole.JUDGE: "JUDGE_LLM",
        }[role]
        config_values = {item.name: getattr(config, item.name) for item in fields(ProviderConfig)}
        config_values["endpoint"] = config.endpoint or _setting(
            f"{role_prefix}_ENDPOINT", "LLM_ENDPOINT", "OPENAI_BASE_URL"
        )
        if not config.api_key and not config.api_key_env and not config.credentials_ref:
            config_values["api_key"] = _setting(f"{role_prefix}_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", default="")
        if not config.api_key_env and not config.credentials_ref:
            config_values["api_key_env"] = _setting(f"{role_prefix}_API_KEY_ENV", "LLM_API_KEY_ENV", default=None)
        if not config.credentials_ref:
            config_values["credentials_ref"] = _setting(
                f"{role_prefix}_CREDENTIALS_REF", "LLM_CREDENTIALS_REF", default=None
            )
        config = ProviderConfig(**config_values)
        if requested_model and config.model != requested_model:
            if requested_model not in config.fallback_models:
                raise ValueError(
                    f"model '{requested_model}' is not configured for role '{role_name}'; "
                    f"configured model is '{config.model}'"
                )
            config_values = {item.name: getattr(config, item.name) for item in fields(ProviderConfig)}
            config_values["model"] = requested_model
            config = ProviderConfig(**config_values)
        return config

    if requested_model and configured_model != requested_model:
        fallback_models = tuple(_setting(f"{role_prefix}_FALLBACK_MODELS", default=[]))
        fallback_value = _setting(f"{role_prefix}_FALLBACK_MODELS", default=fallback_models)
        if isinstance(fallback_value, str):
            fallback_models = tuple(item.strip() for item in fallback_value.split(",") if item.strip())
        else:
            fallback_models = tuple(fallback_value)
        if requested_model not in fallback_models:
            raise ValueError(
                f"model '{requested_model}' is not configured for role '{role.value}'; "
                f"configured model is '{configured_model}'"
            )
    selected_model = requested_model or configured_model

    supports_chat = bool(_setting(f"{role_prefix}_SUPPORTS_CHAT", default=role is not LLMRole.EMBEDDING))
    supports_streaming = bool(_setting(f"{role_prefix}_SUPPORTS_STREAMING", default=False))
    supports_tools = bool(_setting(f"{role_prefix}_SUPPORTS_TOOLS", default=False))
    supports_structured = bool(_setting(f"{role_prefix}_SUPPORTS_STRUCTURED_OUTPUT", default=False))
    supports_reasoning = bool(_setting(f"{role_prefix}_SUPPORTS_REASONING", default=False))
    supports_usage = bool(_setting(f"{role_prefix}_SUPPORTS_USAGE", default=False))
    supports_embeddings = bool(_setting(f"{role_prefix}_SUPPORTS_EMBEDDINGS", default=role is LLMRole.EMBEDDING))
    context_window = _setting(f"{role_prefix}_CONTEXT_WINDOW", default=None)
    embedding_dimensions = _setting(f"{role_prefix}_DIMENSIONS", f"{role_prefix}_EMBEDDING_DIMENSIONS", default=None)
    tokenizer = _setting(f"{role_prefix}_TOKENIZER", default=None)
    capabilities = ProviderCapabilities(
        supports_chat=supports_chat,
        supports_streaming=supports_streaming,
        supports_tools=supports_tools,
        supports_structured_output=supports_structured,
        supports_reasoning=supports_reasoning,
        supports_usage=supports_usage,
        supports_embeddings=supports_embeddings,
        context_window=int(context_window) if context_window is not None else None,
        embedding_dimensions=int(embedding_dimensions) if embedding_dimensions is not None else None,
        tokenizer=tokenizer or None,
    )
    fallback_value = _setting(f"{role_prefix}_FALLBACK_MODELS", default=[])
    fallback_models = (
        tuple(fallback_value)
        if not isinstance(fallback_value, str)
        else tuple(item.strip() for item in fallback_value.split(",") if item.strip())
    )
    return ProviderConfig(
        role=role,
        provider=str(_setting(f"{role_prefix}_PROVIDER", "LLM_PROVIDER", default="openai")),
        model=selected_model,
        endpoint=_setting(f"{role_prefix}_ENDPOINT", "LLM_ENDPOINT", "OPENAI_BASE_URL"),
        credentials_ref=_setting(f"{role_prefix}_CREDENTIALS_REF", "LLM_CREDENTIALS_REF"),
        api_key_env=_setting(f"{role_prefix}_API_KEY_ENV", "LLM_API_KEY_ENV"),
        api_key=_setting(f"{role_prefix}_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", default=""),
        capabilities=capabilities,
        fallback_models=fallback_models,
        parse_mode=_setting(f"{role_prefix}_PARSE_MODE", default=None),
        timeout_seconds=float(_setting(f"{role_prefix}_TIMEOUT", "LLM_TOTAL_TIMEOUT", default=60)),
        max_retries=int(_setting(f"{role_prefix}_MAX_RETRIES", "MAX_LLM_CALL_RETRIES", default=3)),
        allow_global_api_key_fallback=True,
    )


class LLMRegistry:
    """Registry of provider-neutral model metadata and model constructors."""

    LLMS: List[Dict[str, Any]] = []
    _adapter_factories: dict[str, Callable[[ProviderConfig, dict[str, Any]], BaseChatModel]] = {}
    _embedding_adapter_factories: dict[str, Callable[[ProviderConfig], Embeddings]] = {}

    @classmethod
    def _configured_chat_model(cls) -> str:
        """Return configured chat model without assuming a GPT name."""
        explicit = _structured_role_value(LLMRole.CHAT)
        if isinstance(explicit, Mapping) and explicit.get("model"):
            return str(explicit["model"])
        if isinstance(explicit, ProviderConfig):
            return explicit.model
        return str(_setting("CHAT_LLM_MODEL", "CHAT_MODEL", "DEFAULT_LLM_MODEL", default=_LEGACY_MODEL_NAMES[0]))

    @classmethod
    def _model_names(cls) -> list[str]:
        """Build ordered chat model list from configured and explicit fallbacks."""
        configured = _role_config(LLMRole.CHAT, "")
        return list(dict.fromkeys([configured.model, *configured.fallback_models]))

    @classmethod
    def _entry(cls, model_name: str) -> dict[str, Any]:
        """Return metadata for a known chat model."""
        names = cls._model_names()
        if model_name not in names:
            available = ", ".join(names)
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                f"model '{model_name}' not found in registry. available models: {available}",
                model=model_name,
            )
        config = _role_config(LLMRole.CHAT, model_name)
        return {
            "name": model_name,
            "config": config,
            "capabilities": config.capabilities,
            "llm": cls._create_adapter(config, {}),
        }

    @classmethod
    def _create_adapter(cls, config: ProviderConfig, overrides: dict[str, Any]) -> BaseChatModel:
        """Construct an adapter after validating provider and capability gates."""
        provider = config.provider.strip().lower()
        factory = cls._adapter_factories.get(provider)
        if factory is None and provider in _OPENAI_COMPATIBLE_PROVIDERS:
            factory = cls._create_openai_compatible
        if factory is None:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                f"no adapter registered for provider '{config.provider}'",
                provider=config.provider,
                model=config.model,
            )
        cls._validate_overrides(config, overrides)
        return factory(config, overrides)

    @staticmethod
    def _validate_overrides(config: ProviderConfig, overrides: Mapping[str, Any]) -> None:
        """Reject provider parameters that the declared model cannot support."""
        allowed = {
            "tools",
            "streaming",
            "stream_usage",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "structured_output",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "n",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "extra_body",
        }
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                f"unsupported provider override(s): {', '.join(unknown)}",
                provider=config.provider,
                model=config.model,
                retryable=False,
            )
        capability_names = {
            "tools": "tools",
            "streaming": "streaming",
            "stream": "streaming",
            "stream_usage": "usage",
            "reasoning": "reasoning",
            "reasoning_effort": "reasoning",
            "response_format": "structured_output",
            "structured_output": "structured_output",
        }
        for name, capability in capability_names.items():
            value = overrides.get(name)
            if value is not None and value is not False and not config.capabilities.supports(capability):
                raise capability_error(
                    f"provider '{config.provider}' model '{config.model}' does not support {capability}",
                    config,
                )

    @staticmethod
    def _resolve_api_key(config: ProviderConfig) -> str:
        """Resolve a configured credential reference without exposing its value."""
        reference = config.api_key_env or config.credentials_ref
        if reference:
            try:
                env_name = _credential_env_name(reference)
            except ValueError as exc:
                raise ProviderError(
                    ProviderErrorCode.CONFIGURATION,
                    "provider credential reference must name an environment variable",
                    provider=config.provider,
                    model=config.model,
                ) from exc
            api_key = os.getenv(env_name)
            if not api_key:
                raise ProviderError(
                    ProviderErrorCode.CONFIGURATION,
                    f"provider credential environment variable '{env_name}' is not set",
                    provider=config.provider,
                    model=config.model,
                )
        elif config.api_key:
            api_key = config.api_key
        elif config.allow_global_api_key_fallback:
            api_key = str(_setting("OPENAI_API_KEY", default=""))
        else:
            api_key = ""
        return api_key

    @classmethod
    def _create_openai_compatible(cls, config: ProviderConfig, overrides: dict[str, Any]) -> BaseChatModel:
        """Create LangChain ChatOpenAI for an OpenAI-compatible API."""
        constructor: dict[str, Any] = {
            "model": config.model,
            "api_key": SecretStr(cls._resolve_api_key(config)),
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        if config.endpoint:
            constructor["base_url"] = config.endpoint
        max_tokens = _setting("MAX_TOKENS", default=None)
        if max_tokens is not None and not {"max_tokens", "max_completion_tokens"}.intersection(overrides):
            constructor["max_completion_tokens"] = int(max_tokens)

        model_name = config.model
        if config.capabilities.supports_reasoning:
            effort = _REASONING_EFFORTS.get(model_name) or _setting("LLM_REASONING_EFFORT")
            if effort:
                constructor["reasoning"] = {"effort": str(effort)}
        constructor.update(overrides)
        return ChatOpenAI(**constructor)

    @classmethod
    def _create_openai_compatible_embeddings(cls, config: ProviderConfig) -> Embeddings:
        """Create and dimension-check an OpenAI-compatible embedding adapter."""
        dimensions = config.capabilities.embedding_dimensions
        if dimensions is None:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                "embedding_dimensions must be configured for the embedding role",
                provider=config.provider,
                model=config.model,
            )
        constructor: dict[str, Any] = {
            "model": config.model,
            "api_key": SecretStr(cls._resolve_api_key(config)),
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
            "check_embedding_ctx_length": False,
            "tiktoken_enabled": False,
        }
        if config.endpoint:
            constructor["base_url"] = config.endpoint
        return DimensionValidatedEmbeddings(OpenAIEmbeddings(**constructor), config)

    @classmethod
    def register_adapter(
        cls, provider: str, factory: Callable[[ProviderConfig, dict[str, Any]], BaseChatModel]
    ) -> None:
        """Register a provider adapter explicitly."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("provider must not be empty")
        cls._adapter_factories[normalized] = factory

    @classmethod
    def register_embedding_adapter(cls, provider: str, factory: Callable[[ProviderConfig], Embeddings]) -> None:
        """Register a provider-specific embedding adapter explicitly."""
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("provider must not be empty")
        cls._embedding_adapter_factories[normalized] = factory

    @classmethod
    def get_config(cls, model_name: str, role: LLMRole | str = LLMRole.CHAT) -> ProviderConfig:
        """Return provider-neutral configuration for a known model and role."""
        normalized_role = _normalise_role(role)
        try:
            if normalized_role is LLMRole.CHAT and model_name:
                cls._entry(model_name)
            return _role_config(normalized_role, model_name)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                f"invalid provider configuration for role '{normalized_role.value}' ({type(exc).__name__})",
                model=model_name or None,
            ) from exc

    @classmethod
    def get_capabilities(cls, model_name: str, role: LLMRole | str = LLMRole.CHAT) -> ProviderCapabilities:
        """Return declared capabilities without constructing a second adapter."""
        return cls.get_config(model_name, role).capabilities

    @classmethod
    def supports(cls, model_name: str, capability: str, role: LLMRole | str = LLMRole.CHAT) -> bool:
        """Check a declared capability for routing and feature gates."""
        return cls.get_capabilities(model_name, role).supports(capability)

    @classmethod
    def get(cls, model_name: str, **kwargs: Any) -> BaseChatModel:
        """Create a fresh model adapter for a known chat model."""
        role_value = kwargs.pop("role", LLMRole.CHAT)
        role = _normalise_role(role_value)
        if role is LLMRole.EMBEDDING:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                "use get_embeddings() for the embedding role",
                model=model_name or None,
            )
        config = cls.get_config(model_name, role)

        # ``stream`` is the provider-neutral spelling used by callers.  The
        # OpenAI-compatible adapter exposes the explicit ``streaming`` field;
        # normalize the alias before capability validation so it cannot fall
        # through to LangChain's opaque ``model_kwargs``.
        stream = kwargs.pop("stream", None)
        if stream is not None:
            if "streaming" in kwargs and bool(kwargs["streaming"]) != bool(stream):
                raise ValueError("stream and streaming overrides disagree")
            kwargs["streaming"] = bool(stream)
        cls._validate_overrides(config, kwargs)

        # These values are LangChain runnable operations, not constructor
        # arguments.  Removing them before adapter construction prevents
        # ChatOpenAI from accepting them as opaque ``model_kwargs``.
        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("response_format", None)
        structured_output = kwargs.pop("structured_output", None)
        if response_format is not None and structured_output is not None:
            raise ValueError("response_format and structured_output cannot both be supplied")
        structured_schema = response_format if response_format is not None else structured_output

        logger.debug(
            "creating_llm_adapter",
            provider=config.provider,
            model=config.model,
            role=role.value,
            custom_args=list(kwargs.keys()),
        )
        model: Any = cls._create_adapter(config, kwargs)
        if tools is not None:
            model = model.bind_tools(list(tools))
        if structured_schema is not None and structured_schema is not False:
            model = model.with_structured_output(structured_schema)
        return model

    @classmethod
    def get_embeddings(cls, role: LLMRole | str = LLMRole.EMBEDDING) -> Embeddings:
        """Construct a configured embedding adapter with dimension validation."""
        normalized_role = _normalise_role(role)
        if normalized_role is not LLMRole.EMBEDDING:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                "embedding adapters require the embedding role",
            )
        config = cls.get_config("", normalized_role)
        if not config.capabilities.supports_embeddings:
            raise capability_error("embedding role does not declare embedding support", config)
        provider = config.provider.strip().lower()
        factory = cls._embedding_adapter_factories.get(provider)
        if factory is None and provider in _OPENAI_COMPATIBLE_PROVIDERS:
            factory = cls._create_openai_compatible_embeddings
        if factory is None:
            raise ProviderError(
                ProviderErrorCode.CONFIGURATION,
                f"no embedding adapter registered for provider '{config.provider}'",
                provider=config.provider,
                model=config.model,
            )
        return factory(config)

    @classmethod
    def get_pinned_once(cls, config: ProviderConfig, **overrides: Any) -> BaseChatModel:
        """Trusted budgeted caller pins one model and disables SDK retries."""
        config = replace(config, max_retries=0, fallback_models=())
        cls._validate_overrides(config, overrides)
        return cls._create_adapter(config, overrides)

    @classmethod
    def get_embeddings_once(cls) -> Embeddings:
        """Both dimension wrapper and SDK make exactly one bounded attempt."""
        config = replace(cls.get_config("", LLMRole.EMBEDDING), max_retries=0)
        provider = config.provider.strip().lower()
        factory = cls._embedding_adapter_factories.get(provider)
        if factory is None and provider in _OPENAI_COMPATIBLE_PROVIDERS:
            factory = cls._create_openai_compatible_embeddings
        if factory is None:
            raise capability_error("embedding adapter unavailable", config)
        return factory(config)

    @classmethod
    def bind_tools(cls, model_name: str, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Bind tools only when the configured model declares tool support."""
        config = cls.get_config(model_name)
        if not config.capabilities.supports_tools:
            raise capability_error(f"model '{model_name}' does not support tools", config)
        return cls.get(model_name, **kwargs).bind_tools(list(tools))

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Return registered chat model names in fallback order."""
        return cls._model_names()

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """Return model metadata at an index, preserving legacy wrap behavior."""
        names = cls.get_all_names()
        if not names:
            raise ValueError("llm registry has no models")
        normalized_index = index if 0 <= index < len(names) else 0
        return cls._entry(names[normalized_index])

    @classmethod
    def _legacy_llms(cls) -> list[dict[str, Any]]:
        """Build legacy list view lazily so settings changes are observable."""
        return [cls._entry(name) for name in cls.get_all_names()]


# Keep this attribute for callers that inspect the old registry shape. Entries
# are refreshed by ``get_model_at_index``/``get`` and should not be mutated.
LLMRegistry.LLMS = [] if settings.LOCAL_MVP_ONLY else LLMRegistry._legacy_llms()
