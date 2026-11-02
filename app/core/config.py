"""Application configuration management.

This module handles environment-specific configuration loading, parsing, and management
for the application. It includes environment detection, .env file loading, and
configuration value parsing.
"""

import os
import json
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr


WEAK_JWT_SECRET_KEYS = {
    "",
    "changeme",
    "change_me",
    "change_me_to_a_random_32_plus_char_string",
    "supersecretkeythatshouldbechangedforproduction",
    "your-jwt-secret-key",
}


# Define environment types
class Environment(str, Enum):
    """Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production, and test.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# Determine environment
def get_environment() -> Environment:
    """Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


# Load appropriate .env file based on environment
def load_env_file():
    """Load environment-specific .env file."""
    env = get_environment()
    print(f"Loading environment: {env}")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define env files in priority order
    env_files = [
        os.path.join(base_dir, f".env.{env.value}.local"),
        os.path.join(base_dir, f".env.{env.value}"),
        os.path.join(base_dir, ".env.local"),
        os.path.join(base_dir, ".env"),
    ]

    # Load the first env file that exists
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(dotenv_path=env_file)
            print(f"Loaded environment from {env_file}")
            return env_file

    # Fall back to default if no env file found
    return None


ENV_FILE = load_env_file()


# Parse list values from environment variables
def parse_list_from_env(env_key, default=None):
    """Parse a comma-separated list from an environment variable."""
    value = os.getenv(env_key)
    if not value:
        return default or []

    # Remove quotes if they exist
    value = value.strip("\"'")
    # Handle single value case
    if "," not in value:
        return [value]
    # Split comma-separated values
    return [item.strip() for item in value.split(",") if item.strip()]


# Parse dict of lists from environment variables with prefix
def parse_dict_of_lists_from_env(prefix, default_dict=None):
    """Parse dictionary of lists from environment variables with a common prefix."""
    result = default_dict or {}

    # Look for all env vars with the given prefix
    for key, value in os.environ.items():
        if key.startswith(prefix):
            endpoint = key[len(prefix) :].lower()  # Extract endpoint name
            # Parse the values for this endpoint
            if value:
                value = value.strip("\"'")
                if "," in value:
                    result[endpoint] = [item.strip() for item in value.split(",") if item.strip()]
                else:
                    result[endpoint] = [value]

    return result


def _env_first(*keys: str, default: str = "") -> str:
    """Return the first explicitly configured environment value."""
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_bool(key: str, default: bool) -> bool:
    """Parse a boolean environment value without treating arbitrary text as true."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "on"}


def _env_int(key: str, default: int | None = None) -> int | None:
    """Parse an optional positive integer from the environment."""
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{key} must be positive")
    return parsed


def _env_mapping(*keys: str) -> dict | None:
    """Parse one optional JSON object from environment configuration."""
    for key in keys:
        value = os.getenv(key)
        if value is None or not value.strip():
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{key} must contain a valid JSON object") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{key} must contain a JSON object")
        return parsed
    return None


class Settings:
    """Application settings without using pydantic."""

    def __init__(self):
        """Initialize application settings from environment variables.

        Loads and sets all configuration values from environment variables,
        with appropriate defaults for each setting. Also applies
        environment-specific overrides based on the current environment.
        """
        # Set the environment
        self.ENVIRONMENT = get_environment()

        # Application Settings
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "FastAPI LangGraph Template")
        self.VERSION = os.getenv("VERSION", "1.0.0")
        self.DESCRIPTION = os.getenv(
            "DESCRIPTION", "A production-ready FastAPI template with LangGraph and Langfuse integration"
        )
        self.API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
        self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "yes")

        # CORS Settings
        self.ALLOWED_ORIGINS = parse_list_from_env("ALLOWED_ORIGINS", ["*"])

        # Langfuse Configuration
        self.LANGFUSE_TRACING_ENABLED = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in (
            "true",
            "1",
            "t",
            "yes",
        )
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # LangGraph Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-5.6-luna")
        self.SESSION_NAMING_ENABLED = os.getenv("SESSION_NAMING_ENABLED", "true").lower() == "true"
        self.DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.2"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
        self.MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", str(self.MAX_TOKENS)))
        self.MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
        self.MAX_LLM_CALL_RETRIES = int(os.getenv("MAX_LLM_CALL_RETRIES", "3"))
        self.LLM_TOTAL_TIMEOUT = int(os.getenv("LLM_TOTAL_TIMEOUT", "60"))

        # Provider-neutral role configuration. The legacy OpenAI variables remain
        # valid defaults, but each role can now select its own provider and model.
        self.LLM_PROVIDER = _env_first("LLM_PROVIDER", default="openai")
        self.LLM_ENDPOINT = _env_first("LLM_ENDPOINT", "OPENAI_BASE_URL", default="")
        self.LLM_API_KEY = _env_first("LLM_API_KEY", "OPENAI_API_KEY", default=self.OPENAI_API_KEY)
        self.LLM_CREDENTIALS_REF = _env_first("LLM_CREDENTIALS_REF", default="")

        self.CHAT_LLM_PROVIDER = _env_first("CHAT_LLM_PROVIDER", "CHAT_PROVIDER", default=self.LLM_PROVIDER)
        self.CHAT_LLM_MODEL = _env_first("CHAT_LLM_MODEL", "CHAT_MODEL", default=self.DEFAULT_LLM_MODEL)
        self.CHAT_LLM_ENDPOINT = _env_first("CHAT_LLM_ENDPOINT", "CHAT_ENDPOINT", default=self.LLM_ENDPOINT)
        self.CHAT_LLM_API_KEY = _env_first("CHAT_LLM_API_KEY", "CHAT_API_KEY", default=self.LLM_API_KEY)
        self.CHAT_LLM_CREDENTIALS_REF = _env_first(
            "CHAT_LLM_CREDENTIALS_REF", "CHAT_CREDENTIALS_REF", default=self.LLM_CREDENTIALS_REF
        )
        self.CHAT_LLM_FALLBACK_MODELS = parse_list_from_env("CHAT_LLM_FALLBACK_MODELS", [])
        self.CHAT_LLM_SUPPORTS_CHAT = _env_bool("CHAT_LLM_SUPPORTS_CHAT", True)
        self.CHAT_LLM_SUPPORTS_STREAMING = _env_bool("CHAT_LLM_SUPPORTS_STREAMING", True)
        self.CHAT_LLM_SUPPORTS_TOOLS = _env_bool("CHAT_LLM_SUPPORTS_TOOLS", True)
        self.CHAT_LLM_SUPPORTS_STRUCTURED_OUTPUT = _env_bool("CHAT_LLM_SUPPORTS_STRUCTURED_OUTPUT", True)
        self.CHAT_LLM_SUPPORTS_REASONING = _env_bool("CHAT_LLM_SUPPORTS_REASONING", False)
        self.CHAT_LLM_SUPPORTS_USAGE = _env_bool("CHAT_LLM_SUPPORTS_USAGE", True)
        self.CHAT_LLM_CONTEXT_WINDOW = _env_int("CHAT_LLM_CONTEXT_WINDOW", 128000)
        self.CHAT_LLM_TOKENIZER = _env_first("CHAT_LLM_TOKENIZER", default="")

        self.MEMORY_LLM_PROVIDER = _env_first("MEMORY_LLM_PROVIDER", default=self.LLM_PROVIDER)
        self.MEMORY_LLM_MODEL = _env_first("MEMORY_LLM_MODEL", "LONG_TERM_MEMORY_MODEL", default="gpt-5-nano")
        self.MEMORY_LLM_ENDPOINT = _env_first("MEMORY_LLM_ENDPOINT", default=self.LLM_ENDPOINT)
        self.MEMORY_LLM_API_KEY = _env_first("MEMORY_LLM_API_KEY", default=self.LLM_API_KEY)
        self.MEMORY_LLM_CREDENTIALS_REF = _env_first("MEMORY_LLM_CREDENTIALS_REF", default=self.LLM_CREDENTIALS_REF)
        self.MEMORY_LLM_SUPPORTS_CHAT = _env_bool("MEMORY_LLM_SUPPORTS_CHAT", True)
        self.MEMORY_LLM_SUPPORTS_USAGE = _env_bool("MEMORY_LLM_SUPPORTS_USAGE", True)

        self.EMBEDDING_PROVIDER = _env_first("EMBEDDING_PROVIDER", default=self.LLM_PROVIDER)
        self.EMBEDDING_MODEL = _env_first(
            "EMBEDDING_MODEL", "LONG_TERM_MEMORY_EMBEDDER_MODEL", default="text-embedding-3-small"
        )
        self.EMBEDDING_ENDPOINT = _env_first("EMBEDDING_ENDPOINT", default=self.LLM_ENDPOINT)
        self.EMBEDDING_API_KEY = _env_first("EMBEDDING_API_KEY", default=self.LLM_API_KEY)
        self.EMBEDDING_CREDENTIALS_REF = _env_first("EMBEDDING_CREDENTIALS_REF", default=self.LLM_CREDENTIALS_REF)
        self.EMBEDDING_SUPPORTS_EMBEDDINGS = _env_bool("EMBEDDING_SUPPORTS_EMBEDDINGS", True)
        self.EMBEDDING_DIMENSIONS = _env_int("EMBEDDING_DIMENSIONS", None)
        self.EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", str(self.LLM_TOTAL_TIMEOUT)))
        self.EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", str(self.MAX_LLM_CALL_RETRIES)))
        self.EMBEDDING_RETRY_MIN_SECONDS = float(os.getenv("EMBEDDING_RETRY_MIN_SECONDS", "1"))
        self.EMBEDDING_RETRY_MAX_SECONDS = float(os.getenv("EMBEDDING_RETRY_MAX_SECONDS", "8"))

        self.NAMING_LLM_PROVIDER = _env_first("NAMING_LLM_PROVIDER", default=self.CHAT_LLM_PROVIDER)
        self.NAMING_LLM_MODEL = _env_first("NAMING_LLM_MODEL", default=self.CHAT_LLM_MODEL)
        self.NAMING_LLM_ENDPOINT = _env_first("NAMING_LLM_ENDPOINT", default=self.CHAT_LLM_ENDPOINT)
        self.NAMING_LLM_API_KEY = _env_first("NAMING_LLM_API_KEY", default=self.LLM_API_KEY)
        self.NAMING_LLM_CREDENTIALS_REF = _env_first(
            "NAMING_LLM_CREDENTIALS_REF", default=self.CHAT_LLM_CREDENTIALS_REF
        )
        self.NAMING_LLM_SUPPORTS_STRUCTURED_OUTPUT = _env_bool("NAMING_LLM_SUPPORTS_STRUCTURED_OUTPUT", True)
        self.NAMING_LLM_SUPPORTS_CHAT = _env_bool("NAMING_LLM_SUPPORTS_CHAT", True)
        self.NAMING_LLM_SUPPORTS_USAGE = _env_bool("NAMING_LLM_SUPPORTS_USAGE", True)

        self.JUDGE_LLM_PROVIDER = _env_first("JUDGE_LLM_PROVIDER", default=self.LLM_PROVIDER)
        self.JUDGE_LLM_MODEL = _env_first("JUDGE_LLM_MODEL", "EVALUATION_LLM", default=self.CHAT_LLM_MODEL)
        self.JUDGE_LLM_ENDPOINT = _env_first("JUDGE_LLM_ENDPOINT", default=self.LLM_ENDPOINT)
        self.JUDGE_LLM_API_KEY = _env_first("JUDGE_LLM_API_KEY", default=self.LLM_API_KEY)
        self.JUDGE_LLM_CREDENTIALS_REF = _env_first("JUDGE_LLM_CREDENTIALS_REF", default=self.LLM_CREDENTIALS_REF)
        self.JUDGE_LLM_SUPPORTS_STRUCTURED_OUTPUT = _env_bool("JUDGE_LLM_SUPPORTS_STRUCTURED_OUTPUT", True)
        self.JUDGE_LLM_SUPPORTS_CHAT = _env_bool("JUDGE_LLM_SUPPORTS_CHAT", True)
        self.JUDGE_LLM_SUPPORTS_USAGE = _env_bool("JUDGE_LLM_SUPPORTS_USAGE", True)

        # Optional structured role configuration. Invalid JSON fails startup
        # instead of silently falling back to legacy single-role variables.
        self.LLM_ROLE_CONFIGS = _env_mapping("LLM_ROLE_CONFIGS")
        self.PROVIDER_CONFIGS = _env_mapping("PROVIDER_CONFIGS")

        # Long term memory Configuration
        self.LONG_TERM_MEMORY_MODEL = os.getenv("LONG_TERM_MEMORY_MODEL", "gpt-5-nano")
        self.LONG_TERM_MEMORY_EMBEDDER_MODEL = os.getenv("LONG_TERM_MEMORY_EMBEDDER_MODEL", "text-embedding-3-small")
        self.LONG_TERM_MEMORY_COLLECTION_NAME = os.getenv("LONG_TERM_MEMORY_COLLECTION_NAME", "longterm_memory")
        self.MEMORY_ENABLED = _env_bool("MEMORY_ENABLED", True)
        self.LOCAL_MVP_ENABLED = _env_bool("LOCAL_MVP_ENABLED", False)
        self.LOCAL_MVP_ONLY = _env_bool("LOCAL_MVP_ONLY", False)
        self.V1_BACKEND_ENABLED = _env_bool("V1_BACKEND_ENABLED", False)
        self.V1_REAL_MODELS_ENABLED = _env_bool("V1_REAL_MODELS_ENABLED", False)
        self.V11_BACKEND_ENABLED = _env_bool("V11_BACKEND_ENABLED", False)
        self.V11_REAL_READS_ENABLED = _env_bool("V11_REAL_READS_ENABLED", False)
        self.V11_REAL_MODELS_ENABLED = _env_bool("V11_REAL_MODELS_ENABLED", False)
        self.V11_BUDGET_CATEGORY = os.getenv("V11_BUDGET_CATEGORY", "acceptance")
        self.V11_BUDGET_LEDGER_ID = os.getenv("V11_BUDGET_LEDGER_ID", "")
        if self.V11_BUDGET_CATEGORY not in {"smoke", "development", "acceptance"}:
            raise ValueError("v11_budget_category_invalid")
        if self.V11_BACKEND_ENABLED and not self.V1_BACKEND_ENABLED:
            raise ValueError("v11_requires_v1_backend")
        self.V1_SOURCE_OWNER_ID = _env_int("V1_SOURCE_OWNER_ID", 0) or 0
        raw_v1_checks = os.getenv("V1_REQUIRED_CHECKS", "[]")
        v1_checks = json.loads(raw_v1_checks)
        if (
            not isinstance(v1_checks, list)
            or any(not isinstance(v, str) or not v or len(v) > 128 for v in v1_checks)
            or len(set(v1_checks)) != len(v1_checks)
        ):
            raise ValueError("v1_required_checks_json_list_invalid")
        self.V1_REQUIRED_CHECKS = tuple(v1_checks)
        self.V1_CHAT_MODEL = _env_first("V1_CHAT_MODEL", default="deepseek-flash")
        self.V1_GITHUB_READ_TOKEN = SecretStr(os.getenv("V1_GITHUB_READ_TOKEN", ""))
        self.V1_MODEL_BUDGET_CNY = float(os.getenv("V1_MODEL_BUDGET_CNY", "100"))
        if not 0 < self.V1_MODEL_BUDGET_CNY <= 100:
            raise ValueError("v1_budget_must_be_positive_and_at_most_100")
        if self.V1_BACKEND_ENABLED and not self.LOCAL_MVP_ENABLED:
            raise ValueError("v1_requires_local_mvp_enable")
        if self.V1_BACKEND_ENABLED and self.ENVIRONMENT not in {Environment.DEVELOPMENT, Environment.TEST}:
            raise ValueError("v1_requires_development_or_test")
        if self.LOCAL_MVP_ENABLED and self.ENVIRONMENT not in {Environment.DEVELOPMENT, Environment.TEST}:
            raise ValueError("local_mvp_requires_development_or_test")
        if self.LOCAL_MVP_ONLY:
            if not self.LOCAL_MVP_ENABLED:
                raise ValueError("local_mvp_only_requires_explicit_enable")
            self.MEMORY_ENABLED = False
            self.SESSION_NAMING_ENABLED = False
            self.LANGFUSE_TRACING_ENABLED = False
        self.EVIDENCE_PROVIDER = _env_first("EVIDENCE_PROVIDER", default="fixture")
        self.EVIDENCE_MODEL = _env_first("EVIDENCE_MODEL", default="")
        self.EVIDENCE_RETRIEVAL_LIMIT = int(os.getenv("EVIDENCE_RETRIEVAL_LIMIT", "5"))
        self.EVIDENCE_SNIPPET_MAX_CHARS = int(os.getenv("EVIDENCE_SNIPPET_MAX_CHARS", "600"))
        # JWT Configuration
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_DAYS", "30"))
        self.validate_jwt_secret_key()

        # Logging Configuration
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"

        # Profiling Configuration (DEBUG only)
        self.PROFILING_DIR = Path(os.getenv("PROFILING_DIR", "/tmp/fastapi_profiles"))
        self.PROFILING_THRESHOLD_SECONDS = float(os.getenv("PROFILING_THRESHOLD_SECONDS", "2.0"))

        # Postgres Configuration
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "food_order_db")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
        self.POSTGRES_MAX_OVERFLOW = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
        self.CHECKPOINT_TABLES = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]

        # Valkey/Redis Cache Configuration (optional — if host is set, caching is enabled)
        self.VALKEY_HOST = os.getenv("VALKEY_HOST", "")
        self.VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))
        self.VALKEY_DB = int(os.getenv("VALKEY_DB", "0"))
        self.VALKEY_PASSWORD = os.getenv("VALKEY_PASSWORD", "")
        self.VALKEY_MAX_CONNECTIONS = int(os.getenv("VALKEY_MAX_CONNECTIONS", "20"))
        self.CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))

        # Rate Limiting Configuration
        self.RATE_LIMIT_DEFAULT = parse_list_from_env("RATE_LIMIT_DEFAULT", ["200 per day", "50 per hour"])

        # Rate limit endpoints defaults
        default_endpoints = {
            "chat": ["30 per minute"],
            "chat_stream": ["20 per minute"],
            "messages": ["50 per minute"],
            "register": ["10 per hour"],
            "login": ["20 per minute"],
            "root": ["10 per minute"],
            "health": ["20 per minute"],
        }

        # Update rate limit endpoints from environment variables
        self.RATE_LIMIT_ENDPOINTS = default_endpoints.copy()
        for endpoint in default_endpoints:
            env_key = f"RATE_LIMIT_{endpoint.upper()}"
            value = parse_list_from_env(env_key)
            if value:
                self.RATE_LIMIT_ENDPOINTS[endpoint] = value

        # Evaluation Configuration
        self.EVALUATION_LLM = os.getenv("EVALUATION_LLM", "gpt-5")
        self.EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", "https://api.openai.com/v1")
        self.EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY", self.OPENAI_API_KEY)
        self.EVALUATION_SLEEP_TIME = int(os.getenv("EVALUATION_SLEEP_TIME", "10"))

        # Apply environment-specific settings
        self.apply_environment_settings()

    def apply_environment_settings(self):
        """Apply environment-specific settings based on the current environment."""
        env_settings = {
            Environment.DEVELOPMENT: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "200 per hour"],
            },
            Environment.STAGING: {
                "DEBUG": False,
                "LOG_LEVEL": "INFO",
                "RATE_LIMIT_DEFAULT": ["500 per day", "100 per hour"],
            },
            Environment.PRODUCTION: {
                "DEBUG": False,
                "LOG_LEVEL": "WARNING",
                "RATE_LIMIT_DEFAULT": ["200 per day", "50 per hour"],
            },
            Environment.TEST: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "1000 per hour"],  # Relaxed for testing
            },
        }

        # Get settings for current environment
        current_env_settings = env_settings.get(self.ENVIRONMENT, {})

        # Apply settings if not explicitly set in environment variables
        for key, value in current_env_settings.items():
            env_var_name = key.upper()
            # Only override if environment variable wasn't explicitly set
            if env_var_name not in os.environ:
                setattr(self, key, value)

    def validate_jwt_secret_key(self):
        """Reject empty, short, or placeholder JWT secrets outside test runs."""
        if self.ENVIRONMENT == Environment.TEST:
            return

        secret = self.JWT_SECRET_KEY.strip()
        if len(secret) < 32:
            raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters long")

        if secret.lower() in WEAK_JWT_SECRET_KEYS:
            raise RuntimeError("JWT_SECRET_KEY must be a strong random value, not a placeholder or default")


# Create settings instance
settings = Settings()
