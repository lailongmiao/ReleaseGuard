#!/bin/bash
set -e

# Print initial environment values (before loading .env)
echo "Starting with these environment variables:"
echo "APP_ENV: ${APP_ENV:-development}"
echo "Initial Database Host: $( [[ -n ${POSTGRES_HOST:-${DB_HOST:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Initial Database Port: $( [[ -n ${POSTGRES_PORT:-${DB_PORT:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Initial Database Name: $( [[ -n ${POSTGRES_DB:-${DB_NAME:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Initial Database User: $( [[ -n ${POSTGRES_USER:-${DB_USER:-}} ]] && echo 'set' || echo 'Not set' )"

# Load environment variables from the appropriate .env file
if [ -f ".env.${APP_ENV}" ]; then
    echo "Loading environment from .env.${APP_ENV}"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue

        # Extract the key
        key=$(echo "$line" | cut -d '=' -f 1)

        # Only set if not already set in environment
        if [[ -z "${!key}" ]]; then
            export "$line"
        else
            echo "Keeping existing value for $key"
        fi
    done <".env.${APP_ENV}"
elif [ -f ".env" ]; then
    echo "Loading environment from .env"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue

        # Extract the key
        key=$(echo "$line" | cut -d '=' -f 1)

        # Only set if not already set in environment
        if [[ -z "${!key}" ]]; then
            export "$line"
        else
            echo "Keeping existing value for $key"
        fi
    done <".env"
else
    echo "Warning: No .env file found. Using system environment variables."
fi

# Normal Chat mode requires a usable credential. Explicit local-only mode
# validates its nonproduction guard instead and never creates a Provider client.
required_vars=("JWT_SECRET_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

chat_credential_status=$(/app/.venv/bin/python - 2>/dev/null <<'PY'
import json
import os
import re
import sys

if os.getenv("LOCAL_MVP_ONLY", "false").lower() == "true":
    if os.getenv("LOCAL_MVP_ENABLED", "false").lower() != "true" or os.getenv("APP_ENV", "development") not in {"development", "test"}:
        print("invalid")
    else:
        print("available")
    sys.exit(0)


def usable(value):
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().lower()
    return not (
        normalized in {"change_me", "changeme", "placeholder", "your_api_key", "your-api-key", "xxx", "test"}
        or normalized.startswith(("your-", "replace-me", "change-me"))
        or "placeholder" in normalized
    )


config = None
for name in ("LLM_ROLE_CONFIGS", "PROVIDER_CONFIGS"):
    raw = os.getenv(name)
    if not raw:
        continue
    try:
        role_map = json.loads(raw)
    except json.JSONDecodeError:
        print("invalid")
        sys.exit(0)
    if not isinstance(role_map, dict):
        print("invalid")
        sys.exit(0)
    if "chat" in role_map:
        if not isinstance(role_map["chat"], dict):
            print("invalid")
            sys.exit(0)
        config = role_map["chat"]
        break

credential = None
if config is not None:
    api_key_env = config.get("api_key_env")
    credentials_ref = config.get("credentials_ref")
    if api_key_env and credentials_ref:
        left = api_key_env[4:] if isinstance(api_key_env, str) and api_key_env.startswith("env:") else api_key_env
        right = credentials_ref[4:] if isinstance(credentials_ref, str) and credentials_ref.startswith("env:") else credentials_ref
        if left != right:
            print("invalid")
            sys.exit(0)
    if api_key_env or credentials_ref:
        reference = api_key_env or credentials_ref
        if not isinstance(reference, str):
            print("invalid")
            sys.exit(0)
        env_name = reference[4:] if reference.startswith("env:") else reference
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            print("invalid")
            sys.exit(0)
        credential = os.getenv(env_name)
    elif config.get("api_key"):
        credential = config["api_key"]
    else:
        credential = os.getenv("CHAT_LLM_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
else:
    credential = os.getenv("CHAT_LLM_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

print("available" if usable(credential) else "placeholder" if credential else "missing")
PY
)
if [[ "$chat_credential_status" != "available" ]]; then
    if [[ "$chat_credential_status" == "invalid" ]]; then
        missing_vars+=("valid Chat provider configuration and credential reference")
    elif [[ "$chat_credential_status" == "placeholder" ]]; then
        missing_vars+=("a non-placeholder Chat provider API key")
    else
        missing_vars+=("a Chat provider API key (CHAT_LLM_API_KEY or OPENAI_API_KEY compatibility fallback)")
    fi
fi

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: The following required environment variables are missing:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo "Please provide these variables through environment or .env files."
    exit 1
fi

# Print final environment info
echo -e "\nFinal environment configuration:"
echo "Environment: ${APP_ENV:-development}"

echo "Database Host: $( [[ -n ${POSTGRES_HOST:-${DB_HOST:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Database Port: $( [[ -n ${POSTGRES_PORT:-${DB_PORT:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Database Name: $( [[ -n ${POSTGRES_DB:-${DB_NAME:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Database User: $( [[ -n ${POSTGRES_USER:-${DB_USER:-}} ]] && echo 'set' || echo 'Not set' )"

echo "LLM Model: ${DEFAULT_LLM_MODEL:-Not set}"
echo "Debug Mode: ${DEBUG:-false}"

# Run migrations if necessary with `make docker-migrate`.

# Execute the CMD
exec "$@"
