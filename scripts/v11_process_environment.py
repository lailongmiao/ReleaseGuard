"""Explicit subprocess environment; never propagate parent Provider credentials."""

import os
from collections.abc import Mapping


def minimal_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Keep Windows runtime locations while excluding credentials and proxy settings."""
    source = os.environ if parent is None else parent
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PATHEXT",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
    }
    return {key: value for key, value in source.items() if key.upper() in allowed}


def offline_environment() -> dict[str, str]:
    """Local synthetic processes must opt in to their own test DB separately."""
    return {
        **minimal_environment(),
        "APP_ENV": "test",
        "LOCAL_MVP_ONLY": "true",
        "LOCAL_MVP_ENABLED": "true",
        "V1_REAL_MODELS_ENABLED": "false",
        "V11_REAL_MODELS_ENABLED": "false",
        "V11_REAL_READS_ENABLED": "false",
        "LANGFUSE_TRACING_ENABLED": "false",
        "LANGSMITH_TRACING": "false",
        "MEMORY_ENABLED": "false",
        "UV_OFFLINE": "true",
        "PYRIGHT_PYTHON_IGNORE_WARNINGS": "true",
        "PYRIGHT_PYTHON_USE_BUNDLED_PYRIGHT": "true",
    }
