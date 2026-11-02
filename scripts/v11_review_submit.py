"""Local review-only CLI: never load a sending runtime or connect to a database."""

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Install process-only offline guards before importing application modules."""
    os.environ.update(
        APP_ENV="test",
        LOCAL_MVP_ONLY="true",
        LOCAL_MVP_ENABLED="true",
        V1_REAL_MODELS_ENABLED="false",
        V11_REAL_MODELS_ENABLED="false",
        V11_REAL_READS_ENABLED="false",
        LANGFUSE_TRACING_ENABLED="false",
        LANGCHAIN_TRACING_V2="false",
        LANGSMITH_TRACING="false",
        MEMORY_ENABLED="false",
        SESSION_NAMING_ENABLED="false",
        UV_OFFLINE="true",
        PYRIGHT_PYTHON_IGNORE_WARNINGS="true",
        PYRIGHT_PYTHON_USE_BUNDLED_PYRIGHT="true",
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("evals.v11_backend.review_submission", run_name="__main__")


if __name__ == "__main__":
    main()
