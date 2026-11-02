"""Isolated ReleaseGuard evaluation; never a business workflow or live executor."""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from evals.releaseguard.boundary import OfflineBoundary


def bootstrap_logging(**kwargs: object) -> None:
    """Discard import-time application logs; never install its file handler."""
    return None


# Keep legacy app import-time tokenizer/client initialization offline as well.
with (
    patch.object(tempfile, "tempdir", str(Path.cwd())),
    patch.object(sys, "dont_write_bytecode", True),
    patch.object(logging, "basicConfig", bootstrap_logging),
    OfflineBoundary() as bootstrap_boundary,
):
    from app.core.langgraph.graph import build_agent_graph as build_agent_graph
