"""Explicit local evaluation server; a separately signed capture permit is required."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import uvicorn

from app.core.config import settings
from app.main import app
from evals.v11_backend.http_evidence import EvidencePermit, HTTPEvaluationEvidence, router, verify_windows_acl


def main() -> None:
    """No signature is issued here, no API capabilities or real-call flags are enabled."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not settings.LOCAL_MVP_ONLY:
        raise ValueError("local_only_required")
    for path in (args.directory, args.permit, args.key_file):
        if path.is_symlink():
            raise ValueError("private_path_required")
        verify_windows_acl(path)
    raw = json.loads(args.permit.read_text(encoding="utf-8"))
    app.state.review_evaluation = HTTPEvaluationEvidence(
        args.directory, EvidencePermit.model_validate(raw["permit"]), raw["signature"], args.key_file.read_bytes()
    )
    app.include_router(router, prefix="/_evaluation")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port))
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        try:
            runner.run(server.serve())
        finally:
            transport = app.state.review_evaluation.github_transport
            if transport is not None:
                runner.run(transport.shutdown())


if __name__ == "__main__":
    main()
