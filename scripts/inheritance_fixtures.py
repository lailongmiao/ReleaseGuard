"""Legacy synthetic boundaries for the explicitly isolated browser probe only."""

from types import SimpleNamespace
from typing import Any

import httpx

from app.schemas.v1_backend import AnalysisOutput, Claim
from app.services.v1_sources import GitHubSource


class LegacyProbeProvider:
    """No SDK, billing, embedding service, or external fallback."""

    provenance = "unverified"

    def config(self, role: Any) -> Any:
        """Match the deterministic local embedding dimension."""
        return SimpleNamespace(
            model="synthetic", provider="fake", capabilities=SimpleNamespace(embedding_dimensions=3)
        )

    async def embed(self, texts: list[str], *args: Any) -> list[list[float]]:
        """Synthetic vectors are generated locally."""
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def analyze(self, query: str, chunks: list, *args: Any) -> tuple:
        """Use supplied evidence IDs, never fabricate an approval."""
        return AnalysisOutput(
            explanation="Synthetic legacy review / 历史合成审阅",
            claims=[Claim(text="Simulation only", citation_ids=[chunks[0].chunk_id])],
            risk="LOW",
            unknowns=[],
            candidate_steps=["validate_inputs", "simulate", "verify_predictions"],
        ), {"model": "synthetic", "cost": "unknown"}


def legacy_source() -> GitHubSource:
    """HTTP requests are handled in memory; no socket reaches GitHub."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method != "GET" or request.url.host != "api.github.com":
            raise AssertionError("unexpected_legacy_source_request")
        path = request.url.path
        if "/pulls/" in path and not path.endswith("/files"):
            data: Any = {"head": {"sha": "a" * 40}}
        elif path.endswith("check-runs"):
            data = {
                "check_runs": [
                    {"id": 1, "head_sha": "a" * 40, "name": "test", "status": "completed", "conclusion": "success"}
                ],
                "total_count": 1,
            }
        elif path.endswith("/files"):
            data = [{"filename": "docs/审阅/历史.md"}]
        else:
            data = [{"tag_name": "synthetic-v1"}]
        return httpx.Response(200, json=data)

    return GitHubSource(httpx.MockTransport(handle))
