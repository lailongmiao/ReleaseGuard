"""Durable GitHub attempt ceiling for the single, explicitly signed formal batch."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import override

import httpx

from app.services.releaseguard.state_machine import DomainError
from evals.v11_backend.controlled import save_new


class FormalGitHubTransport(httpx.AsyncBaseTransport):
    """Consume an attempt before transport; failed or uncertain attempts are not refunded."""

    def __init__(
        self,
        directory: Path,
        inner: httpx.AsyncBaseTransport | None = None,
        check_time: Callable[[], None] | None = None,
        carried_attempts: int = 0,
    ) -> None:
        """Reopen only a contiguous journal, retaining every previous attempt."""
        self.directory = directory
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0)
        self.lock = asyncio.Lock()
        self.check_time = check_time
        if carried_attempts not in (0, 18):
            raise DomainError("evaluation_github_carryover_invalid")
        numbers = sorted(int(p.stem.removeprefix("github-attempt-")) for p in directory.glob("github-attempt-*.json"))
        if numbers != list(range(carried_attempts + 1, carried_attempts + len(numbers) + 1)):
            raise DomainError("evaluation_github_journal_invalid")
        self.attempts = carried_attempts + len(numbers)

    @override
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """No credentials, headers or response body enter the attempt journal."""
        async with self.lock:
            if self.check_time is not None:
                self.check_time()
            if request.method != "GET" or request.url.scheme != "https" or request.url.host != "api.github.com":
                raise DomainError("evaluation_github_scope_invalid")
            if self.attempts >= 2000:
                raise DomainError("evaluation_github_attempt_limit")
            number = self.attempts + 1
            save_new(
                self.directory / f"github-attempt-{number:04d}.json",
                {
                    "attempt": number,
                    "status": "claimed_no_refund",
                    "path": request.url.path,
                },
            )
            self.attempts = number
        return await self.inner.handle_async_request(request)

    @override
    async def aclose(self) -> None:
        """Individual collection clients do not reset or close the shared batch transport."""

    async def shutdown(self) -> None:
        """The operator server owns shutdown after all requests finish."""
        await self.inner.aclose()
