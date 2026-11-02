"""Session mutations must yield the event loop while SQL row locks are awaited."""

import asyncio
import threading
from unittest.mock import patch

from app.services.database import DatabaseService


def test_session_mutations_run_outside_event_loop() -> None:
    """A synchronous session writer cannot prevent async review readers releasing locks."""
    service = object.__new__(DatabaseService)
    loop_thread = threading.get_ident()
    observed: list[int] = []

    def update(session_id: str, name: str) -> object:
        observed.append(threading.get_ident())
        return (session_id, name)

    def delete(session_id: str) -> bool:
        observed.append(threading.get_ident())
        return session_id == "test-session"

    async def check() -> None:
        with patch.object(service, "_update_session_name", update), patch.object(service, "_delete_session", delete):
            await asyncio.gather(
                service.update_session_name("test-session", "name"), service.delete_session("test-session")
            )

    asyncio.run(check())
    assert len(observed) == 2
    assert all(worker != loop_thread for worker in observed)
