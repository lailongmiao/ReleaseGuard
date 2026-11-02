"""Invalid credentials must never enter auth logging, even as prefixes."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.v1 import auth


@pytest.fixture
def anyio_backend() -> str:
    """Use the project's installed asyncio runtime, without optional trio."""
    return "asyncio"


@pytest.mark.anyio
async def test_invalid_token_never_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspect logger arguments without generating or outputting any real JWT."""
    log = Mock()
    monkeypatch.setattr(auth, "logger", log)
    monkeypatch.setattr(auth, "verify_token", lambda _: None)
    marker = "synthetic-private-invalid-credential"
    with pytest.raises(HTTPException) as error:
        await auth.get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=marker))
    assert error.value.status_code == 401
    assert marker not in repr(log.mock_calls)
    assert marker[:10] not in repr(log.mock_calls)
