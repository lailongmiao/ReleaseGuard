"""Deny external I/O for the trusted, finite offline evaluation composition."""

import builtins
import io
import os
import socket
import subprocess
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch


class OfflineBoundary:
    """Process-local guard, used serially; not a sandbox for untrusted Python."""

    def __init__(self) -> None:
        """Count attempts independently of expected case results."""
        self.network_attempts = 0
        self.process_attempts = 0
        self.write_attempts = 0
        self.stack = ExitStack()
        self.open = builtins.open
        self.io_open = io.open
        self.os_open = os.open

    def network(self, *args: Any, **kwargs: Any) -> Any:
        """Fail before DNS, connect or datagram send."""
        self.network_attempts += 1
        raise PermissionError("offline_network_denied")

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Fail before starting a child process."""
        self.process_attempts += 1
        raise PermissionError("offline_process_denied")

    def write(self, *args: Any, **kwargs: Any) -> Any:
        """Fail before host filesystem mutations."""
        self.write_attempts += 1
        raise PermissionError("offline_host_write_denied")

    def checked_open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        """Allow reads, never file writes from a case."""
        if any(flag in mode for flag in "wax+"):
            return self.write()
        return self.open(file, mode, *args, **kwargs)

    def checked_io_open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        """Pathlib uses io.open rather than builtins.open."""
        if any(flag in mode for flag in "wax+"):
            return self.write()
        return self.io_open(file, mode, *args, **kwargs)

    def checked_os_open(self, file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        """Block low-level writable file descriptors too."""
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return self.write()
        return self.os_open(file, flags, *args, **kwargs)

    def __enter__(self) -> "OfflineBoundary":
        """Patch known I/O boundaries and restore them even on cancellation."""
        for name in ("connect", "connect_ex", "sendto"):
            self.stack.enter_context(patch.object(socket.socket, name, self.network))
        self.stack.enter_context(patch.object(socket, "getaddrinfo", self.network))
        self.stack.enter_context(patch.object(subprocess, "Popen", self.process))
        self.stack.enter_context(patch.object(os, "system", self.process))
        self.stack.enter_context(patch.object(builtins, "open", self.checked_open))
        self.stack.enter_context(patch.object(io, "open", self.checked_io_open))
        self.stack.enter_context(patch.object(os, "open", self.checked_os_open))
        for name in ("mkdir", "remove", "unlink", "rename", "replace", "rmdir"):
            self.stack.enter_context(patch.object(os, name, self.write))
        return self

    def __exit__(self, *args: Any) -> None:
        """Do not swallow failures or retain global patches."""
        self.stack.close()
