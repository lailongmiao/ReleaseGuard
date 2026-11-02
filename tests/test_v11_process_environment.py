"""A failing child must not receive parent credentials or proxy configuration."""

import os
import subprocess
import sys

from scripts.v11_process_environment import minimal_environment, offline_environment


def test_environment_is_allowlist_and_preserves_windows_locations():
    """Exercise the named invariant without external capabilities."""
    parent = {"SystemDrive": "C:", "PATH": "tools", "ANTHROPIC_AUTH_TOKEN": "synthetic-sentinel"}
    assert minimal_environment(parent) == {"SystemDrive": "C:", "PATH": "tools"}


def test_failure_output_does_not_expand_parent_secrets(monkeypatch):
    """Exercise the named invariant without external capabilities."""
    sentinel = "synthetic-do-not-propagate-sentinel"
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "HTTPS_PROXY", "JWT_SECRET_KEY"):
        monkeypatch.setenv(name, sentinel)
    env = offline_environment()
    result = subprocess.run(
        [sys.executable, "-c", "import os,sys; print(dict(os.environ)); sys.exit(7)"],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 7
    assert sentinel.encode() not in result.stdout + result.stderr
    error = subprocess.CalledProcessError(result.returncode, [sys.executable, "synthetic-child"])
    assert sentinel not in str(error)
    assert env["V11_REAL_MODELS_ENABLED"] == "false"
    if os.name == "nt":
        assert any(k.upper() == "SYSTEMDRIVE" for k in env)
