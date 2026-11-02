"""Local configuration static boundaries without opening private env files."""

import json
import ast
from pathlib import Path

import yaml
import pytest

from app.core.config import Settings
from app.services.runtime_readiness import MIGRATION_HEAD

ROOT = Path(__file__).resolve().parents[1]


def test_provider_free_local_mode_is_explicit_and_nonproduction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local-only is opt-in and disables optional model operations."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOCAL_MVP_ENABLED", "true")
    monkeypatch.setenv("LOCAL_MVP_ONLY", "true")
    configuration = Settings()
    assert not configuration.MEMORY_ENABLED
    assert not configuration.SESSION_NAMING_ENABLED
    assert not configuration.LANGFUSE_TRACING_ENABLED
    monkeypatch.setenv("LOCAL_MVP_ENABLED", "false")
    with pytest.raises(ValueError, match="local_mvp_only_requires_explicit_enable"):
        Settings()
    monkeypatch.setenv("LOCAL_MVP_ENABLED", "true")
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValueError, match="local_mvp_requires_development_or_test"):
        Settings()


def test_readiness_head_matches_template_migration() -> None:
    """A new migration requires an explicit readiness contract update."""
    migrations = list((ROOT / "alembic/versions").glob("*.py"))
    revisions, parents = set(), set()
    for migration in migrations:
        module = ast.parse(migration.read_text(encoding="utf-8"))
        for node in module.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Constant)
            ):
                if node.target.id == "revision":
                    revisions.add(node.value.value)
                elif node.target.id == "down_revision" and node.value.value is not None:
                    parents.add(node.value.value)
    assert parents <= revisions
    assert revisions - parents == {MIGRATION_HEAD}


def test_local_ports_and_no_fixed_admin_password() -> None:
    """Container config keeps local access scoped and requires private admin data."""
    config = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for service in config["services"].values():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:")
    assert "=admin" not in " ".join(config["services"]["grafana"]["environment"])
    assert config["services"]["cadvisor"]["profiles"] == ["linux-host"]
    assert any("/ready" in item for item in config["services"]["app"]["healthcheck"]["test"])


def test_dashboard_and_local_only_alerts() -> None:
    """Provisioned datasource/queries are explicit, without notification targets."""
    dashboard = json.loads((ROOT / "grafana/dashboards/json/releaseguard.json").read_text(encoding="utf-8"))
    assert dashboard["uid"] == "releaseguard-local"
    rules = yaml.safe_load((ROOT / "prometheus/rules/releaseguard.yml").read_text(encoding="utf-8"))
    assert len(rules["groups"][0]["rules"]) == 2
    assert "alertmanager" not in (ROOT / "prometheus/prometheus.yml").read_text(encoding="utf-8")
