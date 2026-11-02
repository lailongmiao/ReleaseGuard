"""Real local HTTP/PG/checkpoint acceptance; never print credentials or bodies.

Creates synthetic accounts and finite integer targets. Restarts ONLY a named
releaseguard-mvp-* project. Run against a fresh isolated database; never point
at a production service. Fault modes require mvp_fault_runtime on the server.
"""

import argparse
import json
import secrets
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from pydantic import BaseModel
from rich.console import Console
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential


class ProductReport(BaseModel):
    """Only synthetic verdicts and counts; no raw user/HTTP/Provider payloads."""

    mode: str
    transport: str = "actual_http_postgresql_async_postgres_saver"
    checks: list[dict[str, Any]] = []
    restarts: list[dict[str, str]] = []
    passed: bool = False
    failure: str | None = None
    differing_paths: list[str] = []
    external_provider: str = "unverified_not_called"
    production_execution: str = "disabled"


class ProductProbe:
    """Exercise the product boundary without direct DB writes or forged facts."""

    def __init__(self, url: str, compose: Path, mode: str) -> None:
        """Validate a loopback endpoint and exact isolated project name."""
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("probe_requires_local_http")
        self.args = json.loads(compose.read_text(encoding="utf-8"))
        if self.args[:3] != ["docker", "compose", "--project-name"]:
            raise ValueError("probe_requires_compose_command_file")
        if not self.args[3].startswith("releaseguard-mvp-") or len(self.args) not in {10, 12}:
            raise ValueError("probe_requires_isolated_project")
        self.container = self.args[3] + "-app-1"
        self.client = httpx.Client(base_url=url, trust_env=False, timeout=45)
        self.report = ProductReport(mode=mode)
        self.nonce = uuid4().hex
        self.credentials: dict[int, str] = {}
        self.identities: dict[int, dict] = {}
        self.session_ids: dict[int, str] = {}

    def check(self, name: str, condition: bool) -> None:
        """Report a bounded named verdict; never include assertion raw values."""
        self.report.checks.append({"case": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def equal_facts(self, before: Any, after: Any, path: str = "") -> bool:
        """Diagnose field paths only; never record differing payload values."""
        if before == after:
            return True
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                self.equal_facts(before.get(key), after.get(key), path + "/" + str(key))
        elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            for number, (left, right) in enumerate(zip(before, after, strict=True)):
                self.equal_facts(left, right, path + "/" + str(number))
        else:
            self.report.differing_paths.append(path)
        return False

    def request(
        self, method: str, path: str, body: dict | None = None, user: int | None = 1, expected: int = 200
    ) -> dict:
        """Keep tokens and all failure response bodies out of the report."""
        headers = {"Authorization": "Bearer " + self.credentials[user]} if user is not None else {}
        response = self.client.request(method, path, json=body, headers=headers)
        if response.status_code != expected:
            raise AssertionError(f"http_{method}_{path.split('/')[-1]}_expected_{expected}_got_{response.status_code}")
        if expected >= 500 or not response.content:
            return {}
        return response.json()

    @retry(
        stop=stop_after_delay(60),
        wait=wait_exponential(multiplier=0.5, max=3),
        retry=retry_if_exception_type((httpx.HTTPError, AssertionError)),
        reraise=True,
    )
    def ready(self) -> None:
        """Bounded polling of the actual readiness route, not a startup exit code."""
        result = self.request("GET", "/ready", user=None)
        assert result["ready"] is True and result["production_execution"] == "disabled"

    def restart(self, name: str) -> None:
        """Verify a new container process and the same retained isolated volumes."""

        def started() -> str:
            """Read only the process timestamp, never the container environment."""
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.StartedAt}}", self.container],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            return result.stdout.strip()

        before = started()
        result = subprocess.run([*self.args, "restart", "app"], capture_output=True, timeout=45, check=False)
        if result.returncode:
            raise AssertionError("isolated_restart_failed")
        self.ready()
        after = started()
        self.check(name, before != after)
        self.report.restarts.append({"before": before, "after": after})

    def accounts(self) -> None:
        """Use actual registration/session APIs; credentials stay in memory."""
        for number in (1, 2, 3):
            response = self.request(
                "POST",
                "/api/v1/auth/register",
                {
                    "email": f"synthetic-probe-{self.nonce}-{number}@example.com",
                    "password": secrets.token_urlsafe(24) + "Aa1!",
                    "username": "synthetic-probe",
                },
                user=None,
            )
            self.credentials[number] = response["token"]["access_token"]
            session = self.request("POST", "/api/v1/auth/session", user=number)
            self.credentials[number] = session["token"]["access_token"]
            self.session_ids[number] = session["session_id"]
            self.identities[number] = self.request("GET", "/api/v1/releaseguard/identity", user=number)
        self.check("registered_session_identity", len({x["user_id"] for x in self.identities.values()}) == 3)

    def body(self, key: str, scenario: str = "complete") -> dict:
        """Build a current synthetic ChangeRequest with server-issued identity."""
        now = datetime.now(UTC)
        return {
            "change": {
                "requester_id": self.identities[1]["principal_id"],
                "service": "synthetic-service",
                "environment": "staging",
                "title": "Synthetic local change",
                "description": "Finite register",
                "change_window": {
                    "start": now.isoformat(),
                    "end": (now + timedelta(hours=1)).isoformat(),
                    "timezone": "UTC",
                },
                "rollback_plan": "Restore synthetic integer",
                "verification_plan": "Independent integer observation",
            },
            "approver_user_id": self.identities[2]["user_id"],
            "idempotency_key": self.nonce + ":" + key,
            "evidence_scenario": scenario,
        }

    def prepare(
        self, key: str, scenario: str = "complete", mode: str = "offline_simulation", ttl: int = 600
    ) -> tuple[str, dict]:
        """Create then invoke the real Graph through the product API."""
        created = self.request("POST", "/api/v1/releaseguard/changes", self.body(key, scenario))
        path = "/api/v1/releaseguard/changes/" + created["state"]["change_id"]
        view = self.request(
            "POST",
            path + "/prepare",
            {
                "command_id": "prepare",
                "expected_state_version": 0,
                "mode": mode,
                "approval_ttl_seconds": ttl,
            },
        )
        return path, view

    def decision(self, view: dict, decision: str = "APPROVED", key: str = "decision") -> dict:
        """Only typed intent and exact server-issued authorization binding."""
        approval = view["approval"]["request"]
        return {
            "command_id": key,
            "expected_state_version": view["state"]["state_version"],
            "approval_request_id": approval["approval_request_id"],
            "authorization_hash": approval["authorization_hash"],
            "decision": decision,
            "reason_code": "synthetic_review",
        }

    def approve(self, path: str, view: dict) -> dict:
        """A different designated DB user makes the approval decision."""
        return self.request("POST", path + "/approval", self.decision(view), user=2)

    def run_body(self, view: dict, key: str = "run") -> dict:
        """Explicit CAS and idempotent command identifier."""
        return {"command_id": key, "expected_state_version": view["state"]["state_version"]}

    def normal(self) -> None:
        """Product chain, denial cases and actual approval/receipt restart."""
        paths = [self.prepare("simulation"), self.prepare("sandbox", mode="offline_sandbox")]
        self.check(
            "two_sources_citations_high_risk_wait",
            all(
                v["state"]["status"] == "WAITING_FOR_APPROVAL"
                and len(v["state"]["evidence_items"]) == 2
                and v["state"]["risk_assessment"]["level"] == "HIGH"
                for _, v in paths
            ),
        )
        self.restart("approval_wait_process_restart")
        saved_results = []
        for (path, previous), mode, status, writes in zip(
            paths, ("simulation", "sandbox"), ("SIMULATED", "SUCCEEDED"), (0, 1), strict=True
        ):
            saved = self.request("GET", path)
            self.check(mode + "_approval_recovered", saved == previous)
            self.request("POST", path + "/run", self.run_body(saved, "pending"), expected=409)
            self.check(mode + "_pending_zero_call", self.request("GET", path)["target_write_calls"] == 0)
            approved = self.approve(path, saved)
            command = self.run_body(approved)
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        lambda _, route=path, value=command: self.request("POST", route + "/run", value), range(2)
                    )
                )
            view = results[0]
            self.check(mode + "_concurrent_exact_replay", results[0] == results[1])
            self.check(
                mode + "_independent_receipt",
                view["state"]["receipt"]["final_status"] == status
                and view["target_write_calls"] == writes
                and view["synthetic"] is True
                and view["production_execution"] == "disabled",
            )
            saved_results.append((path, command, view))
        self.restart("completed_facts_process_restart")
        for path, command, view in saved_results:
            self.check(
                "receipt_state_idempotency_recovered_" + view["state"]["receipt"]["final_status"],
                self.request("GET", path) == view
                and self.request("POST", path + "/run", command) == view
                and bool(self.request("GET", path + "/audit")["events"])
                and self.request("GET", path + "/receipt")["receipt"] == view["state"]["receipt"],
            )
        path, view = self.prepare("object-denials")
        self.request("GET", "/api/v1/releaseguard/identity", user=None, expected=401)
        for suffix in ("", "/audit", "/receipt"):
            self.request("GET", path + suffix, user=3, expected=404)
        self.request("POST", path + "/approval", self.decision(view), expected=403)
        self.check("authentication_object_and_approver_denials", self.request("GET", path)["target_write_calls"] == 0)
        for scenario in ("missing", "expired", "conflicting", "unavailable"):
            path, view = self.prepare(scenario, scenario)
            self.request("POST", path + "/run", self.run_body(view), expected=409)
            self.check(
                "evidence_" + scenario + "_zero_call",
                view["state"]["status"] == "WAITING_FOR_EVIDENCE"
                and view["approval"] is None
                and self.request("GET", path)["target_write_calls"] == 0,
            )
        for decision in ("REJECTED", "NEEDS_CHANGES"):
            path, view = self.prepare(decision)
            result = self.request("POST", path + "/approval", self.decision(view, decision), user=2)
            self.request("POST", path + "/run", self.run_body(result), expected=409)
            self.check("approval_" + decision + "_zero_call", self.request("GET", path)["target_write_calls"] == 0)
        path, view = self.prepare("revoke-drift")
        wrong = self.decision(view)
        wrong["authorization_hash"] = "0" * 64
        self.request("POST", path + "/approval", wrong, user=2, expected=409)
        decided = self.request("POST", path + "/approval", self.decision(view, key="valid"), user=2)
        revoked = self.request("POST", path + "/revoke", self.run_body(decided, "revoke"))
        self.request("POST", path + "/run", self.run_body(revoked), expected=409)
        self.check("binding_drift_and_revocation_zero_call", self.request("GET", path)["target_write_calls"] == 0)
        path, view = self.prepare("expiry", ttl=1)
        sleep(1.1)
        expired = self.approve(path, view)
        self.request("POST", path + "/run", self.run_body(expired), expected=409)
        self.check("approval_expiry_zero_call", self.request("GET", path)["target_write_calls"] == 0)
        value = self.body("concurrent-create")
        with ThreadPoolExecutor(max_workers=2) as executor:
            created = list(
                executor.map(lambda _: self.request("POST", "/api/v1/releaseguard/changes", value), range(2))
            )
        self.check("concurrent_create_same_key", created[0] == created[1])
        altered = deepcopy(value)
        altered["change"]["title"] = "Different same key"
        self.request("POST", "/api/v1/releaseguard/changes", altered, expected=409)
        self.check("idempotency_conflict_rejected", True)
        path, view = self.prepare("revision", scenario="missing")
        revised = self.request(
            "POST",
            path + "/revision",
            {
                **self.run_body(view, "revise"),
                "change": self.body("unused")["change"],
                "evidence_scenario": "complete",
            },
        )
        prepared = self.request("POST", path + "/prepare", self.run_body(revised, "prepare-r2"))
        decided = self.approve(path, prepared)
        result = self.request("POST", path + "/run", self.run_body(decided))
        self.check("revision_supplementation", result["state"]["receipt"]["revision"] == 2)
        self.request("DELETE", "/api/v1/auth/session/" + self.session_ids[3], user=3)
        self.request("GET", "/api/v1/releaseguard/identity", user=3, expected=401)
        self.check("deleted_session_revokes_access", True)

    def fault(self) -> None:
        """Real HTTP with explicit server-only injection and durable restart."""
        path, view = self.prepare(self.report.mode, mode="offline_sandbox")
        approved = self.approve(path, view)
        command = self.run_body(approved)
        result = self.request("POST", path + "/run", command, expected=500 if self.report.mode == "orphan" else 200)
        saved = self.request("GET", path)
        expected_calls = 0 if self.report.mode == "orphan" else 1
        self.check(
            "fault_saved_safe_hold", saved["recovery"] == "safe_hold" and saved["target_write_calls"] == expected_calls
        )
        if self.report.mode == "unknown":
            self.check(
                "unknown_not_success",
                result["state"]["status"] == "EXECUTION_UNKNOWN"
                and result["state"]["receipt"]["final_status"] == "UNKNOWN",
            )
        if self.report.mode == "verification":
            self.check(
                "independent_verification_not_success",
                result["state"]["status"] == "RECOVERY_REQUIRED"
                and result["state"]["verification_result"]["status"] == "FAILED"
                and (result["state"]["receipt"] is None or result["state"]["receipt"]["final_status"] != "SUCCEEDED"),
            )
        self.restart("fault_process_restart")
        restored = self.request("GET", path)
        self.check("fault_facts_recovered", self.equal_facts(saved, restored))
        duplicate = self.request("POST", path + "/run", command, expected=409 if self.report.mode == "orphan" else 200)
        if self.report.mode != "orphan":
            self.check("fault_original_result_replayed", duplicate == result)
        self.request("POST", path + "/run", self.run_body(restored, "new-run"), expected=409)
        self.check("fault_zero_resend", self.request("GET", path)["target_write_calls"] == expected_calls)


def main() -> int:
    """Write sanitized evidence even when a product check fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--mode", choices=["normal", "unknown", "orphan", "verification"], default="normal")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    probe = ProductProbe(args.url, args.compose_command, args.mode)
    try:
        probe.ready()
        probe.accounts()
        probe.normal() if args.mode == "normal" else probe.fault()
        probe.report.passed = True
    except Exception as error:
        # Assertion messages are generated by this script, never raw HTTP bodies.
        probe.report.failure = str(error) if isinstance(error, AssertionError) else type(error).__name__
    finally:
        probe.client.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(probe.report.model_dump_json(indent=2), encoding="utf-8")
    Console().print(
        {
            "mode": args.mode,
            "passed": probe.report.passed,
            "checks": len(probe.report.checks),
            "restarts": len(probe.report.restarts),
            "failure": probe.report.failure,
        }
    )
    return 0 if probe.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
