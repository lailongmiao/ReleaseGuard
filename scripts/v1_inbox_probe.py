"""Actual HTTP/PG inbox regression on a new isolated stack, with zero models."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

from rich.console import Console

from scripts.mvp_product_probe import ProductProbe
from scripts.v1_provider_probe import isolated_database
from app.schemas.local_mvp import LocalActor
from app.services.local_mvp import LocalMVPService
from app.services.local_mvp_storage import LocalPostgresStore


async def concurrent_pages(probe: ProductProbe, rows: list[tuple[str, dict]]) -> None:
    """Pause real PG listing between candidate selection and HTTP decisions.

    Only scheduling is injected: decisions use the running API, persisted facts
    use the actual repository/service; no snapshot writes or substitute Graph.
    """

    class BarrierStore(LocalPostgresStore):
        """A deterministic scheduling barrier around the actual SQL query."""

        @override
        async def list_records(self, actor: LocalActor, *, inbox: bool, cursor: str, limit: int) -> list[str]:
            identifiers = await super().list_records(actor, inbox=inbox, cursor=cursor, limit=limit)
            for path, view in rows:
                if view["state"]["change_id"] in identifiers[:2]:
                    await asyncio.to_thread(
                        probe.request, "POST", path + "/approval", probe.decision(view, "REJECTED"), 2
                    )
            return identifiers

    async def no_graph() -> Any:
        raise AssertionError("listing_must_not_execute_graph")

    store = BarrierStore()
    actor = await store.authenticate(probe.session_ids[2], datetime.now(UTC).timestamp() + 3600)
    service = LocalMVPService(no_graph, store)
    cursor = ""
    for number in (1, 3):
        page = await service.listing(actor, inbox=True, cursor=cursor, limit=2)
        probe.check(
            "pg_http_concurrent_empty_page_" + str(number),
            not page["items"] and page["next_cursor"] == rows[number][1]["state"]["change_id"],
        )
        cursor = page["next_cursor"]
    last = await asyncio.to_thread(
        probe.request, "GET", "/api/v1/releaseguard/approvals?limit=2&cursor=" + cursor, None, 2
    )
    probe.check(
        "pg_http_concurrent_final_page",
        len(last["items"]) == 1
        and last["items"][0]["change_id"] == rows[-1][1]["state"]["change_id"]
        and last["next_cursor"] is None,
    )


def main() -> int:
    """Create synthetic users/objects; never modify an existing database."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-command", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:35080")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    probe = ProductProbe(args.url, args.compose_command, "inbox_pagination")
    try:
        probe.ready()
        probe.accounts()
        rows = sorted((probe.prepare("page-" + str(i)) for i in range(6)), key=lambda row: row[0])
        for path, view in rows[:3]:
            probe.request("POST", path + "/approval", probe.decision(view, "REJECTED"), user=2)
        root = "/api/v1/releaseguard/approvals?limit=2"
        first = probe.request("GET", root, user=2)
        probe.check(
            "pending_before_limit",
            [v["change_id"] for v in first["items"]] == [v["state"]["change_id"] for _, v in rows[3:5]],
        )
        second = probe.request("GET", root + "&cursor=" + first["next_cursor"], user=2)
        probe.check(
            "last_page_exact",
            [v["change_id"] for v in second["items"]] == [rows[5][1]["state"]["change_id"]]
            and second["next_cursor"] is None,
        )
        for user in (1, 3):
            probe.check(
                "other_user_empty_" + str(user),
                probe.request("GET", root, user=user) == {"items": [], "next_cursor": None},
            )
        probe.restart("pending_new_process")
        probe.check("pending_page_restored", probe.request("GET", root, user=2) == first)
        for path, view in rows[3:]:
            probe.request("POST", path + "/approval", probe.decision(view, "REJECTED"), user=2)
        raced = sorted((probe.prepare("race-" + str(i)) for i in range(5)), key=lambda row: row[0])
        isolated_database(args.compose_command)
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            runner.run(concurrent_pages(probe, raced))
        probe.report.passed = True
    except Exception as error:
        probe.report.failure = (
            type(error).__name__ + ":" + str(error)[:180]
            if isinstance(error, AssertionError)
            else type(error).__name__
        )
    finally:
        probe.client.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(probe.report.model_dump_json(indent=2), encoding="utf-8")
    Console().print(
        {"passed": probe.report.passed, "checks": len(probe.report.checks), "failure": probe.report.failure}
    )
    return 0 if probe.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
