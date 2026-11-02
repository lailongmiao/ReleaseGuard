"""Enhancement regressions use real services; database evidence is separate."""

import pytest
import test_local_mvp as mvp_fixtures

from test_local_mvp import approval, call, prepared
from app.services.v1_rag import chinese_terms

product = mvp_fixtures.product


def test_chinese_terms_preserve_literal_boundaries() -> None:
    """Punctuation, Latin identifiers and short spans cannot invent matches."""
    assert chinese_terms("模拟步骤，模拟标识符 SHA 单") == ("模拟", "拟步", "步骤", "拟标", "标识", "识符")
    assert chinese_terms("simulate verify_predictions") == ()
    assert chinese_terms("甲，乙；丙") == ()
    assert "甲乙" not in chinese_terms("甲，乙")


def test_chinese_terms_bound_untrusted_input() -> None:
    """Repeated text is deduplicated and long input has a fixed SQL bound."""
    assert chinese_terms("模拟" * 1000) == ("模拟", "拟模")
    text = "".join(chr(0x4E00 + index) for index in range(100))
    assert len(chinese_terms(text)) == 64
    assert all(term in text and len(term) == 2 for term in chinese_terms(text))


@pytest.fixture
def anyio_backend() -> str:
    """The Graph requires asyncio."""
    return "asyncio"


@pytest.mark.anyio
async def test_pending_filter_precedes_pagination(product: tuple) -> None:
    """Closed leading objects must never hide later pending objects."""
    client, _, _ = product
    views = [(await prepared(client, key=f"page-{i}"))[1] for i in range(6)]
    views.sort(key=lambda v: v["state"]["change_id"])
    for view in views[:3]:
        response = await call(
            client, "POST", f"/changes/{view['state']['change_id']}/approval", approval(view, "REJECTED"), user=2
        )
        assert response.status_code == 200
    first = (await call(client, "GET", "/approvals?limit=2", user=2)).json()
    assert [i["change_id"] for i in first["items"]] == [v["state"]["change_id"] for v in views[3:5]]
    second = (await call(client, "GET", "/approvals?limit=2&cursor=" + first["next_cursor"], user=2)).json()
    assert [i["change_id"] for i in second["items"]] == [views[5]["state"]["change_id"]]
    assert second["next_cursor"] is None
    for user in (1, 3):
        assert (await call(client, "GET", "/approvals?limit=2", user=user)).json() == {
            "items": [],
            "next_cursor": None,
        }


@pytest.mark.anyio
async def test_empty_concurrent_page_keeps_scanned_cursor(product: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Status races can empty a live page but may not terminate its traversal."""
    client, store, _ = product
    views = [(await prepared(client, key=f"race-{i}"))[1] for i in range(5)]
    views.sort(key=lambda v: v["state"]["change_id"])
    original = store.list_records

    async def raced(actor, *, inbox, cursor, limit):
        identifiers = await original(actor, inbox=inbox, cursor=cursor, limit=limit)
        for view in views:
            if view["state"]["change_id"] in identifiers[:2]:
                response = await call(
                    client,
                    "POST",
                    f"/changes/{view['state']['change_id']}/approval",
                    approval(view, "REJECTED"),
                    user=2,
                )
                assert response.status_code == 200
        return identifiers

    monkeypatch.setattr(store, "list_records", raced)
    cursor = ""
    for expected in (views[1], views[3]):
        page = (await call(client, "GET", "/approvals?limit=2&cursor=" + cursor, user=2)).json()
        assert page["items"] == []
        assert page["next_cursor"] == expected["state"]["change_id"]
        cursor = page["next_cursor"]
    monkeypatch.setattr(store, "list_records", original)
    last = (await call(client, "GET", "/approvals?limit=2&cursor=" + cursor, user=2)).json()
    assert len(last["items"]) == 1 and last["next_cursor"] is None
