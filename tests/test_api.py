import httpx
import pytest

from fpl import api
from fpl.api import FPLAPIError, FPLClient, cache_key
from fpl.gameweeks import current_gw, last_deadline_gw, next_gw
from tests.conftest import PICKS_GW, TEAM_ID

MY_SQUAD = {
    "Verbruggen", "Tzolakis", "Gabriel", "Calafiori", "Diop", "Ajayi", "Guéhi",
    "Ødegaard", "Rogers", "Groß", "King", "Cherki", "João Pedro", "Haaland", "Wissa",
}


def test_cache_key():
    assert cache_key("entry/1/event/6/picks/") == "entry_1_event_6_picks"
    assert (
        cache_key("leagues-classic/314/standings/", {"phase": 1, "page_standings": 2})
        == "leagues-classic_314_standings_page_standings2_phase1"
    )


def test_bootstrap_offline(client, bootstrap):
    bs = client.bootstrap_static()
    assert len(bs["teams"]) == 20
    assert len(bs["elements"]) > 500
    assert next_gw(bs) == 7
    assert current_gw(bs) == 6


def test_last_deadline_gw(bootstrap):
    from datetime import UTC, datetime

    assert last_deadline_gw(bootstrap, now=datetime(2026, 10, 12, tzinfo=UTC)) == 6
    assert last_deadline_gw(bootstrap, now=datetime(2026, 10, 17, 10, 0, 1, tzinfo=UTC)) == 7


def test_my_picks_offline(client):
    bs = client.bootstrap_static()
    names = {e["id"]: e["web_name"] for e in bs["elements"]}
    picks = client.entry_picks(TEAM_ID, PICKS_GW)
    assert {names[p["element"]] for p in picks["picks"]} == MY_SQUAD
    assert picks["entry_history"]["bank"] == 4


def test_other_endpoints_offline(client):
    assert client.entry(TEAM_ID)["id"] == TEAM_ID
    assert client.entry_history(TEAM_ID)["chips"][0]["name"] == "wildcard"
    assert isinstance(client.entry_transfers(TEAM_ID), list)
    assert len(client.fixtures()) == 380
    assert "history" in client.element_summary(411)
    page = client.league_standings(314, 1)
    assert len(page["standings"]["results"]) == 50
    assert client.event_live(5)["elements"]


def test_missing_fixture_raises(client):
    with pytest.raises(FPLAPIError):
        client.element_summary(999999)


def test_retry_then_success(tmp_path, monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    c = FPLClient(cache_dir=tmp_path, transport=httpx.MockTransport(handler), rate=0)
    assert c.get("thing/") == {"ok": True}
    assert calls["n"] == 3
    # second call served from disk cache
    assert c.get("thing/") == {"ok": True}
    assert calls["n"] == 3
    assert (tmp_path / "thing.json").exists()


def test_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    c = FPLClient(
        cache_dir=tmp_path, transport=httpx.MockTransport(lambda r: httpx.Response(429)), rate=0
    )
    with pytest.raises(FPLAPIError):
        c.get("x/")


def test_throttle_spacing(monkeypatch):
    t = api._Throttle(rate=4)
    slept = []
    monkeypatch.setattr(api.time, "sleep", slept.append)
    for _ in range(5):
        t.wait()
    # sleep is mocked so time doesn't advance: each call is scheduled 0.25s after the last
    assert slept == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=0.05)
