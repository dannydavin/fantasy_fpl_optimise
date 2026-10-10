from __future__ import annotations

import pytest

from fpl import storage
from fpl.ingest import fpl_snapshot as snap
from tests.conftest import NEXT_GW, PICKS_GW, TEAM_ID, load_fixture

SQUAD = {
    "Verbruggen", "Tzolakis", "Gabriel", "Calafiori", "Diop", "Ajayi", "Guéhi", "Ødegaard",
    "Rogers", "Groß", "King", "Cherki", "João Pedro", "Haaland", "Wissa",
}


@pytest.fixture
def snapshot(client, data_root):
    return snap.run(client)


def test_run_writes_all_tables(snapshot):
    assert snapshot == NEXT_GW
    for t in ("players", "teams", "fixtures", "events", "my_entry", "my_team", "my_history",
              "my_transfers"):
        assert storage.exists(t, NEXT_GW), t


def test_my_team_has_the_15_names(snapshot):
    team = storage.read_table("my_team", NEXT_GW)
    players = storage.read_table("players", NEXT_GW).set_index("player_id")
    assert len(team) == 15
    names = {players.loc[i, "web_name"] for i in team["player_id"]}
    assert names == SQUAD
    assert sorted(team["slot"]) == list(range(1, 16))
    assert team["is_captain"].sum() == 1
    assert set(team["gw"]) == {NEXT_GW}


def test_my_entry_bank_ft_chips(snapshot):
    e = storage.read_table("my_entry", NEXT_GW).iloc[0]
    assert e["entry_id"] == TEAM_ID
    assert e["bank"] == pytest.approx(0.4)
    assert e["free_transfers"] == 1
    assert e["picks_gw"] == PICKS_GW
    assert set(e["chips_available"].split(",")) == {"freehit", "bboost", "3xc"}


def test_squad_value_matches_entry(snapshot):
    team = storage.read_table("my_team", NEXT_GW)
    e = storage.read_table("my_entry", NEXT_GW).iloc[0]
    assert team["price"].sum() + e["bank"] == pytest.approx(e["value"], abs=0.35)
    assert (team["sell_price"] <= team["price"] + 1e-9).all()


def test_purchase_price_from_latest_transfer(bootstrap):
    transfers = load_fixture("entry_1608168_transfers")
    picks = load_fixture("entry_1608168_event_6_picks")
    elements = {e["id"]: e for e in bootstrap["elements"]}
    pp = snap.purchase_prices(picks["picks"], elements, transfers)
    last_in = max((t for t in transfers if t["element_in"] == 268), key=lambda t: t["time"])
    assert pp[268] == last_in["element_in_cost"] / 10 == 5.6


@pytest.mark.parametrize(
    "purchase,now,expected",
    [(5.0, 5.0, 5.0), (5.0, 5.1, 5.0), (5.0, 5.2, 5.1), (5.0, 5.5, 5.2), (5.0, 5.7, 5.3),
     (5.0, 4.8, 4.8), (10.0, 10.9, 10.4)],
)
def test_sell_price(purchase, now, expected):
    assert snap.sell_price(purchase, now) == pytest.approx(expected)


def test_free_transfer_rules():
    hist = {"current": [{"event": g, "event_transfers": 0} for g in range(1, 6)], "chips": []}
    assert snap.compute_free_transfers(hist, 5) == 5  # 1 rolled each GW, capped
    hist["current"].append({"event": 6, "event_transfers": 3})
    assert snap.compute_free_transfers(hist, 6) == 3  # 5 - 3 used + 1
    hist["current"][-1]["event_transfers"] = 7
    assert snap.compute_free_transfers(hist, 6) == 1  # hits take bank to 0, then +1
    hist["chips"] = [{"event": 6, "name": "freehit"}]
    assert snap.compute_free_transfers(hist, 6) == 5  # free hit keeps the bank


def test_chip_week_freezes_free_transfers():
    hist = {"current": [{"event": g, "event_transfers": 0} for g in range(1, 4)], "chips": []}
    assert snap.compute_free_transfers(hist, 3) == 3
    hist["current"].append({"event": 4, "event_transfers": 12})
    hist["chips"] = [{"event": 4, "name": "wildcard"}]
    assert snap.compute_free_transfers(hist, 4) == 3  # kept, but no +1 for the WC week


def test_chips_available_respects_windows(bootstrap):
    hist = {"chips": [{"name": "wildcard", "event": 4}]}
    assert set(snap.chips_available(bootstrap, hist, 7)) == {"freehit", "bboost", "3xc"}
    assert set(snap.chips_available(bootstrap, {"chips": []}, 7)) == {
        "wildcard", "freehit", "bboost", "3xc"}
    assert "wildcard" in snap.chips_available(bootstrap, hist, 20)


def test_pending_transfers_applied():
    picks = load_fixture("entry_1608168_event_6_picks")
    out_id = picks["picks"][0]["element"]
    pending = [{"event": 7, "time": "2026-10-11T10:00:00Z", "element_out": out_id,
                "element_out_cost": 60, "element_in": 1, "element_in_cost": 55}]
    new, delta, n = snap.apply_pending(picks, pending, 7)
    assert n == 1 and delta == pytest.approx(0.5)
    assert new["picks"][0]["element"] == 1
    assert picks["picks"][0]["element"] == out_id  # input untouched


def test_players_price_and_risk_columns(bootstrap):
    df = snap.players_table(bootstrap, NEXT_GW)
    h = df[df["web_name"] == "Haaland"].iloc[0]
    raw = next(e for e in bootstrap["elements"] if e["id"] == 411)
    assert h["position"] == "FWD"
    assert h["price"] == raw["now_cost"] / 10
    assert set(df["price_risk"]) <= {"rise", "fall", "stable"}
    assert {"net_transfers_event", "projected_percent", "likelihood"} <= set(df.columns)


def test_history_has_chip(snapshot):
    h = storage.read_table("my_history", NEXT_GW).set_index("gw")
    assert h.loc[4, "chip"] == "wildcard"
    assert h.loc[6, "bank"] == pytest.approx(0.4)
