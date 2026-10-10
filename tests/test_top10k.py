from __future__ import annotations

import httpx

from fpl import storage
from fpl.api import FPLClient
from fpl.ingest import top10k
from tests.conftest import PICKS_GW, TEAM_ID, load_fixture


def test_parse_standings():
    df = top10k.parse_standings(
        load_fixture("leagues-classic_314_standings_page_standings1_phase1"), PICKS_GW
    )
    assert len(df) == 50
    assert {"gw", "entry_id", "rank", "total", "entry_name", "player_name"} <= set(df.columns)
    assert df["rank"].is_monotonic_increasing
    assert (df["gw"] == PICKS_GW).all()


def test_parse_picks():
    raw = load_fixture(f"entry_{TEAM_ID}_event_6_picks")
    df = top10k.parse_picks(raw, TEAM_ID, 7, PICKS_GW)
    assert len(df) == 15
    assert df["is_captain"].sum() == 1
    assert df["is_vice_captain"].sum() == 1
    assert (df["entry_id"] == TEAM_ID).all()
    assert df["multiplier"].sum() >= 12
    assert df["bank"].iloc[0] == raw["entry_history"]["bank"] / 10


def _mock_client(bad: set[int]):
    standings = load_fixture("leagues-classic_314_standings_page_standings1_phase1")
    picks = load_fixture(f"entry_{TEAM_ID}_event_6_picks")

    def handler(req: httpx.Request) -> httpx.Response:
        parts = req.url.path.strip("/").split("/")
        if "leagues-classic" in parts:
            page = int(req.url.params["page_standings"])
            data = {**standings, "standings": {**standings["standings"], "has_next": page < 2}}
            if page == 2:
                data["standings"]["results"] = [
                    {**r, "entry": r["entry"] + 10_000_000, "rank": r["rank"] + 50}
                    for r in standings["standings"]["results"]
                ]
            return httpx.Response(200, json=data)
        if "picks" in parts:
            eid = int(parts[parts.index("entry") + 1])
            if eid in bad:
                return httpx.Response(404, json={})
            return httpx.Response(200, json=picks)
        return httpx.Response(404, json={})

    return FPLClient(transport=httpx.MockTransport(handler), use_cache=False, rate=0)


def test_run_end_to_end(data_root):
    first = load_fixture("leagues-classic_314_standings_page_standings1_phase1")["standings"][
        "results"
    ][0]["entry"]
    with _mock_client({first}) as c:
        gw = top10k.run(c, limit=80, gw=PICKS_GW, workers=3)
    assert gw == PICKS_GW
    entries = storage.read_table("top10k_entries", PICKS_GW)
    picks = storage.read_table("top10k_picks", PICKS_GW)
    assert len(entries) == 79  # 80 limit, one 404 skipped
    assert first not in set(entries["entry_id"])
    assert len(picks) == 79 * 15
    assert picks["entry_id"].nunique() == 79
    assert picks["rank"].max() <= 80


def test_run_stops_when_no_next_page(data_root):
    # limit 40 -> 1 page only; ranks 1..40
    with _mock_client(set()) as c:
        top10k.run(c, limit=40, gw=PICKS_GW)
    assert len(storage.read_table("top10k_entries", PICKS_GW)) == 40
