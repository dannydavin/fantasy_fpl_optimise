import json
from pathlib import Path

import httpx

from fpl import storage
from fpl.ingest import understat as u

FIX = Path(__file__).parent / "fixtures" / "understat"


def _league():
    return json.loads((FIX / "league_EPL_2024_sample.json").read_text(encoding="utf-8"))


def _player():
    return json.loads((FIX / "player_8260_sample.json").read_text(encoding="utf-8"))


def test_season_helpers():
    assert u.season_year("2024-25") == 2024
    assert u.season_label(2025) == "2025-26"
    assert u.strip_accents("Ødegaard Müller") == "Ødegaard Muller"


def test_parse_team_matches():
    df = u.parse_team_matches(_league(), "2024-25")
    assert len(df) == 6  # 3 matches x 2 sides
    storage.conform(df, "understat_team_match")
    first = df[df["team"] == "Manchester United"].iloc[0]
    assert first["opponent"] == "Fulham" and first["is_home"]
    assert first["xg"] > first["xga"]
    assert first["team_short"] == "MUN"
    assert df["npxg"].notna().any()
    # both sides mirror each other
    mu = df[(df["team"] == "Manchester United") & df["is_home"]].iloc[0]
    fu = df[(df["team"] == "Fulham")].iloc[0]
    assert mu["xg"] == fu["xga"]


def test_parse_player_matches_filters_season():
    p = u.parse_player_matches(_player(), "2024-25", ["Manchester City"])
    assert len(p) == 4 and set(p["team"]) == {"Manchester City"}
    storage.conform(p, "understat_player_match")
    assert (p["xg"] >= 0).all() and p["minutes"].max() <= 120
    cur = u.parse_player_matches(_player(), "2026-27", ["Manchester City"])
    assert len(cur) == 2


def test_run_with_mock_transport_and_graceful_failure(data_root, tmp_path):
    league, player = _league(), _player()

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["x-requested-with"] == "XMLHttpRequest"
        if "getLeagueData/EPL/2024" in req.url.path:
            return httpx.Response(200, json=league)
        if "getPlayerData/" in req.url.path:
            return httpx.Response(200, json=player)
        return httpx.Response(404)

    out = u.run(["2024-25", "2023-24"], players=True, cache_dir=tmp_path / "c",
                transport=httpx.MockTransport(handler))
    assert len(out["understat_team_match"]) == 6  # 2023-24 404 -> skipped
    assert len(out["understat_player_match"]) > 0
    assert storage.exists("understat_team_match")
    assert storage.exists("understat_player_match")


def test_run_unreachable_returns_empty(data_root, tmp_path):
    out = u.run(["2024-25"], cache_dir=tmp_path / "c",
                transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    assert out["understat_team_match"].empty
    assert not storage.exists("understat_team_match")
