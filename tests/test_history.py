from pathlib import Path

import httpx
import pandas as pd

from fpl import storage
from fpl.ingest import history

FIX = Path(__file__).parent / "fixtures" / "history"


def _frames(season):
    return (
        pd.read_csv(FIX / f"merged_gw_{season}.csv"),
        pd.read_csv(FIX / f"players_raw_{season}.csv"),
        pd.read_csv(FIX / f"teams_{season}.csv"),
    )


def test_previous_seasons():
    assert history.previous_seasons("2026-27") == ["2023-24", "2024-25", "2025-26"]


def test_normalise_contract_new_season():
    m, pr, tm = _frames("2025-26")
    out = history.normalise_merged_gw(m, "2025-26", pr, tm)
    stored = storage.conform(out, "hist_player_gw")
    assert len(stored) == len(m)
    assert set(out["position"]) <= {"GKP", "DEF", "MID", "FWD"}
    assert out["code"].notna().all()
    assert out["defensive_contribution"].notna().any()
    assert out["opponent"].notna().all()
    assert out["value"].between(3.5, 16).all()  # tenths -> £m
    assert out["kickoff_time"].dt.tz is not None


def test_normalise_old_season_has_no_dc_and_maps_gk():
    m, pr, tm = _frames("2023-24")
    out = history.normalise_merged_gw(m, "2023-24", pr, tm)
    assert out["defensive_contribution"].isna().all()
    assert "GK" not in set(out["position"])
    assert out["expected_goals"].notna().any()


def test_normalise_without_players_raw_and_dgw_rows_kept():
    m, _, _ = _frames("2025-26")
    dup = pd.concat([m, m.iloc[:2]], ignore_index=True)  # simulate a DGW's second fixture
    out = history.normalise_merged_gw(dup, "2025-26")
    assert len(out) == len(dup)
    assert out["code"].isna().all()
    assert out.duplicated(["element", "gw"]).sum() >= 2


def test_mojibake_fix():
    assert history._fix_mojibake("MartÃ­n") == "Martín"
    assert history._fix_mojibake("Plain") == "Plain"


def test_run_offline_with_mock_transport(data_root, tmp_path):
    seasons = {"2025-26": _frames("2025-26")}

    def handler(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/data/")[1].split("/")
        season, fname = parts[0], parts[-1]
        if season not in seasons:
            return httpx.Response(404)
        m, pr, tm = seasons[season]
        body = {"merged_gw.csv": m, "players_raw.csv": pr, "teams.csv": tm}[fname]
        return httpx.Response(200, content=body.to_csv(index=False).encode())

    df = history.run(
        ["2025-26", "2022-23"], cache_dir=tmp_path / "c", transport=httpx.MockTransport(handler)
    )
    assert set(df["season"]) == {"2025-26"}
    assert storage.exists("hist_player_gw")
    back = storage.read_table("hist_player_gw")
    assert len(back) == len(df)
    # cached: second run needs no transport
    again = history.run(["2025-26"], cache_dir=tmp_path / "c", transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))
    assert len(again) == len(df)
