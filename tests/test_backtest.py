from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl.ingest import history as hist
from fpl.models import backtest as bt

HIST = Path(__file__).parent / "fixtures" / "history"


@pytest.fixture(scope="module")
def history():
    """Two seasons of the recorded sample: 2023-24 acts as the prior for 2025-26."""
    frames = []
    for s in ("2023-24", "2025-26"):
        frames.append(hist.normalise_merged_gw(
            pd.read_csv(HIST / f"merged_gw_{s}.csv"), s,
            pd.read_csv(HIST / f"players_raw_{s}.csv"), pd.read_csv(HIST / f"teams_{s}.csv")))
    return pd.concat(frames, ignore_index=True)


def synthetic_season(n_gw=12, seed=0):
    """Small league: 6 teams, 4 outfield players each with different but stable quality."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    rows, fid = [], 0
    for gw in range(1, n_gw + 1):
        order = rng.permutation(teams)
        for h, a in zip(order[0::2], order[1::2], strict=True):
            fid += 1
            hs, as_ = rng.poisson(1.5), rng.poisson(1.2)
            for team, opp, home in ((h, a, True), (a, h, False)):
                for j in range(4):
                    el = int(team[1]) * 10 + j
                    quality = 0.1 + 0.15 * j
                    mins = 90 if j < 3 else 0
                    g = int(rng.random() < quality) if mins else 0
                    rows.append({
                        "season": "2025-26", "gw": gw, "element": el, "code": 1000 + el,
                        "team": team, "player_name": f"P{el}", "position": "FWD" if j else "GKP",
                        "opponent_team": 0, "opponent": opp, "was_home": home,
                        "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC"),
                        "minutes": mins, "total_points": 2 + 4 * g if mins else 0,
                        "goals_scored": g, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
                        "bonus": 0, "bps": 0, "expected_goals": quality if mins else 0.0,
                        "expected_assists": 0.05 if mins else 0.0, "value": 5.0,
                        "starts": 1 if mins else 0, "saves": 0, "own_goals": 0,
                        "team_h_score": hs, "team_a_score": as_, "fixture_id": fid,
                        "defensive_contribution": 3 if mins else 0,
                    })
    return pd.DataFrame(rows)


def test_spearman():
    a = pd.Series([1, 2, 3, 4, 5.0])
    assert bt.spearman(a, a) == pytest.approx(1.0)
    assert bt.spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(bt.spearman(a, pd.Series([1.0] * 5)))


def test_backtest_shape_and_metrics_on_recorded_sample(history):
    res = bt.backtest("2025-26", hist=history, start_gw=5)
    assert {"gw", "subset", "n", "mae_xp", "mae_ppg", "mae_form3", "rho_xp"} <= set(res.columns)
    assert set(res["subset"]) <= set(bt.SUBSETS)
    assert res["gw"].min() >= 5
    assert (res["mae_xp"] >= 0).all()
    s = bt.summarise(res)
    assert "all" in set(s["subset"])


def test_backtest_synthetic_beats_naive_and_ranks(history):
    h = synthetic_season()
    res = bt.backtest("2025-26", hist=h, start_gw=4)
    s = bt.summarise(res).set_index("subset")
    assert s.loc["played", "rho_xp"] >= -0.2  # noisy tiny sample; just sane, not anti-correlated
    assert s.loc["all", "rho_xp"] > 0.3  # starters vs a never-playing 4th player
    assert s.loc["all", "mae_xp"] < s.loc["all", "mae_ppg"] + 0.5


def test_no_lookahead():
    h = synthetic_season(seed=3)
    base = bt.backtest("2025-26", hist=h, start_gw=4)
    k = 8
    scrambled = h.copy()
    m = scrambled["gw"] >= k
    # change everything about GW >= k that is an outcome (not fixture identity)
    for c in ("total_points", "minutes", "starts", "goals_scored", "expected_goals",
              "expected_assists", "bonus", "saves", "defensive_contribution"):
        scrambled.loc[m, c] = scrambled.loc[m, c] * 3 + 1
    alt = bt.backtest("2025-26", hist=scrambled, start_gw=4)
    p0 = base.attrs["predictions"].query("gw == @k").set_index("element")["pred_xp"]
    p1 = alt.attrs["predictions"].query("gw == @k").set_index("element")["pred_xp"]
    assert p0.index.equals(p1.index)
    # team scores of GW k are outcomes too, but are only used for gw < k: predictions equal
    np.testing.assert_allclose(p0.to_numpy(), p1.to_numpy())
    # whereas GW k+1 may use GW k data, so it changes
    q0 = base.attrs["predictions"].query("gw == @k + 1")["pred_xp"].to_numpy()
    q1 = alt.attrs["predictions"].query("gw == @k + 1")["pred_xp"].to_numpy()
    assert not np.allclose(q0, q1)
