from __future__ import annotations

import pandas as pd
import pytest

from fpl import storage
from fpl.features import fixtures as fx
from fpl.ingest import fpl_snapshot as snap
from tests.conftest import NEXT_GW, load_fixture


def _fx(rows):
    return pd.DataFrame(
        [
            {"fixture_id": i, "gw": gw, "team_h": h, "team_a": a, "team_h_difficulty": 3,
             "team_a_difficulty": 3, "finished": fin}
            for i, (gw, h, a, fin) in enumerate(rows, 1)
        ]
    )


def test_blank_and_double_detection():
    f = _fx([(1, 1, 2, True), (2, 1, 2, False), (2, 1, 3, False), (3, 2, 3, False)])
    cal = fx.blank_double_calendar(f).set_index(["team_id", "gw"])["kind"]
    assert cal[(1, 2)] == "double"
    assert cal[(2, 3)] == "single"
    assert cal[(1, 3)] == "blank"
    assert cal[(3, 1)] == "blank"
    counts = fx.gw_fixture_counts(f).set_index(["team_id", "gw"])["n_fixtures"]
    assert counts[(1, 2)] == 2 and (1, 3) not in counts.index


def test_team_match_xg_splits_dgw_and_pairs_opponents():
    f = _fx([(1, 1, 2, True), (2, 1, 2, True), (2, 3, 1, True)])
    pgw = pd.DataFrame(
        [
            {"gw": 1, "team_id": 1, "expected_goals": 2.0},
            {"gw": 1, "team_id": 2, "expected_goals": 0.5},
            {"gw": 2, "team_id": 1, "expected_goals": 3.0},
            {"gw": 2, "team_id": 2, "expected_goals": 1.0},
            {"gw": 2, "team_id": 3, "expected_goals": 0.2},
        ]
    )
    m = fx.team_match_xg(pgw, f)
    t1 = m[(m.team_id == 1) & (m.gw == 2)].set_index("fixture_id")
    assert t1.loc[2, "xgf"] == pytest.approx(1.5)  # DGW xG split in two
    assert t1.loc[2, "xga"] == pytest.approx(1.0)  # team 2 had one fixture in gw2
    assert t1.loc[3, "xga"] == pytest.approx(0.2)


def test_ratings_rank_strong_team_hardest():
    rows = []
    for gw in range(1, 5):
        rows += [
            (gw * 10 + 1, gw, 1, 2, 3.0, 0.3), (gw * 10 + 1, gw, 2, 1, 0.3, 3.0),
            (gw * 10 + 2, gw, 3, 4, 1.0, 1.0), (gw * 10 + 2, gw, 4, 3, 1.0, 1.0),
        ]
    m = pd.DataFrame(rows, columns=["fixture_id", "gw", "team_id", "opponent_id", "xgf", "xga"])
    r = fx.team_ratings(m)
    assert r["strength"].idxmax() == 1 and r["strength"].idxmin() == 2
    f = _fx([(5, 1, 2, False), (5, 3, 4, False)])
    t = fx.ticker_table(f, 5, r).set_index(["team_id"])
    assert t.loc[2, "difficulty"] > t.loc[4, "difficulty"]  # facing strong team 1 vs average 3
    assert t["difficulty"].between(1, 5).all()


def test_ticker_without_ratings_falls_back_to_fdr():
    f = _fx([(5, 1, 2, False)])
    t = fx.ticker_table(f, 5, None)
    assert (t["difficulty"] == t["fdr"]).all()


def test_run_ticker_on_recorded_data(client, data_root):
    snap.run(client)
    t = fx.run(client=client)
    assert storage.exists("fixture_ticker", NEXT_GW)
    assert t["gw"].min() >= NEXT_GW
    assert t["team_id"].nunique() == 20
    assert t["difficulty"].between(1, 5).all()
    raw = load_fixture("fixtures")
    n_future = sum(1 for x in raw if x["event"] and x["event"] >= NEXT_GW)
    assert len(t) == 2 * n_future
    # custom difficulty actually differs from the official FDR
    assert (t["difficulty"] != t["fdr"]).any()
    cal = fx.blank_double_calendar(storage.read_table("fixtures", NEXT_GW))
    assert set(cal["kind"]) <= {"blank", "single", "double"}
