from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl import storage
from fpl.features import ownership as own


def _picks(entries: dict[int, list[tuple]], chips: dict[int, str] | None = None) -> pd.DataFrame:
    """entries: entry_id -> [(player, multiplier, is_cap, is_vice)]."""
    rows = []
    for eid, ps in entries.items():
        for i, (pid, mult, cap, vice) in enumerate(ps, 1):
            rows.append(
                dict(gw=6, entry_id=eid, rank=eid, player_id=pid, slot=i, multiplier=mult,
                     is_captain=cap, is_vice_captain=vice,
                     active_chip=(chips or {}).get(eid), event_transfers=0,
                     event_transfers_cost=0, bank=0.0, value=100.0)
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synth():
    # 4 entries. P1: captain in e1,e2 (x2), TC in e3 (x3), benched in e4. P2: vice e1; BB e4 bench=1
    return _picks(
        {
            1: [(1, 2, True, False), (2, 1, False, True), (3, 0, False, False)],
            2: [(1, 2, True, False), (3, 1, False, False)],
            3: [(1, 3, True, False), (2, 1, False, True)],
            4: [(1, 1, False, False), (2, 1, True, False), (3, 1, False, True)],
        },
        chips={3: "3xc", 4: "bboost"},
    )


def test_eo_maths(synth):
    o = own.compute_ownership(synth, 6).set_index("player_id")
    # P1: mults 2,2,3,1 -> 8/4 = 200%
    assert o.loc[1, "eo"] == pytest.approx(200.0)
    assert o.loc[1, "owned_pct"] == 100.0
    assert o.loc[1, "captain_pct"] == 75.0
    assert o.loc[1, "start_pct"] == 100.0  # BB bench (mult 1) counts as start
    # P2: 1,1,1 over 3 owners -> 3/4 = 75%; vice twice
    assert o.loc[2, "eo"] == pytest.approx(75.0)
    assert o.loc[2, "owned_pct"] == 75.0
    assert o.loc[2, "vice_pct"] == 50.0
    # P3: bench in e1 (0), 1, 1 -> 2/4 = 50 EO; owned 3/4; started 2/4
    assert o.loc[3, "eo"] == pytest.approx(50.0)
    assert o.loc[3, "start_pct"] == 50.0
    assert o.loc[3, "owned_pct"] == 75.0


def test_bench_zero_not_in_eo():
    p = _picks({1: [(1, 0, False, False)], 2: [(2, 1, False, False)]})
    o = own.compute_ownership(p, 6).set_index("player_id")
    assert o.loc[1, "eo"] == 0 and o.loc[1, "owned_pct"] == 50 and o.loc[1, "start_pct"] == 0


def test_chip_and_captaincy(synth):
    c = own.chip_usage(synth).set_index("chip")
    assert c.loc["none", "entries"] == 2 and c.loc["3xc", "pct"] == 25.0
    cap = own.captaincy(synth)
    assert cap.iloc[0]["player_id"] == 1 and cap.iloc[0]["captain_pct"] == 75.0


def test_overall_and_prev_delta(synth):
    o = own.compute_ownership(synth, 6)
    players = pd.DataFrame({"player_id": [1, 2], "selected_by_percent": [60.0, 10.0]})
    o2 = own.add_overall(o, players).set_index("player_id")
    assert o2.loc[1, "delta_vs_overall"] == pytest.approx(40.0)
    assert np.isnan(o2.loc[3, "overall_selected_pct"])
    assert np.isnan(own.add_overall(o, None)["overall_selected_pct"]).all()
    prev = pd.DataFrame({"player_id": [1], "owned_pct": [90.0]})
    o3 = own.add_prev_delta(o, prev).set_index("player_id")
    assert o3.loc[1, "delta_vs_prev_gw"] == pytest.approx(10.0)
    assert o3.loc[2, "delta_vs_prev_gw"] == pytest.approx(75.0)


def test_my_exposure_and_high_eo(synth):
    o = own.compute_ownership(synth, 6)
    mine = pd.DataFrame({"player_id": [2, 9], "multiplier": [2, 1]})
    e = own.my_exposure(o, mine).set_index("player_id")
    assert e.loc[1, "exposure"] == pytest.approx(-200.0)  # short the 200 EO captain
    assert e.loc[2, "exposure"] == pytest.approx(125.0)
    assert e.loc[9, "exposure"] == pytest.approx(100.0)
    h = own.high_eo_not_owned(o, mine, min_eo=60)
    assert list(h["player_id"]) == [1]


def test_rank_bands():
    p = _picks({1: [(1, 1, False, False)], 2: [(2, 1, False, False)], 3: [(1, 1, False, False)]})
    b = own.rank_bands(p, bands=(1, 3))
    top1 = b[b["band"] == 1].set_index("player_id")
    assert top1.loc[1, "owned_pct"] == 100 and top1["n_entries"].iloc[0] == 1
    top3 = b[b["band"] == 3].set_index("player_id")
    assert top3.loc[1, "owned_pct"] == pytest.approx(200 / 3)


def _player_pool():
    rows, pid = [], 1
    for pos, n in {"GKP": 4, "DEF": 8, "MID": 8, "FWD": 5}.items():
        for _ in range(n):
            rows.append(dict(player_id=pid, position=pos, team_id=1 + (pid % 8)))
            pid += 1
    return pd.DataFrame(rows)


def test_template_xi_legal():
    pl = _player_pool()
    o = pd.DataFrame(
        {
            "player_id": pl["player_id"],
            "owned_pct": np.linspace(90, 5, len(pl)),
            "start_pct": np.linspace(80, 3, len(pl)),
            "eo": np.linspace(95, 4, len(pl)),
        }
    )
    t = own.template_xi(o, pl)
    sq, xi = t["squad"], t["xi"]
    assert len(sq) == 15 and len(xi) == 11
    assert sq["position"].value_counts().to_dict() == {"DEF": 5, "MID": 5, "FWD": 3, "GKP": 2}
    assert sq["team_id"].value_counts().max() <= 3
    c = xi["position"].value_counts()
    assert c["GKP"] == 1 and 3 <= c["DEF"] <= 5 and 2 <= c["MID"] <= 5 and 1 <= c["FWD"] <= 3
    assert t["formation"] == f"{c['DEF']}-{c['MID']}-{c['FWD']}"
    assert len(t["bench"]) == 4


def test_run_writes_table_with_deltas(data_root):
    p6 = _picks({1: [(1, 2, True, False)], 2: [(1, 1, False, False), (2, 1, True, False)]})
    p7 = p6.assign(gw=7)
    storage.write_table(p6, "top10k_picks", 6)
    storage.write_table(p7, "top10k_picks", 7)
    own.run(6)
    out = own.run(7)
    assert len(storage.read_table("ownership", 7)) == len(out)
    assert out["delta_vs_prev_gw"].eq(0).all()
    assert out["overall_selected_pct"].isna().all()  # no players table
    trend = own.ownership_trend()
    assert set(trend["gw"]) == {6, 7}
