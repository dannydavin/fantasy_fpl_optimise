"""FPL-Optimization-Tools wrapper: offline export/parse tests; real run is opt-in."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from fpl import storage
from fpl.optimise import fplopt_wrapper as w


def test_projection_csv_and_team_json(ui_data):
    players = storage.read_table("players")
    xp = storage.read_table("xp")
    team = storage.read_table("my_team")
    gw = int(team["gw"].iloc[0])
    df = w.build_projection_csv(xp, players, [gw, gw + 1], "baseline")
    assert {"ID", "Name", "Pos", "Value", f"{gw}_Pts", f"{gw}_xMins", f"{gw + 1}_Pts"} <= set(
        df.columns)
    assert set(df["Pos"]) <= {"G", "D", "M", "F"} and df["ID"].is_unique
    top = xp[(xp["gw"] == gw) & (xp["model"] == "baseline")].sort_values("xp").iloc[-1]
    assert df.set_index("ID").at[top["player_id"], f"{gw}_Pts"] == pytest.approx(top["xp"], abs=1e-3)
    tj = w.build_team_json(team, players, 0.4, 1)
    assert len(tj["picks"]) == 15 and tj["transfers"] == {
        "bank": 4, "limit": 1, "made": 0, "cost": 4, "value": 0}
    sp = team.set_index("player_id")["sell_price"]
    assert all(p["selling_price"] == round(sp[p["element"]] * 10) for p in tj["picks"])


def test_settings_chips():
    s = w.build_settings(7, 3, {8: "bboost", 9: "freehit"}, [1], [2])
    assert s["use_bb"] == [8] and s["use_fh"] == [9] and s["locked"] == [1] and s["banned"] == [2]
    assert s["horizon"] == 3 and s["datasource"] == "ourxp"
    assert "run/solve.py" in w.command()[-1] and "3.14" in w.command()


def test_export_inputs(ui_data, tmp_path):
    if w.find_tool_dir() is None:
        pytest.skip("external/FPL-Optimization-Tools not checked out")
    players, xp, team = (storage.read_table(t) for t in ("players", "xp", "my_team"))
    gw = int(team["gw"].iloc[0])
    w.export_inputs(tmp_path, xp, players, team, 0.4, 1, gw, 2, chips={gw + 1: "bboost"})
    assert (tmp_path / "run" / "solve.py").exists() and (tmp_path / "data" / "ourxp.csv").exists()
    st = json.loads((tmp_path / "data" / "user_settings.json").read_text())
    assert st["use_bb"] == [gw + 1] and st["override_next_gw"] == gw


def test_parse_results():
    def row(i, week, pos, typ, squad, lineup, bench, cap=0, ti=0, to=0, sell=0.0, buy=0.0, chip=""):
        return {"id": i, "week": week, "name": f"p{i}", "pos": pos, "type": typ, "buy_price": buy,
                "sell_price": sell, "xP": 5.0, "squad": squad, "lineup": lineup, "bench": bench,
                "captain": cap, "vicecaptain": 0, "transfer_in": ti, "transfer_out": to,
                "multiplier": 2 if cap else (1 if lineup else 0), "chip": chip, "ft": 1.0}

    rows = [row(1, 7, "GKP", 1, 1, 1, -1, cap=1), row(2, 7, "GKP", 1, 1, 0, 0),
            row(3, 7, "MID", 3, 1, 1, -1, ti=1, buy=6.0), row(4, 7, "MID", 3, 0, 0, -1,
                                                              to=1, sell=5.5)]
    plan = w.parse_results(pd.DataFrame(rows), bank=1.0, free_transfers=1)
    assert plan.transfers.iloc[0].to_dict() == {
        "gw": 7, "player_out": 4, "player_in": 3, "out_price": 5.5, "in_price": 6.0}
    s = plan.summary.iloc[0]
    assert s["transfers"] == 1 and s["hits"] == 0 and s["bank"] == pytest.approx(0.5)
    assert plan.lineups["is_captain"].sum() == 1
    assert sorted(plan.lineups["slot"]) == [1, 2, 12]


@pytest.mark.skipif(os.environ.get("FPLOPT_RUN") != "1", reason="set FPLOPT_RUN=1 (needs network)")
def test_real_run(ui_data):
    ok, why = w.available()
    if not ok:
        pytest.skip(why)
    players, xp, team = (storage.read_table(t) for t in ("players", "xp", "my_team"))
    plan = w.cross_check(xp, players, team, 0.4, 1, int(team["gw"].iloc[0]), 1, secs=60)
    assert len(plan.lineups) == 15
