from __future__ import annotations

import pandas as pd
import pytest

from fpl import storage
from fpl.features import form
from fpl.ingest import fpl_snapshot as snap
from tests.conftest import NEXT_GW


def _live(gw, rows):
    return {
        gw: {
            "elements": [
                {"id": pid, "stats": {"minutes": m, "starts": s, "expected_goals": xg,
                                      "expected_assists": xa, "defensive_contribution": dc,
                                      "total_points": pts}}
                for pid, m, s, xg, xa, dc, pts in rows
            ]
        }
    }


def test_form_table_maths():
    live = {}
    live.update(_live(1, [(1, 90, 1, 0.5, 0.1, 4, 6), (2, 0, 0, 0, 0, 0, 0)]))
    live.update(_live(2, [(1, 45, 0, 0.3, 0.1, 2, 2), (2, 90, 1, 0.0, 0.0, 6, 3)]))
    pgw = form.live_player_gw(live)
    players = pd.DataFrame({"player_id": [1, 2], "team_id": [10, 10], "penalties_order": [1, None]})
    fixtures = pd.DataFrame(
        {"gw": [1, 2], "team_h": [10, 10], "team_a": [11, 12], "finished": [True, True]}
    )
    out = form.form_table(pgw, players, fixtures, windows=(2, 1))
    a = out[(out.player_id == 1) & (out.window == 2)].iloc[0]
    assert a["minutes"] == 135
    assert a["start_pct"] == pytest.approx(50.0)
    assert a["minutes_share"] == pytest.approx(135 / 180)
    assert a["xg_per90"] == pytest.approx(0.8 / 135 * 90)
    assert a["xgi_per90"] == pytest.approx(1.0 / 135 * 90)
    assert a["dc_per90"] == pytest.approx(6 / 135 * 90)
    assert a["points_per90"] == pytest.approx(8 / 135 * 90)
    assert a["penalties_order"] == 1
    # window 1 only sees the latest GW
    b = out[(out.player_id == 1) & (out.window == 1)].iloc[0]
    assert b["minutes"] == 45 and b["start_pct"] == 0
    # player with no minutes in a window gets zeros, not NaN
    c = out[(out.player_id == 2) & (out.window == 2)].iloc[0]
    assert c["minutes"] == 90 and c["points_per90"] == pytest.approx(3.0)


def test_run_form_on_recorded_data(client, data_root):
    snap.run(client)
    df = form.run(client=client)
    assert storage.exists("form", NEXT_GW)
    assert set(df["window"]) == {4, 6}
    players = storage.read_table("players", NEXT_GW)
    assert df.groupby("window")["player_id"].nunique().eq(len(players)).all()
    assert df["start_pct"].between(0, 100).all()
    assert df["minutes_share"].between(0, 1).all()
    assert not df[["xgi_per90", "dc_per90", "points_per90"]].isna().any().any()
    # Haaland (id 411) should have real minutes and non-trivial xGI
    h = df[(df.player_id == 411) & (df.window == 4)].iloc[0]
    assert h["minutes"] > 0 and h["xgi_per90"] > 0
