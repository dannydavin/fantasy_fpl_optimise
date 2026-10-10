from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl import schemas, storage
from fpl.ingest import history as hist
from fpl.models import xp_baseline as xp
from fpl.schemas import POSITIONS
from tests.conftest import load_fixture

HIST = Path(__file__).parent / "fixtures" / "history"
NEXT = 7


def build_inputs(bootstrap=None, fixtures_json=None):
    """players/teams/fixtures frames in the snapshot schema, from recorded JSON."""
    b = bootstrap or load_fixture("bootstrap-static")
    fj = fixtures_json or load_fixture("fixtures")
    e = pd.DataFrame(b["elements"])
    players = pd.DataFrame({
        "player_id": e["id"], "code": e["code"], "web_name": e["web_name"],
        "team_id": e["team"], "element_type": e["element_type"],
        "position": e["element_type"].map(POSITIONS), "status": e["status"],
        "chance_of_playing_next_round": pd.to_numeric(e["chance_of_playing_next_round"]),
        "minutes": e["minutes"], "starts": e["starts"],
        "expected_goals": pd.to_numeric(e["expected_goals"]),
        "expected_assists": pd.to_numeric(e["expected_assists"]),
        "defensive_contribution": e["defensive_contribution"], "bonus": e["bonus"],
        "saves": e["saves"],
    })
    teams = pd.DataFrame(b["teams"]).rename(columns={"id": "team_id"})[
        ["team_id", "name", "short_name"]]
    f = pd.DataFrame(fj)
    fixtures = pd.DataFrame({
        "fixture_id": f["id"], "gw": f["event"], "team_h": f["team_h"], "team_a": f["team_a"],
        "team_h_score": f["team_h_score"], "team_a_score": f["team_a_score"],
        "finished": f["finished"],
    })
    return (_full(players, "players"), _full(teams, "teams"),
            _full(fixtures, "fixtures"))


def _full(df, table):
    """Fill the remaining schema columns with nulls so the frame satisfies the contract."""
    out = df.copy()
    for c in schemas.get_schema(table).columns:
        if c not in out:
            out[c] = np.nan
    return schemas.conform(out, table)


@pytest.fixture(scope="module")
def inputs():
    return build_inputs()


@pytest.fixture(scope="module")
def history():
    m = pd.read_csv(HIST / "merged_gw_2025-26.csv")
    return hist.normalise_merged_gw(
        m, "2025-26", pd.read_csv(HIST / "players_raw_2025-26.csv"),
        pd.read_csv(HIST / "teams_2025-26.csv"))


def test_poisson_helpers():
    assert xp.poisson_sf(np.array([2.0]), 0)[0] == pytest.approx(1.0)
    assert xp.poisson_sf(np.array([2.0]), 1)[0] == pytest.approx(1 - np.exp(-2))
    assert xp.expected_floor_half(np.array([0.0]))[0] == 0.0
    assert 0.3 < xp.expected_floor_half(np.array([1.3]))[0] < 0.5


def test_contract_and_ordering(inputs, data_root):
    players, teams, fixtures = inputs
    gws = list(range(NEXT, NEXT + 6))
    out = xp.project(players, teams, fixtures, gws)
    storage.conform(out, "xp")
    assert set(out["gw"]) == set(gws)
    assert (out["model"] == "baseline").all()
    assert out["xp"].notna().all() and (out["xp"] >= -2).all()
    assert out[["gw", "player_id"]].duplicated().sum() == 0
    g7 = out[out["gw"] == NEXT].merge(players[["player_id", "web_name", "position"]])
    haaland = g7[g7["web_name"] == "Haaland"].iloc[0]
    assert haaland["xp"] > 5
    gks = g7[(g7["position"] == "GKP") & (g7["xmins"] < 5)]
    assert haaland["xp"] > gks["xp"].max() + 3
    assert g7.nlargest(1, "xp")["web_name"].iloc[0] == "Haaland"
    assert g7["xp"].max() < 15


def test_write_and_run(inputs, data_root):
    players, teams, fixtures = inputs
    storage.write_table(players, "players", NEXT)
    storage.write_table(teams, "teams", NEXT)
    storage.write_table(fixtures, "fixtures", NEXT)
    out = xp.run(horizon=3)
    assert sorted(out["gw"].unique()) == [7, 8, 9]
    back = storage.read_table("xp", NEXT)
    assert len(back) == len(out)
    assert storage.table_path("xp", NEXT).parent.name == "gw07"
    assert xp.run(gw=7, horizon=1)["gw"].unique().tolist() == [7]


def test_dgw_sums_and_blank_zero(inputs):
    players, teams, fixtures = inputs
    players = players[players["web_name"].isin(["Haaland", "Salah"])].reset_index(drop=True)
    haaland = players[players["web_name"] == "Haaland"].iloc[0]
    tid = int(haaland["team_id"])
    gw = NEXT
    base_fx = fixtures[(fixtures["gw"] == gw) & ((fixtures["team_h"] == tid)
                                                 | (fixtures["team_a"] == tid))]
    assert len(base_fx) == 1
    single = xp.project(players, teams, fixtures, [gw])
    s = single[single["player_id"] == haaland["player_id"]].iloc[0]
    assert s["n_fixtures"] == 1 and s["xp"] > 3

    # DGW: add a second home fixture vs the same opponent in the same GW
    extra = base_fx.copy()
    extra["fixture_id"] = 99999
    dgw = pd.concat([fixtures, extra], ignore_index=True)
    d = xp.project(players, teams, dgw, [gw])
    dd = d[d["player_id"] == haaland["player_id"]].iloc[0]
    assert dd["n_fixtures"] == 2
    assert dd["xp"] == pytest.approx(2 * s["xp"], rel=1e-9)
    assert dd["xmins"] == pytest.approx(2 * s["xmins"])
    assert dd["p_start"] == pytest.approx(s["p_start"])

    # blank: remove the team's fixture
    blank = fixtures.drop(base_fx.index)
    b = xp.project(players, teams, blank, [gw])
    bb = b[b["player_id"] == haaland["player_id"]].iloc[0]
    assert bb["xp"] == 0 and bb["n_fixtures"] == 0 and bb["opponent"] == ""


def test_injured_zero_and_doubtful_scaled(inputs):
    players, teams, fixtures = inputs
    p = players[players["web_name"] == "Haaland"].copy()
    inj = p.assign(status="i", chance_of_playing_next_round=0.0, player_id=-1)
    dbt = p.assign(status="d", chance_of_playing_next_round=50.0, player_id=-2)
    allp = pd.concat([p, inj, dbt], ignore_index=True)
    out = xp.project(allp, teams, fixtures, [NEXT, NEXT + 1])
    g = out[out["gw"] == NEXT].set_index("player_id")
    assert g.loc[-1, "xp"] == 0
    assert g.loc[-1, "p_start"] == 0
    ratio = g.loc[-2, "xp"] / g.loc[p["player_id"].iloc[0], "xp"]
    assert 0.4 < ratio < 0.6
    # later GW: the injured player drifts back, but stays below the fit one
    g8 = out[out["gw"] == NEXT + 1].set_index("player_id")
    assert 0 < g8.loc[-1, "xp"] < g8.loc[p["player_id"].iloc[0], "xp"]


def test_history_prior_shrinks_rate(inputs, history):
    players, teams, fixtures = inputs
    p = players[players["web_name"] == "Haaland"].copy()
    # no in-season data -> prior only
    fresh = p.assign(minutes=0, starts=0, expected_goals=0.0, expected_assists=0.0,
                     defensive_contribution=0, bonus=0)
    pos_prior, _ = xp.build_priors(None)
    none = xp.estimate_rates(fresh.assign(team_games=5.0), pos_prior)
    assert none["xg90"].iloc[0] == pytest.approx(pos_prior.loc["FWD", "xg90"])
    h = history.copy()
    hp = h[h["code"].notna()]
    code = hp.groupby("code")["minutes"].sum().idxmax()
    pri_pos, pri_pl = xp.build_priors(h)
    assert code in pri_pl.index
    f2 = fresh.assign(code=code, position=hp[hp["code"] == code]["position"].iloc[0],
                      team_games=5.0)
    withp = xp.estimate_rates(f2, pri_pos, pri_pl)
    assert withp["xg90"].iloc[0] >= 0


def test_scoring_override_changes_points(inputs):
    players, teams, fixtures = inputs
    p = players[players["web_name"] == "Haaland"]
    a = xp.project(p, teams, fixtures, [NEXT])["xp"].iloc[0]
    b = xp.project(p, teams, fixtures, [NEXT], scoring={"goals_scored": {"FWD": 8}})["xp"].iloc[0]
    assert b > a


def test_team_ratings_sane(inputs):
    players, teams, fixtures = inputs
    r = xp.team_ratings(fixtures, teams)
    assert len(r) == 20
    assert r["att"].mean() == pytest.approx(1.0)
    assert r["att"].between(0.6, 1.5).all() and r["def"].between(0.6, 1.5).all()
    assert 1.0 < r.attrs["mu"] < 2.0
