"""Fixture ticker: official FDR plus a custom difficulty from rolling team xG.

Also blank/double-GW detection from the fixture list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl import storage
from fpl.api import FPLClient
from fpl.features.form import live_player_gw, load_live

XG_WINDOW = 6  # matches
SHRINK = 3.0  # pseudo-matches pulling team xG toward the league mean
HOME_ADJ = 0.3


def _long(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: team_id, opponent_id, is_home, fdr, plus fixture cols."""
    f = fixtures[fixtures["gw"].notna()]
    home = pd.DataFrame(
        {"fixture_id": f["fixture_id"], "gw": f["gw"], "team_id": f["team_h"],
         "opponent_id": f["team_a"], "is_home": True, "fdr": f["team_h_difficulty"],
         "finished": f["finished"].astype(bool)}
    )
    away = pd.DataFrame(
        {"fixture_id": f["fixture_id"], "gw": f["gw"], "team_id": f["team_a"],
         "opponent_id": f["team_h"], "is_home": False, "fdr": f["team_a_difficulty"],
         "finished": f["finished"].astype(bool)}
    )
    return pd.concat([home, away], ignore_index=True)


def gw_fixture_counts(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Number of fixtures per (team_id, gw); teams with none in a GW are absent."""
    long = _long(fixtures)
    return long.groupby(["team_id", "gw"]).size().rename("n_fixtures").reset_index()


def blank_double_calendar(
    fixtures: pd.DataFrame,
    gws: list[int] | None = None,
    teams: list[int] | None = None,
) -> pd.DataFrame:
    """Full team x GW grid with n_fixtures and kind in {blank, single, double}.

    Defaults: teams from the fixture list, gws from the min to max scheduled GW.
    ``double`` means 2 or more fixtures.
    """
    counts = gw_fixture_counts(fixtures)
    teams = teams if teams is not None else sorted(
        set(fixtures["team_h"]).union(fixtures["team_a"])
    )
    if gws is None:
        scheduled = fixtures["gw"].dropna().astype(int)
        gws = list(range(int(scheduled.min()), int(scheduled.max()) + 1)) if len(scheduled) else []
    grid = pd.MultiIndex.from_product([teams, gws], names=["team_id", "gw"]).to_frame(index=False)
    out = grid.merge(counts, on=["team_id", "gw"], how="left")
    out["n_fixtures"] = out["n_fixtures"].fillna(0).astype(int)
    out["kind"] = out["n_fixtures"].map(lambda n: "blank" if n == 0 else "single" if n == 1 else "double")
    return out


def team_match_xg(
    pgw_with_team: pd.DataFrame, fixtures: pd.DataFrame
) -> pd.DataFrame:
    """Team xG for/against per finished fixture.

    ``pgw_with_team`` has gw, team_id, expected_goals per player-GW. A team's GW xG is
    split evenly across its finished fixtures that GW (live data is GW-aggregated).
    Returns fixture_id, gw, team_id, opponent_id, xgf, xga.
    """
    long = _long(fixtures)
    long = long[long["finished"] & long["gw"].isin(pgw_with_team["gw"].unique())]
    tg = pgw_with_team.groupby(["team_id", "gw"])["expected_goals"].sum().rename("gw_xg")
    n = long.groupby(["team_id", "gw"]).size().rename("n")
    long = long.join(tg, on=["team_id", "gw"]).join(n, on=["team_id", "gw"])
    long["xgf"] = long["gw_xg"].fillna(0.0) / long["n"]
    opp = long[["fixture_id", "team_id", "xgf"]].rename(
        columns={"team_id": "opponent_id", "xgf": "xga"}
    )
    out = long.merge(opp, on=["fixture_id", "opponent_id"], how="left")
    return out[["fixture_id", "gw", "team_id", "opponent_id", "xgf", "xga"]]


def team_ratings(match_xg: pd.DataFrame, window: int = XG_WINDOW) -> pd.DataFrame:
    """Shrunk rolling attack/defence per team and a standardised ``strength`` z-score.

    strength = attack_rel - concede_rel (higher = tougher opponent). Index team_id.
    """
    if match_xg.empty:
        return pd.DataFrame(columns=["att", "conc", "strength"])
    m = match_xg.sort_values(["team_id", "gw", "fixture_id"]).groupby("team_id").tail(window)
    mu = m["xgf"].mean()
    g = m.groupby("team_id").agg(n=("xgf", "size"), f=("xgf", "sum"), a=("xga", "sum"))
    g["att"] = (g["f"] + SHRINK * mu) / (g["n"] + SHRINK)
    g["conc"] = (g["a"] + SHRINK * mu) / (g["n"] + SHRINK)
    raw = (g["att"] - mu) / mu - (g["conc"] - mu) / mu
    sd = raw.std(ddof=0)
    g["strength"] = (raw - raw.mean()) / sd if sd and sd > 0 else 0.0
    return g[["att", "conc", "strength"]]


def ticker_table(
    fixtures: pd.DataFrame, from_gw: int, ratings: pd.DataFrame | None = None
) -> pd.DataFrame:
    """fixture_ticker rows for fixtures with gw >= from_gw (DGW = 2 rows, blank = none)."""
    long = _long(fixtures)
    long = long[long["gw"] >= from_gw].copy()
    if ratings is not None and len(ratings):
        z = long["opponent_id"].map(ratings["strength"])
        adj = long["is_home"].map({True: -HOME_ADJ, False: HOME_ADJ})  # opponent venue
        long["difficulty"] = (3 + 0.9 * z + adj).clip(1, 5)
        long["difficulty"] = long["difficulty"].fillna(long["fdr"].astype(float))
    else:
        long["difficulty"] = long["fdr"].astype(float)
    cols = ["team_id", "gw", "opponent_id", "is_home", "fdr", "difficulty", "fixture_id"]
    return long.sort_values(["team_id", "gw", "fixture_id"])[cols].reset_index(drop=True)


def run(gw: int | None = None, root: Path | None = None, client: FPLClient | None = None):
    """Compute and write ``fixture_ticker`` for snapshot ``gw`` (default: latest)."""
    gw = gw if gw is not None else storage.latest_gw("fixtures", root)
    if gw is None:
        raise FileNotFoundError("no fixtures snapshot; run fpl_snapshot first")
    fixtures = storage.read_table("fixtures", gw, root)
    players = storage.read_table("players", gw, root)
    events = storage.read_table("events", gw, root)
    finished = sorted(int(g) for g in events.loc[events["finished"], "gw"] if g < gw)
    own = client is None
    client = client or FPLClient()
    try:
        live = load_live(client, finished[-XG_WINDOW:])
    finally:
        if own:
            client.close()
    pgw = live_player_gw(live).merge(players[["player_id", "team_id"]], on="player_id")
    ratings = team_ratings(team_match_xg(pgw, fixtures)) if len(pgw) else None
    df = ticker_table(fixtures, gw, ratings)
    storage.write_table(df, "fixture_ticker", gw=gw, root=root)
    return df
