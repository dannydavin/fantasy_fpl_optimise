"""Rolling player form from ``event/{gw}/live`` of finished gameweeks.

Window ``w`` covers the last ``w`` finished GWs up to the snapshot. ``start_pct`` and
``minutes_share`` are relative to the team's finished matches in the window (DGW aware).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl import storage
from fpl.api import FPLAPIError, FPLClient

WINDOWS = (4, 6)
STAT_COLS = ["minutes", "starts", "expected_goals", "expected_assists", "defensive_contribution",
             "total_points"]
SETPIECE_COLS = ["penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order"]


def load_live(client: FPLClient, gws: list[int]) -> dict[int, dict]:
    """event_live for finished GWs (cached forever); GWs that fail to load are skipped."""
    out = {}
    for g in gws:
        try:
            out[g] = client.event_live(g, final=True)
        except FPLAPIError:
            continue
    return out


def live_player_gw(live_by_gw: dict[int, dict]) -> pd.DataFrame:
    """One row per (gw, player_id) with the stats needed for form/xG."""
    rows = []
    for gw, live in live_by_gw.items():
        for el in live.get("elements", []):
            s = el["stats"]
            rows.append(
                {
                    "gw": gw,
                    "player_id": el["id"],
                    **{c: float(s.get(c) or 0) for c in STAT_COLS},
                }
            )
    cols = ["gw", "player_id", *STAT_COLS]
    return pd.DataFrame(rows, columns=cols)


def team_matches(fixtures: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    """Finished matches per (team_id, gw): columns team_id, gw, n_matches."""
    f = fixtures[fixtures["finished"].astype(bool) & fixtures["gw"].isin(gws)]
    long = pd.concat(
        [f[["gw", "team_h"]].rename(columns={"team_h": "team_id"}),
         f[["gw", "team_a"]].rename(columns={"team_a": "team_id"})]
    )
    return long.groupby(["team_id", "gw"]).size().rename("n_matches").reset_index()


def form_table(
    pgw: pd.DataFrame,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    windows: tuple[int, ...] = WINDOWS,
) -> pd.DataFrame:
    """Rolling form per player and window. ``pgw`` from :func:`live_player_gw`."""
    gws_avail = sorted(pgw["gw"].unique()) if len(pgw) else []
    extras = ["team_id", *[c for c in SETPIECE_COLS if c in players.columns]]
    base = players[["player_id", *extras]]
    frames = []
    for w in windows:
        gws = gws_avail[-w:]
        sub = pgw[pgw["gw"].isin(gws)]
        agg = sub.groupby("player_id")[STAT_COLS].sum().reset_index()
        agg = base.merge(agg, on="player_id", how="left")
        agg[STAT_COLS] = agg[STAT_COLS].fillna(0.0)
        tm = team_matches(fixtures, gws).groupby("team_id")["n_matches"].sum()
        n = agg["team_id"].map(tm).fillna(0)
        mins = agg["minutes"]
        safe_n = n.where(n > 0)
        safe_mins = mins.where(mins > 0)

        def per90(col: str, agg=agg, safe_mins=safe_mins) -> pd.Series:
            return (agg[col] / safe_mins * 90).fillna(0.0)

        out = pd.DataFrame(
            {
                "player_id": agg["player_id"],
                "window": w,
                "minutes": mins.astype(int),
                "start_pct": (agg["starts"] / safe_n * 100).fillna(0.0).clip(upper=100),
                "minutes_share": (mins / (90 * safe_n)).fillna(0.0).clip(upper=1),
                "xg_per90": per90("expected_goals"),
                "xa_per90": per90("expected_assists"),
                "xgi_per90": per90("expected_goals") + per90("expected_assists"),
                "dc_per90": per90("defensive_contribution"),
                "points_per90": per90("total_points"),
                "gws_in_window": len(gws),
            }
        )
        for c in extras:
            out[c] = agg[c]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def run(gw: int | None = None, root: Path | None = None, client: FPLClient | None = None):
    """Compute and write ``form`` for snapshot ``gw`` (default: latest snapshot)."""
    gw = gw if gw is not None else storage.latest_gw("players", root)
    if gw is None:
        raise FileNotFoundError("no players snapshot; run fpl_snapshot first")
    players = storage.read_table("players", gw, root)
    fixtures = storage.read_table("fixtures", gw, root)
    events = storage.read_table("events", gw, root)
    finished = sorted(int(g) for g in events.loc[events["finished"], "gw"] if g < gw)
    own = client is None
    client = client or FPLClient()
    try:
        live = load_live(client, finished[-max(WINDOWS):])
    finally:
        if own:
            client.close()
    df = form_table(live_player_gw(live), players, fixtures)
    storage.write_table(df, "form", gw=gw, root=root)
    return df
