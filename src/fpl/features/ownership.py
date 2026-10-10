"""Top-10k ownership, effective ownership and UI helpers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpl import storage
from fpl.schemas import MAX_PER_TEAM, SQUAD_COUNTS, XI_LIMITS

log = logging.getLogger(__name__)


def _n_entries(picks: pd.DataFrame) -> int:
    return int(picks["entry_id"].nunique())


def compute_ownership(picks: pd.DataFrame, gw: int | None = None) -> pd.DataFrame:
    """Per-player ownership from top10k_picks rows (percentages 0-100).

    eo = sum(multiplier)/N*100 (captain 2, TC 3, bench-boost bench 1, bench 0).
    """
    cols = ["gw", "player_id", "owned_pct", "start_pct", "captain_pct", "vice_pct", "eo"]
    n = _n_entries(picks)
    if n == 0:
        return pd.DataFrame(columns=cols)
    g = picks.assign(
        _start=(picks["multiplier"] > 0).astype(float),
        _cap=picks["is_captain"].astype(float),
        _vice=picks["is_vice_captain"].astype(float),
    ).groupby("player_id")
    out = pd.DataFrame(
        {
            "owned_pct": g.size() / n * 100,
            "start_pct": g["_start"].sum() / n * 100,
            "captain_pct": g["_cap"].sum() / n * 100,
            "vice_pct": g["_vice"].sum() / n * 100,
            "eo": g["multiplier"].sum() / n * 100,
        }
    ).reset_index()
    if gw is None:
        gw = int(picks["gw"].iloc[0]) if "gw" in picks else 0
    out.insert(0, "gw", gw)
    return out.sort_values("eo", ascending=False).reset_index(drop=True)


def add_overall(own: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    own = own.copy()
    if players is not None and len(players):
        sel = players.set_index("player_id")["selected_by_percent"]
        own["overall_selected_pct"] = own["player_id"].map(sel).astype(float)
    else:
        own["overall_selected_pct"] = np.nan
    own["delta_vs_overall"] = own["owned_pct"] - own["overall_selected_pct"]
    return own


def add_prev_delta(own: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    own = own.copy()
    if prev is None or prev.empty:
        own["delta_vs_prev_gw"] = np.nan
    else:
        p = prev.set_index("player_id")["owned_pct"]
        own["delta_vs_prev_gw"] = own["owned_pct"] - own["player_id"].map(p).fillna(0.0)
    return own


def run(gw: int | None = None, root=None) -> pd.DataFrame:
    """Build and write the ownership table for picks gw (default: latest top10k_picks)."""
    if gw is None:
        gw = storage.latest_gw("top10k_picks", root)
        if gw is None:
            raise FileNotFoundError("no top10k_picks written; run ingest.top10k first")
    picks = storage.read_table("top10k_picks", gw, root)
    own = compute_ownership(picks, gw)
    own["n_sampled"] = _n_entries(picks)
    players = None
    try:
        players = storage.read_table("players", None, root)
    except FileNotFoundError:
        log.warning("no players table; overall_selected_pct left NaN")
    own = add_overall(own, players)
    prev = None
    if storage.exists("ownership", gw - 1, root):
        prev = storage.read_table("ownership", gw - 1, root)
    elif storage.exists("top10k_picks", gw - 1, root):
        prev = compute_ownership(storage.read_table("top10k_picks", gw - 1, root), gw - 1)
    own = add_prev_delta(own, prev)
    storage.write_table(own, "ownership", gw, root)
    return own


# ---- UI helpers -------------------------------------------------------------------------


def chip_usage(picks: pd.DataFrame) -> pd.DataFrame:
    """Chip -> entries and % of entries (None -> 'none')."""
    e = picks.drop_duplicates("entry_id")
    chip = e["active_chip"].fillna("none").replace("", "none")
    out = chip.value_counts().rename_axis("chip").reset_index(name="entries")
    out["pct"] = out["entries"] / max(len(e), 1) * 100
    return out


def captaincy(picks: pd.DataFrame) -> pd.DataFrame:
    """Captain / vice share per player, ordered by captain share."""
    n = _n_entries(picks)
    cap = picks[picks["is_captain"]].groupby("player_id").size().rename("captains")
    vice = picks[picks["is_vice_captain"]].groupby("player_id").size().rename("vices")
    out = pd.concat([cap, vice], axis=1).fillna(0).astype(int).reset_index()
    out["captain_pct"] = out["captains"] / max(n, 1) * 100
    out["vice_pct"] = out["vices"] / max(n, 1) * 100
    return out.sort_values(["captain_pct", "vice_pct"], ascending=False).reset_index(drop=True)


def template_xi(ownership: pd.DataFrame, players: pd.DataFrame) -> dict:
    """Most-owned legal 15 (2/5/5/3, <=3 per club) and best valid XI.

    ``players`` needs player_id, position, team_id. Returns
    {'squad', 'xi', 'bench' DataFrames, 'formation': 'd-m-f'}. Squad is greedy by owned_pct;
    the XI uses the formation maximising total start_pct.
    """
    df = ownership.merge(players[["player_id", "position", "team_id"]], on="player_id", how="inner")
    df = df.sort_values(["owned_pct", "eo"], ascending=False)
    counts = dict.fromkeys(SQUAD_COUNTS, 0)
    per_team: dict[int, int] = {}
    chosen = []
    for r in df.itertuples():
        if counts[r.position] >= SQUAD_COUNTS[r.position] or per_team.get(r.team_id, 0) >= MAX_PER_TEAM:
            continue
        counts[r.position] += 1
        per_team[r.team_id] = per_team.get(r.team_id, 0) + 1
        chosen.append(r.Index)
    squad = df.loc[chosen]
    by_pos = {
        p: squad[squad["position"] == p].sort_values("start_pct", ascending=False) for p in SQUAD_COUNTS
    }
    best = None
    for d in range(XI_LIMITS["DEF"][0], XI_LIMITS["DEF"][1] + 1):
        for m in range(XI_LIMITS["MID"][0], XI_LIMITS["MID"][1] + 1):
            f = 10 - d - m
            if not XI_LIMITS["FWD"][0] <= f <= XI_LIMITS["FWD"][1]:
                continue
            sizes = [1, d, m, f]
            parts = [by_pos[p].head(n) for p, n in zip(["GKP", "DEF", "MID", "FWD"], sizes, strict=True)]
            if any(len(p) < n for p, n in zip(parts, sizes, strict=True)):
                continue
            xi = pd.concat(parts)
            score = xi["start_pct"].sum()
            if best is None or score > best[0]:
                best = (score, xi, f"{d}-{m}-{f}")
    if best is None:
        return {"squad": squad, "xi": squad.head(0), "bench": squad, "formation": ""}
    xi = best[1]
    bench = squad.drop(xi.index).sort_values("start_pct", ascending=False)
    return {"squad": squad, "xi": xi, "bench": bench, "formation": best[2]}


def my_exposure(ownership: pd.DataFrame, my_team: pd.DataFrame) -> pd.DataFrame:
    """Per player: my multiplier*100 - EO, over my squad plus every sampled player.

    exposure > 0: I gain vs the top 10k if the player scores (differential);
    exposure < 0: I lose ground (short a highly owned player). Sorted ascending (biggest risk first).
    """
    mine = my_team.set_index("player_id")["multiplier"]
    ids = sorted(set(ownership["player_id"]) | set(mine.index))
    out = pd.DataFrame({"player_id": ids})
    out["my_mult"] = out["player_id"].map(mine).fillna(0).astype(int)
    out = out.merge(ownership[["player_id", "eo", "owned_pct"]], on="player_id", how="left")
    out[["eo", "owned_pct"]] = out[["eo", "owned_pct"]].fillna(0.0)
    out["exposure"] = out["my_mult"] * 100 - out["eo"]
    out["owned"] = out["player_id"].isin(mine.index)
    return out.sort_values("exposure").reset_index(drop=True)


def high_eo_not_owned(ownership: pd.DataFrame, my_team: pd.DataFrame, min_eo: float = 20.0) -> pd.DataFrame:
    """Players with EO >= min_eo that are not in my squad, by EO desc."""
    o = ownership[(ownership["eo"] >= min_eo) & ~ownership["player_id"].isin(my_team["player_id"])]
    return o.sort_values("eo", ascending=False).reset_index(drop=True)


def rank_bands(picks: pd.DataFrame, bands: tuple[int, ...] = (100, 1000, 10000)) -> pd.DataFrame:
    """Ownership/EO per cumulative rank band (rank <= band), long format.

    Columns: band, n_entries, player_id, owned_pct, start_pct, captain_pct, vice_pct, eo.
    """
    frames = []
    for b in bands:
        sub = picks[picks["rank"] <= b]
        if sub.empty:
            continue
        o = compute_ownership(sub, 0)
        o.insert(0, "n_entries", _n_entries(sub))
        o.insert(0, "band", b)
        frames.append(o.drop(columns="gw"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ownership_trend(root=None) -> pd.DataFrame:
    """Ownership rows across every GW with a stored ownership table."""
    gws = storage.available_gws("ownership", root)
    if not gws:
        return pd.DataFrame()
    return pd.concat([storage.read_table("ownership", g, root) for g in gws], ignore_index=True)
