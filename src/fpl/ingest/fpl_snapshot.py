"""Per-GW snapshot of FPL state: players, teams, fixtures, events and my team.

Pure transforms (``players_table`` etc.) take API JSON; :func:`run` fetches and writes
``data/raw/gwNN/<table>.parquet`` with NN = the next deadline GW (the GW being planned).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from fpl import gameweeks, storage
from fpl.api import FPLClient
from fpl.config import get_settings
from fpl.schemas import MAX_FREE_TRANSFERS, POSITIONS

# Whether playing a wildcard consumes the week's free transfer. Recorded data for team
# 1608168 (WC in GW4, 1 FT shown for GW7) is only consistent with the WC consuming it;
# Free Hit leaves the bank untouched.
WILDCARD_CONSUMES_FT = True


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---- league-wide tables ---------------------------------------------------------------


def players_table(bootstrap: dict, gw: int) -> pd.DataFrame:
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    rows = []
    for e in bootstrap["elements"]:
        row = dict(e)
        proj = {p["offset"]: p for p in (e.get("price_change_projections") or [])}
        p0 = proj.get(0, {})
        net = (e.get("transfers_in_event") or 0) - (e.get("transfers_out_event") or 0)
        likelihood = p0.get("likelihood")
        if likelihood is None:
            label = "stable"
        else:
            label = {1: "rise", -1: "fall"}.get(likelihood, "stable")
        rows.append(
            {
                **row,
                "gw": gw,
                "player_id": e["id"],
                "team_id": e["team"],
                "team_short": teams.get(e["team"]),
                "position": POSITIONS.get(e["element_type"]),
                "price": e["now_cost"] / 10,
                "cost_change_event": e["cost_change_event"] / 10,
                "cost_change_start": e["cost_change_start"] / 10,
                "net_transfers_event": net,
                "price_change_percent": _f(e.get("price_change_percent")),
                "price_change_hourly_rate": e.get("price_change_hourly_rate"),
                "projected_percent": _f(p0.get("projected_percent")),
                "likelihood": likelihood,
                "price_risk": label,
            }
        )
    df = pd.DataFrame(rows)
    drop = {"price_change_projections", "scout_risks", "overrides"}
    return df.drop(columns=[c for c in drop if c in df.columns])


def teams_table(bootstrap: dict, gw: int) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["teams"]).rename(columns={"id": "team_id"})
    df["gw"] = gw
    return df


def fixtures_table(fixtures: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "fixture_id": f["id"],
                "gw": f.get("event"),
                "kickoff_time": f.get("kickoff_time"),
                "team_h": f["team_h"],
                "team_a": f["team_a"],
                "team_h_score": f.get("team_h_score"),
                "team_a_score": f.get("team_a_score"),
                "team_h_difficulty": f.get("team_h_difficulty"),
                "team_a_difficulty": f.get("team_a_difficulty"),
                "started": bool(f.get("started")),
                "finished": bool(f.get("finished")),
            }
            for f in fixtures
        ]
    )
    return df


def events_table(bootstrap: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gw": e["id"],
                "name": e["name"],
                "deadline_time": e["deadline_time"],
                "finished": e["finished"],
                "is_previous": e["is_previous"],
                "is_current": e["is_current"],
                "is_next": e["is_next"],
                "average_entry_score": e.get("average_entry_score"),
                "highest_score": e.get("highest_score"),
            }
            for e in bootstrap["events"]
        ]
    )


# ---- my team --------------------------------------------------------------------------


def history_table(history: dict) -> pd.DataFrame:
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    rows = [
        {
            "gw": h["event"],
            "points": h["points"],
            "total_points": h["total_points"],
            "rank": h.get("rank"),
            "overall_rank": h.get("overall_rank"),
            "bank": h["bank"] / 10,
            "value": h["value"] / 10,
            "event_transfers": h["event_transfers"],
            "event_transfers_cost": h["event_transfers_cost"],
            "points_on_bench": h["points_on_bench"],
            "chip": chips.get(h["event"]),
        }
        for h in history.get("current", [])
    ]
    return pd.DataFrame(rows)


def transfers_table(transfers: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gw": t["event"],
                "time": t["time"],
                "player_in": t["element_in"],
                "player_in_cost": t["element_in_cost"] / 10,
                "player_out": t["element_out"],
                "player_out_cost": t["element_out_cost"] / 10,
            }
            for t in transfers
        ]
    )


def sell_price(purchase: float, now: float) -> float:
    """FPL selling price: purchase + floor(half the rise) in 0.1 steps; falls pass in full."""
    p, n = round(purchase * 10), round(now * 10)
    if n > p:
        return (p + math.floor((n - p) / 2)) / 10
    return n / 10


def purchase_prices(picks: list[dict], elements: dict[int, dict], transfers: list[dict]) -> dict:
    """Purchase price (£m) per player id: latest transfer-in cost, else the start price."""
    latest: dict[int, tuple[str, int]] = {}
    for t in transfers:
        cur = latest.get(t["element_in"])
        if cur is None or t["time"] > cur[0]:
            latest[t["element_in"]] = (t["time"], t["element_in_cost"])
    out = {}
    for p in picks:
        pid = p["element"]
        if pid in latest:
            out[pid] = latest[pid][1] / 10
        else:
            e = elements[pid]
            out[pid] = (e["now_cost"] - e["cost_change_start"]) / 10
    return out


def apply_pending(picks: dict, transfers: list[dict], gw: int) -> tuple[dict, float, int]:
    """Apply transfers already made for ``gw`` (after the previous deadline) to the picks.

    Returns (picks, bank delta in £m, number of pending transfers).
    """
    pending = sorted((t for t in transfers if t["event"] == gw), key=lambda t: t["time"])
    if not pending:
        return picks, 0.0, 0
    plist = [dict(p) for p in picks["picks"]]
    delta = 0
    for t in pending:
        for p in plist:
            if p["element"] == t["element_out"]:
                p["element"] = t["element_in"]
                break
        delta += t["element_out_cost"] - t["element_in_cost"]
    return {**picks, "picks": plist}, delta / 10, len(pending)


def my_team_table(
    picks: dict, bootstrap: dict, transfers: list[dict], entry_id: int, gw: int
) -> pd.DataFrame:
    elements = {e["id"]: e for e in bootstrap["elements"]}
    pp = purchase_prices(picks["picks"], elements, transfers)
    rows = []
    for p in picks["picks"]:
        now = elements[p["element"]]["now_cost"] / 10
        rows.append(
            {
                "gw": gw,
                "entry_id": entry_id,
                "player_id": p["element"],
                "slot": p["position"],
                "multiplier": p["multiplier"],
                "is_captain": p["is_captain"],
                "is_vice_captain": p["is_vice_captain"],
                "price": now,
                "purchase_price": pp[p["element"]],
                "sell_price": sell_price(pp[p["element"]], now),
            }
        )
    return pd.DataFrame(rows)


def compute_free_transfers(history: dict, upto_gw: int) -> int:
    """Free transfers available for the GW after ``upto_gw``.

    GW1 is unlimited; from GW2 you get 1 per GW, unused ones roll over to a max of 5.
    Transfers beyond the FTs cost hits and leave 0 banked. A Free Hit keeps the bank
    unchanged; a Wildcard consumes the week's FT (see ``WILDCARD_CONSUMES_FT``).
    """
    used_by_gw = {h["event"]: h["event_transfers"] for h in history.get("current", [])}
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    ft = 1  # available going into GW2
    for g in range(2, upto_gw + 1):
        chip = chips.get(g)
        used = used_by_gw.get(g, 0)
        if chip == "freehit":
            remaining = ft
        elif chip == "wildcard" and WILDCARD_CONSUMES_FT:
            remaining = max(ft - max(used, 1), 0)
        elif chip == "wildcard":
            remaining = ft
        else:
            remaining = max(ft - used, 0)
        ft = min(MAX_FREE_TRANSFERS, remaining + 1)
    return ft


def chips_available(bootstrap: dict, history: dict, gw: int) -> list[str]:
    used = history.get("chips", [])
    out = []
    for c in bootstrap.get("chips", []):
        if not (c["start_event"] <= gw <= c["stop_event"]):
            continue
        n_used = sum(
            1 for u in used if u["name"] == c["name"] and c["start_event"] <= u["event"] <= c["stop_event"]
        )
        if n_used < c.get("number", 1):
            out.append(c["name"])
    return out


def my_entry_table(
    entry: dict,
    history: dict,
    picks: dict,
    bootstrap: dict,
    gw: int,
    picks_gw: int,
    bank_delta: float = 0.0,
    pending: int = 0,
) -> pd.DataFrame:
    eh = picks["entry_history"]
    cur = history.get("current", [])
    last = cur[-1] if cur else {}
    return pd.DataFrame(
        [
            {
                "gw": gw,
                "entry_id": entry["id"],
                "entry_name": entry["name"],
                "bank": round(eh["bank"] / 10 + bank_delta, 1),
                "value": eh["value"] / 10,
                "free_transfers": max(compute_free_transfers(history, picks_gw) - pending, 0),
                "total_points": eh.get("total_points", last.get("total_points")),
                "overall_rank": entry.get("summary_overall_rank") or eh.get("overall_rank"),
                "chips_available": ",".join(chips_available(bootstrap, history, gw)),
                "picks_gw": picks_gw,
            }
        ]
    )


# ---- orchestration --------------------------------------------------------------------


def run(client: FPLClient, root: Path | None = None) -> int:
    """Pull and write the snapshot. Returns the snapshot GW (next deadline)."""
    team_id = get_settings().team_id
    bs = client.bootstrap_static()
    gw = gameweeks.next_gw(bs) or gameweeks.current_gw(bs)
    picks_gw = gameweeks.current_gw(bs) or gameweeks.last_deadline_gw(bs)
    fixtures = client.fixtures()
    entry = client.entry(team_id)
    history = client.entry_history(team_id)
    transfers = client.entry_transfers(team_id)
    picks = client.entry_picks(team_id, picks_gw, final=True)

    picks, bank_delta, pending = apply_pending(picks, transfers, gw)
    w = lambda df, t: storage.write_table(df, t, gw=gw, root=root)  # noqa: E731
    w(players_table(bs, gw), "players")
    w(teams_table(bs, gw), "teams")
    w(fixtures_table(fixtures), "fixtures")
    w(events_table(bs), "events")
    w(my_entry_table(entry, history, picks, bs, gw, picks_gw, bank_delta, pending), "my_entry")
    w(my_team_table(picks, bs, transfers, team_id, gw), "my_team")
    w(history_table(history), "my_history")
    w(transfers_table(transfers), "my_transfers")
    return gw
