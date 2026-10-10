"""Solver tests: synthetic pools for exact rule checks, ui_data for the real squad."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl import storage
from fpl.optimise.solver import (
    Plan,
    simulate_free_transfers,
    solve,
    solve_from_storage,
    validate_plan,
)
from fpl.schemas import MAX_PER_TEAM, SQUAD_COUNTS

KINDS = ["GKP", "DEF", "DEF", "MID", "MID", "FWD"]
OPTS = {"time_limit": 30, "mip_gap": 1e-6, "candidate_limit": 100}


def make_players(n_teams: int = 8, price: float = 5.0) -> pd.DataFrame:
    rows = []
    for t in range(1, n_teams + 1):
        for k, pos in enumerate(KINDS, 1):
            rows.append({"player_id": t * 10 + k, "web_name": f"P{t}{k}", "team_id": t,
                         "position": pos, "price": price, "status": "a"})
    return pd.DataFrame(rows)


def initial_squad(players: pd.DataFrame) -> pd.DataFrame:
    """Greedy legal 15 using the highest ids (so low-id 'stars' stay outsiders)."""
    counts, per_team, ids = dict.fromkeys(SQUAD_COUNTS, 0), {}, []
    for r in players.sort_values("player_id", ascending=False).itertuples():
        if counts[r.position] < SQUAD_COUNTS[r.position] and per_team.get(r.team_id, 0) < 3:
            counts[r.position] += 1
            per_team[r.team_id] = per_team.get(r.team_id, 0) + 1
            ids.append(r.player_id)
    return pd.DataFrame({"player_id": ids, "sell_price": 5.0})


def make_xp(players, gws, fn=None, default=2.0) -> pd.DataFrame:
    rows = []
    for g in gws:
        for p in players["player_id"]:
            v = fn(p, g) if fn else None
            rows.append({"gw": g, "player_id": p, "xp": default if v is None else v})
    return pd.DataFrame(rows)


@pytest.fixture
def pool():
    pl = make_players()
    return pl, initial_squad(pl)


def run(pool, xp_fn=None, horizon=1, bank=0.0, ft=1, **kw) -> Plan:
    pl, sq = pool
    xp = make_xp(pl, range(1, 7), xp_fn)
    plan = solve(xp, pl, sq, bank, ft, 1, horizon, **{**OPTS, **kw})
    assert plan.ok, plan.status
    assert validate_plan(plan, pl, sq, bank, ft) == []
    return plan


def squad_at(plan, gw):
    return set(plan.lineups[plan.lineups["gw"] == gw]["player_id"])


def test_wildcard_gives_legal_squad_from_scratch(pool):
    plan = run(pool, lambda p, g: p % 7 + 1, chips={1: "wildcard"})
    assert len(plan.transfers) > 1 and plan.summary["hits"].sum() == 0
    assert plan.summary["chip"].iloc[0] == "wildcard"


def test_max_three_per_club(pool):
    plan = run(pool, lambda p, g: 20 if p // 10 == 8 else 1, chips={1: "wildcard"})
    lg = plan.lineups
    assert (lg["team_id"] == 8).sum() == MAX_PER_TEAM
    assert lg["team_id"].value_counts().max() <= MAX_PER_TEAM


def test_budget_binding():
    pl = make_players()
    pl.loc[pl["player_id"] == 16, "price"] = 9.0  # star FWD costs 9, owned FWD sells for 5
    sq = initial_squad(pl)

    def fn(p, g):
        return 30 if p == 16 else 2

    broke = solve(make_xp(pl, [1], fn), pl, sq, 0.0, 1, 1, **OPTS)
    assert 16 not in squad_at(broke, 1)
    rich = solve(make_xp(pl, [1], fn), pl, sq, 4.0, 1, 1, **OPTS)
    assert 16 in squad_at(rich, 1)
    assert rich.summary["bank"].iloc[0] == pytest.approx(0.0)


def test_sell_price_used_for_originals():
    pl = make_players()
    sq = initial_squad(pl)
    sq["sell_price"] = 4.0  # sell at 4, buying a 5.0 player needs 1.0 extra
    xp = make_xp(pl, [1], lambda p, g: 30 if p == 16 else 2)
    no = solve(xp, pl, sq, 0.5, 1, 1, **OPTS)
    assert 16 not in squad_at(no, 1)
    yes = solve(xp, pl, sq, 1.0, 1, 1, **OPTS)
    t = yes.transfers.iloc[0]
    assert t["player_in"] == 16 and t["out_price"] == 4.0 and t["in_price"] == 5.0
    assert yes.summary["bank"].iloc[0] == pytest.approx(0.0)


def test_ft_rollover_and_cap(pool):
    plan = run(pool, horizon=3, ft=1, ft_value=0.0)
    assert plan.transfers.empty
    assert list(plan.summary["ft_available"]) == [1, 2, 3]
    capped = run(pool, horizon=2, ft=5)
    assert list(capped.summary["ft_available"]) == [5, 5]
    assert simulate_free_transfers(2, [1, 0, 0], [None] * 3) == ([2, 2, 3], [0, 0, 0], 4)
    assert simulate_free_transfers(1, [3], [None]) == ([1], [2], 1)


def test_hit_accounting(pool):
    _, sq = pool
    banned = list(sq["player_id"].iloc[[3, 4]])

    def fn(p, g):
        return 0 if p in banned else 5

    plan = run(pool, fn, ban=banned, ft=1)
    assert plan.summary["transfers"].iloc[0] == 2
    assert plan.summary["hits"].iloc[0] == 1 and plan.summary["hit_cost"].iloc[0] == 4
    free = run(pool, fn, ban=banned, ft=2)
    assert free.summary["hits"].iloc[0] == 0
    # same lineup value; the only objective difference is the 4-point hit (+ft_value of 1 FT)
    assert free.objective - plan.objective == pytest.approx(4.0 + 1.5 * (1 - 1), abs=1e-2)


def test_small_upgrade_does_not_justify_hit(pool):
    plan = run(pool, lambda p, g: 3.0 if p in {11, 12, 13} else 2.0, ft=1)
    assert plan.summary["hits"].iloc[0] == 0
    assert plan.summary["transfers"].iloc[0] <= 1


def test_big_upgrade_justifies_hit(pool):
    _, sq = pool
    owned = set(sq["player_id"])
    out = [p for p in (11, 12, 13, 14, 15, 16) if p not in owned]
    plan = run(pool, lambda p, g: 30.0 if p in out else 2.0, ft=1, bank=0.0)
    assert plan.summary["hits"].iloc[0] >= 1


def test_free_hit_reverts_and_freezes_ft(pool):
    _, sq = pool
    stars = {11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33}

    def fn(p, g):
        return 10 if (g == 2 and p in stars) else 2

    plan = run(pool, fn, horizon=3, ft=2, chips={2: "freehit"}, ft_value=0.0)
    orig = set(sq["player_id"])
    assert squad_at(plan, 2) != orig
    assert (plan.transfers["gw"] == 2).sum() == 0
    assert squad_at(plan, 3) == orig
    fts = list(plan.summary["ft_available"])
    assert fts[2] == fts[1]  # FT frozen across the FH GW


def test_wildcard_freezes_ft_and_no_hits(pool):
    stars = {11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26}
    plan = run(pool, lambda p, g: 10 if (g >= 2 and p in stars) else 2, horizon=3, ft=1,
               chips={2: "wildcard"}, ft_value=0.0)
    s = plan.summary
    assert s["transfers"].iloc[1] >= 5 and s["hits"].sum() == 0
    assert s["ft_available"].iloc[2] == s["ft_available"].iloc[1]
    assert s["chip"].iloc[1] == "wildcard"


def test_bench_boost_counts_bench(pool):
    def fn(p, g):
        return p % 5 + 1

    base = run(pool, fn)
    bb = run(pool, fn, chips={1: "bboost"})
    lg = bb.lineups
    assert (lg["multiplier"] > 0).all()
    assert bb.summary["xp_total"].iloc[0] == pytest.approx((lg["xp"] * lg["multiplier"]).sum())
    assert bb.summary["xp_total"].iloc[0] > base.summary["xp_total"].iloc[0]
    assert bb.summary["xp_xi"].iloc[0] < bb.summary["xp_total"].iloc[0]


def test_triple_captain(pool):
    def fn(p, g):
        return p % 5 + 1

    plan = run(pool, fn, chips={1: "3xc"})
    lg = plan.lineups
    assert lg.loc[lg["is_captain"], "multiplier"].iloc[0] == 3
    assert lg.loc[lg["is_captain"], "xp"].iloc[0] == lg[lg["in_xi"]]["xp"].max()
    plain = run(pool, fn)
    assert plain.lineups.loc[plain.lineups["is_captain"], "multiplier"].iloc[0] == 2
    assert plan.summary["xp_xi"].iloc[0] > plain.summary["xp_xi"].iloc[0]


def test_lock_and_ban(pool):
    pl, sq = pool
    weak = int(sq["player_id"].iloc[0])
    keep = run(pool, lambda p, g: 0 if p == weak else 5, lock=[weak], ft=3)
    assert weak in squad_at(keep, 1)
    bn = run(pool, lambda p, g: 50 if p == 11 else 2, ban=[11], ft=3)
    assert 11 not in squad_at(bn, 1)
    with pytest.raises(ValueError):
        solve(make_xp(pl, [1]), pl, sq, 0, 1, 1, lock=[11], ban=[11])


def test_infeasible_squad_error(pool):
    pl, sq = pool
    with pytest.raises(ValueError, match="infeasible"):
        solve(make_xp(pl, [1]), pl, sq.iloc[:14], 0, 1, 1)


def test_unavailable_not_bought_and_missing_xp_zero(pool):
    pl, sq = pool
    pl = pl.copy()
    pl.loc[pl["player_id"] == 11, "status"] = "u"
    xp = make_xp(pl, [1], lambda p, g: 40 if p == 11 else 2)
    plan = solve(xp[xp["player_id"] != 12], pl, sq, 0.0, 3, 1, **OPTS)
    assert plan.ok and 11 not in squad_at(plan, 1)


def test_multi_gw(pool):
    plan = run(pool, lambda p, g: p % 9 + g, horizon=3, ft=2)
    assert list(plan.summary["gw"]) == [1, 2, 3]
    assert plan.lineups.groupby("gw").size().eq(15).all()


# ---- real squad (ui_data) -------------------------------------------------


def test_real_squad_one_ft_plan(ui_data):
    plan = solve_from_storage(horizon=1, model="baseline", time_limit=30)
    assert plan.ok
    players = storage.read_table("players")
    my_team = storage.read_table("my_team")
    entry = storage.read_table("my_entry").iloc[0]
    assert validate_plan(plan, players, my_team, float(entry["bank"]),
                         int(entry["free_transfers"])) == []
    s = plan.summary.iloc[0]
    assert len(plan.lineups) == 15 and s["bank"] >= 0
    assert s["transfers"] <= 1 or s["hits"] >= 1
    lg = plan.lineups
    assert lg["is_captain"].sum() == 1 and lg.loc[lg["is_captain"], "in_xi"].all()
    assert lg["team_id"].value_counts().max() <= MAX_PER_TEAM


def test_real_multi_gw_with_chip(ui_data):
    plan = solve_from_storage(horizon=3, time_limit=60, mip_gap=5e-3, chips={8: "bboost"})
    assert plan.ok
    assert plan.summary["chip"].fillna("").tolist() == ["", "bboost", ""]
