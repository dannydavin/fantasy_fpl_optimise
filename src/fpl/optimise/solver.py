"""Multi-period FPL transfer MILP (highspy directly, tiny expression layer below).

``solve`` takes explicit frames; ``solve_from_storage`` reads them from ``fpl.storage``.
No Streamlit dependency.

Formulation (per GW ``g`` in the horizon, ``p`` over a pruned candidate pool):

* ``b[p,g]`` carried squad (previous squad + buys - sells); ``s[p,g]`` squad that plays
  in ``g`` (``== b`` except on a Free Hit GW where it is free and ``b`` reverts).
* ``s`` is legal: 15 men (2/5/5/3), <=3 per club. ``x`` XI (formation limits), captain
  ``c`` and vice ``vc`` in XI and distinct, ordered bench ``k[p,g,j]`` (j=0 bench GK,
  j=1..3 outfield order).
* Money: selling an original player yields ``sell_price``, anything else current
  ``price``; buying costs ``price``; original players cannot be re-bought; bank >= 0
  every GW. Free Hit squad is budgeted against bank + value of the carried squad and
  leaves the bank unchanged.
* Free transfers: integer ``ft[g]`` (FTs available in GW g, ``ft[0]`` given). Exact
  split ``t - ft = h - r`` with ``h, r >= 0``, binary ``z`` (``h <= M z``,
  ``r <= M (1-z)``) so ``h = max(0, t - ft)`` (hits) and ``r = max(0, ft - t)``.
  Normal GW: ``ft[g+1] <= min(5, r + 1)``. WC/FH GW: transfers are free (no hit) and
  ``ft[g+1] <= ft[g]`` (FT frozen, no accrual). More FTs never hurt, so the upper
  bounds are tight at the optimum; the reported ``summary`` is re-simulated exactly
  from the chosen transfers/chips rather than read from the variables.
* Chips (forced via ``chips={gw: name}``, or solver-chosen from ``available_chips``):
  ``wildcard``, ``freehit``, ``bboost``, ``3xc``; <=1 per GW, each <=1 in horizon.

Objective: sum_g decay^(g-g0) * (XI xP + captain xP (+ another for 3xc)
+ vice_weight * vice xP + sum bench_weights * bench xP; bboost counts bench fully)
- HIT_COST * hits, plus ``ft_value`` per FT carried into the GW after the horizon.

Defaults: decay=1.0, vice_weight=0.05, bench_weights=(gk 0.03, outfield 0.21/0.06/0.002),
ft_value=1.5, hit_cost=4, time_limit=60 s, mip_gap=1e-3, candidate_limit=30 per position
(top by horizon xP) plus 10 per position by xP-per-price; originals and locks always kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import highspy
import numpy as np
import pandas as pd

from fpl.schemas import (
    HIT_COST,
    MAX_FREE_TRANSFERS,
    MAX_PER_TEAM,
    SQUAD_COUNTS,
    SQUAD_SIZE,
    XI_LIMITS,
    XI_SIZE,
)

CHIPS = ("wildcard", "freehit", "bboost", "3xc")
CHIP_ALIASES = {"wc": "wildcard", "fh": "freehit", "bb": "bboost", "tc": "3xc", "": None,
                "none": None}
DEFAULT_BENCH_WEIGHTS = (0.03, 0.21, 0.06, 0.002)  # bench GK, outfield 1..3
POS = tuple(SQUAD_COUNTS)
BIG = 100.0
EPS = 1e-6


def norm_chip(name: str | None) -> str | None:
    if name is None:
        return None
    n = str(name).strip().lower()
    n = CHIP_ALIASES.get(n, n)
    if n is not None and n not in CHIPS:
        raise ValueError(f"unknown chip {name!r}; expected one of {CHIPS}")
    return n


@dataclass
class Plan:
    status: str
    objective: float
    transfers: pd.DataFrame
    lineups: pd.DataFrame
    summary: pd.DataFrame
    params: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("Optimal", "Feasible")


TRANSFER_COLS = ["gw", "player_out", "player_in", "out_price", "in_price"]
LINEUP_COLS = ["gw", "player_id", "web_name", "position", "team_id", "slot", "multiplier",
               "is_captain", "is_vice_captain", "xp", "in_xi"]
SUMMARY_COLS = ["gw", "ft_available", "transfers", "hits", "hit_cost", "chip", "xp_xi",
                "xp_total", "bank"]


def simulate_free_transfers(ft0: int, transfers: list[int], chips: list[str | None]):
    """Exact FT accounting. Returns (ft_available per GW, hits per GW, ft after last GW)."""
    ft, fts, hits = int(ft0), [], []
    for t, chip in zip(transfers, chips, strict=True):
        fts.append(ft)
        if chip in ("wildcard", "freehit"):
            hits.append(0)
        else:
            hits.append(max(0, t - ft))
            ft = min(MAX_FREE_TRANSFERS, max(0, ft - t) + 1)
    return fts, hits, ft


def _pick_xp(xp: pd.DataFrame, model: str | None) -> pd.DataFrame:
    if "model" in xp.columns and xp["model"].nunique(dropna=True) > 1:
        if model is None:
            model = "baseline" if "baseline" in set(xp["model"]) else str(xp["model"].iloc[0])
        xp = xp[xp["model"] == model]
    return xp.groupby(["gw", "player_id"], as_index=False)["xp"].sum()


def _candidates(players, original, xpd, gws, locks, limit, value_limit):
    keep = set(original) | set(locks)
    df = players.assign(_xp=[sum(xpd.get((p, g), 0.0) for g in gws) for p in players.index])
    df["_val"] = df["_xp"] / df["price"].clip(lower=3.9)
    for pos in POS:
        sub = df[df["position"] == pos]
        keep |= set(sub.nlargest(limit, "_xp").index)
        keep |= set(sub.nlargest(value_limit, "_val").index)
    return keep


def solve(
    xp: pd.DataFrame,
    players: pd.DataFrame,
    squad: pd.DataFrame,
    bank: float,
    free_transfers: int,
    start_gw: int,
    horizon: int = 1,
    *,
    model: str | None = None,
    chips: dict[int, str] | None = None,
    available_chips: Iterable[str] | None = None,
    lock: Iterable[int] = (),
    ban: Iterable[int] = (),
    allow_unavailable: bool = False,
    decay: float = 1.0,
    vice_weight: float = 0.05,
    bench_weights: tuple[float, float, float, float] = DEFAULT_BENCH_WEIGHTS,
    ft_value: float = 1.5,
    hit_cost: float = HIT_COST,
    max_hits_per_gw: int | None = None,
    max_hits_total: int | None = None,
    max_transfers_per_gw: int | None = None,
    max_transfers_total: int | None = None,
    candidate_limit: int = 30,
    value_limit: int = 10,
    time_limit: float = 60.0,
    mip_gap: float = 1e-3,
) -> Plan:
    """Solve the multi-period plan. See module docstring for formulation and defaults.

    ``squad``: current 15 with ``player_id`` and ``sell_price`` (price used if absent).
    ``chips``: forced chips ``{gw: chip}``; ``available_chips``: chips the solver may place
    itself. Raises ``ValueError`` if the current squad is not a legal 15.
    """
    horizon = int(horizon)
    if not 1 <= horizon <= 6:
        raise ValueError("horizon must be 1..6")
    gws = list(range(start_gw, start_gw + horizon))
    P = players.drop_duplicates("player_id").set_index("player_id").copy()
    for col in ("position", "team_id", "price"):
        if col not in P.columns:
            raise ValueError(f"players table lacks {col!r}")
    if "status" not in P.columns:
        P["status"] = "a"
    if "web_name" not in P.columns:
        P["web_name"] = [f"#{i}" for i in P.index]

    # ---- current squad checks
    sq = squad.drop_duplicates("player_id")
    original = [int(p) for p in sq["player_id"]]
    missing = [p for p in original if p not in P.index]
    if missing:
        raise ValueError(f"current squad players missing from players table: {missing}")
    if len(original) != SQUAD_SIZE:
        raise ValueError(f"current squad is infeasible: {len(original)} players, need 15")
    cnt = P.loc[original, "position"].value_counts().to_dict()
    if any(cnt.get(k, 0) != v for k, v in SQUAD_COUNTS.items()):
        raise ValueError(f"current squad is infeasible: positions {cnt}")
    if P.loc[original, "team_id"].value_counts().max() > MAX_PER_TEAM:
        raise ValueError("current squad is infeasible: more than 3 players from one club")
    if bank < -EPS:
        raise ValueError("current squad is infeasible: negative bank")
    sv = {int(p): float(P.at[p, "price"]) for p in P.index}
    if "sell_price" in sq.columns:
        for r in sq.itertuples():
            if pd.notna(r.sell_price):
                sv[int(r.player_id)] = float(r.sell_price)

    # ---- chips
    forced: dict[int, str] = {}
    for g, c in (chips or {}).items():
        c = norm_chip(c)
        if c is None:
            continue
        if int(g) not in gws:
            raise ValueError(f"chip {c} in GW{g} outside horizon {gws}")
        forced[int(g)] = c
    if len(set(forced.values())) != len(forced):
        raise ValueError("each chip can be forced at most once")
    free_chips = {norm_chip(c) for c in (available_chips or ())} - {None}
    chip_set = set(forced.values()) | free_chips

    # ---- xp lookups, locks, bans
    xdf = _pick_xp(xp, model)
    xpd = {(int(r.player_id), int(r.gw)): float(r.xp) for r in xdf.itertuples()
           if pd.notna(r.xp)}
    lock, ban = {int(p) for p in lock}, {int(p) for p in ban}
    if lock & ban:
        raise ValueError("a player cannot be both locked and banned")
    bad = lock - set(P.index)
    if bad:
        raise ValueError(f"locked players not in players table: {sorted(bad)}")

    unbuyable = set() if allow_unavailable else {p for p in P.index if P.at[p, "status"] == "u"}
    pool = _candidates(P, original, xpd, gws, lock, candidate_limit, value_limit)
    pool = sorted({p for p in pool if p in original or (p not in unbuyable and p not in ban)})
    pos_of = {p: P.at[p, "position"] for p in pool}
    team_of = {p: int(P.at[p, "team_id"]) for p in pool}
    price = {p: float(P.at[p, "price"]) for p in pool}
    orig = set(original)

    m = _Model()
    V = m.var
    G = range(horizon)

    def bvar(name, idx):
        return {i: V(f"{name}_{'_'.join(map(str, i)) if isinstance(i, tuple) else i}",
                     cat="Binary") for i in idx}

    pg = [(p, g) for p in pool for g in G]
    s, x, cap, vic, bs, sell = (bvar(n, pg) for n in ("s", "x", "c", "v", "b", "sell"))
    buy = bvar("buy", [(p, g) for p in pool if p not in orig for g in G])
    kk = bvar("k", [(p, g, j) for p in pool for g in G for j in range(4)
                    if (j == 0) == (pos_of[p] == "GKP")])
    bank_v = {g: V(f"bank{g}", lowBound=0) for g in G}
    ft = {g: V(f"ft{g}", lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer")
          for g in range(horizon + 1)}
    hv = {g: V(f"h{g}", lowBound=0, upBound=BIG, cat="Integer") for g in G}
    rv = {g: V(f"r{g}", lowBound=0, upBound=BIG, cat="Integer") for g in G}
    z = bvar("z", list(G))
    chipv = {(c, g): V(f"chip_{c}_{g}", cat="Binary") for c in chip_set for g in G}

    def chip(c, g):
        return chipv[(c, g)] if c in chip_set else 0

    nt = {g: lpsum(buy[p, g] for p in pool if p not in orig) for g in G}
    for c in chip_set:
        m += lpsum(chip(c, g) for g in G) <= 1
    for g in G:
        m += lpsum(chip(c, g) for c in chip_set) <= 1
        if g + start_gw in forced:
            m += chipv[(forced[g + start_gw], g)] == 1
    for c in chip_set - free_chips:  # forced-only chip: nowhere else
        for g in G:
            if forced.get(g + start_gw) != c:
                m += chipv[(c, g)] == 0
    fh = [chip("freehit", g) for g in G]
    wc = [chip("wildcard", g) for g in G]
    w = [fh[g] + wc[g] for g in G]

    for g in G:
        prev_b = {p: ((1 if p in orig else 0) if g == 0 else bs[p, g - 1]) for p in pool}
        for p in pool:
            m += bs[p, g] == prev_b[p] + (buy[p, g] if p not in orig else 0) - sell[p, g]
            m += sell[p, g] <= prev_b[p]
            if p not in orig:
                m += buy[p, g] <= 1 - fh[g]
            m += sell[p, g] <= 1 - fh[g]
            m += s[p, g] <= bs[p, g] + fh[g]
            m += s[p, g] >= bs[p, g] - fh[g]
            m += x[p, g] <= s[p, g]
            m += cap[p, g] <= x[p, g]
            m += vic[p, g] <= x[p, g]
            m += cap[p, g] + vic[p, g] <= 1
            ks = [kk[p, g, j] for j in range(4) if (p, g, j) in kk]
            m += lpsum(ks) == s[p, g] - x[p, g]
            if p in lock:
                m += s[p, g] == 1
            if p in ban:
                m += s[p, g] == 0
        for pos, n in SQUAD_COUNTS.items():
            m += lpsum(s[p, g] for p in pool if pos_of[p] == pos) == n
            lo, hi = XI_LIMITS[pos]
            xs = lpsum(x[p, g] for p in pool if pos_of[p] == pos)
            m += xs >= lo
            m += xs <= hi
        for t in set(team_of.values()):
            m += lpsum(s[p, g] for p in pool if team_of[p] == t) <= MAX_PER_TEAM
        m += lpsum(x[p, g] for p in pool) == XI_SIZE
        m += lpsum(cap[p, g] for p in pool) == 1
        m += lpsum(vic[p, g] for p in pool) == 1
        for j in range(4):
            m += lpsum(kk[p, g, j] for p in pool if (p, g, j) in kk) == 1
        # money
        prev_bank = bank if g == 0 else bank_v[g - 1]
        m += bank_v[g] == prev_bank + lpsum(sell[p, g] * sv[p] for p in pool) - lpsum(
            buy[p, g] * price[p] for p in pool if p not in orig)
        m += lpsum(s[p, g] * sv[p] for p in pool) <= (
            prev_bank + lpsum(prev_b[p] * sv[p] for p in pool) + 1000 * (1 - fh[g]))
        # transfers / FT
        m += hv[g] - rv[g] >= nt[g] - ft[g] - BIG * w[g]
        m += hv[g] - rv[g] <= nt[g] - ft[g] + BIG * w[g]
        m += hv[g] <= BIG * z[g]
        m += rv[g] <= BIG * (1 - z[g])
        m += hv[g] <= BIG * (1 - w[g])
        m += ft[g + 1] <= rv[g] + 1 + BIG * w[g]
        m += ft[g + 1] <= ft[g] + BIG * (1 - w[g])
        if max_hits_per_gw is not None:
            m += hv[g] <= max_hits_per_gw
        if max_transfers_per_gw is not None:
            m += nt[g] <= max_transfers_per_gw + BIG * wc[g]
    m += ft[0] == int(free_transfers)
    if max_hits_total is not None:
        m += lpsum(hv.values()) <= max_hits_total
    if max_transfers_total is not None:
        m += lpsum(nt.values()) <= max_transfers_total

    # ---- objective
    bw = list(bench_weights)
    obj = []
    for g in G:
        d = decay ** g
        gw = gws[g]
        e = []
        for p in pool:
            xv = xpd.get((p, gw), 0.0)
            if xv == 0.0:
                continue
            e.append(xv * (x[p, g] + cap[p, g] + vice_weight * vic[p, g]))
            if "3xc" in chip_set:
                cc = V(f"cc_{p}_{g}", lowBound=0, upBound=1)
                m += cc <= cap[p, g]
                m += cc <= chipv[("3xc", g)]
                e.append(xv * cc)
            for j in range(4):
                if (p, g, j) not in kk:
                    continue
                e.append(xv * bw[j] * kk[p, g, j])
                if "bboost" in chip_set:
                    bbk = V(f"bb_{p}_{g}_{j}", lowBound=0, upBound=1)
                    m += bbk <= kk[p, g, j]
                    m += bbk <= chipv[("bboost", g)]
                    e.append(xv * (1 - bw[j]) * bbk)
        obj.append(d * (lpsum(e) - hit_cost * hv[g]))
    obj.append(ft_value * ft[horizon])
    obj.append(-1e-3 * lpsum(nt.values()))  # break ties against pointless transfers
    # tiny tie-break towards cheaper squads
    obj.append(-1e-5 * lpsum(s[p, horizon - 1] * price[p] for p in pool))
    m += lpsum(obj)

    status = m.solve(time_limit, mip_gap)
    if status not in ("Optimal", "Feasible"):
        return Plan(status, float("nan"), pd.DataFrame(columns=TRANSFER_COLS),
                    pd.DataFrame(columns=LINEUP_COLS), pd.DataFrame(columns=SUMMARY_COLS))
    return _extract(locals(), status)


class Expr:
    """Minimal linear expression (terms by column index + constant); also the Var type."""

    __hash__ = object.__hash__

    def __init__(self, terms=None, const=0.0):
        self.terms: dict[int, float] = terms or {}
        self.const = const
        self.varValue: float | None = None

    @staticmethod
    def of(o):
        return o if isinstance(o, Expr) else Expr({}, float(o))

    def __add__(self, o):
        o = Expr.of(o)
        terms = dict(self.terms)
        for k, v in o.terms.items():
            terms[k] = terms.get(k, 0.0) + v
        return Expr(terms, self.const + o.const)

    __radd__ = __add__

    def __neg__(self):
        return Expr({k: -v for k, v in self.terms.items()}, -self.const)

    def __sub__(self, o):
        return self + (-Expr.of(o))

    def __rsub__(self, o):
        return Expr.of(o) + (-self)

    def __mul__(self, c):
        c = float(c)
        return Expr({k: v * c for k, v in self.terms.items()}, self.const * c)

    __rmul__ = __mul__

    def __le__(self, o):
        return (self - o, -np.inf, 0.0)

    def __ge__(self, o):
        return (self - o, 0.0, np.inf)

    def __eq__(self, o):  # type: ignore[override]
        return (self - o, 0.0, 0.0)


def lpsum(items) -> Expr:
    terms: dict[int, float] = {}
    const = 0.0
    for it in items:
        if isinstance(it, Expr):
            for k, v in it.terms.items():
                terms[k] = terms.get(k, 0.0) + v
            const += it.const
        else:
            const += float(it)
    return Expr(terms, const)


class _Model:
    def __init__(self):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.vars: list[Expr] = []
        self.objective: Expr | None = None

    def var(self, name="", lowBound=None, upBound=None, cat="Continuous") -> Expr:
        lo = 0.0 if cat == "Binary" else (-np.inf if lowBound is None else float(lowBound))
        hi = 1.0 if cat == "Binary" else (np.inf if upBound is None else float(upBound))
        idx = len(self.vars)
        self.h.addVar(lo, hi)
        if cat in ("Binary", "Integer"):
            self.h.changeColIntegrality(idx, highspy.HighsVarType.kInteger)
        v = Expr({idx: 1.0})
        self.vars.append(v)
        return v

    def __iadd__(self, c):
        if isinstance(c, Expr):
            self.objective = c
            return self
        e, lo, hi = c
        if not e.terms:
            if e.const < lo - 1e-9 or e.const > hi + 1e-9:
                raise ValueError("trivially infeasible constraint")
            return self
        idx = np.fromiter(e.terms.keys(), dtype=np.int32)
        val = np.fromiter(e.terms.values(), dtype=np.float64)
        self.h.addRow(lo - e.const, hi - e.const, len(idx), idx, val)
        return self

    def solve(self, time_limit: float, mip_gap: float) -> str:
        h = self.h
        obj = self.objective
        n = len(self.vars)
        cost = np.zeros(n)
        for k, v in obj.terms.items():
            cost[k] = v
        h.changeColsCost(n, np.arange(n, dtype=np.int32), cost)
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", float(mip_gap))
        h.setOptionValue("threads", 1)
        h.setOptionValue("random_seed", 0)
        h.run()
        status = h.getModelStatus()
        feasible = h.getInfo().primal_solution_status == 2
        if feasible:
            vals = h.getSolution().col_value
            for i, v in enumerate(self.vars):
                v.varValue = float(vals[i])
        if status == highspy.HighsModelStatus.kOptimal:
            return "Optimal"
        if feasible:
            return "Feasible"
        if status == highspy.HighsModelStatus.kInfeasible:
            return "Infeasible"
        return str(h.modelStatusToString(status))

    def objective_value(self) -> float:
        o = self.objective
        return o.const + sum(v * self.vars[k].varValue for k, v in o.terms.items())


def _val(v) -> float:
    return float(v.varValue or 0.0) if hasattr(v, "varValue") else float(v)


def _extract(L: dict, status: str) -> Plan:
    pool, G, P, gws = L["pool"], L["G"], L["P"], L["gws"]
    orig, sv, price, xpd = L["orig"], L["sv"], L["price"], L["xpd"]
    s, x, cap, vic, kk, buy, sell = (L[k] for k in ("s", "x", "cap", "vic", "kk", "buy", "sell"))
    chipv, chip_set = L["chipv"], L["chip_set"]

    def one(v):
        return _val(v) > 0.5

    trans_rows, line_rows, chips_used, tcount = [], [], [], []
    for g in G:
        gw = gws[g]
        c = next((c for c in sorted(chip_set) if one(chipv[(c, g)])), None)
        chips_used.append(c)
        oq = [p for p in pool if one(sell[p, g])]
        iq = [p for p in pool if p not in orig and one(buy[p, g])]
        tcount.append(len(iq))
        for pos in POS:  # pair like-for-like first
            for o in [o for o in oq if P.at[o, "position"] == pos]:
                i = next((i for i in iq if P.at[i, "position"] == pos), None)
                if i is not None:
                    oq.remove(o)
                    iq.remove(i)
                    trans_rows.append((gw, o, i))
        trans_rows += [(gw, o, i) for o, i in zip(oq, iq, strict=False)]
        squad = [p for p in pool if one(s[p, g])]
        xi = [p for p in squad if one(x[p, g])]
        order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        xi.sort(key=lambda p: (order[P.at[p, "position"]], -xpd.get((p, gw), 0.0), p))
        slot_of = {p: n for n, p in enumerate(xi, 1)}
        for j in range(4):
            p = next(p for p in squad if (p, g, j) in kk and one(kk[p, g, j]))
            slot_of[p] = 12 + j
        for p in squad:
            is_c, is_v = one(cap[p, g]), one(vic[p, g])
            mult = 0
            if p in xi:
                mult = (3 if c == "3xc" else 2) if is_c else 1
            elif c == "bboost":
                mult = 1
            line_rows.append({
                "gw": gw, "player_id": p, "web_name": P.at[p, "web_name"],
                "position": P.at[p, "position"], "team_id": int(P.at[p, "team_id"]),
                "slot": slot_of[p], "multiplier": mult, "is_captain": is_c,
                "is_vice_captain": is_v, "xp": xpd.get((p, gw), 0.0), "in_xi": p in xi})
    lineups = pd.DataFrame(line_rows, columns=LINEUP_COLS).sort_values(["gw", "slot"])
    fts, hits, _ = simulate_free_transfers(int(round(_val(L["ft"][0]))), tcount, chips_used)
    transfers = pd.DataFrame(
        [{"gw": gw, "player_out": o, "player_in": i, "out_price": sv[o], "in_price": price[i]}
         for gw, o, i in trans_rows], columns=TRANSFER_COLS)
    bank, rows = L["bank"], []
    for g in G:
        gw = gws[g]
        t = transfers[transfers["gw"] == gw]
        bank = bank + t["out_price"].sum() - t["in_price"].sum()
        lg = lineups[lineups["gw"] == gw]
        xp_xi = float((lg["xp"] * lg["multiplier"] * lg["in_xi"]).sum())
        rows.append({"gw": gw, "ft_available": fts[g], "transfers": tcount[g], "hits": hits[g],
                     "hit_cost": hits[g] * L["hit_cost"], "chip": chips_used[g], "xp_xi": xp_xi,
                     "xp_total": float((lg["xp"] * lg["multiplier"]).sum()),
                     "bank": round(float(bank), 4)})
    summary = pd.DataFrame(rows, columns=SUMMARY_COLS)
    return Plan(status, float(L["m"].objective_value()), transfers,
                lineups.reset_index(drop=True), summary,
                params={"start_gw": gws[0], "horizon": len(gws)})


def validate_plan(plan: Plan, players: pd.DataFrame, squad: pd.DataFrame | None = None,
                  bank: float | None = None, free_transfers: int | None = None) -> list[str]:
    """Return a list of rule violations (empty = legal)."""
    errs: list[str] = []
    if plan.lineups.empty:
        return ["plan has no lineups"]
    P = players.drop_duplicates("player_id").set_index("player_id")
    cur = {int(p) for p in squad["player_id"]} if squad is not None else None
    base = set(cur) if cur is not None else None
    cash = bank
    chips = []
    for r in plan.summary.itertuples():
        gw = r.gw
        lg = plan.lineups[plan.lineups["gw"] == gw]
        ids = list(lg["player_id"])
        chips.append(r.chip)
        if len(ids) != SQUAD_SIZE or len(set(ids)) != SQUAD_SIZE:
            errs.append(f"GW{gw}: squad has {len(ids)} rows / {len(set(ids))} unique")
            continue
        pos = lg["position"].value_counts().to_dict()
        for k, n in SQUAD_COUNTS.items():
            if pos.get(k, 0) != n:
                errs.append(f"GW{gw}: {pos.get(k, 0)} {k}, need {n}")
        if lg["team_id"].value_counts().max() > MAX_PER_TEAM:
            errs.append(f"GW{gw}: more than {MAX_PER_TEAM} players from one club")
        xi = lg[lg["in_xi"]]
        if len(xi) != XI_SIZE:
            errs.append(f"GW{gw}: XI has {len(xi)} players")
        xp_ = xi["position"].value_counts().to_dict()
        for k, (lo, hi) in XI_LIMITS.items():
            if not lo <= xp_.get(k, 0) <= hi:
                errs.append(f"GW{gw}: invalid formation ({k}={xp_.get(k, 0)})")
        cap_ = lg[lg["is_captain"]]
        vc_ = lg[lg["is_vice_captain"]]
        if len(cap_) != 1 or not bool(cap_["in_xi"].all()):
            errs.append(f"GW{gw}: captain missing or not in XI")
        if len(vc_) != 1 or not bool(vc_["in_xi"].all()):
            errs.append(f"GW{gw}: vice-captain missing or not in XI")
        if len(cap_) == 1 and len(vc_) == 1 and cap_["player_id"].iloc[0] == vc_["player_id"].iloc[0]:
            errs.append(f"GW{gw}: captain == vice-captain")
        if sorted(lg["slot"]) != list(range(1, 16)):
            errs.append(f"GW{gw}: slots are not 1..15")
        if sorted(lg[lg["in_xi"]]["slot"]) != list(range(1, 12)):
            errs.append(f"GW{gw}: XI must occupy slots 1..11")
        bench = lg[~lg["in_xi"]].sort_values("slot")
        if len(bench) and bench["position"].iloc[0] != "GKP":
            errs.append(f"GW{gw}: first bench slot must be the spare goalkeeper")
        for p in ids:
            if p in P.index and P.at[p, "status"] == "u" and (base is None or p not in base):
                errs.append(f"GW{gw}: unavailable player {p} bought")
        if base is None:
            continue
        t = plan.transfers[plan.transfers["gw"] == gw]
        outs, ins = set(t["player_out"]), set(t["player_in"])
        if not outs <= base:
            errs.append(f"GW{gw}: selling players not in squad: {sorted(outs - base)}")
        for p in ins:
            if p in P.index and abs(P.at[p, "price"] - float(
                    t[t["player_in"] == p]["in_price"].iloc[0])) > 1e-6:
                errs.append(f"GW{gw}: in_price of {p} differs from current price")
        base = (base - outs) | ins
        if r.chip != "freehit" and set(ids) != base:
            errs.append(f"GW{gw}: lineup does not match squad after transfers")
        if r.chip == "freehit" and len(t):
            errs.append(f"GW{gw}: transfers listed on a Free Hit GW")
        if cash is not None:
            cash = cash + t["out_price"].sum() - t["in_price"].sum()
            if cash < -1e-6:
                errs.append(f"GW{gw}: bank negative ({cash:.2f})")
            if abs(cash - r.bank) > 1e-3:
                errs.append(f"GW{gw}: summary bank {r.bank} != recomputed {cash:.2f}")
    ft0 = free_transfers if free_transfers is not None else int(plan.summary["ft_available"].iloc[0])
    fts, hits, _ = simulate_free_transfers(
        ft0, [int(v) for v in plan.summary["transfers"]], chips)
    for r, f, h in zip(plan.summary.itertuples(), fts, hits, strict=True):
        if r.ft_available != f or r.hits != h or abs(r.hit_cost - h * HIT_COST) > 1e-6:
            if abs(r.hit_cost - h * HIT_COST) > 1e-6 and r.hits == h:
                continue  # custom hit_cost
            errs.append(f"GW{r.gw}: FT/hit accounting mismatch")
    return errs


def solve_from_storage(gw: int | None = None, horizon: int = 1, model: str = "baseline",
                       root: Path | None = None, **opts) -> Plan:
    """Solve from the stored tables (players, my_team, my_entry, xp) for the planning GW."""
    from fpl import storage

    players = storage.read_table("players", gw, root)
    my_team = storage.read_table("my_team", gw, root)
    entry = storage.read_table("my_entry", gw, root)
    xp = storage.read_table("xp", None, root) if not storage.exists("xp", gw, root) \
        else storage.read_table("xp", gw, root)
    if gw is None:
        gw = int(my_team["gw"].iloc[0])
    if "model" in xp.columns:
        xp = xp[xp["model"] == model] if model in set(xp["model"]) else xp
    e = entry.iloc[0]
    opts.setdefault("bank", float(e["bank"]))
    opts.setdefault("free_transfers", int(e["free_transfers"]))
    bank = opts.pop("bank")
    ft = opts.pop("free_transfers")
    return solve(xp, players, my_team[["player_id", "sell_price"]], bank, ft, gw, horizon,
                 model=model, **opts)
