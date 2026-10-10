"""Baseline expected-points (xP) model.

Per player per target GW (DGW fixtures summed, blank = 0)::

    xP = appearance pts (1 sub/short, 2 for 60+)
       + goal_pts[pos] * xG90 * xmins/90 * attack_adj
       + assist_pts   * xA90 * xmins/90 * attack_adj
       + cs_pts[pos]  * P(clean sheet) * P(60+)
       + dc_pts[pos]  * P(defensive-contribution threshold)
       + bonus90 * xmins/90
       - goals-conceded penalty (GKP/DEF)  + saves pts (GKP)

Rates (xG/xA/bonus/saves/DC per 90) are this season's totals shrunk (minutes-weighted)
towards a prior built from past seasons (matched by player ``code``) and positional means.
Team strength comes from this season's goals / Understat xG shrunk to last season's xG.
The per-fixture engine :func:`fixture_points` is shared with :mod:`fpl.models.backtest`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl import storage
from fpl.config import get_settings

log = logging.getLogger(__name__)

MODEL = "baseline"
HOME = 1.12  # home goals multiplier (away uses 1/HOME)
DEFAULT_MU = 1.40  # goals per team per match
DC_THRESHOLD = {"GKP": 99, "DEF": 10, "MID": 12, "FWD": 12}
SUB_MINUTES = 22.0  # average minutes of a substitute appearance
STARTER_MINUTES = 85.0

DEFAULT_SCORING = {
    "long_play": 2, "short_play": 1, "assists": 3, "saves": 1, "bonus": 1,
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
}
POS_PRIOR_DEFAULT = pd.DataFrame(
    {
        "xg90": [0.0, 0.05, 0.14, 0.32],
        "xa90": [0.01, 0.06, 0.11, 0.10],
        "bonus90": [0.16, 0.17, 0.17, 0.20],
        "saves90": [3.0, 0.0, 0.0, 0.0],
        "dc90": [0.5, 7.0, 4.5, 2.5],
    },
    index=["GKP", "DEF", "MID", "FWD"],
)
RATE_K = {"xg90": 900.0, "xa90": 900.0, "bonus90": 1500.0, "saves90": 900.0, "dc90": 900.0}
POS_PRIOR_MINUTES = 450.0
SEASON_WEIGHT = [1.0, 0.5, 0.25]  # most recent first


def resolve_scoring(scoring: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_SCORING.items()}
    for k, v in (scoring or {}).items():
        if k in out:
            out[k] = {**out[k], **v} if isinstance(v, dict) else v
    return out


# ---- maths helpers ----------------------------------------------------------------------


def poisson_sf(lam: np.ndarray, k: np.ndarray | int) -> np.ndarray:
    """P(X >= k) for X ~ Poisson(lam); ``k`` scalar or array (>=0), vectorised."""
    lam = np.asarray(lam, dtype=float)
    k = np.broadcast_to(np.asarray(k, dtype=int), lam.shape)
    cdf = np.zeros_like(lam)
    pmf = np.exp(-lam)
    for i in range(int(k.max(initial=0))):
        cdf = cdf + np.where(i < k, pmf, 0.0)
        pmf = pmf * lam / (i + 1)
    return np.clip(1.0 - cdf, 0.0, 1.0)


def expected_floor_half(lam: np.ndarray, kmax: int = 14) -> np.ndarray:
    """E[floor(X/2)] for X ~ Poisson(lam) (goals-conceded penalty: -1 per 2 goals)."""
    lam = np.asarray(lam, dtype=float)
    pmf = np.exp(-lam)
    tot = np.zeros_like(lam)
    for i in range(kmax + 1):
        tot += pmf * (i // 2)
        pmf = pmf * lam / (i + 1)
    return tot


# ---- priors -----------------------------------------------------------------------------


def build_priors(history: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positional means and per-player (``code``) per-90 priors from past seasons.

    Returns ``(pos_prior, player_prior)``; ``player_prior`` is indexed by code with columns
    ``xg90, xa90, bonus90, saves90, dc90`` (minutes-weighted over seasons, recent heavier),
    ``p_mins`` (weighted minutes), ``start_rate`` (starts / 38 style rate, NaN if unknown).
    """
    pos_prior = POS_PRIOR_DEFAULT.copy()
    empty = pd.DataFrame(
        columns=["xg90", "xa90", "bonus90", "saves90", "dc90", "p_mins", "start_rate"],
        index=pd.Index([], name="code", dtype="int64"),
    )
    if history is None or history.empty:
        return pos_prior, empty
    h = history.copy()
    seasons = sorted(h["season"].unique(), reverse=True)
    w = {s: SEASON_WEIGHT[min(i, len(SEASON_WEIGHT) - 1)] for i, s in enumerate(seasons)}
    h["w"] = h["season"].map(w)
    for c in ("minutes", "expected_goals", "expected_assists", "bonus", "saves",
              "defensive_contribution", "starts"):
        h[c] = pd.to_numeric(h.get(c, np.nan), errors="coerce")
    cols = {"xg90": "expected_goals", "xa90": "expected_assists", "bonus90": "bonus",
            "saves90": "saves", "dc90": "defensive_contribution"}

    # positional means from the latest season, players with >= 450 minutes
    last = h[h["season"] == seasons[0]]
    tot = last.groupby(["element", "position"], as_index=False)[
        ["minutes", *cols.values()]
    ].sum(min_count=1)
    tot = tot[tot["minutes"] >= 450]
    if len(tot):
        for pos, g in tot.groupby("position"):
            if pos not in pos_prior.index:
                continue
            for name, col in cols.items():
                ok = g[col].notna()
                if ok.sum() >= 10 and g.loc[ok, "minutes"].sum() > 0:
                    pos_prior.loc[pos, name] = g.loc[ok, col].sum() * 90 / g.loc[ok, "minutes"].sum()
    hc = h[h["code"].notna()]
    if hc.empty:
        return pos_prior, empty
    # accumulate weighted sums per code (vectorised; per-GW rows are many)
    for name, col in cols.items():
        v = hc[col]
        hc = hc.assign(**{
            f"{name}_x": (v * hc["w"]).where(v.notna(), 0.0),
            f"{name}_m": (hc["minutes"] * hc["w"]).where(v.notna(), 0.0),
        })
    hc = hc.assign(p_mins=hc["minutes"] * hc["w"], s_x=(hc["starts"] * hc["w"]).fillna(0),
                   s_w=hc["w"].where(hc["starts"].notna(), 0.0))
    grp = hc.groupby("code").sum(numeric_only=True)
    pp = pd.DataFrame(index=grp.index)
    for name in cols:
        pp[name] = np.where(grp[f"{name}_m"] > 0, grp[f"{name}_x"] * 90 / grp[f"{name}_m"].where(
            grp[f"{name}_m"] > 0, 1), np.nan)
    pp["p_mins"] = grp["p_mins"]
    # start rate: weighted starts / weighted appearances-fixtures (rows) in the squad lists
    rows = hc.groupby("code")["w"].sum()
    pp["start_rate"] = np.where(grp["s_w"] > 0, grp["s_x"] / rows.where(rows > 0, 1), np.nan)
    return pos_prior, pp


# ---- team ratings -----------------------------------------------------------------------


def team_ratings(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    understat: pd.DataFrame | None = None,
    season: str | None = None,
    k_games: float = 8.0,
) -> pd.DataFrame:
    """Attack/defence ratings (1 = league average; def > 1 = leaky) plus games and mu.

    Observations: this season's Understat xG/xGA when available for the team, else goals
    from finished fixtures. Prior: previous season's Understat xG/xGA per game (relative to
    that season's mean); teams absent from it (promoted) get a weak prior (att .93, def 1.07).
    """
    season = season or get_settings().season
    t = teams[["team_id"]].copy()
    fin = fixtures[fixtures["finished"].fillna(False).astype(bool)
                   & fixtures["team_h_score"].notna()]
    h = pd.DataFrame({"team_id": fin["team_h"], "gf": fin["team_h_score"],
                      "ga": fin["team_a_score"]})
    a = pd.DataFrame({"team_id": fin["team_a"], "gf": fin["team_a_score"],
                      "ga": fin["team_h_score"]})
    obs = pd.concat([h, a]).astype(float).groupby("team_id").agg(
        g=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"))
    t = t.merge(obs, left_on="team_id", right_index=True, how="left").fillna(
        {"g": 0.0, "gf": 0.0, "ga": 0.0})
    short = teams.set_index("team_id")["short_name"] if "short_name" in teams else None
    prior_att = pd.Series(1.0, index=t["team_id"])
    prior_def = pd.Series(1.0, index=t["team_id"])
    if understat is not None and len(understat) and short is not None:
        us = understat.copy()
        if "team_short" not in us:
            from fpl.ingest.understat import TEAM_SHORT
            us["team_short"] = us["team"].map(TEAM_SHORT)
        past = us[us["season"] < season]
        if len(past):
            ps = past[past["season"] == past["season"].max()]
            per = ps.groupby("team_short")[["xg", "xga"]].mean()
            mu_p = per["xg"].mean()
            by_id = short.map(lambda s: s)
            prior_att = by_id.map(per["xg"] / mu_p).fillna(0.93)
            prior_def = by_id.map(per["xga"] / mu_p).fillna(1.07)
        cur = us[us["season"] == season]
        if len(cur):
            c = cur.groupby("team_short").agg(g=("xg", "size"), xf=("xg", "sum"),
                                              xa=("xga", "sum"))
            idx = t["team_id"].map(short)
            has = idx.map(c["g"]).notna().to_numpy()
            t.loc[has, "g"] = idx[has].map(c["g"]).to_numpy()
            t.loc[has, "gf"] = idx[has].map(c["xf"]).to_numpy()
            t.loc[has, "ga"] = idx[has].map(c["xa"]).to_numpy()
    out = shrunk_ratings(t, prior_att, prior_def, k_games)
    return out


def shrunk_ratings(
    obs: pd.DataFrame, prior_att: pd.Series, prior_def: pd.Series, k_games: float = 8.0
) -> pd.DataFrame:
    """Shrink observed per-team goals for/against towards priors.

    ``obs``: team_id, g (games), gf, ga. ``prior_att/def``: relative ratings indexed by
    team_id (1 = average). Returns team_id, att, def (means normalised to 1), games, with
    league goals-per-team-game in ``attrs["mu"]``.
    """
    n_obs = obs["g"].sum()
    mu = (obs["gf"].sum() + DEFAULT_MU * 40) / (n_obs + 40)
    pa = obs["team_id"].map(prior_att).fillna(1.0).to_numpy()
    pd_ = obs["team_id"].map(prior_def).fillna(1.0).to_numpy()
    att = (obs["gf"] + k_games * mu * pa) / ((obs["g"] + k_games) * mu)
    dfn = (obs["ga"] + k_games * mu * pd_) / ((obs["g"] + k_games) * mu)
    out = pd.DataFrame({"team_id": obs["team_id"], "att": att / att.mean(),
                        "def": dfn / dfn.mean(), "games": obs["g"]})
    out.attrs["mu"] = float(mu)
    return out


# ---- rate estimation (shared with backtest) ---------------------------------------------


def estimate_rates(
    stats: pd.DataFrame, pos_prior: pd.DataFrame, player_prior: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Shrunk per-90 rates and playing-time parameters.

    ``stats`` columns: position, code (optional), minutes, starts, team_games, xg, xa,
    bonus, saves, dc (totals this season; NaN dc allowed), optional ``recent_start``
    (start share over the last few GWs) and ``use_dc``.
    Returns a frame aligned to ``stats`` with xg90, xa90, bonus90, saves90, dc90,
    p_start_base, mps (expected minutes when starting), p_sub (sub-appearance prob if not
    starting).
    """
    s = stats
    pos = s["position"]
    pri = pd.DataFrame(index=s.index)
    if player_prior is not None and len(player_prior) and "code" in s:
        pp = player_prior.reindex(s["code"].to_numpy())
        pp.index = s.index
    else:
        pp = pd.DataFrame(np.nan, index=s.index, columns=player_prior.columns
                          if player_prior is not None else ["p_mins", "start_rate"])
    pm = pp["p_mins"].fillna(0.0) if "p_mins" in pp else pd.Series(0.0, index=s.index)
    out = pd.DataFrame(index=s.index)
    mins = s["minutes"].astype(float).fillna(0.0)
    for name, col in (("xg90", "xg"), ("xa90", "xa"), ("bonus90", "bonus"),
                      ("saves90", "saves"), ("dc90", "dc")):
        posm = pos.map(pos_prior[name]).astype(float)
        if name in pp:
            pw = pm.clip(upper=3000.0)
            ph = pp[name].astype(float)
            ph = ph.where(ph.notna(), posm)
            wph = pw.where(pp[name].notna(), 0.0)
            prior = (ph * wph + posm * POS_PRIOR_MINUTES) / (wph + POS_PRIOR_MINUTES)
        else:
            prior = posm
        pri[name] = prior
        tot = pd.to_numeric(s[col], errors="coerce") if col in s else pd.Series(
            np.nan, index=s.index)
        have = tot.notna()
        k = RATE_K[name]
        est = (tot.fillna(0.0) + prior * k / 90.0) / ((mins.where(have, 0.0) + k) / 90.0)
        out[name] = est
    tg = s["team_games"].astype(float).fillna(0.0)
    starts = s["starts"].astype(float).fillna(0.0)
    prior_start = pp["start_rate"].astype(float) if "start_rate" in pp else np.nan
    prior_start = pd.Series(prior_start, index=s.index).fillna(0.3).clip(0.02, 0.95)
    psb = (starts + 1.5 * prior_start) / (tg + 1.5)
    if "recent_start" in s:
        rs = s["recent_start"].astype(float)
        psb = np.where(rs.notna(), 0.5 * psb + 0.5 * rs, psb)
    out["p_start_base"] = np.clip(psb, 0.0, 0.98)
    mps_raw = (mins / starts.where(starts > 0)).clip(20, 90)
    out["mps"] = (mps_raw.fillna(75.0) * starts + 82.0 * 2) / (starts + 2)
    non_games = (tg - starts).clip(lower=0)
    non_mins = (mins - STARTER_MINUTES * starts).clip(lower=0)
    out["p_sub"] = ((non_mins / SUB_MINUTES + 0.15 * 1.5) / (non_games + 1.5)).clip(0.02, 0.85)
    return out


# ---- availability -----------------------------------------------------------------------


def availability(
    status: pd.Series, chance: pd.Series, offset: int
) -> pd.Series:
    """P(available) for a GW ``offset`` weeks after the snapshot's next GW.

    The FPL chance figure applies to the next GW. For later GWs, doubtful/injured players
    drift back towards full availability over ~6 GWs (to 60% of the missing share);
    unavailable (``u``), suspended (``s``) and not-in-squad (``n``) stay at their value.
    """
    status = status.fillna("a").astype(str)
    c = pd.to_numeric(chance, errors="coerce")
    default = status.map({"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}).fillna(1)
    base = (c / 100.0).where(c.notna(), default).clip(0, 1)
    ramp = min(1.0, offset / 6.0) * 0.6
    drift = status.isin(["a", "d", "i"]).astype(float)
    # suspensions are short: recover fully after offset 1-2
    susp = (status == "s").astype(float) * min(1.0, offset / 3.0)
    return (base + (1 - base) * (ramp * drift + susp)).clip(0, 1)


# ---- per-fixture engine -----------------------------------------------------------------


def fixture_points(f: pd.DataFrame, scoring: dict | None = None) -> pd.DataFrame:
    """Expected points for each player-fixture row.

    Required columns: position, p_start (P(start) incl. availability), p_sub (P(sub app) if
    not starting, already availability-scaled), mps, xg90, xa90, bonus90, saves90, dc90,
    lam_for, lam_against, att_adj, save_adj; optional ``use_dc`` (bool, default True).
    Returns xp, xmins, p_cs, p_dc, xg, xa plus component points.
    """
    sc = resolve_scoring(scoring)
    pos = f["position"]
    ps = f["p_start"].to_numpy(float)
    psub = f["p_sub"].to_numpy(float)
    mps = f["mps"].to_numpy(float)
    p_app = np.clip(ps + psub, 0, 1)
    xmins = ps * mps + psub * SUB_MINUTES
    p60_start = np.clip((mps - 30.0) / 55.0, 0.0, 1.0)
    p60 = ps * p60_start
    mins_app = np.where(p_app > 1e-9, xmins / np.where(p_app > 1e-9, p_app, 1), 0.0)

    app_pts = sc["long_play"] * p60 + sc["short_play"] * (p_app - p60)
    att_adj = np.clip(f["att_adj"].to_numpy(float), 0.5, 1.8)
    save_adj = np.clip(f["save_adj"].to_numpy(float), 0.5, 1.8)
    goal_pts = pos.map(sc["goals_scored"]).to_numpy(float)
    xg = f["xg90"].to_numpy(float) * xmins / 90.0 * att_adj
    xa = f["xa90"].to_numpy(float) * xmins / 90.0 * att_adj
    goals = goal_pts * xg
    assists = sc["assists"] * xa

    lam_a = f["lam_against"].to_numpy(float)
    p_cs = np.exp(-lam_a)
    cs = pos.map(sc["clean_sheets"]).to_numpy(float) * p_cs * p60
    gc_pen = pos.map(sc["goals_conceded"]).to_numpy(float)
    gc = gc_pen * p_app * expected_floor_half(lam_a * mins_app / 90.0)

    is_gk = (pos == "GKP").to_numpy()
    saves = np.where(is_gk, sc["saves"] / 3.0 * f["saves90"].to_numpy(float) * xmins / 90.0
                     * save_adj, 0.0)

    dc_pts = pos.map(sc["defensive_contribution"]).to_numpy(float)
    thr = pos.map(DC_THRESHOLD).to_numpy(int)
    lam_dc = f["dc90"].to_numpy(float) * mins_app / 90.0
    p_dc_app = poisson_sf(lam_dc, thr)
    use_dc = f["use_dc"].to_numpy(bool) if "use_dc" in f else np.ones(len(f), bool)
    p_dc = np.where(use_dc, p_app * p_dc_app, 0.0)
    dc = dc_pts * p_dc

    bonus = sc["bonus"] * f["bonus90"].to_numpy(float) * xmins / 90.0
    xp = app_pts + goals + assists + cs + gc + saves + dc + bonus
    return pd.DataFrame({
        "xp": xp, "xmins": xmins, "p_start": ps, "p_cs": p_cs * (p60 > 0), "p_dc": p_dc,
        "xg": xg, "xa": xa,
        "pts_app": app_pts, "pts_goal": goals, "pts_assist": assists, "pts_cs": cs,
        "pts_gc": gc, "pts_saves": saves, "pts_dc": dc, "pts_bonus": bonus,
    }, index=f.index)


# ---- projection -------------------------------------------------------------------------


def _team_fixture_rows(fixtures: pd.DataFrame, teams: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    fx = fixtures[fixtures["gw"].isin(gws)]
    short = teams.set_index("team_id")["short_name"] if "short_name" in teams else None
    h = pd.DataFrame({"gw": fx["gw"], "fixture_id": fx["fixture_id"], "team_id": fx["team_h"],
                      "opp_id": fx["team_a"], "is_home": True})
    a = pd.DataFrame({"gw": fx["gw"], "fixture_id": fx["fixture_id"], "team_id": fx["team_a"],
                      "opp_id": fx["team_h"], "is_home": False})
    rows = pd.concat([h, a], ignore_index=True)
    if short is not None:
        rows["opp"] = rows["opp_id"].map(short).fillna("?") + np.where(
            rows["is_home"], "(H)", "(A)")
    else:
        rows["opp"] = rows["opp_id"].astype(str)
    return rows


def project(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    gws: list[int],
    history: pd.DataFrame | None = None,
    form: pd.DataFrame | None = None,
    scoring: dict | None = None,
    understat: pd.DataFrame | None = None,
    ratings: pd.DataFrame | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Project xP for every player in ``players`` for each GW in ``gws``.

    Returns contract columns ``gw, player_id, xp, p_start, xmins, model`` plus extras
    ``xg, xa, p_cs, p_dc`` (DGW: xp/xmins/xg/xa summed; p_start/p_cs/p_dc averaged over
    fixtures), ``n_fixtures`` and ``opponent`` ("MCI(H), ARS(A)"; blank = "").
    """
    gws = [int(g) for g in gws]
    sc = resolve_scoring(scoring)
    pl = players.copy()
    if "position" not in pl or pl["position"].isna().any():
        from fpl.schemas import POSITIONS
        pl["position"] = pl["position"].fillna(pl["element_type"].map(POSITIONS)) \
            if "position" in pl else pl["element_type"].map(POSITIONS)
    if ratings is None:
        ratings = team_ratings(fixtures, teams, understat, season)
    mu = float(ratings.attrs.get("mu", DEFAULT_MU))
    rt = ratings.set_index("team_id")

    fin = fixtures[fixtures["finished"].fillna(False).astype(bool)]
    played = pd.concat([fin["team_h"], fin["team_a"]]).value_counts()
    pl["team_games"] = pl["team_id"].map(played).fillna(0).astype(float)
    pl["xg"] = pd.to_numeric(pl.get("expected_goals"), errors="coerce")
    pl["xa"] = pd.to_numeric(pl.get("expected_assists"), errors="coerce")
    pl["dc"] = pd.to_numeric(pl.get("defensive_contribution"), errors="coerce")
    pl["saves"] = pd.to_numeric(pl["saves"], errors="coerce") if "saves" in pl else np.nan
    if form is not None and len(form):
        fw = form[form["window"] == form["window"].min()].set_index("player_id")["start_pct"]
        rs = pl["player_id"].map(fw)
        pl["recent_start"] = rs / 100.0 if rs.max() > 1.5 else rs
    pos_prior, player_prior = build_priors(history)
    rates = estimate_rates(pl, pos_prior, player_prior)
    pl = pl.join(rates)

    rows = _team_fixture_rows(fixtures, teams, gws)
    pl = pl.drop(columns=[c for c in ("gw", "fixture_id", "opp", "opp_id", "is_home")
                         if c in pl])
    long = rows.merge(pl, on="team_id", how="inner")
    long["offset"] = long["gw"] - gws[0]
    avail = np.ones(len(long))
    for off, idx in long.groupby("offset").groups.items():
        sub = long.loc[idx]
        avail[long.index.get_indexer(idx)] = availability(
            sub["status"], sub["chance_of_playing_next_round"], int(off)).to_numpy()
    p_start = long["p_start_base"].to_numpy() * avail
    p_sub = (1 - long["p_start_base"].to_numpy()) * long["p_sub"].to_numpy() * avail

    att_o = long["opp_id"].map(rt["att"]).fillna(1.0).to_numpy()
    def_o = long["opp_id"].map(rt["def"]).fillna(1.0).to_numpy()
    att_t = long["team_id"].map(rt["att"]).fillna(1.0).to_numpy()
    def_t = long["team_id"].map(rt["def"]).fillna(1.0).to_numpy()
    hf = np.where(long["is_home"], HOME, 1 / HOME)
    feat = pd.DataFrame({
        "position": long["position"], "p_start": p_start, "p_sub": p_sub, "mps": long["mps"],
        "xg90": long["xg90"], "xa90": long["xa90"], "bonus90": long["bonus90"],
        "saves90": long["saves90"], "dc90": long["dc90"],
        "lam_for": mu * att_t * def_o * hf,
        "lam_against": mu * att_o * def_t / hf,
        "att_adj": def_o * hf,
        "save_adj": att_o / hf,
    })
    pts = fixture_points(feat, sc)
    pts["gw"] = long["gw"].to_numpy()
    pts["player_id"] = long["player_id"].to_numpy()
    pts["opponent"] = long["opp"].to_numpy()
    g = pts.groupby(["gw", "player_id"], sort=True)
    agg = g.agg(xp=("xp", "sum"), xmins=("xmins", "sum"), xg=("xg", "sum"), xa=("xa", "sum"),
                p_start=("p_start", "mean"), p_cs=("p_cs", "mean"), p_dc=("p_dc", "mean"),
                n_fixtures=("xp", "size"), opponent=("opponent", ", ".join)).reset_index()

    # blanks: every player x every target GW gets a row
    full = pd.MultiIndex.from_product([gws, pl["player_id"].tolist()],
                                      names=["gw", "player_id"]).to_frame(index=False)
    out = full.merge(agg, on=["gw", "player_id"], how="left")
    out["n_fixtures"] = out["n_fixtures"].fillna(0).astype(int)
    out["opponent"] = out["opponent"].fillna("")
    for c in ("xp", "xmins", "xg", "xa", "p_start", "p_cs", "p_dc"):
        out[c] = out[c].fillna(0.0)
    out["model"] = MODEL
    cols = ["gw", "player_id", "xp", "p_start", "xmins", "model", "xg", "xa", "p_cs", "p_dc",
            "n_fixtures", "opponent"]
    return out[cols].sort_values(["gw", "xp"], ascending=[True, False]).reset_index(drop=True)


def _read_optional(table: str, root: Path | None, gw: int | None = None) -> pd.DataFrame | None:
    try:
        return storage.read_table(table, gw=gw if storage.get_schema(table).per_gw else None,
                                  root=root)
    except FileNotFoundError:
        return None


def run(
    gw: int | None = None,
    horizon: int = 6,
    root: Path | None = None,
    client=None,
    scoring: dict | None = None,
) -> pd.DataFrame:
    """Project GWs ``gw .. gw+horizon-1`` from the latest snapshot and write ``xp``.

    ``gw`` defaults to the latest snapshot GW. Optional inputs used when present: ``form``
    (processed), ``hist_player_gw`` and ``understat_team_match`` (raw). ``client`` is
    accepted for interface compatibility; the model degrades gracefully without it.
    """
    snap = gw if gw is not None else storage.latest_gw("players", root)
    if snap is None:
        raise FileNotFoundError("no players snapshot; run the snapshot ingest first")
    players = storage.read_table("players", snap, root)
    teams = storage.read_table("teams", snap, root)
    fixtures = storage.read_table("fixtures", snap, root)
    form = _read_optional("form", root, snap)
    history = _read_optional("hist_player_gw", root)
    understat = _read_optional("understat_team_match", root)
    if scoring is None and client is not None:
        try:
            scoring = client.bootstrap_static().get("game_config", {}).get("scoring")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("could not fetch scoring config: %s", exc)
    gws = [g for g in range(snap, snap + horizon) if g <= 38]
    out = project(players, teams, fixtures, gws, history, form, scoring, understat)
    storage.write_table(out, "xp", gw=snap, root=root)
    return out
