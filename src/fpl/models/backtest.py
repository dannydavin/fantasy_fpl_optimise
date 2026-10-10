"""Walk-forward backtest of the baseline xP model on past seasons (hist_player_gw).

For each GW ``k >= start_gw`` only rows with ``gw < k`` of the season (plus earlier seasons
for the per-player prior) are used to project GW ``k``; fixtures of GW ``k`` (opponent, home
or away) are treated as known. Predictions use the same engine as
:func:`fpl.models.xp_baseline.project` (``estimate_rates`` + ``fixture_points``) but no
injury news (not in history), so availability is inferred from starts only.

Baselines: ``ppg`` (points per appearance so far x fixtures) and ``form3`` (mean points over
the last 3 team games x fixtures).
Subsets: ``all`` (every player with a fixture row), ``played`` (actual minutes > 0; conditions
on the outcome), ``regular`` (>= 270 minutes before k; no look-ahead).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fpl import storage
from fpl.models import xp_baseline as xp

SUBSETS = ("all", "played", "regular")


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation via pandas ranks (average ties)."""
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def _results(cur: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture: gw, home, away, hs, as (team names)."""
    r = cur[cur["team_h_score"].notna() & cur["fixture_id"].notna()]
    r = r.drop_duplicates("fixture_id")
    home = np.where(r["was_home"], r["team"], r["opponent"])
    away = np.where(r["was_home"], r["opponent"], r["team"])
    return pd.DataFrame({"gw": r["gw"].astype(int).to_numpy(), "home": home, "away": away,
                         "hs": r["team_h_score"].astype(float).to_numpy(),
                         "as_": r["team_a_score"].astype(float).to_numpy()})


def _team_obs(res: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame({"team_id": res["home"], "gf": res["hs"], "ga": res["as_"]})
    a = pd.DataFrame({"team_id": res["away"], "gf": res["as_"], "ga": res["hs"]})
    return pd.concat([h, a]).groupby("team_id").agg(g=("gf", "size"), gf=("gf", "sum"),
                                                    ga=("ga", "sum")).reset_index()


def _prior_ratings(prev: pd.DataFrame | None) -> tuple[pd.Series, pd.Series]:
    if prev is None or prev.empty or "opponent" not in prev:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    last = prev[prev["season"] == prev["season"].max()]
    obs = _team_obs(_results(last)).set_index("team_id")
    if obs.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    mu = obs["gf"].sum() / obs["g"].sum()
    return (obs["gf"] / obs["g"] / mu), (obs["ga"] / obs["g"] / mu)


def _naive(before: pd.DataFrame, target: pd.DataFrame, k: int, team_games: pd.Series) -> pd.DataFrame:
    app = before[before["minutes"] > 0]
    ppg = app.groupby("element")["total_points"].mean()
    recent = before[before["gw"] >= k - 3].groupby("element")["total_points"].sum()
    # points over the last three GWs / number of the team's games in that window
    win = pd.Series(np.minimum(3, k - 1), index=target.index, dtype=float)
    out = pd.DataFrame(index=target.index)
    out["ppg"] = target["element"].map(ppg).fillna(0.0)
    out["form3"] = target["element"].map(recent).fillna(0.0) / win.clip(lower=1)
    return out


def backtest(
    season: str,
    root: Path | None = None,
    start_gw: int = 5,
    hist: pd.DataFrame | None = None,
    end_gw: int | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest of ``season``; returns one row per (gw, subset).

    Columns: gw, subset, n, mae_xp, mae_ppg, mae_form3, rho_xp, rho_ppg, rho_form3.
    Per-player predictions are attached as ``result.attrs["predictions"]``.
    """
    if hist is None:
        hist = storage.read_table("hist_player_gw", root=root)
    cur = hist[hist["season"] == season].copy()
    if cur.empty:
        raise ValueError(f"no history for {season}")
    for c in ("minutes", "total_points", "starts", "expected_goals", "expected_assists",
              "bonus", "saves", "defensive_contribution", "gw", "element"):
        cur[c] = pd.to_numeric(cur[c], errors="coerce")
    prev = hist[hist["season"] < season]
    pos_prior, player_prior = xp.build_priors(prev if len(prev) else None)
    prior_att, prior_def = _prior_ratings(prev)
    results = _results(cur)
    use_dc = season >= "2025-26"
    last_gw = int(cur["gw"].max()) if end_gw is None else end_gw
    out_rows = []
    preds = []
    for k in range(start_gw, last_gw + 1):
        target = cur[cur["gw"] == k].copy()
        if target.empty:
            continue
        before = cur[cur["gw"] < k]
        res = results[results["gw"] < k]
        ratings = xp.shrunk_ratings(_team_obs(res), prior_att, prior_def) if len(res) else None
        if ratings is None:
            continue
        mu = ratings.attrs["mu"]
        rt = ratings.set_index("team_id")
        team_games = ratings.set_index("team_id")["games"]

        g = before.groupby("element")
        stats = pd.DataFrame({
            "minutes": g["minutes"].sum(), "starts": g["starts"].sum(),
            "xg": g["expected_goals"].sum(min_count=1), "xa": g["expected_assists"].sum(min_count=1),
            "bonus": g["bonus"].sum(), "saves": g["saves"].sum(),
            "dc": g["defensive_contribution"].sum(min_count=1) if use_dc else np.nan,
        })
        meta = target.groupby("element").agg(position=("position", "first"), code=("code", "first"),
                                             team=("team", "first"))
        stats = meta.join(stats, how="left")
        stats[["minutes", "starts", "bonus", "saves"]] = stats[
            ["minutes", "starts", "bonus", "saves"]].fillna(0)
        stats["team_games"] = stats["team"].map(team_games).fillna(0.0)
        win = before[before["gw"] >= k - 5]
        wg = win.groupby("element")["starts"].sum()
        wn = pd.Series(min(5, k - 1), index=stats.index, dtype=float).clip(lower=1)
        stats["recent_start"] = (stats.index.to_series().map(wg).fillna(0) / wn
                                 ).where(stats["minutes"] > 0, np.nan)
        stats["code"] = stats["code"].astype("float").astype("Int64").astype("float")
        rates = xp.estimate_rates(stats, pos_prior, player_prior)

        t = target.merge(rates.join(stats[["team", "minutes"]].rename(
            columns={"minutes": "prior_minutes", "team": "team_"}), how="left"),
            left_on="element", right_index=True, how="left")
        opp, own = t["opponent"], t["team"]
        att_o = opp.map(rt["att"]).fillna(1.0).to_numpy()
        def_o = opp.map(rt["def"]).fillna(1.0).to_numpy()
        att_t = own.map(rt["att"]).fillna(1.0).to_numpy()
        def_t = own.map(rt["def"]).fillna(1.0).to_numpy()
        hf = np.where(t["was_home"].astype(bool), xp.HOME, 1 / xp.HOME)
        feat = pd.DataFrame({
            "position": t["position"], "p_start": t["p_start_base"],
            "p_sub": (1 - t["p_start_base"]) * t["p_sub"], "mps": t["mps"],
            "xg90": t["xg90"], "xa90": t["xa90"], "bonus90": t["bonus90"],
            "saves90": t["saves90"], "dc90": t["dc90"],
            "lam_for": mu * att_t * def_o * hf, "lam_against": mu * att_o * def_t / hf,
            "att_adj": def_o * hf, "save_adj": att_o / hf,
            "use_dc": use_dc,
        })
        pts = xp.fixture_points(feat)
        t["pred_xp"] = pts["xp"].to_numpy()
        naive = _naive(before, t, k, team_games)
        t["pred_ppg"] = naive["ppg"].to_numpy()
        t["pred_form3"] = naive["form3"].to_numpy()
        agg = t.groupby("element").agg(
            actual=("total_points", "sum"), minutes=("minutes", "sum"),
            pred_xp=("pred_xp", "sum"), pred_ppg=("pred_ppg", "sum"),
            pred_form3=("pred_form3", "sum"), prior_minutes=("prior_minutes", "first"))
        preds.append(agg.reset_index().assign(gw=k))
        masks = {
            "all": pd.Series(True, index=agg.index),
            "played": agg["minutes"] > 0,
            "regular": agg["prior_minutes"].fillna(0) >= 270,
        }
        for name, m in masks.items():
            a = agg[m]
            if a.empty:
                continue
            row = {"gw": k, "subset": name, "n": len(a)}
            for model in ("xp", "ppg", "form3"):
                row[f"mae_{model}"] = float((a[f"pred_{model}"] - a["actual"]).abs().mean())
                row[f"rho_{model}"] = spearman(a[f"pred_{model}"], a["actual"])
            out_rows.append(row)
    result = pd.DataFrame(out_rows)
    result.attrs["predictions"] = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return result


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-GW results: n-weighted MAE, mean per-GW Spearman, per subset."""
    rows = []
    for name, g in bt.groupby("subset"):
        row = {"subset": name, "gws": g["gw"].nunique(), "n": int(g["n"].sum())}
        for model in ("xp", "ppg", "form3"):
            row[f"mae_{model}"] = float(np.average(g[f"mae_{model}"], weights=g["n"]))
            row[f"rho_{model}"] = float(g[f"rho_{model}"].mean())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("subset").reindex(list(SUBSETS)).dropna(how="all")
    return out.reset_index()
