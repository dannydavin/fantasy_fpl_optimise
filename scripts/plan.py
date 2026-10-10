"""Print the solver's transfer plan for the planning GW, optionally cross-checked externally.

    uv run python scripts/plan.py                          # 5-GW horizon from stored tables
    uv run python scripts/plan.py --horizon 6 --cross-check
    uv run python scripts/plan.py --chip 9:bboost --lock Haaland 411 --ban Saka
    uv run python scripts/plan.py --compare-chip bboost    # chip in each GW vs holding it
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without the package installed
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fpl import storage  # noqa: E402
from fpl.optimise import fplopt_wrapper, solver  # noqa: E402


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name).replace("ø", "o").replace("Ø", "O"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().replace(".", " ").strip()


def resolve(tokens: list[str], players: pd.DataFrame, squad_ids: set[int]) -> list[int]:
    """Player ids from ids or names (web_name, accent/case-insensitive; squad wins ties)."""
    keys = players["web_name"].map(_norm)
    ids = []
    for tok in tokens:
        if tok.isdigit():
            ids.append(int(tok))
            continue
        hit = players.loc[keys == _norm(tok), "player_id"].tolist()
        if not hit:  # fall back to substring, e.g. "Pedro" -> "João Pedro"
            hit = players.loc[keys.str.contains(_norm(tok), regex=False), "player_id"].tolist()
        if len(hit) > 1:
            hit = [i for i in hit if i in squad_ids] or hit
        if len(hit) != 1:
            names = players.set_index("player_id").loc[hit, "web_name"].tolist() if hit else []
            raise SystemExit(f"--lock/--ban {tok!r}: {'ambiguous ' + str(names) if hit else 'no match'}")
        ids.append(int(hit[0]))
    return ids


def show(title: str, plan: solver.Plan, names: pd.Series, gw: int) -> None:
    print(f"\n== {title}: {plan.status}, objective {plan.objective:.2f}")
    tr = plan.transfers.copy()
    if tr.empty:
        print("no transfers")
    else:
        tr["out"], tr["in"] = tr["player_out"].map(names), tr["player_in"].map(names)
        print(tr[["gw", "out", "out_price", "in", "in_price"]].to_string(index=False))
    print(plan.summary.to_string(index=False))
    xi = plan.lineups[(plan.lineups["gw"] == gw)]
    cap = xi.loc[xi["is_captain"], "player_id"].map(names).tolist()
    vice = xi.loc[xi["is_vice_captain"], "player_id"].map(names).tolist()
    xp_xi = plan.summary.loc[plan.summary["gw"] == gw, "xp_xi"].sum()
    print(f"GW{gw} captain {cap} vice {vice}, XI xP {xp_xi:.2f} (incl. captain)")
    print(f"net xP over horizon: {net_xp(plan):.2f}")


def net_xp(plan: solver.Plan) -> float:
    return float(plan.summary["xp_total"].sum() - plan.summary["hit_cost"].sum())


def calendar(gws: list[int]) -> None:
    """Teams with a blank or double in each GW of the horizon."""
    from fpl.features.fixtures import blank_double_calendar

    cal = blank_double_calendar(storage.read_table("fixtures"), gws)
    short = storage.read_table("teams").set_index("team_id")["short_name"]
    for g in gws:
        row = cal[cal["gw"] == g]
        parts = [f"{k}s: {', '.join(short.reindex(row.loc[row['kind'] == k, 'team_id']).astype(str))}"
                 for k in ("double", "blank") if (row["kind"] == k).any()]
        print(f"GW{g}: {'; '.join(parts) or 'all single'}")


def compare_chip(chip: str, gw: int, horizon: int, names: pd.Series, model: str, opts: dict) -> None:
    """Re-solve with ``chip`` in each GW of the horizon and tabulate against holding it."""
    gws = list(range(gw, gw + horizon))
    print(f"\n== {chip} by GW (other options unchanged)")
    calendar(gws)
    base_chips = dict(opts.pop("chips") or {})
    rows = []
    for g in [None, *gws]:
        if g in base_chips:
            continue
        chips = {**base_chips, **({g: chip} if g else {})}
        plan = solver.solve_from_storage(gw, horizon, model=model, chips=chips or None, **opts)
        lu = plan.lineups[plan.lineups["gw"] == g]
        extra = (float("nan") if g is None else lu.loc[~lu["in_xi"], "xp"].sum() if chip == "bboost"
                 else lu.loc[lu["is_captain"], "xp"].sum() if chip == "3xc" else float("nan"))
        moves = plan.transfers[plan.transfers["gw"] == gw]
        first = ", ".join(f"{names.get(o)} -> {names.get(i)}"
                          for o, i in zip(moves["player_out"], moves["player_in"], strict=True)) or "roll"
        rows.append({"chip_gw": g or "hold", "net_xp": net_xp(plan), "chip_extra_xp": extra,
                     "hits": int(plan.summary["hits"].sum()), f"gw{gw}_move": first})
    df = pd.DataFrame(rows)
    df.insert(2, "vs_hold", df["net_xp"] - df.loc[df["chip_gw"] == "hold", "net_xp"].iat[0])
    print(df.round(2).to_string(index=False))
    print("chip_extra_xp: bench xP that GW (bboost) / extra captain xP (3xc); hits over the horizon")
    print("hold = chip not played inside this horizon; its value in a later GW is not counted")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gw", type=int, default=None)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--model", default="baseline")
    p.add_argument("--lock", nargs="*", default=[], help="player ids or names to keep/buy")
    p.add_argument("--ban", nargs="*", default=[], help="player ids or names to exclude")
    p.add_argument("--chip", action="append", default=[], help="GW:chip, e.g. 9:bboost")
    p.add_argument("--compare-chip", metavar="CHIP", help="table of CHIP played in each GW vs hold")
    p.add_argument("--cross-check", action="store_true", help="also run FPL-Optimization-Tools")
    p.add_argument("--secs", type=int, default=120, help="external solver time limit")
    a = p.parse_args(argv)

    chips = {int(g): c for g, c in (s.split(":", 1) for s in a.chip)}
    players = storage.read_table("players", a.gw)
    my_team = storage.read_table("my_team", a.gw)
    entry = storage.read_table("my_entry", a.gw).iloc[0]
    gw = a.gw or int(my_team["gw"].iloc[0])
    bank, ft = float(entry["bank"]), int(entry["free_transfers"])
    names = players.set_index("player_id")["web_name"]
    print(f"GW{gw}: bank £{bank:.1f}m, {ft} FT, chips {entry['chips_available']}")

    squad_ids = set(my_team["player_id"].astype(int))
    lock, ban = resolve(a.lock, players, squad_ids), resolve(a.ban, players, set())
    if lock or ban:
        print(f"lock {names.reindex(lock).tolist()} ban {names.reindex(ban).tolist()}")
    opts = dict(chips=chips or None, lock=lock, ban=ban)
    if a.compare_chip:
        compare_chip(solver.norm_chip(a.compare_chip), gw, a.horizon, names, a.model, opts)
        return 0
    t = time.time()
    plan = solver.solve_from_storage(gw, a.horizon, model=a.model, **opts)
    show(f"MILP ({time.time() - t:.0f}s)", plan, names, gw)
    errs = solver.validate_plan(plan, players, my_team, bank, ft)
    print("legal" if not errs else f"VIOLATIONS: {errs}")
    hold = solver.solve_from_storage(gw, a.horizon, model=a.model, max_transfers_total=0, **opts)
    print(f"gain vs no transfers over {a.horizon} GWs: {net_xp(plan) - net_xp(hold):+.2f} xP (net of hits)")

    if a.cross_check:
        ok, why = fplopt_wrapper.available()
        if not ok:
            print(f"\ncross-check unavailable: {why}")
            return 0
        xp = storage.read_table("xp", gw) if storage.exists("xp", gw) else storage.read_table("xp")
        t = time.time()
        try:
            ext = fplopt_wrapper.cross_check(xp, players, my_team, bank, ft, gw, a.horizon,
                                             model=a.model, secs=a.secs, **opts)
        except RuntimeError as e:
            print(f"\ncross-check failed: {e}")
            return 1
        show(f"FPL-Optimization-Tools ({time.time() - t:.0f}s)", ext, names, gw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
