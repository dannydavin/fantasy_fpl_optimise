"""Print the solver's transfer plan for the planning GW, optionally cross-checked externally.

    uv run python scripts/plan.py                          # 5-GW horizon from stored tables
    uv run python scripts/plan.py --horizon 6 --cross-check
    uv run python scripts/plan.py --chip 9:bboost --lock 411 --ban 165
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without the package installed
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fpl import storage  # noqa: E402
from fpl.optimise import fplopt_wrapper, solver  # noqa: E402


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


def net_xp(plan: solver.Plan) -> float:
    return float(plan.summary["xp_total"].sum() - plan.summary["hit_cost"].sum())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gw", type=int, default=None)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--model", default="baseline")
    p.add_argument("--lock", type=int, nargs="*", default=[])
    p.add_argument("--ban", type=int, nargs="*", default=[])
    p.add_argument("--chip", action="append", default=[], help="GW:chip, e.g. 9:bboost")
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

    opts = dict(chips=chips or None, lock=a.lock, ban=a.ban)
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
