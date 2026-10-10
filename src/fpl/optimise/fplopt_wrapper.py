"""Cross-check our solver against sertalpbilal/FPL-Optimization-Tools on the SAME xP.

The tool needs Python >= 3.14 (we run on 3.12) and writes into its own ``data/`` dir, so we
never import or modify ``external/``. Instead :func:`export_inputs` copies the tool's source
(paths.py, utils.py, dev/, run/, data/comprehensive_settings.json) into a scratch work dir,
adds our xP as ``data/ourxp.csv`` (columns ``ID, Name, Pos, Value, <gw>_Pts, <gw>_xMins``),
my squad as ``data/team.json`` (``team_data: json``; sell prices in tenths, bank, FTs) and
``data/user_settings.json``; :func:`run_tool` executes::

    uv run --python 3.14 --no-project --with pandas --with numpy --with requests \\
        --with fuzzywuzzy --with matplotlib --with highspy --with tabulate \\
        python run/solve.py          # cwd = work dir, PYTHONPATH = work dir

(the tool itself still reads bootstrap-static / fixtures from the live FPL API.)
:func:`parse_results` turns its ``data/results/*.csv`` into our :class:`~fpl.optimise.solver.Plan`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from fpl.optimise.solver import (
    LINEUP_COLS,
    SUMMARY_COLS,
    TRANSFER_COLS,
    Plan,
    norm_chip,
    simulate_free_transfers,
)
from fpl.schemas import HIT_COST

TOOL_REL = Path("external") / "FPL-Optimization-Tools"
CHIP_KEY = {"wildcard": "use_wc", "freehit": "use_fh", "bboost": "use_bb", "3xc": "use_tc"}
CHIP_TEXT = {"WC": "wildcard", "FH": "freehit", "BB": "bboost", "TC": "3xc", "": None}
PY_VERSION = "3.14"
TOOL_DEPS = ["pandas", "numpy", "requests", "fuzzywuzzy", "matplotlib", "highspy", "tabulate"]
POS_SHORT = {"GKP": "G", "DEF": "D", "MID": "M", "FWD": "F"}
POS_LONG = {v: k for k, v in POS_SHORT.items()}


def find_tool_dir() -> Path | None:
    """Locate the tool: $FPLOPT_DIR, else ``external/FPL-Optimization-Tools`` in this repo
    or any ancestor directory (covers git worktrees whose submodule is not checked out)."""
    env = os.environ.get("FPLOPT_DIR")
    cands = [Path(env)] if env else []
    cands += [p / TOOL_REL for p in Path(__file__).resolve().parents]
    return next((c for c in cands if (c / "run" / "solve.py").exists()), None)


def available() -> tuple[bool, str]:
    """(ok, reason). Needs the tool sources, ``uv`` and a Python 3.14 interpreter."""
    if find_tool_dir() is None:
        return False, "external/FPL-Optimization-Tools not found (git submodule update --init)"
    uv = shutil.which("uv")
    if uv is None:
        return False, "uv not on PATH"
    r = subprocess.run([uv, "python", "find", PY_VERSION], capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"Python {PY_VERSION} not available (uv python install {PY_VERSION})"
    return True, ""


# ---- inputs ---------------------------------------------------------------


def build_projection_csv(xp: pd.DataFrame, players: pd.DataFrame, gws: list[int],
                         model: str | None = None) -> pd.DataFrame:
    """Our xP -> the tool's wide projection table (``ID, Name, Pos, Value, <gw>_Pts, <gw>_xMins``)."""
    x = xp
    if "model" in x.columns and x["model"].nunique(dropna=True) > 1:
        model = model or ("baseline" if "baseline" in set(x["model"]) else x["model"].iloc[0])
        x = x[x["model"] == model]
    x = x[x["gw"].isin(gws)]
    pts = x.pivot_table(index="player_id", columns="gw", values="xp", aggfunc="sum")
    mins = (x.pivot_table(index="player_id", columns="gw", values="xmins", aggfunc="sum")
            if "xmins" in x.columns else None)
    out = players[["player_id", "web_name", "position", "price"]].rename(
        columns={"player_id": "ID", "web_name": "Name", "position": "Pos", "price": "Value"})
    out = out.drop_duplicates("ID").reset_index(drop=True)
    out["Pos"] = out["Pos"].map(POS_SHORT)
    for g in gws:
        p = pts[g] if g in pts.columns else pd.Series(dtype=float)
        out[f"{g}_Pts"] = out["ID"].map(p).fillna(0.0).round(4)
        if mins is not None and g in mins.columns:
            m = out["ID"].map(mins[g])
            m = m.where(m.notna(), 90.0 * (out[f"{g}_Pts"] > 0))
        else:
            m = pd.Series(90.0, index=out.index).where(out[f"{g}_Pts"] > 0, 0.0)
        out[f"{g}_xMins"] = m.astype(float).round(1)
    return out


def build_team_json(my_team: pd.DataFrame, players: pd.DataFrame, bank: float, ft: int) -> dict:
    """my_team (+ sell prices) -> the tool's ``team.json`` (prices in tenths)."""
    price = players.drop_duplicates("player_id").set_index("player_id")["price"]
    picks = []
    for r in my_team.sort_values("slot").itertuples():
        sell = getattr(r, "sell_price", None)
        sell = price[r.player_id] if sell is None or pd.isna(sell) else sell
        buy = getattr(r, "purchase_price", None)
        buy = sell if buy is None or pd.isna(buy) else buy
        picks.append({"element": int(r.player_id), "position": int(r.slot),
                      "selling_price": int(round(sell * 10)),
                      "purchase_price": int(round(buy * 10)), "multiplier": 1,
                      "is_captain": False, "is_vice_captain": False})
    return {"picks": picks, "chips": [],
            "transfers": {"bank": int(round(bank * 10)), "limit": int(ft), "made": 0,
                          "cost": HIT_COST, "value": 0}}


def build_settings(gw: int, horizon: int, chips: dict[int, str] | None, lock, ban,
                   decay: float = 1.0, secs: int = 120, **extra) -> dict:
    s = {
        "datasource": "ourxp", "team_data": "json", "team_id": None, "horizon": int(horizon),
        "override_next_gw": int(gw), "no_transfer_last_gws": 0, "decay_base": decay,
        "xmin_lb": 0, "ev_per_price_cutoff": 0, "keep_top_ev_percent": 100,
        "banned": [int(p) for p in ban], "locked": [int(p) for p in lock],
        "use_wc": [], "use_bb": [], "use_fh": [], "use_tc": [], "secs": int(secs), "gap": 0.001,
        "verbose": False, "export_image": 0, "print_squads": False,
        "print_result_table": True, "print_transfer_chip_summary": True,
    }
    for g, c in (chips or {}).items():
        c = norm_chip(c)
        if c:
            s[CHIP_KEY[c]].append(int(g))
    s.update(extra)
    return s


def export_inputs(workdir: Path, xp: pd.DataFrame, players: pd.DataFrame, my_team: pd.DataFrame,
                  bank: float, ft: int, gw: int, horizon: int, *, chips=None, lock=(), ban=(),
                  model: str | None = None, decay: float = 1.0, secs: int = 120,
                  tool_dir: Path | None = None, **extra) -> dict:
    """Populate ``workdir`` with a runnable copy of the tool + our inputs; returns settings."""
    tool_dir = tool_dir or find_tool_dir()
    if tool_dir is None:
        raise FileNotFoundError("FPL-Optimization-Tools sources not found")
    workdir = Path(workdir)
    for rel in ("paths.py", "utils.py", "dev", "run", "data/comprehensive_settings.json"):
        src, dst = tool_dir / rel, workdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "tmp"))
        else:
            shutil.copy2(src, dst)
    (workdir / "data" / "results").mkdir(parents=True, exist_ok=True)
    gws = list(range(gw, gw + horizon))
    build_projection_csv(xp, players, gws, model).to_csv(workdir / "data" / "ourxp.csv",
                                                         index=False)
    (workdir / "data" / "team.json").write_text(
        json.dumps(build_team_json(my_team, players, bank, ft), indent=1))
    settings = build_settings(gw, horizon, chips, lock, ban, decay, secs, **extra)
    (workdir / "data" / "user_settings.json").write_text(json.dumps(settings, indent=1))
    return settings


def command() -> list[str]:
    cmd = ["uv", "run", "--python", PY_VERSION, "--no-project"]
    for d in TOOL_DEPS:
        cmd += ["--with", d]
    return [*cmd, "python", "run/solve.py"]


def run_tool(workdir: Path, timeout: float = 900) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(workdir), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(command(), cwd=workdir, env=env, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")


# ---- outputs --------------------------------------------------------------


def parse_results(path: Path | pd.DataFrame, bank: float | None = None,
                  free_transfers: int | None = None) -> Plan:
    """Tool result CSV (``picks`` frame) -> Plan."""
    p = pd.read_csv(path) if not isinstance(path, pd.DataFrame) else path
    p = p[p["squad"] > 0.5] if "squad" in p.columns else p
    allp = path if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    lines, trans, rows = [], [], []
    cash = bank
    for gw in sorted(allp["week"].unique()):
        g = allp[allp["week"] == gw]
        sq = g[g["squad"] > 0.5]
        chip = CHIP_TEXT.get(str(g["chip"].fillna("").iloc[0]))
        xi = sq[sq["lineup"] > 0.5].sort_values(["type", "xP"], ascending=[True, False])
        slot = {int(i): n for n, i in enumerate(xi["id"], 1)}
        bench = sq[sq["bench"] > -0.5].sort_values("bench")
        for n, i in enumerate(bench["id"]):
            slot[int(i)] = 12 + n
        for r in sq.itertuples():
            is_c = r.captain > 0.5
            lines.append({
                "gw": int(gw), "player_id": int(r.id), "web_name": r.name,
                "position": POS_LONG.get(r.pos, r.pos), "team_id": 0, "slot": slot[int(r.id)],
                "multiplier": int(r.multiplier), "is_captain": bool(is_c),
                "is_vice_captain": bool(r.vicecaptain > 0.5), "xp": float(r.xP),
                "in_xi": bool(r.lineup > 0.5)})
        outs, ins = g[g["transfer_out"] > 0.5], g[g["transfer_in"] > 0.5]
        for pos in (1, 2, 3, 4):
            for o, i in zip(outs[outs["type"] == pos].itertuples(),
                            ins[ins["type"] == pos].itertuples(), strict=False):
                trans.append({"gw": int(gw), "player_out": int(o.id), "player_in": int(i.id),
                              "out_price": float(o.sell_price), "in_price": float(i.buy_price)})
        t = len(ins)
        ft = int(round(g["ft"].iloc[0]))
        hits = 0 if chip in ("wildcard", "freehit") else max(0, t - ft)
        xi_xp = float((xi["xP"] * xi["multiplier"]).sum())
        tot = float((sq["xP"] * sq["multiplier"]).sum())
        if cash is not None:
            cash += float(outs["sell_price"].sum() - ins["buy_price"].sum())
        rows.append({"gw": int(gw), "ft_available": ft, "transfers": t, "hits": hits,
                     "hit_cost": hits * HIT_COST, "chip": chip, "xp_xi": xi_xp, "xp_total": tot,
                     "bank": None if cash is None else round(cash, 4)})
    summary = pd.DataFrame(rows, columns=SUMMARY_COLS)
    if free_transfers is not None and len(summary):
        simulate_free_transfers(free_transfers, [int(t) for t in summary["transfers"]],
                                list(summary["chip"]))
    return Plan("Optimal", float(summary["xp_total"].sum() - summary["hit_cost"].sum()),
                pd.DataFrame(trans, columns=TRANSFER_COLS),
                pd.DataFrame(lines, columns=LINEUP_COLS), summary, params={"source": "fplopt"})


def cross_check(xp: pd.DataFrame, players: pd.DataFrame, my_team: pd.DataFrame, bank: float,
                free_transfers: int, gw: int, horizon: int, *, chips=None, lock=(), ban=(),
                model: str | None = None, decay: float = 1.0, secs: int = 120,
                workdir: Path | None = None) -> Plan:
    """Run the external tool on our xP and return its plan (raises RuntimeError on failure)."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    work = Path(workdir or tempfile.mkdtemp(prefix="fplopt_"))
    export_inputs(work, xp, players, my_team, bank, free_transfers, gw, horizon, chips=chips,
                  lock=lock, ban=ban, model=model, decay=decay, secs=secs)
    r = run_tool(work, timeout=secs + 600)
    results = sorted((work / "data" / "results").glob("*.csv"))
    if r.returncode != 0 or not results:
        raise RuntimeError(f"FPL-Optimization-Tools failed (rc={r.returncode}): "
                           f"{(r.stderr or r.stdout)[-1500:]}")
    plan = parse_results(results[-1], bank, free_transfers)
    plan.params["workdir"] = str(work)
    plan.params["stdout"] = r.stdout
    return plan
