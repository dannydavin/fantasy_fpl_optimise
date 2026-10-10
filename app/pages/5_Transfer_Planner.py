import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import components  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fpl.optimise.solver import DEFAULT_BENCH_WEIGHTS, solve, validate_plan  # noqa: E402
from fpl.schemas import HIT_COST  # noqa: E402

components.page_setup("Transfer Planner", ":arrows_counterclockwise:")

players = components.load("players")
my_team = components.load("my_team")
my_entry = components.load("my_entry")
xp_all = components.load("xp")
if players.empty or my_team.empty or my_entry.empty or xp_all.empty:
    st.info("Missing data (players / my_team / my_entry / xp). "
            "Run `uv run python scripts/update.py` first.")
    st.stop()

entry = my_entry.iloc[0]
gw0 = int(my_team["gw"].iloc[0])
labels = components.player_labels()
shorts = components.team_shorts()
chips_have = [c for c in str(entry.get("chips_available") or "").split(",") if c]

# ---- controls
with st.sidebar:
    st.header("Plan settings")
    horizon = st.slider("Horizon (GWs)", 1, 6, 1, key="horizon")
    models = sorted(xp_all["model"].dropna().unique()) if "model" in xp_all.columns else []
    model = st.selectbox("xP model", models, index=0, key="model") if models else None
    upload = st.file_uploader("...or upload xP CSV (gw, player_id, xp)", type="csv")
    bank = st.number_input("Bank (£m)", 0.0, 100.0, float(entry["bank"]), 0.1, key="bank")
    ft = st.number_input("Free transfers", 0, 5, int(entry["free_transfers"]), key="ft")
    hit_cost = st.number_input("Hit cost", 0.0, 12.0, float(HIT_COST), 0.5, key="hit_cost")
    max_hits = st.number_input("Max hits per GW (-1 = unlimited)", -1, 5, -1, key="max_hits")
    decay = st.slider("GW decay", 0.5, 1.0, 1.0, 0.01, key="decay")
    ft_value = st.number_input("Value of a banked FT (xP)", 0.0, 5.0, 1.5, 0.1, key="ft_value")
    vice_weight = st.number_input("Vice-captain weight", 0.0, 1.0, 0.05, 0.01, key="vice_w")
    with st.expander("Bench weights"):
        bw = [st.number_input(n, 0.0, 1.0, float(v), 0.005, format="%.3f", key=f"bw{i}")
              for i, (n, v) in enumerate(zip(["Bench GK", "Bench 1", "Bench 2", "Bench 3"],
                                             DEFAULT_BENCH_WEIGHTS, strict=True))]
    time_limit = st.number_input("Time limit (s)", 5, 600, 60, key="time_limit")

squad_ids = [int(p) for p in my_team["player_id"]]
all_ids = sorted(int(p) for p in players["player_id"])
fmt = lambda p: labels.get(int(p), f"#{p}")  # noqa: E731
c1, c2 = st.columns(2)
lock = c1.multiselect("Lock (keep in squad)", all_ids, default=[], format_func=fmt, key="lock")
ban = c2.multiselect("Ban (never in squad)", all_ids, default=[], format_func=fmt, key="ban")

st.subheader("Chips")
chip_opts = list(chips_have)
if st.checkbox("Wildcard available", value=False, key="wc_avail") and "wildcard" not in chip_opts:
    chip_opts.append("wildcard")
chip_plan: dict[int, str] = {}
if chip_opts:
    cols = st.columns(horizon)
    for i in range(horizon):
        sel = cols[i].selectbox(f"GW{gw0 + i}", ["none", *chip_opts], key=f"chip{i}")
        if sel != "none":
            chip_plan[gw0 + i] = sel
else:
    st.caption("No chips available.")
chip_dupes = len(set(chip_plan.values())) != len(chip_plan)
if chip_dupes:
    st.error("Each chip can only be played once.")

# ---- xP source
xp = xp_all
if upload is not None:
    try:
        xp = pd.read_csv(upload)
        assert {"gw", "player_id", "xp"} <= set(xp.columns)
        model = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read uploaded xP CSV: {exc}")
        xp = xp_all
elif model is not None:
    xp = xp_all[xp_all["model"] == model]


def run_solve():
    return solve(
        xp, players, my_team[["player_id", "sell_price"]], float(bank), int(ft), gw0, int(horizon),
        model=model, chips=chip_plan, lock=lock, ban=ban, decay=float(decay),
        ft_value=float(ft_value), hit_cost=float(hit_cost), vice_weight=float(vice_weight),
        bench_weights=tuple(bw), max_hits_per_gw=None if max_hits < 0 else int(max_hits),
        time_limit=float(time_limit),
    )


b1, b2 = st.columns(2)
if b1.button("Solve", type="primary", disabled=chip_dupes):
    try:
        with st.spinner("Solving..."):
            st.session_state["plan"] = run_solve()
            st.session_state["plan_args"] = (float(bank), int(ft))
    except ValueError as exc:
        st.session_state.pop("plan", None)
        st.error(str(exc))

if b2.button("Cross-check with FPL-Optimization-Tools"):
    try:
        from fpl.optimise import fplopt_wrapper

        ok, why = fplopt_wrapper.available()
        if not ok:
            st.warning(f"FPL-Optimization-Tools unavailable: {why}")
        else:
            with st.spinner("Running FPL-Optimization-Tools..."):
                st.session_state["ext_plan"] = fplopt_wrapper.cross_check(
                    xp, players, my_team, float(bank), int(ft), gw0, int(horizon),
                    chips=chip_plan)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Cross-check failed: {exc}")


def pl_name(pid: int) -> str:
    return labels.get(int(pid), f"#{pid}")


def show_plan(plan, title: str, check: bool = True) -> None:
    st.subheader(title)
    if not plan.ok:
        st.error(f"Solver status: {plan.status}")
        return
    s = plan.summary.copy()
    net = float(s["xp_total"].sum() - s["hit_cost"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("xP total (after hits)", f"{net:.1f}")
    m2.metric("Hits", int(s["hits"].sum()))
    m3.metric("Transfers", int(s["transfers"].sum()))
    st.dataframe(s.round(2), hide_index=True, use_container_width=True)
    if check:
        errs = validate_plan(plan, players, my_team, *st.session_state["plan_args"])
        if errs:
            st.error("Plan fails validation:\n- " + "\n- ".join(errs))
        else:
            st.success("validate_plan: plan is legal.")
    for r in s.itertuples():
        gw = int(r.gw)
        chip_txt = f" [{r.chip}]" if isinstance(r.chip, str) else ""
        with st.expander(f"GW{gw}{chip_txt}: {r.xp_total:.1f} xP, {r.ft_available} FT available",
                         expanded=gw == gw0):
            t = plan.transfers[plan.transfers["gw"] == gw]
            if t.empty:
                st.write("No transfers.")
            else:
                st.dataframe(pd.DataFrame({
                    "Out": [pl_name(p) for p in t["player_out"]],
                    "Sell": [f"£{v:.1f}m" for v in t["out_price"]],
                    "In": [pl_name(p) for p in t["player_in"]],
                    "Buy": [f"£{v:.1f}m" for v in t["in_price"]],
                }), hide_index=True)
            st.caption(f"Hits: {r.hits} (-{r.hit_cost:.0f}); bank after: £{r.bank:.1f}m")
            lg = plan.lineups[plan.lineups["gw"] == gw].sort_values("slot")
            lg = lg.assign(
                Player=[f"{n} ({shorts.get(int(t_), '?')})"
                        + (" (C)" if c else " (VC)" if v else "")
                        for n, t_, c, v in zip(lg["web_name"], lg["team_id"], lg["is_captain"],
                                               lg["is_vice_captain"], strict=True)],
                Role=["XI" if x else "Bench" for x in lg["in_xi"]])
            st.dataframe(lg[["slot", "Role", "position", "Player", "multiplier", "xp"]]
                         .round(2), hide_index=True, use_container_width=True)


if "plan" in st.session_state:
    show_plan(st.session_state["plan"], "Plan")
if "ext_plan" in st.session_state:
    show_plan(st.session_state["ext_plan"], "FPL-Optimization-Tools plan", check=False)
