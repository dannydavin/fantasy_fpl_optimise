"""Top 10k: ownership, EO and rank-risk versus my squad."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import components  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from fpl import storage  # noqa: E402
from fpl.features import ownership as own  # noqa: E402

HINT = "Run `uv run python scripts/update.py --top10k --limit 500` to pull top-10k picks."
OWN_COLS = ["owned_pct", "start_pct", "captain_pct", "eo"]


def squad_ownership(squad: pd.DataFrame, ownership: pd.DataFrame) -> pd.DataFrame:
    """My 15 with top-10k ownership columns (absent players -> 0%) and exposure."""
    cols = ["player_id", *OWN_COLS, "delta_vs_overall", "delta_vs_prev_gw"]
    o = ownership[[c for c in cols if c in ownership.columns]]
    out = squad.merge(o, on="player_id", how="left")
    for c in cols[1:]:
        if c not in out.columns:
            out[c] = float("nan")
    out[OWN_COLS] = out[OWN_COLS].fillna(0.0)
    out["exposure"] = out["multiplier"] * 100 - out["eo"]
    return out


def sample_size(entries: pd.DataFrame, picks: pd.DataFrame) -> tuple[int, int | None]:
    n = int(picks["entry_id"].nunique())
    limit = None
    if not entries.empty and "limit" in entries.columns and entries["limit"].notna().any():
        limit = int(entries["limit"].dropna().iloc[0])
    return n, limit


def _names(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    p = players[["player_id", "web_name", "team_short", "position", "price"]]
    return df.merge(p, on="player_id", how="left")


def main() -> None:
    st.set_page_config(page_title="Top 10k", layout="wide")
    components.reload_button()
    st.title("Top 10k")
    gws = storage.available_gws("top10k_picks")
    if not gws:
        st.info(f"No top-10k data yet. {HINT}")
        return
    gw = st.sidebar.selectbox("Picks GW", sorted(gws, reverse=True)) if len(gws) > 1 else gws[0]
    picks = components.load("top10k_picks", gw)
    entries = components.load("top10k_entries", gw)
    players = components.load("players")
    squad = components.my_squad()
    ownership = components.load("ownership", gw) if components.has("ownership", gw) else None
    if ownership is None or ownership.empty:
        ownership = own.add_overall(own.compute_ownership(picks, gw), players)
        ownership["delta_vs_prev_gw"] = float("nan")
    n, limit = sample_size(entries, picks)
    plan_gw = int(squad["gw"].iloc[0]) if not squad.empty else None
    st.caption(
        f"Picks GW{gw} | {n} entries sampled"
        + (f" (limit {limit})" if limit else "")
        + (f" | my squad is for planning GW{plan_gw}" if plan_gw else "")
    )
    if squad.empty or players.empty:
        st.info("No snapshot of my team; run `uv run python scripts/update.py`.")
        return

    mine = squad_ownership(squad, ownership)
    my_cap = mine.loc[mine["is_captain"].astype(bool), "web_name"]

    st.header("My squad vs top 10k")
    delta_cols = ["exposure", "delta_vs_overall", "delta_vs_prev_gw"]
    show = mine[["slot", "web_name", "team_short", "position", "multiplier", *OWN_COLS, *delta_cols]]
    pct = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: pct for c in [*OWN_COLS, *delta_cols]})

    st.header("Rank risk")
    st.caption("Negative = short a highly owned player (risk); positive = differential.")
    exp = _names(own.my_exposure(ownership, squad), players)
    top = pd.concat([exp[exp["owned"]], exp[~exp["owned"]].head(10)]).sort_values("exposure")
    top["side"] = top["exposure"].map(lambda v: "differential" if v > 0 else "risk")
    fig = px.bar(top, x="exposure", y="web_name", orientation="h", color="side",
                 color_discrete_map={"risk": "#d62728", "differential": "#2ca02c"},
                 hover_data=["eo", "my_mult"])
    fig.update_layout(height=max(350, 24 * len(top)), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

    st.header("High-EO players I don't own")
    min_eo = st.slider("Min EO %", 0, 100, 20)
    hi = _names(own.high_eo_not_owned(ownership, squad, float(min_eo)), players)
    if hi.empty:
        st.success("None above that EO.")
    else:
        st.dataframe(hi[["web_name", "team_short", "position", "price", *OWN_COLS]],
                     hide_index=True, width="stretch",
                     column_config={"price": st.column_config.NumberColumn(format="£%.1fm")})

    st.header("Captaincy")
    cap = _names(own.captaincy(picks), players).head(10)
    if my_cap.size:
        st.caption(f"My captain: {my_cap.iloc[0]}")
    figc = px.bar(cap.melt(id_vars="web_name", value_vars=["captain_pct", "vice_pct"]),
                  x="web_name", y="value", color="variable", barmode="group")
    figc.update_layout(xaxis_title=None, yaxis_title="% of entries")
    st.plotly_chart(figc, width="stretch")

    st.header("Ownership deltas")
    d = _names(ownership, players)
    for col, label in (("delta_vs_prev_gw", "vs previous GW"), ("delta_vs_overall", "vs overall")):
        dd = d.dropna(subset=[col])
        st.subheader(label)
        if dd.empty:
            st.info("No data for this comparison.")
            continue
        a, b = st.columns(2)
        cfg = {col: st.column_config.NumberColumn(format="%+.1f")}
        view = ["web_name", "team_short", "owned_pct", col]
        a.caption("Risers")
        a.dataframe(dd.nlargest(10, col)[view], hide_index=True, column_config=cfg)
        b.caption("Fallers")
        b.dataframe(dd.nsmallest(10, col)[view], hide_index=True, column_config=cfg)
    trend = own.ownership_trend()
    if not trend.empty and trend["gw"].nunique() >= 2:
        labels = dict(zip(players["player_id"], players["web_name"], strict=False))
        default = squad["player_id"].head(3).tolist()
        chosen = st.multiselect("Ownership trend for", sorted(trend["player_id"].unique()),
                                default=default, format_func=lambda i: labels.get(i, str(i)))
        t = trend[trend["player_id"].isin(chosen)].assign(
            player=lambda x: x["player_id"].map(labels))
        st.plotly_chart(px.line(t, x="gw", y="owned_pct", color="player", markers=True),
                        width="stretch")
    else:
        st.info("Ownership trend needs at least two GWs of ownership data.")

    st.header("Chip usage")
    chips = own.chip_usage(picks)
    a, b = st.columns(2)
    a.dataframe(chips, hide_index=True, column_config={"pct": pct})
    b.plotly_chart(px.bar(chips, x="chip", y="pct"), width="stretch")

    st.header("Template XI")
    tx = own.template_xi(ownership, players)
    mine_ids = set(squad["player_id"])
    st.caption(f"Formation {tx['formation'] or 'n/a'}; green = in my squad")
    for title, part in (("XI", tx["xi"]), ("Bench", tx["bench"])):
        if part.empty:
            continue
        t = _names(part[["player_id", "owned_pct", "start_pct", "eo"]], players)
        t["mine"] = t["player_id"].isin(mine_ids)
        st.subheader(title)
        view = t[["mine", "web_name", "team_short", "position", "owned_pct", "start_pct", "eo"]]
        st.dataframe(
            view.style.apply(
                lambda r: ["background-color: #2ca02c55" if r["mine"] else "" for _ in r], axis=1),
            hide_index=True, width="stretch")

    st.header("Rank bands")
    bands = own.rank_bands(picks)
    if bands.empty:
        st.info("No rank data.")
    else:
        mine_set = set(squad["player_id"])
        ids = mine_set | set(ownership.nlargest(8, "eo")["player_id"])
        b = _names(bands[bands["player_id"].isin(ids)], players)
        b["band"] = "Top " + b["band"].astype(str)
        if st.checkbox("My players only", value=False):
            b = b[b["player_id"].isin(mine_set)]
        st.plotly_chart(px.bar(b, x="web_name", y="eo", color="band", barmode="group"),
                        width="stretch")


if __name__ == "__main__":
    main()
