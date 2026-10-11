import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import components  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="FPL - Home", page_icon=":soccer:", layout="wide")
components.reload_button()

CHIP_NAMES = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost",
              "3xc": "Triple Captain", "manager": "Assistant Manager"}

entry = components.load("my_entry")
if entry.empty:
    st.title("FPL Toolkit")
    st.info("No data yet. Run `uv run python scripts/update.py` then press Reload data.")
    st.stop()

e = entry.iloc[0]
gw = int(e["gw"])
events = components.load("events")
history = components.load("my_history")
sq = components.my_squad()

deadline = None
if not events.empty:
    row = events[events["gw"] == gw]
    if not row.empty:
        deadline = pd.Timestamp(row["deadline_time"].iloc[0])


def countdown(ts: pd.Timestamp) -> str:
    left = ts - pd.Timestamp.now(tz="UTC")
    if left.total_seconds() <= 0:
        return "deadline passed"
    d, h, m = left.days, left.seconds // 3600, left.seconds % 3600 // 60
    return f"{d}d {h}h {m}m to go" if d else f"{h}h {m}m to go"


chips = [x for x in str(e["chips_available"] or "").split(",") if x]
subtitle = f"Gameweek {gw}"
if deadline is not None:
    subtitle += f" · deadline {deadline.strftime('%a %d %b %H:%M')} UTC · {countdown(deadline)}"
components.hero(str(e["entry_name"]), subtitle,
                [f"Chip: {CHIP_NAMES.get(c, c)}" for c in chips] or ["No chips left"])

# ---- KPI row
last = prev = None
if not history.empty:
    h = history.sort_values("gw")
    last = h.iloc[-1]
    prev = h.iloc[-2] if len(h) > 1 else None
avg = None
if last is not None and not events.empty:
    m = events[events["gw"] == int(last["gw"])]
    avg = int(m["average_entry_score"].iloc[0]) if not m.empty else None

c = st.columns(5)
rank = e["overall_rank"]
rank_delta = None
if prev is not None and pd.notna(prev["overall_rank"]) and pd.notna(rank):
    moved = int(prev["overall_rank"]) - int(rank)
    rank_delta = f"{moved:+,} places"
c[0].metric("Overall rank", f"{int(rank):,}" if pd.notna(rank) else "-", rank_delta,
            border=True)
c[1].metric("Total points", int(e["total_points"]),
            f"{int(last['points'])} last GW" + (f" (avg {avg})" if avg is not None else "")
            if last is not None else None,
            delta_color="off", border=True)
c[2].metric("Bank", f"£{float(e['bank']):.1f}m", border=True)
c[3].metric("Free transfers", int(e["free_transfers"]), border=True)
c[4].metric("Squad value", f"£{float(e['value']):.1f}m", border=True)

# ---- season charts
if not history.empty:
    h = history.sort_values("gw")
    if not events.empty:
        h = h.merge(events[["gw", "average_entry_score"]], on="gw", how="left")
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Points per GW vs average**")
        fig = go.Figure()
        fig.add_bar(x=h["gw"], y=h["points"], name="Me",
                    hovertemplate="GW%{x}: %{y} pts<extra></extra>")
        if "average_entry_score" in h:
            fig.add_scatter(x=h["gw"], y=h["average_entry_score"], name="Average",
                            mode="lines+markers", line={"width": 2, "dash": "dot"},
                            hovertemplate="GW%{x} avg: %{y}<extra></extra>")
        fig.update_layout(height=280, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                          xaxis={"title": "GW", "dtick": 1}, yaxis_title="Points",
                          legend={"orientation": "h", "y": 1.12}, bargap=0.35)
        st.plotly_chart(fig, width="stretch")
    with right.container(border=True):
        st.markdown("**Overall rank** (up is better)")
        fig = go.Figure(go.Scatter(
            x=h["gw"], y=h["overall_rank"], mode="lines+markers", line={"width": 2},
            marker={"size": 8},
            hovertemplate="GW%{x}: %{y:,}<extra></extra>"))
        fig.update_layout(height=280, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                          xaxis={"title": "GW", "dtick": 1,
                                 "range": [h["gw"].min() - 0.5, h["gw"].max() + 0.5]},
                          yaxis={"autorange": "reversed", "tickformat": "~s"})
        st.plotly_chart(fig, width="stretch")

# ---- this week
if sq.empty:
    st.info("No squad data (my_team) yet.")
else:
    xp_col = f"xp_gw{gw}"
    has_xp = xp_col in sq.columns and sq[xp_col].notna().any()
    a, b, cc = st.columns([1, 1.4, 1])
    with a.container(border=True, height="stretch"):
        st.markdown(f"**Captaincy GW{gw}**")
        cap = sq.loc[sq["is_captain"].astype(bool), "web_name"]
        st.metric("Current captain", cap.iloc[0] if not cap.empty else "-")
        if has_xp:
            top = sq[sq["slot"] <= 11].nlargest(2, xp_col)
            st.metric("Model's pick", str(top.iloc[0]["web_name"]),
                      f"{top.iloc[0][xp_col]:.1f} xP", delta_color="off")
            if len(top) > 1:
                st.caption(f"Vice: {top.iloc[1]['web_name']} ({top.iloc[1][xp_col]:.1f} xP)")
    with b.container(border=True, height="stretch"):
        st.markdown("**Best in my squad, next 6 GWs**")
        if sq["xp_next6"].notna().any():
            top = sq.nlargest(6, "xp_next6").iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["xp_next6"], y=top["web_name"] + " · " + top["position"], orientation="h",
                text=[f"{v:.1f}" for v in top["xp_next6"]], textposition="outside", cliponaxis=False,
                customdata=top[["team_short", "position"]],
                hovertemplate="%{y} (%{customdata[0]} %{customdata[1]}): %{x:.1f} xP"
                              "<extra></extra>"))
            fig.update_layout(height=230, margin={"l": 0, "r": 30, "t": 0, "b": 0},
                              xaxis={"visible": False, "range": [0, top["xp_next6"].max() * 1.15]},
                              bargap=0.3)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No xP projections yet.")
    with cc.container(border=True, height="stretch"):
        flagged = sq[(sq["status"] != "a")
                     | sq["chance_of_playing_next_round"].lt(100).fillna(False)]
        st.markdown(f"**Flagged players: {len(flagged)}**")
        if flagged.empty:
            st.success("Everyone fit.")
        for r in flagged.itertuples():
            cop = r.chance_of_playing_next_round
            pct = f" · {int(cop)}%" if pd.notna(cop) else ""
            icon = ":material/warning:" if r.status == "d" else ":material/block:"
            st.markdown(f"{icon} **{r.web_name}** ({r.team_short}){pct}  \n"
                        f":gray[{r.news or components.STATUS_TEXT.get(r.status, '')}]")

# ---- navigation
st.markdown("#### Go to")
pages = [("pages/1_My_Team.py", "My Team", ":material/groups:"),
         ("pages/5_Transfer_Planner.py", "Transfers", ":material/swap_horiz:"),
         ("pages/2_Top_10k.py", "Top 10k", ":material/leaderboard:"),
         ("pages/3_Players.py", "Players", ":material/person_search:"),
         ("pages/4_Fixtures.py", "Fixtures", ":material/calendar_month:"),
         ("pages/6_News.py", "News", ":material/newspaper:")]
for col, (path, label, icon) in zip(st.columns(len(pages)), pages, strict=True):
    with col.container(border=True):
        st.page_link(path, label=label, icon=icon)
