import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import components  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

components.page_setup("Players", ":bar_chart:")

players = components.load("players")
form = components.load("form")
if players.empty:
    st.info("No players table yet. Run scripts/update.py.")
    st.stop()
if form.empty:
    st.info("No form table yet; showing basic player data only.")

window = st.sidebar.radio("Form window (GWs)", [4, 6], index=1)
f = form[form["window"] == window] if not form.empty else form
fcols = [c for c in (
    "player_id", "minutes", "start_pct", "minutes_share", "xg_per90", "xa_per90", "xgi_per90",
    "dc_per90", "points_per90", "gws_in_window",
) if c in f.columns]
df = players.merge(
    f[fcols].rename(columns={"minutes": "win_minutes"}) if not f.empty else
    pd.DataFrame(columns=["player_id"]),
    on="player_id", how="left",
)

xp = components.load("xp")
if not xp.empty:
    gw0 = components.planning_gw() or int(xp["gw"].min())
    x = xp[(xp["gw"] >= gw0) & (xp["gw"] < gw0 + 6)]
    if x["model"].nunique() > 1:
        x = x[x["model"] == x["model"].iloc[0]]
    df = df.merge(x.groupby("player_id")["xp"].sum().rename("xp_next6"), on="player_id",
                  how="left")
else:
    df["xp_next6"] = float("nan")

squad_ids = set(components.my_squad()["player_id"]) if components.has("my_team") else set()
df["mine"] = df["player_id"].isin(squad_ids)

# ---- filters
st.sidebar.header("Filters")
pos = st.sidebar.multiselect("Position", components.POSITION_ORDER, default=components.POSITION_ORDER)
teams = st.sidebar.multiselect("Team", sorted(df["team_short"].dropna().unique()))
pmin, pmax = float(df["price"].min()), float(df["price"].max())
price = st.sidebar.slider("Price (£m)", pmin, pmax, (pmin, pmax), step=0.1)
max_min = int(df["win_minutes"].max()) if df["win_minutes"].notna().any() else 0
min_minutes = st.sidebar.slider(f"Min minutes (last {window} GWs)", 0, max(max_min, 1), 0)
status_opts = sorted(df["status"].dropna().astype(str).unique())
status = st.sidebar.multiselect("Status", status_opts, default=status_opts)
only_mine = st.sidebar.checkbox("My squad only")

m = df["position"].isin(pos) & df["price"].between(*price) & df["status"].astype(str).isin(status)
if teams:
    m &= df["team_short"].isin(teams)
if "win_minutes" in df.columns:
    m &= df["win_minutes"].fillna(0) >= min_minutes
if only_mine:
    m &= df["mine"]
d = df[m].copy()
st.caption(f"{len(d)} players")

# ---- scatter
if "xgi_per90" in d.columns and d["xgi_per90"].notna().any():
    sc = d.dropna(subset=["xgi_per90"]).copy()
    sc["Squad"] = sc["mine"].map({True: "Mine", False: ""})
    fig = px.scatter(
        sc, x="price", y="xgi_per90", color="position",
        color_discrete_map=components.POSITION_COLOURS,
        category_orders={"position": components.POSITION_ORDER},
        hover_name="web_name",
        hover_data={"team_short": True, "price": ":.1f", "xgi_per90": ":.2f",
                    "win_minutes": True, "position": False},
        labels={"price": "Price (£m)", "xgi_per90": f"xGI/90 (last {window} GWs)"},
    )
    mine = sc[sc["mine"]]
    if not mine.empty:
        fig.add_scatter(
            x=mine["price"], y=mine["xgi_per90"], mode="markers+text", text=mine["web_name"],
            textposition="top center", name="My squad", hoverinfo="skip",
            marker=dict(size=14, symbol="circle-open", line=dict(width=2, color="black")),
        )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

# ---- table
cols = ["web_name", "team_short", "position", "price", "status", "selected_by_percent",
        "xp_next6", "start_pct", "minutes_share", "win_minutes", "xg_per90", "xa_per90",
        "xgi_per90", "dc_per90", "points_per90", "penalties_order",
        "corners_and_indirect_freekicks_order", "direct_freekicks_order", "mine"]
cols = [c for c in cols if c in d.columns]
d["mine"] = d["mine"].map({True: "★", False: ""})
d = d.sort_values("xp_next6", ascending=False, na_position="last")
st.dataframe(
    d[cols],
    hide_index=True,
    height=600,
    column_config={
        "web_name": "Player", "team_short": "Team", "position": "Pos",
        "price": st.column_config.NumberColumn("£", format="%.1f"),
        "selected_by_percent": st.column_config.NumberColumn("Sel %", format="%.1f"),
        "xp_next6": st.column_config.NumberColumn("xP next 6", format="%.1f"),
        "start_pct": st.column_config.NumberColumn("Start %", format="%.0f"),
        "minutes_share": st.column_config.NumberColumn("Min share", format="%.2f"),
        "win_minutes": st.column_config.NumberColumn("Mins", format="%d"),
        "xg_per90": st.column_config.NumberColumn("xG/90", format="%.2f"),
        "xa_per90": st.column_config.NumberColumn("xA/90", format="%.2f"),
        "xgi_per90": st.column_config.NumberColumn("xGI/90", format="%.2f"),
        "dc_per90": st.column_config.NumberColumn("DC/90", format="%.2f"),
        "points_per90": st.column_config.NumberColumn("Pts/90", format="%.1f"),
        "penalties_order": st.column_config.NumberColumn("Pen", format="%d"),
        "corners_and_indirect_freekicks_order": st.column_config.NumberColumn("Corners", format="%d"),
        "direct_freekicks_order": st.column_config.NumberColumn("DFK", format="%d"),
        "mine": "Mine",
    },
)
