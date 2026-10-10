import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import components  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fpl.features.fixtures import blank_double_calendar  # noqa: E402

components.page_setup("Fixtures", ":calendar:")

ticker = components.load("fixture_ticker")
if ticker.empty:
    st.info("No fixture_ticker yet. Run scripts/update.py.")
    st.stop()

shorts = components.team_shorts()
first = int(ticker["gw"].min())
last = int(ticker["gw"].max())
n = st.slider("Gameweeks", 1, min(12, last - first + 1), min(6, last - first + 1))
metric = "fdr" if st.toggle("Use official FPL FDR instead of custom difficulty") else "difficulty"

cells = components.fixture_cells(ticker, shorts, first, n, metric)
cells["team"] = cells["team_id"].map(shorts)
avg = cells.groupby("team")["value"].mean().sort_values()
order = avg.index.tolist()  # easiest first

lab = cells.pivot(index="team", columns="gw", values="label").loc[order]
val = cells.pivot(index="team", columns="gw", values="value").loc[order]
cs = [[i / 4, c] for i, c in enumerate(components.DIFFICULTY_COLOURS)]
fig = go.Figure(
    go.Heatmap(
        z=val.fillna(0).values,
        x=[f"GW{g}" for g in val.columns],
        y=[f"{t} ({avg[t]:.2f})" for t in order],
        text=lab.values,
        texttemplate="%{text}",
        colorscale=cs,
        zmin=1,
        zmax=5,
        showscale=False,
        hoverinfo="text",
        xgap=2,
        ygap=2,
    )
)
# blank cells -> grey overlay
bl = (cells.pivot(index="team", columns="gw", values="n_fixtures").loc[order] == 0)
fig.add_trace(
    go.Heatmap(
        z=bl.astype(float).where(bl).values,
        x=[f"GW{g}" for g in val.columns],
        y=[f"{t} ({avg[t]:.2f})" for t in order],
        colorscale=[[0, components.BLANK_COLOUR], [1, components.BLANK_COLOUR]],
        showscale=False,
        hoverinfo="skip",
        xgap=2,
        ygap=2,
    )
)
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(side="top")
fig.update_layout(height=max(500, 28 * len(order) + 80), margin=dict(l=10, r=10, t=40, b=10))
st.subheader("Fixture difficulty (easiest first, avg in brackets)")
st.plotly_chart(fig, width="stretch")
st.caption("Grey = blank GW. DGW cells show both opponents; colour is the mean difficulty.")

st.subheader("Blank / double calendar")
fx = components.load("fixtures")
if fx.empty:
    st.info("No fixtures table.")
else:
    cal = blank_double_calendar(fx, gws=list(range(first, last + 1)))
    cal["team"] = cal["team_id"].map(shorts)
    grid = cal.pivot(index="team", columns="gw", values="n_fixtures")
    grid = grid.loc[sorted(grid.index)]
    fig2 = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[f"GW{g}" for g in grid.columns],
            y=list(grid.index),
            text=grid.values,
            texttemplate="%{text}",
            colorscale=[[0, "#d9534f"], [0.5, "#e6e6e6"], [1, "#2e9e5b"]],
            zmin=0,
            zmax=2,
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )
    fig2.update_yaxes(autorange="reversed")
    fig2.update_xaxes(side="top")
    fig2.update_layout(height=max(500, 24 * len(grid) + 80), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")
    odd = cal[cal["kind"] != "single"]
    odd = odd.assign(GW=odd["gw"])
    if odd.empty:
        st.write("No blank or double gameweeks scheduled.")
    else:
        summ = (
            odd.groupby(["gw", "kind"])["team"].apply(lambda s: ", ".join(sorted(s)))
            .reset_index().rename(columns={"gw": "GW", "team": "Teams"})
        )
        st.dataframe(summ, hide_index=True)
