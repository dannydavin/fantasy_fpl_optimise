import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import components  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

components.page_setup("FPL Toolkit", ":soccer:")

entry = components.load("my_entry")
if entry.empty:
    st.info("No data yet. Run `uv run python scripts/update.py` then press Reload data.")
    st.stop()

e = entry.iloc[0]
st.subheader(str(e["entry_name"]))

events = components.load("events")
deadline = None
if not events.empty:
    row = events[events["gw"] == int(e["gw"])]
    if not row.empty:
        deadline = pd.Timestamp(row["deadline_time"].iloc[0])

c = st.columns(5)
c[0].metric("Planning GW", int(e["gw"]))
c[1].metric("Deadline", deadline.strftime("%a %d %b %H:%M UTC") if deadline is not None else "-")
c[2].metric("Bank", f"£{float(e['bank']):.1f}m")
c[3].metric("Free transfers", int(e["free_transfers"]))
rank = e["overall_rank"]
c[4].metric("Overall rank", f"{int(rank):,}" if pd.notna(rank) else "-")
chips = [x for x in str(e["chips_available"] or "").split(",") if x]
st.write("**Chips available:** " + (", ".join(chips) if chips else "none"))
st.write(f"**Squad value:** £{float(e['value']):.1f}m  |  **Total points:** {int(e['total_points'])}")

sq = components.my_squad()
if sq.empty:
    st.info("No squad data (my_team) yet.")
else:
    st.subheader("Top xP picks in my squad (next 6 GWs)")
    if sq["xp_next6"].notna().any():
        top = sq.sort_values("xp_next6", ascending=False).head(5)
        st.dataframe(
            top[["web_name", "team_short", "position", "xp_next6"]],
            hide_index=True,
            column_config={"xp_next6": st.column_config.NumberColumn("xP next 6", format="%.1f")},
        )
    else:
        st.info("No xP projections yet.")
    flagged = sq[(sq["status"] != "a") | sq["chance_of_playing_next_round"].lt(100).fillna(False)]
    st.subheader(f"Flagged players: {len(flagged)}")
    if not flagged.empty:
        st.dataframe(
            flagged[["web_name", "team_short", "status", "chance_of_playing_next_round", "news"]],
            hide_index=True,
        )

st.caption("Pages: My Team, Top 10k, Players, Fixtures, Transfer Planner, News (sidebar).")
