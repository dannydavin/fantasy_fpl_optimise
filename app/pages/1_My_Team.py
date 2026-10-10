import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import components  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

components.page_setup("My Team", ":shirt:")

sq = components.my_squad()
if sq.empty:
    st.info("No squad data (my_team) yet. Run scripts/update.py.")
    st.stop()

entry = components.load("my_entry")
if not entry.empty:
    e = entry.iloc[0]
    c = st.columns(4)
    c[0].metric("Bank", f"£{float(e['bank']):.1f}m")
    c[1].metric("Free transfers", int(e["free_transfers"]))
    c[2].metric("Squad value", f"£{float(e['value']):.1f}m")
    c[3].metric("Chips", str(e["chips_available"] or "none").replace(",", ", "))

gw0 = int(sq["gw"].iloc[0])
gws = list(range(gw0, gw0 + 6))

PRICE_LABEL = {"rise": "Rise risk", "fall": "Fall risk", "stable": ""}
STATUS_LABEL = {"a": "", "d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable",
                "n": "Not in squad"}


def flag(r) -> str:
    out = []
    st_ = STATUS_LABEL.get(str(r.status), str(r.status))
    if st_:
        cop = r.chance_of_playing_next_round
        out.append(st_ + (f" ({int(cop)}%)" if pd.notna(cop) else ""))
    return ", ".join(out)


def role(r) -> str:
    return "C" if r.is_captain else "VC" if r.is_vice_captain else ""


df = sq.copy()
df["role"] = [role(r) for r in df.itertuples()]
df["flag"] = [flag(r) for r in df.itertuples()]
if "price_risk" in df.columns:
    df["price_flag"] = df["price_risk"].map(PRICE_LABEL).fillna("")
else:
    df["price_flag"] = ""
df["where"] = ["XI" if s <= 11 else "Bench" for s in df["slot"]]

ticker = components.load("fixture_ticker")
shorts = components.team_shorts()
cells = components.fixture_cells(ticker, shorts, gw0, 6) if not ticker.empty else None
fix_cols = [f"GW{g}" for g in gws]
diff = pd.DataFrame(index=df.index, columns=fix_cols, dtype=float)
if cells is not None and not cells.empty:
    lab = cells.pivot(index="team_id", columns="gw", values="label")
    val = cells.pivot(index="team_id", columns="gw", values="value")
    for g, col in zip(gws, fix_cols, strict=True):
        df[col] = df["team_id"].map(lab[g]) if g in lab.columns else "-"
        diff[col] = df["team_id"].map(val[g]).astype(float) if g in val.columns else float("nan")
        df[col] = df[col].fillna("-")
else:
    st.info("No fixture_ticker yet; fixtures hidden.")

st.subheader("Squad")
view = st.radio("View", ["Pitch", "Table"], horizontal=True)

if view == "Pitch":
    xi = df[df["slot"] <= 11]
    for pos in ("FWD", "MID", "DEF", "GKP"):
        grp = xi[xi["position"] == pos]
        cols = st.columns(max(len(grp), 1))
        for col, r in zip(cols, grp.itertuples(), strict=False):
            mark = f" ({r.role})" if r.role else ""
            xp = f"{r.xp_next6:.1f}" if pd.notna(r.xp_next6) else "-"
            warn = f"  \n:warning: {r.flag}" if r.flag else ""
            col.markdown(
                f"**{r.web_name}**{mark}  \n{r.team_short} £{r.price:.1f}m  \nxP6 {xp}"
                f"{warn}"
            )
    st.markdown("**Bench:** " + " | ".join(
        f"{r.slot - 11}. {r.web_name} ({r.position})" for r in df[df["slot"] > 11].itertuples()
    ))

xp_cols = [f"xp_gw{g}" for g in gws]
show = ["slot", "where", "web_name", "role", "position", "team_short", "price", "purchase_price",
        "sell_price", *xp_cols, "xp_next6", "flag", "price_flag"]
show = [c for c in show if c in df.columns] + [c for c in fix_cols if c in df.columns]
table = df[show].copy()


def colour_fixtures(_):
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    for col in fix_cols:
        if col in styles.columns:
            for i in table.index:
                bg = components.difficulty_colour(diff.at[i, col])
                styles.at[i, col] = f"background-color: {bg}; color: #111"
    return styles


fmt = {c: "{:.1f}" for c in [*xp_cols, "xp_next6"]}
fmt.update({"price": "£{:.1f}", "purchase_price": "£{:.1f}", "sell_price": "£{:.1f}"})
fmt = {k: v for k, v in fmt.items() if k in table.columns}
st.dataframe(
    table.style.format(fmt, na_rep="-").apply(colour_fixtures, axis=None),
    hide_index=True,
    height=35 * (len(table) + 1) + 3,
)
st.caption(
    "Fixture colour = custom difficulty (green easy, red hard); '-' = blank GW; "
    "'+' = double GW. Price flag from net transfers vs thresholds."
)

if "net_transfers_event" in df.columns:
    with st.expander("Price movement detail"):
        extra = [c for c in ("web_name", "price_risk", "net_transfers_event") if c in df.columns]
        st.dataframe(df[extra], hide_index=True)
