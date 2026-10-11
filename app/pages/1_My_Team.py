import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import components  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

components.page_setup("My Team", ":shirt:")

CHIP_ABBR = {"wildcard": "WC", "freehit": "FH", "bboost": "BB", "3xc": "TC", "manager": "AM"}

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
    chips = [x for x in str(e["chips_available"] or "").split(",") if x]
    c[3].metric("Chips", " · ".join(CHIP_ABBR.get(x, x) for x in chips) or "none",
                help=", ".join(chips) or None)

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


def _num(v, fmt: str) -> str:
    return fmt.format(v) if pd.notna(v) else "-"


INFO = {
    "Next fixture": lambda r: r.get(fix_cols[0], "-"),
    f"xP GW{gw0}": lambda r: _num(r.get(f"xp_gw{gw0}"), "{:.1f} xP"),
    "xP next 6": lambda r: _num(r.get("xp_next6"), "{:.1f} xP"),
    "Price": lambda r: _num(r.get("price"), "£{:.1f}m"),
}

st.subheader(f"Squad for GW{gw0}")
left, right = st.columns([3, 1])
with right:
    show_info = st.radio("Under each player", list(INFO), index=1)
    xi = df[df["slot"] <= 11]
    xp_col = f"xp_gw{gw0}"
    if xp_col in xi.columns and xi[xp_col].notna().any():
        st.metric(f"XI xP GW{gw0} (C doubled)",
                  f"{(xi[xp_col] * xi['multiplier'].clip(lower=1)).sum():.1f}")
        best = xi.loc[xi[xp_col].idxmax()]
        st.metric("Top xP in XI", str(best["web_name"]), f"{best[xp_col]:.1f} xP",
                  delta_color="off")
    flagged = df[df["flag"] != ""]
    if not flagged.empty:
        st.warning("\n".join(f"- **{r.web_name}**: {r.flag}" for r in flagged.itertuples()))
pitch = df.assign(info=[INFO[show_info](r) for r in df.to_dict("records")])
left.html(components.pitch_html(pitch))

st.subheader("Fixtures and xP")
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
