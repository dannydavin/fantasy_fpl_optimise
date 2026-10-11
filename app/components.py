"""Shared Streamlit helpers: cached table loaders, my squad, labels, team colours.

Contract (Wave 2): pages rely on these signatures; extend freely but keep them stable.

    load(table, gw=None) -> DataFrame        # empty contract frame if missing
    my_squad(horizon=6) -> DataFrame         # my_team ⨝ players ⨝ next-N xp
    player_label(player_id) -> str           # "Haaland (MCI £14.5m)"
    player_labels() -> dict[int, str]
    team_shorts() -> dict[int, str]          # team_id -> "ARS"
    team_colour(short_or_id) -> str          # hex primary colour
    TEAM_COLOURS: dict[str, str]

Data root comes from ``fpl.config.get_settings().data_dir`` (env ``FPL_DATA_DIR``).
Pure builders (``build_squad``) are importable without a Streamlit runtime for tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow `streamlit run` without the package installed
    sys.path.insert(0, str(ROOT / "src"))

from fpl import storage  # noqa: E402
from fpl.schemas import empty, get_schema  # noqa: E402

CACHE_TTL = 600

# Primary kit colours by FPL short name (2026/27 PL + recent clubs).
TEAM_COLOURS: dict[str, str] = {
    "ARS": "#EF0107", "AVL": "#670E36", "BOU": "#DA291C", "BRE": "#E30613",
    "BHA": "#0057B8", "BUR": "#6C1D45", "CHE": "#034694", "COV": "#59CBE8",
    "CRY": "#1B458F", "EVE": "#003399", "FUL": "#000000", "HUL": "#F5A12D",
    "IPS": "#3A64A3", "LEE": "#FFCD00", "LEI": "#003090", "LIV": "#C8102E",
    "LUT": "#F78F1E", "MCI": "#6CABDD", "MUN": "#DA291C", "NEW": "#241F20",
    "NFO": "#DD0000", "SHU": "#EE2737", "SOU": "#D71920", "SUN": "#EB172B",
    "TOT": "#132257", "WHU": "#7A263A", "WOL": "#FDB913",
}
DEFAULT_COLOUR = "#888888"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load(table: str, gw: int | None = None) -> pd.DataFrame:
    """Read a stored table (latest GW when ``gw`` is None); empty contract frame if absent."""
    try:
        return storage.read_table(table, gw)
    except FileNotFoundError:
        return empty(table)


def has(table: str, gw: int | None = None) -> bool:
    """True if the table has been written (any GW when ``gw`` is None)."""
    get_schema(table)
    return storage.exists(table, gw)


def build_squad(
    my_team: pd.DataFrame,
    players: pd.DataFrame,
    xp: pd.DataFrame | None = None,
    horizon: int = 6,
    start_gw: int | None = None,
) -> pd.DataFrame:
    """my_team ⨝ players (+ xp pivot). Adds ``xp_gw{N}`` columns and ``xp_next{horizon}``.

    Sorted by slot. ``start_gw`` defaults to my_team.gw (the GW being planned).
    """
    if my_team.empty:
        return my_team.copy()
    pcols = [c for c in (
        "player_id", "web_name", "team_id", "team_short", "position", "element_type",
        "status", "chance_of_playing_next_round", "news", "selected_by_percent", "form",
        "total_points", "price_risk", "net_transfers_event", "team_code",
    ) if c in players.columns]
    out = my_team.merge(players[pcols], on="player_id", how="left")
    if start_gw is None:
        start_gw = int(my_team["gw"].iloc[0])
    gws = list(range(start_gw, start_gw + horizon))
    if xp is not None and not xp.empty:
        x = xp[xp["gw"].isin(gws)]
        if "model" in x.columns and x["model"].nunique() > 1:
            x = x[x["model"] == x["model"].iloc[0]]
        piv = x.pivot_table(index="player_id", columns="gw", values="xp", aggfunc="sum")
        piv = piv.reindex(columns=gws)
        piv.columns = [f"xp_gw{int(g)}" for g in piv.columns]
        out = out.merge(piv, left_on="player_id", right_index=True, how="left")
    for g in gws:
        out[f"xp_gw{g}"] = out.get(f"xp_gw{g}", pd.Series(float("nan"), index=out.index))
    out[f"xp_next{horizon}"] = out[[f"xp_gw{g}" for g in gws]].sum(axis=1, min_count=1)
    return out.sort_values("slot").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def my_squad(horizon: int = 6) -> pd.DataFrame:
    """My current 15 with player info and per-GW xP for the next ``horizon`` GWs."""
    return build_squad(load("my_team"), load("players"), load("xp"), horizon)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def player_labels() -> dict[int, str]:
    p = load("players")
    if p.empty:
        return {}
    return {
        int(r.player_id): f"{r.web_name} ({r.team_short} £{r.price:.1f}m)"
        for r in p.itertuples()
    }


def player_label(player_id: int) -> str:
    return player_labels().get(int(player_id), f"#{player_id}")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def team_shorts() -> dict[int, str]:
    t = load("teams")
    return {int(r.team_id): str(r.short_name) for r in t.itertuples()}


def team_colour(team: str | int) -> str:
    short = team_shorts().get(int(team), "") if isinstance(team, int) else str(team)
    return TEAM_COLOURS.get(short, DEFAULT_COLOUR)


def reload_button() -> None:
    """Sidebar button that clears cached tables (after running scripts/update.py)."""
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------- Wave 2 E additions

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]
POSITION_COLOURS = {"GKP": "#e8a317", "DEF": "#2a9d8f", "MID": "#457b9d", "FWD": "#d1495b"}
# difficulty 1 (easy) -> 5 (hard)
DIFFICULTY_COLOURS = ["#2e9e5b", "#8fd18f", "#e6e6e6", "#f2a07b", "#d9534f"]
BLANK_COLOUR = "#bdbdbd"


def page_setup(title: str, icon: str = ":soccer:") -> None:
    """Set page config, title and sidebar reload button (call first on each page)."""
    st.set_page_config(page_title=f"FPL - {title}", page_icon=icon, layout="wide")
    st.title(title)
    reload_button()


def difficulty_colour(difficulty: float | None) -> str:
    """Hex colour for a 1-5 difficulty (floats are rounded); grey when missing."""
    if difficulty is None or pd.isna(difficulty):
        return BLANK_COLOUR
    idx = min(max(int(round(float(difficulty))), 1), 5) - 1
    return DIFFICULTY_COLOURS[idx]


def planning_gw() -> int | None:
    """GW being planned (my_entry.gw, else players.gw)."""
    for table in ("my_entry", "players"):
        df = load(table)
        if not df.empty and df["gw"].notna().any():
            return int(df["gw"].iloc[0])
    return None


def fixture_cells(
    ticker: pd.DataFrame,
    shorts: dict[int, str],
    start_gw: int,
    n: int = 6,
    metric: str = "difficulty",
) -> pd.DataFrame:
    """Long table (team_id, gw, label, value, n_fixtures) for every team x GW in the window.

    ``label`` is e.g. "BHA (H)"; a DGW joins with " + "; a blank GW has label "-" and
    value NaN. ``value`` is the mean of ``metric`` over that GW's fixtures.
    """
    cols = ["team_id", "gw", "label", "value", "n_fixtures"]
    if ticker.empty:
        return pd.DataFrame(columns=cols)
    gws = list(range(start_gw, start_gw + n))
    teams = sorted(set(ticker["team_id"].astype(int)))
    t = ticker[ticker["gw"].isin(gws)]
    groups = {k: g for k, g in t.groupby(["team_id", "gw"])}
    rows = []
    for team in teams:
        for gw in gws:
            g = groups.get((team, gw))
            if g is None or g.empty:
                rows.append({"team_id": team, "gw": gw, "label": "-", "value": float("nan"),
                             "n_fixtures": 0})
                continue
            labels = [
                f"{shorts.get(int(r.opponent_id), r.opponent_id)} ({'H' if r.is_home else 'A'})"
                for r in g.itertuples()
            ]
            rows.append({"team_id": team, "gw": gw, "label": " + ".join(labels),
                         "value": float(g[metric].mean()), "n_fixtures": len(g)})
    return pd.DataFrame(rows, columns=cols)


def team_fixture_labels(
    ticker: pd.DataFrame, team_id: int, shorts: dict[int, str], start_gw: int, n: int = 6
) -> list[str]:
    """Next-``n`` GW fixture labels for one team ("-" for blanks)."""
    c = fixture_cells(ticker, shorts, start_gw, n)
    c = c[c["team_id"] == int(team_id)].sort_values("gw")
    return c["label"].tolist() if not c.empty else ["-"] * n


# ---------------------------------------------------------------- visual helpers

SHIRT_URL = "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}{gk}-110.webp"
STATUS_TEXT = {"d": "Doubtful", "i": "Injured", "s": "Suspended", "u": "Unavailable",
               "n": "Not in squad"}


def hero(title: str, subtitle: str = "", pills: list[str] | None = None) -> None:
    """Blue gradient banner (fixed colours so it reads the same in light and dark mode)."""
    chips = "".join(f'<span class="hero-pill">{p}</span>' for p in pills or [])
    st.html(f"""
<style>
.hero {{ background: linear-gradient(120deg, #0b2a5b 0%, #1d5fd1 60%, #3b8cff 100%);
  color: #fff; border-radius: 14px; padding: 22px 26px; position: relative; overflow: hidden; }}
.hero::after {{ content: ""; position: absolute; right: -60px; top: -60px; width: 240px;
  height: 240px; border-radius: 50%; border: 28px solid rgba(255,255,255,.08); }}
.hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; color: #fff; line-height: 1.15; }}
.hero p {{ margin: 6px 0 12px; opacity: .85; font-size: 1rem; }}
.hero-pill {{ display: inline-block; background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.28); border-radius: 999px; padding: 3px 12px;
  margin: 0 6px 6px 0; font-size: .85rem; font-weight: 600; }}
</style>
<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>""")


def shirt_url(team_code: int | float | None, is_gk: bool = False) -> str | None:
    if team_code is None or pd.isna(team_code):
        return None
    return SHIRT_URL.format(code=int(team_code), gk="_1" if is_gk else "")


def _pitch_card(r: dict) -> str:
    img = shirt_url(r.get("team_code"), r.get("position") == "GKP")
    shirt = (f'<img src="{img}" alt="{r.get("team_short", "")}">' if img else
             f'<div class="shirt-fallback">{r.get("team_short", "")}</div>')
    badges = ""
    if r.get("role"):
        badges += f'<span class="badge cap">{r["role"]}</span>'
    status = r.get("status")
    if status and status != "a":
        cop = r.get("chance_of_playing_next_round")
        sev = "warn" if status == "d" and pd.notna(cop) and cop >= 50 else "out"
        tip = STATUS_TEXT.get(status, status) + (f" {int(cop)}%" if pd.notna(cop) else "")
        badges += f'<span class="badge {sev}" title="{tip}">!</span>'
    return (f'<div class="pcard">{badges}{shirt}'
            f'<div class="pname">{r["web_name"]}</div>'
            f'<div class="pinfo">{r.get("info", "")}</div></div>')


def pitch_html(squad: pd.DataFrame) -> str:
    """FPL-style pitch for a squad frame with columns web_name, position, slot and optional
    team_code, team_short, role, status, chance_of_playing_next_round, info (sub-label)."""
    rows = squad.to_dict("records")
    xi = [r for r in rows if r["slot"] <= 11]
    bench = sorted((r for r in rows if r["slot"] > 11), key=lambda r: r["slot"])
    lines = "".join(
        '<div class="prow">' + "".join(_pitch_card(r) for r in xi if r["position"] == pos)
        + "</div>" for pos in POSITION_ORDER
    )
    bench_cards = "".join(_pitch_card(r) for r in bench)
    return f"""
<style>
.pitch {{ position: relative; border-radius: 14px 14px 0 0; padding: 18px 8px 10px;
  background: repeating-linear-gradient(180deg, #1f8f4e 0 56px, #23a058 56px 112px);
  overflow: hidden; }}
.pitch .lines {{ position: absolute; inset: 10px; border: 2px solid rgba(255,255,255,.55);
  border-radius: 4px; pointer-events: none; }}
.pitch .box {{ position: absolute; left: 30%; right: 30%; top: 10px; height: 13%;
  border: 2px solid rgba(255,255,255,.55); border-top: none; }}
.pitch .circle {{ position: absolute; left: 50%; bottom: -70px; width: 140px; height: 140px;
  transform: translateX(-50%); border: 2px solid rgba(255,255,255,.55); border-radius: 50%; }}
.prow {{ position: relative; display: flex; justify-content: center; gap: 2%;
  margin: 10px 0 18px; }}
.pcard {{ position: relative; width: 104px; text-align: center; }}
.pcard img {{ width: 64px; height: auto; filter: drop-shadow(0 3px 3px rgba(0,0,0,.35)); }}
.shirt-fallback {{ width: 56px; height: 60px; margin: 0 auto; border-radius: 10px 10px 6px 6px;
  background: #0b2a5b; color: #fff; font-weight: 700; display: flex; align-items: center;
  justify-content: center; }}
.pname {{ background: #0b2a5b; color: #fff; font-weight: 700; font-size: .8rem;
  border-radius: 6px 6px 0 0; padding: 3px 4px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
.pinfo {{ background: #e8f0ff; color: #0b1a33; font-size: .75rem; border-radius: 0 0 6px 6px;
  padding: 2px 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.badge {{ position: absolute; top: 0; width: 22px; height: 22px; border-radius: 50%;
  font-size: .72rem; font-weight: 800; line-height: 22px; color: #fff;
  box-shadow: 0 0 0 2px #fff; }}
.badge.cap {{ right: 14px; background: #0b1a33; }}
.badge.warn {{ left: 14px; background: #f2a900; color: #0b1a33; }}
.badge.out {{ left: 14px; background: #d9342b; }}
.bench {{ background: linear-gradient(180deg, #d6e4fb, #c3d6f6); border-radius: 0 0 14px 14px;
  padding: 10px 8px 12px; }}
.bench .label {{ color: #0b2a5b; font-weight: 800; font-size: .75rem; letter-spacing: .08em;
  text-align: center; margin-bottom: 6px; }}
@media (max-width: 640px) {{ .pcard {{ width: 19%; }} .pcard img {{ width: 44px; }}
  .pname, .pinfo {{ font-size: .65rem; }} }}
</style>
<div class="pitch"><div class="lines"></div><div class="box"></div><div class="circle"></div>
{lines}</div>
<div class="bench"><div class="label">BENCH</div><div class="prow" style="margin:0">
{bench_cards}</div></div>"""
