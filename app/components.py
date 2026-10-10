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
        "total_points", "price_risk", "net_transfers_event",
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
