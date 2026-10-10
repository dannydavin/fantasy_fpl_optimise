"""Column contracts for every parquet table.

Conventions (apply to all tables):
- IDs: ``player_id`` (FPL element id, this season), ``team_id`` (1–20), ``entry_id``
  (manager), ``fixture_id``. ``code`` is the cross-season stable player code.
- ``gw`` is the gameweek the row refers to:
    * snapshot tables (players, teams, my_*): the GW being planned, i.e. the next
      deadline (``is_next`` event; GW7 at the time of writing);
    * fixtures: the fixture's own event (null if unscheduled);
    * top10k_*: the GW whose picks were pulled (latest passed deadline);
    * xp / fixture_ticker: the target GW being projected.
- Money is in £m as float (API tenths / 10): ``price`` 4.5, ``bank`` 0.4.
- Timestamps are timezone-aware UTC.
- Tables may carry extra columns beyond the contract; required columns must exist
  and are cast to the listed dtype by :func:`conform`.

Storage layout (see :mod:`fpl.storage`):
``data/<layer>/gwNN/<table>.parquet`` for ``per_gw`` tables and
``data/<layer>/<table>.parquet`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
# Starting XI min/max per position (formation rules)
XI_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_TEAM = 3
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_FREE_TRANSFERS = 5
HIT_COST = 4

INT = "Int64"
FLOAT = "float64"
STR = "string"
BOOL = "boolean"
TS = "datetime64[ns, UTC]"


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: dict[str, str]
    layer: Literal["raw", "processed"] = "raw"
    per_gw: bool = True
    key: tuple[str, ...] = ()
    description: str = ""
    extra: dict = field(default_factory=dict)


class SchemaError(ValueError):
    pass


_T: dict[str, TableSchema] = {}


def _register(schema: TableSchema) -> TableSchema:
    _T[schema.name] = schema
    return schema


# ---- raw: FPL snapshot (Agent A) -------------------------------------------------------

PLAYERS = _register(TableSchema(
    "players",
    {
        "gw": INT, "player_id": INT, "code": INT, "web_name": STR, "first_name": STR,
        "second_name": STR, "team_id": INT, "team_short": STR, "element_type": INT,
        "position": STR, "price": FLOAT, "cost_change_event": FLOAT,
        "cost_change_start": FLOAT, "selected_by_percent": FLOAT, "status": STR,
        "chance_of_playing_next_round": FLOAT, "news": STR, "news_added": TS,
        "form": FLOAT, "points_per_game": FLOAT, "total_points": INT, "event_points": INT,
        "minutes": INT, "starts": INT, "goals_scored": INT, "assists": INT,
        "clean_sheets": INT, "goals_conceded": INT, "bonus": INT, "bps": INT,
        "expected_goals": FLOAT, "expected_assists": FLOAT,
        "expected_goal_involvements": FLOAT, "expected_goals_conceded": FLOAT,
        "defensive_contribution": INT, "transfers_in_event": INT,
        "transfers_out_event": INT, "ep_next": FLOAT, "penalties_order": INT,
        "corners_and_indirect_freekicks_order": INT, "direct_freekicks_order": INT,
    },
    key=("player_id",),
    description="bootstrap-static elements at snapshot time, one row per player.",
))

TEAMS = _register(TableSchema(
    "teams",
    {
        "gw": INT, "team_id": INT, "code": INT, "name": STR, "short_name": STR,
        "strength": INT, "strength_overall_home": INT, "strength_overall_away": INT,
        "strength_attack_home": INT, "strength_attack_away": INT,
        "strength_defence_home": INT, "strength_defence_away": INT,
    },
    key=("team_id",),
))

FIXTURES = _register(TableSchema(
    "fixtures",
    {
        "fixture_id": INT, "gw": INT, "kickoff_time": TS, "team_h": INT, "team_a": INT,
        "team_h_score": INT, "team_a_score": INT, "team_h_difficulty": INT,
        "team_a_difficulty": INT, "started": BOOL, "finished": BOOL,
    },
    key=("fixture_id",),
    description="Whole-season fixture list as of the snapshot; gw = fixture event.",
))

EVENTS = _register(TableSchema(
    "events",
    {
        "gw": INT, "name": STR, "deadline_time": TS, "finished": BOOL, "is_previous": BOOL,
        "is_current": BOOL, "is_next": BOOL, "average_entry_score": INT,
        "highest_score": INT,
    },
    key=("gw",),
    description="bootstrap-static events (stored inside each snapshot dir).",
))

MY_ENTRY = _register(TableSchema(
    "my_entry",
    {
        "gw": INT, "entry_id": INT, "entry_name": STR, "bank": FLOAT, "value": FLOAT,
        "free_transfers": INT, "total_points": INT, "overall_rank": INT,
        "chips_available": STR,  # comma-separated chip names, e.g. "freehit,bboost,3xc"
        "picks_gw": INT,  # GW of the picks the squad was taken from
    },
    key=("entry_id",),
    description="One row: my manager state going into snapshot gw.",
))

MY_TEAM = _register(TableSchema(
    "my_team",
    {
        "gw": INT, "entry_id": INT, "player_id": INT, "slot": INT, "multiplier": INT,
        "is_captain": BOOL, "is_vice_captain": BOOL, "price": FLOAT,
        "purchase_price": FLOAT, "sell_price": FLOAT,
    },
    key=("player_id",),
    description="15 rows: current squad; slot 1–11 XI, 12–15 bench order. "
                "sell_price = purchase + floor half of any rise (0.1 steps).",
))

MY_HISTORY = _register(TableSchema(
    "my_history",
    {
        "gw": INT, "points": INT, "total_points": INT, "rank": INT, "overall_rank": INT,
        "bank": FLOAT, "value": FLOAT, "event_transfers": INT,
        "event_transfers_cost": INT, "points_on_bench": INT, "chip": STR,
    },
    per_gw=True,
    key=("gw",),
    description="entry/{id}/history current[], one row per played GW (stored in snapshot dir).",
))

MY_TRANSFERS = _register(TableSchema(
    "my_transfers",
    {
        "gw": INT, "time": TS, "player_in": INT, "player_in_cost": FLOAT,
        "player_out": INT, "player_out_cost": FLOAT,
    },
    description="entry/{id}/transfers; gw = event of the transfer.",
))

# ---- raw: top 10k (Agent B) ------------------------------------------------------------

TOP10K_ENTRIES = _register(TableSchema(
    "top10k_entries",
    {"gw": INT, "entry_id": INT, "rank": INT, "total": INT, "entry_name": STR,
     "player_name": STR},
    key=("entry_id",),
    description="Overall league 314 standings rows used for the pull.",
))

TOP10K_PICKS = _register(TableSchema(
    "top10k_picks",
    {
        "gw": INT, "entry_id": INT, "rank": INT, "player_id": INT, "slot": INT,
        "multiplier": INT, "is_captain": BOOL, "is_vice_captain": BOOL,
        "active_chip": STR, "event_transfers": INT, "event_transfers_cost": INT,
        "bank": FLOAT, "value": FLOAT,
    },
    key=("entry_id", "player_id"),
    description="15 rows per top-10k entry for picks gw.",
))

# ---- raw: news (Agent C) ---------------------------------------------------------------

NEWS = _register(TableSchema(
    "news",
    {
        "item_id": STR, "published": TS, "source": STR, "kind": STR, "title": STR,
        "summary": STR, "url": STR, "player_id": INT, "player_name": STR,
    },
    key=("item_id", "player_id"),
    description="kind in {'fpl_flag','rss','reddit'}. One row per (item, tagged player); "
                "untagged items have null player_id.",
))

# ---- raw: history / understat (Agent D) — season-level, not per GW --------------------

HIST_PLAYER_GW = _register(TableSchema(
    "hist_player_gw",
    {
        "season": STR, "gw": INT, "element": INT, "code": INT, "player_name": STR,
        "team": STR, "position": STR, "opponent_team": INT, "was_home": BOOL,
        "kickoff_time": TS, "minutes": INT, "total_points": INT, "goals_scored": INT,
        "assists": INT, "clean_sheets": INT, "goals_conceded": INT, "bonus": INT,
        "bps": INT, "expected_goals": FLOAT, "expected_assists": FLOAT, "value": FLOAT,
    },
    per_gw=False,
    description="vaastav per-GW player rows for past seasons (element is season-specific).",
))

UNDERSTAT_PLAYER_MATCH = _register(TableSchema(
    "understat_player_match",
    {"season": STR, "date": TS, "player_name": STR, "team": STR, "minutes": INT,
     "xg": FLOAT, "xa": FLOAT, "npxg": FLOAT, "shots": INT, "key_passes": INT},
    per_gw=False,
))

UNDERSTAT_TEAM_MATCH = _register(TableSchema(
    "understat_team_match",
    {"season": STR, "date": TS, "team": STR, "opponent": STR, "is_home": BOOL,
     "xg": FLOAT, "xga": FLOAT, "npxg": FLOAT, "npxga": FLOAT},
    per_gw=False,
))

# ---- processed -------------------------------------------------------------------------

XP = _register(TableSchema(
    "xp",
    {"gw": INT, "player_id": INT, "xp": FLOAT, "p_start": FLOAT, "xmins": FLOAT,
     "model": STR},
    layer="processed",
    key=("gw", "player_id", "model"),
    description="Projected points per player per target GW (DGW summed; blank = 0). "
                "Stored under the snapshot gw dir; the solver consumes (gw, player_id, xp).",
))

OWNERSHIP = _register(TableSchema(
    "ownership",
    {
        "gw": INT, "player_id": INT, "owned_pct": FLOAT, "start_pct": FLOAT,
        "captain_pct": FLOAT, "vice_pct": FLOAT, "eo": FLOAT,
        "overall_selected_pct": FLOAT, "delta_vs_overall": FLOAT,
        "delta_vs_prev_gw": FLOAT,
    },
    layer="processed",
    key=("gw", "player_id"),
    description="Top-10k ownership for picks gw. Percentages 0–100; eo = sum(multiplier)/N*100.",
))

FIXTURE_TICKER = _register(TableSchema(
    "fixture_ticker",
    {"team_id": INT, "gw": INT, "opponent_id": INT, "is_home": BOOL, "fdr": INT,
     "difficulty": FLOAT, "fixture_id": INT},
    layer="processed",
    key=("team_id", "gw", "fixture_id"),
    description="One row per team per fixture (DGW = 2 rows, blank = 0 rows). "
                "difficulty: custom 1–5 scale from rolling xG (higher = harder).",
))

FORM = _register(TableSchema(
    "form",
    {"player_id": INT, "window": INT, "minutes": INT, "start_pct": FLOAT,
     "minutes_share": FLOAT, "xg_per90": FLOAT, "xa_per90": FLOAT, "xgi_per90": FLOAT,
     "dc_per90": FLOAT, "points_per90": FLOAT},
    layer="processed",
    key=("player_id", "window"),
    description="Rolling form over the last `window` GWs as of snapshot gw.",
))

TABLES: dict[str, TableSchema] = _T


def get_schema(table: str) -> TableSchema:
    try:
        return TABLES[table]
    except KeyError:
        raise SchemaError(f"unknown table {table!r}; known: {sorted(TABLES)}") from None


def conform(df: pd.DataFrame, table: str | TableSchema) -> pd.DataFrame:
    """Check required columns exist and cast them; required columns first, extras after."""
    schema = table if isinstance(table, TableSchema) else get_schema(table)
    missing = [c for c in schema.columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{schema.name}: missing columns {missing}")
    out = df.copy()
    for col, dtype in schema.columns.items():
        try:
            out[col] = _cast(out[col], dtype)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{schema.name}.{col}: cannot cast to {dtype}: {exc}") from exc
    extras = [c for c in out.columns if c not in schema.columns]
    return out[list(schema.columns) + extras]


def _cast(s: pd.Series, dtype: str) -> pd.Series:
    if dtype == TS:
        return pd.to_datetime(s, utc=True, errors="coerce")
    if dtype == INT:
        return pd.to_numeric(s, errors="coerce").round().astype(INT)
    if dtype == FLOAT:
        return pd.to_numeric(s, errors="coerce").astype(FLOAT)
    if dtype == BOOL:
        return s.astype(BOOL)
    if dtype == STR:
        return s.astype(STR)
    return s.astype(dtype)


def empty(table: str) -> pd.DataFrame:
    """Empty frame with the table's required columns and dtypes."""
    schema = get_schema(table)
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in schema.columns.items()})
