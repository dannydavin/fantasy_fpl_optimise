"""Understat EPL team / player per-match xG, xA, npxG.

Since 2025 understat.com serves JSON from XHR endpoints (checked live Oct 2026):
- ``/getLeagueData/EPL/<start_year>``: ``teams`` (per-team match history incl. npxG),
  ``players`` (season totals, with ids), ``dates`` (every fixture with both sides' xG).
- ``/getPlayerData/<player_id>``: ``matches`` for every season the player has played.
Both need ``X-Requested-With: XMLHttpRequest`` and a same-site ``Referer``.

Team-level data is the default (1 request per season). Player-level needs one request per
player (~600 per season-set) so it is opt-in (``players=True``) and throttled.
If Understat is unreachable the affected season is skipped with a warning (graceful
degradation); parsers are pure and tested against recorded samples.
"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from pathlib import Path

import pandas as pd

from fpl import storage
from fpl.config import get_settings
from fpl.ingest.history import http_get, previous_seasons

log = logging.getLogger(__name__)

BASE = "https://understat.com"
CURRENT_TTL = 12 * 3600

# Understat club title -> FPL short_name (covers clubs seen in recent EPL seasons)
TEAM_SHORT = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Leicester": "LEI", "Liverpool": "LIV", "Luton": "LUT", "Manchester City": "MCI",
    "Manchester United": "MUN", "Newcastle United": "NEW", "Nottingham Forest": "NFO",
    "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN",
    "Tottenham": "TOT", "West Ham": "WHU", "Wolverhampton Wanderers": "WOL",
    "Coventry": "COV", "Hull": "HUL", "Leicester City": "LEI",
}


def season_year(season: str) -> int:
    return int(season[:4])


def season_label(year: int | str) -> str:
    y = int(year)
    return f"{y}-{str(y + 1)[2:]}"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _headers() -> dict:
    return {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE}/league/EPL",
        "Accept": "application/json, text/javascript, */*",
    }


def _fetch_json(
    path: str, cache_dir: Path, name: str, ttl: float | None, transport=None
) -> dict | None:
    cache = cache_dir / "understat" / f"{name}.json"
    if cache.exists() and (ttl is None or time.time() - cache.stat().st_mtime < ttl):
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        data = http_get(BASE + path, headers=_headers(), transport=transport).json()
    except Exception as exc:  # unreachable / blocked / schema change
        log.warning("understat %s unavailable: %s", path, exc)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# ---- pure parsers -----------------------------------------------------------------------


def parse_team_matches(payload: dict, season: str) -> pd.DataFrame:
    """League payload -> ``understat_team_match`` (two rows per played match).

    Opponent/xGA come from ``dates``; npxG/npxGA are joined from each team's ``teams``
    history (same team + kickoff datetime). Extras: ``team_short``, ``scored``, ``conceded``.
    """
    np_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for t in payload.get("teams", {}).values():
        for h in t.get("history", []):
            np_lookup[(t["title"], h["date"])] = (float(h["npxG"]), float(h["npxGA"]))
    rows = []
    for m in payload.get("dates", []):
        if not m.get("isResult"):
            continue
        dt = m["datetime"]
        for side, other in (("h", "a"), ("a", "h")):
            team, opp = m[side]["title"], m[other]["title"]
            npxg, npxga = np_lookup.get((team, dt), (float("nan"), float("nan")))
            rows.append({
                "season": season,
                "date": pd.Timestamp(dt, tz="UTC"),
                "team": team,
                "opponent": opp,
                "is_home": side == "h",
                "xg": float(m["xG"][side]),
                "xga": float(m["xG"][other]),
                "npxg": npxg,
                "npxga": npxga,
                "team_short": TEAM_SHORT.get(team),
                "scored": int(m["goals"][side]),
                "conceded": int(m["goals"][other]),
            })
    cols = ["season", "date", "team", "opponent", "is_home", "xg", "xga", "npxg", "npxga",
            "team_short", "scored", "conceded"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def league_players(payload: dict) -> pd.DataFrame:
    """League payload ``players`` -> id/name/team/season totals."""
    df = pd.DataFrame(payload.get("players", []))
    if df.empty:
        return df
    for c in ("games", "time", "goals", "assists", "shots", "key_passes"):
        df[c] = pd.to_numeric(df[c])
    for c in ("xG", "xA", "npxG"):
        df[c] = pd.to_numeric(df[c])
    return df


def parse_player_matches(
    payload: dict, season: str, team_titles: list[str] | None = None
) -> pd.DataFrame:
    """``getPlayerData`` payload -> ``understat_player_match`` rows for one season.

    ``team_titles``: the player's club(s) that season (league payload ``team_title`` split
    on commas); the one playing in each match is used as ``team``.
    """
    year = str(season_year(season))
    pl = payload.get("player", {})
    rows = []
    for m in payload.get("matches", []):
        if str(m.get("season")) != year:
            continue
        team = next(
            (t for t in (team_titles or []) if t in (m["h_team"], m["a_team"])), m["h_team"]
        )
        rows.append({
            "season": season,
            "date": pd.Timestamp(m["date"], tz="UTC"),
            "player_name": pl.get("name"),
            "team": team,
            "minutes": int(m["time"]),
            "xg": float(m["xG"]),
            "xa": float(m["xA"]),
            "npxg": float(m["npxG"]),
            "shots": int(m["shots"]),
            "key_passes": int(m["key_passes"]),
            "player_id": int(pl["id"]) if pl.get("id") else None,
            "match_id": int(m["id"]),
            "position": m.get("position"),
            "goals": int(m["goals"]),
            "assists": int(m["assists"]),
        })
    return pd.DataFrame(rows)


# ---- orchestration ----------------------------------------------------------------------


def _merge_store(new: pd.DataFrame, table: str, root: Path | None) -> None:
    if storage.exists(table, root=root):
        old = storage.read_table(table, root=root)
        old = old[~old["season"].isin(new["season"].unique())]
        if len(old):
            new = pd.concat([old, new], ignore_index=True)
    storage.write_table(new, table, root=root)


def run(
    seasons: list[str] | None = None,
    root: Path | None = None,
    players: bool = False,
    cache_dir: Path | None = None,
    transport=None,
) -> dict[str, pd.DataFrame]:
    """Fetch Understat data for ``seasons`` (default: last 3 completed).

    Returns ``{"understat_team_match": df, "understat_player_match": df}`` (player frame is
    empty unless ``players=True``). Seasons that can't be fetched are skipped.
    """
    settings = get_settings()
    seasons = seasons or previous_seasons(settings.season)
    cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir
    team_frames, player_frames = [], []
    for season in seasons:
        year = season_year(season)
        ttl = CURRENT_TTL if season == settings.season else None
        league = _fetch_json(f"/getLeagueData/EPL/{year}", cache_dir, f"league_EPL_{year}",
                             ttl, transport)
        if league is None:
            continue
        tm = parse_team_matches(league, season)
        if len(tm):
            team_frames.append(tm)
        if players:
            lp = league_players(league)
            for _, p in lp[lp["time"] > 0].iterrows():
                data = _fetch_json(f"/getPlayerData/{p['id']}", cache_dir, f"player_{p['id']}",
                                   ttl, transport)
                if data is None:
                    continue
                titles = [t.strip() for t in str(p["team_title"]).split(",")]
                pm = parse_player_matches(data, season, titles)
                if len(pm):
                    player_frames.append(pm)
    out: dict[str, pd.DataFrame] = {}
    if team_frames:
        out["understat_team_match"] = pd.concat(team_frames, ignore_index=True)
        _merge_store(out["understat_team_match"], "understat_team_match", root)
    else:
        out["understat_team_match"] = storage.conform(pd.DataFrame(), "understat_team_match") \
            if False else pd.DataFrame()
        log.warning("no Understat team data fetched")
    if player_frames:
        out["understat_player_match"] = pd.concat(player_frames, ignore_index=True)
        _merge_store(out["understat_player_match"], "understat_player_match", root)
    else:
        out["understat_player_match"] = pd.DataFrame()
    return out
