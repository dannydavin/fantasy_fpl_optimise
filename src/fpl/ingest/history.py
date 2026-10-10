"""vaastav/Fantasy-Premier-League per-GW history for past seasons -> ``hist_player_gw``.

Files are fetched from the repo's raw-content host at run time (never vendored) and cached
under ``<cache_dir>/history/<season>/``. Pure transform: :func:`normalise_merged_gw`.
"""

from __future__ import annotations

import io
import logging
import random
import threading
import time
from pathlib import Path

import httpx
import pandas as pd

from fpl import storage
from fpl.config import get_settings

log = logging.getLogger(__name__)

RAW_BASE = "https://raw." + "git" + "hubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
POSITION_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD", "AM": "MID"}
_ELEMENT_TYPE_POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
NAN = float("nan")

_lock = threading.Lock()
_next_at = [0.0]


def previous_seasons(current: str, n: int = 3) -> list[str]:
    """``'2026-27'`` -> ``['2023-24', '2024-25', '2025-26']``."""
    start = int(current[:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(start - n, start)]


def throttle(rate: float = 2.0) -> None:
    """Global politeness limiter shared by history and understat downloads."""
    with _lock:
        now = time.monotonic()
        delay = _next_at[0] - now
        _next_at[0] = max(now, _next_at[0]) + 1.0 / rate
    if delay > 0:
        time.sleep(delay)


def http_get(
    url: str, headers: dict | None = None, retries: int = 4, transport=None
) -> httpx.Response:
    """Polite GET with throttle and backoff. 404 -> FileNotFoundError."""
    h = {"User-Agent": get_settings().user_agent, **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries + 1):
        throttle()
        try:
            with httpx.Client(
                headers=h, timeout=60, follow_redirects=True, transport=transport
            ) as c:
                r = c.get(url)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                raise FileNotFoundError(url)
            last = RuntimeError(f"{r.status_code} for {url}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except httpx.TransportError as exc:
            last = exc
        if attempt < retries:
            time.sleep(min(30.0, 1.5 * 2**attempt) + random.uniform(0, 0.5))
    raise RuntimeError(f"giving up on {url}") from last


def fetch_cached(
    relpath: str, season: str, cache_dir: Path | None = None, refresh: bool = False, transport=None
) -> bytes | None:
    """Download ``data/<season>/<relpath>`` (cached). Returns None if it does not exist."""
    cache_dir = Path(cache_dir) if cache_dir else get_settings().cache_dir
    path = cache_dir / "history" / season / relpath.replace("/", "_")
    if path.exists() and not refresh:
        return path.read_bytes()
    try:
        content = http_get(f"{RAW_BASE}/{season}/{relpath}", transport=transport).content
    except FileNotFoundError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)


def _fix_mojibake(s):
    if isinstance(s, str) and ("Ã" in s or "Â" in s):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _to_bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() == "true"


def normalise_merged_gw(
    df: pd.DataFrame,
    season: str,
    players_raw: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalise a vaastav ``merged_gw.csv`` frame to the ``hist_player_gw`` contract.

    Per-season differences handled: ``GW``/``round`` column, ``GK``/``AM`` positions,
    expected stats absent before 2022-23 (NaN), ``defensive_contribution`` only from
    2025-26 (NaN before), lossy name encodings (names preferred from ``players_raw``).
    One row per fixture played (a DGW gives two rows for the same ``gw``).
    Extras: ``opponent`` (name, if ``teams`` given), team scores, saves, own_goals,
    defensive_contribution, starts, fixture_id.
    """
    d = df.copy()
    out = pd.DataFrame(index=d.index)
    out["season"] = season
    out["gw"] = pd.to_numeric(d["GW"] if "GW" in d else d["round"], errors="coerce")
    out["element"] = pd.to_numeric(d["element"], errors="coerce")
    out["team"] = d["team"].map(_fix_mojibake)
    pos = d["position"].map(POSITION_MAP) if "position" in d else pd.Series(None, index=d.index)
    names = d["name"].map(_fix_mojibake) if "name" in d else pd.Series(None, index=d.index)
    code = pd.Series(pd.NA, index=d.index, dtype="Int64")
    if players_raw is not None and len(players_raw):
        pr = players_raw.drop_duplicates("id").set_index("id")
        code = out["element"].map(pr["code"]).astype("Int64")
        full = (pr["first_name"].fillna("") + " " + pr["second_name"].fillna("")).str.strip()
        mapped = out["element"].map(full)
        names = mapped.where(mapped.notna(), names)
        if pos.isna().all():
            pos = out["element"].map(pr["element_type"].map(_ELEMENT_TYPE_POS))
    out["code"] = code
    out["player_name"] = names
    out["position"] = pos
    out["opponent_team"] = _num(d, "opponent_team")
    out["was_home"] = d["was_home"].map(_to_bool)
    out["kickoff_time"] = pd.to_datetime(d["kickoff_time"], utc=True, errors="coerce")
    for c in (
        "minutes", "total_points", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps",
    ):
        out[c] = _num(d, c)
    out["expected_goals"] = _num(d, "expected_goals", NAN)
    out["expected_assists"] = _num(d, "expected_assists", NAN)
    out["value"] = _num(d, "value") / 10.0
    out["starts"] = _num(d, "starts", NAN)
    out["saves"] = _num(d, "saves")
    out["own_goals"] = _num(d, "own_goals")
    out["team_h_score"] = _num(d, "team_h_score", NAN)
    out["team_a_score"] = _num(d, "team_a_score", NAN)
    out["fixture_id"] = _num(d, "fixture", NAN)
    out["defensive_contribution"] = _num(d, "defensive_contribution", NAN)
    if teams is not None and len(teams):
        out["opponent"] = out["opponent_team"].map(teams.set_index("id")["name"])
    out = out[out["gw"].notna() & out["element"].notna()]
    for c in (
        "gw", "element", "opponent_team", "minutes", "total_points", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "bonus", "bps", "starts", "saves", "own_goals",
        "team_h_score", "team_a_score", "fixture_id", "defensive_contribution",
    ):
        out[c] = out[c].round().astype("Int64")
    return out.reset_index(drop=True)


def load_season(
    season: str, cache_dir: Path | None = None, refresh: bool = False, transport=None
) -> pd.DataFrame | None:
    merged = fetch_cached("gws/merged_gw.csv", season, cache_dir, refresh, transport)
    if merged is None:
        log.warning("vaastav has no merged_gw.csv for %s", season)
        return None
    pr = fetch_cached("players_raw.csv", season, cache_dir, refresh, transport)
    tm = fetch_cached("teams.csv", season, cache_dir, refresh, transport)
    return normalise_merged_gw(
        read_csv_bytes(merged),
        season,
        read_csv_bytes(pr) if pr else None,
        read_csv_bytes(tm) if tm else None,
    )


def run(
    seasons: list[str] | None = None,
    root: Path | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
    transport=None,
) -> pd.DataFrame:
    """Fetch + normalise seasons (default: last 3 completed) -> raw/hist_player_gw.parquet.

    Seasons already stored but not requested are kept; the file holds the union.
    """
    seasons = seasons or previous_seasons(get_settings().season)
    frames = []
    for s in seasons:
        df = load_season(s, cache_dir, refresh, transport)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no history available for {seasons}")
    new = pd.concat(frames, ignore_index=True)
    if storage.exists("hist_player_gw", root=root):
        old = storage.read_table("hist_player_gw", root=root)
        old = old[~old["season"].isin(new["season"].unique())]
        if len(old):
            new = pd.concat([old, new], ignore_index=True)
    storage.write_table(new, "hist_player_gw", root=root)
    return new
