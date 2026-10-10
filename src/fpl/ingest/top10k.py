"""Top-10k ingest: overall league 314 standings -> per-entry picks for the last passed deadline.

Resumable: ``entry_picks(final=True)`` is cached forever by the client, so a re-run after an
interruption only fetches entries not yet cached. A small progress checkpoint is also written
to ``<cache_dir>/top10k_progress_gwNN.json``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fpl import gameweeks, storage
from fpl.api import FPLAPIError, FPLClient
from fpl.schemas import empty

log = logging.getLogger(__name__)

PAGE_SIZE = 50
LOG_EVERY = 250


def parse_standings(page: dict, gw: int) -> pd.DataFrame:
    """One standings page -> top10k_entries rows."""
    rows = [
        {
            "gw": gw,
            "entry_id": r["entry"],
            "rank": r["rank"],
            "total": r["total"],
            "entry_name": r.get("entry_name"),
            "player_name": r.get("player_name"),
            "event_total": r.get("event_total"),
        }
        for r in page["standings"]["results"]
    ]
    return pd.DataFrame(rows)


def parse_picks(picks: dict, entry_id: int, rank: int, gw: int) -> pd.DataFrame:
    """One picks response -> 15 top10k_picks rows."""
    eh = picks.get("entry_history") or {}
    chip = picks.get("active_chip")
    rows = [
        {
            "gw": gw,
            "entry_id": entry_id,
            "rank": rank,
            "player_id": p["element"],
            "slot": p["position"],
            "multiplier": p["multiplier"],
            "is_captain": bool(p["is_captain"]),
            "is_vice_captain": bool(p["is_vice_captain"]),
            "active_chip": chip,
            "event_transfers": eh.get("event_transfers", 0),
            "event_transfers_cost": eh.get("event_transfers_cost", 0),
            "bank": (eh.get("bank") or 0) / 10,
            "value": (eh.get("value") or 0) / 10,
            "event_points": eh.get("points"),
            "points_on_bench": eh.get("points_on_bench"),
        }
        for p in picks["picks"]
    ]
    return pd.DataFrame(rows)


def fetch_standings(client: FPLClient, limit: int, gw: int) -> pd.DataFrame:
    league = client.settings.overall_league_id
    frames = []
    seen: set[int] = set()
    # Standings shift while a GW is live, so entries can repeat across pages; keep paging
    # (bounded) until `limit` unique entries are collected.
    max_pages = math.ceil(limit / PAGE_SIZE) + 2
    for page in range(1, max_pages + 1):
        data = client.league_standings(league, page=page)
        df = parse_standings(data, gw)
        frames.append(df)
        seen.update(df["entry_id"].tolist())
        if len(seen) >= limit or not data["standings"].get("has_next"):
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates("entry_id")
    return df.sort_values("rank").head(limit).reset_index(drop=True)


def _checkpoint(client: FPLClient, gw: int, done: int, total: int, failed: int) -> None:
    try:
        p = Path(client.cache_dir) / f"top10k_progress_gw{gw:02d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"done": done, "total": total, "failed": failed, "ts": time.time()}))
    except OSError:  # progress is best-effort
        pass


def fetch_all_picks(
    client: FPLClient, entries: pd.DataFrame, gw: int, workers: int = 4
) -> tuple[pd.DataFrame, list[int]]:
    """Fetch picks for every entry; returns (picks frame, failed entry ids)."""
    ranks = dict(zip(entries["entry_id"], entries["rank"], strict=True))
    total = len(ranks)
    frames: list[pd.DataFrame] = []
    failed: list[int] = []
    start = time.time()

    def one(eid: int):
        return eid, client.entry_picks(eid, gw, final=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, int(e)): int(e) for e in ranks}
        for done, fut in enumerate(as_completed(futs), 1):
            eid = futs[fut]
            try:
                _, data = fut.result()
                frames.append(parse_picks(data, eid, int(ranks[eid]), gw))
            except (FPLAPIError, KeyError) as exc:
                log.warning("skipping entry %s: %s", eid, exc)
                failed.append(eid)
            if done % LOG_EVERY == 0 or done == total:
                elapsed = time.time() - start
                eta = elapsed / done * (total - done)
                log.info("top10k picks %d/%d (failed %d) eta %.0fs", done, total, len(failed), eta)
                _checkpoint(client, gw, done, total, len(failed))
    picks = pd.concat(frames, ignore_index=True) if frames else empty("top10k_picks")
    return picks, sorted(failed)


def run(
    client: FPLClient,
    limit: int = 10000,
    gw: int | None = None,
    root=None,
    workers: int = 4,
) -> int:
    """Pull top-`limit` entries' picks for ``gw`` (default: last passed deadline). Returns gw."""
    if gw is None:
        gw = gameweeks.last_deadline_gw(client.bootstrap_static())
        if gw is None:
            raise RuntimeError("no gameweek deadline has passed yet; picks are not public")
    entries = fetch_standings(client, limit, gw)
    picks, failed = fetch_all_picks(client, entries, gw, workers)
    ok = entries[~entries["entry_id"].isin(failed)].copy()
    if not picks.empty:
        extra = picks.groupby("entry_id").agg(
            event_points=("event_points", "first"),
            points_on_bench=("points_on_bench", "first"),
            active_chip=("active_chip", "first"),
        )
        ok = ok.merge(extra, left_on="entry_id", right_index=True, how="left")
    ok["limit"] = limit
    ok["n_sampled"] = len(ok)
    storage.write_table(ok, "top10k_entries", gw, root)
    storage.write_table(picks, "top10k_picks", gw, root)
    log.info("top10k gw%d: %d entries ok, %d failed %s", gw, len(ok), len(failed), failed[:10])
    return gw
