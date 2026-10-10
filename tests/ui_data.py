"""Synthetic-but-realistic data dir for offline UI / solver tests.

``write_ui_data(root)`` runs the real snapshot -> fixtures -> form -> xp pipeline against the
recorded API fixtures (GW7 planning, my real 15), then adds:
- news: FPL flags plus two synthetic RSS items (one tagged Haaland, one untagged);
- top10k_entries / top10k_picks for GW5 and GW6: N deterministic random legal squads that
  favour high-``selected_by_percent`` players (so my squad players appear), with captains,
  bench, and a few chips;
- ownership for GW5 and GW6 (GW6 has delta_vs_prev_gw).

Use via the ``ui_data`` fixture in conftest (sets FPL_DATA_DIR) or call directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fpl import storage
from fpl.api import offline_client
from fpl.schemas import MAX_PER_TEAM, SQUAD_COUNTS

PICKS_GWS = (5, 6)
N_ENTRIES = 60
CHIPS = ["wildcard", "freehit", "bboost", "3xc"]


def _random_squad(players: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Legal 15 (2/5/5/3, <=3/club), sampled with weight ~ selected_by_percent."""
    w = players["selected_by_percent"].fillna(0).clip(lower=0.05).to_numpy() ** 1.5
    order = rng.choice(len(players), size=len(players), replace=False, p=w / w.sum())
    counts = dict.fromkeys(SQUAD_COUNTS, 0)
    per_team: dict[int, int] = {}
    chosen = []
    for i in order:
        r = players.iloc[i]
        if counts[r.position] >= SQUAD_COUNTS[r.position]:
            continue
        if per_team.get(r.team_id, 0) >= MAX_PER_TEAM:
            continue
        counts[r.position] += 1
        per_team[r.team_id] = per_team.get(r.team_id, 0) + 1
        chosen.append(i)
        if len(chosen) == 15:
            break
    return players.iloc[chosen]


def _entry_picks(sq: pd.DataFrame, entry_id: int, rank: int, gw: int, chip: str | None) -> list[dict]:
    """Order as XI (1 GKP, 4 DEF, 4 MID, 2 FWD by selection) then bench; captain = top FWD/MID."""
    sq = sq.sort_values("selected_by_percent", ascending=False)
    by = {p: sq[sq["position"] == p] for p in SQUAD_COUNTS}
    xi = pd.concat([by["GKP"].head(1), by["DEF"].head(4), by["MID"].head(4), by["FWD"].head(2)])
    bench = pd.concat([by["GKP"].iloc[1:], sq.drop(xi.index).query("position != 'GKP'")])
    attackers = xi[xi["position"].isin(["MID", "FWD"])]
    cap, vice = attackers["player_id"].iloc[0], attackers["player_id"].iloc[1]
    rows = []
    for slot, r in enumerate(pd.concat([xi, bench]).itertuples(), 1):
        mult = 1 if slot <= 11 else 0
        if chip == "bboost":
            mult = 1
        if r.player_id == cap:
            mult = 3 if chip == "3xc" else 2
        rows.append({
            "gw": gw, "entry_id": entry_id, "rank": rank, "player_id": int(r.player_id),
            "slot": slot, "multiplier": mult, "is_captain": r.player_id == cap,
            "is_vice_captain": r.player_id == vice, "active_chip": chip,
            "event_transfers": 1, "event_transfers_cost": 0, "bank": 0.5, "value": 102.0,
        })
    return rows


def synthetic_top10k(players: pd.DataFrame, gw: int, n: int = N_ENTRIES, seed: int = 0):
    rng = np.random.default_rng(seed + gw)
    pool = players[players["status"].fillna("a") != "u"]
    entries, picks = [], []
    for k in range(n):
        eid, rank = 9_000_000 + k, k + 1
        chip = CHIPS[k % len(CHIPS)] if k % 10 == 0 else None
        picks += _entry_picks(_random_squad(pool, rng), eid, rank, gw, chip)
        entries.append({"gw": gw, "entry_id": eid, "rank": rank, "total": 500 - k,
                        "entry_name": f"Team {k}", "player_name": f"Manager {k}"})
    return pd.DataFrame(entries), pd.DataFrame(picks)


def synthetic_news_rows(players: pd.DataFrame) -> list[dict]:
    h = players[players["web_name"] == "Haaland"].iloc[0]
    now = pd.Timestamp("2026-10-10T12:00:00Z")
    base = {"kind": "rss", "status": None, "chance_of_playing_next_round": None}
    return [
        {**base, "item_id": "synthetic-haaland", "published": now, "source": "BBC Sport",
         "title": "Haaland hat-trick fires City top", "summary": "Haaland scored three.",
         "url": "https://example.com/haaland", "player_id": int(h.player_id),
         "player_name": "Haaland", "team_id": int(h.team_id), "team_ids": str(int(h.team_id))},
        {**base, "item_id": "synthetic-untagged", "published": now - pd.Timedelta(hours=2),
         "source": "The Guardian", "title": "Premier League weekend preview",
         "summary": "All the talking points.", "url": "https://example.com/preview",
         "player_id": None, "player_name": None, "team_id": None, "team_ids": None},
    ]


def write_ui_data(root: Path, n_entries: int = N_ENTRIES) -> Path:
    """Populate ``root`` with every table the app/solver read. Returns ``root``."""
    from fpl.features import fixtures, form, ownership
    from fpl.ingest import fpl_snapshot, news
    from fpl.models import xp_baseline

    root = Path(root)
    with offline_client() as client:
        gw = fpl_snapshot.run(client, root=root)
        fixtures.run(gw=gw, root=root, client=client)
        form.run(gw=gw, root=root, client=client)
        xp_baseline.run(gw=gw, root=root)
        nd = news.run(client, gw=gw, root=root, sources=[])
    players = storage.read_table("players", gw, root)
    nd = news.build_table(nd.to_dict("records") + synthetic_news_rows(players))
    storage.write_table(nd, "news", gw, root)
    for pgw in PICKS_GWS:
        entries, picks = synthetic_top10k(players, pgw, n_entries)
        storage.write_table(entries, "top10k_entries", pgw, root)
        storage.write_table(picks, "top10k_picks", pgw, root)
        ownership.run(gw=pgw, root=root)
    return root


if __name__ == "__main__":  # python tests/ui_data.py <dir>  -> data dir for manual app checks
    import sys

    print(write_ui_data(Path(sys.argv[1] if len(sys.argv) > 1 else "data/_ui")))
