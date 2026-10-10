"""One-command weekly refresh: snapshot -> fixtures -> form -> news -> xP [-> top10k -> ownership].

    uv run python scripts/update.py                       # light refresh
    uv run python scripts/update.py --top10k --limit 500  # smoke test of the heavy pull
    uv run python scripts/update.py --offline             # dry run against tests/fixtures
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without the package installed
    sys.path.insert(0, str(ROOT / "src"))


class Step:
    """Run ``module.func`` lazily; absent modules/functions are reported as skipped."""

    def __init__(self, name: str):
        self.name = name
        self.status = "pending"

    def call(self, module: str, func: str, *args, **kwargs):
        try:
            fn = getattr(importlib.import_module(module), func)
        except (ImportError, AttributeError):
            self.status = "skipped (not available)"
            print(f"[{self.name}] skipped (not available)")
            return None
        t0 = time.time()
        try:
            res = fn(*args, **kwargs)
        except Exception as exc:  # one failing source must not kill the run
            self.status = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[{self.name}] FAILED: {exc}")
            traceback.print_exc()
            return None
        self.status = f"ok ({time.time() - t0:.1f}s)"
        print(f"[{self.name}] {self.status}")
        return res


def _summary(steps: list[Step], gw: int | None, root) -> None:
    from fpl import storage

    print("\n=== summary ===")
    for s in steps:
        print(f"{s.name:<12} {s.status}")
    if gw is None:
        return
    base = storage.data_root(root)
    print(f"\ndata under {base} (snapshot gw {gw}):")
    for layer in ("raw", "processed"):
        d = base / layer / storage.gw_dirname(gw)
        if d.exists():
            for p in sorted(d.glob("*.parquet")):
                print(f"  {p.relative_to(base)}")
    try:
        entry = storage.read_table("my_entry", gw, root).iloc[0]
        team = storage.read_table("my_team", gw, root)
        players = storage.read_table("players", gw, root).set_index("player_id")
        names = [players.loc[i, "web_name"] for i in team.sort_values("slot")["player_id"]]
        print(f"\nteam: {entry['entry_name']}  bank £{entry['bank']:.1f}m  "
              f"FT {entry['free_transfers']}  chips: {entry['chips_available'] or '-'}")
        print("squad: " + ", ".join(names))
    except Exception as exc:
        print(f"(could not summarise my team: {exc})")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--top10k", action="store_true", help="also pull top-10k picks + ownership")
    ap.add_argument("--limit", type=int, default=10000, help="top-10k entries to pull")
    ap.add_argument("--workers", type=int, default=4, help="top-10k worker threads")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--history", action="store_true", help="run vaastav/understat backfill")
    ap.add_argument("--offline", action="store_true", help="use recorded fixtures, no network")
    ap.add_argument("--data-dir", type=Path, help="override data directory")
    args = ap.parse_args(argv)

    from fpl.api import FPLClient, offline_client
    from fpl.config import get_settings

    root = args.data_dir
    if args.offline and root is None:
        root = get_settings().data_dir / "_offline"  # keep recorded data apart from real pulls

    client = offline_client() if args.offline else FPLClient()
    steps: list[Step] = []
    gw: int | None = None
    try:
        if args.history:
            s = Step("history")
            steps.append(s)
            s.call("fpl.ingest.history", "run", root=root)
            s = Step("understat")
            steps.append(s)
            s.call("fpl.ingest.understat", "run", root=root)

        s = Step("snapshot")
        steps.append(s)
        gw = s.call("fpl.ingest.fpl_snapshot", "run", client, root=root)
        if gw is None:
            print("snapshot failed; later steps depend on it")
            return 1

        s = Step("fixtures")
        steps.append(s)
        s.call("fpl.features.fixtures", "run", gw=gw, root=root, client=client)
        s = Step("form")
        steps.append(s)
        s.call("fpl.features.form", "run", gw=gw, root=root, client=client)

        s = Step("news")
        steps.append(s)
        if args.skip_news:
            s.status = "skipped (--skip-news)"
        else:
            s.call("fpl.ingest.news", "run", client, gw=gw, root=root)

        s = Step("xp")
        steps.append(s)
        s.call("fpl.models.xp_baseline", "run", gw=gw, root=root)

        if args.top10k:
            s = Step("top10k")
            steps.append(s)
            picks_gw = s.call(
                "fpl.ingest.top10k", "run", client,
                limit=args.limit, root=root, workers=args.workers,
            )
            s = Step("ownership")
            steps.append(s)
            s.call("fpl.features.ownership", "run", gw=picks_gw, root=root)
    finally:
        client.close()
        _summary(steps, gw, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
