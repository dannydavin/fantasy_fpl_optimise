# fantasy_fpl_optimise

Personal FPL decision toolkit for team **1608168**: pulls FPL/community data, tracks what the
top 10k managers do, projects points with a home-built baseline model, and plans transfers
with a MILP solver. Everything is viewed in a local Streamlit app.

## Setup

```bash
uv sync
git submodule update --init   # external/ reference repos (needed for the cross-check)
```

Config lives in `config.toml` (`team_id`, `season`, rate limit). Any field can be
overridden with an `FPL_` env var, e.g. `FPL_TEAM_ID=123`.

## Layout

| Path | What |
|---|---|
| `src/fpl/api.py` | FPL API client — on-disk JSON cache (`data/cache/`), ~4 req/s throttle, retry/backoff |
| `src/fpl/schemas.py` | Column contracts for every parquet table |
| `src/fpl/storage.py` | `write_table` / `read_table` / DuckDB `connect()` over `data/<layer>/gwNN/<table>.parquet` |
| `src/fpl/ingest/` | snapshot, top-10k, news, history/Understat pulls |
| `src/fpl/features/` | ownership/EO, fixture ticker, form |
| `src/fpl/models/` | baseline xP |
| `src/fpl/optimise/` | MILP transfer solver (highspy) + FPL-Optimization-Tools wrapper |
| `app/` | Streamlit app |
| `scripts/update.py` | one-command data refresh |
| `tests/fixtures/` | recorded API responses (one real pull) for offline tests |

## Weekly usage

Run after each GW's deadline has passed and before the next one (prices change overnight,
so the day of the deadline is best for a final refresh).

```bash
uv run python scripts/update.py                        # snapshot, fixtures, form, news, xP (~15 s)
uv run python scripts/update.py --top10k --limit 10000 # + top-10k picks/ownership (~40 min first run)
uv run streamlit run app/Home.py                       # open the app
uv run python scripts/plan.py --cross-check           # solver plan + FPL-Optimization-Tools check
```

- `--top10k --limit 500` is a quick smoke test (a couple of minutes). Picks for a passed
  deadline are cached forever under `data/cache/`, so re-runs only fetch new entries.
- `--history` re-pulls the vaastav past-season CSVs and Understat (needed once per season;
  the xP model uses them for priors and team ratings).
- `--skip-news`, `--offline` (recorded fixtures, writes to `data/_offline`), `--data-dir DIR`.
- In the app, hit **Reload data** in the sidebar after re-running the update.

### Pages

| Page | What |
|---|---|
| Home | bank / FT / chips / rank, top projected players, flagged players |
| My Team | XI + bench with sell prices, xP for the next 6 GWs, fixtures, status and price-risk flags |
| Top 10k | my squad's ownership/EO vs the top 10k, rank-risk exposure, high-EO players I don't own, captaincy, risers/fallers, chip usage, template XI, top 100/1k/10k bands |
| Players | price vs xGI/90 scatter, filters, form, set-piece takers |
| Fixtures | difficulty heatmap (custom xG-based or FPL FDR), blank/double calendar |
| Transfer Planner | MILP solver over 1–6 GWs with lock/ban, hits, chips; optional cross-check with FPL-Optimization-Tools |
| News | FPL injury flags (mine first) and RSS feed tagged by player |

### Transfer solver from Python

```python
from fpl.optimise.solver import solve_from_storage, validate_plan

plan = solve_from_storage(horizon=4)          # latest planning GW, my squad/bank/FTs, baseline xP
plan.transfers, plan.summary, plan.lineups    # per-GW moves, FT/hit accounting, XI/bench/captain
plan = solve_from_storage(horizon=6, chips={9: "bboost"}, lock=[411], ban=[165], ft_value=1.5)
```

`solve()` takes any `(gw, player_id, xp)` frame, so external projections can be plugged in
(also via CSV upload on the Transfer Planner page). Rules: 2/5/5/3, ≤3 per club, sell prices
for current players, FTs roll over to 5, −4 per extra transfer, and a WC/FH week freezes the
FT count. `fpl.optimise.fplopt_wrapper.cross_check(...)` runs
`external/FPL-Optimization-Tools` on the same xP. It needs network access (the tool fetches
bootstrap-static itself) and Python 3.14, which `uv` fetches on demand: the wrapper shells out to
`uv run --python 3.14 --no-project --with pandas --with numpy --with requests --with fuzzywuzzy
--with matplotlib --with highspy --with tabulate python run/solve.py` in a temp copy of the tool.

## Development

```bash
uv run pytest        # offline; live-API tests are marked `network` and deselected
FPLOPT_RUN=1 uv run pytest tests/test_fplopt_wrapper.py   # also runs the external solver
uv run ruff check
```
