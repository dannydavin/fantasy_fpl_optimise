# fantasy_fpl_optimise

Personal FPL decision toolkit for team **1608168**: pulls FPL/community data, tracks what the
top 10k managers do, projects points with a home-built baseline model, and plans transfers
with a MILP solver. Everything is viewed in a local Streamlit app.

## Setup

```bash
uv sync
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
| `src/fpl/optimise/` | MILP transfer solver |
| `app/` | Streamlit app |
| `scripts/update.py` | one-command data refresh |
| `tests/fixtures/` | recorded API responses (one real pull) for offline tests |

## Development

```bash
uv run pytest        # offline; live-API tests are marked `network` and deselected
uv run ruff check
```
