"""Parquet read/write helpers with path conventions, plus a DuckDB view layer.

Paths (``root`` defaults to ``settings.data_dir``):
- per-GW table:  ``<root>/<layer>/gwNN/<table>.parquet``
- season table:  ``<root>/<layer>/<table>.parquet``

All writes go through :func:`fpl.schemas.conform`, so on-disk tables always satisfy
their contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from fpl.config import get_settings
from fpl.schemas import TABLES, conform, get_schema

_GW_DIR = re.compile(r"^gw(\d{2,})$")


def data_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else get_settings().data_dir


def gw_dirname(gw: int) -> str:
    return f"gw{int(gw):02d}"


def table_path(table: str, gw: int | None = None, root: Path | None = None) -> Path:
    schema = get_schema(table)
    base = data_root(root) / schema.layer
    if schema.per_gw:
        if gw is None:
            raise ValueError(f"{table} is per-GW; pass gw")
        return base / gw_dirname(gw) / f"{table}.parquet"
    return base / f"{table}.parquet"


def write_table(
    df: pd.DataFrame, table: str, gw: int | None = None, root: Path | None = None
) -> Path:
    """Validate against the schema and write atomically. Returns the written path."""
    path = table_path(table, gw, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = conform(df, table)
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def available_gws(table: str, root: Path | None = None) -> list[int]:
    """Sorted GWs for which ``table`` has been written."""
    schema = get_schema(table)
    base = data_root(root) / schema.layer
    if not schema.per_gw or not base.exists():
        return []
    gws = []
    for d in base.iterdir():
        m = _GW_DIR.match(d.name)
        if m and (d / f"{table}.parquet").exists():
            gws.append(int(m.group(1)))
    return sorted(gws)


def latest_gw(table: str, root: Path | None = None) -> int | None:
    gws = available_gws(table, root)
    return gws[-1] if gws else None


def exists(table: str, gw: int | None = None, root: Path | None = None) -> bool:
    schema = get_schema(table)
    if schema.per_gw and gw is None:
        return latest_gw(table, root) is not None
    return table_path(table, gw, root).exists()


def read_table(table: str, gw: int | None = None, root: Path | None = None) -> pd.DataFrame:
    """Read a table. For per-GW tables ``gw=None`` means the latest available GW."""
    schema = get_schema(table)
    if schema.per_gw and gw is None:
        gw = latest_gw(table, root)
        if gw is None:
            raise FileNotFoundError(f"no {table} written under {data_root(root) / schema.layer}")
    path = table_path(table, gw, root)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def read_all_gws(table: str, root: Path | None = None) -> pd.DataFrame:
    """Concatenate a per-GW table across all GW dirs, adding ``snapshot_gw``."""
    frames = []
    for gw in available_gws(table, root):
        df = read_table(table, gw, root)
        df["snapshot_gw"] = gw
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {table} written")
    return pd.concat(frames, ignore_index=True)


def connect(root: Path | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a view per table.

    ``<table>`` -> latest GW (per-GW tables) or the season file;
    ``<table>_all`` -> every GW, with a ``snapshot_gw`` column.
    """
    con = duckdb.connect()
    for name, schema in TABLES.items():
        if schema.per_gw:
            gw = latest_gw(name, root)
            if gw is None:
                continue
            latest = table_path(name, gw, root).as_posix()
            glob = (data_root(root) / schema.layer / "gw*" / f"{name}.parquet").as_posix()
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{latest}')")
            con.execute(
                f"CREATE VIEW {name}_all AS SELECT *, "
                f"CAST(regexp_extract(filename, 'gw(\\d+)', 1) AS INTEGER) AS snapshot_gw "
                f"FROM read_parquet('{glob}', filename=true, union_by_name=true)"
            )
        else:
            path = table_path(name, root=root)
            if path.exists():
                con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    return con
