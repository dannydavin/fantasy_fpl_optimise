import pandas as pd
import pytest

from fpl import storage
from fpl.schemas import TABLES, SchemaError, conform, empty


def _players(gw, n=3):
    df = empty("players")
    df = pd.DataFrame({c: [None] * n for c in df.columns})
    df["gw"] = gw
    df["player_id"] = range(1, n + 1)
    df["web_name"] = [f"p{i}" for i in range(n)]
    df["price"] = [4.5, 5.0, 6.5]
    return df


def test_conform_missing_column():
    with pytest.raises(SchemaError):
        conform(pd.DataFrame({"gw": [1]}), "players")


def test_conform_casts_and_orders():
    df = _players(7)
    df["extra"] = 1
    out = conform(df, "players")
    assert list(out.columns)[: len(TABLES["players"].columns)] == list(TABLES["players"].columns)
    assert out.columns[-1] == "extra"
    assert str(out["player_id"].dtype) == "Int64"


def test_roundtrip_and_latest(data_root):
    storage.write_table(_players(6), "players", gw=6)
    p = storage.write_table(_players(7), "players", gw=7)
    assert p == data_root / "raw" / "gw07" / "players.parquet"
    assert storage.available_gws("players") == [6, 7]
    assert storage.read_table("players")["gw"].iloc[0] == 7
    assert len(storage.read_all_gws("players")) == 6


def test_duckdb_views(data_root):
    storage.write_table(_players(6), "players", gw=6)
    storage.write_table(_players(7), "players", gw=7)
    con = storage.connect()
    assert con.execute("select max(gw) from players").fetchone()[0] == 7
    rows = con.execute("select snapshot_gw, count(*) from players_all group by 1 order by 1").fetchall()
    assert rows == [(6, 3), (7, 3)]


def test_per_gw_requires_gw(data_root):
    with pytest.raises(ValueError):
        storage.table_path("players")
    with pytest.raises(FileNotFoundError):
        storage.read_table("players")


def test_season_table_path(data_root):
    assert storage.table_path("hist_player_gw") == data_root / "raw" / "hist_player_gw.parquet"
