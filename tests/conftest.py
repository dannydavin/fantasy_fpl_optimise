"""Shared offline fixtures. Tests never hit the network unless marked ``network``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl.api import FIXTURES_DIR, offline_client
from fpl.config import get_settings

TEAM_ID = 1608168
PICKS_GW = 6  # recorded picks are for GW6; GW7 is the next deadline in the fixtures
NEXT_GW = 7


def load_fixture(name: str):
    """Load ``tests/fixtures/<name>.json`` (name = api.cache_key of the endpoint)."""
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def client():
    """FPLClient that serves recorded responses from tests/fixtures (no network, no cache)."""
    with offline_client() as c:
        yield c


@pytest.fixture(scope="session")
def bootstrap() -> dict:
    return load_fixture("bootstrap-static")


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch) -> Path:
    """Point settings.data_dir at a temp dir so storage reads/writes are isolated."""
    monkeypatch.setenv("FPL_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield tmp_path / "data"
    get_settings.cache_clear()
