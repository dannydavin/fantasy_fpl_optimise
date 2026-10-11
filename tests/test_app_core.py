"""Core Streamlit pages (AppTest) and the pure helpers added to app/components.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import components  # noqa: E402

PAGES = ["app/Home.py", "app/pages/1_My_Team.py", "app/pages/3_Players.py",
         "app/pages/4_Fixtures.py"]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_data(ui_data, page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_empty(data_root, page):
    import streamlit as st

    st.cache_data.clear()
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.info


def test_home_content(ui_data):
    at = _run("app/Home.py")
    assert any("Gunners" in h.proto.body for h in at.get("html"))
    assert len(at.metric) >= 5


def test_my_team_shows_15(ui_data):
    at = _run("app/pages/1_My_Team.py")
    assert not at.exception
    pitch = " ".join(h.proto.body for h in at.get("html"))
    assert "Haaland" in pitch and "shirt_" in pitch
    assert len(components.my_squad()) == 15


def test_players_and_fixtures_widgets(ui_data):
    at = _run("app/pages/3_Players.py")
    assert not at.exception
    at.sidebar.radio[0].set_value(4).run()
    assert not at.exception
    at = _run("app/pages/4_Fixtures.py")
    at.toggle[0].set_value(True).run()
    assert not at.exception


def test_difficulty_colour():
    assert components.difficulty_colour(1) == components.DIFFICULTY_COLOURS[0]
    assert components.difficulty_colour(4.6) == components.DIFFICULTY_COLOURS[4]
    assert components.difficulty_colour(99) == components.DIFFICULTY_COLOURS[4]
    assert components.difficulty_colour(float("nan")) == components.BLANK_COLOUR
    assert components.difficulty_colour(None) == components.BLANK_COLOUR


def test_fixture_cells_dgw_and_blank():
    ticker = pd.DataFrame({
        "team_id": [1, 1, 1, 2],
        "gw": [7, 8, 8, 7],
        "opponent_id": [2, 2, 3, 1],
        "is_home": [True, False, True, False],
        "fdr": [2, 3, 4, 3],
        "difficulty": [2.0, 3.0, 5.0, 3.0],
        "fixture_id": [1, 2, 3, 1],
    })
    shorts = {1: "ARS", 2: "AVL", 3: "BOU"}
    c = components.fixture_cells(ticker, shorts, 7, 3)
    get = lambda t, g: c[(c.team_id == t) & (c.gw == g)].iloc[0]  # noqa: E731
    assert get(1, 7).label == "AVL (H)"
    assert get(1, 8).label == "AVL (A) + BOU (H)"
    assert get(1, 8).value == 4.0 and get(1, 8).n_fixtures == 2
    assert get(1, 9).label == "-" and pd.isna(get(1, 9).value)
    assert get(2, 8).label == "-"
    assert components.team_fixture_labels(ticker, 1, shorts, 7, 3) == [
        "AVL (H)", "AVL (A) + BOU (H)", "-"]
    assert components.fixture_cells(ticker.iloc[0:0], shorts, 7, 3).empty
