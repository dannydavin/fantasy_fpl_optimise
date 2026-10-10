"""Crowd pages: Top 10k and News."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def _page(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "app/pages" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(page):
    at = AppTest.from_file(str(ROOT / "app/pages" / page), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_squad_ownership_zero_fill():
    m = _page("2_Top_10k.py")
    squad = pd.DataFrame({"player_id": [1, 2], "multiplier": [2, 1]})
    own = pd.DataFrame({"player_id": [1], "owned_pct": [50.0], "start_pct": [40.0],
                        "captain_pct": [10.0], "eo": [60.0]})
    out = m.squad_ownership(squad, own)
    assert len(out) == 2
    assert out.loc[1, "eo"] == 0 and out.loc[1, "exposure"] == 100
    assert out.loc[0, "exposure"] == 140


def test_group_items_and_filters():
    m = _page("6_News.py")
    news = pd.DataFrame({
        "item_id": ["a", "a", "b"], "published": pd.to_datetime(["2026-10-01"] * 3, utc=True),
        "source": ["BBC", "BBC", "G"], "kind": ["rss"] * 3, "title": ["t1", "t1", "t2"],
        "summary": ["s", "s", None], "url": ["u", "u", None], "player_id": [1, 2, None],
        "player_name": ["A", "B", None],
    })
    g = m.group_items(news, {2})
    assert len(g) == 2 and g.iloc[0]["players"] == "A, B" and g.iloc[0]["mine"]
    assert len(m.filter_items(g, ["BBC", "G"], ["rss"], "", True)) == 1
    assert len(m.filter_items(g, ["BBC", "G"], ["rss"], "t2", False)) == 1


@pytest.mark.parametrize("page", ["2_Top_10k.py", "6_News.py"])
def test_pages_empty(data_root, page):
    at = _run(page)
    assert at.info


def test_top10k_page(ui_data):
    at = _run("2_Top_10k.py")
    first = at.dataframe[0].value
    assert len(first) == 15 and "Haaland" in set(first["web_name"])
    assert first["owned_pct"].notna().all()


def test_news_page(ui_data):
    at = _run("6_News.py")
    text = " ".join(h.value for h in at.header)
    assert "My squad flags" in text and "League-wide flags" in text
    md = " ".join(x.value for x in at.markdown)
    assert "Haaland" in md
