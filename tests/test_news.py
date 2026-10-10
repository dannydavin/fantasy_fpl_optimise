"""Offline tests for fpl.ingest.news (recorded feeds in tests/fixtures/news)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest

from fpl import storage
from fpl.ingest import news
from tests.conftest import NEXT_GW, load_fixture

FIX = Path(__file__).parent / "fixtures" / "news"

BY_URL = {
    "https://www.fantasyfootballscout.co.uk/feed/": "fantasyfootballscout.xml",
    "https://feeds.bbci.co.uk/sport/football/rss.xml": "bbc_football.xml",
    "https://www.skysports.com/rss/12040": "skysports_football.xml",
    "https://www.theguardian.com/football/rss": "guardian_football.xml",
    "https://www.reddit.com/r/FantasyPL/.rss": "reddit_fantasypl.xml",
}


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "hot.json" in url:
        return httpx.Response(403, text="blocked")
    name = BY_URL.get(url)
    if name is None:
        return httpx.Response(404)
    return httpx.Response(200, content=(FIX / name).read_bytes())


@pytest.fixture(autouse=True)
def _fast_http(monkeypatch):
    monkeypatch.setattr(news, "HTTP_RATE", 1000.0)


@pytest.fixture(scope="module")
def bs():
    return load_fixture("bootstrap-static")


@pytest.fixture(scope="module")
def tagger(bs):
    return news.PlayerTagger(bs)


def names(tagger, text, fpl=False):
    pids, _ = tagger.tag(text, fpl)
    return [tagger.players[p]["web_name"] for p in pids]


# ---- parsing -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fname,kind",
    [("fantasyfootballscout.xml", "rss"), ("bbc_football.xml", "rss"),
     ("skysports_football.xml", "rss"), ("guardian_football.xml", "rss"),
     ("reddit_fantasypl.xml", "reddit")],
)
def test_parse_feed_samples(fname, kind):
    items = news.parse_feed((FIX / fname).read_bytes(), "src", kind)
    assert 1 <= len(items) <= 12
    for it in items:
        assert it["title"] and "<" not in it["title"] and "<" not in it["summary"]
        assert it["kind"] == kind and len(it["item_id"]) == 40
        assert pd.isna(it["published"]) or it["published"].tzinfo is not None


def test_parse_reddit_json():
    items = news.parse_reddit_json((FIX / "reddit_fantasypl.json").read_bytes())
    assert len(items) == 5
    assert items[1]["summary"] == "[Injury] Arsenal's Ødegaard was seen in training & looked fine."
    assert items[0]["url"].startswith("https://www.reddit.com/r/FantasyPL/")
    assert items[0]["published"].tzinfo is not None


def test_strip_html():
    assert news.strip_html("<p>Hi&nbsp;<b>there</b> &amp; you</p><script>x()</script>") == (
        "Hi there & you"
    )


# ---- FPL flags -----------------------------------------------------------------------


def test_fpl_flags(bs):
    items = news.fpl_flag_items(bs)
    flagged = [e for e in bs["elements"] if (e["news"] or "").strip()]
    assert len(items) == len(flagged) > 0
    it = items[0]
    e = next(e for e in flagged if e["id"] == it["player_id"])
    short = next(t["short_name"] for t in bs["teams"] if t["id"] == e["team"])
    assert it["title"] == f"{e['web_name']} ({short}): {e['news']}"
    assert it["kind"] == "fpl_flag" and it["source"] == "FPL" and it["url"] is None
    assert f"Status: {e['status']}" in it["summary"]
    assert it["published"].tzinfo is not None


# ---- tagging -------------------------------------------------------------------------


def test_accents(tagger):
    assert names(tagger, "Odegaard and Guehi both start; Gross benched") == [
        "Ødegaard", "Guéhi", "Groß"]
    assert names(tagger, "ØDEGAARD, Guéhi, Groß") == ["Ødegaard", "Guéhi", "Groß"]
    assert names(tagger, "Joao Pedro scores again") == ["João Pedro"]
    assert names(tagger, "João Pedro's brace") == ["João Pedro"]


def test_word_boundaries(tagger):
    assert names(tagger, "Haalandesque finish") == []
    assert names(tagger, "Haaland's hat-trick") == ["Haaland"]


def test_ambiguous_surname_needs_team(tagger):
    # two players called King: Everton (Tom) and Fulham (Josh)
    assert names(tagger, "King is a doubt") == []
    assert names(tagger, "Everton's King is a doubt") == ["King"]
    pids, _ = tagger.tag("Everton's King is a doubt")
    assert tagger.players[pids[0]]["first_name"] == "Tom"
    pids, _ = tagger.tag("Fulham midfielder King returns")
    assert tagger.players[pids[0]]["first_name"] == "Josh"
    pids, _ = tagger.tag("Josh King and Tom King both start")
    assert sorted(tagger.players[p]["first_name"] for p in pids) == ["Josh", "Tom"]


def test_common_word_false_positives(tagger):
    assert names(tagger, "The king of the hill pointed at the rogers family") == []
    assert names(tagger, "long live the king") == []
    # capitalised but no team context is still skipped for a common word
    assert names(tagger, "The King opens parliament") == []


def test_team_context_unlocks_general_news(tagger):
    assert names(tagger, "England fans hope Rogers scores") == []
    assert names(tagger, "Chelsea beat Hull City as Rogers scored") == ["Rogers"]


def test_fpl_context_relaxes_surnames(tagger):
    assert names(tagger, "Haaland or Wissa to captain?", fpl=True) == ["Haaland", "Wissa"]
    assert names(tagger, "Rogers looks great", fpl=True) == ["Rogers"]


def test_first_name_only_not_tagged_in_general_news(tagger):
    assert names(tagger, "Enzo Maresca on Saturday") == []
    assert names(tagger, "David Beckham heroics") == []


def test_team_detection(tagger, bs):
    _, teams = tagger.tag("Spurs travel to Manchester United; Hull and Ipswich win")
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    assert {short[t] for t in teams} >= {"TOT", "MUN", "HUL", "IPS"}


# ---- tables / run --------------------------------------------------------------------


def test_tag_items_one_row_per_player_and_untagged(tagger):
    items = [
        {"item_id": "a", "published": pd.Timestamp("2026-10-10", tz="UTC"), "source": "s",
         "kind": "rss", "title": "Haaland and Wissa", "summary": "", "url": "u1",
         "fpl_context": True},
        {"item_id": "b", "published": pd.Timestamp("2026-10-09", tz="UTC"), "source": "s",
         "kind": "rss", "title": "Nothing here", "summary": "", "url": "u2"},
    ]
    rows = news.tag_items(items, tagger)
    assert [(r["item_id"], r["player_name"]) for r in rows] == [
        ("a", "Haaland"), ("a", "Wissa"), ("b", None)]
    assert rows[2]["player_id"] is None


def test_run_end_to_end(client, data_root):
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        df = news.run(client, http=http, use_cache=False)
    path = storage.table_path("news", NEXT_GW)
    assert path.exists()
    back = storage.read_table("news", NEXT_GW)
    assert len(back) == len(df) > 0
    assert set(back["kind"]) == {"fpl_flag", "rss", "reddit"}
    assert {"FPL", "Fantasy Football Scout", "r/FantasyPL"} <= set(back["source"])
    assert back["published"].is_monotonic_decreasing or back["published"].isna().any()
    assert not back.duplicated(["item_id", "player_id"]).any()
    # untagged items kept with null player_id
    assert back["player_id"].isna().any() and back["player_id"].notna().any()
    # extras present
    assert {"status", "chance_of_playing_next_round", "team_id", "team_ids"} <= set(back.columns)


def test_run_survives_dead_sources(client, data_root):
    def dead(request):
        if "fantasyfootballscout" in str(request.url):
            return httpx.Response(200, content=(FIX / "fantasyfootballscout.xml").read_bytes())
        if "bbci" in str(request.url):
            raise httpx.ConnectError("boom")
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(dead)) as http:
        df = news.run(client, http=http, use_cache=False)
    assert set(df["source"]) == {"FPL", "Fantasy Football Scout"}


def test_reddit_json_used_when_available(client, data_root):
    def h(request):
        if "hot.json" in str(request.url):
            return httpx.Response(200, content=(FIX / "reddit_fantasypl.json").read_bytes())
        return httpx.Response(404)

    srcs = [s for s in news.SOURCES if s.name == "r/FantasyPL"]
    with httpx.Client(transport=httpx.MockTransport(h)) as http:
        df = news.run(client, http=http, sources=srcs, use_cache=False)
    red = df[df["kind"] == "reddit"]
    assert red["item_id"].nunique() == 5
    tagged = set(red["player_name"].dropna())
    assert {"Haaland", "Wissa", "Ødegaard", "Groß", "Guéhi"} <= tagged
    king_item = red[red["title"].str.startswith("King of the hill")]
    assert king_item["player_id"].isna().all()
