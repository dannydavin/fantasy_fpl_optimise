"""News ingest: FPL injury flags + RSS feeds + r/FantasyPL, tagged by player.

``run(client, gw=None, root=None, sources=None, http=None)`` writes the ``news`` table
(``raw/gwNN``, NN = next GW). Pure helpers (``fpl_flag_items``, ``parse_feed``,
``parse_reddit_json``, ``PlayerTagger``, ``build_table``) are unit-testable on in-memory data.

Sources (checked live 2026-10-10):
- Fantasy Football Scout: https://www.fantasyfootballscout.co.uk/feed/  (works)
- Fantasy Football Hub: no public feed (all /feed, /rss paths 404) -> not included.
- Premier League site: no RSS (news.rss / rss 404) -> replaced by BBC Sport football,
  Sky Sports football and Guardian football feeds.
- r/FantasyPL: www.reddit.com JSON returns 403 without OAuth, so the JSON endpoint is tried
  first and the ``.rss`` (Atom) endpoint is the fallback (works, rate limited).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import feedparser
import httpx
import pandas as pd

from fpl import storage
from fpl.api import _Throttle
from fpl.config import get_settings
from fpl.gameweeks import next_gw

log = logging.getLogger(__name__)

USER_AGENT = "fpl-toolkit/0.1 (personal FPL news reader; contact dan.m.vincent@hotmail.com)"
CACHE_TTL = 15 * 60
HTTP_RATE = 1.0  # requests/second to non-FPL sites


@dataclass(frozen=True)
class Source:
    name: str
    urls: tuple[str, ...]  # tried in order; first that works wins
    kind: str = "rss"  # 'rss' | 'reddit'
    fpl_context: bool = False  # fantasy-specific source: surnames are almost surely players


# Edit this list to add/remove feeds.
SOURCES: list[Source] = [
    Source("Fantasy Football Scout", ("https://www.fantasyfootballscout.co.uk/feed/",),
           fpl_context=True),
    Source("BBC Sport Football", ("https://feeds.bbci.co.uk/sport/football/rss.xml",)),
    Source("Sky Sports Football", ("https://www.skysports.com/rss/12040",)),
    Source("Guardian Football", ("https://www.theguardian.com/football/rss",)),
    Source(
        "r/FantasyPL",
        (
            "https://www.reddit.com/r/FantasyPL/hot.json?limit=50",
            "https://www.reddit.com/r/FantasyPL/.rss",
            "https://www.reddit.com/r/FantasyPL/new/.rss",
        ),
        kind="reddit",
        fpl_context=True,
    ),
]


# ---------------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------------

_TRANSLIT = str.maketrans(
    {"ß": "ss", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "đ": "d", "Đ": "D", "ł": "l",
     "Ł": "L", "œ": "oe", "Œ": "OE", "ı": "i", "’": "'", "‘": "'"}
)


def fold(text: str) -> str:
    """Accent-fold (keeps case): Ødegaard->Odegaard, Guéhi->Guehi, Groß->Gross."""
    text = (text or "").translate(_TRANSLIT)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


class _Stripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("br", "p", "div", "li"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    s = _Stripper()
    try:
        s.feed(text)
        s.close()
        out = "".join(s.parts)
    except Exception:  # pragma: no cover - malformed markup
        out = re.sub(r"<[^>]+>", " ", text)
    out = html.unescape(out).replace("\xa0", " ")
    out = re.sub(r"\[(?:link|comments)\]|submitted by\s+/u/\S+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def item_id(url: str | None, source: str, title: str) -> str:
    basis = url if url else f"{source}|{title}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def _to_ts(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------------------
# player tagging
# ---------------------------------------------------------------------------------------

# Single-token aliases that are ordinary English words/places: need a capitalised match
# AND the player's team mentioned in the text (full names always work).
COMMON_WORDS = {
    "king", "young", "white", "ward", "gray", "grey", "cash", "mount", "will", "strand",
    "bull", "long", "lamb", "hill", "brown", "wood", "stone", "gordon", "miller", "rice",
    "mason", "sterling", "major", "best", "sharp", "kelly", "brooks", "march", "ball",
    "bell", "rose", "cook", "moore", "palmer", "sanchez", "mac", "mate", "lewis", "evans",
    "smith", "johnson", "jones", "bruce", "burn", "dawson", "robinson", "doyle", "wilson",
    "ramsey", "walker", "hall", "keane", "pickford", "welbeck",
}
# Surnames that are plain names in fantasy text but ambiguous in general football news
# (England squad talk etc.): strict outside fantasy-specific sources.
GENERAL_STRICT = {"rogers", "palmer", "walker", "henderson"}
# Extra nicknames per team name (folded, lowercase). Team name/short_name are added too.
TEAM_NICKNAMES = {
    "spurs": ["tottenham", "spurs", "tottenham hotspur"],
    "man city": ["manchester city", "man city"],
    "man utd": ["manchester united", "man utd", "man united"],
    "wolves": ["wolverhampton", "wolves"],
    "nott'm forest": ["nottingham forest", "nott'm forest"],
    "newcastle": ["newcastle", "newcastle united"],
    "west ham": ["west ham", "west ham united"],
    "brighton": ["brighton", "brighton and hove albion"],
    "leeds": ["leeds", "leeds united"],
    "aston villa": ["aston villa", "villa"],
    "crystal palace": ["crystal palace", "palace"],
}


def _word_re(alias: str, cs_first: bool = False) -> str:
    """Regex fragment for alias with word boundaries; first letter case-sensitive upper."""
    body = re.escape(alias)
    body = body.replace(r"\ ", r"\s+")
    if cs_first and alias[:1].isalpha():
        body = f"(?-i:{re.escape(alias[0].upper())})(?i:{re.escape(alias[1:])})".replace(
            r"\ ", r"\s+"
        )
    return body


@dataclass
class _Alias:
    text: str  # folded alias
    players: list[int] = field(default_factory=list)
    single: bool = True


class PlayerTagger:
    """Accent-insensitive, word-boundary player/team matcher built from bootstrap-static."""

    def __init__(self, bootstrap: dict):
        self.players: dict[int, dict] = {}
        self.teams: dict[int, dict] = {}
        for t in bootstrap["teams"]:
            self.teams[t["id"]] = t
        aliases: dict[str, _Alias] = {}

        def add(alias: str, pid: int):
            a = fold(alias).strip()
            key = a.lower()
            if len(key) < 3:
                return
            ent = aliases.setdefault(key, _Alias(a, single=" " not in key))
            if pid not in ent.players:
                ent.players.append(pid)

        for e in bootstrap["elements"]:
            pid = e["id"]
            self.players[pid] = e
            first, second = e.get("first_name") or "", e.get("second_name") or ""
            for a in (e.get("web_name"), e.get("known_name"), second, f"{first} {second}"):
                if a:
                    add(a, pid)
            if e.get("web_name") and first:
                add(f"{first} {e['web_name']}", pid)
        self.aliases = aliases

        # team patterns
        self.team_aliases: dict[str, set[int]] = {}
        for tid, t in self.teams.items():
            names = {fold(t["name"]).lower()}
            names.update(TEAM_NICKNAMES.get(fold(t["name"]).lower(), []))
            for suffix in (" city", " town"):  # Hull City -> Hull, Ipswich Town -> Ipswich
                base = fold(t["name"]).lower()
                if base.endswith(suffix) and not base.startswith("man"):
                    names.add(base[: -len(suffix)])
            for n in names:
                self.team_aliases.setdefault(n, set()).add(tid)
        self._team_re = re.compile(
            r"(?<!\w)(" + "|".join(
                _word_re(n) for n in sorted(self.team_aliases, key=len, reverse=True)
            ) + r")(?!\w)",
            re.I,
        )
        self._short = {t["short_name"]: tid for tid, t in self.teams.items()}
        self._short_re = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, self._short)) + r")(?!\w)")

        # strict single-token aliases need a capitalised match plus team context.
        firsts = {fold(e.get("first_name") or "").lower() for e in bootstrap["elements"]}
        self._cs: dict[str, bool] = {}  # strict in general (non-fantasy) text
        self._cs_fpl: dict[str, bool] = {}  # strict in fantasy-specific text
        for key, ent in aliases.items():
            common = ent.single and (key in COMMON_WORDS or len(key) <= 3)
            self._cs_fpl[key] = common
            self._cs[key] = common or (ent.single and (key in firsts or key in GENERAL_STRICT))
        ordered = sorted(aliases, key=len, reverse=True)
        self._re = re.compile(
            r"(?<!\w)(?:" + "|".join(_word_re(aliases[k].text) for k in ordered) + r")(?!\w)",
            re.I,
        )

    def teams_mentioned(self, text: str) -> set[int]:
        text = fold(text)
        found: set[int] = set()
        for m in self._team_re.finditer(text):
            k = re.sub(r"\s+", " ", m.group(1).lower())
            found |= self.team_aliases.get(k, set())
        for m in self._short_re.finditer(text):
            found.add(self._short[m.group(1)])
        return found

    def tag(self, text: str, fpl_context: bool = False) -> tuple[list[int], set[int]]:
        """Return (player_ids in order of first mention, team_ids mentioned)."""
        text = fold(text or "")
        teams = self.teams_mentioned(text)
        out: list[int] = []
        for m in self._re.finditer(text):
            raw = re.sub(r"\s+", " ", m.group(0))
            key = raw.lower()
            ent = self.aliases.get(key)
            if ent is None:
                continue
            cands = ent.players
            if ent.single and not raw[:1].isupper():
                continue
            if ent.single and not fpl_context:
                # part of a longer capitalised name ("Enzo Maresca", "David Beckham")
                nxt = re.match(r"\s+([A-Z][\w'-]+)", text[m.end():])
                if nxt and fold(nxt.group(1)).lower() not in self.aliases:
                    continue
            if (self._cs_fpl if fpl_context else self._cs)[key]:
                # common word / first name / short token: team context required
                cands = [p for p in cands if self.players[p]["team"] in teams]
            elif len(cands) > 1:
                cands = [p for p in cands if self.players[p]["team"] in teams]
            if len(cands) != 1:
                continue
            if cands[0] not in out:
                out.append(cands[0])
        return out, teams


# ---------------------------------------------------------------------------------------
# item builders
# ---------------------------------------------------------------------------------------


def fpl_flag_items(bootstrap: dict) -> list[dict]:
    """One item per player with non-empty FPL ``news``, already tagged with that player."""
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    items = []
    for e in bootstrap["elements"]:
        news = (e.get("news") or "").strip()
        if not news:
            continue
        short = teams.get(e["team"], "")
        chance = e.get("chance_of_playing_next_round")
        chance_txt = "n/a" if chance is None else f"{int(chance)}%"
        added = e.get("news_added")
        items.append({
            "item_id": hashlib.sha1(f"fpl|{e['id']}|{news}|{added}".encode()).hexdigest(),
            "published": _to_ts(added),
            "source": "FPL",
            "kind": "fpl_flag",
            "title": f"{e['web_name']} ({short}): {news}",
            "summary": f"{news} Status: {e.get('status')}. "
                       f"Chance of playing next round: {chance_txt}.",
            "url": None,
            "player_id": e["id"],
            "player_name": e["web_name"],
            "status": e.get("status"),
            "chance_of_playing_next_round": chance,
            "team_id": e["team"],
            "team_ids": str(e["team"]),
        })
    return items


def parse_feed(content: bytes | str, source: str, kind: str = "rss") -> list[dict]:
    """Parse RSS/Atom bytes into untagged item dicts."""
    feed = feedparser.parse(content)
    items = []
    for ent in feed.entries:
        title = strip_html(ent.get("title"))
        if not title:
            continue
        body = ent.get("summary") or ""
        if not body and ent.get("content"):
            body = ent["content"][0].get("value", "")
        summary = strip_html(body)
        url = ent.get("link") or None
        t = ent.get("published_parsed") or ent.get("updated_parsed")
        pub = pd.Timestamp(datetime(*t[:6], tzinfo=UTC)) if t else pd.NaT
        items.append({
            "item_id": item_id(url, source, title), "published": pub, "source": source,
            "kind": kind, "title": title, "summary": summary[:1000], "url": url,
        })
    return items


def parse_reddit_json(content: bytes | str, source: str = "r/FantasyPL") -> list[dict]:
    data = json.loads(content)
    items = []
    for ch in data.get("data", {}).get("children", []):
        d = ch.get("data", {})
        title = strip_html(d.get("title"))
        if not title:
            continue
        link = d.get("permalink")
        url = f"https://www.reddit.com{link}" if link else d.get("url")
        ts = d.get("created_utc")
        pub = pd.Timestamp(ts, unit="s", tz="UTC") if ts else pd.NaT
        flair = d.get("link_flair_text")
        summary = strip_html(d.get("selftext") or "")
        if flair:
            summary = f"[{flair}] {summary}".strip()
        items.append({
            "item_id": item_id(url, source, title), "published": pub, "source": source,
            "kind": "reddit", "title": title, "summary": summary[:1000], "url": url,
        })
    return items


def parse_payload(content: bytes, source: Source) -> list[dict]:
    head = content.lstrip()[:1]
    if head in (b"{", b"["):
        return parse_reddit_json(content, source.name)
    return parse_feed(content, source.name, source.kind)


def tag_items(items: list[dict], tagger: PlayerTagger) -> list[dict]:
    """Expand items to one row per tagged player. ``it['fpl_context']`` relaxes tagging."""
    # (null player_id when nothing matches)
    rows: list[dict] = []
    for it in items:
        pids, teams = tagger.tag(
            f"{it['title']}. {it.get('summary') or ''}", bool(it.get("fpl_context"))
        )
        team_ids = ",".join(str(t) for t in sorted(teams))
        base = {k: v for k, v in it.items() if k != "fpl_context"}
        base["team_ids"] = team_ids
        if not pids:
            rows.append({**base, "player_id": None, "player_name": None})
            continue
        for pid in pids:
            p = tagger.players[pid]
            rows.append({**base, "player_id": pid, "player_name": p["web_name"],
                         "team_id": p["team"]})
    return rows


def build_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["item_id", "published", "source", "kind", "title", "summary", "url",
            "player_id", "player_name", "status", "chance_of_playing_next_round",
            "team_id", "team_ids"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.drop_duplicates(["item_id", "player_id"], keep="first")
    return df.sort_values("published", ascending=False, na_position="last",
                          kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# HTTP (non-FPL)
# ---------------------------------------------------------------------------------------


class _Fetcher:
    def __init__(self, http: httpx.Client | None, cache_dir: Path, ttl: int, rate: float):
        self.own = http is None
        self.http = http or httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            timeout=httpx.Timeout(15.0, connect=8.0), follow_redirects=True,
        )
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.throttle = _Throttle(rate)

    def get(self, url: str) -> bytes:
        p = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".bin")
        if self.ttl > 0 and p.exists() and time.time() - p.stat().st_mtime < self.ttl:
            return p.read_bytes()
        self.throttle.wait()
        r = self.http.get(url)
        r.raise_for_status()
        if self.ttl > 0:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content)
        return r.content

    def close(self):
        if self.own:
            self.http.close()


def fetch_source(fetcher: _Fetcher, source: Source) -> list[dict]:
    last: Exception | None = None
    for url in source.urls:
        try:
            items = parse_payload(fetcher.get(url), source)
            if items:
                return items
            last = RuntimeError("no items")
        except Exception as exc:  # noqa: BLE001 - per-source resilience
            last = exc
            log.warning("news source %s url %s failed: %s", source.name, url, exc)
    log.warning("news source %s skipped (%s)", source.name, last)
    return []


def run(
    client,
    gw: int | None = None,
    root: Path | None = None,
    sources: Iterable[Source] | None = None,
    http: httpx.Client | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build and write the ``news`` table for the next GW; returns the DataFrame."""
    bootstrap = client.bootstrap_static()
    gw = gw if gw is not None else next_gw(bootstrap)
    if gw is None:
        raise RuntimeError("no next gameweek in bootstrap-static")
    tagger = PlayerTagger(bootstrap)
    rows = fpl_flag_items(bootstrap)  # already tagged with their player

    cache_base = (Path(root) / "cache") if root is not None else get_settings().cache_dir
    fetcher = _Fetcher(http, cache_base / "news", CACHE_TTL if use_cache else 0, HTTP_RATE)
    ext: list[dict] = []
    try:
        for src in SOURCES if sources is None else list(sources):
            items = fetch_source(fetcher, src)
            for it in items:
                it["fpl_context"] = src.fpl_context
            log.info("news %s: %d items", src.name, len(items))
            ext.extend(items)
    finally:
        fetcher.close()
    seen: set[str] = set()
    uniq = []
    for it in ext:
        if it["item_id"] not in seen:
            seen.add(it["item_id"])
            uniq.append(it)
    rows.extend(tag_items(uniq, tagger))
    df = build_table(rows)
    storage.write_table(df, "news", gw, root)
    return df
