"""FPL API client with on-disk JSON cache, throttling and retry/backoff.

Every endpoint method returns parsed JSON (dict/list). Responses are cached under
``data/cache/<key>.json`` where ``<key>`` is the endpoint path with ``/`` replaced
by ``_`` (see :func:`cache_key`). The same key names the recorded fixtures in
``tests/fixtures/``, so tests can serve them through :func:`fixture_transport`.

The client is thread-safe: one shared throttle caps the request rate across
threads, so callers may use a thread pool for bulk pulls (e.g. top-10k picks).
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from fpl.config import Settings, get_settings

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api/"
RETRY_STATUS = {429, 500, 502, 503, 504}

# TTL sentinels
FOREVER = -1  # immutable data (e.g. picks for a finished GW)
NO_CACHE = 0


def cache_key(path: str, params: dict[str, Any] | None = None) -> str:
    """``entry/1608168/event/6/picks/`` -> ``entry_1608168_event_6_picks``."""
    key = path.strip("/").replace("/", "_")
    if params:
        key += "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    return key


class FPLAPIError(RuntimeError):
    pass


class _Throttle:
    def __init__(self, rate: float):
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class FPLClient:
    """Thin client for the public FPL API.

    Args:
        settings: project settings (defaults to config.toml).
        cache_dir: override cache location; ``None`` uses ``settings.cache_dir``.
        use_cache: disable to skip reading and writing the on-disk cache.
        transport: optional httpx transport (tests pass :func:`fixture_transport`).
        rate: requests per second; defaults to ``settings.requests_per_second``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        transport: httpx.BaseTransport | None = None,
        rate: float | None = None,
    ):
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else self.settings.cache_dir
        self.use_cache = use_cache
        self._throttle = _Throttle(rate if rate is not None else self.settings.requests_per_second)
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"User-Agent": self.settings.user_agent, "Accept": "application/json"},
            timeout=httpx.Timeout(20.0, connect=10.0),
            transport=transport,
            follow_redirects=True,
        )
        self.requests_made = 0

    # ---- plumbing -------------------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FPLClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str, ttl: int) -> Any | None:
        if not self.use_cache or ttl == NO_CACHE:
            return None
        p = self._cache_path(key)
        if not p.exists():
            return None
        if ttl != FOREVER and time.time() - p.stat().st_mtime > ttl:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache(self, key: str, data: Any) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def get(self, path: str, params: dict[str, Any] | None = None, ttl: int | None = None) -> Any:
        """GET ``path`` (relative to the API root) with caching and retries."""
        if not path.endswith("/"):
            path += "/"
        ttl = self.settings.cache_ttl_live if ttl is None else ttl
        key = cache_key(path, params)
        cached = self._read_cache(key, ttl)
        if cached is not None:
            return cached

        url = path + ("?" + urlencode(params) if params else "")
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            self._throttle.wait()
            try:
                resp = self._http.get(url)
                self.requests_made += 1
            except httpx.TransportError as exc:
                last_exc = exc
                self._backoff(attempt, None)
                continue
            if resp.status_code == 200:
                data = resp.json()
                if self.use_cache and ttl != NO_CACHE:
                    self._write_cache(key, data)
                return data
            if resp.status_code == 404:
                raise FPLAPIError(f"404 Not Found: {url}")
            if resp.status_code in RETRY_STATUS:
                last_exc = FPLAPIError(f"{resp.status_code} for {url}")
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            raise FPLAPIError(f"{resp.status_code} for {url}: {resp.text[:200]}")
        raise FPLAPIError(f"giving up on {url} after retries") from last_exc

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> None:
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = min(60.0, 1.5 * 2**attempt) + random.uniform(0, 0.5)
        log.warning("FPL API retry %d in %.1fs", attempt + 1, delay)
        time.sleep(delay)

    # ---- endpoints ------------------------------------------------------------------

    def bootstrap_static(self) -> dict:
        """Players (``elements``), ``teams``, ``events``, ``element_types``, ``chips``."""
        return self.get("bootstrap-static/")

    def fixtures(self, event: int | None = None) -> list[dict]:
        """All season fixtures, or only those in ``event``."""
        params = {"event": event} if event is not None else None
        return self.get("fixtures/", params=params)

    def element_summary(self, element_id: int) -> dict:
        """Per-player ``fixtures`` (upcoming), ``history`` (this season), ``history_past``."""
        return self.get(f"element-summary/{element_id}/")

    def entry(self, entry_id: int) -> dict:
        """Manager overview: name, rank, bank/value (``last_deadline_*``), ``leagues``."""
        return self.get(f"entry/{entry_id}/")

    def entry_history(self, entry_id: int) -> dict:
        """``current`` (per GW points/rank/bank/value/transfers), ``past``, ``chips``."""
        return self.get(f"entry/{entry_id}/history/")

    def entry_picks(self, entry_id: int, event: int, final: bool = False) -> dict:
        """``picks`` (15, with multiplier/captain), ``entry_history``, ``active_chip``.

        Picks are only public after the GW deadline. Pass ``final=True`` for a GW whose
        deadline has passed to cache the response forever.
        """
        return self.get(f"entry/{entry_id}/event/{event}/picks/", ttl=FOREVER if final else None)

    def entry_transfers(self, entry_id: int) -> list[dict]:
        """All transfers this season (element_in/out, costs, event, time)."""
        return self.get(f"entry/{entry_id}/transfers/")

    def league_standings(
        self, league_id: int, page: int = 1, phase: int = 1, ttl: int | None = None
    ) -> dict:
        """Classic league standings page (50 entries). ``standings.has_next`` for paging."""
        return self.get(
            f"leagues-classic/{league_id}/standings/",
            params={"page_standings": page, "phase": phase},
            ttl=ttl,
        )

    def event_live(self, event: int, final: bool = False) -> dict:
        """Live per-player stats/points for ``event``."""
        return self.get(f"event/{event}/live/", ttl=FOREVER if final else None)


# ---- offline helpers (tests) -----------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def fixture_transport(fixtures_dir: Path = FIXTURES_DIR) -> httpx.MockTransport:
    """httpx transport serving ``tests/fixtures/<cache_key>.json``; 404 if missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/")
        params = dict(request.url.params)
        p = fixtures_dir / f"{cache_key(path, params or None)}.json"
        if not p.exists():
            return httpx.Response(404, json={"detail": "fixture missing", "key": p.name})
        return httpx.Response(200, content=p.read_bytes(), headers={"Content-Type": "application/json"})

    return httpx.MockTransport(handler)


def offline_client(fixtures_dir: Path = FIXTURES_DIR, **kwargs) -> FPLClient:
    """Client that never touches the network or the real cache."""
    return FPLClient(transport=fixture_transport(fixtures_dir), use_cache=False, rate=0, **kwargs)
