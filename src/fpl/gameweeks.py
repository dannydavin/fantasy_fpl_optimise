"""Gameweek helpers over bootstrap-static ``events``."""

from __future__ import annotations

from datetime import UTC, datetime


def _deadline(ev: dict) -> datetime:
    return datetime.fromisoformat(ev["deadline_time"].replace("Z", "+00:00"))


def next_gw(bootstrap: dict) -> int | None:
    """The GW whose deadline is next (the one being planned). None after GW38."""
    ev = next((e for e in bootstrap["events"] if e["is_next"]), None)
    return ev["id"] if ev else None


def current_gw(bootstrap: dict) -> int | None:
    """The GW whose deadline most recently passed (may be in progress)."""
    ev = next((e for e in bootstrap["events"] if e["is_current"]), None)
    return ev["id"] if ev else None


def last_deadline_gw(bootstrap: dict, now: datetime | None = None) -> int | None:
    """Latest GW whose deadline is in the past — picks are public for it."""
    now = now or datetime.now(UTC)
    passed = [e["id"] for e in bootstrap["events"] if _deadline(e) <= now]
    return max(passed) if passed else None


def last_finished_gw(bootstrap: dict) -> int | None:
    done = [e["id"] for e in bootstrap["events"] if e["finished"]]
    return max(done) if done else None


def deadline(bootstrap: dict, gw: int) -> datetime:
    return _deadline(next(e for e in bootstrap["events"] if e["id"] == gw))
