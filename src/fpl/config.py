"""Project configuration loaded from config.toml (overridable via FPL_* env vars)."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.toml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPL_")

    team_id: int = 1608168
    season: str = "2026-27"
    overall_league_id: int = 314
    data_dir: Path = ROOT / "data"
    # API politeness
    requests_per_second: float = 4.0
    max_retries: int = 5
    # Default cache TTLs (seconds); 0 disables caching for that call class.
    cache_ttl_live: int = 15 * 60
    cache_ttl_static: int = 7 * 24 * 3600
    user_agent: str = "Mozilla/5.0 (fantasy_fpl_optimise personal toolkit)"

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Env vars beat config.toml values (which arrive as init kwargs).
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    values: dict = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as f:
            values = tomllib.load(f)
    return Settings(**values)
