"""Small helpers for local dataframe caching."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def league_game_log_cache_path(
    cache_dir: str | Path,
    season_start_year: int,
    season_type: str = "Regular Season",
) -> Path:
    """Return the expected cache path for an NBA LeagueGameLog season."""

    cache_root = Path(cache_dir)
    prefix = "league_game_log" if cache_root.name.lower() == "nba" else "nba_league_game_log"
    if season_type == "Regular Season":
        filename = f"{prefix}_{season_start_year}.parquet"
    else:
        safe_season_type = season_type.lower().replace(" ", "_")
        filename = f"{prefix}_{season_start_year}_{safe_season_type}.parquet"
    return cache_root / filename


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a cached dataframe from parquet."""

    cache_path = Path(path)
    logger.info("Loading cached dataframe from %s", cache_path)
    try:
        return pd.read_parquet(cache_path)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Parquet support is unavailable. Install pyarrow with "
            "`python -m pip install pyarrow`."
        ) from exc


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe to parquet and return the path."""

    cache_path = Path(path)
    ensure_directory(cache_path.parent)
    logger.info("Writing dataframe to %s", cache_path)
    try:
        df.to_parquet(cache_path, index=False)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Parquet support is unavailable. Install pyarrow with "
            "`python -m pip install pyarrow`."
        ) from exc
    return cache_path
