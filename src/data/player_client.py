"""NBA player game-log download and cache helpers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .nba_client import season_int_to_nba_season
from .progress import tqdm


logger = logging.getLogger(__name__)


def player_game_log_cache_path(
    cache_dir: str | Path,
    season_start_year: int,
    season_type: str = "Regular Season",
) -> Path:
    cache_root = Path(cache_dir)
    if season_type == "Regular Season":
        filename = f"player_game_log_{season_start_year}.parquet"
    else:
        safe = season_type.lower().replace(" ", "_")
        filename = f"player_game_log_{season_start_year}_{safe}.parquet"
    return cache_root / filename


def _read_table(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError, RuntimeError):
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError, RuntimeError):
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def fetch_league_player_game_log(
    season: str,
    season_type: str = "Regular Season",
    retries: int = 3,
    sleep_seconds: float = 2.0,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch one season of player-level NBA game logs from NBA.com through nba_api."""

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            from nba_api.stats.endpoints import leaguegamelog

            logger.info(
                "Fetching NBA player LeagueGameLog season=%s season_type=%s attempt=%s/%s",
                season,
                season_type,
                attempt,
                retries,
            )
            endpoint = leaguegamelog.LeagueGameLog(
                player_or_team_abbreviation="P",
                season=season,
                season_type_all_star=season_type,
                timeout=timeout,
            )
            frames = endpoint.get_data_frames()
            if not frames:
                raise RuntimeError("nba_api returned no data frames")
            df = frames[0].copy()
            df["nba_season"] = season
            df["season_type"] = season_type
            return df
        except Exception as exc:
            last_error = exc
            logger.warning(
                "NBA player fetch failed for %s on attempt %s/%s: %s",
                season,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Failed to fetch NBA player LeagueGameLog for {season}") from last_error


def download_player_seasons(
    start_season: int,
    end_season: int,
    cache_dir: str | Path = "data/raw/nba/player",
    season_type: str = "Regular Season",
    force: bool = False,
    strict: bool = False,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    timeout: int = 60,
) -> list[pd.DataFrame]:
    """Download or load cached NBA player game logs for a season range."""

    if start_season > end_season:
        raise ValueError("start_season must be less than or equal to end_season")

    cache_root = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for season_start_year in tqdm(
        range(start_season, end_season + 1),
        desc="NBA player seasons",
        unit="season",
    ):
        nba_season = season_int_to_nba_season(season_start_year)
        cache_path = player_game_log_cache_path(cache_root, season_start_year, season_type=season_type)
        try:
            if not force and (cache_path.exists() or cache_path.with_suffix(".csv").exists()):
                df = _read_table(cache_path)
            else:
                df = fetch_league_player_game_log(
                    nba_season,
                    season_type=season_type,
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                    timeout=timeout,
                )
                _write_table(df, cache_path)
            df["season_start_year"] = season_start_year
            df["nba_season"] = nba_season
            df["season_type"] = season_type
            frames.append(df)
            time.sleep(sleep_seconds)
        except Exception as exc:
            message = f"Could not process NBA player season {nba_season}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            logger.error(message)
    return frames


def load_raw_player_logs(cache_dir: str | Path = "data/raw/nba/player") -> pd.DataFrame:
    """Load cached player LeagueGameLog files from parquet or CSV."""

    cache_root = Path(cache_dir)
    files = sorted({*cache_root.glob("player_game_log_*.parquet"), *cache_root.glob("player_game_log_*.csv")})
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        frame = _read_table(path)
        frame["source_file"] = path.name
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if "GAME_DATE" in output.columns:
        output["GAME_DATE"] = pd.to_datetime(output["GAME_DATE"], errors="coerce")
    return output
