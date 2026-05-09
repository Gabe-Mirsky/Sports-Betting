"""NBA data download helpers built around nba_api."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .cache import league_game_log_cache_path, read_dataframe, write_dataframe


logger = logging.getLogger(__name__)


def season_int_to_nba_season(year: int) -> str:
    """Convert a start year like 2023 to an NBA season label like 2023-24."""

    if year < 1946:
        raise ValueError(f"NBA season start year looks invalid: {year}")

    next_year_suffix = (year + 1) % 100
    return f"{year}-{next_year_suffix:02d}"


def fetch_league_game_log(
    season: str,
    season_type: str = "Regular Season",
    retries: int = 3,
    sleep_seconds: float = 2.0,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch one season of team-level game logs from NBA.com through nba_api."""

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            from nba_api.stats.endpoints import leaguegamelog

            logger.info(
                "Fetching NBA LeagueGameLog season=%s season_type=%s attempt=%s/%s",
                season,
                season_type,
                attempt,
                retries,
            )
            endpoint = leaguegamelog.LeagueGameLog(
                player_or_team_abbreviation="T",
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
            logger.info("Fetched %s team-game rows for %s", len(df), season)
            return df
        except Exception as exc:  # NBA.com can fail in several transport-specific ways.
            last_error = exc
            logger.warning(
                "NBA fetch failed for %s on attempt %s/%s: %s",
                season,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Failed to fetch NBA LeagueGameLog for {season}") from last_error


def _add_download_metadata(
    df: pd.DataFrame,
    season_start_year: int,
    nba_season: str,
    season_type: str,
) -> pd.DataFrame:
    output = df.copy()
    output["season_start_year"] = season_start_year
    output["nba_season"] = nba_season
    output["season_type"] = season_type
    return output


def download_seasons(
    start_season: int,
    end_season: int,
    cache_dir: str | Path = "data/raw",
    season_type: str = "Regular Season",
    force: bool = False,
    strict: bool = False,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    timeout: int = 60,
) -> list[pd.DataFrame]:
    """Download or load cached NBA team game logs for a range of seasons."""

    if start_season > end_season:
        raise ValueError("start_season must be less than or equal to end_season")

    cache_root = Path(cache_dir)
    downloaded: list[pd.DataFrame] = []

    for season_start_year in tqdm(
        range(start_season, end_season + 1),
        desc="NBA seasons",
        unit="season",
    ):
        nba_season = season_int_to_nba_season(season_start_year)
        cache_path = league_game_log_cache_path(
            cache_root,
            season_start_year,
            season_type=season_type,
        )

        try:
            if cache_path.exists() and not force:
                logger.info("Using cached NBA data for %s at %s", nba_season, cache_path)
                df = read_dataframe(cache_path)
                df = _add_download_metadata(
                    df,
                    season_start_year=season_start_year,
                    nba_season=nba_season,
                    season_type=season_type,
                )
            else:
                df = fetch_league_game_log(
                    nba_season,
                    season_type=season_type,
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                    timeout=timeout,
                )
                df = _add_download_metadata(
                    df,
                    season_start_year=season_start_year,
                    nba_season=nba_season,
                    season_type=season_type,
                )
                write_dataframe(df, cache_path)

            downloaded.append(df)
            time.sleep(sleep_seconds)
        except Exception as exc:
            message = f"Could not process NBA season {nba_season}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            logger.error(message)

    return downloaded


def download_nba_seasons(
    start_season: int,
    end_season: int,
    force: bool = False,
    season_type: str = "Regular Season",
    cache_dir: str | Path = "data/raw/nba",
) -> pd.DataFrame:
    """Download NBA seasons and return one concatenated team-game dataframe."""

    frames = download_seasons(
        start_season=start_season,
        end_season=end_season,
        cache_dir=cache_dir,
        season_type=season_type,
        force=force,
    )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
