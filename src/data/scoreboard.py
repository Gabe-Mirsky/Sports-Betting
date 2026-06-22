"""Upcoming NBA schedule helpers using the free nba_api package."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd


logger = logging.getLogger(__name__)


def _load_nba_scoreboard_endpoints():
    try:
        from nba_api.stats.endpoints import scoreboardv2, scoreboardv3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nba_api is required to fetch official NBA scoreboard data. "
            "Install project requirements before running live scoreboard refreshes."
        ) from exc
    return scoreboardv2, scoreboardv3


def _to_naive_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is not None:
        return parsed.dt.tz_localize(None)
    return parsed


def infer_season_from_date(game_date: pd.Timestamp | date | datetime) -> int:
    """Infer NBA season start year from a calendar date."""

    timestamp = pd.Timestamp(game_date)
    return int(timestamp.year if timestamp.month >= 9 else timestamp.year - 1)


def infer_season_type_from_date(game_date: pd.Timestamp | date | datetime) -> str:
    """Infer regular season vs playoffs for scoreboard rows."""

    timestamp = pd.Timestamp(game_date)
    if timestamp.month in {5, 6} or (timestamp.month == 4 and timestamp.day >= 15):
        return "Playoffs"
    return "Regular Season"


def parse_scoreboard_frames(
    game_header: pd.DataFrame,
    line_score: pd.DataFrame,
    season_type: str | None = None,
) -> pd.DataFrame:
    """Convert ScoreboardV2 frames into game rows used by the model."""

    if game_header.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "season",
                "season_type",
                "home_team_id",
                "home_team_abbr",
                "away_team_id",
                "away_team_abbr",
                "game_status_id",
                "upcoming_status",
            ]
        )

    header = game_header.copy()
    header["game_id"] = header["GAME_ID"].astype(str)
    header["game_date"] = _to_naive_datetime(header["GAME_DATE_EST"])
    header["home_team_id"] = pd.to_numeric(header["HOME_TEAM_ID"], errors="coerce").astype("Int64")
    header["away_team_id"] = pd.to_numeric(header["VISITOR_TEAM_ID"], errors="coerce").astype("Int64")
    header["game_status_id"] = pd.to_numeric(header["GAME_STATUS_ID"], errors="coerce").astype("Int64")
    header["upcoming_status"] = header["GAME_STATUS_TEXT"].astype(str)

    if "SEASON" in header.columns:
        header["season"] = pd.to_numeric(header["SEASON"], errors="coerce")
    else:
        header["season"] = pd.NA
    header["season"] = header["season"].fillna(header["game_date"].map(infer_season_from_date)).astype(int)
    header["season_type"] = season_type or header["game_date"].map(infer_season_type_from_date)

    if line_score.empty:
        header["home_team_abbr"] = pd.NA
        header["away_team_abbr"] = pd.NA
    else:
        lines = line_score[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]].copy()
        lines["GAME_ID"] = lines["GAME_ID"].astype(str)
        lines["TEAM_ID"] = pd.to_numeric(lines["TEAM_ID"], errors="coerce").astype("Int64")
        home_lines = lines.rename(
            columns={
                "GAME_ID": "game_id",
                "TEAM_ID": "home_team_id",
                "TEAM_ABBREVIATION": "home_team_abbr",
            }
        )
        away_lines = lines.rename(
            columns={
                "GAME_ID": "game_id",
                "TEAM_ID": "away_team_id",
                "TEAM_ABBREVIATION": "away_team_abbr",
            }
        )
        header = header.merge(
            home_lines,
            on=["game_id", "home_team_id"],
            how="left",
        )
        header = header.merge(
            away_lines,
            on=["game_id", "away_team_id"],
            how="left",
        )

    output_columns = [
        "game_id",
        "game_date",
        "season",
        "season_type",
        "home_team_id",
        "home_team_abbr",
        "away_team_id",
        "away_team_abbr",
        "game_status_id",
        "upcoming_status",
    ]
    return header[output_columns].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def parse_scoreboard_v3_frames(
    game_header: pd.DataFrame,
    line_score: pd.DataFrame,
    season_type: str | None = None,
) -> pd.DataFrame:
    """Convert ScoreboardV3 frames into game rows used by the model."""

    if game_header.empty:
        return parse_scoreboard_frames(pd.DataFrame(), pd.DataFrame())

    header = game_header.copy()
    header["game_id"] = header["gameId"].astype(str)
    header["game_date"] = _to_naive_datetime(
        header["gameEt"] if "gameEt" in header.columns else header["gameTimeUTC"],
    )
    header["game_status_id"] = pd.to_numeric(header["gameStatus"], errors="coerce").astype("Int64")
    header["upcoming_status"] = header["gameStatusText"].astype(str)
    header["season"] = header["game_date"].map(infer_season_from_date).astype(int)

    if season_type:
        header["season_type"] = season_type
    elif "poRoundDesc" in header.columns:
        header["season_type"] = header["poRoundDesc"].astype(str).where(
            header["poRoundDesc"].astype(str).str.len() > 0,
            header["game_date"].map(infer_season_type_from_date),
        )
        header["season_type"] = header["season_type"].map(
            lambda value: "Playoffs" if str(value).lower() not in {"", "nan", "none"} else "Regular Season"
        )
    else:
        header["season_type"] = header["game_date"].map(infer_season_type_from_date)

    lines = line_score.copy()
    if lines.empty:
        header["home_team_id"] = pd.NA
        header["away_team_id"] = pd.NA
        header["home_team_abbr"] = pd.NA
        header["away_team_abbr"] = pd.NA
    else:
        lines["gameId"] = lines["gameId"].astype(str)
        lines["teamId"] = pd.to_numeric(lines["teamId"], errors="coerce").astype("Int64")
        lines["teamTricode"] = lines["teamTricode"].astype(str)
        team_lookup = lines.set_index(["gameId", "teamTricode"])[["teamId"]].to_dict()["teamId"]

        away_abbrs: list[str | None] = []
        home_abbrs: list[str | None] = []
        away_ids: list[int | None] = []
        home_ids: list[int | None] = []
        for _, row in header.iterrows():
            game_code = str(row.get("gameCode", ""))
            matchup_code = game_code.split("/", maxsplit=1)[-1]
            if len(matchup_code) >= 6:
                away_abbr = matchup_code[:3]
                home_abbr = matchup_code[-3:]
            else:
                game_lines = lines[lines["gameId"] == row["game_id"]]
                away_abbr = str(game_lines["teamTricode"].iloc[1]) if len(game_lines) > 1 else None
                home_abbr = str(game_lines["teamTricode"].iloc[0]) if len(game_lines) > 0 else None
            away_abbrs.append(away_abbr)
            home_abbrs.append(home_abbr)
            away_ids.append(team_lookup.get((row["game_id"], away_abbr)))
            home_ids.append(team_lookup.get((row["game_id"], home_abbr)))

        header["away_team_abbr"] = away_abbrs
        header["home_team_abbr"] = home_abbrs
        header["away_team_id"] = away_ids
        header["home_team_id"] = home_ids

    output_columns = [
        "game_id",
        "game_date",
        "season",
        "season_type",
        "home_team_id",
        "home_team_abbr",
        "away_team_id",
        "away_team_abbr",
        "game_status_id",
        "upcoming_status",
    ]
    return header[output_columns].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fetch_scoreboard_games(
    game_date: date | datetime | str,
    season_type: str | None = None,
    retries: int = 3,
    sleep_seconds: float = 1.5,
    timeout: int = 30,
    log_warnings: bool = True,
) -> pd.DataFrame:
    """Fetch scheduled NBA games for one date from ScoreboardV2."""

    date_text = pd.Timestamp(game_date).date().isoformat()
    scoreboardv2, scoreboardv3 = _load_nba_scoreboard_endpoints()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            endpoint_v3 = scoreboardv3.ScoreboardV3(game_date=date_text, timeout=timeout)
            games = parse_scoreboard_v3_frames(
                endpoint_v3.game_header.get_data_frame(),
                endpoint_v3.line_score.get_data_frame(),
                season_type=season_type,
            )
            if not games.empty:
                return games

            endpoint_v2 = scoreboardv2.ScoreboardV2(game_date=date_text, timeout=timeout)
            return parse_scoreboard_frames(
                endpoint_v2.game_header.get_data_frame(),
                endpoint_v2.line_score.get_data_frame(),
                season_type=season_type,
            )
        except Exception as exc:  # pragma: no cover - network failures vary
            last_error = exc
            if log_warnings:
                logger.warning(
                    "Scoreboard request failed for %s on attempt %s/%s: %s",
                    date_text,
                    attempt,
                    retries,
                    exc,
                )
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Could not fetch scoreboard for {date_text}") from last_error


def fetch_upcoming_games(
    start_date: date | datetime | str | None = None,
    days: int = 14,
    season_type: str | None = None,
    include_in_progress: bool = True,
) -> pd.DataFrame:
    """Fetch scheduled games from start_date through the next N calendar days."""

    start = pd.Timestamp(start_date).date() if start_date else date.today()
    frames: list[pd.DataFrame] = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        games = fetch_scoreboard_games(current, season_type=season_type)
        if games.empty:
            continue
        frames.append(games)
        time.sleep(0.8)

    if not frames:
        return parse_scoreboard_frames(pd.DataFrame(), pd.DataFrame())

    upcoming = pd.concat(frames, ignore_index=True)
    status = pd.to_numeric(upcoming["game_status_id"], errors="coerce")
    if include_in_progress:
        upcoming = upcoming[status < 3].copy()
    else:
        upcoming = upcoming[status == 1].copy()
    return upcoming.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fill_team_abbreviations_from_history(
    upcoming_games: pd.DataFrame,
    historical_games: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing team abbreviations using IDs from the local historical dataset."""

    if upcoming_games.empty:
        return upcoming_games

    home_map = historical_games[["home_team_id", "home_team_abbr"]].rename(
        columns={"home_team_id": "team_id", "home_team_abbr": "team_abbr"}
    )
    away_map = historical_games[["away_team_id", "away_team_abbr"]].rename(
        columns={"away_team_id": "team_id", "away_team_abbr": "team_abbr"}
    )
    team_map = (
        pd.concat([home_map, away_map], ignore_index=True)
        .dropna()
        .drop_duplicates(subset=["team_id"], keep="last")
        .set_index("team_id")["team_abbr"]
        .to_dict()
    )

    output = upcoming_games.copy()
    for side in ["home", "away"]:
        id_column = f"{side}_team_id"
        abbr_column = f"{side}_team_abbr"
        output[abbr_column] = output[abbr_column].where(
            output[abbr_column].notna(),
            output[id_column].map(team_map),
        )
    return output
