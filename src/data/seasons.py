"""NBA season date helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd


TRAIN_START_SEASON = 2018
TRAIN_END_SEASON = 2023
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
MIN_TRAINING_SPORTSBOOK_MATCH_RATE = 0.80
FREE_ODDS_SPLIT_MODE = "strict_full_seasons"
FREE_ODDS_SPLIT_MODES = {"latest_available", "strict_full_seasons"}


def current_nba_season_start_year(today: date | None = None) -> int:
    """Return the NBA season start year for a calendar date.

    NBA season labels use the first calendar year in the season. For example,
    games played in May 2026 belong to the 2025-26 season, so this returns 2025.
    """

    current = today or date.today()
    if current.month >= 7:
        return current.year
    return current.year - 1


def nba_season_display_label(season_start_year: int) -> str:
    """Return a friendly season label like 2025-26."""

    return f"{season_start_year}-{(season_start_year + 1) % 100:02d}"


def season_start_year_from_dates(values: pd.Series) -> pd.Series:
    """Return NBA season start years inferred from game dates."""

    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)


def assign_dataset_split(
    frame: pd.DataFrame,
    season_column: str = "season",
    date_column: str = "game_date",
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    allow_outside: bool = False,
) -> pd.DataFrame:
    """Add the canonical chronological train/validation/test split."""

    if season_column in frame.columns:
        seasons = pd.to_numeric(frame[season_column], errors="coerce")
        if seasons.isna().any() and date_column in frame.columns:
            seasons = seasons.fillna(season_start_year_from_dates(frame[date_column]))
    elif date_column in frame.columns:
        seasons = season_start_year_from_dates(frame[date_column])
    else:
        raise ValueError(f"Need either {season_column!r} or {date_column!r} to assign dataset_split.")

    output = frame.copy()
    output["dataset_split"] = pd.NA
    output.loc[seasons.between(train_start_season, train_end_season, inclusive="both"), "dataset_split"] = "train"
    output.loc[seasons.eq(validation_season), "dataset_split"] = "validation"
    output.loc[seasons.eq(test_season), "dataset_split"] = "test"

    unassigned = output["dataset_split"].isna()
    if allow_outside:
        output.loc[unassigned, "dataset_split"] = "outside_split"
        return output
    if unassigned.any():
        bad_seasons = sorted(
            {
                int(season)
                for season in seasons.loc[unassigned].dropna().unique()
            }
        )
        raise ValueError(
            "Rows fall outside the configured train/validation/test seasons: "
            f"{bad_seasons}. Expected train {train_start_season}-{train_end_season}, "
            f"validation {validation_season}, test {test_season}."
        )

    return output


def infer_train_start_from_sportsbook_coverage(
    coverage: pd.DataFrame,
    default_train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    minimum_match_rate: float = MIN_TRAINING_SPORTSBOOK_MATCH_RATE,
) -> int:
    """Pick the first training season with usable sportsbook coverage."""

    if coverage.empty or "season" not in coverage.columns:
        return default_train_start_season
    working = coverage.copy()
    working["season"] = pd.to_numeric(working["season"], errors="coerce")
    if "match_rate" in working.columns:
        rate_column = "match_rate"
    elif "sportsbook_match_rate" in working.columns:
        rate_column = "sportsbook_match_rate"
    else:
        return default_train_start_season
    working[rate_column] = pd.to_numeric(working[rate_column], errors="coerce").fillna(0.0)
    candidates = working[
        working["season"].between(default_train_start_season, train_end_season, inclusive="both")
        & working[rate_column].ge(minimum_match_rate)
    ].sort_values("season")
    if not candidates.empty:
        return int(candidates.iloc[0]["season"])

    count_column = None
    for column in ["matched_games", "sportsbook_games"]:
        if column in working.columns:
            count_column = column
            break
    if count_column is None:
        return default_train_start_season
    working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    any_coverage = working[
        working["season"].between(default_train_start_season, train_end_season, inclusive="both")
        & working[count_column].gt(0)
    ].sort_values("season")
    if any_coverage.empty:
        return default_train_start_season
    return int(any_coverage.iloc[0]["season"])


def infer_train_bounds_from_sportsbook_coverage(
    coverage: pd.DataFrame,
    default_train_start_season: int = TRAIN_START_SEASON,
    default_train_end_season: int = TRAIN_END_SEASON,
    minimum_match_rate: float = MIN_TRAINING_SPORTSBOOK_MATCH_RATE,
) -> tuple[int, int]:
    """Return contiguous training bounds that exclude seasons with no sportsbook odds."""

    train_start = infer_train_start_from_sportsbook_coverage(
        coverage,
        default_train_start_season=default_train_start_season,
        train_end_season=default_train_end_season,
        minimum_match_rate=minimum_match_rate,
    )
    if coverage.empty or "season" not in coverage.columns:
        return train_start, default_train_end_season
    working = coverage.copy()
    working["season"] = pd.to_numeric(working["season"], errors="coerce")
    count_column = None
    for column in ["matched_games", "sportsbook_games"]:
        if column in working.columns:
            count_column = column
            break
    if count_column is None:
        return train_start, default_train_end_season
    working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    by_season = {
        int(row["season"]): int(row[count_column])
        for _, row in working.dropna(subset=["season"]).iterrows()
    }
    train_end = train_start - 1
    for season in range(train_start, default_train_end_season + 1):
        if by_season.get(season, 0) <= 0:
            break
        train_end = season
    if train_end < train_start:
        return train_start, default_train_end_season
    return train_start, train_end


def _coverage_count_column(coverage: pd.DataFrame) -> str | None:
    for column in ["matched_games", "sportsbook_games"]:
        if column in coverage.columns:
            return column
    return None


def _coverage_rate_column(coverage: pd.DataFrame) -> str | None:
    for column in ["match_rate", "sportsbook_match_rate"]:
        if column in coverage.columns:
            return column
    return None


def build_free_odds_split_plan(
    coverage: pd.DataFrame,
    mode: str = FREE_ODDS_SPLIT_MODE,
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    test_season: int = TEST_SEASON,
    minimum_match_rate: float = MIN_TRAINING_SPORTSBOOK_MATCH_RATE,
) -> dict[str, object]:
    """Build the market-proxy train/validation/test split from free odds coverage."""

    if mode not in FREE_ODDS_SPLIT_MODES:
        raise ValueError(f"Unsupported free odds split mode: {mode}. Expected one of {sorted(FREE_ODDS_SPLIT_MODES)}")

    working = coverage.copy() if not coverage.empty else pd.DataFrame()
    if not working.empty and "season" in working.columns:
        working["season"] = pd.to_numeric(working["season"], errors="coerce")
    count_column = _coverage_count_column(working)
    rate_column = _coverage_rate_column(working)
    if count_column:
        working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0)
    if rate_column:
        working[rate_column] = pd.to_numeric(working[rate_column], errors="coerce").fillna(0.0)

    eligible = working[
        working["season"].between(train_start_season, train_end_season, inclusive="both")
    ].copy() if not working.empty and "season" in working.columns else pd.DataFrame()

    if count_column:
        covered = eligible[eligible[count_column].gt(0)].sort_values("season")
    else:
        covered = pd.DataFrame()
    if rate_column:
        full = eligible[eligible[rate_column].ge(minimum_match_rate)].sort_values("season")
    else:
        full = pd.DataFrame()

    if mode == "latest_available":
        validation_season = int(covered.iloc[-1]["season"]) if not covered.empty else None
        train_seasons = [
            int(season)
            for season in covered["season"].dropna().astype(int).tolist()
            if validation_season is not None and int(season) < validation_season
        ]
    else:
        validation_season = int(full.iloc[-1]["season"]) if not full.empty else None
        train_seasons = [
            int(season)
            for season in full["season"].dropna().astype(int).tolist()
            if validation_season is not None and int(season) < validation_season
        ]

    season_splits: dict[int, str] = {}
    for season in working["season"].dropna().astype(int).unique().tolist() if "season" in working.columns else []:
        season_splits[int(season)] = "outside_split"
    for season in range(train_start_season, test_season + 1):
        season_splits.setdefault(season, "outside_split")
    train_label = "strict_train" if mode == "strict_full_seasons" else "train"
    validation_label = "strict_validation" if mode == "strict_full_seasons" else "validation"
    for season in train_seasons:
        season_splits[season] = train_label
    if validation_season is not None:
        season_splits[validation_season] = validation_label
    season_splits[test_season] = "test/live"

    excluded_due_to_missing_odds = []
    partial_validation_warning = ""
    validation_match_rate = None
    if validation_season is not None and not eligible.empty:
        validation_row = eligible[eligible["season"].eq(validation_season)]
        if not validation_row.empty and rate_column:
            validation_match_rate = float(validation_row.iloc[0][rate_column])
            if validation_match_rate < minimum_match_rate:
                partial_validation_warning = (
                    f"{nba_season_display_label(validation_season)} is partial free odds coverage, "
                    "so validation uses only matched sportsbook games."
                )
    exclusion_scan = working[
        working["season"].between(train_start_season, test_season - 1, inclusive="both")
    ].copy() if not working.empty and "season" in working.columns else pd.DataFrame()
    for _, row in exclusion_scan.iterrows():
        season = int(row["season"])
        if count_column and int(row[count_column]) <= 0:
            excluded_due_to_missing_odds.append(season)

    return {
        "mode": mode,
        "train_seasons": train_seasons,
        "validation_season": validation_season,
        "test_season": test_season,
        "season_splits": season_splits,
        "validation_match_rate": validation_match_rate,
        "excluded_due_to_missing_odds": excluded_due_to_missing_odds,
        "partial_validation_warning": partial_validation_warning,
        "minimum_match_rate": minimum_match_rate,
    }
