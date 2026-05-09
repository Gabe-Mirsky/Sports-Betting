"""Coverage summaries for NBA games, Kalshi markets, matches, and prices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        csv_path = table_path.with_suffix(".csv")
        if csv_path.exists():
            table_path = csv_path
        else:
            return pd.DataFrame()
    if table_path.suffix.lower() == ".csv":
        return pd.read_csv(table_path, dtype={"game_id": str})
    try:
        return pd.read_parquet(table_path)
    except (ImportError, ValueError, RuntimeError):
        csv_path = table_path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, dtype={"game_id": str})
        raise


def _parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def load_default_game_universe(project_root: str | Path | None = None) -> pd.DataFrame:
    root = Path(project_root) if project_root else PROJECT_ROOT
    frames: list[pd.DataFrame] = []
    for path in [
        root / "data" / "reports" / "all_game_predictions.csv",
        root / "data" / "reports" / "upcoming_predictions.csv",
    ]:
        frame = _read_table(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    if "game_id" in games.columns:
        games = games.drop_duplicates(subset=["game_id"], keep="last")
    if "game_date" in games.columns:
        games["game_date"] = _parse_dates(games["game_date"]).dt.normalize()
    return games.reset_index(drop=True)


def build_kalshi_coverage_report(project_root: str | Path | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(project_root) if project_root else PROJECT_ROOT
    games = load_default_game_universe(root)
    markets = _read_table(root / "data" / "processed" / "kalshi_possible_nba_markets.parquet")
    matches = _read_table(root / "data" / "processed" / "kalshi_game_market_matches.csv")
    prices = _read_table(root / "data" / "processed" / "kalshi_pregame_prices.csv")
    starts = _read_table(root / "data" / "interim" / "nba_game_start_times.csv")

    for frame in [markets, matches, prices, starts]:
        if not frame.empty and "game_date" in frame.columns:
            frame["game_date"] = _parse_dates(frame["game_date"]).dt.normalize()

    usable_prices = prices[prices["price_quality"] != "missing"].copy() if "price_quality" in prices.columns else prices
    auto_matches = matches[matches["match_status"] == "auto_matched"].copy() if "match_status" in matches.columns else matches
    review_matches = matches[matches["match_status"] == "needs_review"].copy() if "match_status" in matches.columns else pd.DataFrame()

    market_key_columns = {"game_date", "home_team_abbr", "away_team_abbr"}
    games_with_markets = (
        int(markets[["game_date", "home_team_abbr", "away_team_abbr"]].drop_duplicates().shape[0])
        if market_key_columns.issubset(markets.columns)
        else int(markets["game_id"].nunique())
        if "game_id" in markets.columns
        else 0
    )

    summary = {
        "games_in_prediction_universe": int(games["game_id"].nunique()) if "game_id" in games.columns else int(len(games)),
        "kalshi_market_rows": int(len(markets)),
        "games_with_kalshi_markets": games_with_markets,
        "auto_matched_games": int(auto_matches["game_id"].nunique()) if "game_id" in auto_matches.columns else 0,
        "needs_review_games": int(review_matches["game_id"].nunique()) if "game_id" in review_matches.columns else 0,
        "pregame_price_rows": int(len(prices)),
        "games_with_usable_pregame_price": int(usable_prices["game_id"].nunique()) if "game_id" in usable_prices.columns else 0,
        "games_with_start_times": int(starts.drop_duplicates(["game_date", "home_team_abbr", "away_team_abbr"]).shape[0])
        if {"game_date", "home_team_abbr", "away_team_abbr"}.issubset(starts.columns)
        else 0,
        "market_date_min": str(markets["game_date"].min().date()) if "game_date" in markets.columns and markets["game_date"].notna().any() else None,
        "market_date_max": str(markets["game_date"].max().date()) if "game_date" in markets.columns and markets["game_date"].notna().any() else None,
    }
    if not prices.empty and "price_quality" in prices.columns:
        summary["price_quality_counts"] = prices["price_quality"].value_counts(dropna=False).to_dict()
    if not prices.empty and "period_interval" in prices.columns:
        summary["period_interval_counts"] = prices["period_interval"].value_counts(dropna=False).to_dict()

    if games.empty or "game_date" not in games.columns:
        monthly = pd.DataFrame()
    else:
        monthly = games[["game_id", "game_date"]].drop_duplicates().copy()
        monthly["month"] = monthly["game_date"].dt.to_period("M").astype(str)
        monthly = monthly.groupby("month", as_index=False).agg(games=("game_id", "nunique"))

        if {"game_date", "home_team_abbr", "away_team_abbr"}.issubset(markets.columns):
            market_monthly = markets[["game_date", "home_team_abbr", "away_team_abbr"]].drop_duplicates()
            market_monthly["game_id"] = (
                market_monthly["game_date"].astype(str)
                + "|"
                + market_monthly["away_team_abbr"].astype(str)
                + "|"
                + market_monthly["home_team_abbr"].astype(str)
            )
        elif "game_id" in markets.columns and markets["game_id"].notna().any():
            market_monthly = markets[["game_id"]].drop_duplicates().merge(
                games[["game_id", "game_date"]],
                on="game_id",
                how="left",
            )
        else:
            market_monthly = pd.DataFrame()
        if not market_monthly.empty:
            market_monthly["month"] = market_monthly["game_date"].dt.to_period("M").astype(str)
            market_monthly = market_monthly.groupby("month", as_index=False).agg(games_with_markets=("game_id", "nunique"))
            monthly = monthly.merge(market_monthly, on="month", how="left")
        if "game_id" in usable_prices.columns:
            price_monthly = usable_prices[["game_id"]].drop_duplicates().merge(
                games[["game_id", "game_date"]],
                on="game_id",
                how="left",
            )
            price_monthly["month"] = price_monthly["game_date"].dt.to_period("M").astype(str)
            price_monthly = price_monthly.groupby("month", as_index=False).agg(games_with_prices=("game_id", "nunique"))
            monthly = monthly.merge(price_monthly, on="month", how="left")
        for column in ["games_with_markets", "games_with_prices"]:
            if column not in monthly.columns:
                monthly[column] = 0
            monthly[column] = monthly[column].fillna(0).astype(int)
        monthly["market_coverage_pct"] = monthly["games_with_markets"] / monthly["games"]
        monthly["price_coverage_pct"] = monthly["games_with_prices"] / monthly["games"]

    return summary, monthly


def _classify_unmatched_market(row: pd.Series) -> str:
    game_date = row.get("game_date")
    title = str(row.get("market_title", "")).lower()
    result = str(row.get("result", "")).lower()
    status = str(row.get("status", "")).lower()
    away_team = str(row.get("away_team_abbr", "")).upper()
    home_team = str(row.get("home_team_abbr", "")).upper()

    if status in {"active", "initialized"}:
        return "future_or_active_market"
    if result == "scalar":
        return "unusual_scalar_settlement"
    if away_team not in {
        "ATL",
        "BKN",
        "BOS",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    } or home_team not in {
        "ATL",
        "BKN",
        "BOS",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    }:
        return "exhibition_non_nba_opponent"
    if pd.notna(game_date):
        timestamp = pd.Timestamp(game_date)
        if timestamp.month == 10 and timestamp.day < 20:
            return "preseason_market"
        if timestamp.month == 4 and 13 <= timestamp.day <= 18 and "game " not in title:
            return "play_in_or_special_postseason_market"
    return "no_model_game_row"


def build_kalshi_gap_report(project_root: str | Path | None = None) -> pd.DataFrame:
    """List Kalshi game markets that are not represented by an auto-matched model game."""

    root = Path(project_root) if project_root else PROJECT_ROOT
    markets = _read_table(root / "data" / "processed" / "kalshi_possible_nba_markets.parquet")
    matches = _read_table(root / "data" / "processed" / "kalshi_game_market_matches.csv")
    if markets.empty:
        return pd.DataFrame()
    if "game_date" in markets.columns:
        markets["game_date"] = _parse_dates(markets["game_date"]).dt.normalize()
    if not matches.empty and "game_date" in matches.columns:
        matches["game_date"] = _parse_dates(matches["game_date"]).dt.normalize()

    required = {"game_date", "home_team_abbr", "away_team_abbr"}
    if not required.issubset(markets.columns):
        return pd.DataFrame()
    auto_matches = matches[matches["match_status"] == "auto_matched"].copy() if "match_status" in matches.columns else matches
    auto_keys = (
        auto_matches[["game_date", "home_team_abbr", "away_team_abbr"]].drop_duplicates()
        if required.issubset(auto_matches.columns)
        else pd.DataFrame(columns=["game_date", "home_team_abbr", "away_team_abbr"])
    )
    unmatched = markets.merge(
        auto_keys,
        on=["game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns=["_merge"]).copy()
    if unmatched.empty:
        return pd.DataFrame()
    unmatched["gap_reason"] = unmatched.apply(_classify_unmatched_market, axis=1)
    columns = [
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
        "market_ticker",
        "market_title",
        "yes_team_abbr",
        "status",
        "result",
        "gap_reason",
    ]
    available = [column for column in columns if column in unmatched.columns]
    output = unmatched[available].drop_duplicates().sort_values(
        [column for column in ["game_date", "market_ticker"] if column in available]
    )
    return output.reset_index(drop=True)


def save_kalshi_coverage_report(
    summary: dict[str, Any],
    monthly: pd.DataFrame,
    summary_path: str | Path | None = None,
    monthly_path: str | Path | None = None,
    gap_report: pd.DataFrame | None = None,
    gap_path: str | Path | None = None,
) -> None:
    summary_output = (
        Path(summary_path)
        if summary_path
        else PROJECT_ROOT / "data" / "reports" / "kalshi_coverage_summary.json"
    )
    monthly_output = (
        Path(monthly_path)
        if monthly_path
        else PROJECT_ROOT / "data" / "reports" / "kalshi_coverage_by_month.csv"
    )
    gap_output = (
        Path(gap_path)
        if gap_path
        else PROJECT_ROOT / "data" / "reports" / "kalshi_unmatched_market_gap_report.csv"
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    monthly.to_csv(monthly_output, index=False)
    if gap_report is not None:
        gap_output.parent.mkdir(parents=True, exist_ok=True)
        gap_report.to_csv(gap_output, index=False)
