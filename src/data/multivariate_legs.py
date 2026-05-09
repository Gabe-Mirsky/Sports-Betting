"""Extract NBA legs embedded inside Kalshi multivariate markets."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from data.team_aliases import CURRENT_TEAM_ABBRS


NBA_LEG_PREFIX_TO_CATEGORY = {
    "KXNBAGAME": "game_winner",
    "KXNBASPREAD": "spread_handicap",
    "KXNBATOTAL": "total_points_over_under",
    "KXNBAPTS": "player_points_rebounds_assists",
    "KXNBAREB": "player_points_rebounds_assists",
    "KXNBAAST": "player_points_rebounds_assists",
    "KXNBA3PT": "player_points_rebounds_assists",
    "KXNBABLK": "player_points_rebounds_assists",
    "KXNBASTL": "player_points_rebounds_assists",
}

NBA_PROP_STAT_BY_PREFIX = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBA3PT": "three_pointers",
    "KXNBABLK": "blocks",
    "KXNBASTL": "steals",
}

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _normalize_market_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "market_ticker" not in output.columns and "ticker" in output.columns:
        output["market_ticker"] = output["ticker"]
    if "market_title" not in output.columns and "title" in output.columns:
        output["market_title"] = output["title"]
    return output


def _parse_legs(value: object) -> list[dict[str, Any]]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _ticker_prefix(ticker: str) -> str:
    return ticker.split("-", 1)[0].upper()


def _is_nba_leg_ticker(ticker: str) -> bool:
    return _ticker_prefix(ticker) in NBA_LEG_PREFIX_TO_CATEGORY


def _line_value_from_ticker(ticker: str, category: str) -> float | None:
    if category == "game_winner":
        return None
    if category == "total_points_over_under":
        match = re.search(r"-(\d+(?:\.\d+)?)$", ticker)
    elif category == "spread_handicap":
        match = re.search(r"-[A-Z]{2,3}(-?\d+(?:\.\d+)?)$", ticker)
    else:
        match = re.search(r"-(\d+(?:\.\d+)?)$", ticker)
    return float(match.group(1)) if match else None


def _split_matchup_code(matchup: str) -> tuple[str, str]:
    for first in sorted(CURRENT_TEAM_ABBRS, key=len, reverse=True):
        if matchup.startswith(first):
            second = matchup[len(first):]
            if second in CURRENT_TEAM_ABBRS:
                return first, second
    return "", ""


def _parse_event_parts(event_ticker: str) -> dict[str, str]:
    parts = event_ticker.upper().split("-")
    if len(parts) < 2:
        return {"game_date": "", "away_team_abbr": "", "home_team_abbr": ""}
    event_code = parts[1]
    match = re.match(r"(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<teams>[A-Z]+)", event_code)
    if not match:
        return {"game_date": "", "away_team_abbr": "", "home_team_abbr": ""}
    year = 2000 + int(match.group("yy"))
    month = MONTHS.get(match.group("mon"), 0)
    day = int(match.group("dd"))
    game_date = f"{year:04d}-{month:02d}-{day:02d}" if month else ""
    away, home = _split_matchup_code(match.group("teams"))
    return {"game_date": game_date, "away_team_abbr": away, "home_team_abbr": home}


def _leg_row(parent: pd.Series, leg: dict[str, Any]) -> dict[str, Any]:
    leg_market_ticker = str(leg.get("market_ticker", "")).upper()
    leg_event_ticker = str(leg.get("event_ticker", "")).upper()
    prefix = _ticker_prefix(leg_market_ticker)
    category = NBA_LEG_PREFIX_TO_CATEGORY.get(prefix, "unknown")
    event_parts = _parse_event_parts(leg_event_ticker)
    return {
        "parent_market_ticker": str(parent.get("market_ticker", "")),
        "parent_market_title": parent.get("market_title", ""),
        "parent_status": parent.get("status", ""),
        "parent_close_time": parent.get("close_time", ""),
        "leg_event_ticker": leg_event_ticker,
        "leg_market_ticker": leg_market_ticker,
        "leg_side": str(leg.get("side", "")),
        "leg_prefix": prefix,
        "leg_category": category,
        "leg_stat_type": NBA_PROP_STAT_BY_PREFIX.get(prefix, "winner" if category == "game_winner" else category),
        "leg_line_value": _line_value_from_ticker(leg_market_ticker, category),
        "leg_game_date": event_parts["game_date"],
        "away_team_abbr": event_parts["away_team_abbr"],
        "home_team_abbr": event_parts["home_team_abbr"],
        "is_direct_single_market_price": False,
        "leg_usage_status": "inventory_only_combo_price_not_single_leg_price",
    }


def extract_multivariate_nba_legs(markets: pd.DataFrame) -> pd.DataFrame:
    """Return NBA-related selected legs from cached multivariate markets."""

    if markets.empty or "mve_selected_legs" not in markets.columns:
        return pd.DataFrame()
    frame = _normalize_market_columns(markets)
    rows: list[dict[str, Any]] = []
    for _, parent in frame.iterrows():
        for leg in _parse_legs(parent.get("mve_selected_legs")):
            ticker = str(leg.get("market_ticker", "")).upper()
            if _is_nba_leg_ticker(ticker):
                rows.append(_leg_row(parent, leg))
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values(["leg_category", "leg_event_ticker", "leg_market_ticker"]).reset_index(drop=True)


def summarize_multivariate_nba_legs(legs: pd.DataFrame) -> dict[str, Any]:
    if legs.empty:
        return {
            "rows": 0,
            "unique_legs": 0,
            "directly_backtestable_rows": 0,
            "blocked": True,
            "note": "No NBA legs were found inside cached multivariate markets.",
        }
    category_counts = legs["leg_category"].value_counts(dropna=False).to_dict()
    prefix_counts = legs["leg_prefix"].value_counts(dropna=False).to_dict()
    line_leg_rows = int(legs["leg_category"].isin(["spread_handicap", "total_points_over_under"]).sum())
    prop_leg_rows = int(legs["leg_category"].eq("player_points_rebounds_assists").sum())
    unique_line_legs = int(
        legs.loc[
            legs["leg_category"].isin(["spread_handicap", "total_points_over_under"]),
            "leg_market_ticker",
        ].nunique()
    )
    return {
        "rows": int(len(legs)),
        "unique_legs": int(legs["leg_market_ticker"].nunique()),
        "unique_parent_markets": int(legs["parent_market_ticker"].nunique()),
        "category_counts": {str(key): int(value) for key, value in category_counts.items()},
        "prefix_counts": {str(key): int(value) for key, value in prefix_counts.items()},
        "spread_total_leg_rows": line_leg_rows,
        "unique_spread_total_legs": unique_line_legs,
        "player_prop_leg_rows": prop_leg_rows,
        "directly_backtestable_rows": 0,
        "blocked": True,
        "note": (
            "These are NBA legs embedded inside multivariate combo markets. "
            "They prove line/prop inventory exists, but combo prices must not be used as single-leg prices."
        ),
    }


def extract_multivariate_nba_legs_from_file(
    input_path: str | Path,
    legs_output_path: str | Path,
    summary_output_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(input_path)
    markets = pd.read_csv(source) if source.exists() else pd.DataFrame()
    legs = extract_multivariate_nba_legs(markets)
    summary = summarize_multivariate_nba_legs(legs)
    legs_output = Path(legs_output_path)
    summary_output = Path(summary_output_path)
    legs_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    legs.to_csv(legs_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return legs, summary
