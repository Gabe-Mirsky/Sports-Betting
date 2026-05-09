"""Match NBA games to likely Kalshi NBA team-win markets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_backfill import kalshi_event_ticker_for_game
from .kalshi_backfill import NBA_TEAM_ALIASES, teams_mentioned_in_text
from .team_aliases import normalize_team_abbr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCH_OUTPUT_COLUMNS = [
    "game_id",
    "game_date",
    "home_team_abbr",
    "away_team_abbr",
    "market_ticker",
    "series_ticker",
    "event_ticker",
    "market_title",
    "market_subtitle",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "expiration_time",
    "latest_expiration_time",
    "yes_team_abbr",
    "no_team_abbr",
    "match_score",
    "match_status",
    "match_notes",
]

NEGATIVE_PROP_TERMS = [
    "spread",
    "total",
    "over",
    "under",
    "points",
    "rebounds",
    "assists",
]
NEGATIVE_FUTURES_TERMS = [
    "series",
    "championship",
    "finals",
    "conference",
]
WIN_TERMS = ["win", "wins", "beat", "defeat"]


def normalize_market_text(value: object) -> str:
    """Lowercase market text and remove punctuation for matching."""

    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def _market_text(row: pd.Series) -> str:
    title = row.get("market_title", row.get("title", ""))
    subtitle = row.get("market_subtitle", row.get("subtitle", ""))
    return normalize_market_text(f"{title} {subtitle}")


def _team_aliases(abbr: str) -> list[str]:
    return [normalize_market_text(alias) for alias in NBA_TEAM_ALIASES.get(abbr, [abbr])]


def _team_appears(text: str, abbr: str) -> bool:
    return any(alias and alias in text for alias in _team_aliases(abbr))


def _first_team_mentioned(text: str, teams: list[str]) -> str | None:
    positions: list[tuple[int, str]] = []
    for abbr in teams:
        for alias in _team_aliases(abbr):
            if not alias:
                continue
            position = text.find(alias)
            if position >= 0:
                positions.append((position, abbr))
    if not positions:
        return None
    return sorted(positions, key=lambda item: item[0])[0][1]


def _ticker_yes_team(row: pd.Series, teams: set[str]) -> str | None:
    ticker = str(row.get("market_ticker", row.get("ticker", ""))).upper()
    if "-" not in ticker:
        return None
    suffix = ticker.rsplit("-", maxsplit=1)[-1]
    normalized = normalize_team_abbr(suffix)
    if normalized in teams:
        return normalized
    return None


def identify_yes_team(row: pd.Series, home_team: str, away_team: str) -> str | None:
    """Infer which NBA team the YES side refers to, returning None when ambiguous."""

    teams = {home_team, away_team}
    for column in ["yes_team_abbr", "yes_team", "yes_subtitle"]:
        if column in row and pd.notna(row[column]):
            candidate = normalize_team_abbr(row[column])
            if candidate in teams:
                return candidate

    ticker_team = _ticker_yes_team(row, teams)
    if ticker_team:
        return ticker_team

    text = _market_text(row)
    if not any(term in text for term in WIN_TERMS):
        return None
    return _first_team_mentioned(text, [home_team, away_team])


def _market_datetime(row: pd.Series) -> pd.Timestamp | pd.NaT:
    for column in [
        "close_time",
        "expected_expiration_time",
        "latest_expiration_time",
        "expiration_time",
        "open_time",
    ]:
        if column in row and pd.notna(row[column]):
            return pd.to_datetime(row[column], errors="coerce", utc=True)
    return pd.NaT


def _market_game_date(row: pd.Series) -> pd.Timestamp | pd.NaT:
    if "game_date" in row and pd.notna(row["game_date"]):
        parsed = pd.to_datetime(row["game_date"], errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
    market_datetime = _market_datetime(row)
    if pd.isna(market_datetime):
        return pd.NaT
    return market_datetime.tz_convert(None).normalize() if market_datetime.tzinfo else market_datetime.normalize()


def _candidate_markets_for_game(
    game_date: pd.Timestamp,
    markets: pd.DataFrame,
    search_days_before: int,
    search_days_after: int,
) -> pd.DataFrame:
    if markets.empty or "market_date" not in markets.columns:
        return markets

    start_date = game_date.normalize() - pd.Timedelta(days=search_days_before)
    end_date = game_date.normalize() + pd.Timedelta(days=search_days_after)
    dated = markets["market_date"].notna()
    within = dated & (markets["market_date"] >= start_date) & (markets["market_date"] <= end_date)
    return markets.loc[within | ~dated].copy()


def _score_market_for_game(
    row: pd.Series,
    game: pd.Series,
    auto_match_threshold: float,
    review_match_threshold: float,
) -> dict[str, Any]:
    home_team = normalize_team_abbr(game["home_team_abbr"])
    away_team = normalize_team_abbr(game["away_team_abbr"])
    game_date = pd.to_datetime(game["game_date"], errors="coerce").normalize()
    text = _market_text(row)
    title = normalize_market_text(row.get("market_title", row.get("title", "")))
    expected_event_ticker = kalshi_event_ticker_for_game(game_date, away_team, home_team)
    market_ticker = str(row.get("market_ticker", row.get("ticker", "")))

    home_found = _team_appears(text, home_team)
    away_found = _team_appears(text, away_team)
    mentioned_teams = teams_mentioned_in_text(text)
    yes_team = identify_yes_team(row, home_team, away_team)
    score = 0.0
    notes: list[str] = []

    if home_found:
        score += 0.35
        notes.append("home team mentioned")
    if away_found:
        score += 0.35
        notes.append("away team mentioned")
    if any(term in text for term in WIN_TERMS):
        score += 0.15
        notes.append("win language found")
    if market_ticker.startswith(f"{expected_event_ticker}-"):
        score += 0.70
        notes.append("exact Kalshi matchup ticker found")

    market_date = row.get("market_date")
    if pd.notna(market_date) and pd.Timestamp(market_date).normalize() == game_date:
        score += 0.10
        notes.append("market date matches game date")

    if home_found and away_found and (" vs " in title or " beat " in title or " defeat " in title):
        score += 0.05
        notes.append("exact matchup language found")

    if any(term in text for term in NEGATIVE_PROP_TERMS):
        score -= 0.40
        notes.append("prop/spread/total term found")
    if any(term in text for term in NEGATIVE_FUTURES_TERMS):
        score -= 0.30
        notes.append("series/futures term found")
    if len(set(mentioned_teams)) > 2:
        score -= 0.20
        notes.append("more than two NBA teams mentioned")
    if yes_team is None:
        score -= 0.20
        notes.append("YES team could not be identified")

    score = max(0.0, min(1.0, score))
    if score >= auto_match_threshold:
        status = "auto_matched"
    elif score >= review_match_threshold:
        status = "needs_review"
    else:
        status = "no_match"

    no_team = ""
    if yes_team == home_team:
        no_team = away_team
    elif yes_team == away_team:
        no_team = home_team

    return {
        "market_ticker": row.get("market_ticker", row.get("ticker", "")),
        "series_ticker": row.get("series_ticker", ""),
        "event_ticker": row.get("event_ticker", ""),
        "market_title": row.get("market_title", row.get("title", "")),
        "market_subtitle": row.get("market_subtitle", row.get("subtitle", "")),
        "open_time": row.get("open_time", ""),
        "close_time": row.get("close_time", ""),
        "expected_expiration_time": row.get("expected_expiration_time", ""),
        "expiration_time": row.get("expiration_time", ""),
        "latest_expiration_time": row.get("latest_expiration_time", ""),
        "yes_team_abbr": yes_team or "",
        "no_team_abbr": no_team,
        "match_score": round(score, 4),
        "match_status": status,
        "match_notes": "; ".join(notes),
    }


def match_games_to_kalshi_markets(
    nba_games_df: pd.DataFrame,
    kalshi_markets_df: pd.DataFrame,
    auto_match_threshold: float = 0.85,
    review_match_threshold: float = 0.60,
    search_days_before: int = 1,
    search_days_after: int = 1,
) -> pd.DataFrame:
    """Find the best Kalshi market match for each NBA game."""

    games = nba_games_df.copy()
    markets = kalshi_markets_df.copy()
    if games.empty:
        return pd.DataFrame(columns=MATCH_OUTPUT_COLUMNS)

    games["game_date"] = _parse_dates(games["game_date"]).dt.normalize()
    games["home_team_abbr"] = games["home_team_abbr"].map(normalize_team_abbr)
    games["away_team_abbr"] = games["away_team_abbr"].map(normalize_team_abbr)
    games = games.dropna(subset=["game_date", "home_team_abbr", "away_team_abbr"]).copy()

    if not markets.empty:
        markets["market_date"] = markets.apply(_market_game_date, axis=1)

    rows: list[dict[str, Any]] = []
    for _, game in games.sort_values(["game_date", "game_id"]).iterrows():
        base_row = {
            "game_id": str(game.get("game_id", "")),
            "game_date": pd.Timestamp(game["game_date"]).date().isoformat(),
            "home_team_abbr": game["home_team_abbr"],
            "away_team_abbr": game["away_team_abbr"],
        }
        candidates = _candidate_markets_for_game(
            game["game_date"],
            markets,
            search_days_before=search_days_before,
            search_days_after=search_days_after,
        )
        scored = [
            _score_market_for_game(
                row,
                game,
                auto_match_threshold=auto_match_threshold,
                review_match_threshold=review_match_threshold,
            )
            for _, row in candidates.iterrows()
        ]
        if scored:
            best = sorted(scored, key=lambda item: item["match_score"], reverse=True)[0]
        else:
            best = {
                "market_ticker": "",
                "series_ticker": "",
                "event_ticker": "",
                "market_title": "",
                "market_subtitle": "",
                "yes_team_abbr": "",
                "no_team_abbr": "",
                "match_score": 0.0,
                "match_status": "no_match",
                "match_notes": "no market candidates in search window",
            }
        rows.append({**base_row, **best})

    return pd.DataFrame(rows, columns=MATCH_OUTPUT_COLUMNS)


def save_match_outputs(
    matches: pd.DataFrame,
    matches_path: str | Path | None = None,
    review_path: str | Path | None = None,
) -> None:
    """Save all matches and the needs-review subset."""

    output_path = Path(matches_path) if matches_path else PROJECT_ROOT / "data" / "processed" / "kalshi_game_market_matches.csv"
    review_output_path = (
        Path(review_path)
        if review_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_matches_needs_review.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    matches[matches["match_status"] == "needs_review"].to_csv(review_output_path, index=False)
