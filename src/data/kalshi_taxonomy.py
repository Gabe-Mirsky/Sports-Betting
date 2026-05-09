"""Classify Kalshi NBA markets into model-ready market types."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_backfill import KALSHI_MONTH_LOOKUP, _parse_kxnbagame_ticker, teams_mentioned_in_text
from .kalshi_matcher import normalize_market_text
from .team_aliases import CURRENT_TEAM_ABBRS, normalize_team_abbr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_KALSHI_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

MARKET_CATEGORIES = [
    "game_winner",
    "spread_handicap",
    "total_points_over_under",
    "team_total",
    "player_points_rebounds_assists",
    "series_playoff",
    "weird_ambiguous",
]

PLAYER_STAT_TERMS = {
    "points": ["points", "pts"],
    "rebounds": ["rebounds", "rebs"],
    "assists": ["assists", "asts"],
    "points_rebounds_assists": ["pra", "points rebounds assists"],
    "three_pointers": ["threes", "three pointers", "three-pointers", "3pt", "3 pointers"],
    "blocks": ["blocks", "blks"],
    "steals": ["steals", "stls"],
}
PLAYER_PROP_TICKER_STAT_TYPES = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBATRB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBAPRA": "points_rebounds_assists",
    "KXNBA3PT": "three_pointers",
    "KXNBABLK": "blocks",
    "KXNBASTL": "steals",
    "KXNBA2D": "double_double",
    "KXNBA3D": "triple_double",
}
SPREAD_TICKER_PREFIXES = {
    "KXNBASPREAD": "spread",
    "KXNBA1HSPREAD": "first_half_spread",
    "KXNBA2HSPREAD": "second_half_spread",
}
TOTAL_TICKER_PREFIXES = {
    "KXNBATOTAL": "total_points",
    "KXNBA1HTOTAL": "first_half_total_points",
    "KXNBA2HTOTAL": "second_half_total_points",
}
WINNER_TICKER_PREFIXES = {
    "KXNBAGAME": "winner",
    "KXNBA1HWINNER": "first_half_winner",
    "KXNBA2HWINNER": "second_half_winner",
}
SERIES_PLAYOFF_TICKER_PREFIXES = [
    "KXNBASERIES",
    "KXNBAPLAYOFF",
    "KXNBAPIADVANCE",
    "KXNBAPLAYIN",
    "KXNBAWINS",
    "KXNBAEAST",
    "KXNBAWEST",
]
SERIES_TERMS = [
    "series",
    "championship",
    "conference",
    "finals",
    "advance",
    "make the playoffs",
    "win the nba",
]
SPREAD_TERMS = ["spread", "handicap", "cover", "cover the"]
TOTAL_TERMS = ["total points", "combined points", "combined score", "game total"]
TEAM_TOTAL_TERMS = ["team total", "score over", "score under", "points in the game"]
WIN_TERMS = ["winner", " win", " wins", " beat", " defeat"]
OVER_UNDER_TERMS = ["over", "under", "at least", "more than", "less than"]


def _read_market_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError, RuntimeError):
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError, RuntimeError):
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def _first_present(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return row[name]
    return ""


def _normalize_market_columns(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    if "market_ticker" not in output.columns and "ticker" in output.columns:
        output["market_ticker"] = output["ticker"]
    if "market_title" not in output.columns and "title" in output.columns:
        output["market_title"] = output["title"]
    if "market_subtitle" not in output.columns and "subtitle" in output.columns:
        output["market_subtitle"] = output["subtitle"]
    if "series_ticker" not in output.columns:
        output["series_ticker"] = output["market_ticker"].astype(str).str.extract(r"^([A-Z0-9]+)-", expand=False)
    output["source_file"] = source_file
    return output


def load_cached_kalshi_markets(
    raw_dir: str | Path = RAW_KALSHI_DIR,
    include_processed_possible: bool = True,
) -> pd.DataFrame:
    """Load cached Kalshi market rows and dedupe by market ticker."""

    root = Path(raw_dir)
    candidate_paths = [
        root / "live_markets.parquet",
        root / "live_markets.csv",
        root / "historical_markets.parquet",
        root / "historical_markets.csv",
        root / "historical_series_markets.parquet",
        root / "historical_series_markets.csv",
        root / "targeted_nba_game_markets.parquet",
        root / "targeted_nba_game_markets.csv",
        root / "kxnbagame_markets_by_series.parquet",
        root / "kxnbagame_markets_by_series.csv",
        root / "broad_nba_markets.parquet",
        root / "broad_nba_markets.csv",
        root / "underlying_nba_leg_markets.parquet",
        root / "underlying_nba_leg_markets.csv",
    ]
    if include_processed_possible:
        candidate_paths.extend(
            [
                PROCESSED_DIR / "kalshi_possible_nba_markets.parquet",
                PROCESSED_DIR / "kalshi_possible_nba_markets.csv",
            ]
        )

    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()
    for path in candidate_paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        frame = _read_market_cache(path)
        if frame.empty:
            continue
        frames.append(_normalize_market_columns(frame, path.name))

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[combined["market_ticker"].notna()].copy()
    combined["market_ticker"] = combined["market_ticker"].astype(str)
    combined = combined[combined["market_ticker"].str.strip().ne("")]
    combined["source_files"] = combined.groupby("market_ticker")["source_file"].transform(
        lambda values: ",".join(sorted(set(values.dropna().astype(str))))
    )
    combined = combined.drop_duplicates(subset=["market_ticker"], keep="last")
    return combined.reset_index(drop=True)


def _combined_text(row: pd.Series) -> str:
    fields = [
        _first_present(row, ["market_title", "title"]),
        _first_present(row, ["market_subtitle", "subtitle"]),
        row.get("yes_sub_title", ""),
        row.get("no_sub_title", ""),
        row.get("rules_primary", ""),
        row.get("rules_secondary", ""),
    ]
    return " ".join(str(value) for value in fields if pd.notna(value))


def _extract_line_value(text: str) -> float | None:
    patterns = [
        r"(?:over|under|at least|more than|less than)\s+(-?\d+(?:\.\d+)?)",
        r"\b(-?\d+(?:\.\d+)?)\+",
        r"([+-]\d+(?:\.\d+)?)",
        r"by\s+(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _extract_direction(text: str) -> str:
    if re.search(r"\b(over|more than|at least)\b", text):
        return "over"
    if re.search(r"\b\d+(?:\.\d+)?\+", text):
        return "over"
    if re.search(r"\b(under|less than)\b", text):
        return "under"
    return ""


def _extract_player_stat_type(text: str) -> str:
    found = []
    for stat_type, aliases in PLAYER_STAT_TERMS.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            found.append(stat_type)
    if "points_rebounds_assists" in found:
        return "points_rebounds_assists"
    if len(found) == 1:
        return found[0]
    if found:
        return "+".join(found)
    return ""


def _extract_player_name(raw_text: str, normalized_text: str) -> str:
    raw_match = re.search(r"^\s*(.+?):\s*\d+(?:\.\d+)?\+", raw_text)
    if raw_match:
        candidate = raw_match.group(1).strip()
        if candidate:
            return candidate.title()
    patterns = [
        r"will\s+(.+?)\s+(?:score|record|have|get)\b",
        r"(.+?)\s+(?:over|under|at least|more than|less than)\s+\d",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_text)
        if match:
            candidate = match.group(1).strip()
            if candidate and not any(term in candidate for term in ["team", "total", "game"]):
                return candidate.title()
    return ""


def _team_from_yes_subtitle(row: pd.Series, mentioned_teams: list[str]) -> str:
    yes_text = normalize_market_text(row.get("yes_sub_title", ""))
    if not yes_text:
        return ""
    for team in mentioned_teams:
        if normalize_market_text(team) == yes_text:
            return team
    candidate = normalize_team_abbr(row.get("yes_sub_title", ""))
    return candidate if candidate in set(mentioned_teams) else ""


def _parse_kxnba_market_ticker(ticker: str) -> dict[str, Any] | None:
    match = re.match(
        r"^(?P<series>KXNBA[A-Z0-9]+)-(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<day>\d{2})"
        r"(?P<matchup>[A-Z]{6})-(?P<rest>.+)$",
        ticker.upper(),
    )
    if not match:
        return None
    month = KALSHI_MONTH_LOOKUP.get(match.group("month"))
    if month is None:
        return None
    matchup = match.group("matchup")
    away = normalize_team_abbr(matchup[:3])
    home = normalize_team_abbr(matchup[3:])
    if away not in CURRENT_TEAM_ABBRS or home not in CURRENT_TEAM_ABBRS:
        return None
    rest = match.group("rest")
    yes_team = ""
    for length in [3, 2]:
        candidate = normalize_team_abbr(rest[:length])
        if candidate in CURRENT_TEAM_ABBRS:
            yes_team = candidate
            break
    no_team = home if yes_team == away else away if yes_team == home else ""
    event_ticker = (
        f"{match.group('series')}-{match.group('year')}{match.group('month')}{match.group('day')}{away}{home}"
    )
    game_date = pd.Timestamp(
        year=2000 + int(match.group("year")),
        month=month,
        day=int(match.group("day")),
    )
    return {
        "game_date": game_date.date().isoformat(),
        "away_team_abbr": away,
        "home_team_abbr": home,
        "yes_team_abbr": yes_team,
        "no_team_abbr": no_team,
        "series_ticker": match.group("series"),
        "event_ticker": event_ticker,
    }


def _category_from_ticker(ticker: str) -> tuple[str, str, str, float] | None:
    for prefix, stat_type in SPREAD_TICKER_PREFIXES.items():
        if ticker.startswith(f"{prefix}-"):
            return "spread_handicap", "game", stat_type, 0.90
    for prefix, stat_type in TOTAL_TICKER_PREFIXES.items():
        if ticker.startswith(f"{prefix}-"):
            return "total_points_over_under", "game", stat_type, 0.90
    if ticker.startswith("KXNBATEAMTOTAL-"):
        return "team_total", "team", "team_points", 0.90
    for prefix, stat_type in WINNER_TICKER_PREFIXES.items():
        if ticker.startswith(f"{prefix}-"):
            return "game_winner", "game", stat_type, 0.95
    for prefix, stat_type in PLAYER_PROP_TICKER_STAT_TYPES.items():
        if ticker.startswith(f"{prefix}-"):
            return "player_points_rebounds_assists", "player", stat_type, 0.85
    if ticker.startswith("KXNBAPTSLEADER-"):
        return "player_points_rebounds_assists", "player", "points_leader", 0.70
    if any(ticker.startswith(f"{prefix}-") for prefix in SERIES_PLAYOFF_TICKER_PREFIXES):
        return "series_playoff", "series", "series_or_future", 0.85
    return None


def _category_from_text(ticker: str, text: str, mentioned_teams: list[str]) -> tuple[str, str, str, float]:
    ticker_category = _category_from_ticker(ticker)
    if ticker_category:
        return ticker_category

    has_ou = any(re.search(rf"\b{re.escape(term)}\b", text) for term in OVER_UNDER_TERMS)
    has_player_stat = bool(_extract_player_stat_type(text))
    has_points = bool(re.search(r"\b(points|pts)\b", text))
    has_win = any(term in f" {text}" for term in WIN_TERMS)
    has_series = any(term in text for term in SERIES_TERMS)
    has_spread = any(term in text for term in SPREAD_TERMS)
    has_total = any(term in text for term in TOTAL_TERMS)
    has_team_total = any(term in text for term in TEAM_TOTAL_TERMS)

    if has_series and not ticker.startswith("KXNBAGAME-"):
        return "series_playoff", "series", "series_or_future", 0.85
    if has_player_stat and has_ou and not has_total and (not has_team_total or not mentioned_teams):
        return "player_points_rebounds_assists", "player", _extract_player_stat_type(text), 0.75
    if has_spread:
        return "spread_handicap", "game", "spread", 0.80
    if has_team_total and has_points and has_ou:
        return "team_total", "team", "team_points", 0.75
    if has_total and has_ou:
        return "total_points_over_under", "game", "total_points", 0.80
    if ticker.startswith("KXNBAGAME-") or (has_win and len(set(mentioned_teams)) <= 2):
        return "game_winner", "game", "winner", 0.95 if ticker.startswith("KXNBAGAME-") else 0.75
    return "weird_ambiguous", "unknown", "", 0.30


def classify_kalshi_market(row: pd.Series) -> dict[str, Any]:
    """Classify one market row into a broad NBA market type."""

    ticker = str(row.get("market_ticker", row.get("ticker", ""))).upper()
    raw_text = _combined_text(row)
    raw_lower = raw_text.lower()
    text = normalize_market_text(raw_text)
    mentioned_teams = teams_mentioned_in_text(raw_text)
    parsed_ticker = _parse_kxnbagame_ticker(ticker) or _parse_kxnba_market_ticker(ticker) or {}
    if parsed_ticker:
        for team in [parsed_ticker.get("home_team_abbr"), parsed_ticker.get("away_team_abbr")]:
            if team and team not in mentioned_teams:
                mentioned_teams.append(team)

    is_multivariate = ticker.startswith("KXMVE") or any(
        pd.notna(row.get(column, "")) and str(row.get(column, "")).strip()
        for column in ["mve_collection_ticker", "mve_selected_legs"]
    )
    if is_multivariate:
        category, scope, stat_type, confidence = "weird_ambiguous", "multivariate", "multivariate", 0.40
    else:
        category, scope, stat_type, confidence = _category_from_text(ticker, text, mentioned_teams)
    line_value = _extract_line_value(raw_lower)
    direction = _extract_direction(raw_lower) or _extract_direction(text)
    yes_team = parsed_ticker.get("yes_team_abbr", "") or _team_from_yes_subtitle(row, mentioned_teams)
    no_team = parsed_ticker.get("no_team_abbr", "")
    player_name = _extract_player_name(raw_text, text) if category == "player_points_rebounds_assists" else ""
    notes = []
    if category == "weird_ambiguous":
        notes.append("no high-confidence taxonomy rule matched")
    if is_multivariate:
        notes.append("multivariate combination market; exclude until portfolio modeling is explicit")
    if category in {"spread_handicap", "total_points_over_under", "team_total", "player_points_rebounds_assists"} and line_value is None:
        notes.append("line value not found in title/rules")
        confidence = min(confidence, 0.55)
    if category == "player_points_rebounds_assists" and not player_name:
        notes.append("player name not extracted")
        confidence = min(confidence, 0.60)

    return {
        "market_ticker": ticker,
        "series_ticker": row.get("series_ticker", parsed_ticker.get("series_ticker", "")),
        "event_ticker": row.get("event_ticker", parsed_ticker.get("event_ticker", "")),
        "market_title": _first_present(row, ["market_title", "title"]),
        "market_subtitle": _first_present(row, ["market_subtitle", "subtitle"]),
        "yes_sub_title": row.get("yes_sub_title", ""),
        "no_sub_title": row.get("no_sub_title", ""),
        "market_category": category,
        "market_scope": scope,
        "stat_type": stat_type,
        "line_value": line_value,
        "direction": direction,
        "player_name": player_name,
        "mentioned_team_abbrs": ",".join(sorted(set(mentioned_teams))),
        "home_team_abbr": parsed_ticker.get("home_team_abbr", row.get("home_team_abbr", "")),
        "away_team_abbr": parsed_ticker.get("away_team_abbr", row.get("away_team_abbr", "")),
        "yes_team_abbr": yes_team,
        "no_team_abbr": no_team,
        "game_date": parsed_ticker.get("game_date", row.get("game_date", "")),
        "taxonomy_confidence": round(float(confidence), 4),
        "taxonomy_notes": "; ".join(notes),
        "source_files": row.get("source_files", row.get("source_file", "")),
    }


def build_market_taxonomy(markets: pd.DataFrame) -> pd.DataFrame:
    """Classify a dataframe of market rows."""

    if markets.empty:
        return pd.DataFrame(
            columns=[
                "market_ticker",
                "series_ticker",
                "event_ticker",
                "market_title",
                "market_category",
                "taxonomy_confidence",
            ]
        )
    rows = [classify_kalshi_market(row) for _, row in markets.iterrows()]
    taxonomy = pd.DataFrame(rows)
    taxonomy["market_category"] = pd.Categorical(
        taxonomy["market_category"],
        categories=MARKET_CATEGORIES,
        ordered=False,
    )
    return taxonomy.sort_values(["market_category", "game_date", "market_ticker"]).reset_index(drop=True)


def summarize_market_taxonomy(taxonomy: pd.DataFrame) -> dict[str, Any]:
    if taxonomy.empty:
        return {
            "market_rows": 0,
            "category_counts": {},
            "low_confidence_rows": 0,
            "categories_ready_for_modeling": [],
        }
    counts = taxonomy["market_category"].astype(str).value_counts(dropna=False).to_dict()
    scope_counts = (
        taxonomy.get("market_scope", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .replace("", "unknown")
        .value_counts(dropna=False)
        .to_dict()
    )
    stat_type_counts = (
        taxonomy.get("stat_type", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .replace("", "unknown")
        .value_counts(dropna=False)
        .to_dict()
    )
    ready = sorted(
        taxonomy.loc[taxonomy["market_category"].isin(["game_winner", "spread_handicap", "total_points_over_under"]), "market_category"]
        .astype(str)
        .unique()
        .tolist()
    )
    return {
        "market_rows": int(len(taxonomy)),
        "unique_markets": int(taxonomy["market_ticker"].nunique()),
        "category_counts": {str(key): int(value) for key, value in counts.items()},
        "scope_counts": {str(key): int(value) for key, value in scope_counts.items()},
        "stat_type_counts": {str(key): int(value) for key, value in stat_type_counts.items()},
        "low_confidence_rows": int((taxonomy["taxonomy_confidence"] < 0.70).sum()),
        "categories_ready_for_modeling": ready,
        "note": "Current cached NBA Kalshi markets are mostly game-winner rows; broader market types will appear once raw caches include them.",
    }


def write_market_taxonomy_outputs(
    raw_dir: str | Path = RAW_KALSHI_DIR,
    taxonomy_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    include_processed_possible: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    markets = load_cached_kalshi_markets(raw_dir, include_processed_possible=include_processed_possible)
    taxonomy = build_market_taxonomy(markets)
    summary = summarize_market_taxonomy(taxonomy)

    taxonomy_output = Path(taxonomy_path) if taxonomy_path else PROCESSED_DIR / "kalshi_market_taxonomy.csv"
    summary_output = Path(summary_path) if summary_path else REPORTS_DIR / "kalshi_market_taxonomy_summary.json"
    _write_table(taxonomy, taxonomy_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return taxonomy, summary
