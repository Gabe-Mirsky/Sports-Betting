"""Broader NBA market discovery beyond the KXNBAGAME winner series."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_backfill import _date_to_ts, teams_mentioned_in_text
from .kalshi_client import KalshiAPIClient
from .kalshi_taxonomy import build_market_taxonomy, summarize_market_taxonomy
from .player_client import load_raw_player_logs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_KALSHI_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

BASKETBALL_DISCOVERY_TERMS = [
    "nba",
    "basketball",
    "professional basketball",
    "pro basketball",
    "national basketball association",
]


def _read_cache(path: Path) -> pd.DataFrame:
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


def _write_cache(
    df: pd.DataFrame,
    path: Path,
    dedupe_column: str = "market_ticker",
    append: bool = True,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    existing = _read_cache(path) if append else pd.DataFrame()
    if not existing.empty:
        output = pd.concat([existing, output], ignore_index=True, sort=False)
    if dedupe_column in output.columns:
        output = output.drop_duplicates(subset=[dedupe_column], keep="last")
    if path.suffix.lower() == ".csv":
        output.to_csv(path, index=False)
    else:
        try:
            output.to_parquet(path, index=False)
        except (ImportError, ValueError, RuntimeError):
            output.to_csv(path.with_suffix(".csv"), index=False)
    return output.reset_index(drop=True)


def _market_text(markets: pd.DataFrame) -> pd.Series:
    def column(name: str, fallback: str = "") -> pd.Series:
        if name in markets.columns:
            return markets[name].fillna("").astype(str)
        return pd.Series(fallback, index=markets.index)

    title = column("market_title") if "market_title" in markets.columns else column("title")
    subtitle = column("market_subtitle") if "market_subtitle" in markets.columns else column("subtitle")
    return (
        title
        + " "
        + subtitle
        + " "
        + column("yes_sub_title")
        + " "
        + column("no_sub_title")
        + " "
        + column("rules_primary")
        + " "
        + column("rules_secondary")
    ).str.lower()


def load_cached_nba_player_names(cache_dir: str | Path = "data/raw/nba/player") -> list[str]:
    """Load unique NBA player names from cached player logs for prop discovery."""

    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    player_logs = load_raw_player_logs(cache_path)
    if player_logs.empty or "PLAYER_NAME" not in player_logs.columns:
        return []
    names = (
        player_logs["PLAYER_NAME"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    return sorted(name for name in names if len(name.split()) >= 2)


def _contains_any_player_name(text: str, player_names: list[str]) -> bool:
    if not player_names:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    for name in player_names:
        player = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if player and re.search(rf"\b{re.escape(player)}\b", normalized):
            return True
    return False


def _contains_discovery_term(text: str, term: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    term_normalized = re.sub(r"[^a-z0-9]+", " ", term.lower()).strip()
    return bool(term_normalized and re.search(rf"\b{re.escape(term_normalized)}\b", normalized))


def filter_broad_nba_markets(
    markets: pd.DataFrame,
    player_names: list[str] | None = None,
) -> pd.DataFrame:
    """Keep likely NBA markets without excluding spreads, totals, or player props."""

    if markets.empty:
        return markets.copy()
    working = markets.copy()
    text = _market_text(working)
    player_names = player_names or []
    has_team = text.map(lambda value: bool(teams_mentioned_in_text(value)))
    has_basketball_term = text.map(
        lambda value: any(_contains_discovery_term(value, term) for term in BASKETBALL_DISCOVERY_TERMS)
    )
    has_player_name = text.map(lambda value: _contains_any_player_name(value, player_names))
    output = working.loc[has_team | has_basketball_term | has_player_name].copy()
    output["nba_discovery_reason"] = ""
    output.loc[has_team.loc[output.index], "nba_discovery_reason"] += "team;"
    output.loc[has_basketball_term.loc[output.index], "nba_discovery_reason"] += "basketball_term;"
    output.loc[has_player_name.loc[output.index], "nba_discovery_reason"] += "player_name;"
    return output.reset_index(drop=True)


def discover_recent_nba_markets(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    client: KalshiAPIClient | None = None,
    max_pages: int = 20,
    player_names: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch recent/live markets across all series and keep likely NBA rows."""

    kalshi = client or KalshiAPIClient.from_env()
    params: dict[str, Any] = {
        "min_close_ts": _date_to_ts(start_date),
        "max_close_ts": _date_to_ts(end_date, end_of_day=True),
        "limit": 1000,
        "max_pages": max_pages,
    }
    markets = kalshi.get_markets(params)
    if markets.empty:
        return markets
    markets["broad_discovery_window_start"] = pd.Timestamp(start_date).date().isoformat()
    markets["broad_discovery_window_end"] = pd.Timestamp(end_date).date().isoformat()
    markets["broad_discovery_route"] = "recent"
    return filter_broad_nba_markets(markets, player_names=player_names)


def discover_historical_nba_markets(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    client: KalshiAPIClient | None = None,
    max_pages: int = 20,
    player_names: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch archived markets across all series and keep likely NBA rows."""

    kalshi = client or KalshiAPIClient.from_env()
    params: dict[str, Any] = {
        "min_close_ts": _date_to_ts(start_date),
        "max_close_ts": _date_to_ts(end_date, end_of_day=True),
        "limit": 1000,
        "max_pages": max_pages,
    }
    markets = kalshi.get_historical_markets(params)
    if markets.empty:
        return markets
    markets["broad_discovery_window_start"] = pd.Timestamp(start_date).date().isoformat()
    markets["broad_discovery_window_end"] = pd.Timestamp(end_date).date().isoformat()
    markets["broad_discovery_route"] = "historical"
    return filter_broad_nba_markets(markets, player_names=player_names)


def write_broad_nba_market_discovery(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    client: KalshiAPIClient | None = None,
    max_pages: int = 20,
    output_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    replace_cache: bool = False,
    include_historical: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Discover, cache, classify, and summarize broad NBA-related markets."""

    player_names = load_cached_nba_player_names()
    recent = discover_recent_nba_markets(
        start_date=start_date,
        end_date=end_date,
        client=client,
        max_pages=max_pages,
        player_names=player_names,
    )
    frames = [recent]
    historical = pd.DataFrame()
    if include_historical:
        historical = discover_historical_nba_markets(
            start_date=start_date,
            end_date=end_date,
            client=client,
            max_pages=max_pages,
            player_names=player_names,
        )
        frames.append(historical)
    discovered = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if any(
        not frame.empty for frame in frames
    ) else pd.DataFrame()
    output = Path(output_path) if output_path else RAW_KALSHI_DIR / "broad_nba_markets.csv"
    cached = _write_cache(discovered, output, append=not replace_cache)

    taxonomy = build_market_taxonomy(cached)
    taxonomy_output = Path(taxonomy_path) if taxonomy_path else PROCESSED_DIR / "kalshi_broad_market_taxonomy.csv"
    taxonomy_output.parent.mkdir(parents=True, exist_ok=True)
    taxonomy.to_csv(taxonomy_output, index=False)

    summary = summarize_market_taxonomy(taxonomy)
    summary["discovery_window_start"] = pd.Timestamp(start_date).date().isoformat()
    summary["discovery_window_end"] = pd.Timestamp(end_date).date().isoformat()
    summary["player_names_loaded"] = len(player_names)
    summary["raw_discovered_rows_this_run"] = int(len(discovered))
    summary["raw_recent_rows_this_run"] = int(len(recent))
    summary["raw_historical_rows_this_run"] = int(len(historical))
    summary["include_historical"] = bool(include_historical)
    summary_output = Path(summary_path) if summary_path else REPORTS_DIR / "kalshi_broad_market_taxonomy_summary.json"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cached, taxonomy, summary


def write_cached_broad_nba_market_taxonomy(
    input_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild broad-market taxonomy outputs from the local broad discovery cache."""

    source = Path(input_path) if input_path else RAW_KALSHI_DIR / "broad_nba_markets.csv"
    cached = _read_cache(source)
    taxonomy = build_market_taxonomy(cached)
    taxonomy_output = Path(taxonomy_path) if taxonomy_path else PROCESSED_DIR / "kalshi_broad_market_taxonomy.csv"
    taxonomy_output.parent.mkdir(parents=True, exist_ok=True)
    taxonomy.to_csv(taxonomy_output, index=False)

    summary = summarize_market_taxonomy(taxonomy)
    summary["source_cache"] = str(source)
    summary_output = Path(summary_path) if summary_path else REPORTS_DIR / "kalshi_broad_market_taxonomy_summary.json"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return taxonomy, summary
