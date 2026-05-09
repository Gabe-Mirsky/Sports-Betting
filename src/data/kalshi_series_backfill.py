"""Series-first Kalshi historical market backfill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_backfill import filter_possible_nba_markets
from .kalshi_client import KalshiAPIClient
from .kalshi_taxonomy import load_cached_kalshi_markets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_KALSHI_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

NBA_SERIES_TICKERS = [
    "KXNBAGAME",
    "KXNBASPREAD",
    "KXNBATOTAL",
    "KXNBATEAMTOTAL",
    "KXNBA1HWINNER",
    "KXNBA1HSPREAD",
    "KXNBA1HTOTAL",
    "KXNBA2HWINNER",
    "KXNBA2HSPREAD",
    "KXNBA2HTOTAL",
    "KXNBAOVERTIME",
    "KXNBAPTS",
    "KXNBAREB",
    "KXNBAAST",
    "KXNBA3PT",
    "KXNBABLK",
    "KXNBASTL",
    "KXNBA2D",
    "KXNBA3D",
    "KXNBAPTSLEADER",
    "KXNBASERIES",
    "KXNBASERIESGAMES",
    "KXNBASERIESSCORE",
    "KXNBASERIESSPREAD",
    "KXNBASERIESROADWIN",
    "KXNBASERIESPTSLEADER",
    "KXNBAPLAYOFF",
    "KXNBAPLAYOFFWINS",
    "KXNBAPLAYOFFPTS",
    "KXNBAPIADVANCE",
    "KXNBAPLAYIN",
    "KXNBAPREPACK2ML",
    "KXNBAPREPACK3ML",
    "KXNBAMENTION",
]


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
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


def _write_cache(df: pd.DataFrame, path: Path, append: bool = True) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    existing = _read_cache(path) if append else pd.DataFrame()
    if not existing.empty:
        output = pd.concat([existing, output], ignore_index=True, sort=False)
    if "market_ticker" in output.columns:
        output["market_ticker"] = output["market_ticker"].astype(str)
        output = output.drop_duplicates(subset=["market_ticker"], keep="last")
    if path.suffix.lower() == ".csv":
        output.to_csv(path, index=False)
    else:
        try:
            output.to_parquet(path, index=False)
        except (ImportError, ValueError, RuntimeError):
            output.to_csv(path.with_suffix(".csv"), index=False)
    return output.reset_index(drop=True)


def _series_summary_rows(markets: pd.DataFrame) -> list[dict[str, Any]]:
    if markets.empty or "series_ticker" not in markets.columns:
        return []
    working = markets.copy()
    for column in ["close_time", "expected_expiration_time", "open_time", "created_time"]:
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce", utc=True)
    rows = []
    for series, frame in working.groupby("series_ticker", dropna=False):
        row: dict[str, Any] = {
            "series_ticker": str(series),
            "rows": int(len(frame)),
            "unique_markets": int(frame["market_ticker"].nunique()) if "market_ticker" in frame.columns else int(len(frame)),
        }
        for column in ["expected_expiration_time", "close_time", "open_time", "created_time"]:
            if column in frame.columns and frame[column].notna().any():
                row[f"{column}_min"] = frame[column].min().isoformat()
                row[f"{column}_max"] = frame[column].max().isoformat()
        rows.append(row)
    return sorted(rows, key=lambda item: item["series_ticker"])


def crawl_kalshi_series_markets(
    series_tickers: list[str] | None = None,
    client: KalshiAPIClient | None = None,
    include_historical: bool = True,
    include_recent: bool = False,
    max_pages: int = 100,
    limit: int = 1000,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    possible_output_path: str | Path | None = None,
    append: bool = True,
    rebuild_possible_from_all_raw: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Crawl Kalshi markets by series ticker and cache the raw results."""

    selected_series = [str(item).strip().upper() for item in (series_tickers or NBA_SERIES_TICKERS) if str(item).strip()]
    kalshi = client or KalshiAPIClient.from_env()
    frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []

    for series_ticker in selected_series:
        for route in (["historical"] if include_historical else []) + (["recent"] if include_recent else []):
            params = {
                "series_ticker": series_ticker,
                "limit": int(limit),
                "max_pages": int(max_pages),
            }
            markets = (
                kalshi.get_historical_markets(params)
                if route == "historical"
                else kalshi.get_markets(params)
            )
            error = str(markets.attrs.get("error", ""))
            if not markets.empty:
                markets = markets.copy()
                markets["series_backfill_route"] = route
                markets["series_backfill_series"] = series_ticker
                if "series_ticker" not in markets.columns:
                    markets["series_ticker"] = series_ticker
                else:
                    markets["series_ticker"] = markets["series_ticker"].fillna(series_ticker)
                frames.append(markets)
            run_rows.append(
                {
                    "series_ticker": series_ticker,
                    "route": route,
                    "rows_this_run": int(len(markets)),
                    "unique_markets_this_run": int(markets["market_ticker"].nunique())
                    if not markets.empty and "market_ticker" in markets.columns
                    else 0,
                    "error": error,
                }
            )

    discovered = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    output = Path(output_path) if output_path else RAW_KALSHI_DIR / "historical_series_markets.csv"
    cached = _write_cache(discovered, output, append=append)

    possible_source = load_cached_kalshi_markets(include_processed_possible=False) if rebuild_possible_from_all_raw else cached
    if possible_source.empty:
        possible_source = cached
    possible = filter_possible_nba_markets(possible_source)
    possible_output = (
        Path(possible_output_path)
        if possible_output_path
        else PROCESSED_DIR / "kalshi_possible_nba_markets.csv"
    )
    possible_output.parent.mkdir(parents=True, exist_ok=True)
    possible.to_csv(possible_output, index=False)

    summary = {
        "series_tickers": selected_series,
        "include_historical": bool(include_historical),
        "include_recent": bool(include_recent),
        "max_pages": int(max_pages),
        "limit": int(limit),
        "rows_this_run": int(len(discovered)),
        "unique_markets_this_run": int(discovered["market_ticker"].nunique())
        if not discovered.empty and "market_ticker" in discovered.columns
        else 0,
        "cached_rows": int(len(cached)),
        "cached_unique_markets": int(cached["market_ticker"].nunique())
        if not cached.empty and "market_ticker" in cached.columns
        else 0,
        "possible_game_winner_rows": int(len(possible)),
        "possible_rebuilt_from_all_raw": bool(rebuild_possible_from_all_raw),
        "run_by_series": run_rows,
        "cached_by_series": _series_summary_rows(cached),
        "raw_cache_path": str(output),
        "possible_markets_path": str(possible_output),
        "note": (
            "Historical market discovery is series-first because Kalshi historical markets support "
            "series_ticker pagination, not date-window search."
        ),
    }
    summary_output = Path(summary_path) if summary_path else REPORTS_DIR / "kalshi_historical_series_backfill_summary.json"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cached, possible, summary
