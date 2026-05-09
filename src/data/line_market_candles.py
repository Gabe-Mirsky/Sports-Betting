"""Download pregame candles for direct NBA spread, total, and prop markets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_candles import (
    EASTERN,
    SNAPSHOT_TARGETS,
    _download_market_candles,
    _extract_cutoff_ts,
    _extract_snapshot,
    _read_candle_cache,
    _with_normalized_candle_columns,
    _write_candle_cache,
)
from .kalshi_client import KalshiAPIClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKETS_PATH = PROJECT_ROOT / "data" / "raw" / "kalshi" / "underlying_nba_leg_markets.csv"
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "data" / "processed" / "kalshi_market_taxonomy.csv"
DEFAULT_CANDLE_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi" / "line_candles"
DEFAULT_PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "kalshi_line_pregame_prices.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "reports" / "kalshi_line_candle_summary.json"
DEFAULT_CATEGORIES = ["spread_handicap", "total_points_over_under"]


def _read_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _series_from_ticker(ticker: Any) -> str:
    text = str(ticker).upper()
    return text.split("-", 1)[0] if "-" in text else text


def _first_present(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return row[column]
    return ""


def _market_start_timestamp(row: pd.Series) -> tuple[pd.Timestamp | pd.NaT, str]:
    for column in ["occurrence_datetime", "expected_expiration_time", "expiration_time", "latest_expiration_time"]:
        value = _first_present(row, [column, f"{column}_raw"])
        if value != "":
            timestamp = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.notna(timestamp):
                return timestamp, f"market_{column}"

    game_date = pd.to_datetime(_first_present(row, ["game_date"]), errors="coerce")
    if pd.isna(game_date):
        return pd.NaT, "missing"
    estimated = pd.Timestamp(
        year=game_date.year,
        month=game_date.month,
        day=game_date.day,
        hour=19,
        minute=0,
        tz=EASTERN,
    )
    return estimated.tz_convert("UTC"), "estimated_7pm_eastern"


def _prepare_line_markets(
    markets: pd.DataFrame,
    taxonomy: pd.DataFrame,
    categories: list[str] | None = None,
    max_markets: int | None = None,
) -> pd.DataFrame:
    if markets.empty or taxonomy.empty:
        return pd.DataFrame()
    selected_categories = categories or DEFAULT_CATEGORIES
    if "market_ticker" not in markets.columns or "market_ticker" not in taxonomy.columns:
        return pd.DataFrame()

    taxonomy_subset = taxonomy[taxonomy["market_category"].astype(str).isin(selected_categories)].copy()
    merged = taxonomy_subset.merge(markets, on="market_ticker", how="inner", suffixes=("", "_raw"))
    if merged.empty:
        return merged

    merged["series_ticker"] = merged.apply(
        lambda row: _first_present(row, ["series_ticker", "series_ticker_raw"]) or _series_from_ticker(row["market_ticker"]),
        axis=1,
    )
    merged["market_title"] = merged.apply(
        lambda row: _first_present(row, ["market_title", "market_title_raw", "title"]),
        axis=1,
    )
    merged["line_value"] = pd.to_numeric(merged.get("line_value"), errors="coerce")
    merged = merged.sort_values(["market_category", "game_date", "market_ticker"]).drop_duplicates(
        subset=["market_ticker"],
        keep="last",
    )
    if max_markets is not None:
        merged = merged.head(int(max_markets))
    return merged.reset_index(drop=True)


def download_line_market_candles(
    markets: pd.DataFrame,
    taxonomy: pd.DataFrame,
    client: KalshiAPIClient | None = None,
    categories: list[str] | None = None,
    max_markets: int | None = None,
    force: bool = False,
    candle_dir: str | Path = DEFAULT_CANDLE_DIR,
    prices_path: str | Path = DEFAULT_PRICES_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download candles and extract pregame snapshots for direct line markets."""

    kalshi = client or KalshiAPIClient.from_env()
    line_markets = _prepare_line_markets(markets, taxonomy, categories=categories, max_markets=max_markets)
    candle_root = Path(candle_dir)
    candle_root.mkdir(parents=True, exist_ok=True)
    prices_output = Path(prices_path)
    prices_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cached_before = 0
    downloaded = 0
    failed = 0
    skipped_missing_time = 0
    cutoff_ts: int | None = None
    intervals = [1, 60, 1440]

    for _, market in line_markets.iterrows():
        market_ticker = str(market["market_ticker"])
        series_ticker = str(market.get("series_ticker") or _series_from_ticker(market_ticker))
        game_start, time_quality = _market_start_timestamp(market)
        if pd.isna(game_start):
            skipped_missing_time += 1
            continue
        game_start_ts = int(game_start.timestamp())
        start_ts = game_start_ts - 24 * 60 * 60
        candle_path = candle_root / f"{market_ticker}.parquet"

        candles = pd.DataFrame()
        if not force:
            candles = _read_candle_cache(candle_path)
            candles = _with_normalized_candle_columns(candles)
            if not candles.empty:
                cached_before += 1

        if candles.empty:
            if cutoff_ts is None:
                cutoff_ts = _extract_cutoff_ts(kalshi.get_historical_cutoff())
            candles = _download_market_candles(
                kalshi,
                market_ticker=market_ticker,
                series_ticker=series_ticker,
                start_ts=start_ts,
                end_ts=game_start_ts,
                intervals=intervals,
                cutoff_ts=cutoff_ts,
            )
            if candles.empty:
                failed += 1
            else:
                downloaded += 1
                _write_candle_cache(candles, candle_path)

        for snapshot_target, minutes in SNAPSHOT_TARGETS.items():
            snapshot = _extract_snapshot(
                candles=candles,
                game_id=market_ticker,
                market_ticker=market_ticker,
                series_ticker=series_ticker,
                game_start_ts=game_start_ts,
                snapshot_target=snapshot_target,
                minutes_before_tipoff=minutes,
                time_quality=time_quality,
            )
            snapshot.update(
                {
                    "market_category": market.get("market_category", ""),
                    "stat_type": market.get("stat_type", ""),
                    "line_value": market.get("line_value", ""),
                    "direction": market.get("direction", ""),
                    "game_date": market.get("game_date", ""),
                    "home_team_abbr": market.get("home_team_abbr", ""),
                    "away_team_abbr": market.get("away_team_abbr", ""),
                    "yes_team_abbr": market.get("yes_team_abbr", ""),
                    "market_title": market.get("market_title", ""),
                }
            )
            rows.append(snapshot)

    prices = pd.DataFrame(rows)
    prices.to_csv(prices_output, index=False)
    usable = prices[prices.get("price_quality", pd.Series(dtype=str)).astype(str).ne("missing")] if not prices.empty else prices
    summary = {
        "candidate_markets": int(len(line_markets)),
        "categories": categories or DEFAULT_CATEGORIES,
        "cached_before": int(cached_before),
        "downloaded": int(downloaded),
        "failed": int(failed),
        "skipped_missing_time": int(skipped_missing_time),
        "snapshot_rows": int(len(prices)),
        "usable_snapshot_rows": int(len(usable)),
        "usable_60m_rows": int(
            ((prices.get("snapshot_target") == "pregame_60m") & (prices.get("price_quality") != "missing")).sum()
        )
        if not prices.empty
        else 0,
        "price_path": str(prices_output),
        "candle_dir": str(candle_root),
        "note": "Line-market candles are separate from game-winner candles and are not used by the headline backtest yet.",
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return prices, summary


def download_line_market_candles_from_files(
    markets_path: str | Path = DEFAULT_MARKETS_PATH,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    markets = _read_csv(markets_path)
    taxonomy = _read_csv(taxonomy_path)
    return download_line_market_candles(markets, taxonomy, **kwargs)
