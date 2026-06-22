"""Raw-first public Kalshi Sports/NBA market backfill."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_backfill import filter_possible_nba_markets
from .kalshi_client import KalshiAPIClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PUBLIC_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi" / "public_api"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NBA_SERIES_TERMS = [
    "nba",
    "pro basketball",
    "basketball",
    "celtics",
    "lakers",
    "knicks",
    "warriors",
]


def _optional_status_params(status: str | None) -> dict[str, str]:
    if status is None:
        return {}
    value = str(status).strip()
    if not value or value.lower() in {"all", "any", "none", "omit"}:
        return {}
    return {"status": value}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))
    return cleaned.strip("_") or "root"


def _cache_key(path: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"path": path, "params": params}, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_name(path)}_{digest}.json"


def _raw_request(
    client: KalshiAPIClient,
    path: str,
    params: dict[str, Any],
    raw_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / _cache_key(path, params)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = client.get_json(path, params)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def _paged_records(
    client: KalshiAPIClient,
    path: str,
    params: dict[str, Any],
    record_key: str,
    raw_dir: Path,
    force: bool = False,
    max_pages: int = 100,
    sleep_seconds: float = 0.0,
    alternate_record_keys: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_params = dict(params)
    request_params.setdefault("limit", 200)
    cursor = str(request_params.pop("cursor", "") or "")
    records: list[dict[str, Any]] = []
    raw_payloads: list[dict[str, Any]] = []
    for _ in range(max_pages):
        page_params = dict(request_params)
        if cursor:
            page_params["cursor"] = cursor
        payload = _raw_request(client, path, page_params, raw_dir, force=force)
        raw_payloads.append(payload)
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
        if payload.get("error"):
            break
        page_records = payload.get(record_key) or []
        if not page_records:
            for alternate_key in alternate_record_keys or []:
                page_records = payload.get(alternate_key) or []
                if page_records:
                    break
        if isinstance(page_records, list):
            records.extend([record for record in page_records if isinstance(record, dict)])
        cursor = str(payload.get("cursor") or "")
        if not cursor:
            break
    return records, raw_payloads


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


def _normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep="_")


def _normalize_series(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _normalize_records(records)
    if frame.empty:
        return frame
    if "ticker" in frame.columns and "series_ticker" not in frame.columns:
        frame["series_ticker"] = frame["ticker"]
    text_columns = [column for column in ["series_ticker", "title", "subtitle", "category"] if column in frame.columns]
    frame["series_match_text"] = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    frame["is_nba_candidate"] = frame["series_match_text"].map(
        lambda text: any(term in text for term in NBA_SERIES_TERMS)
    )
    return frame


def _normalize_events(records: list[dict[str, Any]], series_ticker: str) -> pd.DataFrame:
    frame = _normalize_records(records)
    if frame.empty:
        return frame
    if "event_ticker" not in frame.columns and "ticker" in frame.columns:
        frame["event_ticker"] = frame["ticker"]
    if "series_ticker" not in frame.columns:
        frame["series_ticker"] = series_ticker
    if "title" in frame.columns and "event_title" not in frame.columns:
        frame["event_title"] = frame["title"]
    return frame


def _normalize_markets(records: list[dict[str, Any]], event: pd.Series | None = None) -> pd.DataFrame:
    frame = _normalize_records(records)
    if frame.empty:
        return frame
    rename_map = {
        "ticker": "market_ticker",
        "title": "market_title",
        "subtitle": "market_subtitle",
    }
    frame = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})
    if event is not None:
        for column in ["event_ticker", "series_ticker", "event_title"]:
            if column not in frame.columns:
                frame[column] = event.get(column, "")
            else:
                frame[column] = frame[column].where(frame[column].notna(), event.get(column, ""))
    for column in [
        "open_time",
        "close_time",
        "expiration_time",
        "expected_expiration_time",
        "latest_expiration_time",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    text_columns = [
        column
        for column in ["market_ticker", "market_title", "market_subtitle", "event_title"]
        if column in frame.columns
    ]
    frame["market_match_text"] = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    return frame


def discover_nba_series(series: pd.DataFrame) -> pd.DataFrame:
    """Return Sports series likely to contain NBA or pro basketball markets."""

    if series.empty:
        return series.copy()
    frame = series.copy()
    if "is_nba_candidate" not in frame.columns:
        text = frame.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        frame["is_nba_candidate"] = text.map(lambda value: any(term in value for term in NBA_SERIES_TERMS))
    return frame[frame["is_nba_candidate"]].copy().reset_index(drop=True)


def backfill_public_sports_nba_markets(
    client: KalshiAPIClient | None = None,
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    category: str = "Sports",
    event_status: str = "settled",
    market_status: str = "settled",
    include_historical_markets: bool = True,
    force: bool = False,
    max_pages: int = 100,
    series_tickers: list[str] | None = None,
    max_events_per_series: int | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    """Backfill raw Kalshi Sports series, NBA events, and NBA market candidates."""

    kalshi = client or KalshiAPIClient.from_env()
    raw_root = Path(raw_dir) if raw_dir else RAW_PUBLIC_DIR
    table_root = Path(output_dir) if output_dir else RAW_PUBLIC_DIR
    raw_json_dir = raw_root / "json"

    series_records, _ = _paged_records(
        kalshi,
        "/series",
        {"category": category, "include_volume": "true"},
        "series",
        raw_json_dir,
        force=force,
        max_pages=max_pages,
        sleep_seconds=sleep_seconds,
    )
    sports_series = _normalize_series(series_records)
    nba_series = discover_nba_series(sports_series)
    if series_tickers:
        requested = {ticker.strip().upper() for ticker in series_tickers if ticker.strip()}
        if "series_ticker" in nba_series.columns:
            matching_candidates = nba_series["series_ticker"].astype(str).str.upper().isin(requested)
            forced_candidates = (
                sports_series["series_ticker"].astype(str).str.upper().isin(requested)
                if "series_ticker" in sports_series.columns
                else pd.Series(False, index=sports_series.index)
            )
            nba_series = pd.concat(
                [nba_series[matching_candidates], sports_series[forced_candidates]],
                ignore_index=True,
                sort=False,
            ).drop_duplicates(subset=["series_ticker"], keep="first")

    all_events: list[pd.DataFrame] = []
    all_markets: list[pd.DataFrame] = []
    series_tickers = nba_series["series_ticker"].dropna().astype(str).unique().tolist() if "series_ticker" in nba_series.columns else []
    for series_ticker in series_tickers:
        event_records, _ = _paged_records(
            kalshi,
            "/events",
            {
                "series_ticker": series_ticker,
                "with_milestones": "true",
                **_optional_status_params(event_status),
            },
            "events",
            raw_json_dir,
            force=force,
            max_pages=max_pages,
            sleep_seconds=sleep_seconds,
        )
        events = _normalize_events(event_records, series_ticker)
        if events.empty:
            continue
        if max_events_per_series is not None:
            events = events.head(int(max_events_per_series)).copy()
        all_events.append(events)
        for _, event in events.iterrows():
            event_ticker = str(event.get("event_ticker", "") or event.get("ticker", ""))
            if not event_ticker:
                continue
            market_records, _ = _paged_records(
                kalshi,
                "/markets",
                {"event_ticker": event_ticker, **_optional_status_params(market_status)},
                "markets",
                raw_json_dir,
                force=force,
                max_pages=max_pages,
                sleep_seconds=sleep_seconds,
                alternate_record_keys=["markets"],
            )
            markets = _normalize_markets(market_records, event)
            if not markets.empty:
                markets["kalshi_data_tier"] = "recent_or_live"
                all_markets.append(markets)

        if include_historical_markets:
            historical_records, _ = _paged_records(
                kalshi,
                "/historical/markets",
                {"series_ticker": series_ticker},
                "historical_markets",
                raw_json_dir,
                force=force,
                max_pages=max_pages,
                sleep_seconds=sleep_seconds,
                alternate_record_keys=["markets"],
            )
            historical = _normalize_markets(historical_records)
            if not historical.empty:
                historical["series_ticker"] = historical.get("series_ticker", series_ticker)
                historical["kalshi_data_tier"] = "historical"
                all_markets.append(historical)

    events_output = pd.concat(all_events, ignore_index=True, sort=False) if all_events else pd.DataFrame()
    markets_output = pd.concat(all_markets, ignore_index=True, sort=False) if all_markets else pd.DataFrame()
    if not markets_output.empty and "market_ticker" in markets_output.columns:
        markets_output = markets_output.drop_duplicates(subset=["market_ticker"], keep="last")
    possible = filter_possible_nba_markets(markets_output) if not markets_output.empty else pd.DataFrame()

    possible_output_path = (
        PROCESSED_DIR / "kalshi_public_possible_nba_markets.csv"
        if output_dir is None
        else table_root / "kalshi_public_possible_nba_markets.csv"
    )
    paths = {
        "sports_series": _write_table(sports_series, table_root / "sports_series.csv"),
        "nba_series": _write_table(nba_series, table_root / "nba_series.csv"),
        "nba_events": _write_table(events_output, table_root / "nba_events.csv"),
        "nba_markets": _write_table(markets_output, table_root / "nba_markets.csv"),
        "possible_nba_markets": _write_table(possible, possible_output_path),
    }
    return {
        "sports_series": sports_series,
        "nba_series": nba_series,
        "nba_events": events_output,
        "nba_markets": markets_output,
        "possible_nba_markets": possible,
        "summary": {
            "sports_series": int(len(sports_series)),
            "nba_series": int(len(nba_series)),
            "nba_events": int(len(events_output)),
            "nba_markets": int(len(markets_output)),
            "possible_nba_markets": int(len(possible)),
            "raw_json_dir": str(raw_json_dir),
            "paths": {key: str(value) for key, value in paths.items()},
        },
    }
