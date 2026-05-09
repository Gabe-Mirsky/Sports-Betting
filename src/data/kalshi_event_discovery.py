"""Discover NBA Kalshi series through series and events endpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_backfill import NBA_TEAM_ALIASES
from .kalshi_client import KalshiAPIClient
from .kalshi_series_backfill import NBA_SERIES_TICKERS, crawl_kalshi_series_markets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_KALSHI_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

NBA_DISCOVERY_TERMS = [
    "nba",
    "basketball",
    "professional basketball",
    "national basketball association",
]
NBA_SERIES_PREFIX = "KXNBA"


def _json_safe_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _normalized_blob(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            parts.append(_json_safe_text(value))
        elif not pd.isna(value):
            parts.append(str(value))
    text = " ".join(parts).lower()
    return re.sub(r"[^a-z0-9]+", " ", text)


def _has_nba_signal(text: str) -> bool:
    if any(re.search(rf"\b{re.escape(term)}\b", text) for term in NBA_DISCOVERY_TERMS):
        return True
    for aliases in NBA_TEAM_ALIASES.values():
        for alias in aliases:
            normalized = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
            if normalized and re.search(rf"\b{re.escape(normalized)}\b", text):
                return True
    return False


def _kxnba_series_tickers(values: pd.Series | list[str]) -> list[str]:
    series = pd.Series(values, dtype="object").dropna().astype(str).str.upper().str.strip()
    return sorted(series[series.str.startswith(NBA_SERIES_PREFIX)].unique().tolist())


def _write_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_series_list_candidates(
    client: KalshiAPIClient | None = None,
    output_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fetch the Kalshi series list and keep NBA/basketball-looking series."""

    kalshi = client or KalshiAPIClient.from_env()
    all_series = kalshi.get_series_list({"include_product_metadata": True, "include_volume": True})
    if all_series.empty:
        candidates = pd.DataFrame()
    else:
        working = all_series.copy()
        working["series_discovery_text"] = working.apply(
            lambda row: _normalized_blob(
                row,
                [
                    "ticker",
                    "frequency",
                    "title",
                    "category",
                    "tags",
                    "settlement_sources",
                    "product_metadata",
                ],
            ),
            axis=1,
        )
        ticker_is_nba = working["ticker"].fillna("").astype(str).str.upper().str.startswith("KXNBA")
        text_is_nba = working["series_discovery_text"].map(_has_nba_signal)
        candidates = working.loc[ticker_is_nba | text_is_nba].copy()
        candidates["series_discovery_reason"] = ""
        candidates.loc[ticker_is_nba.loc[candidates.index], "series_discovery_reason"] += "ticker_prefix;"
        candidates.loc[text_is_nba.loc[candidates.index], "series_discovery_reason"] += "nba_text;"

    output = Path(output_path) if output_path else RAW_KALSHI_DIR / "series_list.csv"
    candidate_output = Path(candidates_path) if candidates_path else REPORTS_DIR / "kalshi_nba_series_candidates.csv"
    _write_csv(all_series, output)
    _write_csv(candidates, candidate_output)
    summary = {
        "series_rows": int(len(all_series)),
        "candidate_rows": int(len(candidates)),
        "candidate_tickers": sorted(candidates["ticker"].dropna().astype(str).str.upper().unique().tolist())
        if not candidates.empty and "ticker" in candidates.columns
        else [],
        "series_list_path": str(output),
        "candidates_path": str(candidate_output),
    }
    return all_series, candidates, summary


def fetch_nba_event_candidates(
    client: KalshiAPIClient | None = None,
    series_tickers: list[str] | None = None,
    max_pages: int = 50,
    output_path: str | Path | None = None,
    milestones_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fetch events and keep NBA/basketball-looking events."""

    kalshi = client or KalshiAPIClient.from_env()
    selected_series = sorted(set(series_tickers or []))
    event_frames: list[pd.DataFrame] = []
    milestone_frames: list[pd.DataFrame] = []
    routes: list[dict[str, Any]] = []

    if selected_series:
        for series_ticker in selected_series:
            events, milestones = kalshi.get_events_with_milestones(
                {"series_ticker": series_ticker, "limit": 200, "max_pages": max_pages}
            )
            if not events.empty:
                events["event_discovery_route"] = f"series:{series_ticker}"
                event_frames.append(events)
            if not milestones.empty:
                milestones["event_discovery_route"] = f"series:{series_ticker}"
                milestone_frames.append(milestones)
            routes.append(
                {
                    "route": f"series:{series_ticker}",
                    "events": int(len(events)),
                    "milestones": int(len(milestones)),
                    "error": str(events.attrs.get("error", "")),
                }
            )
    else:
        events, milestones = kalshi.get_events_with_milestones({"limit": 200, "max_pages": max_pages})
        if not events.empty:
            events["event_discovery_route"] = "all"
            event_frames.append(events)
        if not milestones.empty:
            milestones["event_discovery_route"] = "all"
            milestone_frames.append(milestones)
        routes.append(
            {
                "route": "all",
                "events": int(len(events)),
                "milestones": int(len(milestones)),
                "error": str(events.attrs.get("error", "")),
            }
        )

    events_all = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    milestones_all = pd.concat(milestone_frames, ignore_index=True, sort=False) if milestone_frames else pd.DataFrame()
    if not events_all.empty and "event_ticker" in events_all.columns:
        events_all = events_all.drop_duplicates(subset=["event_ticker"], keep="last")
    if not milestones_all.empty and "id" in milestones_all.columns:
        milestones_all = milestones_all.drop_duplicates(subset=["id"], keep="last")

    if events_all.empty:
        candidates = pd.DataFrame()
    else:
        working = events_all.copy()
        working["event_discovery_text"] = working.apply(
            lambda row: _normalized_blob(
                row,
                [
                    "event_ticker",
                    "series_ticker",
                    "sub_title",
                    "title",
                    "category",
                    "product_metadata",
                    "strike_period",
                ],
            ),
            axis=1,
        )
        ticker_is_nba = working["series_ticker"].fillna("").astype(str).str.upper().str.startswith("KXNBA")
        text_is_nba = working["event_discovery_text"].map(_has_nba_signal)
        candidates = working.loc[ticker_is_nba | text_is_nba].copy()
        candidates["event_discovery_reason"] = ""
        candidates.loc[ticker_is_nba.loc[candidates.index], "event_discovery_reason"] += "series_prefix;"
        candidates.loc[text_is_nba.loc[candidates.index], "event_discovery_reason"] += "nba_text;"

    output = Path(output_path) if output_path else RAW_KALSHI_DIR / "events_discovery.csv"
    milestone_output = Path(milestones_path) if milestones_path else RAW_KALSHI_DIR / "events_milestones.csv"
    candidate_output = Path(candidates_path) if candidates_path else REPORTS_DIR / "kalshi_nba_event_candidates.csv"
    _write_csv(events_all, output)
    _write_csv(milestones_all, milestone_output)
    _write_csv(candidates, candidate_output)
    summary = {
        "event_rows": int(len(events_all)),
        "milestone_rows": int(len(milestones_all)),
        "candidate_event_rows": int(len(candidates)),
        "candidate_series_tickers": sorted(candidates["series_ticker"].dropna().astype(str).str.upper().unique().tolist())
        if not candidates.empty and "series_ticker" in candidates.columns
        else [],
        "routes": routes,
        "events_path": str(output),
        "milestones_path": str(milestone_output),
        "candidates_path": str(candidate_output),
    }
    return events_all, milestones_all, candidates, summary


def discover_and_backfill_nba_series_from_series_and_events(
    client: KalshiAPIClient | None = None,
    event_max_pages: int = 50,
    market_max_pages: int = 100,
    include_all_events_scan: bool = False,
) -> dict[str, Any]:
    """Find NBA series candidates, crawl their historical markets, and summarize additions."""

    kalshi = client or KalshiAPIClient.from_env()
    _, series_candidates, series_summary = fetch_series_list_candidates(client=kalshi)
    series_from_list = (
        series_candidates["ticker"].dropna().astype(str).str.upper().tolist()
        if not series_candidates.empty and "ticker" in series_candidates.columns
        else []
    )
    # Team/city aliases make a useful review report, but are too broad for API crawling
    # because names like Indiana or Washington occur outside NBA contexts.
    series_from_list = _kxnba_series_tickers(series_from_list)
    seed_series = sorted(set(NBA_SERIES_TICKERS + series_from_list))
    _, _, event_candidates, event_summary = fetch_nba_event_candidates(
        client=kalshi,
        series_tickers=seed_series,
        max_pages=event_max_pages,
    )
    event_series = (
        event_candidates["series_ticker"].dropna().astype(str).str.upper().tolist()
        if not event_candidates.empty and "series_ticker" in event_candidates.columns
        else []
    )
    event_series = _kxnba_series_tickers(event_series)
    all_event_summary: dict[str, Any] | None = None
    if include_all_events_scan:
        _, _, all_event_candidates, all_event_summary = fetch_nba_event_candidates(
            client=kalshi,
            series_tickers=None,
            max_pages=event_max_pages,
            output_path=RAW_KALSHI_DIR / "events_discovery_all_scan.csv",
            milestones_path=RAW_KALSHI_DIR / "events_milestones_all_scan.csv",
            candidates_path=REPORTS_DIR / "kalshi_nba_event_candidates_all_scan.csv",
        )
        if not all_event_candidates.empty and "series_ticker" in all_event_candidates.columns:
            event_series.extend(_kxnba_series_tickers(all_event_candidates["series_ticker"].tolist()))

    series_to_crawl = sorted(set(seed_series + event_series))
    cached, possible, backfill_summary = crawl_kalshi_series_markets(
        series_tickers=series_to_crawl,
        client=kalshi,
        max_pages=market_max_pages,
    )
    summary = {
        "series_list": series_summary,
        "events_by_candidate_series": event_summary,
        "events_all_scan": all_event_summary,
        "series_to_crawl": series_to_crawl,
        "historical_backfill": backfill_summary,
        "cached_markets": int(len(cached)),
        "possible_game_winner_markets": int(len(possible)),
        "note": "Series and event discovery can reveal unknown NBA series; historical markets still provide the actual archived contract rows.",
    }
    summary_path = REPORTS_DIR / "kalshi_series_event_discovery_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
