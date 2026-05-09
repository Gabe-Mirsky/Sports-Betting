"""Fetch direct market rows for NBA legs discovered inside combo markets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.kalshi_client import KalshiAPIClient


LINE_AND_PROP_CATEGORIES = {
    "spread_handicap",
    "total_points_over_under",
    "player_points_rebounds_assists",
}

CATEGORY_PRIORITY = {
    "spread_handicap": 0,
    "total_points_over_under": 1,
    "player_points_rebounds_assists": 2,
    "game_winner": 3,
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_cache(existing: pd.DataFrame, fetched: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame for frame in [existing, fetched] if not frame.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if "market_ticker" in combined.columns:
        combined["market_ticker"] = combined["market_ticker"].astype(str)
        combined = combined.drop_duplicates(subset=["market_ticker"], keep="last")
    combined.to_csv(output_path, index=False)
    return combined.reset_index(drop=True)


def _candidate_tickers(legs: pd.DataFrame, include_game_winners: bool, max_tickers: int | None) -> list[str]:
    if legs.empty or "leg_market_ticker" not in legs.columns:
        return []
    candidates = legs.copy()
    if not include_game_winners and "leg_category" in candidates.columns:
        candidates = candidates[candidates["leg_category"].isin(LINE_AND_PROP_CATEGORIES)].copy()
    if "leg_category" in candidates.columns:
        candidates["_category_priority"] = candidates["leg_category"].map(CATEGORY_PRIORITY).fillna(99)
        candidates = candidates.sort_values(["_category_priority", "leg_market_ticker"])
    tickers = candidates["leg_market_ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if max_tickers is not None:
        tickers = tickers[: int(max_tickers)]
    return tickers


def fetch_underlying_leg_markets(
    legs: pd.DataFrame,
    client: KalshiAPIClient | None = None,
    existing_markets: pd.DataFrame | None = None,
    include_game_winners: bool = False,
    max_tickers: int | None = None,
    max_consecutive_failures: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fetch direct market rows for unique selected-leg tickers."""

    kalshi = client or KalshiAPIClient.from_env()
    existing = existing_markets.copy() if existing_markets is not None else pd.DataFrame()
    existing_tickers = set(existing.get("market_ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    tickers = [ticker for ticker in _candidate_tickers(legs, include_game_winners, max_tickers) if ticker not in existing_tickers]

    fetched_frames: list[pd.DataFrame] = []
    request_rows: list[dict[str, Any]] = []
    consecutive_failures = 0
    for ticker in tickers:
        route = "recent"
        market = kalshi.get_market(ticker)
        if market.empty:
            route = "historical"
            market = kalshi.get_historical_market(ticker)
        ok = not market.empty
        if ok:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        request_rows.append(
            {
                "market_ticker": ticker,
                "route": route,
                "status": "fetched" if ok else "missing",
                "error": str(market.attrs.get("error", "")) if market.empty else "",
            }
        )
        if ok:
            market = market.copy()
            market["direct_fetch_route"] = route
            fetched_frames.append(market)
        if max_consecutive_failures and consecutive_failures >= int(max_consecutive_failures):
            break

    fetched = pd.concat(fetched_frames, ignore_index=True, sort=False) if fetched_frames else pd.DataFrame()
    requests = pd.DataFrame(request_rows)
    summary = {
        "candidate_tickers": len(tickers),
        "fetched_rows": int(len(fetched)),
        "fetched_unique_markets": int(fetched["market_ticker"].nunique()) if "market_ticker" in fetched.columns else 0,
        "missing_tickers": int(requests["status"].eq("missing").sum()) if not requests.empty else 0,
        "attempted_tickers": int(len(requests)),
        "stopped_after_consecutive_failures": bool(
            max_consecutive_failures and consecutive_failures >= int(max_consecutive_failures)
        ),
        "max_consecutive_failures": int(max_consecutive_failures),
        "include_game_winners": bool(include_game_winners),
        "max_tickers": max_tickers,
        "note": "These are direct underlying market rows; only these can eventually support single-leg spread/total/prop backtests.",
    }
    return fetched.reset_index(drop=True), requests, summary


def fetch_underlying_leg_markets_from_files(
    legs_path: str | Path,
    output_path: str | Path,
    requests_path: str | Path,
    summary_path: str | Path,
    client: KalshiAPIClient | None = None,
    include_game_winners: bool = False,
    max_tickers: int | None = None,
    max_consecutive_failures: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    legs = _read_csv(Path(legs_path))
    output = Path(output_path)
    existing = _read_csv(output)
    fetched, requests, summary = fetch_underlying_leg_markets(
        legs,
        client=client,
        existing_markets=existing,
        include_game_winners=include_game_winners,
        max_tickers=max_tickers,
        max_consecutive_failures=max_consecutive_failures,
    )
    combined = _write_cache(existing, fetched, output)
    summary = {
        **summary,
        "cached_rows": int(len(combined)),
        "cached_unique_markets": int(combined["market_ticker"].nunique()) if "market_ticker" in combined.columns else 0,
    }
    requests_output = Path(requests_path)
    summary_output = Path(summary_path)
    requests_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    requests.to_csv(requests_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return combined, requests, summary
