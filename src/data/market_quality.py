"""Market data quality reporting for local Kalshi-style CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RESOLVED_SETTLEMENTS = {"yes", "y", "1", "true", "win", "won", "no", "n", "0", "false", "loss", "lost"}


def _row_labels(markets: pd.DataFrame, mask: pd.Series) -> list[str]:
    if "market_ticker" in markets.columns:
        return markets.loc[mask, "market_ticker"].astype(str).tolist()
    return [str(index) for index in markets.index[mask]]


def analyze_market_data_quality(
    markets_df: pd.DataFrame,
    matched_df: pd.DataFrame | None = None,
    min_sample_size: int = 30,
) -> dict[str, Any]:
    """Build a warning-oriented market data quality report."""

    markets = markets_df.copy()
    report: dict[str, Any] = {
        "rows": int(len(markets)),
        "matched_rows": int(len(matched_df)) if matched_df is not None else None,
        "warnings": [],
        "price_source_counts": {},
        "price_quality": {},
        "settlement_quality": {},
        "volume_quality": {},
        "spread_quality": {},
    }

    if markets.empty:
        report["warnings"].append("Market file is empty.")
        return report

    if len(markets) < min_sample_size:
        report["warnings"].append(
            f"Small sample size: {len(markets)} markets. Results may be noisy."
        )

    price_source = (
        markets["price_source"].fillna("missing").astype(str)
        if "price_source" in markets.columns
        else pd.Series("unknown", index=markets.index)
    )
    report["price_source_counts"] = {
        str(source): int(count)
        for source, count in price_source.value_counts(dropna=False).sort_index().items()
    }

    close_price_mask = price_source.eq("close_price")
    if close_price_mask.any():
        report["warnings"].append(
            f"{int(close_price_mask.sum())} markets use close_price_cents as the trading price. "
            "That can make historical backtests look too good if the price was not known before the game."
        )

    missing_price_mask = price_source.eq("missing") | pd.to_numeric(
        markets.get("yes_mid_cents", pd.Series(pd.NA, index=markets.index)),
        errors="coerce",
    ).isna()
    report["price_quality"] = {
        "missing_price_count": int(missing_price_mask.sum()),
        "close_price_only_count": int(close_price_mask.sum()),
        "missing_price_rows": _row_labels(markets, missing_price_mask),
        "close_price_rows": _row_labels(markets, close_price_mask),
    }
    if missing_price_mask.any():
        report["warnings"].append(f"{int(missing_price_mask.sum())} markets are missing usable prices.")

    bid = pd.to_numeric(markets.get("yes_bid_cents", pd.Series(pd.NA, index=markets.index)), errors="coerce")
    ask = pd.to_numeric(markets.get("yes_ask_cents", pd.Series(pd.NA, index=markets.index)), errors="coerce")
    has_spread = bid.notna() & ask.notna()
    missing_spread = ~has_spread
    spread = ask - bid
    report["spread_quality"] = {
        "rows_with_bid_ask": int(has_spread.sum()),
        "rows_missing_bid_or_ask": int(missing_spread.sum()),
        "average_spread_cents": float(spread[has_spread].mean()) if has_spread.any() else None,
        "median_spread_cents": float(spread[has_spread].median()) if has_spread.any() else None,
        "max_spread_cents": float(spread[has_spread].max()) if has_spread.any() else None,
    }
    if missing_spread.any():
        report["warnings"].append(
            f"{int(missing_spread.sum())} markets are missing bid/ask spread data."
        )

    settlement = markets.get("settlement", pd.Series("", index=markets.index)).astype(str).str.strip().str.lower()
    unresolved = settlement.eq("") | settlement.isin({"nan", "none"})
    resolved = settlement.isin(RESOLVED_SETTLEMENTS)
    report["settlement_quality"] = {
        "resolved_count": int(resolved.sum()),
        "unresolved_count": int(unresolved.sum()),
        "unrecognized_count": int((~resolved & ~unresolved).sum()),
        "unresolved_rows": _row_labels(markets, unresolved),
    }
    if unresolved.any():
        report["warnings"].append(
            f"{int(unresolved.sum())} markets have unresolved settlement values."
        )

    volume = pd.to_numeric(markets.get("volume", pd.Series(pd.NA, index=markets.index)), errors="coerce")
    missing_volume = volume.isna()
    zero_volume = volume.fillna(0).eq(0)
    report["volume_quality"] = {
        "missing_volume_count": int(missing_volume.sum()),
        "zero_volume_count": int(zero_volume.sum()),
        "average_volume": float(volume.dropna().mean()) if volume.notna().any() else None,
    }
    if missing_volume.any():
        report["warnings"].append(f"{int(missing_volume.sum())} markets are missing volume values.")

    if matched_df is not None:
        unmatched = len(markets) - len(matched_df)
        report["unmatched_rows"] = int(unmatched)
        if unmatched:
            report["warnings"].append(f"{unmatched} markets did not match prediction rows.")

    return report


def save_market_quality_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save a market quality report as JSON."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
