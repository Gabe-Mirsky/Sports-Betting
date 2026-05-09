"""Audit real Kalshi line extraction before spread/total backtests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


LINE_MARKET_TYPES = [
    "spread_handicap",
    "total_points_over_under",
    "team_total",
    "player_points_rebounds_assists",
]
DEFERRED_MARKET_TYPES = {
    "player_points_rebounds_assists",
}


def build_market_line_coverage(taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize whether non-winner market rows have extracted lines."""

    if taxonomy.empty:
        return pd.DataFrame(), {
            "market_rows": 0,
            "line_market_rows": 0,
            "ready_market_types": [],
            "blocked_market_types": LINE_MARKET_TYPES,
            "note": "No market taxonomy rows are available yet.",
        }
    frame = taxonomy.copy()
    if "market_category" not in frame.columns:
        raise ValueError("Market line audit requires market_category.")
    for column in ["line_value", "direction", "taxonomy_confidence"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["line_value"] = pd.to_numeric(frame["line_value"], errors="coerce")
    frame["direction"] = frame["direction"].fillna("").astype(str)
    frame["taxonomy_confidence"] = pd.to_numeric(frame["taxonomy_confidence"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for market_type in LINE_MARKET_TYPES:
        subset = frame[frame["market_category"].astype(str).eq(market_type)].copy()
        rows_count = int(len(subset))
        line_count = int(subset["line_value"].notna().sum())
        direction_count = int(subset["direction"].str.strip().ne("").sum())
        high_confidence_count = int((subset["taxonomy_confidence"] >= 0.70).sum())
        line_coverage = float(line_count / rows_count) if rows_count else 0.0
        high_confidence_coverage = float(high_confidence_count / rows_count) if rows_count else 0.0
        line_extraction_ready = bool(rows_count >= 50 and line_coverage >= 0.90 and high_confidence_coverage >= 0.80)
        ready = bool(line_extraction_ready and market_type not in DEFERRED_MARKET_TYPES)
        if ready:
            blocked_reason = ""
        elif market_type in DEFERRED_MARKET_TYPES and rows_count:
            blocked_reason = "player_props_deferred_until_spread_total_models_are_ready"
        else:
            blocked_reason = "needs_more_real_kalshi_lines"
        rows.append(
            {
                "market_type": market_type,
                "rows": rows_count,
                "rows_with_line": line_count,
                "rows_with_direction": direction_count,
                "high_confidence_rows": high_confidence_count,
                "line_coverage_pct": line_coverage,
                "high_confidence_coverage_pct": high_confidence_coverage,
                "line_extraction_ready": line_extraction_ready,
                "ready_for_market_specific_backtest": ready,
                "blocked_reason": blocked_reason,
            }
        )

    coverage = pd.DataFrame(rows)
    ready_types = coverage.loc[coverage["ready_for_market_specific_backtest"], "market_type"].tolist()
    summary = {
        "market_rows": int(len(frame)),
        "line_market_rows": int(coverage["rows"].sum()) if not coverage.empty else 0,
        "ready_market_types": ready_types,
        "blocked_market_types": [
            market_type for market_type in LINE_MARKET_TYPES if market_type not in set(ready_types)
        ],
        "spread_ready": "spread_handicap" in ready_types,
        "total_ready": "total_points_over_under" in ready_types,
        "note": (
            "Spread and total models remain separate projects. "
            "Do not backtest those markets until real Kalshi line extraction passes this audit. "
            "Player props stay deferred until spread and total models are stable."
        ),
    }
    return coverage, summary


def save_market_line_coverage(
    coverage: pd.DataFrame,
    summary: dict[str, Any],
    coverage_path: str | Path,
    summary_path: str | Path,
) -> None:
    coverage_output = Path(coverage_path)
    summary_output = Path(summary_path)
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
