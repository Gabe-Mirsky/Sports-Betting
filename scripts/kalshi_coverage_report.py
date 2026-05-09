"""Summarize Kalshi market and pregame price coverage."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.coverage import (  # noqa: E402
    build_kalshi_coverage_report,
    build_kalshi_gap_report,
    save_kalshi_coverage_report,
)


def main() -> None:
    summary, monthly = build_kalshi_coverage_report(PROJECT_ROOT)
    gap_report = build_kalshi_gap_report(PROJECT_ROOT)
    save_kalshi_coverage_report(summary, monthly, gap_report=gap_report)

    print(f"Games in prediction universe: {summary['games_in_prediction_universe']:,}")
    print(f"Kalshi market rows: {summary['kalshi_market_rows']:,}")
    print(f"Games with Kalshi markets: {summary['games_with_kalshi_markets']:,}")
    print(f"Auto-matched games: {summary['auto_matched_games']:,}")
    print(f"Games with usable pregame prices: {summary['games_with_usable_pregame_price']:,}")
    print(f"Market date range: {summary['market_date_min']} to {summary['market_date_max']}")
    print(f"Saved summary to: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_coverage_summary.json'}")
    print(f"Saved monthly coverage to: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_coverage_by_month.csv'}")
    print(f"Saved unmatched market gap report to: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_unmatched_market_gap_report.csv'}")


if __name__ == "__main__":
    main()
