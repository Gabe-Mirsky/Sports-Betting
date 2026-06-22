"""Audit recommendation grading artifacts for correctness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.recommendation_grading_audit import build_recommendation_grading_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit recommendation grading outputs.")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--recommendations-path", default=None)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--graded-single-path", default=None)
    parser.add_argument("--snapshot-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"
    recommendations_path = (
        Path(args.recommendations_path) if args.recommendations_path else reports_dir / "fair_price_signals.csv"
    )
    results_path = Path(args.results_path) if args.results_path else reports_dir / "backtest_trades.csv"
    graded_single_path = (
        Path(args.graded_single_path) if args.graded_single_path else reports_dir / "graded_single_recommendations.csv"
    )
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else reports_dir / "recommendation_snapshots"
    summary = build_recommendation_grading_audit(
        recommendations_path,
        results_path,
        graded_single_path,
        snapshot_dir,
        reports_dir,
    )
    print(f"Recommendations: {summary['match_quality']['recommendations']:,}")
    print(f"Graded: {summary['match_quality']['graded']:,}")
    print(f"Unmatched: {summary['match_quality']['unmatched']:,}")
    print(f"Profit math failures: {summary['payout_profit_math']['profit_math_failures']:,}")
    print(f"CLV math failures: {summary['clv_math']['clv_math_failures']:,}")
    print(f"Missing CLV rows: {summary['clv_math']['missing_clv_rows']:,}")
    print(f"Trusted: {summary['trusted']}")
    print(f"Saved audit summary to: {reports_dir / 'recommendation_grading_audit_summary.json'}")


if __name__ == "__main__":
    main()
