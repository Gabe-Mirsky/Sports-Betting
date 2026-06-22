"""Build the single-game market truth audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.coverage import build_market_truth_audit, save_market_truth_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit matched Kalshi game markets and pregame prices.")
    parser.add_argument("--max-spread-cents", type=float, default=10.0)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit, summary = build_market_truth_audit(
        PROJECT_ROOT,
        max_spread_cents=args.max_spread_cents,
        min_volume=args.min_volume,
    )
    save_market_truth_audit(
        audit,
        summary,
        audit_path=args.audit_path,
        summary_path=args.summary_path,
    )

    usable = summary["usable_price_counts"]
    print(f"Matched game-markets: {summary['matched_game_markets']:,}")
    print(f"Auto-matched: {summary['auto_matched']:,}")
    print(f"Needs review: {summary['needs_review']:,}")
    print(f"Usable 60m prices: {usable['pregame_60m']:,}")
    print(f"Usable 30m prices: {usable['pregame_30m']:,}")
    print(f"Usable 5m prices: {usable['pregame_5m']:,}")
    print(f"Ticker/team mapping mismatches: {summary['ticker_mapping_mismatch_count']:,}")
    print(f"Wide spreads: {summary['wide_spread_count']:,}")
    print(f"Low-liquidity rows: {summary['low_liquidity_count']:,}")
    print(f"Saved audit to: {args.audit_path or PROJECT_ROOT / 'data' / 'reports' / 'market_truth_audit.csv'}")
    print(
        "Saved summary to: "
        f"{args.summary_path or PROJECT_ROOT / 'data' / 'reports' / 'market_truth_audit_summary.json'}"
    )


if __name__ == "__main__":
    main()
