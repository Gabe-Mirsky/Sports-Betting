"""Review Kalshi unmatched-market gaps and audit matched market quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.market_review import write_market_review_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review unmatched Kalshi markets and audit matched markets.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, audit = write_market_review_outputs(
        PROJECT_ROOT,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )
    gap = payload["gap_decision"]
    audit_summary = payload["audit_summary"]

    print("Gap decision:")
    print(f"- Play-in games: {gap['decision']['play_in_games']}")
    print(f"- Preseason markets: {gap['decision']['preseason_markets']}")
    print(f"- Total gap games: {gap['total_gap_games']:,}")
    print("Gap games by reason:")
    for reason, count in gap["gap_games_by_reason"].items():
        print(f"- {reason}: {count:,}")
    print("Audit sample:")
    print(f"- Rows audited: {len(audit):,}")
    print(f"- Ticker failures: {audit_summary['ticker_failures']:,}")
    print(f"- Invalid YES team rows: {audit_summary['invalid_yes_team_rows']:,}")
    print(f"- Title alias warnings: {audit_summary['title_alias_warnings']:,}")
    print(f"Saved summary to: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_market_review_summary.json'}")
    print(f"Saved audit sample to: {PROJECT_ROOT / 'data' / 'reports' / 'kalshi_match_audit_sample_50.csv'}")


if __name__ == "__main__":
    main()
