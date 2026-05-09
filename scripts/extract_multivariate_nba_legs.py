"""Extract NBA legs embedded inside broad multivariate Kalshi markets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.multivariate_legs import extract_multivariate_nba_legs_from_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract NBA legs from cached multivariate Kalshi markets.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path) if args.input_path else PROJECT_ROOT / "data" / "raw" / "kalshi" / "broad_nba_markets.csv"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_multivariate_nba_legs.csv"
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else PROJECT_ROOT / "data" / "reports" / "kalshi_multivariate_nba_legs_summary.json"
    )
    legs, summary = extract_multivariate_nba_legs_from_file(input_path, output_path, summary_path)

    print(f"NBA multivariate leg rows: {len(legs):,}")
    print(f"Unique NBA legs: {summary.get('unique_legs', 0):,}")
    print(f"Spread/total leg rows: {summary.get('spread_total_leg_rows', 0):,}")
    print(f"Player-prop leg rows: {summary.get('player_prop_leg_rows', 0):,}")
    print(f"Directly backtestable rows: {summary.get('directly_backtestable_rows', 0):,}")
    print(f"Saved legs to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
