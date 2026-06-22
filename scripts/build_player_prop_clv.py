"""Build the player-prop CLV reports (research-only).

Compares early vs closing-like snapshots per prop market and writes:
  - data/reports/player_prop_clv_summary.json
  - data/reports/player_prop_clv.csv
  - data/reports/player_prop_clv_by_bookmaker.csv
  - data/reports/player_prop_clv_by_prop_type.csv
  - data/reports/player_prop_clv.md

Settlement is not required. If no early/closing pair exists yet, the reports
are still written and explain what is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.player_prop_clv import write_clv_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-prop CLV reports.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = write_clv_reports(PROJECT_ROOT)

    print(f"Prop CLV: {'READY' if summary['clv_ready'] else 'NOT READY'}")
    print(f"  {summary['verdict']}")
    print(f"  Markets with CLV: {summary['markets_with_clv']} (NBA: {summary['nba_markets_with_clv']})")
    print(f"  NBA closing-like snapshots: {summary['nba_closing_like_snapshots']}")
    for warning in summary["warnings"]:
        print(f"  WARNING: {warning}")
    for key, path in summary["outputs"].items():
        print(f"Wrote: {path}")
    print("Research-only: CLV measurement only; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
