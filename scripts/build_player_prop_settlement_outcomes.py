"""Build the player-prop settlement outcomes report (research-only).

Reads the enriched prop snapshots and the likely-main-line audit and writes:
  - data/reports/player_prop_settlement_outcomes_summary.json
  - data/reports/player_prop_settlement_outcomes.csv
  - data/reports/player_prop_settlement_outcomes.md

If nothing has settled yet, the report says so honestly (which game is
pending and why) instead of forcing settlement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.player_prop_settlement_outcomes import write_settlement_outcome_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-prop settlement outcome reports.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = write_settlement_outcome_reports(PROJECT_ROOT)

    print(f"Settlement outcomes: {summary['settled_props']} settled, {summary['pending_props']} pending")
    overall = summary["overall"]
    if summary["settled_props"]:
        print(
            f"  Overall: over {overall['over_won']} ({overall['over_win_rate']:.1%}) / "
            f"under {overall['under_won']} ({overall['under_win_rate']:.1%}) / push {overall['push']}"
        )
    for game in summary["pending_games"]:
        print(
            f"  Pending: {game['canonical_game_key']} ({game['pending_props']} props) — {game['reason']}"
        )
    for warning in summary["warnings"]:
        print(f"  WARNING: {warning}")
    for key, path in summary["outputs"].items():
        print(f"Wrote: {path}")
    print("Research-only: settlement history only; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
