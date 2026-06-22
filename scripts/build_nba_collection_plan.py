"""Build the NBA near-tip-off prop collection plan + CLV readiness reports.

Reads the collected prop snapshots, plans the pre-tip collection windows
(24h / 6h / 2h / 60m / 30m / 10m before tip) for every known NBA game, and
writes:
  - data/reports/nba_prop_closing_collection_plan.json
  - data/reports/nba_prop_closing_collection_plan.md
  - data/reports/nba_prop_closing_coverage.csv
  - data/reports/nba_prop_clv_readiness_summary.json

Research-only: planning only; no models, recommendations, or betting changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.nba_collection_planner import write_collection_plan_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the NBA near-tip prop collection plan.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    plan = write_collection_plan_reports(PROJECT_ROOT)

    print(f"NBA collection plan: {plan['games_total']} game(s), {plan['games_upcoming']} upcoming")
    for game in plan["games"]:
        print(
            f"  {game['game']}: {game['minutes_until_game']:.0f} min to tip "
            f"({game['timing_classification']}), hit={game['windows_hit']}, "
            f"missed={game['windows_missed']}, collect_now={game['collection_needed_now']}"
        )
    print(f"Collection needed now: {plan['collection_needed_now']}")
    print(f"Next recommended collection (UTC): {plan['next_recommended_collection_time_utc'] or 'n/a'}")
    for warning in plan["warnings"]:
        print(f"  WARNING: {warning}")
    for key, path in plan["outputs"].items():
        print(f"Wrote: {path}")
    print("Research-only: planning only; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
