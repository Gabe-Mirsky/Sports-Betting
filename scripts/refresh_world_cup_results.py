"""Refresh World Cup results and settle easy markets (research-only).

Fetches recent FIFA World Cup scores from The Odds API (guarded by the World Cup
quota floor) and grades the simple game/team markets we collected (1X2, totals,
BTTS) as won/lost/push. Player-prop settlement is NOT attempted.

Default is --dry-run (no /scores call, no credits). Pass --real to spend credits.
Enables no models, recommendations, predictions, approved bets, or parlays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.world_cup_collection import load_world_cup_config  # noqa: E402
from data.world_cup_results import run_world_cup_results_refresh  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh World Cup results + settle easy markets.")
    p.add_argument("--real", action="store_true", help="Make the real /scores call (spends credits).")
    p.add_argument("--days-from", type=int, default=1, help="Score lookback days (Odds API).")
    p.add_argument("--results-min-remaining", type=float, default=None,
                   help="One-time override of the credit floor for this run only.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "config" / "world_cup_collection.yaml"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_world_cup_config(args.config)
    if args.results_min_remaining is not None:
        config.setdefault("quota", {})["results_min_remaining"] = float(args.results_min_remaining)

    summary = run_world_cup_results_refresh(
        config, PROJECT_ROOT, dry_run=not args.real, days_from=args.days_from,
    )
    print(f"Results refresh: status={summary['status']} | events_with_scores={summary['events_with_scores']} | "
          f"completed={summary['completed_events']} | settled={summary['settled_rows']} "
          f"(won={summary['won']} lost={summary['lost']} push={summary['push']}) | "
          f"by_market={summary['by_market_type']}")
    if summary.get("blockers"):
        print(f"  blockers: {summary['blockers']}")
    print("Research-only: settlement labels only; no bets, parlays, predictions, or gate changes.")


if __name__ == "__main__":
    main()
