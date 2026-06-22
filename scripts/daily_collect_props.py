"""Daily multi-sport player-prop odds collector (research-only).

Loads config/prop_collection.yaml, collects player-prop odds for every enabled
league from the enabled sources, saves raw responses, and appends normalized
snapshots to data/processed/player_prop_snapshots_normalized.csv.

No models, no recommendations, no betting: approved bets and parlays stay blocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.prop_collection import load_prop_collection_config, run_prop_collection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect multi-sport player-prop odds snapshots.")
    parser.add_argument("--config", default=None, help="Path to prop_collection.yaml")
    parser.add_argument("--max-events", type=int, default=None, help="Override per-league event cap")
    parser.add_argument("--horizon-hours", type=float, default=None, help="Override event horizon")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config" / "prop_collection.yaml"
    config = load_prop_collection_config(config_path)
    if args.max_events is not None:
        config["defaults"]["max_events_per_league_per_run"] = args.max_events
    if args.horizon_hours is not None:
        config["defaults"]["event_horizon_hours"] = args.horizon_hours

    summary = run_prop_collection(config, PROJECT_ROOT)

    totals = summary["totals"]
    print(f"Run {summary['run_id']}: status={summary['status']}")
    for record in summary["leagues"]:
        source = record.get("source") or "-"
        extra = f" ({record.get('snapshots_collected', 0)} snapshots)" if record["status"] == "collected" else ""
        print(f"  {record['league']:<6} {source:<10} {record['status']}{extra}")
    print(
        f"Snapshots: +{totals['snapshots_added']} added, {totals['duplicates_removed']} duplicates removed, "
        f"{totals['snapshots_total']} total ({totals['closing_snapshots_total']} closing-like)"
    )
    print(f"Normalized CSV: {summary['outputs']['processed_path']}")
    print(f"Run log: {summary['outputs']['run_log']}")
    print("Research-only: no recommendations were produced; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
