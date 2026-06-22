"""Build the player-prop manual review reports (research-only data inspection).

Reads:
  - data/processed/player_prop_snapshots_normalized.csv
  - data/processed/player_prop_snapshots_enriched.csv (if it exists)
  - config/prop_collection.yaml

Writes (under data/reports/):
  - player_prop_manual_review_summary.json
  - player_prop_manual_review.md
  - player_prop_counts_by_league_prop.csv
  - player_prop_counts_by_bookmaker.csv
  - player_prop_missing_fields.csv
  - nba_player_prop_review.csv
  - latest_collected_props.csv

No models, no recommendations, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.player_prop_manual_review import write_manual_review_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-prop manual review reports.")
    parser.add_argument("--normalized", default=None, help="Path to player_prop_snapshots_normalized.csv")
    parser.add_argument("--enriched", default=None, help="Path to player_prop_snapshots_enriched.csv")
    parser.add_argument("--config", default=None, help="Path to prop_collection.yaml")
    parser.add_argument("--reports-dir", default=None, help="Output directory (default data/reports)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = write_manual_review_reports(
        PROJECT_ROOT,
        normalized_path=args.normalized,
        enriched_path=args.enriched,
        config_path=args.config,
        reports_dir=args.reports_dir,
    )

    print(f"Total snapshots reviewed: {summary['total_snapshots']}")
    print(f"Latest snapshot (UTC): {summary['latest_snapshot_time_utc'] or 'n/a'}")
    print(f"Closing-like snapshots: {summary['closing_like_snapshots']}")
    print("Snapshots by league:")
    for league, count in summary["snapshots_by_league"].items():
        print(f"  - {league}: {count}")
    print(f"NBA players collected: {len(summary['nba_players'])}")
    match_rate = summary["nba_player_match_rate"]
    print(f"NBA player match rate: {f'{match_rate:.1%}' if match_rate is not None else 'n/a'}")
    print(f"Leagues configured but not collected: {summary['configured_leagues_not_collected'] or 'none'}")
    if summary["configured_prop_types_not_collected"]:
        print("Configured prop types not collected:")
        for league, props in sorted(summary["configured_prop_types_not_collected"].items()):
            print(f"  - {league}: {', '.join(props)}")
    print(f"NBA usable for future grading: {summary['nba_usable_for_grading']}")
    print(f"Verdict: {summary['nba_usability_verdict']}")
    for key, path in summary["outputs"].items():
        print(f"Wrote: {path}")
    print("Research-only: manual data inspection; no recommendations or betting changes.")


if __name__ == "__main__":
    main()
