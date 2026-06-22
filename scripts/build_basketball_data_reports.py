"""Build the basketball join-readiness and player-prop-path reports.

Reads the normalized ehallmar + zachht outputs from data/processed and writes:
    data/reports/basketball_data_join_readiness.json
    data/reports/basketball_data_join_readiness.md
    data/reports/player_prop_data_path.md

Reporting only - no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.basketball_join_readiness import (  # noqa: E402
    build_join_readiness_reports,
    build_player_prop_path_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build basketball data join-readiness + prop-path reports.")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir) if args.processed_dir else PROJECT_ROOT / "data" / "processed"
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"

    readiness = build_join_readiness_reports(processed_dir, reports_dir)
    build_player_prop_path_report(readiness, reports_dir)

    src = readiness["sources"]
    print("Basketball join readiness:")
    print(f"  ehallmar NBA games: {src['ehallmar']['nba_games']:,} | player logs: {src['ehallmar']['player_game_logs']:,}")
    print(f"  zachht snapshots: {src['zachht']['snapshot_rows']:,} | unique games: {src['zachht']['unique_games']:,}")
    print(f"  player-logs->games id join: {readiness['id_joins']['within_ehallmar_player_logs_to_games']['coverage_pct']}%")
    print(f"  cross-source date+team overlap: {readiness['date_team_join']['overlapping_game_keys']:,} game keys")
    print(f"  player props available: {readiness['player_props_available']}")
    print(f"  wrote: {reports_dir / 'basketball_data_join_readiness.md'}, {reports_dir / 'player_prop_data_path.md'}")


if __name__ == "__main__":
    main()
