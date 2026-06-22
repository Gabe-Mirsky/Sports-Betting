"""Refresh NBA results and settle pending player-prop snapshots.

Re-imports the nba_api current games + player game logs from the local caches
(or re-downloads them first with --download), reruns the NBA prop snapshot
enrichment, and reports how settlement moved (pending before/after, newly
settled, still pending).

Updates:
    data/processed/player_prop_snapshots_enriched.csv
    data/reports/player_prop_enrichment_summary.json
    data/reports/player_prop_unmatched_players.csv
    data/reports/player_prop_unmatched_games.csv
    data/reports/player_prop_settlement_refresh_summary.json
    data/reports/player_prop_newly_settled.csv

Research-only: no recommendations, no predictions, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.nba_current_actuals import DEFAULT_MIN_SEASON  # noqa: E402
from data.prop_settlement_refresh import run_results_refresh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh NBA actuals and settle pending player-prop snapshots."
    )
    parser.add_argument("--download", action="store_true",
                        help="Re-download nba_api caches before refreshing (default: cache-only).")
    parser.add_argument("--min-season", type=int, default=DEFAULT_MIN_SEASON,
                        help="Earliest season start year to include (default %(default)s).")
    parser.add_argument("--max-season", type=int, default=None,
                        help="Latest season start year to refresh when --download is set.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = run_results_refresh(
        PROJECT_ROOT,
        download=args.download,
        min_season=args.min_season,
        max_season=args.max_season,
    )

    settlement = summary["settlement"]
    enrichment = summary["enrichment"]
    print(f"Settlement refresh {summary['refresh_version']} @ {summary['generated_at_utc']}")
    print(f"  Mode:                 {summary['mode']}")
    print(f"  Actuals import:       {summary['actuals_import']['status']}")
    print(f"  NBA snapshots:        {enrichment['nba_snapshots']}")
    print(f"  Player match rate:    {enrichment['player_match_rate']:.1%}")
    print(f"  Game match rate:      {enrichment['game_match_rate']:.1%}")
    print(f"  Pending before:       {settlement['pending_before_refresh']}")
    print(f"  Pending after:        {settlement['pending_after_refresh']}")
    print(f"  Newly settled:        {settlement['newly_settled']}")
    print(f"  Still pending:        {settlement['still_pending']}")
    print(f"  Settled total:        {settlement['settled_total']}")
    if settlement["settled_by_prop_type"]:
        print("  Settled by prop type:")
        for prop_type, count in sorted(settlement["settled_by_prop_type"].items()):
            print(f"    {prop_type}: {count}")
    if settlement["unsettled_games"]:
        print(f"  Unsettled games:      {len(settlement['unsettled_games'])}")
    print(f"  Refresh summary: {summary['outputs']['refresh_summary_path']}")
    print("Research-only: no recommendations were produced; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
