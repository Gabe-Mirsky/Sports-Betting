"""Import current/recent NBA actuals from the cached nba_api downloads.

Writes:
    data/processed/nba_current_games_normalized.csv
    data/processed/nba_current_player_game_logs_normalized.csv
    data/reports/nba_current_actuals_import_summary.json

Optionally refreshes the raw caches first with --download (uses nba_api over
the network; default is cache-only). Actuals ingestion only - no model logic,
no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.nba_current_actuals import DEFAULT_MIN_SEASON, import_nba_current_actuals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import current NBA actuals from nba_api caches.")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--min-season", type=int, default=DEFAULT_MIN_SEASON,
                        help="Earliest season start year to include (default %(default)s).")
    parser.add_argument("--max-season", type=int, default=None,
                        help="Latest season start year to refresh when --download is set.")
    parser.add_argument("--download", action="store_true",
                        help="Refresh raw caches from NBA.com via nba_api before normalizing.")
    return parser.parse_args()


def _refresh_caches(args: argparse.Namespace) -> None:
    from data.nba_client import download_seasons
    from data.player_client import download_player_seasons

    end_season = args.max_season if args.max_season is not None else args.min_season + 1
    for season_type in ("Regular Season", "Playoffs"):
        download_seasons(
            args.min_season, end_season,
            cache_dir=Path(args.raw_dir) / "nba",
            season_type=season_type, force=True,
        )
        download_player_seasons(
            args.min_season, end_season,
            cache_dir=Path(args.raw_dir) / "nba" / "player",
            season_type=season_type, force=True,
        )


def main() -> None:
    args = parse_args()
    if args.download:
        _refresh_caches(args)
    summary = import_nba_current_actuals(
        args.raw_dir, args.processed_dir, args.reports_dir, min_season=args.min_season
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
