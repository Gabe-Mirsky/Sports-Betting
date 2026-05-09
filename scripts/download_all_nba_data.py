"""Download regular-season and playoff NBA team game logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.nba_client import download_seasons  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NBA regular-season and playoff logs.")
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--regular-season-only", action="store_true")
    parser.add_argument("--playoffs-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    start_season = args.start_season or config.data.start_season
    end_season = args.end_season or config.data.end_season
    cache_dir = resolve_project_path(config.data.cache_dir)

    if args.regular_season_only and args.playoffs_only:
        raise SystemExit("Choose at most one of --regular-season-only or --playoffs-only.")

    season_types = ["Regular Season", "Playoffs"]
    if args.regular_season_only:
        season_types = ["Regular Season"]
    if args.playoffs_only:
        season_types = ["Playoffs"]

    total_frames = 0
    total_rows = 0
    for season_type in season_types:
        frames = download_seasons(
            start_season=start_season,
            end_season=end_season,
            cache_dir=cache_dir,
            season_type=season_type,
            force=args.force,
            strict=args.strict,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
        )
        rows = sum(len(frame) for frame in frames)
        total_frames += len(frames)
        total_rows += rows
        print(f"{season_type}: {len(frames)} seasons, {rows:,} team-game rows.")

    print(f"Finished {total_frames} downloads/loads with {total_rows:,} team-game rows.")
    print(f"Raw cache folder: {cache_dir}")


if __name__ == "__main__":
    main()
