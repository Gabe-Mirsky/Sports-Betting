"""Download NBA player game logs with local caching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.player_client import download_player_seasons  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download cached NBA player game logs.")
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--force", action="store_true", help="Re-download even if cache exists.")
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed season.")
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
    cache_dir = resolve_project_path(config.data.player_cache_dir)

    frames = download_player_seasons(
        start_season=start_season,
        end_season=end_season,
        cache_dir=cache_dir,
        season_type=args.season_type,
        force=args.force,
        strict=args.strict,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
    )

    total_rows = sum(len(frame) for frame in frames)
    print(f"Downloaded or loaded {len(frames)} seasons with {total_rows:,} player-game rows.")
    print(f"Raw player cache folder: {cache_dir}")
    if args.season_type == "Regular Season":
        print("Run again with --season-type Playoffs if you want playoff player logs too.")


if __name__ == "__main__":
    main()
