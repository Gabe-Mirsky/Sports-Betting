"""Create a local injury/availability CSV template with player impact weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.availability_template import build_availability_template, write_availability_template  # noqa: E402
from data.player_client import load_raw_player_logs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an injury/availability entry template.")
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--player-cache-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--players-per-team", type=int, default=10)
    parser.add_argument("--lookback-games", type=int, default=10)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    games_path = Path(args.games_path) if args.games_path else PROJECT_ROOT / "data" / "interim" / "nba_games.parquet"
    player_cache_dir = (
        Path(args.player_cache_dir)
        if args.player_cache_dir
        else resolve_project_path(config.data.player_cache_dir)
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "data" / "raw" / "nba" / "injuries" / "availability_template.csv"
    )

    if not games_path.exists():
        raise SystemExit(f"Games file not found: {games_path}. Run scripts/build_features.py first.")
    games = pd.read_parquet(games_path) if games_path.suffix.lower() == ".parquet" else pd.read_csv(games_path)
    player_logs = load_raw_player_logs(player_cache_dir)
    if player_logs.empty:
        raise SystemExit(
            f"No player logs found in {player_cache_dir}. Run scripts/download_nba_player_data.py first."
        )
    template = build_availability_template(
        games,
        player_logs,
        start_date=args.start_date,
        end_date=args.end_date,
        players_per_team=args.players_per_team,
        lookback_games=args.lookback_games,
    )
    write_availability_template(template, output_path)
    print(f"Template rows: {len(template):,}")
    print(f"Saved template to: {output_path}")
    print("Fill the status column from a free/allowed injury source, then save as availability.csv.")


if __name__ == "__main__":
    main()
