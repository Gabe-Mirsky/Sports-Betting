"""Build a manual team-availability template from upcoming fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.availability_template_builder import (  # noqa: E402
    build_availability_template_from_fixtures,
    write_availability_template,
)
from data.fixtures_loader import load_fixtures, normalize_fixtures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual team-availability template from fixtures.")
    parser.add_argument("--fixtures-path", default="data/processed/fixtures_today.csv")
    parser.add_argument("--output-path", default="data/manual/current_fixture_availability_template.csv")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--include-placeholder-players", action="store_true")
    parser.add_argument("--players-per-team", type=int, default=1)
    parser.add_argument("--aliases-path", default="data/manual/team_aliases_template.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {"aliases_path": args.aliases_path} if args.aliases_path and Path(args.aliases_path).exists() else {}
    fixtures = normalize_fixtures(load_fixtures(args.fixtures_path), config)
    template = build_availability_template_from_fixtures(
        fixtures,
        as_of_date=args.as_of_date,
        include_placeholder_players=args.include_placeholder_players,
        players_per_team=args.players_per_team,
    )
    write_availability_template(template, args.output_path)
    fixture_teams = sorted(set(template["team"].dropna().astype(str))) if not template.empty else []
    print(f"Built availability template rows: {len(template):,}")
    print(f"Fixture teams covered: {len(fixture_teams):,}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
