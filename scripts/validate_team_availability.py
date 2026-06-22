"""Validate team availability coverage for upcoming fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.fixtures_loader import load_fixtures, normalize_fixtures  # noqa: E402
from data.injuries_loader import load_injuries, normalize_injuries  # noqa: E402
from quality.team_availability_validation import (  # noqa: E402
    STATUS_FAIL,
    build_team_availability_validation_report,
    render_team_availability_markdown,
    save_team_availability_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate team availability coverage.")
    parser.add_argument("--fixtures-path", default="data/processed/fixtures_today.csv")
    parser.add_argument("--injuries-path", default="data/processed/injuries.csv")
    parser.add_argument("--aliases-path", default="data/manual/team_aliases_template.csv")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--allow-unknown-player-rows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {"aliases_path": args.aliases_path} if args.aliases_path and Path(args.aliases_path).exists() else {}

    try:
        fixtures = normalize_fixtures(load_fixtures(args.fixtures_path), config)
        raw_injuries = load_injuries(args.injuries_path)
        injuries = normalize_injuries(raw_injuries, config)
        report = build_team_availability_validation_report(
            fixtures,
            injuries,
            raw_injuries=raw_injuries,
            as_of_date=args.as_of_date,
            allow_unknown_player_rows=args.allow_unknown_player_rows,
        )
    except Exception as exc:
        report = {
            "overall_status": STATUS_FAIL,
            "coverage": {
                "total_fixture_teams": 0,
                "fixture_teams_with_availability": 0,
                "fixture_teams_missing_availability": 0,
                "coverage_percentage": 0.0,
                "missing_teams": [],
            },
            "injury_data": {
                "rows_loaded": 0,
                "valid_status_rows": 0,
                "invalid_status_rows": 0,
                "invalid_statuses": [],
                "stale_rows_older_than_48h": 0,
                "missing_player_names": 0,
                "unknown_player_rows": 0,
                "missing_importance_scores": 0,
                "teams_not_found_in_fixtures": [],
                "duplicate_team_player_rows": 0,
                "status_counts": {},
            },
            "team_rows": [],
            "issues": [f"Could not validate team availability: {exc}"],
            "warnings": [],
        }

    json_path, md_path = save_team_availability_validation_report(report, args.output_dir)
    print(render_team_availability_markdown(report))
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    return 1 if report.get("overall_status") == STATUS_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
