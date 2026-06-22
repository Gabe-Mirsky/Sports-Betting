"""Import upcoming international soccer fixtures into fixtures_today.csv.

This is a no-odds importer. It reads future, unscored rows from the martj42
international_results dataset and writes the fixture schema consumed by the
matchup prediction pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.international_fixtures_importer import (  # noqa: E402
    DEFAULT_ALIASES_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RAW_OUTPUT_PATH,
    build_fixture_import_summary,
    load_raw_fixture_source,
    normalize_international_fixtures,
    render_fixture_summary_text,
)
from data.team_name_map import load_team_aliases  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import upcoming international soccer fixtures (no odds).")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days-ahead", type=int, default=14)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--team-filter", default=None, help="Comma-separated team list to keep.")
    parser.add_argument("--include-past-today", action="store_true")
    parser.add_argument("--write-raw-copy", action="store_true")
    parser.add_argument("--raw-output-path", default=DEFAULT_RAW_OUTPUT_PATH)
    parser.add_argument("--aliases-path", default=DEFAULT_ALIASES_PATH)
    parser.add_argument("--reports-dir", default="data/reports")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        raw, source_label = load_raw_fixture_source(
            input_path=args.input_path,
            source_url=args.source_url,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    raw_copy_path = None
    if args.write_raw_copy:
        raw_copy_path = Path(args.raw_output_path)
        raw_copy_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_copy_path, index=False)

    aliases = load_team_aliases(args.aliases_path) if args.aliases_path else None
    team_filter = [t.strip() for t in args.team_filter.split(",")] if args.team_filter else None

    try:
        fixtures, stats = normalize_international_fixtures(
            raw,
            aliases=aliases,
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            end_date=args.end_date,
            days_ahead=args.days_ahead,
            team_filter=team_filter,
            include_past_today=args.include_past_today,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)

    summary = build_fixture_import_summary(fixtures, stats, source_label, output_path, raw_copy_path)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "international_fixtures_import_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(render_fixture_summary_text(summary))
    if summary["warnings"]:
        print("Warnings: " + "; ".join(summary["warnings"]))
    print(f"Wrote {summary['rows_written']:,} fixtures to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
