"""Import real international soccer results into the matchup schema (no odds).

Example
-------
    python scripts/import_international_soccer_results.py \
        --output-path data/processed/match_results.csv

By default it reads the bundled local copy of the martj42/international_results
dataset (or downloads it if missing), cleans it, and writes
``data/processed/match_results.csv`` plus an import summary. No betting odds,
prices, CLV, or sportsbook data are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.international_soccer_importer import (  # noqa: E402
    DEFAULT_ALIASES_PATH,
    DEFAULT_RAW_OUTPUT_PATH,
    build_import_summary,
    load_raw_source,
    normalize_international_results,
    render_summary_markdown,
    render_summary_text,
)
from data.team_name_map import load_team_aliases  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import real international soccer results (no odds).")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", default="data/processed/match_results.csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    friendlies = parser.add_mutually_exclusive_group()
    friendlies.add_argument("--include-friendlies", dest="include_friendlies", action="store_true", default=True)
    friendlies.add_argument("--exclude-friendlies", dest="include_friendlies", action="store_false")
    parser.add_argument("--team-filter", default=None, help="Comma-separated team list to keep.")
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
        raw, source_label = load_raw_source(
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

    clean, stats = normalize_international_results(
        raw,
        aliases=aliases,
        start_date=args.start_date,
        end_date=args.end_date,
        min_year=args.min_year,
        max_year=args.max_year,
        include_friendlies=args.include_friendlies,
        team_filter=team_filter,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    summary = build_import_summary(clean, stats, source_label, output_path, raw_copy_path)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "international_soccer_import_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (reports_dir / "international_soccer_import_summary.md").write_text(
        render_summary_markdown(summary), encoding="utf-8"
    )

    print(render_summary_text(summary))
    if summary["warnings"]:
        print("Warnings: " + "; ".join(summary["warnings"]))
    print(f"Wrote {summary['rows_written']:,} games to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
