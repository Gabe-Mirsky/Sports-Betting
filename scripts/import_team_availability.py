"""Import manual team availability CSVs into data/processed/injuries.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.team_availability_importer import import_team_availability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manual team availability CSVs.")
    parser.add_argument("--input-path", action="append", required=True)
    parser.add_argument("--output-path", default="data/processed/injuries.csv")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--fixtures-path", default=None, help="Accepted for workflow symmetry; not required by importer.")
    parser.add_argument("--aliases-path", default="data/manual/team_aliases_template.csv")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--drop-unknown-player-rows", action="store_true")
    parser.add_argument("--keep-unknown-status", action="store_true", default=True)
    parser.add_argument("--summary-path", default="data/reports/team_availability_import_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary = import_team_availability(
        input_paths=args.input_path,
        output_path=args.output_path,
        append=args.append,
        aliases_path=args.aliases_path,
        as_of_date=args.as_of_date,
        drop_unknown_player_rows=args.drop_unknown_player_rows,
        keep_unknown_status=args.keep_unknown_status,
    )

    summary_data = summary.as_dict()
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    print(f"Availability rows read: {summary.rows_read:,}")
    print(f"Availability rows written: {summary.rows_written:,}")
    print(f"Statuses found: {', '.join(summary.statuses_found) if summary.statuses_found else 'none'}")
    if summary.warnings:
        print("Warnings:")
        for warning in summary.warnings:
            print(f"- {warning}")
    print(f"Output: {args.output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
