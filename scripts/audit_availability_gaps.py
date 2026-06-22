"""Report missing player availability statuses from the generated template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.availability_gaps import build_availability_gap_report, save_availability_gap_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit missing injury/availability statuses.")
    parser.add_argument("--template-path", default=None)
    parser.add_argument("--availability-path", default=None)
    parser.add_argument("--high-impact-minutes", type=float, default=20.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="availability_gap")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"game_id": str, "player_id": str})


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    template_path = (
        Path(args.template_path)
        if args.template_path
        else PROJECT_ROOT / "data" / "raw" / "nba" / "injuries" / "availability_template.csv"
    )
    availability_path = (
        Path(args.availability_path)
        if args.availability_path
        else PROJECT_ROOT / "data" / "raw" / "nba" / "injuries" / "availability.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "reports"
    template = _read_csv_if_exists(template_path)
    availability = _read_csv_if_exists(availability_path)
    gaps, summary = build_availability_gap_report(
        template,
        availability,
        high_impact_minutes=args.high_impact_minutes,
    )
    save_availability_gap_report(gaps, summary, output_dir, prefix=args.prefix)
    print(f"Availability gap status: {summary.get('status', 'n/a')}")
    print(f"Template rows: {summary.get('template_rows', 0):,}")
    print(f"Missing statuses: {summary.get('missing_rows', 0):,}")
    print(f"High-impact missing statuses: {summary.get('high_impact_missing_rows', 0):,}")
    print(f"Saved availability gap report to: {output_dir}")


if __name__ == "__main__":
    main()
