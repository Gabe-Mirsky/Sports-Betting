"""Audit whether real Kalshi non-winner markets have usable line values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.market_line_audit import build_market_line_coverage, save_market_line_coverage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Kalshi spread/total/team-total/prop line extraction.")
    parser.add_argument("--taxonomy-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy_path = (
        Path(args.taxonomy_path)
        if args.taxonomy_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_market_taxonomy.csv"
    )
    reports_dir = PROJECT_ROOT / "data" / "reports"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "market_line_coverage.csv"
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else reports_dir / "market_line_coverage_summary.json"
    )

    taxonomy = pd.read_csv(taxonomy_path) if taxonomy_path.exists() else pd.DataFrame()
    coverage, summary = build_market_line_coverage(taxonomy)
    save_market_line_coverage(coverage, summary, output_path, summary_path)

    print(f"Line-market rows: {summary.get('line_market_rows', 0):,}")
    print(f"Ready market types: {summary.get('ready_market_types', [])}")
    print(f"Blocked market types: {summary.get('blocked_market_types', [])}")
    print(f"Saved line coverage to: {output_path}")
    print(f"Saved line summary to: {summary_path}")


if __name__ == "__main__":
    main()
