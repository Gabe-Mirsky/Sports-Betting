"""Classify cached Kalshi NBA markets into market types."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_taxonomy import write_market_taxonomy_outputs  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a taxonomy table for cached NBA Kalshi markets.")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--taxonomy-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--raw-only", action="store_true", help="Skip processed possible-NBA market cache.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    raw_dir = Path(args.raw_dir) if args.raw_dir else PROJECT_ROOT / "data" / "raw" / "kalshi"
    taxonomy_path = Path(args.taxonomy_path) if args.taxonomy_path else None
    summary_path = Path(args.summary_path) if args.summary_path else None

    taxonomy, summary = write_market_taxonomy_outputs(
        raw_dir=raw_dir,
        taxonomy_path=taxonomy_path,
        summary_path=summary_path,
        include_processed_possible=not args.raw_only,
    )

    print(f"Classified markets: {len(taxonomy):,}")
    for category, count in summary.get("category_counts", {}).items():
        print(f"- {category}: {count:,}")
    print(f"Low-confidence rows: {summary.get('low_confidence_rows', 0):,}")
    print(f"Taxonomy table: {taxonomy_path or PROJECT_ROOT / 'data' / 'processed' / 'kalshi_market_taxonomy.csv'}")
    print(f"Summary: {summary_path or PROJECT_ROOT / 'data' / 'reports' / 'kalshi_market_taxonomy_summary.json'}")


if __name__ == "__main__":
    main()
