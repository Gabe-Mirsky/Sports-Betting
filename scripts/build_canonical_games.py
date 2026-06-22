"""Build the cross-source canonical games table + key summary report.

Writes:
    data/processed/canonical_games.csv
    data/reports/canonical_game_key_summary.json

Identity plumbing only - no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.canonical_games import build_canonical_games_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical games table.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_canonical_games_outputs(args.processed_dir, args.reports_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
