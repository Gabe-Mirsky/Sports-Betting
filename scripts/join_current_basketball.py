"""Join zachht game-odds snapshots to nba_api current games via canonical keys.

Writes:
    data/reports/current_basketball_join_summary.json
    data/reports/current_basketball_join_examples.csv

Research-only join diagnostics - no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.current_basketball_join import build_current_basketball_join_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join zachht odds to nba_api current games.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_current_basketball_join_outputs(args.processed_dir, args.reports_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
