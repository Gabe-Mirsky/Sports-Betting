"""Write the player-prop snapshot schema template + summary report.

Writes:
    data/templates/player_prop_snapshot_template.csv
    data/reports/player_prop_schema_summary.md

Schema plumbing only - no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.player_prop_schema import build_player_prop_schema_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the player prop snapshot schema outputs.")
    parser.add_argument("--templates-dir", default=str(PROJECT_ROOT / "data" / "templates"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_player_prop_schema_outputs(args.templates_dir, args.reports_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
