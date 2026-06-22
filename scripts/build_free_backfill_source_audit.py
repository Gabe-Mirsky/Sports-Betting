"""Write the free-backfill source audit (markdown + CSV inventory).

Research-only planning artifact. Downloads nothing. Emits
``data/reports/free_backfill_source_audit.md`` and
``data/reports/free_backfill_source_inventory.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logging_setup import setup_logging  # noqa: E402
from reports.free_backfill_audit import write_source_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the free-backfill source audit.")
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    paths = write_source_audit(args.reports_dir)
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
