"""Write the historical game inventory data-quality audit (JSON + MD).

Read-only and research-only. Reads local files only (no network), enables no
betting/predictions/parlays, and touches no model or readiness gates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logging_setup import setup_logging  # noqa: E402
from data.free_backfill import INVENTORY_PATH, STAGING_DIR  # noqa: E402
from reports.historical_inventory_quality import build_quality_audit, write_quality_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the historical game inventory quality.")
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--inventory-path", default=str(INVENTORY_PATH))
    parser.add_argument("--staging-dir", default=str(STAGING_DIR))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    paths = write_quality_audit(
        reports_dir=args.reports_dir,
        inventory_path=args.inventory_path,
        staging_dir=args.staging_dir,
    )
    audit = build_quality_audit(inventory_path=args.inventory_path, staging_dir=args.staging_dir)
    print(f"Verdict: {audit['verdict']}  (total games: {audit['total_games']:,})")
    if audit["critical_issues"]:
        print("Critical:")
        for issue in audit["critical_issues"]:
            print(f"  - {issue}")
    if audit["warning_issues"]:
        print("Warnings:")
        for issue in audit["warning_issues"]:
            print(f"  - {issue}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
