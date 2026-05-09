"""Validate saved NBA, Kalshi, candle, and audit artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.data_validation import write_validation_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate saved research data artifacts.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    summary, issues = write_validation_outputs(project_root)

    print(f"Validation status: {summary['validation_status']}")
    print(f"Error checks: {summary['error_checks']}")
    print(f"Warning checks: {summary['warning_checks']}")
    print(f"Auto matches: {summary.get('matches', {}).get('auto_matches', 'n/a')}")
    print(f"Strict eligible 60m price rows: {summary.get('prices', {}).get('strict_eligible_60m_rows', 'n/a')}")
    print(f"Issues written: {len(issues):,}")
    print(f"Summary: {project_root / 'data' / 'reports' / 'data_validation_summary.json'}")
    print(f"Issues: {project_root / 'data' / 'reports' / 'data_validation_issues.csv'}")


if __name__ == "__main__":
    main()
