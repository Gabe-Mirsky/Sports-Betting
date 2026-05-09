"""Streamlit entry point for the interactive dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.interactive_dashboard import run_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive Streamlit dashboard.")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_app(args.report_dir)


if __name__ == "__main__":
    main()
