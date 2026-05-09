"""Build a self-contained local HTML dashboard from saved reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.dashboard import write_dashboard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local HTML dashboard.")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "dashboard.html"

    dashboard_path = write_dashboard(reports_dir, output_path)
    print(f"Built dashboard: {dashboard_path}")
    print("Open that HTML file in your browser.")


if __name__ == "__main__":
    main()
