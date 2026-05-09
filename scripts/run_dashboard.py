"""Launch the optional Streamlit dashboard."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the interactive dashboard.")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "Streamlit is not installed. Run: "
            "python -m pip install -r requirements-dashboard.txt"
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "scripts" / "dashboard_app.py"),
            "--",
            "--report-dir",
            args.report_dir,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
