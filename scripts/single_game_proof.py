"""Build single-game edge proof gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.single_game_proof import build_single_game_proof_report_from_files, save_single_game_proof_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score whether single-game edge has been proven.")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--output-gates-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"
    gates_path = Path(args.output_gates_path) if args.output_gates_path else reports_dir / "single_game_proof_gates.csv"
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "single_game_proof_summary.json"
    )
    gates, summary = build_single_game_proof_report_from_files(reports_dir)
    save_single_game_proof_report(gates, summary, gates_path, summary_path)
    print(f"Single-game proof status: {summary.get('status', 'n/a')}")
    print(f"Hard failures: {summary.get('hard_failures', 0):,}")
    print(f"Warning failures: {summary.get('warning_failures', 0):,}")
    print(f"Parlay research allowed: {summary.get('parlay_research_allowed', False)}")
    print(f"Saved proof gates to: {gates_path}")
    print(f"Saved proof summary to: {summary_path}")


if __name__ == "__main__":
    main()
