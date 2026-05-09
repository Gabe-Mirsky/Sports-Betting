"""Build the default headline slate-settled paper-trading summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.headline import build_headline_backtest_summary, save_headline_backtest_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose the default slate-settled headline backtest result.")
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output_path) if args.output_path else reports_dir / "headline_backtest_summary.json"
    summary = build_headline_backtest_summary(reports_dir)
    save_headline_backtest_summary(summary, output_path)

    print(f"Headline result: {summary.get('headline_label', 'n/a')}")
    print(f"Settlement mode: {summary.get('settlement_mode', 'n/a')}")
    print(
        f"Headline trades ({summary.get('trade_timeline', 'n/a')}): "
        f"{summary.get('num_selected_trades', 0):,}"
    )
    print(f"Ending bankroll: ${summary.get('ending_bankroll', 0.0):.2f}")
    print(f"Readiness: {summary.get('readiness_status', 'unknown')}")
    print(f"Parlays blocked: {summary.get('parlays_blocked', True)}")
    print(f"Saved headline summary to: {output_path}")


if __name__ == "__main__":
    main()
