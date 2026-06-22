"""Report the baseline validation backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print baseline validation backtest summary.")
    parser.add_argument("--summary-path", default=str(PROJECT_ROOT / "outputs" / "validation_backtest_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    if not summary_path.exists():
        raise SystemExit(f"Validation backtest summary not found: {summary_path}. Run scripts/train_model.py first.")
    summary = pd.read_csv(summary_path)
    print("Validation backtest by edge threshold")
    print("-------------------------------------")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
