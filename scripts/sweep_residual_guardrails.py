"""Walk-forward sweep residual guardrails for calibrated single-game signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.residual_audit import run_residual_guardrail_sweep  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep prior-history residual guardrails.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="residual_guardrail")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_calibrated_trades.csv"
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    rules, selected, summary = run_residual_guardrail_sweep(rows, signal_column=args.signal_column)
    rules.to_csv(output_dir / f"{args.prefix}_rules.csv", index=False)
    selected.to_csv(output_dir / f"{args.prefix}_selected.csv", index=False)
    (output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Rules tested: {summary.get('rules_tested', 0):,}")
    print(f"Candidates: {summary.get('candidates', 0):,}")
    print(f"Best status: {summary.get('best_status', 'n/a')}")
    print(f"Best signals: {summary.get('best_signals', 0):,}")
    print(f"Best positive CLV rate: {summary.get('best_positive_clv_rate', 0.0):.1%}")
    print(f"Saved residual guardrail sweep to: {output_dir}")


if __name__ == "__main__":
    main()
