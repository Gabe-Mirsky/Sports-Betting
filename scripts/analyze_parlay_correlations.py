"""Analyze same-slate signal pairs before any parlay research."""

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
from strategy.parlay_research import (  # noqa: E402
    apply_strategy_readiness_gate,
    build_parlay_pair_frame,
    save_parlay_research_outputs,
    summarize_parlay_pairs,
)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze parlay correlation readiness.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="consensus_trade")
    parser.add_argument("--readiness-summary-path", default=None)
    parser.add_argument("--output-pairs-path", default=None)
    parser.add_argument("--output-report-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--include-same-game", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_consensus_calibrated_trades.csv"
    readiness_path = (
        Path(args.readiness_summary_path)
        if args.readiness_summary_path
        else reports_dir / "strategy_readiness_summary.json"
    )
    pairs_path = Path(args.output_pairs_path) if args.output_pairs_path else reports_dir / "parlay_pair_rows.csv"
    report_path = (
        Path(args.output_report_path)
        if args.output_report_path
        else reports_dir / "parlay_correlation_report.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "parlay_correlation_summary.json"
    )

    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    pair_rows = build_parlay_pair_frame(
        rows,
        signal_column=args.signal_column,
        exclude_same_game=not args.include_same_game,
    )
    report, summary = summarize_parlay_pairs(pair_rows)
    summary = apply_strategy_readiness_gate(summary, _read_json(readiness_path))
    save_parlay_research_outputs(pair_rows, report, summary, pairs_path, report_path, summary_path)

    print(f"Parlay pair observations ({summary.get('timeline', 'n/a')}): {summary.get('pair_rows', 0):,}")
    print(f"Slates with pairs: {summary.get('slates_with_pairs', 0):,}")
    print(f"Pair win rate: {summary.get('pair_win_rate', 0.0) * 100:.2f}%")
    print(f"Leg outcome correlation: {summary.get('leg_outcome_correlation', 0.0):.4f}")
    print(f"Status: {summary.get('status', 'n/a')}")
    print(f"Parlay ready: {summary.get('parlay_ready', False)}")
    print(f"Saved parlay pair rows to: {pairs_path}")
    print(f"Saved parlay report to: {report_path}")
    print(f"Saved parlay summary to: {summary_path}")


if __name__ == "__main__":
    main()
