"""Sweep market-anchored model probability blends."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.market_anchor import save_market_anchor_outputs, sweep_market_anchor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep market-anchored probability blends.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--max-bet-fraction", type=float, default=0.03)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="market_anchor")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "matched_markets.csv"
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    markets = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    results, summary = sweep_market_anchor(
        markets,
        bankroll=args.bankroll,
        max_bet_fraction=args.max_bet_fraction,
    )
    save_market_anchor_outputs(results, summary, output_dir, prefix=args.prefix)

    print(f"Market-anchor status: {summary.get('status', 'n/a')}")
    print(f"Rules tested: {summary.get('rules_tested', 0):,}")
    print(f"Candidate rules: {summary.get('candidate_rules', 0):,}")
    print(f"Best model weight: {summary.get('best_model_weight', 0.0):.2f}")
    print(f"Best edge threshold: {summary.get('best_edge_threshold', 0.0):.1%}")
    print(f"Best trades: {summary.get('best_trades', 0):,}")
    print(f"Best average CLV: {summary.get('best_average_clv_cents', 0.0):+.2f} cents")
    print(f"Best profit/trade: {summary.get('best_average_profit_per_trade', 0.0):+.3f}")
    print(f"Saved market-anchor sweep to: {output_dir}")


if __name__ == "__main__":
    main()
