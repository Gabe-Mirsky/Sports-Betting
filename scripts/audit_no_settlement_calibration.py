"""Audit NO-side CLV against settlement profit and calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from strategy.no_settlement_calibration import (  # noqa: E402
    build_no_settlement_calibration_audit,
    save_no_settlement_calibration_outputs,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_canonical_backtest_summary(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("market_source") != "kalshi":
        raise RuntimeError(f"Refusing NO settlement audit from source={summary.get('market_source')!r}.")
    if summary.get("price_source") != "kalshi_candlesticks_bid_ask":
        raise RuntimeError("NO settlement audit requires canonical Kalshi bid/ask candle pricing.")
    if not bool(summary.get("canonical_kalshi_backtest", False)):
        raise RuntimeError("NO settlement audit requires canonical_kalshi_backtest=true.")
    if bool(summary.get("stale_artifacts_detected", False)):
        raise RuntimeError("NO settlement audit refused stale backtest artifacts.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NO-side settlement calibration.")
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--backtest-summary-path", default=None)
    parser.add_argument("--proof-summary-path", default=None)
    parser.add_argument("--fair-price-summary-path", default=None)
    parser.add_argument("--parlay-summary-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "backtest_trades.csv"
    backtest_summary_path = (
        Path(args.backtest_summary_path) if args.backtest_summary_path else reports_dir / "backtest_summary.json"
    )
    proof_summary_path = (
        Path(args.proof_summary_path) if args.proof_summary_path else reports_dir / "single_game_proof_summary.json"
    )
    fair_summary_path = (
        Path(args.fair_price_summary_path) if args.fair_price_summary_path else reports_dir / "fair_price_summary.json"
    )
    parlay_summary_path = (
        Path(args.parlay_summary_path) if args.parlay_summary_path else reports_dir / "parlay_recommendations_summary.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    if not trades_path.exists():
        raise FileNotFoundError(f"Backtest trades not found: {trades_path}")
    validate_canonical_backtest_summary(backtest_summary_path)
    trades = pd.read_csv(trades_path, dtype={"game_id": str, "market_ticker": str})
    summary, by_bucket, clv_profit, failures, sweep, walk, recommendations = build_no_settlement_calibration_audit(
        trades,
        proof_summary=_read_json(proof_summary_path),
        fair_price_summary=_read_json(fair_summary_path),
        parlay_summary=_read_json(parlay_summary_path),
        starting_bankroll=float(config.strategy.starting_bankroll),
        max_bet_fraction=float(config.strategy.max_bet_fraction),
    )
    save_no_settlement_calibration_outputs(
        summary,
        by_bucket,
        clv_profit,
        failures,
        sweep,
        walk,
        recommendations,
        output_dir,
    )
    best = summary.get("best_suppression_rule", {}) or {}
    print(f"NO settlement status: {summary.get('status')}")
    print(f"NO rows: {summary.get('rows', 0):,}")
    print(f"Actual NO win rate: {summary.get('actual_no_win_rate', 0.0):.1%}")
    print(f"Avg predicted NO probability: {summary.get('avg_predicted_no_probability', 0.0):.1%}")
    print(f"Calibration error: {summary.get('calibration_error', 0.0):+.1%}")
    print(f"Avg NO CLV: {summary.get('avg_clv_cents', 0.0):+.3f} cents")
    print(f"Positive CLV profit: {summary.get('positive_clv_profit', 0.0):+.2f}")
    print(f"Best suppression rule: {best.get('rule', 'n/a')}")
    print(f"Best suppression status: {best.get('final_status', 'n/a')}")
    print(f"Saved NO settlement audit to: {output_dir}")


if __name__ == "__main__":
    main()
