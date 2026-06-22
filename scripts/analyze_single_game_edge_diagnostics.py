"""Build root-cause diagnostics for failed single-game edge proof gates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.single_game_edge_diagnostics import (  # noqa: E402
    build_single_game_edge_diagnostics,
    save_single_game_edge_diagnostics,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why the single-game Kalshi edge is not proven.")
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--fair-price-path", default=None)
    parser.add_argument("--backtest-summary-path", default=None)
    parser.add_argument("--clv-summary-path", default=None)
    parser.add_argument("--proof-summary-path", default=None)
    parser.add_argument("--proof-gates-path", default=None)
    parser.add_argument("--parlay-summary-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--generated-date", default=date.today().isoformat())
    parser.add_argument("--min-rows", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    trades_path = Path(args.trades_path) if args.trades_path else reports_dir / "backtest_trades.csv"
    fair_price_path = Path(args.fair_price_path) if args.fair_price_path else reports_dir / "fair_price_signals.csv"
    backtest_summary_path = (
        Path(args.backtest_summary_path) if args.backtest_summary_path else reports_dir / "backtest_summary.json"
    )
    clv_summary_path = Path(args.clv_summary_path) if args.clv_summary_path else reports_dir / "clv_summary.json"
    proof_summary_path = (
        Path(args.proof_summary_path) if args.proof_summary_path else reports_dir / "single_game_proof_summary.json"
    )
    proof_gates_path = (
        Path(args.proof_gates_path) if args.proof_gates_path else reports_dir / "single_game_proof_gates.csv"
    )
    parlay_summary_path = (
        Path(args.parlay_summary_path)
        if args.parlay_summary_path
        else reports_dir / "parlay_recommendations_summary.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    if not trades_path.exists():
        raise FileNotFoundError(f"Backtest trades not found: {trades_path}")

    trades = _read_csv(trades_path, dtype={"game_id": str, "market_ticker": str})
    fair_price_signals = _read_csv(fair_price_path, dtype={"game_id": str, "market_ticker": str})
    proof_gates = _read_csv(proof_gates_path)
    summary, diagnostics, failure_segments, walk_forward, recommendations = build_single_game_edge_diagnostics(
        trades=trades,
        fair_price_signals=fair_price_signals,
        backtest_summary=_read_json(backtest_summary_path),
        clv_summary=_read_json(clv_summary_path),
        proof_summary=_read_json(proof_summary_path),
        proof_gates=proof_gates,
        parlay_summary=_read_json(parlay_summary_path),
        generated_date=args.generated_date,
        min_rows=args.min_rows,
    )
    save_single_game_edge_diagnostics(
        summary,
        diagnostics,
        failure_segments,
        walk_forward,
        recommendations,
        output_dir,
    )

    print(f"Single-game edge proven: {summary.get('single_game_edge_proven', False)}")
    print(f"Canonical Kalshi backtest: {summary.get('canonical_kalshi_backtest', False)}")
    print(f"Trades: {summary.get('trades', 0):,}")
    print(f"Average CLV: {summary.get('average_clv_cents', 0.0):+.3f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Fair-price actionable bets: {summary.get('actionable_fair_price_bets', 0):,}")
    print(f"Parlay status: {summary.get('parlay_status', 'unknown')}")
    print(f"Saved diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
