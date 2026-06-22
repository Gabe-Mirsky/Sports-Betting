"""Run research-only side-specific probability shrinkage sweep."""

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
from strategy.shrinkage_policy_sweep import (  # noqa: E402
    run_side_specific_shrinkage_sweep,
    save_side_specific_shrinkage_outputs,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_canonical_backtest_summary(path: Path) -> dict[str, Any]:
    """Require canonical Kalshi bid/ask backtest metadata before research sweeps."""

    summary = _read_json(path)
    if summary.get("market_source") != "kalshi":
        raise RuntimeError(f"Refusing shrinkage sweep from non-Kalshi source={summary.get('market_source')!r}.")
    if summary.get("price_source") != "kalshi_candlesticks_bid_ask":
        raise RuntimeError("Shrinkage sweep requires Kalshi candlestick bid/ask prices.")
    if not bool(summary.get("canonical_kalshi_backtest", False)):
        raise RuntimeError("Shrinkage sweep requires canonical_kalshi_backtest=true.")
    if bool(summary.get("stale_artifacts_detected", False)):
        raise RuntimeError("Shrinkage sweep refused stale canonical backtest artifacts.")
    return summary


def _baseline_from_backtest(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "profit": float(summary.get("ending_bankroll", 0.0) or 0.0) - float(summary.get("starting_bankroll", 0.0) or 0.0),
        "average_clv_cents": float(summary.get("average_clv_cents", 0.0) or 0.0),
        "positive_clv_rate": float(summary.get("positive_clv_rate", 0.0) or 0.0),
        "yes_profit": float(summary.get("yes_profit", 0.0) or 0.0),
        "no_profit": float(summary.get("no_profit", 0.0) or 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only side-specific probability shrinkage sweep.")
    parser.add_argument("--markets-path", default=None)
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
    markets_path = Path(args.markets_path) if args.markets_path else reports_dir / "matched_markets.csv"
    backtest_summary_path = (
        Path(args.backtest_summary_path) if args.backtest_summary_path else reports_dir / "backtest_summary.json"
    )
    proof_summary_path = (
        Path(args.proof_summary_path) if args.proof_summary_path else reports_dir / "single_game_proof_summary.json"
    )
    fair_price_summary_path = (
        Path(args.fair_price_summary_path) if args.fair_price_summary_path else reports_dir / "fair_price_summary.json"
    )
    parlay_summary_path = (
        Path(args.parlay_summary_path) if args.parlay_summary_path else reports_dir / "parlay_recommendations_summary.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir

    if not markets_path.exists():
        raise FileNotFoundError(f"Matched markets file not found: {markets_path}")
    backtest_summary = validate_canonical_backtest_summary(backtest_summary_path)
    proof_summary = _read_json(proof_summary_path)
    fair_summary = _read_json(fair_price_summary_path)
    parlay_summary = _read_json(parlay_summary_path)

    markets = pd.read_csv(markets_path, dtype={"game_id": str, "market_ticker": str})
    sweep, walk_forward, summary, recommendations = run_side_specific_shrinkage_sweep(
        markets,
        starting_bankroll=float(config.strategy.starting_bankroll),
        max_bet_fraction=float(config.strategy.max_bet_fraction),
        baseline=_baseline_from_backtest(backtest_summary),
    )
    summary["validated_backtest_market_source"] = backtest_summary.get("market_source")
    summary["validated_backtest_price_source"] = backtest_summary.get("price_source")
    summary["canonical_kalshi_backtest"] = bool(backtest_summary.get("canonical_kalshi_backtest", False))
    summary["stale_artifacts_detected"] = bool(backtest_summary.get("stale_artifacts_detected", False))
    summary["proof_status"] = proof_summary.get("status", "unknown")
    summary["proof_single_game_edge_proven"] = bool(proof_summary.get("single_game_edge_proven", False))
    summary["fair_price_bets_current"] = int(fair_summary.get("bets", 0) or 0)
    summary["fair_price_proof_gate_status"] = fair_summary.get("proof_gate_status", "unknown")
    summary["parlay_status_current"] = parlay_summary.get("status", "unknown")
    summary["parlay_recommendations_allowed_current"] = bool(
        parlay_summary.get("parlay_recommendations_allowed", False)
    )
    save_side_specific_shrinkage_outputs(sweep, walk_forward, summary, recommendations, output_dir)

    best = summary.get("best_policy", {}) or {}
    print(f"Shrinkage sweep status: {summary.get('status')}")
    print(f"Policies tested: {summary.get('policies_tested', 0):,}")
    print(f"Candidate policies: {summary.get('candidate_policies', 0):,}")
    print(f"Watchlist policies: {summary.get('watchlist_policies', 0):,}")
    print(f"Best policy: {best.get('policy', 'n/a')}")
    print(f"Best final status: {best.get('final_status', 'n/a')}")
    print(f"Best profit: {float(best.get('profit', 0.0) or 0.0):+.2f}")
    print(f"Best average CLV: {float(best.get('average_clv_cents', 0.0) or 0.0):+.3f} cents")
    print(f"Best positive CLV rate: {float(best.get('positive_clv_rate', 0.0) or 0.0):.1%}")
    print(f"Proof status remains: {summary.get('proof_status', 'unknown')}")
    print(f"Parlay status remains: {summary.get('parlay_status_current', 'unknown')}")
    print(f"Saved shrinkage reports to: {output_dir}")


if __name__ == "__main__":
    main()
