"""Build fair-price recommendations from matched single-game markets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from strategy.fair_price import (  # noqa: E402
    apply_single_game_proof_gate,
    build_fair_price_signals,
    save_fair_price_signals,
    summarize_fair_price_signals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fair-price signals for single-game markets.")
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--backtest-summary-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--proof-summary-path", default=None)
    parser.add_argument("--skip-proof-gate", action="store_true")
    parser.add_argument("--edge-threshold", type=float, default=0.03)
    parser.add_argument("--max-spread-cents", type=float, default=None)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--fee-penalty", type=float, default=0.005)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.02)
    parser.add_argument("--spread-penalty-fraction", type=float, default=0.5)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--max-bet-fraction", type=float, default=None)
    parser.add_argument("--research-lean-min-edge", type=float, default=None)
    parser.add_argument("--paper-trade-min-edge", type=float, default=0.05)
    parser.add_argument("--paper-trade-min-model-probability", type=float, default=0.55)
    parser.add_argument("--paper-trade-min-volume", type=float, default=None)
    parser.add_argument("--paper-trade-max-spread-cents", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def validate_canonical_backtest_summary(summary_path: Path) -> dict:
    """Require canonical fair-price inputs to come from Kalshi candlestick backtests."""

    if not summary_path.exists():
        raise FileNotFoundError(f"Backtest summary not found for source validation: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("market_source") != "kalshi":
        raise RuntimeError(
            "Refusing to build canonical fair-price signals from non-Kalshi backtest "
            f"source={summary.get('market_source')!r}."
        )
    if not bool(summary.get("canonical_kalshi_backtest", False)):
        raise RuntimeError("Backtest summary is not marked as the canonical Kalshi backtest.")
    if summary.get("price_source") != "kalshi_candlesticks_bid_ask":
        raise RuntimeError(
            "Canonical fair-price signals require bid/ask Kalshi candle pricing; "
            f"found price_source={summary.get('price_source')!r}."
        )
    if not bool(summary.get("bid_ask_required", False)):
        raise RuntimeError("Canonical Kalshi backtest did not require bid/ask pricing.")
    if bool(summary.get("stale_artifacts_detected", False)):
        raise RuntimeError(
            "Canonical Kalshi backtest summary reports stale artifacts: "
            + ", ".join(str(item) for item in summary.get("artifact_warnings", []))
        )
    return summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    markets_path = Path(args.markets_path) if args.markets_path else reports_dir / "matched_markets.csv"
    backtest_summary_path = (
        Path(args.backtest_summary_path)
        if args.backtest_summary_path
        else reports_dir / "backtest_summary.json"
    )
    output_path = Path(args.output_path) if args.output_path else reports_dir / "fair_price_signals.csv"
    summary_path = Path(args.summary_path) if args.summary_path else reports_dir / "fair_price_summary.json"
    proof_summary_path = (
        Path(args.proof_summary_path)
        if args.proof_summary_path
        else reports_dir / "single_game_proof_summary.json"
    )
    if not markets_path.exists():
        raise FileNotFoundError(f"Matched markets file not found: {markets_path}")
    backtest_summary = validate_canonical_backtest_summary(backtest_summary_path)

    markets = pd.read_csv(markets_path, dtype={"game_id": str, "market_ticker": str})
    signals = build_fair_price_signals(
        markets,
        edge_threshold=args.edge_threshold,
        max_spread_cents=(
            args.max_spread_cents
            if args.max_spread_cents is not None
            else config.backtest.max_bid_ask_spread_cents
        ),
        min_volume=args.min_volume if args.min_volume is not None else config.backtest.min_volume,
        fee_penalty=args.fee_penalty,
        uncertainty_penalty=args.uncertainty_penalty,
        spread_penalty_fraction=args.spread_penalty_fraction,
        starting_bankroll=args.bankroll if args.bankroll is not None else config.strategy.starting_bankroll,
        max_bet_fraction=(
            args.max_bet_fraction
            if args.max_bet_fraction is not None
            else config.strategy.max_bet_fraction
        ),
        research_lean_min_edge=args.research_lean_min_edge,
        paper_trade_min_edge=args.paper_trade_min_edge,
        paper_trade_min_model_probability=args.paper_trade_min_model_probability,
        paper_trade_min_volume=args.paper_trade_min_volume,
        paper_trade_max_spread_cents=args.paper_trade_max_spread_cents,
    )
    if not args.skip_proof_gate:
        proof_summary = (
            json.loads(proof_summary_path.read_text(encoding="utf-8"))
            if proof_summary_path.exists()
            else {"single_game_edge_proven": False, "status": "missing"}
        )
        signals = apply_single_game_proof_gate(
            signals,
            single_game_edge_proven=bool(proof_summary.get("single_game_edge_proven", False)),
            proof_status=str(proof_summary.get("status", "unknown")),
        )
    summary = summarize_fair_price_signals(signals)
    summary["validated_backtest_market_source"] = backtest_summary.get("market_source")
    summary["validated_backtest_price_source"] = backtest_summary.get("price_source")
    summary["validated_backtest_snapshot_target"] = backtest_summary.get("snapshot_target", "")
    summary["validated_backtest_generated_at_utc"] = backtest_summary.get("generated_at_utc", "")
    save_fair_price_signals(signals, summary, output_path, summary_path)
    print(f"Fair-price rows: {summary.get('rows', 0):,}")
    print(f"Recommended bets: {summary.get('bets', 0):,}")
    print(f"Approved bets: {summary.get('approved_bets_count', 0):,}")
    print(f"Paper trade candidates: {summary.get('paper_trade_candidates_count', 0):,}")
    print(f"Research leans: {summary.get('research_leans_count', 0):,}")
    if "ungated_bets" in summary:
        print(f"Ungated research bets: {summary.get('ungated_bets', 0):,}")
    if "proof_gate_status" in summary:
        print(f"Proof gate status: {summary.get('proof_gate_status', 'unknown')}")
    print(f"YES bets: {summary.get('yes_bets', 0):,}")
    print(f"NO bets: {summary.get('no_bets', 0):,}")
    print(f"Saved fair-price signals to: {output_path}")
    print(f"Saved fair-price summary to: {summary_path}")


if __name__ == "__main__":
    main()
