"""Build conservative two-leg parlay recommendation candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.parlay_recommendations import (  # noqa: E402
    build_parlay_recommendations,
    build_research_parlay_candidates,
    save_parlay_recommendations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative parlay recommendation candidates.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--fair-price-summary-path", default=None)
    parser.add_argument("--proof-summary-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--research-output-path", default=None)
    parser.add_argument("--paper-output-path", default=None)
    parser.add_argument("--research-summary-path", default=None)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--min-leg-edge", type=float, default=0.02)
    parser.add_argument("--min-combined-edge", type=float, default=0.03)
    parser.add_argument("--min-research-average-leg-edge", type=float, default=0.03)
    parser.add_argument("--max-recommendations", type=int, default=20)
    return parser.parse_args()


def validate_fair_price_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Fair-price summary not found for source validation: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("validated_backtest_market_source") != "kalshi":
        raise RuntimeError(
            "Refusing to build parlay recommendations from fair-price rows without "
            "validated Kalshi backtest source."
        )
    if summary.get("validated_backtest_price_source") != "kalshi_candlesticks_bid_ask":
        raise RuntimeError("Parlay recommendations require fair-price rows from bid/ask Kalshi candle pricing.")
    return summary


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "fair_price_signals.csv"
    proof_path = (
        Path(args.proof_summary_path)
        if args.proof_summary_path
        else reports_dir / "single_game_proof_summary.json"
    )
    fair_price_summary_path = (
        Path(args.fair_price_summary_path)
        if args.fair_price_summary_path
        else reports_dir / "fair_price_summary.json"
    )
    output_path = Path(args.output_path) if args.output_path else reports_dir / "parlay_recommendations.csv"
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "parlay_recommendations_summary.json"
    )
    research_output_path = (
        Path(args.research_output_path)
        if args.research_output_path
        else reports_dir / "research_parlay_candidates.csv"
    )
    paper_output_path = (
        Path(args.paper_output_path)
        if args.paper_output_path
        else reports_dir / "paper_parlay_candidates.csv"
    )
    research_summary_path = (
        Path(args.research_summary_path)
        if args.research_summary_path
        else reports_dir / "parlay_research_summary.json"
    )

    fair_price_signals = pd.read_csv(input_path) if input_path.exists() else pd.DataFrame()
    fair_price_summary = validate_fair_price_summary(fair_price_summary_path)
    proof_summary = json.loads(proof_path.read_text(encoding="utf-8")) if proof_path.exists() else {}
    recommendations, summary = build_parlay_recommendations(
        fair_price_signals,
        proof_summary=proof_summary,
        bankroll=args.bankroll,
        min_leg_edge=args.min_leg_edge,
        min_combined_edge=args.min_combined_edge,
        max_recommendations=args.max_recommendations,
    )
    summary["validated_backtest_market_source"] = fair_price_summary.get("validated_backtest_market_source")
    summary["validated_backtest_price_source"] = fair_price_summary.get("validated_backtest_price_source")
    summary["validated_backtest_snapshot_target"] = fair_price_summary.get("validated_backtest_snapshot_target", "")
    save_parlay_recommendations(recommendations, summary, output_path, summary_path)
    research_parlays, research_summary = build_research_parlay_candidates(
        fair_price_signals,
        parlay_tier="research_parlay",
        max_candidates=args.max_recommendations,
        min_average_leg_edge=args.min_research_average_leg_edge,
    )
    paper_parlays, paper_summary = build_research_parlay_candidates(
        fair_price_signals,
        parlay_tier="paper_parlay",
        max_candidates=args.max_recommendations,
        min_average_leg_edge=args.min_research_average_leg_edge,
    )
    research_output_path.parent.mkdir(parents=True, exist_ok=True)
    paper_output_path.parent.mkdir(parents=True, exist_ok=True)
    research_summary_path.parent.mkdir(parents=True, exist_ok=True)
    research_parlays.to_csv(research_output_path, index=False)
    paper_parlays.to_csv(paper_output_path, index=False)
    combined_research_summary = {
        "status": "research_only_generated",
        "approved_parlay_status": summary.get("status"),
        "approved_parlays_allowed": bool(summary.get("parlay_recommendations_allowed", False)),
        "single_game_edge_proven": bool(summary.get("single_game_edge_proven", False)),
        "research_only": True,
        "approved": False,
        "research_parlay": research_summary,
        "paper_parlay": paper_summary,
        "warning": "Research and paper parlays are experimental only. They are not approved bets and include no stake sizing.",
    }
    research_summary_path.write_text(json.dumps(combined_research_summary, indent=2), encoding="utf-8")

    print(f"Parlay recommendation status: {summary.get('status')}")
    print(f"Eligible single-game legs: {summary.get('eligible_single_game_legs', 0):,}")
    print(f"Parlays: {summary.get('parlays', 0):,}")
    print(f"Research parlays: {len(research_parlays):,}")
    print(f"Paper parlays: {len(paper_parlays):,}")
    print(f"Saved parlay recommendations to: {output_path}")
    print(f"Saved parlay summary to: {summary_path}")
    print(f"Saved research parlay candidates to: {research_output_path}")
    print(f"Saved paper parlay candidates to: {paper_output_path}")
    print(f"Saved parlay research summary to: {research_summary_path}")


if __name__ == "__main__":
    main()
