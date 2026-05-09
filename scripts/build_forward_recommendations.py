"""Build forward-looking paper recommendations for the local website."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from strategy.forward import build_forward_recommendations_from_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build forward paper recommendations from saved predictions and odds.")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--suggestions-path", default=None)
    parser.add_argument("--readiness-summary-path", default=None)
    parser.add_argument("--rule-sweep-summary-path", default=None)
    parser.add_argument("--rule-validation-summary-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--max-bet-fraction", type=float, default=None)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--ignore-readiness-gate", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    predictions_path = Path(args.predictions_path) if args.predictions_path else reports_dir / "upcoming_predictions.csv"
    suggestions_path = (
        Path(args.suggestions_path)
        if args.suggestions_path
        else reports_dir / "upcoming_market_suggestions.csv"
    )
    readiness_summary_path = (
        Path(args.readiness_summary_path)
        if args.readiness_summary_path
        else reports_dir / "strategy_readiness_summary.json"
    )
    rule_sweep_summary_path = (
        Path(args.rule_sweep_summary_path)
        if args.rule_sweep_summary_path
        else reports_dir / "signal_rule_sweep_summary.json"
    )
    rule_validation_summary_path = (
        Path(args.rule_validation_summary_path)
        if args.rule_validation_summary_path
        else reports_dir / "signal_rule_walk_forward_summary.json"
    )
    output_path = Path(args.output_path) if args.output_path else reports_dir / "forward_recommendations.csv"
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "forward_recommendations_summary.json"
    )
    bankroll = args.bankroll if args.bankroll is not None else config.strategy.starting_bankroll
    max_bet_fraction = (
        args.max_bet_fraction
        if args.max_bet_fraction is not None
        else config.strategy.max_bet_fraction
    )
    recommendations, summary = build_forward_recommendations_from_files(
        predictions_path=predictions_path,
        suggestions_path=suggestions_path,
        readiness_summary_path=readiness_summary_path,
        rule_sweep_summary_path=rule_sweep_summary_path,
        rule_validation_summary_path=rule_validation_summary_path,
        output_path=output_path,
        summary_path=summary_path,
        starting_bankroll=bankroll,
        max_bet_fraction=max_bet_fraction,
        respect_readiness_gate=not args.ignore_readiness_gate,
        as_of_date=args.as_of_date,
    )
    print(f"Forward games ({summary.get('timeline', 'n/a')}): {summary.get('games', 0):,}")
    print(f"Games with Kalshi odds: {summary.get('games_with_kalshi_odds', 0):,}")
    print(f"Edge signals: {summary.get('edge_signals', 0):,}")
    print(f"Recommended paper bets: {summary.get('paper_bets', 0):,}")
    print(f"Hypothetical edge-rule bets: {summary.get('hypothetical_paper_bets', 0):,}")
    print(f"Best sweep-rule passes: {summary.get('best_sweep_rule_passes', 0):,}")
    print(f"Rule validation status: {summary.get('rule_validation_status', 'n/a')}")
    print(f"Readiness gate: {summary.get('readiness_gate', 'n/a')}")
    print(f"Saved forward recommendations to: {output_path}")
    print(f"Saved forward summary to: {summary_path}")
    if recommendations.empty:
        print("No forward games were available in the saved prediction file.")


if __name__ == "__main__":
    main()
