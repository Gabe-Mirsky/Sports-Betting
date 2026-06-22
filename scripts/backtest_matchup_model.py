"""Walk-forward backtest of the no-odds matchup model.

Example
-------
    python scripts/backtest_matchup_model.py \
        --results-path data/processed/match_results.csv \
        --sport soccer \
        --output-dir data/reports

The backtest measures prediction quality (accuracy, log loss, Brier score,
calibration) only. It never computes ROI and never needs odds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.injuries_loader import load_injuries, normalize_injuries  # noqa: E402
from data.match_results_loader import (  # noqa: E402
    load_match_results,
    normalize_match_results,
)
from data.sport_rules import normalize_sport  # noqa: E402
from evaluation.backtest_matchup_model import (  # noqa: E402
    evaluate_probability_predictions,
    summarize_backtest_by_bucket,
    walk_forward_backtest,
)
from logging_setup import setup_logging  # noqa: E402
from reports.matchup_prediction_report import build_backtest_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the no-odds matchup model.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--injuries-path", default=None)
    parser.add_argument("--sport", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--aliases-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _print_summary(metrics: dict, buckets: pd.DataFrame) -> None:
    print("\n=== Matchup model backtest (model probabilities, no odds) ===")
    if metrics.get("n_games", 0) == 0:
        print(metrics.get("note", "No games were scored."))
        return
    print(f"Games scored:        {metrics['n_games']}")
    print(f"Accuracy:            {metrics['accuracy']:.1%}")
    print(f"Log loss:            {metrics['log_loss']:.4f} (lower is better)")
    print(f"Brier score:         {metrics['brier_score']:.4f} (lower is better)")
    print(f"Mean prob of actual: {metrics['mean_prob_of_actual_outcome']:.3f}")
    print(f"Favourite win rate:  {metrics['favorite_win_rate']:.1%}")

    by_sport = metrics.get("accuracy_by_sport", {})
    if by_sport:
        print("\nAccuracy by sport:")
        for sport, stats in sorted(by_sport.items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"  {sport:<16} {stats['accuracy']:.1%}  (n={stats['n']})")

    by_conf = metrics.get("accuracy_by_confidence", {})
    if by_conf:
        print("\nAccuracy by confidence level:")
        order = {"High": 0, "Medium": 1, "Low": 2, "Very low": 3}
        for level, stats in sorted(by_conf.items(), key=lambda kv: order.get(kv[0], 9)):
            print(f"  {level:<10} {stats['accuracy']:.1%}  (n={stats['n']})")

    if "draw_quality" in metrics:
        dq = metrics["draw_quality"]
        print(
            f"\nDraws: actual {dq['actual_draw_rate']:.1%} vs predicted "
            f"{dq['predicted_draw_rate']:.1%} of games."
        )

    if not buckets.empty:
        worst = buckets.iloc[buckets["calibration_gap"].abs().argmax()]
        print(
            f"\nCalibration: largest gap in bucket {worst['prob_bucket']} "
            f"(predicted {worst['mean_predicted_prob']:.2f} vs actual {worst['actual_win_rate']:.2f})."
        )
    print("\nReminder: these are model probabilities, not bet recommendations.\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = {"aliases_path": args.aliases_path} if args.aliases_path else {}
    if args.sport:
        config["default_sport"] = normalize_sport(args.sport)

    results = normalize_match_results(load_match_results(args.results_path), config)
    injuries = (
        normalize_injuries(load_injuries(args.injuries_path), config)
        if args.injuries_path
        else None
    )

    if args.league and "league" in results.columns:
        results = results[results["league"].astype(str).str.lower() == args.league.lower()].copy()

    if results.empty:
        print("No results to backtest after filtering.")
        return

    # One model family per sport (mixing sports in a single model is avoided).
    sports = [normalize_sport(args.sport)] if args.sport else sorted(results["sport"].unique())
    all_predictions: list[pd.DataFrame] = []
    for sport in sports:
        preds = walk_forward_backtest(results, injuries, sport=sport, config=config)
        if not preds.empty:
            all_predictions.append(preds)

    if not all_predictions:
        print("Backtest produced no predictions (not enough history).")
        return

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = evaluate_probability_predictions(predictions)
    buckets = summarize_backtest_by_bucket(predictions)

    build_backtest_report(metrics, buckets, Path(args.output_dir))
    _print_summary(metrics, buckets)
    print(f"Saved: {Path(args.output_dir) / 'matchup_model_backtest.json'}")
    print(f"Saved: {Path(args.output_dir) / 'matchup_model_backtest_by_bucket.csv'}")


if __name__ == "__main__":
    main()
