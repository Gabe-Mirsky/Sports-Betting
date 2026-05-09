"""Tune home-win model families with walk-forward validation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.tuning import tune_and_save  # noqa: E402
from reports.plots import save_calibration_plot, save_probability_distribution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune NBA home-win model settings with walk-forward validation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--first-test-season", type=int, default=None)
    parser.add_argument("--last-test-season", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    modeling_path = (
        Path(args.modeling_path)
        if args.modeling_path
        else resolve_project_path(config.data.processed_dir) / "modeling_dataset.parquet"
    )
    reports_dir = PROJECT_ROOT / "data" / "reports"
    results_path = reports_dir / "model_tuning_results.csv"
    summary_path = reports_dir / "model_tuning_summary.json"
    predictions_path = reports_dir / "tuned_walk_forward_predictions.csv"
    model_output_path = PROJECT_ROOT / "data" / "models" / "home_win_model_tuned.joblib"

    results, predictions, summary = tune_and_save(
        modeling_path=modeling_path,
        results_output_path=results_path,
        summary_output_path=summary_path,
        predictions_output_path=predictions_path,
        model_output_path=model_output_path,
        target_column=config.model.target,
        train_start_season=config.model.train_start_season,
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        random_seed=config.project.random_seed,
    )
    save_calibration_plot(
        predictions,
        reports_dir / "tuned_walk_forward_calibration_curve.png",
        probability_column="model_home_win_prob",
        target_column="actual_home_win",
    )
    save_probability_distribution(
        predictions,
        reports_dir / "tuned_walk_forward_probability_distribution.png",
        probability_column="model_home_win_prob",
    )

    best = summary.get("best_overall", {})
    print(
        "Best tuned model: "
        f"{summary.get('best_model_name')} / {summary.get('best_feature_set')} "
        f"with log_loss={best.get('log_loss', float('nan')):.4f}, "
        f"brier={best.get('brier_score', float('nan')):.4f}, "
        f"auc={best.get('roc_auc', float('nan')):.4f}"
    )
    print(f"Candidates tested: {len(results):,}")
    print(f"Saved tuning results to: {results_path}")
    print(f"Saved tuned predictions to: {predictions_path}")
    print(f"Saved tuned model to: {model_output_path}")


if __name__ == "__main__":
    main()
