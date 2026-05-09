"""Run walk-forward out-of-sample season predictions."""

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
from models.calibration import calibration_curve_frame  # noqa: E402
from models.walk_forward import walk_forward_and_save  # noqa: E402
from reports.plots import save_calibration_plot, save_probability_distribution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward NBA predictions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--first-test-season", type=int, default=None)
    parser.add_argument("--last-test-season", type=int, default=None)
    parser.add_argument("--model-type", default="logistic_regression")
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
    predictions_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"
    metrics_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_metrics.json"

    predictions, metrics = walk_forward_and_save(
        modeling_path=modeling_path,
        predictions_output_path=predictions_path,
        metrics_output_path=metrics_path,
        target_column=config.model.target,
        train_start_season=config.model.train_start_season,
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        model_type=args.model_type,
        random_seed=config.project.random_seed,
    )

    calibration_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_calibration_curve.csv"
    calibration_plot_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_calibration_curve.png"
    probability_plot_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_probability_distribution.png"

    calibration = calibration_curve_frame(
        predictions["actual_home_win"],
        predictions["model_home_win_prob"],
    )
    calibration.to_csv(calibration_path, index=False)
    save_calibration_plot(predictions, calibration_plot_path)
    save_probability_distribution(predictions, probability_plot_path)

    overall = metrics["overall"]["model"]
    print(
        "Walk-forward results: "
        f"rows={len(predictions):,}, "
        f"accuracy={overall['accuracy']:.3f}, "
        f"log_loss={overall['log_loss']:.3f}, "
        f"brier={overall['brier_score']:.3f}, "
        f"auc={overall['roc_auc']:.3f}"
    )
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved calibration data to: {calibration_path}")


if __name__ == "__main__":
    main()
