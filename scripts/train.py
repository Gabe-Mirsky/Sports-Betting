"""Train baseline NBA win-probability models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from models.calibration import calibration_curve_frame  # noqa: E402
from models.train_model import model_feature_diagnostics  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.train_model import train_and_save  # noqa: E402
from models.predict import predict_game_probabilities  # noqa: E402
from reports.plots import save_calibration_plot, save_probability_distribution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline NBA home-win models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modeling-path", default=None)
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
    model_output_path = PROJECT_ROOT / "data" / "models" / "home_win_model.joblib"
    metrics_output_path = PROJECT_ROOT / "data" / "reports" / "model_metrics.json"
    predictions_output_path = PROJECT_ROOT / "data" / "reports" / "model_predictions.csv"
    all_predictions_output_path = PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv"

    model_bundle, metrics, predictions = train_and_save(
        modeling_path=modeling_path,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        predictions_output_path=predictions_output_path,
        target_column=config.model.target,
        train_start_season=config.model.train_start_season,
        train_end_season=config.model.train_end_season,
        test_season=config.model.test_season,
        random_seed=config.project.random_seed,
    )

    calibration_path = PROJECT_ROOT / "data" / "reports" / "calibration_curve.csv"
    calibration_plot_path = PROJECT_ROOT / "data" / "reports" / "calibration_curve.png"
    probability_plot_path = PROJECT_ROOT / "data" / "reports" / "probability_distribution.png"
    feature_diagnostics_path = PROJECT_ROOT / "data" / "reports" / "model_feature_diagnostics.csv"

    calibration = calibration_curve_frame(
        predictions["actual_home_win"],
        predictions["model_home_win_prob"],
    )
    calibration.to_csv(calibration_path, index=False)
    save_calibration_plot(predictions, calibration_plot_path)
    save_probability_distribution(predictions, probability_plot_path)

    feature_diagnostics = model_feature_diagnostics(model_bundle)
    feature_diagnostics.to_csv(feature_diagnostics_path, index=False)

    modeling_df = pd.read_parquet(modeling_path)
    all_predictions = predict_game_probabilities(model_bundle, modeling_df)
    all_predictions["actual_home_win"] = modeling_df["target_home_win"].astype(int).to_numpy()
    all_predictions["split"] = "all_completed_games_final_model"
    all_predictions.to_csv(all_predictions_output_path, index=False)

    best_model = metrics["best_model"]
    best_metrics = metrics["models"][best_model]
    print(f"Trained models. Best model: {best_model}")
    print(
        "Test metrics: "
        f"accuracy={best_metrics['accuracy']:.3f}, "
        f"log_loss={best_metrics['log_loss']:.3f}, "
        f"brier={best_metrics['brier_score']:.3f}, "
        f"auc={best_metrics['roc_auc']:.3f}"
    )
    print(f"Saved model to: {model_output_path}")
    print(f"Saved metrics to: {metrics_output_path}")
    print(f"Saved {len(predictions):,} test predictions to: {predictions_output_path}")
    print(f"Saved {len(all_predictions):,} all-game predictions to: {all_predictions_output_path}")
    print(f"Saved calibration data to: {calibration_path}")
    print(f"Saved feature diagnostics to: {feature_diagnostics_path}")


if __name__ == "__main__":
    main()
