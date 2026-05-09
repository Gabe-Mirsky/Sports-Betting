"""Train spread and total-points model engines."""

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
from models.market_type_models import train_market_type_models_and_save  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA spread and total-points prediction engines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--games-path", default=None)
    parser.add_argument("--first-test-season", type=int, default=None)
    parser.add_argument("--last-test-season", type=int, default=None)
    parser.add_argument("--model-type", default="ridge", choices=["ridge", "hist_gradient_boosting"])
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
    games_path = (
        Path(args.games_path)
        if args.games_path
        else PROJECT_ROOT / "data" / "interim" / "nba_games.parquet"
    )
    model_output_path = PROJECT_ROOT / "data" / "models" / "margin_total_model.joblib"
    predictions_output_path = PROJECT_ROOT / "data" / "reports" / "market_type_predictions.csv"
    metrics_output_path = PROJECT_ROOT / "data" / "reports" / "market_type_model_metrics.json"
    calibration_output_path = PROJECT_ROOT / "data" / "reports" / "market_type_probability_calibration.csv"
    calibration_summary_output_path = PROJECT_ROOT / "data" / "reports" / "market_type_calibration_summary.json"

    _, predictions, metrics = train_market_type_models_and_save(
        modeling_path=modeling_path,
        games_path=games_path,
        model_output_path=model_output_path,
        predictions_output_path=predictions_output_path,
        metrics_output_path=metrics_output_path,
        calibration_output_path=calibration_output_path,
        calibration_summary_output_path=calibration_summary_output_path,
        train_start_season=config.model.train_start_season,
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        model_type=args.model_type,
        random_seed=config.project.random_seed,
    )

    margin = metrics.get("overall", {}).get("margin", {})
    total = metrics.get("overall", {}).get("total", {})
    print(
        "Market-type models: "
        f"rows={len(predictions):,}, "
        f"margin_mae={margin.get('mae', float('nan')):.2f}, "
        f"margin_rmse={margin.get('rmse', float('nan')):.2f}, "
        f"total_mae={total.get('mae', float('nan')):.2f}, "
        f"total_rmse={total.get('rmse', float('nan')):.2f}"
    )
    print(f"Saved model bundle to: {model_output_path}")
    print(f"Saved predictions to: {predictions_output_path}")
    print(f"Saved metrics to: {metrics_output_path}")
    print(f"Saved calibration to: {calibration_output_path}")


if __name__ == "__main__":
    main()
