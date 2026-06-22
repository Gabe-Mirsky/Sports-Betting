"""Train baseline NBA win-probability models."""

from __future__ import annotations

import argparse
import json
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
from data.seasons import (  # noqa: E402
    assign_dataset_split,
    build_free_odds_split_plan,
    nba_season_display_label,
)
from data.sportsbook_odds import load_sportsbook_odds, sportsbook_match_report_by_season  # noqa: E402
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


def _sportsbook_adjusted_split(
    modeling_path: Path,
    configured_train_start: int,
    configured_train_end: int,
    configured_validation_season: int,
    split_mode: str,
    sportsbook_odds_path: Path,
    split_config_path: Path,
) -> tuple[int, int, int]:
    if split_config_path.exists():
        try:
            payload = json.loads(split_config_path.read_text(encoding="utf-8"))
            start_value = int(payload.get("train_start_season", configured_train_start))
            end_value = int(payload.get("train_end_season", configured_train_end))
            validation_value = int(payload.get("validation_season"))
            if configured_train_start <= start_value <= end_value <= configured_train_end:
                return start_value, end_value, validation_value
        except Exception as exc:
            print(f"WARNING: Could not read sportsbook split config {split_config_path}: {exc}")

    if not sportsbook_odds_path.exists():
        return configured_train_start, configured_train_end, configured_validation_season

    try:
        modeling_df = pd.read_parquet(modeling_path)
        odds = load_sportsbook_odds(sportsbook_odds_path)
        report = sportsbook_match_report_by_season(modeling_df, odds)
    except Exception as exc:
        print(f"WARNING: Could not infer sportsbook-adjusted split: {exc}")
        return configured_train_start, configured_train_end, configured_validation_season

    plan = build_free_odds_split_plan(report, mode=split_mode)
    train_seasons = [int(season) for season in plan["train_seasons"]]
    if not train_seasons or plan["validation_season"] is None:
        return configured_train_start, configured_train_end, configured_validation_season
    return min(train_seasons), max(train_seasons), int(plan["validation_season"])


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
    sportsbook_odds_path = resolve_project_path(config.data.sportsbook_odds_path)
    split_config_path = PROJECT_ROOT / "data" / "processed" / "sportsbook_split_config.json"
    train_start_season, train_end_season, validation_season = _sportsbook_adjusted_split(
        modeling_path=modeling_path,
        configured_train_start=config.model.train_start_season,
        configured_train_end=config.model.train_end_season,
        configured_validation_season=config.model.validation_season,
        split_mode=config.data.free_odds_split_mode,
        sportsbook_odds_path=sportsbook_odds_path,
        split_config_path=split_config_path,
    )
    if (
        train_start_season != config.model.train_start_season
        or train_end_season != config.model.train_end_season
        or validation_season != config.model.validation_season
    ):
        print(
            "Adjusted training seasons from free Kaggle sportsbook coverage: "
            f"{nba_season_display_label(train_start_season)} through {nba_season_display_label(train_end_season)}. "
            f"Validation: {nba_season_display_label(validation_season)}. "
            "Seasons without usable sportsbook odds are excluded from train/validation."
        )

    model_bundle, metrics, predictions = train_and_save(
        modeling_path=modeling_path,
        model_output_path=model_output_path,
        metrics_output_path=metrics_output_path,
        predictions_output_path=predictions_output_path,
        target_column=config.model.target,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
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
    modeling_df = assign_dataset_split(
        modeling_df,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
        test_season=config.model.test_season,
        allow_outside=True,
    )
    all_predictions = predict_game_probabilities(model_bundle, modeling_df)
    all_predictions["actual_home_win"] = modeling_df["target_home_win"].astype(int).to_numpy()
    if "dataset_split" in modeling_df.columns:
        all_predictions["dataset_split"] = modeling_df["dataset_split"].to_numpy()
    all_predictions["split"] = "all_completed_games_train_fit_model"
    all_predictions.to_csv(all_predictions_output_path, index=False)

    best_model = metrics["best_model"]
    validation_metrics = metrics["models"][best_model]
    best_metrics = metrics["final_test"][best_model]
    print(f"Trained models. Best model: {best_model}")
    print(
        "Validation metrics used for model selection: "
        f"accuracy={validation_metrics['accuracy']:.3f}, "
        f"log_loss={validation_metrics['log_loss']:.3f}, "
        f"brier={validation_metrics['brier_score']:.3f}, "
        f"auc={validation_metrics['roc_auc']:.3f}"
    )
    print(
        "Final test metrics: "
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
