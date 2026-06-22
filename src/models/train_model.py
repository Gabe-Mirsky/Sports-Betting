"""Train baseline NBA home-win probability models."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.seasons import (
    TEST_SEASON,
    TRAIN_END_SEASON,
    TRAIN_START_SEASON,
    VALIDATION_SEASON,
    assign_dataset_split,
)
from models.evaluate import evaluate_binary_probabilities
from models.predict import predict_game_probabilities


logger = logging.getLogger(__name__)


BASELINE_FEATURE_COLUMNS = [
    "elo_diff_pre",
    "elo_home_win_prob",
    "rest_diff",
    "home_is_back_to_back",
    "away_is_back_to_back",
    "last_5_win_pct_diff",
    "last_10_win_pct_diff",
    "last_5_point_diff_diff",
    "last_10_point_diff_diff",
    "season_win_pct_diff",
    "season_avg_margin_diff",
]

RICH_TEAM_FORM_FEATURE_COLUMNS = [
    "neutral_site",
    "is_playoffs",
    "last_5_fg_pct_diff",
    "last_10_fg_pct_diff",
    "last_5_fg3_pct_diff",
    "last_10_fg3_pct_diff",
    "last_5_ft_pct_diff",
    "last_10_ft_pct_diff",
    "last_5_reb_diff",
    "last_10_reb_diff",
    "last_5_ast_diff",
    "last_10_ast_diff",
    "last_5_tov_diff",
    "last_10_tov_diff",
]

PLAYER_ROTATION_FEATURE_COLUMNS = [
    "player_prior_games_last10_diff",
    "player_top3_minutes_last10_diff",
    "player_top5_minutes_last10_diff",
    "player_top8_minutes_last10_diff",
    "player_top8_points_last10_diff",
    "player_top8_reb_last10_diff",
    "player_top8_ast_last10_diff",
    "player_top8_stock_last10_diff",
    "player_top8_tov_last10_diff",
    "player_top8_plus_minus_last10_diff",
    "player_top8_value_last10_diff",
    "player_top8_games_played_share_last10_diff",
    "player_active_count_last5_diff",
    "player_rotation_continuity_last5_diff",
    "player_top_player_minutes_last10_diff",
    "player_top_player_days_since_seen_diff",
    "player_top3_available_last_game_share_diff",
    "player_top8_available_last_game_share_diff",
    "player_top3_minutes_last_game_diff",
    "player_top8_minutes_last_game_diff",
    "player_top8_value_last_game_diff",
    "player_key_absence_minutes_last_game_diff",
    "player_top8_minutes_gap_last_game_diff",
]

AVAILABILITY_FEATURE_COLUMNS = [
    "availability_report_present_diff",
    "availability_reported_players_diff",
    "availability_players_out_diff",
    "availability_players_doubtful_diff",
    "availability_players_questionable_diff",
    "availability_players_probable_diff",
    "availability_players_available_diff",
    "availability_players_unknown_diff",
    "availability_out_or_doubtful_diff",
    "availability_questionable_or_worse_diff",
    "availability_out_weighted_diff",
    "availability_doubtful_weighted_diff",
    "availability_questionable_weighted_diff",
    "availability_questionable_or_worse_weighted_diff",
    "availability_status_severity_weighted_diff",
    "availability_projected_minutes_lost_diff",
    "home_availability_report_present",
    "away_availability_report_present",
    "home_availability_players_out",
    "away_availability_players_out",
    "home_availability_out_or_doubtful",
    "away_availability_out_or_doubtful",
    "home_availability_questionable_or_worse",
    "away_availability_questionable_or_worse",
    "home_availability_projected_minutes_lost",
    "away_availability_projected_minutes_lost",
    "home_availability_status_severity_weighted",
    "away_availability_status_severity_weighted",
]

DEFAULT_FEATURE_COLUMNS = (
    BASELINE_FEATURE_COLUMNS
    + RICH_TEAM_FORM_FEATURE_COLUMNS
    + PLAYER_ROTATION_FEATURE_COLUMNS
    + AVAILABILITY_FEATURE_COLUMNS
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def available_feature_columns(
    modeling_df: pd.DataFrame,
    requested_columns: list[str] | None = None,
) -> list[str]:
    """Return usable feature columns from the modeling dataset."""

    if requested_columns is None:
        missing_baseline = [
            column for column in BASELINE_FEATURE_COLUMNS if column not in modeling_df.columns
        ]
        if missing_baseline:
            raise ValueError(f"Modeling dataset is missing baseline feature columns: {missing_baseline}")
        return [column for column in DEFAULT_FEATURE_COLUMNS if column in modeling_df.columns]

    missing = [column for column in requested_columns if column not in modeling_df.columns]
    if missing:
        raise ValueError(f"Modeling dataset is missing feature columns: {missing}")
    return requested_columns


def make_model_candidates(random_seed: int = 42) -> dict[str, Any]:
    """Create the first baseline model candidates."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, C=0.01, random_state=random_seed),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        random_state=random_seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.04,
                        l2_regularization=0.01,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }


def time_based_split(
    modeling_df: pd.DataFrame,
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split games by fixed chronological seasons without shuffling."""

    working = assign_dataset_split(
        modeling_df,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
        test_season=test_season,
        allow_outside=True,
    ).sort_values(["game_date", "game_id"]).copy()
    train = working[working["dataset_split"].eq("train")].copy()
    validation = working[working["dataset_split"].eq("validation")].copy()
    test = working[working["dataset_split"].eq("test")].copy()

    if train.empty:
        raise ValueError("Training split is empty. Check train_start_season/train_end_season.")
    if validation.empty:
        raise ValueError(f"Validation split is empty for season {validation_season}.")
    if test.empty:
        raise ValueError(f"Test split is empty for season {test_season}.")

    train_seasons = set(pd.to_numeric(train["season"], errors="coerce").dropna().astype(int))
    forbidden = {validation_season, test_season}
    leaked = sorted(train_seasons & forbidden)
    if leaked:
        raise ValueError(f"Training data contains held-out seasons: {leaked}")

    return train, validation, test


def train_models(
    modeling_df: pd.DataFrame,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Train candidate models and return the best model bundle plus metrics."""

    if target_column not in modeling_df.columns and target_column == "home_team_win":
        target_column = "target_home_win"
    if target_column not in modeling_df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    feature_columns = available_feature_columns(modeling_df, feature_columns)
    train, validation, test = time_based_split(
        modeling_df,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
        test_season=test_season,
    )

    print(
        "Dataset split counts before training: "
        f"train={len(train):,}, validation={len(validation):,}, test={len(test):,}"
    )
    print(
        "Leakage checks passed: no "
        f"{validation_season}-{(validation_season + 1) % 100:02d} or "
        f"{test_season}-{(test_season + 1) % 100:02d} games appear in training."
    )

    x_train = train[feature_columns]
    y_train = train[target_column].astype(int)
    x_validation = validation[feature_columns]
    y_validation = validation[target_column].astype(int)
    x_test = test[feature_columns]
    y_test = test[target_column].astype(int)

    metrics: dict[str, Any] = {
        "split": {
            "train_start_season": train_start_season,
            "train_end_season": train_end_season,
            "validation_season": validation_season,
            "test_season": test_season,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "features": feature_columns,
        "models": {},
        "final_test": {},
    }

    metrics["models"]["elo_baseline"] = evaluate_binary_probabilities(
        y_validation,
        validation["elo_home_win_prob"],
    )
    metrics["final_test"]["elo_baseline"] = evaluate_binary_probabilities(y_test, test["elo_home_win_prob"])

    candidates = make_model_candidates(random_seed=random_seed)
    trained_models: dict[str, Any] = {}
    for name, model in candidates.items():
        logger.info("Training %s on %s games", name, len(train))
        try:
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            trained_models[name] = model
            metrics["models"][name] = evaluate_binary_probabilities(y_validation, probabilities)
        except Exception as exc:
            logger.warning("Skipping %s because training failed: %s", name, exc)
            metrics["models"][name] = {"error": str(exc)}

    if not trained_models:
        raise RuntimeError("No sklearn model candidates trained successfully.")

    def score_for_selection(model_name: str) -> float:
        value = metrics["models"][model_name].get("log_loss")
        if value is None or math.isnan(float(value)):
            return float("inf")
        return float(value)

    best_model_name = min(trained_models, key=score_for_selection)
    metrics["best_model"] = best_model_name
    test_probabilities = trained_models[best_model_name].predict_proba(x_test)[:, 1]
    metrics["final_test"][best_model_name] = evaluate_binary_probabilities(y_test, test_probabilities)
    predictions = predict_game_probabilities(
        {
            "model": trained_models[best_model_name],
            "feature_columns": feature_columns,
        },
        test,
    )
    predictions["actual_home_win"] = y_test.to_numpy()
    predictions["split"] = "test"
    predictions["dataset_split"] = "test"

    final_model = clone(trained_models[best_model_name])
    final_model.fit(x_train, y_train)
    metrics["final_fit_rows"] = int(len(train))
    metrics["final_fit_split"] = "train"

    best_bundle = {
        "model": final_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "final_fit_rows": int(len(train)),
        "final_fit_split": "train",
    }
    return best_bundle, metrics, predictions


def train_and_save(
    modeling_path: str | Path,
    model_output_path: str | Path,
    metrics_output_path: str | Path,
    predictions_output_path: str | Path,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    random_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Load modeling data, train candidates, and save model artifacts."""

    modeling_df = pd.read_parquet(modeling_path)
    best_bundle, metrics, predictions = train_models(
        modeling_df,
        target_column=target_column,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
        test_season=test_season,
        random_seed=random_seed,
    )

    model_output_path = Path(model_output_path)
    metrics_output_path = Path(metrics_output_path)
    predictions_output_path = Path(predictions_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_bundle, model_output_path)
    metrics_output_path.write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(predictions_output_path, index=False)
    return best_bundle, metrics, predictions


def model_feature_diagnostics(model_bundle: dict[str, Any]) -> pd.DataFrame:
    """Return feature coefficients/importances for models that expose them."""

    model = model_bundle["model"]
    model_name = model_bundle["model_name"]
    feature_columns = list(model_bundle["feature_columns"])
    final_estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model

    if hasattr(final_estimator, "coef_"):
        values = final_estimator.coef_[0]
        metric_name = "coefficient"
    elif hasattr(final_estimator, "feature_importances_"):
        values = final_estimator.feature_importances_
        metric_name = "importance"
    else:
        return pd.DataFrame(columns=["model_name", "feature", "metric", "value", "abs_value"])

    diagnostics = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": feature_columns,
            "metric": metric_name,
            "value": values,
        }
    )
    diagnostics["abs_value"] = diagnostics["value"].abs()
    return diagnostics.sort_values("abs_value", ascending=False).reset_index(drop=True)
