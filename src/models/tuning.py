"""Time-aware model tuning for NBA home-win predictions."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.evaluate import evaluate_binary_probabilities
from models.predict import PREDICTION_ID_COLUMNS
from models.train_model import (
    BASELINE_FEATURE_COLUMNS,
    DEFAULT_FEATURE_COLUMNS,
    PLAYER_ROTATION_FEATURE_COLUMNS,
    RICH_TEAM_FORM_FEATURE_COLUMNS,
    _json_safe,
)
from data.seasons import (
    TEST_SEASON,
    TRAIN_END_SEASON,
    TRAIN_START_SEASON,
    VALIDATION_SEASON,
    assign_dataset_split,
)
from models.walk_forward import available_walk_forward_seasons


@dataclass(frozen=True)
class TuningCandidate:
    model_name: str
    feature_set_name: str
    feature_columns: list[str]
    model: Pipeline


def available_feature_sets(modeling_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return candidate feature sets with only columns present in the dataset."""

    def present(columns: list[str]) -> list[str]:
        return [column for column in columns if column in modeling_df.columns]

    baseline = present(BASELINE_FEATURE_COLUMNS)
    team_form = present(BASELINE_FEATURE_COLUMNS + RICH_TEAM_FORM_FEATURE_COLUMNS)
    player_rotation = present(BASELINE_FEATURE_COLUMNS + PLAYER_ROTATION_FEATURE_COLUMNS)
    full = present(DEFAULT_FEATURE_COLUMNS)
    return {
        "baseline": baseline,
        "team_form": team_form,
        "player_rotation": player_rotation,
        "full": full,
    }


def logistic_candidate(model_name: str, c_value: float, random_seed: int) -> Pipeline:
    """Create a calibrated-shape logistic candidate with a chosen regularization level."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1500, C=float(c_value), random_state=random_seed),
            ),
        ]
    )


def make_tuning_candidates(
    modeling_df: pd.DataFrame,
    random_seed: int = 42,
    logistic_c_values: list[float] | None = None,
) -> list[TuningCandidate]:
    """Build a compact, time-safe tuning grid."""

    logistic_c_values = logistic_c_values or [0.003, 0.006, 0.01, 0.02, 0.04, 0.08]
    feature_sets = available_feature_sets(modeling_df)
    candidates: list[TuningCandidate] = []
    for feature_set_name, feature_columns in feature_sets.items():
        if not feature_columns:
            continue
        for c_value in logistic_c_values:
            label = f"logistic_c_{c_value:g}"
            candidates.append(
                TuningCandidate(
                    model_name=label,
                    feature_set_name=feature_set_name,
                    feature_columns=feature_columns,
                    model=logistic_candidate(label, c_value, random_seed),
                )
            )
    return candidates


def _score_value(metrics: dict[str, Any], selection_metric: str) -> float:
    value = metrics.get(selection_metric)
    if value is None:
        return float("inf")
    numeric = float(value)
    if math.isnan(numeric):
        return float("inf")
    return numeric


def evaluate_tuning_candidate(
    modeling_df: pd.DataFrame,
    candidate: TuningCandidate,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one candidate with expanding walk-forward folds."""

    working = modeling_df.sort_values(["game_date", "game_id"]).copy()
    seasons = available_walk_forward_seasons(
        working,
        train_start_season=train_start_season,
        first_test_season=first_test_season,
        last_test_season=last_test_season,
    )
    prediction_frames: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    for test_season in seasons:
        train = working[(working["season"] >= train_start_season) & (working["season"] < test_season)].copy()
        test = working[working["season"] == test_season].copy()
        if train.empty or test.empty:
            continue
        model = clone(candidate.model)
        model.fit(train[candidate.feature_columns], train[target_column].astype(int))
        probabilities = model.predict_proba(test[candidate.feature_columns])[:, 1]
        y_test = test[target_column].astype(int)
        metrics = evaluate_binary_probabilities(y_test, probabilities)
        folds.append(
            {
                "test_season": int(test_season),
                "train_start_season": int(train_start_season),
                "train_end_season": int(test_season - 1),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "metrics": metrics,
            }
        )
        id_columns = [column for column in PREDICTION_ID_COLUMNS if column in test.columns]
        predictions = test[id_columns].copy()
        predictions["model_home_win_prob"] = probabilities
        predictions["model_away_win_prob"] = 1.0 - predictions["model_home_win_prob"]
        predictions["actual_home_win"] = y_test.to_numpy()
        predictions["split"] = "walk_forward_tuned"
        predictions["model_name"] = candidate.model_name
        predictions["feature_set"] = candidate.feature_set_name
        predictions["train_start_season"] = int(train_start_season)
        predictions["train_end_season"] = int(test_season - 1)
        prediction_frames.append(predictions)

    if not prediction_frames:
        raise ValueError(f"No tuning predictions were produced for {candidate.model_name}/{candidate.feature_set_name}.")
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = evaluate_binary_probabilities(
        all_predictions["actual_home_win"],
        all_predictions["model_home_win_prob"],
    )
    metrics_output = {
        "model_name": candidate.model_name,
        "feature_set": candidate.feature_set_name,
        "feature_count": int(len(candidate.feature_columns)),
        "features": candidate.feature_columns,
        "num_predictions": int(len(all_predictions)),
        "overall": overall,
        "folds": folds,
    }
    return all_predictions, metrics_output


def tune_home_win_model(
    modeling_df: pd.DataFrame,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    validation_season: int = VALIDATION_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    random_seed: int = 42,
    selection_metric: str = "log_loss",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Evaluate candidates and return results, best predictions, and best metrics."""

    if first_test_season is None:
        first_test_season = validation_season
    if last_test_season is None:
        last_test_season = validation_season

    candidates = make_tuning_candidates(modeling_df, random_seed=random_seed)
    rows: list[dict[str, Any]] = []
    best_predictions = pd.DataFrame()
    best_metrics: dict[str, Any] = {}
    best_score = float("inf")

    for candidate in candidates:
        predictions, metrics = evaluate_tuning_candidate(
            modeling_df,
            candidate,
            target_column=target_column,
            train_start_season=train_start_season,
            first_test_season=first_test_season,
            last_test_season=last_test_season,
        )
        overall = metrics["overall"]
        rows.append(
            {
                "model_name": candidate.model_name,
                "feature_set": candidate.feature_set_name,
                "feature_count": len(candidate.feature_columns),
                "num_predictions": metrics["num_predictions"],
                "accuracy": overall.get("accuracy"),
                "log_loss": overall.get("log_loss"),
                "brier_score": overall.get("brier_score"),
                "roc_auc": overall.get("roc_auc"),
            }
        )
        score = _score_value(overall, selection_metric)
        if score < best_score:
            best_score = score
            best_predictions = predictions
            best_metrics = metrics

    results = pd.DataFrame(rows).sort_values([selection_metric, "brier_score"], ascending=True).reset_index(drop=True)
    summary = {
        "selection_metric": selection_metric,
        "best_model_name": best_metrics.get("model_name"),
        "best_feature_set": best_metrics.get("feature_set"),
        "best_feature_count": best_metrics.get("feature_count"),
        "best_overall": best_metrics.get("overall", {}),
        "num_candidates": int(len(results)),
        "num_predictions": int(len(best_predictions)),
        "top_candidates": results.head(10).to_dict(orient="records"),
        "best_metrics": best_metrics,
    }
    return results, best_predictions, summary, pd.DataFrame(summary["top_candidates"])


def train_final_tuned_model(
    modeling_df: pd.DataFrame,
    summary: dict[str, Any],
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit the tuned best candidate on the training split only."""

    best_model_name = str(summary["best_model_name"])
    best_feature_set = str(summary["best_feature_set"])
    c_value = float(best_model_name.replace("logistic_c_", ""))
    feature_columns = available_feature_sets(modeling_df)[best_feature_set]
    split_df = assign_dataset_split(
        modeling_df,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
        test_season=test_season,
    )
    train = split_df[split_df["dataset_split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Training split is empty for tuned model final fit.")
    model = logistic_candidate(best_model_name, c_value, random_seed)
    model.fit(train[feature_columns], train[target_column].astype(int))
    return {
        "model": model,
        "model_name": best_model_name,
        "feature_set": best_feature_set,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "final_fit_rows": int(len(train)),
        "final_fit_split": "train",
        "tuning_summary": summary,
    }


def tune_and_save(
    modeling_path: str | Path,
    results_output_path: str | Path,
    summary_output_path: str | Path,
    predictions_output_path: str | Path,
    model_output_path: str | Path,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
    validation_season: int = VALIDATION_SEASON,
    test_season: int = TEST_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run tuning and save all artifacts."""

    modeling_df = pd.read_parquet(modeling_path)
    if first_test_season is None:
        first_test_season = validation_season
    if last_test_season is None:
        last_test_season = validation_season
    results, predictions, summary, _ = tune_home_win_model(
        modeling_df,
        target_column=target_column,
        train_start_season=train_start_season,
        validation_season=validation_season,
        first_test_season=first_test_season,
        last_test_season=last_test_season,
        random_seed=random_seed,
    )
    bundle = train_final_tuned_model(
        modeling_df,
        summary,
        target_column=target_column,
        train_start_season=train_start_season,
        train_end_season=train_end_season,
        validation_season=validation_season,
        test_season=test_season,
        random_seed=random_seed,
    )

    results_output = Path(results_output_path)
    summary_output = Path(summary_output_path)
    predictions_output = Path(predictions_output_path)
    model_output = Path(model_output_path)
    results_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_output, index=False)
    predictions.to_csv(predictions_output, index=False)
    summary_output.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    joblib.dump(bundle, model_output)
    return results, predictions, summary
