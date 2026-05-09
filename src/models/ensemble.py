"""Home-win ensemble diagnostics from saved walk-forward predictions."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.evaluate import evaluate_binary_probabilities
from models.train_model import _json_safe


DEFAULT_PROBABILITY_COLUMNS = [
    "base_home_win_prob",
    "tuned_home_win_prob",
    "margin_home_win_prob",
]


def prepare_home_win_ensemble_frame(
    base_predictions: pd.DataFrame,
    tuned_predictions: pd.DataFrame,
    margin_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge model, tuned model, and margin-derived home-win probabilities."""

    required_base = {
        "game_id",
        "game_date",
        "season",
        "season_type",
        "home_team_abbr",
        "away_team_abbr",
        "model_home_win_prob",
        "actual_home_win",
    }
    missing_base = sorted(required_base - set(base_predictions.columns))
    if missing_base:
        raise ValueError(f"Base predictions are missing columns: {missing_base}")
    if "model_home_win_prob" not in tuned_predictions.columns:
        raise ValueError("Tuned predictions are missing model_home_win_prob.")
    if "prob_home_win_from_margin" not in margin_predictions.columns:
        raise ValueError("Margin predictions are missing prob_home_win_from_margin.")

    base = base_predictions[list(required_base)].copy()
    base["game_id"] = base["game_id"].astype(str)
    base = base.rename(columns={"model_home_win_prob": "base_home_win_prob"})

    tuned = tuned_predictions[["game_id", "model_home_win_prob"]].copy()
    tuned["game_id"] = tuned["game_id"].astype(str)
    tuned = tuned.rename(columns={"model_home_win_prob": "tuned_home_win_prob"})

    margin = margin_predictions[["game_id", "prob_home_win_from_margin"]].copy()
    margin["game_id"] = margin["game_id"].astype(str)
    margin = margin.rename(columns={"prob_home_win_from_margin": "margin_home_win_prob"})

    frame = base.merge(tuned, on="game_id", how="inner").merge(margin, on="game_id", how="inner")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["actual_home_win"] = pd.to_numeric(frame["actual_home_win"], errors="coerce")
    for column in DEFAULT_PROBABILITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["game_date", "season", "actual_home_win", *DEFAULT_PROBABILITY_COLUMNS]
    ).copy()
    frame["season"] = frame["season"].astype(int)
    frame["actual_home_win"] = frame["actual_home_win"].astype(int)
    return frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def simplex_weight_grid(n_models: int, step: float = 0.1) -> list[tuple[float, ...]]:
    """Return non-negative weights that sum to one."""

    if n_models < 1:
        raise ValueError("n_models must be positive.")
    units = int(round(1.0 / step))
    weights: list[tuple[float, ...]] = []
    for values in product(range(units + 1), repeat=n_models - 1):
        remaining = units - sum(values)
        if remaining < 0:
            continue
        full = (*values, remaining)
        weights.append(tuple(round(item / units, 10) for item in full))
    return weights


def apply_weights(frame: pd.DataFrame, probability_columns: list[str], weights: tuple[float, ...]) -> pd.Series:
    """Return weighted-average probabilities."""

    if len(probability_columns) != len(weights):
        raise ValueError("Number of probability columns must match number of weights.")
    values = np.zeros(len(frame), dtype=float)
    for column, weight in zip(probability_columns, weights):
        values += float(weight) * frame[column].to_numpy(dtype=float)
    return pd.Series(np.clip(values, 1e-6, 1.0 - 1e-6), index=frame.index)


def choose_best_weights(
    train: pd.DataFrame,
    probability_columns: list[str] | None = None,
    step: float = 0.1,
    selection_metric: str = "log_loss",
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Choose weights on a training frame."""

    probability_columns = probability_columns or DEFAULT_PROBABILITY_COLUMNS
    if train.empty:
        weights = tuple(1.0 / len(probability_columns) for _ in probability_columns)
        return weights, {"reason": "no_prior_rows", "rows": 0}

    best_weights: tuple[float, ...] | None = None
    best_metrics: dict[str, Any] = {}
    best_score = float("inf")
    for weights in simplex_weight_grid(len(probability_columns), step=step):
        probabilities = apply_weights(train, probability_columns, weights)
        metrics = evaluate_binary_probabilities(train["actual_home_win"], probabilities)
        score = float(metrics.get(selection_metric, float("inf")))
        if np.isnan(score):
            score = float("inf")
        if score < best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics
    if best_weights is None:
        best_weights = tuple(1.0 / len(probability_columns) for _ in probability_columns)
        best_metrics = {"reason": "no_valid_weight_grid"}
    best_metrics["rows"] = int(len(train))
    return best_weights, best_metrics


def build_home_win_ensemble(
    ensemble_frame: pd.DataFrame,
    probability_columns: list[str] | None = None,
    weight_step: float = 0.1,
    min_train_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build expanding-season ensemble predictions and metrics."""

    if ensemble_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0}
    probability_columns = probability_columns or DEFAULT_PROBABILITY_COLUMNS
    frame = ensemble_frame.sort_values(["game_date", "game_id"]).reset_index(drop=True).copy()

    prediction_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    equal_weights = tuple(1.0 / len(probability_columns) for _ in probability_columns)
    for season in sorted(frame["season"].dropna().unique()):
        test = frame[frame["season"].eq(season)].copy()
        train = frame[frame["season"].lt(season)].copy()
        if len(train) >= min_train_rows:
            weights, train_metrics = choose_best_weights(
                train,
                probability_columns=probability_columns,
                step=weight_step,
            )
            weight_source = "prior_seasons_grid_search"
        else:
            weights = equal_weights
            train_metrics = {"reason": "warmup_equal_weights", "rows": int(len(train))}
            weight_source = "warmup_equal_weights"

        probabilities = apply_weights(test, probability_columns, weights)
        output = test[
            [
                "game_id",
                "game_date",
                "season",
                "season_type",
                "home_team_abbr",
                "away_team_abbr",
                "actual_home_win",
                *probability_columns,
            ]
        ].copy()
        output["ensemble_home_win_prob"] = probabilities
        output["model_home_win_prob"] = output["ensemble_home_win_prob"]
        output["model_away_win_prob"] = 1.0 - output["model_home_win_prob"]
        output["split"] = "walk_forward_home_win_ensemble"
        output["ensemble_weight_source"] = weight_source
        for column, weight in zip(probability_columns, weights):
            output[f"weight_{column}"] = float(weight)
        prediction_frames.append(output)

        fold_metrics = evaluate_binary_probabilities(test["actual_home_win"], probabilities)
        weight_row = {
            "season": int(season),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "weight_source": weight_source,
            "train_selection_log_loss": train_metrics.get("log_loss"),
            "test_log_loss": fold_metrics.get("log_loss"),
            "test_brier_score": fold_metrics.get("brier_score"),
            "test_accuracy": fold_metrics.get("accuracy"),
            "test_roc_auc": fold_metrics.get("roc_auc"),
        }
        for column, weight in zip(probability_columns, weights):
            weight_row[f"weight_{column}"] = float(weight)
        weight_rows.append(weight_row)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    weights_df = pd.DataFrame(weight_rows)
    component_metrics = {
        column: evaluate_binary_probabilities(frame["actual_home_win"], frame[column])
        for column in probability_columns
    }
    ensemble_metrics = evaluate_binary_probabilities(
        predictions["actual_home_win"],
        predictions["ensemble_home_win_prob"],
    ) if not predictions.empty else {}
    best_component_name = min(
        component_metrics,
        key=lambda name: float(component_metrics[name].get("log_loss", float("inf"))),
    )
    best_component_metrics = component_metrics[best_component_name]
    log_loss_delta_vs_best = (
        float(ensemble_metrics.get("log_loss", float("nan")))
        - float(best_component_metrics.get("log_loss", float("nan")))
    )
    summary = {
        "rows": int(len(predictions)),
        "timeline": _timeline(predictions),
        "probability_columns": probability_columns,
        "weight_step": float(weight_step),
        "min_train_rows": int(min_train_rows),
        "ensemble": ensemble_metrics,
        "components": component_metrics,
        "best_component": best_component_name,
        "log_loss_delta_vs_best_component": log_loss_delta_vs_best,
        "adoption_status": "candidate" if log_loss_delta_vs_best < -0.001 else "research_only",
        "note": "Ensemble weights are selected on prior seasons only. Research-only unless it beats the best component by a meaningful margin.",
    }
    return predictions, weights_df, summary


def build_static_blend_audit(
    ensemble_frame: pd.DataFrame,
    probability_columns: list[str] | None = None,
    weight_step: float = 0.1,
) -> pd.DataFrame:
    """Evaluate fixed weights on the merged out-of-sample frame for audit only."""

    probability_columns = probability_columns or DEFAULT_PROBABILITY_COLUMNS
    rows: list[dict[str, Any]] = []
    for weights in simplex_weight_grid(len(probability_columns), step=weight_step):
        probabilities = apply_weights(ensemble_frame, probability_columns, weights)
        metrics = evaluate_binary_probabilities(ensemble_frame["actual_home_win"], probabilities)
        row = {
            "rows": int(len(ensemble_frame)),
            "accuracy": metrics.get("accuracy"),
            "brier_score": metrics.get("brier_score"),
            "log_loss": metrics.get("log_loss"),
            "roc_auc": metrics.get("roc_auc"),
            "note": "Fixed-weight audit uses the full out-of-sample frame for selection; do not treat it as a deployment score.",
        }
        for column, weight in zip(probability_columns, weights):
            row[f"weight_{column}"] = float(weight)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)


def _timeline(predictions: pd.DataFrame) -> str:
    if predictions.empty or "game_date" not in predictions.columns:
        return "n/a"
    dates = pd.to_datetime(predictions["game_date"], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    return start if start == end else f"{start} to {end}"


def save_home_win_ensemble_outputs(
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    static_audit: pd.DataFrame,
    summary: dict[str, Any],
    predictions_path: str | Path,
    weights_path: str | Path,
    static_audit_path: str | Path,
    summary_path: str | Path,
) -> None:
    predictions_output = Path(predictions_path)
    weights_output = Path(weights_path)
    static_output = Path(static_audit_path)
    summary_output = Path(summary_path)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    static_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_output, index=False)
    weights.to_csv(weights_output, index=False)
    static_audit.to_csv(static_output, index=False)
    summary_output.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
