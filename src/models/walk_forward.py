"""Walk-forward out-of-sample model evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.seasons import TRAIN_START_SEASON
from models.evaluate import evaluate_binary_probabilities
from models.predict import PREDICTION_ID_COLUMNS
from models.train_model import (
    _json_safe,
    available_feature_columns,
    make_model_candidates,
)


logger = logging.getLogger(__name__)


def available_walk_forward_seasons(
    modeling_df: pd.DataFrame,
    train_start_season: int,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
) -> list[int]:
    """Return seasons that can be predicted with prior training data."""

    seasons = sorted(int(season) for season in modeling_df["season"].dropna().unique())
    if not seasons:
        return []

    minimum_test_season = max(train_start_season + 1, min(seasons) + 1)
    first = first_test_season or minimum_test_season
    last = last_test_season or max(seasons)
    return [
        season
        for season in seasons
        if first <= season <= last and any(train_start_season <= prior < season for prior in seasons)
    ]


def _train_single_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_type: str,
    random_seed: int,
) -> Any:
    candidates = make_model_candidates(random_seed=random_seed)
    if model_type not in candidates:
        raise ValueError(
            f"Unknown model_type={model_type!r}. Choose one of: {sorted(candidates)}"
        )
    model = candidates[model_type]
    model.fit(train[feature_columns], train[target_column].astype(int))
    return model


def walk_forward_predict(
    modeling_df: pd.DataFrame,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    model_type: str = "logistic_regression",
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on past seasons, predict each next season, and combine results."""

    if target_column not in modeling_df.columns and target_column == "home_team_win":
        target_column = "target_home_win"
    if target_column not in modeling_df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    feature_columns = available_feature_columns(
        modeling_df,
        feature_columns,
    )
    working = modeling_df.sort_values(["game_date", "game_id"]).copy()
    seasons = available_walk_forward_seasons(
        working,
        train_start_season=train_start_season,
        first_test_season=first_test_season,
        last_test_season=last_test_season,
    )
    if not seasons:
        raise ValueError("No walk-forward test seasons are available.")

    prediction_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for test_season in seasons:
        train = working[
            (working["season"] >= train_start_season) & (working["season"] < test_season)
        ].copy()
        test = working[working["season"] == test_season].copy()
        if train.empty or test.empty:
            logger.warning(
                "Skipping season %s because train_rows=%s test_rows=%s",
                test_season,
                len(train),
                len(test),
            )
            continue

        logger.info(
            "Walk-forward fold: train %s-%s, predict %s",
            train_start_season,
            test_season - 1,
            test_season,
        )
        model = _train_single_model(
            train,
            feature_columns=feature_columns,
            target_column=target_column,
            model_type=model_type,
            random_seed=random_seed,
        )
        probabilities = model.predict_proba(test[feature_columns])[:, 1]
        y_test = test[target_column].astype(int)
        metrics = evaluate_binary_probabilities(y_test, probabilities)
        elo_metrics = evaluate_binary_probabilities(y_test, test["elo_home_win_prob"])

        fold_metrics.append(
            {
                "test_season": int(test_season),
                "train_start_season": int(train_start_season),
                "train_end_season": int(test_season - 1),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "model": metrics,
                "elo_baseline": elo_metrics,
            }
        )

        id_columns = [column for column in PREDICTION_ID_COLUMNS if column in test.columns]
        predictions = test[id_columns].copy()
        predictions["model_home_win_prob"] = probabilities
        predictions["model_away_win_prob"] = 1.0 - predictions["model_home_win_prob"]
        predictions["actual_home_win"] = y_test.to_numpy()
        predictions["split"] = "walk_forward"
        predictions["train_start_season"] = int(train_start_season)
        predictions["train_end_season"] = int(test_season - 1)
        prediction_frames.append(predictions)

    if not prediction_frames:
        raise ValueError("No walk-forward predictions were produced.")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    overall_metrics = evaluate_binary_probabilities(
        all_predictions["actual_home_win"],
        all_predictions["model_home_win_prob"],
    )
    overall_elo_metrics = evaluate_binary_probabilities(
        all_predictions["actual_home_win"],
        working.loc[working["game_id"].isin(all_predictions["game_id"]), "elo_home_win_prob"],
    )

    metrics_output = {
        "model_type": model_type,
        "target_column": target_column,
        "features": feature_columns,
        "first_test_season": int(min(seasons)),
        "last_test_season": int(max(seasons)),
        "num_predictions": int(len(all_predictions)),
        "overall": {
            "model": overall_metrics,
            "elo_baseline": overall_elo_metrics,
        },
        "folds": fold_metrics,
    }
    return all_predictions, metrics_output


def walk_forward_and_save(
    modeling_path: str | Path,
    predictions_output_path: str | Path,
    metrics_output_path: str | Path,
    target_column: str = "target_home_win",
    train_start_season: int = TRAIN_START_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    model_type: str = "logistic_regression",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run walk-forward predictions and save CSV/JSON outputs."""

    modeling_df = pd.read_parquet(modeling_path)
    predictions, metrics = walk_forward_predict(
        modeling_df,
        target_column=target_column,
        train_start_season=train_start_season,
        first_test_season=first_test_season,
        last_test_season=last_test_season,
        model_type=model_type,
        random_seed=random_seed,
    )

    predictions_output_path = Path(predictions_output_path)
    metrics_output_path = Path(metrics_output_path)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_output_path, index=False)
    metrics_output_path.write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    return predictions, metrics
