"""Prediction helpers for trained home-win models."""

from __future__ import annotations

from typing import Any

import pandas as pd


PREDICTION_ID_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "season_type",
    "home_team_abbr",
    "away_team_abbr",
]


def _get_model_and_features(model_or_bundle: Any) -> tuple[Any, list[str]]:
    if isinstance(model_or_bundle, dict):
        return model_or_bundle["model"], list(model_or_bundle["feature_columns"])

    feature_columns = getattr(model_or_bundle, "feature_columns", None)
    if feature_columns is None:
        raise ValueError("Model bundle must include feature_columns.")
    return model_or_bundle, list(feature_columns)


def predict_game_probabilities(model_or_bundle: Any, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return home and away win probabilities for each game."""

    model, feature_columns = _get_model_and_features(model_or_bundle)
    missing = [column for column in feature_columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Feature dataframe is missing model columns: {missing}")

    probabilities = model.predict_proba(feature_df[feature_columns])[:, 1]
    id_columns = [column for column in PREDICTION_ID_COLUMNS if column in feature_df.columns]
    output = feature_df[id_columns].copy()
    output["model_home_win_prob"] = probabilities
    output["model_away_win_prob"] = 1.0 - output["model_home_win_prob"]
    return output
