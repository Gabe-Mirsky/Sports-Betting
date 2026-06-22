"""Walk-forward backtest for the no-odds matchup model.

The backtest answers *prediction-quality* questions only – it never computes
ROI, never needs odds, and never judges the model purely on win rate:

* Is the model calibrated? (does a 60% prediction win ~60% of the time?)
* Does the confidence label mean anything?
* Which sports / leagues / competition types are strongest?
* Are draws predicted reasonably for soccer?

Validation is strictly walk-forward (expanding window) so games are never
shuffled and the model only ever trains on the past.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.sport_rules import normalize_sport, sport_allows_draws
from features.matchup_features import build_training_features
from models.matchup_model import PROB_COLUMNS, predict_matchup_probabilities, train_matchup_model
from models.prediction_explainer import assign_confidence_level
from quality.matchup_data_quality import assign_prediction_data_quality

logger = logging.getLogger(__name__)

_SIDE_FROM_CLASS = {0: "team_a", 1: "draw", 2: "team_b"}
_EPS = 1e-12


def _actual_side(row) -> str:
    if int(row["result_team_a_win"]) == 1:
        return "team_a"
    if int(row["result_draw"]) == 1:
        return "draw"
    return "team_b"


def walk_forward_backtest(
    results_df: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    sport: str | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Run an expanding-window backtest and return per-game predictions.

    Each fold trains on every game strictly before the fold's first game date
    and predicts that fold's games. The result is one row per scored game with
    predicted probabilities, the predicted/actual outcome, a confidence label,
    and a data-quality tag.
    """

    config = config or {}
    if results_df.empty:
        return pd.DataFrame()

    work = results_df.copy()
    if sport is not None:
        work = work[work["sport"].map(normalize_sport) == normalize_sport(sport)].copy()
    if work.empty:
        logger.warning("No games for sport=%s in backtest.", sport)
        return pd.DataFrame()

    feats = build_training_features(work, injuries_df, config)
    feats["game_date"] = pd.to_datetime(feats["game_date"], errors="coerce")

    # Keep only decided games and sort chronologically.
    decided = (
        feats["result_team_a_win"].astype(int)
        + feats["result_draw"].astype(int)
        + feats["result_team_b_win"].astype(int)
    ) == 1
    feats = feats[decided].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(feats)
    if n < 40:
        logger.warning("Only %d decided games; backtest may be unreliable.", n)

    min_train = int(config.get("min_train_games", max(40, int(n * 0.3))))
    retrain_every = int(config.get("retrain_every", max(20, n // 25)))
    if min_train >= n:
        logger.warning("Not enough games (%d) for the requested min_train (%d).", n, min_train)
        return pd.DataFrame()

    predictions: list[pd.DataFrame] = []
    start = min_train
    while start < n:
        end = min(start + retrain_every, n)
        test = feats.iloc[start:end].copy()
        cutoff_date = test["game_date"].min()
        train = feats[feats["game_date"] < cutoff_date]
        try:
            bundle = train_matchup_model(train, sport or _dominant_sport(train), config)
            preds = predict_matchup_probabilities(bundle, test)
        except ValueError as exc:
            logger.info("Skipping fold [%d:%d]: %s", start, end, exc)
            start = end
            continue
        predictions.append(preds)
        start = end

    if not predictions:
        return pd.DataFrame()

    out = pd.concat(predictions, ignore_index=True)
    out["actual_side"] = out.apply(_actual_side, axis=1)
    out["correct"] = (out["predicted_side"] == out["actual_side"]).astype(int)
    out["data_quality"] = out.apply(assign_prediction_data_quality, axis=1)
    out["confidence_level"] = out.apply(assign_confidence_level, axis=1)
    return out


def _dominant_sport(df: pd.DataFrame) -> str:
    if "sport" in df.columns and not df.empty:
        return str(df["sport"].mode().iloc[0])
    return "soccer"


def _prob_of_side(row, side: str) -> float:
    return float(row[{"team_a": "prob_team_a_win", "draw": "prob_draw", "team_b": "prob_team_b_win"}[side]])


def evaluate_probability_predictions(predictions_df: pd.DataFrame) -> dict:
    """Compute calibration/accuracy metrics for backtest predictions.

    Returns a JSON-serializable dict – accuracy, log loss, Brier score, mean
    probability of the actual outcome, plus accuracy broken down by confidence
    level, sport, league, competition type, and (for draw sports) draw quality.
    """

    if predictions_df is None or predictions_df.empty:
        return {"n_games": 0, "note": "No predictions to evaluate."}

    df = predictions_df.copy()
    probs = df[PROB_COLUMNS].to_numpy(dtype=float)
    side_to_idx = {"team_a": 0, "draw": 1, "team_b": 2}
    actual_idx = df["actual_side"].map(side_to_idx).to_numpy()

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(df)), actual_idx] = 1.0
    p_actual = probs[np.arange(len(df)), actual_idx]

    log_loss = float(-np.mean(np.log(np.clip(p_actual, _EPS, 1.0))))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    accuracy = float(df["correct"].mean())
    mean_p_actual = float(np.mean(p_actual))

    # Favorite = the team (ignoring draw) the model prefers.
    fav_team_a = probs[:, 0] >= probs[:, 2]
    fav_actual_win = np.where(fav_team_a, df["actual_side"] == "team_a", df["actual_side"] == "team_b")
    favorite_win_rate = float(np.mean(fav_actual_win))

    metrics = {
        "n_games": int(len(df)),
        "accuracy": round(accuracy, 4),
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
        "mean_prob_of_actual_outcome": round(mean_p_actual, 4),
        "favorite_win_rate": round(favorite_win_rate, 4),
        "underdog_win_rate": round(1.0 - favorite_win_rate, 4),
        "accuracy_by_confidence": _accuracy_by(df, "confidence_level"),
        "accuracy_by_sport": _accuracy_by(df, "sport"),
        "accuracy_by_league": _accuracy_by(df, "league"),
        "accuracy_by_competition_type": _accuracy_by(df, "competition_type"),
    }

    if (df["prob_draw"] > 0).any() or (df["actual_side"] == "draw").any():
        metrics["draw_quality"] = _draw_quality(df)

    return metrics


def _accuracy_by(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {}
    grouped = df.groupby(column)["correct"].agg(["count", "mean"])
    return {
        str(key): {"n": int(row["count"]), "accuracy": round(float(row["mean"]), 4)}
        for key, row in grouped.iterrows()
    }


def _draw_quality(df: pd.DataFrame) -> dict:
    predicted_draw = df["predicted_side"] == "draw"
    actual_draw = df["actual_side"] == "draw"
    n_pred = int(predicted_draw.sum())
    n_actual = int(actual_draw.sum())
    precision = float((predicted_draw & actual_draw).sum() / n_pred) if n_pred else None
    recall = float((predicted_draw & actual_draw).sum() / n_actual) if n_actual else None
    return {
        "actual_draw_rate": round(float(actual_draw.mean()), 4),
        "predicted_draw_rate": round(float(predicted_draw.mean()), 4),
        "mean_predicted_draw_prob": round(float(df["prob_draw"].mean()), 4),
        "draw_precision": round(precision, 4) if precision is not None else None,
        "draw_recall": round(recall, 4) if recall is not None else None,
    }


def summarize_backtest_by_bucket(
    predictions_df: pd.DataFrame,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Return a calibration table keyed on the predicted outcome's probability.

    For each probability bucket we report the number of games, the mean
    predicted probability, and the actual hit rate – so you can see whether a
    "60%" prediction really wins about 60% of the time.
    """

    columns = ["prob_bucket", "n_games", "mean_predicted_prob", "actual_win_rate", "calibration_gap"]
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=columns)

    df = predictions_df.copy()
    top_prob = df[PROB_COLUMNS].max(axis=1)
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(n_buckets)]
    df["prob_bucket"] = pd.cut(top_prob, bins=edges, labels=labels, include_lowest=True)
    df["_top_prob"] = top_prob

    rows = []
    for bucket, group in df.groupby("prob_bucket", observed=True):
        if group.empty:
            continue
        mean_pred = float(group["_top_prob"].mean())
        actual = float(group["correct"].mean())
        rows.append(
            {
                "prob_bucket": str(bucket),
                "n_games": int(len(group)),
                "mean_predicted_prob": round(mean_pred, 4),
                "actual_win_rate": round(actual, 4),
                "calibration_gap": round(mean_pred - actual, 4),
            }
        )
    return pd.DataFrame(rows, columns=columns)
