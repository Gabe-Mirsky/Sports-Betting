"""Market-aware probability blending for Kalshi matched games."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(probabilities / (1.0 - probabilities))


def _fit_logistic_blend(
    features: np.ndarray,
    target: np.ndarray,
    l2: float = 0.05,
    market_feature_index: int = 1,
) -> np.ndarray:
    x = np.column_stack([np.ones(len(features)), features])
    y = np.asarray(target, dtype=float)
    beta = np.zeros(x.shape[1])
    prior = np.zeros_like(beta)
    prior_index = 1 + market_feature_index
    if 0 <= prior_index < len(prior):
        prior[prior_index] = 1.0
    penalty = np.diag([0.0, *([l2] * (len(beta) - 1))])

    for _ in range(60):
        probabilities = _sigmoid(x @ beta)
        weights = probabilities * (1.0 - probabilities)
        gradient = x.T @ (probabilities - y) + penalty @ (beta - prior)
        hessian = x.T @ (x * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return beta


def _predict_logistic_blend(beta: np.ndarray, features: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(features)), features])
    return _sigmoid(x @ beta)


def _roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positive = p[y == 1]
    negative = p[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    rank_sum_positive = ranks[y == 1].sum()
    return float((rank_sum_positive - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def binary_probability_metrics(y_true: pd.Series | np.ndarray, probabilities: pd.Series | np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "accuracy": float(((p >= 0.5) == y).mean()),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "roc_auc": _roc_auc(y, p),
    }


def add_market_blended_probabilities(
    matched_markets: pd.DataFrame,
    min_train_rows: int = 250,
    use_playoff_features: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add expanding-window market-aware probabilities to resolved matched rows.

    The blend expands by slate date, not by individual row, so games on the same
    date cannot train on one another's outcomes.
    """

    data = matched_markets.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data = data.dropna(subset=["game_date", "actual_yes_win", "model_yes_prob", "yes_mid_cents"]).copy()
    data = data.sort_values(["game_date", "game_id", "market_ticker"]).reset_index(drop=True)
    data["actual_yes_win"] = data["actual_yes_win"].astype(int)
    data["market_yes_prob"] = pd.to_numeric(data["yes_mid_cents"], errors="coerce") / 100.0
    data = data.dropna(subset=["market_yes_prob"]).copy()
    if "is_playoffs" in data.columns:
        data["is_playoffs"] = data["is_playoffs"].astype(str).str.lower().isin({"true", "1", "yes"})
    elif "season_type" in data.columns:
        data["is_playoffs"] = data["season_type"].astype(str).str.contains("Playoffs", case=False, na=False)
    else:
        data["is_playoffs"] = False

    model_logit = _logit(data["model_yes_prob"])
    market_logit = _logit(data["market_yes_prob"])
    feature_arrays = [model_logit, market_logit]
    feature_names = ["model_logit", "market_logit"]
    if use_playoff_features:
        playoff_flag = data["is_playoffs"].astype(float).to_numpy()
        feature_arrays.extend([playoff_flag, model_logit * playoff_flag, market_logit * playoff_flag])
        feature_names.extend(["is_playoffs", "model_logit_x_playoffs", "market_logit_x_playoffs"])
    features = np.column_stack(feature_arrays)
    target = data["actual_yes_win"].to_numpy(dtype=int)
    blended: list[float] = []
    methods: list[str] = []

    for _, slate in data.groupby(data["game_date"].dt.date, sort=True):
        slate_index = slate.index.to_numpy()
        first_slate_index = int(slate_index.min())
        train_index = np.arange(first_slate_index)
        if len(train_index) >= min_train_rows and len(np.unique(target[train_index])) == 2:
            beta = _fit_logistic_blend(features[train_index], target[train_index], market_feature_index=1)
            slate_probabilities = _predict_logistic_blend(beta, features[slate_index])
            method = "expanding_market_playoff_blend" if use_playoff_features else "expanding_market_blend"
            for probability in slate_probabilities:
                blended.append(float(probability))
                methods.append(method)
        else:
            for index in slate_index:
                probability = float(0.5 * data.loc[index, "model_yes_prob"] + 0.5 * data.loc[index, "market_yes_prob"])
                blended.append(probability)
                methods.append("warmup_half_model_half_market")

    data["blended_yes_prob"] = blended
    data["blend_method"] = methods
    metrics = {
        "model": binary_probability_metrics(data["actual_yes_win"], data["model_yes_prob"]),
        "market": binary_probability_metrics(data["actual_yes_win"], data["market_yes_prob"]),
        "market_blend": binary_probability_metrics(data["actual_yes_win"], data["blended_yes_prob"]),
        "regular_season": {},
        "playoffs": {},
        "rows": int(len(data)),
        "playoff_rows": int(data["is_playoffs"].sum()),
        "min_train_rows": int(min_train_rows),
        "use_playoff_features": bool(use_playoff_features),
        "features": feature_names,
        "note": "Market blend is expanding-window by prior slate date; same-date games are not used for training.",
    }
    regular = data[~data["is_playoffs"]]
    playoffs = data[data["is_playoffs"]]
    if not regular.empty:
        metrics["regular_season"] = {
            "model": binary_probability_metrics(regular["actual_yes_win"], regular["model_yes_prob"]),
            "market": binary_probability_metrics(regular["actual_yes_win"], regular["market_yes_prob"]),
            "market_blend": binary_probability_metrics(regular["actual_yes_win"], regular["blended_yes_prob"]),
            "rows": int(len(regular)),
        }
    if not playoffs.empty:
        metrics["playoffs"] = {
            "model": binary_probability_metrics(playoffs["actual_yes_win"], playoffs["model_yes_prob"]),
            "market": binary_probability_metrics(playoffs["actual_yes_win"], playoffs["market_yes_prob"]),
            "market_blend": binary_probability_metrics(playoffs["actual_yes_win"], playoffs["blended_yes_prob"]),
            "rows": int(len(playoffs)),
        }
    return data, metrics


def save_market_blend_outputs(
    blended: pd.DataFrame,
    metrics: dict[str, Any],
    predictions_path: str | Path,
    metrics_path: str | Path,
) -> None:
    predictions_output = Path(predictions_path)
    metrics_output = Path(metrics_path)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(predictions_output, index=False)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
