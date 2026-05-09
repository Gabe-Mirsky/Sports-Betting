"""Model evaluation helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_binary_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate binary probability forecasts."""

    y = np.asarray(y_true).astype(int)
    probs = np.asarray(probabilities).astype(float)
    preds = (probs >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, preds)),
        "brier_score": float(brier_score_loss(y, probs)),
    }

    try:
        metrics["log_loss"] = float(log_loss(y, probs, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = math.nan

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    except ValueError:
        metrics["roc_auc"] = math.nan

    return metrics
