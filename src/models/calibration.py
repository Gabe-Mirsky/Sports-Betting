"""Calibration helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import calibration_curve


def calibration_curve_frame(
    y_true: pd.Series,
    probabilities: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return a dataframe with calibration curve points."""

    observed, predicted = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_win_rate": observed,
        }
    )
