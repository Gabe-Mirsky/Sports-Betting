"""Probability shrinkage utilities for research-only strategy sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shrink_probability(
    model_probability: float | pd.Series,
    market_probability: float | pd.Series,
    shrink_factor: float,
) -> float | pd.Series:
    """Shrink model probability toward market probability and clamp to [0, 1]."""

    adjusted = market_probability + shrink_factor * (model_probability - market_probability)
    if isinstance(adjusted, pd.Series):
        return pd.to_numeric(adjusted, errors="coerce").clip(lower=0.0, upper=1.0)
    if isinstance(adjusted, np.ndarray):
        return pd.Series(adjusted).clip(lower=0.0, upper=1.0)
    if adjusted is None or not np.isfinite(float(adjusted)):
        return float("nan")
    return float(min(max(float(adjusted), 0.0), 1.0))


def side_shrink_factor(side: str, yes_shrink_factor: float, no_shrink_factor: float) -> float:
    """Return the side-specific shrink factor for a YES/NO contract side."""

    return float(no_shrink_factor if str(side).upper() == "NO" else yes_shrink_factor)


def adjusted_edge(
    model_probability: float | pd.Series,
    market_probability: float | pd.Series,
    shrink_factor: float,
    uncertainty_penalty: float | pd.Series = 0.0,
) -> float | pd.Series:
    """Calculate edge after probability shrinkage and uncertainty penalty."""

    adjusted_probability = shrink_probability(model_probability, market_probability, shrink_factor)
    edge = adjusted_probability - market_probability - uncertainty_penalty
    if isinstance(edge, pd.Series):
        return pd.to_numeric(edge, errors="coerce")
    if edge is None or not np.isfinite(float(edge)):
        return float("nan")
    return float(edge)
