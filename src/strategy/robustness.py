"""Lower-confidence-bound screens for calibrated paper-trade signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def _timeline(values: pd.Series) -> tuple[str | None, str | None, str]:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None, None, "n/a"
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    return start, end, start if start == end else f"{start} to {end}"


def _lower_yes_rate(probability: pd.Series, sample_size: pd.Series, confidence_z: float) -> pd.Series:
    p = pd.to_numeric(probability, errors="coerce").clip(0.0, 1.0)
    n = pd.to_numeric(sample_size, errors="coerce")
    standard_error = np.sqrt((p * (1.0 - p)) / n.where(n > 0))
    return (p - float(confidence_z) * standard_error).clip(0.0, 1.0)


def add_confidence_screen(
    calibrated: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    expected_roi_column: str = "calibrated_expected_roi",
    probability_column: str = "calibrated_yes_rate",
    sample_size_column: str = "edge_bin_history_rows",
    blend_probability_column: str | None = None,
    blend_sample_size_column: str | None = None,
    cost_column: str = "contract_cost",
    min_history_rows: int = 100,
    confidence_z: float = 0.75,
    min_lower_profit_per_share: float = 0.0,
    min_expected_roi: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a robust signal flag based on lower-bound calibrated profitability."""

    if calibrated.empty:
        return pd.DataFrame(), {"rows": 0}
    required = [signal_column, expected_roi_column, probability_column, sample_size_column, cost_column]
    missing = [column for column in required if column not in calibrated.columns]
    if missing:
        raise ValueError(f"Calibrated rows are missing robustness columns: {missing}")

    output = calibrated.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce") if "date" in output.columns else pd.NaT
    output["base_signal"] = _coerce_bool(output[signal_column])
    output["base_expected_roi_for_screen"] = pd.to_numeric(output[expected_roi_column], errors="coerce")
    output["confidence_sample_size"] = pd.to_numeric(output[sample_size_column], errors="coerce").fillna(0)
    output["contract_cost"] = pd.to_numeric(output[cost_column], errors="coerce")
    output["confidence_lower_yes_rate"] = _lower_yes_rate(
        output[probability_column],
        output["confidence_sample_size"],
        confidence_z,
    )
    output["confidence_lower_profit_per_share"] = output["confidence_lower_yes_rate"] - output["contract_cost"]
    output["confidence_lower_expected_roi"] = (
        output["confidence_lower_profit_per_share"] / output["contract_cost"]
    )
    output["confidence_source"] = "single_calibration_lower_bound"

    if (
        blend_probability_column
        and blend_sample_size_column
        and blend_probability_column in output.columns
        and blend_sample_size_column in output.columns
    ):
        output["blend_confidence_sample_size"] = pd.to_numeric(
            output[blend_sample_size_column],
            errors="coerce",
        ).fillna(0)
        output["blend_confidence_lower_yes_rate"] = _lower_yes_rate(
            output[blend_probability_column],
            output["blend_confidence_sample_size"],
            confidence_z,
        )
        output["blend_confidence_lower_profit_per_share"] = (
            output["blend_confidence_lower_yes_rate"] - output["contract_cost"]
        )
        output["blend_confidence_lower_expected_roi"] = (
            output["blend_confidence_lower_profit_per_share"] / output["contract_cost"]
        )
        output["confidence_sample_size"] = output[["confidence_sample_size", "blend_confidence_sample_size"]].min(axis=1)
        output["confidence_lower_yes_rate"] = output[
            ["confidence_lower_yes_rate", "blend_confidence_lower_yes_rate"]
        ].min(axis=1)
        output["confidence_lower_profit_per_share"] = output[
            ["confidence_lower_profit_per_share", "blend_confidence_lower_profit_per_share"]
        ].min(axis=1)
        output["confidence_lower_expected_roi"] = output[
            ["confidence_lower_expected_roi", "blend_confidence_lower_expected_roi"]
        ].min(axis=1)
        output["confidence_source"] = "raw_and_market_blend_lower_bound"

    output["robust_calibrated_trade"] = (
        output["base_signal"]
        & (output["confidence_sample_size"] >= int(min_history_rows))
        & (output["base_expected_roi_for_screen"] >= float(min_expected_roi))
        & (output["confidence_lower_profit_per_share"] >= float(min_lower_profit_per_share))
    )
    output["robust_expected_roi"] = output["confidence_lower_expected_roi"]
    output["robust_reason"] = np.select(
        [
            ~output["base_signal"],
            output["confidence_sample_size"] < int(min_history_rows),
            output["base_expected_roi_for_screen"] < float(min_expected_roi),
            output["confidence_lower_profit_per_share"] < float(min_lower_profit_per_share),
        ],
        [
            "base_signal_false",
            "insufficient_confidence_history",
            "expected_roi_below_threshold",
            "lower_bound_not_profitable",
        ],
        default="robust_signal_met",
    )

    selected = output[output["robust_calibrated_trade"]].copy()
    start, end, timeline = _timeline(selected["date"]) if not selected.empty else (None, None, "n/a")
    if "actual_yes_win" in selected.columns:
        actual_yes = _coerce_bool(selected["actual_yes_win"])
    else:
        actual_yes = pd.Series(dtype=bool)
    realized_profit = (
        pd.to_numeric(selected["realized_profit_per_share"], errors="coerce")
        if "realized_profit_per_share" in selected.columns
        else pd.Series(dtype=float)
    )
    summary = {
        "rows": int(len(output)),
        "base_signals": int(output["base_signal"].sum()),
        "robust_signals": int(output["robust_calibrated_trade"].sum()),
        "trade_start_date": start,
        "trade_end_date": end,
        "trade_timeline": timeline,
        "min_history_rows": int(min_history_rows),
        "confidence_z": float(confidence_z),
        "min_lower_profit_per_share": float(min_lower_profit_per_share),
        "min_expected_roi": float(min_expected_roi),
        "confidence_source": str(output["confidence_source"].iloc[0]) if len(output) else "n/a",
        "win_rate": float(actual_yes.mean()) if len(actual_yes) else 0.0,
        "avg_realized_profit_per_share": float(realized_profit.mean()) if len(realized_profit) else 0.0,
        "avg_lower_expected_roi": float(selected["robust_expected_roi"].mean()) if len(selected) else 0.0,
        "note": "Robust signals require the lower confidence bound of calibrated yes rate to clear contract cost.",
    }
    return output.reset_index(drop=True), summary


def save_confidence_screen_outputs(
    screened: pd.DataFrame,
    summary: dict[str, Any],
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    output = Path(output_path)
    summary_output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    screened.to_csv(output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
