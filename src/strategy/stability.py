"""Time stability diagnostics for calibrated paper-trade signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def summarize_signal_stability(
    rows: pd.DataFrame,
    signal_column: str,
    expected_roi_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize signal outcomes by month."""

    if rows.empty:
        return pd.DataFrame(), {"rows": 0}
    required = ["date", signal_column, "actual_yes_win"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Signal rows are missing stability columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["signal"] = _coerce_bool(frame[signal_column])
    frame["actual_yes_win"] = _coerce_bool(frame["actual_yes_win"])
    if "realized_profit_per_share" in frame.columns:
        frame["realized_profit_per_share"] = pd.to_numeric(frame["realized_profit_per_share"], errors="coerce")
    else:
        cost_column = "contract_cost" if "contract_cost" in frame.columns else "market_prob"
        frame[cost_column] = pd.to_numeric(frame[cost_column], errors="coerce")
        frame["realized_profit_per_share"] = np.where(
            frame["actual_yes_win"],
            1.0 - frame[cost_column],
            -frame[cost_column],
        )
    if "edge" in frame.columns:
        frame["edge"] = pd.to_numeric(frame["edge"], errors="coerce")
    if expected_roi_column and expected_roi_column in frame.columns:
        frame["expected_roi_for_stability"] = pd.to_numeric(frame[expected_roi_column], errors="coerce")
    else:
        frame["expected_roi_for_stability"] = np.nan

    signals = frame[frame["signal"]].copy()
    if signals.empty:
        return pd.DataFrame(), {
            "rows": int(len(frame)),
            "signals": 0,
            "timeline": "n/a",
            "positive_months": 0,
            "months": 0,
        }
    signals["month"] = signals["date"].dt.to_period("M").astype(str)
    monthly = (
        signals.groupby("month", as_index=False)
        .agg(
            signals=("signal", "size"),
            win_rate=("actual_yes_win", "mean"),
            avg_profit_per_share=("realized_profit_per_share", "mean"),
            total_profit_per_share=("realized_profit_per_share", "sum"),
            avg_edge=("edge", "mean"),
            avg_expected_roi=("expected_roi_for_stability", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    positive_months = int((monthly["avg_profit_per_share"] > 0).sum())
    dates = signals["date"]
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    worst_index = monthly["avg_profit_per_share"].idxmin()
    best_index = monthly["avg_profit_per_share"].idxmax()
    summary = {
        "rows": int(len(frame)),
        "signals": int(len(signals)),
        "timeline": start if start == end else f"{start} to {end}",
        "months": int(len(monthly)),
        "positive_months": positive_months,
        "positive_month_share": float(positive_months / len(monthly)) if len(monthly) else 0.0,
        "overall_win_rate": float(signals["actual_yes_win"].mean()),
        "overall_avg_profit_per_share": float(signals["realized_profit_per_share"].mean()),
        "worst_month": str(monthly.loc[worst_index, "month"]) if len(monthly) else None,
        "worst_month_avg_profit_per_share": float(monthly.loc[worst_index, "avg_profit_per_share"])
        if len(monthly)
        else 0.0,
        "best_month": str(monthly.loc[best_index, "month"]) if len(monthly) else None,
        "best_month_avg_profit_per_share": float(monthly.loc[best_index, "avg_profit_per_share"])
        if len(monthly)
        else 0.0,
        "note": "Stability is measured on rows where the selected signal column is true.",
    }
    return monthly, summary


def save_signal_stability_outputs(
    monthly: pd.DataFrame,
    summary: dict[str, Any],
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    output = Path(output_path)
    summary_output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
