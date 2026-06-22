"""Audit calibrated NO signals against outcomes and closing-line value."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _prepare_no_rows(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    required = [
        "date",
        "price_cents",
        "clv_cents",
        "realized_profit_per_share",
        signal_column,
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"NO calibration audit rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    if "calibrated_side" in frame.columns:
        frame["_audit_side"] = frame["calibrated_side"]
    elif "candidate_side" in frame.columns:
        frame["_audit_side"] = frame["candidate_side"]
    elif "side" in frame.columns:
        frame["_audit_side"] = frame["side"]
    else:
        raise ValueError("NO calibration audit rows need one of: calibrated_side, candidate_side, side")

    frame["_audit_side"] = frame["_audit_side"].fillna("").astype(str).str.upper()
    frame = _numeric(
        frame,
        [
            "price_cents",
            "clv_cents",
            "clv_reference_price_cents",
            "realized_profit_per_share",
            "calibrated_win_rate",
            "calibrated_yes_rate",
            "model_prob",
            "model_yes_prob",
            "market_prob",
            "edge",
            "calibrated_expected_roi",
            "volume",
            "open_interest",
        ],
    )
    selected = frame[frame["_signal"] & frame["_audit_side"].eq("NO") & frame["date"].notna()].copy()
    if selected.empty:
        return selected.reset_index(drop=True)

    selected["month"] = selected["date"].dt.to_period("M").astype(str)
    if "actual_contract_win" in selected.columns:
        selected["actual_contract_win_bool"] = _coerce_bool(selected["actual_contract_win"])
    elif "actual_yes_win" in selected.columns:
        selected["actual_contract_win_bool"] = ~_coerce_bool(selected["actual_yes_win"])
    else:
        selected["actual_contract_win_bool"] = selected["realized_profit_per_share"] > 0

    if "calibrated_win_rate" in selected.columns:
        selected["forecast_contract_win_rate"] = selected["calibrated_win_rate"]
    elif "model_prob" in selected.columns:
        selected["forecast_contract_win_rate"] = selected["model_prob"]
    else:
        selected["forecast_contract_win_rate"] = np.nan

    selected["positive_clv"] = selected["clv_cents"] > 0
    selected["positive_clv_loss"] = selected["positive_clv"] & ~selected["actual_contract_win_bool"]
    selected["forecast_error"] = (
        selected["actual_contract_win_bool"].astype(float) - selected["forecast_contract_win_rate"]
    )
    return _bucketize(selected).reset_index(drop=True)


def _bucketize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["forecast_win_bucket"] = pd.cut(
        output.get("forecast_contract_win_rate", pd.Series(np.nan, index=output.index)),
        bins=[0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0],
        include_lowest=True,
    ).astype(str)
    output["entry_price_bucket"] = pd.cut(
        output.get("price_cents", pd.Series(np.nan, index=output.index)),
        bins=[0, 10, 20, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    output["edge_bucket"] = pd.cut(
        output.get("edge", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, -0.10, -0.05, 0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<-10%", "-10--5%", "-5-0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    output["clv_bucket"] = pd.cut(
        output.get("clv_cents", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, -10, -2, 0, 2, 10, 25, np.inf],
        labels=["<-10c", "-10--2c", "-2-0c", "0-2c", "2-10c", "10-25c", "25c+"],
    ).astype(str)
    output["liquidity_bucket"] = pd.cut(
        output.get("volume", pd.Series(np.nan, index=output.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
    return output


def _group_summary(frame: pd.DataFrame, group_columns: list[str], min_rows: int = 1) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=False):
        if len(group) < min_rows:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        avg_forecast = float(group["forecast_contract_win_rate"].mean())
        actual_rate = float(group["actual_contract_win_bool"].mean())
        row.update(
            {
                "rows": int(len(group)),
                "avg_forecast_win_rate": avg_forecast,
                "actual_win_rate": actual_rate,
                "calibration_error": actual_rate - avg_forecast,
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "positive_clv_rate": float(group["positive_clv"].mean()),
                "positive_clv_loss_rate": float(group["positive_clv_loss"].mean()),
                "avg_profit_per_share": float(group["realized_profit_per_share"].mean()),
                "avg_price_cents": float(group["price_cents"].mean()),
                "avg_edge": float(group["edge"].mean()) if "edge" in group.columns else np.nan,
                "avg_volume": float(group["volume"].mean()) if "volume" in group.columns else np.nan,
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    ordered = group_columns + [column for column in output.columns if column not in group_columns]
    return output[ordered].reset_index(drop=True)


def build_no_calibration_audit(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_segment_rows: int = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build diagnostics for calibrated NO probability quality."""

    selected = _prepare_no_rows(rows, signal_column=signal_column)
    reports: dict[str, pd.DataFrame] = {
        "rows": selected,
        "by_month": _group_summary(selected, ["month"], min_rows=1),
        "by_forecast_win_bucket": _group_summary(selected, ["forecast_win_bucket"], min_rows=1),
        "by_entry_price_bucket": _group_summary(selected, ["entry_price_bucket"], min_rows=1),
        "by_edge_bucket": _group_summary(selected, ["edge_bucket"], min_rows=1),
        "by_clv_bucket": _group_summary(selected, ["clv_bucket"], min_rows=1),
        "by_liquidity_bucket": _group_summary(selected, ["liquidity_bucket"], min_rows=1),
        "by_month_price": _group_summary(selected, ["month", "entry_price_bucket"], min_rows=min_segment_rows),
        "by_forecast_price": _group_summary(
            selected,
            ["forecast_win_bucket", "entry_price_bucket"],
            min_rows=min_segment_rows,
        ),
    }
    if selected.empty:
        summary = {
            "selected_no_rows": 0,
            "status": "no_selected_no_rows",
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
            "diagnosis": "No selected NO signals were available for calibration audit.",
        }
        return reports, summary

    reports["positive_clv_losses"] = selected[selected["positive_clv_loss"]].sort_values(
        ["clv_cents", "forecast_contract_win_rate"],
        ascending=[False, False],
    ).reset_index(drop=True)

    valid_forecast = selected["forecast_contract_win_rate"].notna()
    avg_forecast = float(selected.loc[valid_forecast, "forecast_contract_win_rate"].mean())
    actual_rate = float(selected["actual_contract_win_bool"].mean())
    avg_clv = float(selected["clv_cents"].mean())
    positive_clv_rate = float(selected["positive_clv"].mean())
    calibration_error = actual_rate - avg_forecast
    status = "not_ready"
    if len(selected) >= 300 and avg_clv > 0 and positive_clv_rate >= 0.50 and abs(calibration_error) <= 0.03:
        status = "watchlist"

    if avg_clv > 0 and calibration_error < -0.03:
        diagnosis = (
            "NO signals show small positive CLV, but settlement outcomes trail calibrated win rates. "
            "This points to probability calibration or noisy market-movement signal rather than a proven edge."
        )
    elif avg_clv <= 0:
        diagnosis = "NO signals do not beat the later pregame price on average."
    else:
        diagnosis = "NO signals need more positive-CLV frequency and month-to-month repeatability before proof gates can pass."

    summary = {
        "selected_no_rows": int(len(selected)),
        "status": status,
        "avg_forecast_win_rate": avg_forecast,
        "actual_win_rate": actual_rate,
        "calibration_error": calibration_error,
        "avg_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv_rate,
        "positive_clv_loss_count": int(selected["positive_clv_loss"].sum()),
        "positive_clv_loss_rate": float(selected["positive_clv_loss"].mean()),
        "avg_profit_per_share": float(selected["realized_profit_per_share"].mean()),
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "diagnosis": diagnosis,
    }
    return reports, summary


def save_no_calibration_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "no_calibration",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
