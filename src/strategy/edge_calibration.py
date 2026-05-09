"""Calibrate model-vs-market edges before portfolio selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EDGE_BINS = [-1.0, -0.15, -0.10, -0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 1.0]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def prepare_edge_calibration_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize saved backtest rows for edge calibration."""

    required = ["date", "market_ticker", "model_yes_prob", "market_prob", "edge", "price_cents", "actual_yes_win"]
    missing = [column for column in required if column not in trades.columns]
    if missing:
        raise ValueError(f"Trade rows are missing edge-calibration columns: {missing}")

    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["model_yes_prob", "market_prob", "edge", "price_cents"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "model_yes_prob", "market_prob", "edge", "price_cents"]).copy()
    frame["actual_yes_win"] = _coerce_bool(frame["actual_yes_win"])
    frame["contract_cost"] = frame["price_cents"] / 100.0
    frame = frame[(frame["contract_cost"] > 0) & (frame["contract_cost"] < 1)].copy()
    frame["realized_profit_per_share"] = np.where(
        frame["actual_yes_win"],
        1.0 - frame["contract_cost"],
        -frame["contract_cost"],
    )
    frame["model_expected_profit_per_share"] = frame["model_yes_prob"] - frame["contract_cost"]
    frame["model_expected_roi"] = frame["model_expected_profit_per_share"] / frame["contract_cost"]
    return frame.sort_values(["date", "market_ticker"]).reset_index(drop=True)


def edge_bin_summary(
    trades: pd.DataFrame,
    edge_bins: list[float] | None = None,
) -> pd.DataFrame:
    """Summarize historical outcomes by model edge bucket."""

    edge_bins = edge_bins or DEFAULT_EDGE_BINS
    frame = prepare_edge_calibration_frame(trades)
    if frame.empty:
        return pd.DataFrame()
    frame["edge_bin"] = pd.cut(frame["edge"], bins=edge_bins, include_lowest=True, right=True).astype(str)
    grouped = (
        frame.groupby("edge_bin", observed=False)
        .agg(
            markets=("edge", "size"),
            avg_edge=("edge", "mean"),
            avg_model_prob=("model_yes_prob", "mean"),
            avg_market_prob=("market_prob", "mean"),
            observed_yes_rate=("actual_yes_win", "mean"),
            avg_model_expected_profit_per_share=("model_expected_profit_per_share", "mean"),
            avg_realized_profit_per_share=("realized_profit_per_share", "mean"),
            avg_model_expected_roi=("model_expected_roi", "mean"),
        )
        .reset_index()
    )
    grouped["realized_roi_on_cost"] = grouped["avg_realized_profit_per_share"] / grouped["avg_market_prob"]
    grouped["calibration_gap"] = (
        grouped["avg_realized_profit_per_share"] - grouped["avg_model_expected_profit_per_share"]
    )
    return grouped


def _date_timeline(values: pd.Series) -> tuple[str | None, str | None, str]:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None, None, "n/a"
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    return start, end, start if start == end else f"{start} to {end}"


def audit_calibrated_edges(calibrated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit calibrated signals, with special attention to negative raw edges."""

    if calibrated.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0}
    required = [
        "date",
        "market_ticker",
        "edge",
        "actual_yes_win",
        "contract_cost",
        "calibrated_trade",
        "calibrated_expected_roi",
    ]
    missing = [column for column in required if column not in calibrated.columns]
    if missing:
        raise ValueError(f"Calibrated rows are missing audit columns: {missing}")

    frame = calibrated.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in [
        "edge",
        "market_prob",
        "model_yes_prob",
        "price_cents",
        "contract_cost",
        "calibrated_yes_rate",
        "calibrated_expected_roi",
        "calibrated_expected_profit_per_share",
        "edge_bin_history_rows",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["actual_yes_win"] = _coerce_bool(frame["actual_yes_win"])
    frame["calibrated_trade"] = _coerce_bool(frame["calibrated_trade"])
    if "realized_profit_per_share" not in frame.columns:
        frame["realized_profit_per_share"] = np.where(
            frame["actual_yes_win"],
            1.0 - frame["contract_cost"],
            -frame["contract_cost"],
        )
    else:
        frame["realized_profit_per_share"] = pd.to_numeric(
            frame["realized_profit_per_share"],
            errors="coerce",
        )

    audit_rows: list[dict[str, Any]] = []
    for edge_bin, group in frame.groupby("edge_bin", dropna=False, sort=False):
        signals = group[group["calibrated_trade"]]
        negative_signals = signals[signals["edge"] < 0]
        audit_rows.append(
            {
                "edge_bin": str(edge_bin),
                "markets": int(len(group)),
                "calibrated_signals": int(len(signals)),
                "avg_edge": float(group["edge"].mean()) if len(group) else 0.0,
                "avg_calibrated_expected_roi": float(group["calibrated_expected_roi"].mean()) if len(group) else 0.0,
                "observed_yes_rate": float(group["actual_yes_win"].mean()) if len(group) else 0.0,
                "all_avg_realized_profit_per_share": float(group["realized_profit_per_share"].mean())
                if len(group)
                else 0.0,
                "signal_win_rate": float(signals["actual_yes_win"].mean()) if len(signals) else 0.0,
                "signal_avg_realized_profit_per_share": float(signals["realized_profit_per_share"].mean())
                if len(signals)
                else 0.0,
                "signal_avg_calibrated_expected_roi": float(signals["calibrated_expected_roi"].mean())
                if len(signals)
                else 0.0,
                "negative_raw_edge_signals": int(len(negative_signals)),
                "negative_raw_edge_signal_win_rate": float(negative_signals["actual_yes_win"].mean())
                if len(negative_signals)
                else 0.0,
                "negative_raw_edge_signal_profit_per_share": float(
                    negative_signals["realized_profit_per_share"].mean()
                )
                if len(negative_signals)
                else 0.0,
            }
        )

    audit = pd.DataFrame(audit_rows)
    selected = frame[frame["calibrated_trade"]].copy()
    negative = selected[selected["edge"] < 0].copy()
    negative = negative.sort_values(["date", "market_ticker"]).reset_index(drop=True)
    negative_columns = [
        "date",
        "game_id",
        "market_ticker",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "edge_bin",
        "edge",
        "model_yes_prob",
        "market_prob",
        "calibrated_yes_rate",
        "calibrated_expected_roi",
        "price_cents",
        "actual_yes_win",
        "realized_profit_per_share",
        "edge_bin_history_rows",
        "calibration_reason",
    ]
    negative = negative[[column for column in negative_columns if column in negative.columns]].copy()

    selected_start, selected_end, selected_timeline = _date_timeline(selected["date"]) if not selected.empty else (None, None, "n/a")
    negative_start, negative_end, negative_timeline = _date_timeline(negative["date"]) if not negative.empty else (None, None, "n/a")
    positive = selected[selected["edge"] >= 0].copy()
    summary = {
        "rows": int(len(frame)),
        "calibrated_trades": int(len(selected)),
        "trade_start_date": selected_start,
        "trade_end_date": selected_end,
        "trade_timeline": selected_timeline,
        "negative_raw_edge_calibrated_trades": int(len(negative)),
        "negative_raw_edge_trade_start_date": negative_start,
        "negative_raw_edge_trade_end_date": negative_end,
        "negative_raw_edge_trade_timeline": negative_timeline,
        "negative_raw_edge_share_of_calibrated_trades": float(len(negative) / len(selected)) if len(selected) else 0.0,
        "negative_raw_edge_win_rate": float(negative["actual_yes_win"].mean()) if len(negative) else 0.0,
        "negative_raw_edge_avg_profit_per_share": float(negative["realized_profit_per_share"].mean())
        if len(negative)
        else 0.0,
        "positive_raw_edge_calibrated_trades": int(len(positive)),
        "positive_raw_edge_win_rate": float(positive["actual_yes_win"].mean()) if len(positive) else 0.0,
        "positive_raw_edge_avg_profit_per_share": float(positive["realized_profit_per_share"].mean())
        if len(positive)
        else 0.0,
        "audit_note": "Negative raw-edge calibrated trades mean historical market-vs-model residuals supported the market side more than the raw model edge.",
    }
    return audit, negative, summary


def add_expanding_edge_calibration(
    trades: pd.DataFrame,
    edge_bins: list[float] | None = None,
    min_history_rows: int = 75,
    min_calibrated_profit_per_share: float = 0.0,
    min_calibrated_roi: float | None = None,
    min_observed_yes_rate: float | None = None,
    shrinkage_rows: int = 150,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add expanding-window calibrated edge estimates to each market row.

    Calibration updates after each slate date, not after each row. That avoids
    letting one game from a date leak into another market on the same date.
    """

    edge_bins = edge_bins or DEFAULT_EDGE_BINS
    frame = prepare_edge_calibration_frame(trades)
    if frame.empty:
        return frame, {"rows": 0}

    frame["edge_bin"] = pd.cut(frame["edge"], bins=edge_bins, include_lowest=True, right=True).astype(str)
    global_seen = 0
    global_wins = 0
    bin_seen: dict[str, int] = {}
    bin_wins: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for _, slate in frame.groupby(frame["date"].dt.date, sort=True):
        slate_outputs: list[dict[str, Any]] = []
        for _, row in slate.iterrows():
            edge_bin = str(row["edge_bin"])
            seen = bin_seen.get(edge_bin, 0)
            wins = bin_wins.get(edge_bin, 0)
            global_rate = global_wins / global_seen if global_seen else float(row["model_yes_prob"])
            bin_rate = wins / seen if seen else global_rate
            weight = seen / (seen + float(shrinkage_rows)) if seen else 0.0
            calibrated_yes_rate = weight * bin_rate + (1.0 - weight) * global_rate
            calibrated_profit = calibrated_yes_rate - float(row["contract_cost"])
            calibrated_roi = calibrated_profit / float(row["contract_cost"])
            has_history = seen >= int(min_history_rows)
            rate_ok = True if min_observed_yes_rate is None else calibrated_yes_rate >= float(min_observed_yes_rate)
            roi_ok = True if min_calibrated_roi is None else calibrated_roi >= float(min_calibrated_roi)
            calibrated_trade = bool(
                has_history
                and rate_ok
                and roi_ok
                and calibrated_profit >= float(min_calibrated_profit_per_share)
            )
            if not has_history:
                reason = "insufficient_prior_edge_history"
            elif not rate_ok:
                reason = "calibrated_yes_rate_below_threshold"
            elif not roi_ok:
                reason = "calibrated_roi_below_threshold"
            elif calibrated_profit < float(min_calibrated_profit_per_share):
                reason = "calibrated_edge_below_threshold"
            else:
                reason = "calibrated_edge_met"

            output = row.to_dict()
            output.update(
                {
                    "edge_bin_history_rows": int(seen),
                    "edge_bin_history_yes_rate": float(bin_rate),
                    "edge_global_history_rows": int(global_seen),
                    "edge_global_history_yes_rate": float(global_rate),
                    "calibration_source": "prior_dates_same_edge_bin" if has_history else "insufficient_prior_dates",
                    "calibrated_yes_rate": float(calibrated_yes_rate),
                    "calibrated_expected_profit_per_share": float(calibrated_profit),
                    "calibrated_expected_roi": float(calibrated_roi),
                    "calibrated_trade": calibrated_trade,
                    "calibration_reason": reason,
                }
            )
            slate_outputs.append(output)

        rows.extend(slate_outputs)

        for _, row in slate.iterrows():
            edge_bin = str(row["edge_bin"])
            actual = bool(row["actual_yes_win"])
            bin_seen[edge_bin] = bin_seen.get(edge_bin, 0) + 1
            bin_wins[edge_bin] = bin_wins.get(edge_bin, 0) + int(actual)
            global_seen += 1
            global_wins += int(actual)

    calibrated = pd.DataFrame(rows)
    summary = {
        "rows": int(len(calibrated)),
        "calibrated_trades": int(calibrated["calibrated_trade"].sum()),
        "trade_start_date": calibrated.loc[calibrated["calibrated_trade"], "date"].min().date().isoformat()
        if calibrated["calibrated_trade"].any()
        else None,
        "trade_end_date": calibrated.loc[calibrated["calibrated_trade"], "date"].max().date().isoformat()
        if calibrated["calibrated_trade"].any()
        else None,
        "min_history_rows": int(min_history_rows),
        "min_calibrated_profit_per_share": float(min_calibrated_profit_per_share),
        "min_calibrated_roi": None if min_calibrated_roi is None else float(min_calibrated_roi),
        "min_observed_yes_rate": None if min_observed_yes_rate is None else float(min_observed_yes_rate),
        "shrinkage_rows": int(shrinkage_rows),
        "edge_bins": edge_bins,
        "note": "Calibration is expanding-window only: each row uses outcomes from prior market dates, not the same slate date.",
    }
    if summary["trade_start_date"] and summary["trade_end_date"]:
        start = summary["trade_start_date"]
        end = summary["trade_end_date"]
        summary["trade_timeline"] = start if start == end else f"{start} to {end}"
    else:
        summary["trade_timeline"] = "n/a"
    return calibrated, summary


def calibrate_edges_and_save(
    trades_path: str | Path,
    calibrated_output_path: str | Path,
    bins_output_path: str | Path,
    summary_output_path: str | Path,
    audit_output_path: str | Path | None = None,
    negative_edge_output_path: str | Path | None = None,
    audit_summary_output_path: str | Path | None = None,
    min_history_rows: int = 75,
    min_calibrated_profit_per_share: float = 0.0,
    min_calibrated_roi: float | None = None,
    min_observed_yes_rate: float | None = None,
    shrinkage_rows: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load backtest rows, calibrate edges, and save artifacts."""

    trades = pd.read_csv(trades_path, dtype={"game_id": str, "market_ticker": str})
    calibrated, summary = add_expanding_edge_calibration(
        trades,
        min_history_rows=min_history_rows,
        min_calibrated_profit_per_share=min_calibrated_profit_per_share,
        min_calibrated_roi=min_calibrated_roi,
        min_observed_yes_rate=min_observed_yes_rate,
        shrinkage_rows=shrinkage_rows,
    )
    bins = edge_bin_summary(trades)
    audit, negative_edge_signals, audit_summary = audit_calibrated_edges(calibrated)
    calibrated_output = Path(calibrated_output_path)
    bins_output = Path(bins_output_path)
    summary_output = Path(summary_output_path)
    calibrated_output.parent.mkdir(parents=True, exist_ok=True)
    bins_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(calibrated_output, index=False)
    bins.to_csv(bins_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if audit_output_path is not None:
        audit_output = Path(audit_output_path)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_output, index=False)
    if negative_edge_output_path is not None:
        negative_output = Path(negative_edge_output_path)
        negative_output.parent.mkdir(parents=True, exist_ok=True)
        negative_edge_signals.to_csv(negative_output, index=False)
    if audit_summary_output_path is not None:
        audit_summary_output = Path(audit_summary_output_path)
        audit_summary_output.parent.mkdir(parents=True, exist_ok=True)
        audit_summary_output.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    return calibrated, bins, summary
