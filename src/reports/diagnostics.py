"""Additional diagnostics for model predictions and paper backtests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Convert common saved bool formats to a bool series."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def prediction_probability_bins(
    predictions: pd.DataFrame,
    probability_column: str = "model_home_win_prob",
    target_column: str = "actual_home_win",
    bins: int = 10,
) -> pd.DataFrame:
    """Summarize calibration by model probability bucket."""

    if predictions.empty:
        return pd.DataFrame()

    working = predictions.copy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    working["probability_bin"] = pd.cut(
        working[probability_column],
        bins=edges,
        include_lowest=True,
        right=True,
    )
    grouped = (
        working.groupby("probability_bin", observed=False)
        .agg(
            games=(target_column, "size"),
            avg_predicted_prob=(probability_column, "mean"),
            observed_win_rate=(target_column, "mean"),
        )
        .reset_index()
    )
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    grouped["calibration_error"] = grouped["observed_win_rate"] - grouped["avg_predicted_prob"]
    grouped["abs_calibration_error"] = grouped["calibration_error"].abs()
    return grouped


def season_prediction_summary(
    predictions: pd.DataFrame,
    probability_column: str = "model_home_win_prob",
    target_column: str = "actual_home_win",
) -> pd.DataFrame:
    """Summarize prediction quality by season."""

    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for season, group in predictions.groupby("season", sort=True):
        probs = group[probability_column].astype(float)
        actual = group[target_column].astype(int)
        picks = (probs >= 0.5).astype(int)
        accuracy = float((picks == actual).mean())
        brier = float(((probs - actual) ** 2).mean())
        rows.append(
            {
                "season": int(season),
                "games": int(len(group)),
                "accuracy": accuracy,
                "brier_score": brier,
                "avg_predicted_home_win_prob": float(probs.mean()),
                "actual_home_win_rate": float(actual.mean()),
                "mean_absolute_calibration_error": float((probs - actual).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def backtest_edge_bins(
    trades: pd.DataFrame,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Summarize market outcomes and paper P/L by edge bucket."""

    if trades.empty:
        return pd.DataFrame()

    bins = bins or [-1.0, -0.10, -0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 1.0]
    working = trades.copy()
    working["_trade_bool"] = _coerce_bool(working["trade"])
    working["_actual_yes_win_bool"] = _coerce_bool(working["actual_yes_win"])
    working["edge_bin"] = pd.cut(
        working["edge"].astype(float),
        bins=bins,
        include_lowest=True,
        right=True,
    )
    grouped = (
        working.groupby("edge_bin", observed=False)
        .agg(
            markets=("edge", "size"),
            trades=("_trade_bool", "sum"),
            avg_edge=("edge", "mean"),
            win_rate=("_actual_yes_win_bool", "mean"),
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
            amount_risked=("cost", "sum"),
        )
        .reset_index()
    )

    traded_only = working[working["_trade_bool"]]
    if traded_only.empty:
        grouped["traded_win_rate"] = np.nan
    else:
        traded_rates = (
            traded_only.groupby("edge_bin", observed=False)["_actual_yes_win_bool"]
            .mean()
            .rename("traded_win_rate")
            .reset_index()
        )
        grouped = grouped.merge(traded_rates, on="edge_bin", how="left")

    grouped["edge_bin"] = grouped["edge_bin"].astype(str)
    grouped["roi_on_amount_risked"] = grouped.apply(
        lambda row: row["total_profit"] / row["amount_risked"]
        if row["amount_risked"]
        else 0.0,
        axis=1,
    )
    return grouped


def top_backtest_trades(trades: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return traded rows with the largest absolute paper P/L."""

    if trades.empty:
        return pd.DataFrame()

    working = trades.copy()
    working["_trade_bool"] = _coerce_bool(working["trade"])
    traded = working[working["_trade_bool"]].copy()
    if traded.empty:
        return traded
    traded["abs_profit"] = traded["profit"].astype(float).abs()
    traded["result_type"] = np.where(traded["profit"].astype(float) >= 0, "win", "loss")
    return traded.sort_values("abs_profit", ascending=False).head(n).drop(columns=["_trade_bool"])


def _read_csv_with_ids(path: str | Path) -> pd.DataFrame:
    """Read report CSVs while keeping identifier columns as strings."""

    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(
        path,
        dtype={
            "game_id": "string",
            "market_ticker": "string",
            "event_ticker": "string",
        },
    )


def generate_diagnostics(
    predictions_path: str | Path,
    trades_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate diagnostic CSV and JSON artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    predictions = _read_csv_with_ids(predictions_path)
    trades = _read_csv_with_ids(trades_path)

    probability_bins = prediction_probability_bins(predictions)
    season_summary = season_prediction_summary(predictions)
    edge_bins = backtest_edge_bins(trades)
    top_trades = top_backtest_trades(trades)

    paths = {
        "probability_bins": output / "prediction_probability_bins.csv",
        "season_summary": output / "prediction_season_summary.csv",
        "edge_bins": output / "backtest_edge_bins.csv",
        "top_trades": output / "top_backtest_trades.csv",
        "summary": output / "diagnostics_summary.json",
    }
    probability_bins.to_csv(paths["probability_bins"], index=False)
    season_summary.to_csv(paths["season_summary"], index=False)
    edge_bins.to_csv(paths["edge_bins"], index=False)
    top_trades.to_csv(paths["top_trades"], index=False)

    summary = {
        "prediction_rows": int(len(predictions)),
        "trade_rows": int(len(trades)),
        "probability_bins": int(len(probability_bins)),
        "season_rows": int(len(season_summary)),
        "edge_bins": int(len(edge_bins)),
        "top_trade_rows": int(len(top_trades)),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths
