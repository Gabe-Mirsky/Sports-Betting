"""Evaluate spread and total model probabilities against direct Kalshi line markets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.market_type_models import probability_margin_exceeds, probability_total_over


def _to_date_string(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.date.astype(str)


def _price_to_prob(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    if number > 1.0:
        number = number / 100.0
    if number <= 0.0 or number >= 1.0:
        return number
    return number


def _safe_log_loss(actual: pd.Series, predicted: pd.Series) -> float:
    if actual.empty:
        return float("nan")
    y = actual.to_numpy(dtype=float)
    p = np.clip(predicted.to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _yes_probability(row: pd.Series) -> float:
    category = str(row.get("market_category", ""))
    line_value = float(row["line_value"])
    if category == "spread_handicap":
        pred_margin = float(row["pred_home_margin"])
        margin_std = float(row["margin_residual_std_train"])
        yes_team = str(row.get("yes_team_abbr", ""))
        if yes_team == str(row.get("home_team_abbr", "")):
            return probability_margin_exceeds(pred_margin, margin_std, line_value)
        if yes_team == str(row.get("away_team_abbr", "")):
            return 1.0 - probability_margin_exceeds(pred_margin, margin_std, -line_value)
        return float("nan")
    if category == "total_points_over_under":
        over_prob = probability_total_over(
            float(row["pred_total_points"]),
            float(row["total_residual_std_train"]),
            line_value,
        )
        return 1.0 - over_prob if str(row.get("direction", "")).lower() == "under" else over_prob
    return float("nan")


def _actual_yes(row: pd.Series) -> bool:
    category = str(row.get("market_category", ""))
    line_value = float(row["line_value"])
    if category == "spread_handicap":
        margin = float(row["target_home_margin"])
        yes_team = str(row.get("yes_team_abbr", ""))
        if yes_team == str(row.get("home_team_abbr", "")):
            return bool(margin > line_value)
        if yes_team == str(row.get("away_team_abbr", "")):
            return bool(-margin > line_value)
        return False
    if category == "total_points_over_under":
        over = float(row["target_total_points"]) > line_value
        return bool(not over) if str(row.get("direction", "")).lower() == "under" else bool(over)
    return False


def prepare_line_market_model_eval(
    line_prices: pd.DataFrame,
    market_type_predictions: pd.DataFrame,
    snapshot_target: str = "pregame_60m",
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    """Join direct line-market prices to spread/total predictions."""

    if line_prices.empty or market_type_predictions.empty:
        return pd.DataFrame()
    prices = line_prices.copy()
    predictions = market_type_predictions.copy()
    required_price_columns = {
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
        "market_category",
        "line_value",
        "market_ticker",
    }
    required_prediction_columns = {
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
        "pred_home_margin",
        "pred_total_points",
        "margin_residual_std_train",
        "total_residual_std_train",
        "target_home_margin",
        "target_total_points",
    }
    if not required_price_columns.issubset(prices.columns) or not required_prediction_columns.issubset(predictions.columns):
        return pd.DataFrame()

    prices = prices[prices["snapshot_target"].astype(str).eq(snapshot_target)].copy()
    prices = prices[prices["price_quality"].astype(str).ne("missing")].copy()
    prices["line_value"] = pd.to_numeric(prices["line_value"], errors="coerce")
    prices["yes_price"] = pd.to_numeric(prices.get("yes_price"), errors="coerce")
    prices = prices[prices["line_value"].notna() & prices["yes_price"].notna()].copy()
    if prices.empty:
        return pd.DataFrame()

    prices["_game_date_key"] = _to_date_string(prices["game_date"])
    predictions["_game_date_key"] = _to_date_string(predictions["game_date"])
    merged = prices.merge(
        predictions,
        on=["_game_date_key", "home_team_abbr", "away_team_abbr"],
        how="inner",
        suffixes=("", "_prediction"),
    )
    if merged.empty:
        return merged

    merged["market_prob"] = merged["yes_price"].map(_price_to_prob)
    merged["model_yes_prob"] = merged.apply(_yes_probability, axis=1)
    merged = merged[merged["market_prob"].notna() & merged["model_yes_prob"].notna()].copy()
    merged["actual_yes"] = merged.apply(_actual_yes, axis=1)
    merged["edge"] = merged["model_yes_prob"] - merged["market_prob"]
    merged["trade_signal"] = merged["edge"] >= float(edge_threshold)
    merged["profit_per_contract"] = np.where(
        merged["actual_yes"],
        1.0 - merged["market_prob"],
        -merged["market_prob"],
    )
    merged["signal_profit_per_contract"] = np.where(merged["trade_signal"], merged["profit_per_contract"], 0.0)
    return merged.sort_values(["game_date", "market_category", "market_ticker"]).reset_index(drop=True)


def summarize_line_market_model_eval(eval_rows: pd.DataFrame, edge_threshold: float = 0.05) -> dict[str, Any]:
    if eval_rows.empty:
        return {
            "rows": 0,
            "edge_threshold": edge_threshold,
            "status": "not_ready",
            "note": "No direct line-market model evaluation rows are available.",
        }
    summaries = []
    for category, frame in eval_rows.groupby("market_category"):
        actual = frame["actual_yes"].astype(int)
        predicted = frame["model_yes_prob"].astype(float).clip(0.0, 1.0)
        signals = frame[frame["trade_signal"]].copy()
        summaries.append(
            {
                "market_category": str(category),
                "rows": int(len(frame)),
                "signals": int(len(signals)),
                "brier_score": float(np.mean((predicted.to_numpy() - actual.to_numpy()) ** 2)),
                "log_loss": _safe_log_loss(actual, predicted),
                "avg_edge": float(frame["edge"].mean()),
                "signal_win_rate": float(signals["actual_yes"].mean()) if not signals.empty else None,
                "signal_profit_per_contract": float(signals["profit_per_contract"].sum()) if not signals.empty else 0.0,
                "avg_signal_profit_per_contract": float(signals["profit_per_contract"].mean()) if not signals.empty else None,
            }
        )

    timeline_dates = pd.to_datetime(eval_rows["game_date"], errors="coerce").dropna()
    timeline = "n/a"
    if not timeline_dates.empty:
        start = timeline_dates.min().date().isoformat()
        end = timeline_dates.max().date().isoformat()
        timeline = start if start == end else f"{start} to {end}"
    enough_rows = len(eval_rows) >= 100
    positive_signal_economics = any(
        item["signals"] >= 30 and (item["avg_signal_profit_per_contract"] or 0.0) > 0.0 for item in summaries
    )
    failed_checks = []
    if not enough_rows:
        failed_checks.append("needs_at_least_100_direct_line_rows")
    if not positive_signal_economics:
        failed_checks.append("needs_positive_signal_economics")
    status = "watchlist" if enough_rows and positive_signal_economics else "not_ready"
    return {
        "rows": int(len(eval_rows)),
        "timeline": timeline,
        "edge_threshold": edge_threshold,
        "signals": int(eval_rows["trade_signal"].sum()),
        "status": status,
        "failed_checks": failed_checks,
        "by_market_category": summaries,
        "note": (
            "This is an exploratory direct spread/total evaluation. "
            "It is blocked from headline trading until it has enough out-of-sample rows and stable positive signal economics."
        ),
    }


def save_line_market_model_eval(
    eval_rows: pd.DataFrame,
    summary: dict[str, Any],
    rows_path: str | Path,
    summary_path: str | Path,
) -> None:
    rows_output = Path(rows_path)
    summary_output = Path(summary_path)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    eval_rows.to_csv(rows_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
