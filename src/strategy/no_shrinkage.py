"""Research-only conservative shrinkage for calibrated NO signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


PRICE_ZONES = [
    {"price_zone": "all", "min_price": 0.0, "max_price": 100.0},
    {"price_zone": "price_10_40", "min_price": 10.0, "max_price": 40.0},
    {"price_zone": "price_10_30", "min_price": 10.0, "max_price": 30.0},
]

FORECAST_ZONES = [
    {"forecast_zone": "all", "min_forecast": 0.0, "max_forecast": 1.0},
    {"forecast_zone": "forecast_20_40", "min_forecast": 0.20, "max_forecast": 0.40},
    {"forecast_zone": "forecast_le_40", "min_forecast": 0.0, "max_forecast": 0.40},
]


def _settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for price_zone in PRICE_ZONES:
        for forecast_zone in FORECAST_ZONES:
            for edge_multiplier in [1.0, 0.75, 0.50, 0.25]:
                for win_haircut in [0.0, 0.025, 0.05, 0.075, 0.10]:
                    for min_adjusted_profit in [0.02, 0.04, 0.06]:
                        settings.append(
                            {
                                **price_zone,
                                **forecast_zone,
                                "edge_multiplier": edge_multiplier,
                                "win_haircut": win_haircut,
                                "min_adjusted_profit": min_adjusted_profit,
                                "policy": (
                                    f"{price_zone['price_zone']}|{forecast_zone['forecast_zone']}|"
                                    f"edge_x{edge_multiplier:g}|haircut_{win_haircut:.3f}|"
                                    f"min_profit_{min_adjusted_profit:.2f}"
                                ),
                            }
                        )
    return settings


SETTINGS = _settings()


def _prepare(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    required = [
        "date",
        signal_column,
        "calibrated_side",
        "price_cents",
        "calibrated_win_rate",
        "clv_cents",
        "realized_profit_per_share",
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"NO shrinkage rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["_side"] = frame["calibrated_side"].fillna("").astype(str).str.upper()
    for column in [
        "price_cents",
        "calibrated_win_rate",
        "clv_cents",
        "realized_profit_per_share",
        "contract_cost",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else np.nan
    frame["contract_cost"] = frame["contract_cost"].fillna(frame["price_cents"] / 100.0)
    signals = frame[
        frame["_signal"]
        & frame["_side"].eq("NO")
        & frame["date"].notna()
        & frame["price_cents"].notna()
        & frame["calibrated_win_rate"].notna()
        & frame["contract_cost"].notna()
        & frame["clv_cents"].notna()
        & frame["realized_profit_per_share"].notna()
    ].copy()
    signals["month"] = signals["date"].dt.to_period("M")
    if "actual_contract_win" in signals.columns:
        signals["actual_contract_win_bool"] = _coerce_bool(signals["actual_contract_win"])
    else:
        signals["actual_contract_win_bool"] = signals["realized_profit_per_share"] > 0
    return signals.reset_index(drop=True)


def _apply_setting(frame: pd.DataFrame, setting: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    forecast = pd.to_numeric(frame["calibrated_win_rate"], errors="coerce")
    cost = pd.to_numeric(frame["contract_cost"], errors="coerce")
    price = pd.to_numeric(frame["price_cents"], errors="coerce")
    raw_edge = forecast - cost
    adjusted_win_rate = cost + raw_edge * float(setting["edge_multiplier"]) - float(setting["win_haircut"])
    adjusted_win_rate = adjusted_win_rate.clip(lower=0.0, upper=1.0)
    adjusted_profit = adjusted_win_rate - cost
    mask = (
        price.ge(float(setting["min_price"]))
        & price.le(float(setting["max_price"]))
        & forecast.ge(float(setting["min_forecast"]))
        & forecast.le(float(setting["max_forecast"]))
        & adjusted_profit.ge(float(setting["min_adjusted_profit"]))
    ).fillna(False)
    return mask, adjusted_win_rate, adjusted_profit


def _metrics(frame: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy_name,
            "signals": 0,
            "months": 0,
            "avg_adjusted_win_rate": 0.0,
            "actual_win_rate": 0.0,
            "calibration_error": 0.0,
            "avg_adjusted_profit_per_share": 0.0,
            "avg_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "avg_profit_per_share": 0.0,
            "positive_month_share": 0.0,
            "score": -999.0,
            "status": "not_ready",
        }
    monthly = frame.groupby("month", as_index=False).agg(avg_clv_cents=("clv_cents", "mean"))
    avg_adjusted = float(frame["no_shrink_adjusted_win_rate"].mean())
    actual_rate = float(frame["actual_contract_win_bool"].mean())
    calibration_error = actual_rate - avg_adjusted
    avg_adjusted_profit = float(frame["no_shrink_adjusted_profit_per_share"].mean())
    avg_clv = float(frame["clv_cents"].mean())
    positive_clv_rate = float((frame["clv_cents"] > 0).mean())
    avg_profit = float(frame["realized_profit_per_share"].mean())
    positive_month_share = float((monthly["avg_clv_cents"] > 0).mean()) if not monthly.empty else 0.0
    signals = int(len(frame))
    months = int(frame["month"].nunique())
    status = (
        "research_candidate"
        if signals >= 100
        and months >= 3
        and avg_clv > 0
        and positive_clv_rate >= 0.50
        and avg_profit > 0
        and abs(calibration_error) <= 0.05
        else "watchlist"
        if signals >= 30 and avg_clv > 0 and avg_profit > 0 and abs(calibration_error) <= 0.08
        else "not_ready"
    )
    score = float(
        avg_profit * 0.30
        + avg_adjusted_profit * 0.15
        + (avg_clv / 100.0) * 0.25
        + positive_clv_rate * 0.20
        + positive_month_share * 0.15
        - abs(calibration_error) * 0.10
    )
    return {
        "policy": policy_name,
        "signals": signals,
        "months": months,
        "avg_adjusted_win_rate": avg_adjusted,
        "actual_win_rate": actual_rate,
        "calibration_error": calibration_error,
        "avg_adjusted_profit_per_share": avg_adjusted_profit,
        "avg_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv_rate,
        "avg_profit_per_share": avg_profit,
        "positive_month_share": positive_month_share,
        "score": score,
        "status": status,
    }


def _evaluate_settings(frame: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        mask, adjusted_win_rate, adjusted_profit = _apply_setting(frame, setting)
        selected = frame[mask].copy()
        if selected.empty:
            metrics = _metrics(selected, str(setting["policy"]))
        else:
            selected["no_shrink_adjusted_win_rate"] = adjusted_win_rate.loc[selected.index]
            selected["no_shrink_adjusted_profit_per_share"] = adjusted_profit.loc[selected.index]
            metrics = _metrics(selected, str(setting["policy"]))
        if int(metrics["signals"]) >= int(min_rows):
            rows.append({**setting, **metrics})
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=list(_metrics(pd.DataFrame(), "none").keys()))
    rank = {"research_candidate": 0, "watchlist": 1, "not_ready": 2}
    output["_rank"] = output["status"].map(rank).fillna(99)
    return output.sort_values(["_rank", "score", "signals"], ascending=[True, False, False]).drop(
        columns="_rank"
    ).reset_index(drop=True)


def run_no_shrinkage_research(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_train_months: int = 2,
    min_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Walk-forward test conservative NO probability haircuts without promoting them to live logic."""

    signals = _prepare(rows, signal_column=signal_column)
    descriptive = _evaluate_settings(signals, min_rows=min_rows)
    if signals.empty:
        return descriptive, pd.DataFrame(), pd.DataFrame(), {
            "status": "no_signals",
            "signals": 0,
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
        }

    validated_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for month in sorted(signals["month"].dropna().unique()):
        train = signals[signals["month"] < month].copy()
        test = signals[signals["month"] == month].copy()
        train_months = int(train["month"].nunique()) if not train.empty else 0
        if train_months < int(min_train_months):
            test = test.copy()
            test["no_shrinkage_signal"] = False
            test["no_shrinkage_policy"] = "skipped_insufficient_prior_months"
            validated_frames.append(test)
            fold_rows.append(
                {
                    "test_month": str(month),
                    "status": "skipped_insufficient_prior_months",
                    "train_rows": int(len(train)),
                    "train_months": train_months,
                    "test_rows": int(len(test)),
                    "signals": 0,
                }
            )
            continue

        train_settings = _evaluate_settings(train, min_rows=min_rows)
        if train_settings.empty:
            test = test.copy()
            test["no_shrinkage_signal"] = False
            test["no_shrinkage_policy"] = "skipped_no_prior_policy"
            validated_frames.append(test)
            fold_rows.append(
                {
                    "test_month": str(month),
                    "status": "skipped_no_prior_policy",
                    "train_rows": int(len(train)),
                    "train_months": train_months,
                    "test_rows": int(len(test)),
                    "signals": 0,
                }
            )
            continue

        best_policy_name = str(train_settings.iloc[0]["policy"])
        setting = next(setting for setting in SETTINGS if setting["policy"] == best_policy_name)
        mask, adjusted_win_rate, adjusted_profit = _apply_setting(test, setting)
        test = test.copy()
        test["no_shrinkage_signal"] = mask
        test["no_shrinkage_policy"] = best_policy_name
        test["no_shrink_adjusted_win_rate"] = adjusted_win_rate
        test["no_shrink_adjusted_profit_per_share"] = adjusted_profit
        validated_frames.append(test)
        selected = test[mask].copy()
        metrics = _metrics(selected, best_policy_name)
        fold_rows.append(
            {
                "test_month": str(month),
                "status": "evaluated",
                "train_rows": int(len(train)),
                "train_months": train_months,
                "test_rows": int(len(test)),
                "selected_policy": best_policy_name,
                "train_policy_status": str(train_settings.iloc[0]["status"]),
                "train_policy_signals": int(train_settings.iloc[0]["signals"]),
                "signals": int(metrics["signals"]),
                "test_avg_adjusted_win_rate": float(metrics["avg_adjusted_win_rate"]),
                "test_actual_win_rate": float(metrics["actual_win_rate"]),
                "test_calibration_error": float(metrics["calibration_error"]),
                "test_avg_clv_cents": float(metrics["avg_clv_cents"]),
                "test_positive_clv_rate": float(metrics["positive_clv_rate"]),
                "test_avg_profit_per_share": float(metrics["avg_profit_per_share"]),
            }
        )

    validated = pd.concat(validated_frames, ignore_index=True, sort=False) if validated_frames else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    selected_validated = (
        validated[validated["no_shrinkage_signal"].fillna(False)].copy()
        if not validated.empty and "no_shrinkage_signal" in validated.columns
        else pd.DataFrame()
    )
    walk_metrics = _metrics(selected_validated, "walk_forward_selected")
    status = (
        "research_candidate"
        if int(walk_metrics["signals"]) >= 100
        and int(walk_metrics["months"]) >= 3
        and float(walk_metrics["avg_clv_cents"]) > 0
        and float(walk_metrics["positive_clv_rate"]) >= 0.50
        and float(walk_metrics["avg_profit_per_share"]) > 0
        and abs(float(walk_metrics["calibration_error"])) <= 0.05
        else "not_ready"
    )
    summary = {
        **walk_metrics,
        "status": status,
        "evaluated_months": int(folds["status"].eq("evaluated").sum()) if not folds.empty else 0,
        "skipped_months": int(folds["status"].astype(str).str.startswith("skipped").sum()) if not folds.empty else 0,
        "descriptive_best_policy": str(descriptive.iloc[0]["policy"]) if not descriptive.empty else "n/a",
        "descriptive_best_status": str(descriptive.iloc[0]["status"]) if not descriptive.empty else "n/a",
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "note": (
            "Research-only NO shrinkage sweep. It tests conservative probability haircuts for NO candidates "
            "and must not be promoted unless out-of-sample CLV, profit, and repeatability gates pass."
        ),
    }
    return descriptive, validated.reset_index(drop=True), folds.reset_index(drop=True), summary


def save_no_shrinkage_outputs(
    descriptive: pd.DataFrame,
    validated: pd.DataFrame,
    folds: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "no_shrinkage",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(output_root / f"{prefix}_descriptive.csv", index=False)
    validated.to_csv(output_root / f"{prefix}_walk_forward_rows.csv", index=False)
    folds.to_csv(output_root / f"{prefix}_walk_forward_folds.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
