"""Walk-forward research guardrails for calibrated NO signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


POLICIES = [
    {"policy": "no_all", "min_price": 0.0, "max_price": 100.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_price_10_20", "min_price": 10.0, "max_price": 20.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_price_10_30", "min_price": 10.0, "max_price": 30.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_price_10_40", "min_price": 10.0, "max_price": 40.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_price_20_40", "min_price": 20.0, "max_price": 40.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_price_le_30", "min_price": 0.0, "max_price": 30.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_price_le_40", "min_price": 0.0, "max_price": 40.0, "min_forecast": 0.0, "max_forecast": 1.0},
    {"policy": "no_forecast_le_30", "min_price": 0.0, "max_price": 100.0, "min_forecast": 0.0, "max_forecast": 0.30},
    {"policy": "no_forecast_le_40", "min_price": 0.0, "max_price": 100.0, "min_forecast": 0.0, "max_forecast": 0.40},
    {"policy": "no_forecast_20_40", "min_price": 0.0, "max_price": 100.0, "min_forecast": 0.20, "max_forecast": 0.40},
    {"policy": "no_price_10_40_forecast_le_40", "min_price": 10.0, "max_price": 40.0, "min_forecast": 0.0, "max_forecast": 0.40},
    {
        "policy": "no_player_edge_higher",
        "min_price": 0.0,
        "max_price": 100.0,
        "min_forecast": 0.0,
        "max_forecast": 1.0,
        "require_player_edge_higher": True,
    },
    {
        "policy": "no_same_side_player_edge_higher",
        "min_price": 0.0,
        "max_price": 100.0,
        "min_forecast": 0.0,
        "max_forecast": 1.0,
        "require_player_edge_higher": True,
        "require_same_side": True,
    },
    {
        "policy": "no_price_10_40_player_edge_higher",
        "min_price": 10.0,
        "max_price": 40.0,
        "min_forecast": 0.0,
        "max_forecast": 1.0,
        "require_player_edge_higher": True,
    },
    {
        "policy": "no_forecast_20_40_player_edge_higher",
        "min_price": 0.0,
        "max_price": 100.0,
        "min_forecast": 0.20,
        "max_forecast": 0.40,
        "require_player_edge_higher": True,
    },
]


def _prepare(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    required = ["date", signal_column, "calibrated_side", "price_cents", "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"NO guardrail rows are missing columns: {missing}")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["_side"] = frame["calibrated_side"].fillna("").astype(str).str.upper()
    for column in [
        "price_cents",
        "clv_cents",
        "realized_profit_per_share",
        "calibrated_win_rate",
        "edge",
        "volume",
        "edge_delta_vs_team",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "calibrated_win_rate" not in frame.columns:
        frame["calibrated_win_rate"] = np.nan
    signals = frame[
        frame["_signal"]
        & frame["_side"].eq("NO")
        & frame["date"].notna()
        & frame["price_cents"].notna()
        & frame["clv_cents"].notna()
        & frame["realized_profit_per_share"].notna()
    ].copy()
    signals["month"] = signals["date"].dt.to_period("M")
    if "actual_contract_win" in signals.columns:
        signals["actual_contract_win_bool"] = _coerce_bool(signals["actual_contract_win"])
    else:
        signals["actual_contract_win_bool"] = signals["realized_profit_per_share"] > 0
    return signals.reset_index(drop=True)


def _apply_policy(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    price = pd.to_numeric(frame["price_cents"], errors="coerce")
    forecast = pd.to_numeric(frame["calibrated_win_rate"], errors="coerce")
    mask = (
        price.ge(float(policy["min_price"]))
        & price.le(float(policy["max_price"]))
        & forecast.ge(float(policy["min_forecast"]))
        & forecast.le(float(policy["max_forecast"]))
    ).fillna(False)
    if bool(policy.get("require_player_edge_higher", False)):
        if "player_edge_higher" in frame.columns:
            player_edge_higher = _coerce_bool(frame["player_edge_higher"])
        elif "edge_delta_vs_team" in frame.columns:
            player_edge_higher = pd.to_numeric(frame["edge_delta_vs_team"], errors="coerce").gt(0)
        else:
            player_edge_higher = pd.Series(False, index=frame.index)
        mask = mask & player_edge_higher.fillna(False)
    if bool(policy.get("require_same_side", False)):
        same_side = _coerce_bool(frame["same_side"]) if "same_side" in frame.columns else pd.Series(False, index=frame.index)
        mask = mask & same_side.fillna(False)
    return mask


def _metrics(frame: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy_name,
            "signals": 0,
            "months": 0,
            "avg_forecast_win_rate": 0.0,
            "actual_win_rate": 0.0,
            "calibration_error": 0.0,
            "avg_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "avg_profit_per_share": 0.0,
            "positive_month_share": 0.0,
            "score": -999.0,
            "status": "not_ready",
        }
    monthly = frame.groupby("month", as_index=False).agg(avg_clv_cents=("clv_cents", "mean"))
    avg_forecast = float(frame["calibrated_win_rate"].mean())
    actual_rate = float(frame["actual_contract_win_bool"].mean())
    avg_clv = float(frame["clv_cents"].mean())
    positive_clv_rate = float((frame["clv_cents"] > 0).mean())
    avg_profit = float(frame["realized_profit_per_share"].mean())
    positive_month_share = float((monthly["avg_clv_cents"] > 0).mean()) if not monthly.empty else 0.0
    calibration_error = actual_rate - avg_forecast
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
        if signals >= 30 and avg_clv > 0 and avg_profit > 0
        else "not_ready"
    )
    score = float(
        avg_profit * 0.35
        + (avg_clv / 100.0) * 0.25
        + positive_clv_rate * 0.20
        + positive_month_share * 0.15
        - abs(calibration_error) * 0.05
    )
    return {
        "policy": policy_name,
        "signals": signals,
        "months": months,
        "avg_forecast_win_rate": avg_forecast,
        "actual_win_rate": actual_rate,
        "calibration_error": calibration_error,
        "avg_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv_rate,
        "avg_profit_per_share": avg_profit,
        "positive_month_share": positive_month_share,
        "score": score,
        "status": status,
    }


def _evaluate_policies(frame: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        selected = frame[_apply_policy(frame, policy)].copy()
        metrics = _metrics(selected, str(policy["policy"]))
        if int(metrics["signals"]) >= int(min_rows):
            rows.append(metrics)
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=list(_metrics(pd.DataFrame(), "none").keys()))
    rank = {"research_candidate": 0, "watchlist": 1, "not_ready": 2}
    output["_rank"] = output["status"].map(rank).fillna(99)
    return output.sort_values(["_rank", "score", "signals"], ascending=[True, False, False]).drop(
        columns="_rank"
    ).reset_index(drop=True)


def run_no_calibration_guardrail_research(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_train_months: int = 2,
    min_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate NO guardrail policies descriptively and with monthly walk-forward selection."""

    signals = _prepare(rows, signal_column=signal_column)
    descriptive = _evaluate_policies(signals, min_rows=min_rows)
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
            test["no_guardrail_signal"] = False
            test["no_guardrail_policy"] = "skipped_insufficient_prior_months"
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

        train_rules = _evaluate_policies(train, min_rows=min_rows)
        if train_rules.empty:
            test = test.copy()
            test["no_guardrail_signal"] = False
            test["no_guardrail_policy"] = "skipped_no_prior_policy"
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

        best_policy_name = str(train_rules.iloc[0]["policy"])
        policy = next(policy for policy in POLICIES if policy["policy"] == best_policy_name)
        mask = _apply_policy(test, policy)
        selected = test[mask].copy()
        test = test.copy()
        test["no_guardrail_signal"] = mask
        test["no_guardrail_policy"] = best_policy_name
        validated_frames.append(test)
        metrics = _metrics(selected, best_policy_name)
        fold_rows.append(
            {
                "test_month": str(month),
                "status": "evaluated",
                "train_rows": int(len(train)),
                "train_months": train_months,
                "test_rows": int(len(test)),
                "selected_policy": best_policy_name,
                "train_policy_status": str(train_rules.iloc[0]["status"]),
                "train_policy_signals": int(train_rules.iloc[0]["signals"]),
                "signals": int(metrics["signals"]),
                "test_avg_forecast_win_rate": float(metrics["avg_forecast_win_rate"]),
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
        validated[validated["no_guardrail_signal"].fillna(False)].copy()
        if not validated.empty and "no_guardrail_signal" in validated.columns
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
        "note": "Research-only NO guardrail sweep. Do not use as betting logic unless proof gates pass out of sample.",
    }
    return descriptive, validated.reset_index(drop=True), folds.reset_index(drop=True), summary


def save_no_calibration_guardrail_outputs(
    descriptive: pd.DataFrame,
    validated: pd.DataFrame,
    folds: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "no_calibration_guardrail",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(output_root / f"{prefix}_descriptive.csv", index=False)
    validated.to_csv(output_root / f"{prefix}_walk_forward_rows.csv", index=False)
    folds.to_csv(output_root / f"{prefix}_walk_forward_folds.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
