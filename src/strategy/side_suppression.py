"""Research-only side-suppression tests for calibrated single-game signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


POLICIES = [
    {"policy": "all_calibrated", "allowed_sides": {"YES", "NO"}, "min_yes_edge": -np.inf},
    {"policy": "no_only", "allowed_sides": {"NO"}, "min_yes_edge": np.inf},
    {"policy": "yes_only", "allowed_sides": {"YES"}, "min_yes_edge": -np.inf},
    {"policy": "no_plus_yes_edge_ge_2pct", "allowed_sides": {"YES", "NO"}, "min_yes_edge": 0.02},
    {"policy": "no_plus_yes_edge_ge_5pct", "allowed_sides": {"YES", "NO"}, "min_yes_edge": 0.05},
    {"policy": "no_plus_yes_edge_ge_8pct", "allowed_sides": {"YES", "NO"}, "min_yes_edge": 0.08},
    {"policy": "no_plus_yes_edge_ge_12pct", "allowed_sides": {"YES", "NO"}, "min_yes_edge": 0.12},
]


def _side_series(frame: pd.DataFrame) -> pd.Series:
    if "calibrated_side" in frame.columns:
        side = frame["calibrated_side"]
    elif "candidate_side" in frame.columns:
        side = frame["candidate_side"]
    elif "side" in frame.columns:
        side = frame["side"]
    else:
        side = pd.Series("YES", index=frame.index)
    side = side.fillna("").astype(str).str.upper()
    return side.where(side.isin({"YES", "NO"}), "YES")


def _prepare(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    required = ["date", signal_column, "clv_cents", "realized_profit_per_share", "price_cents"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Side-suppression rows are missing columns: {missing}")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["_side"] = _side_series(frame)
    for column in ["clv_cents", "realized_profit_per_share", "price_cents", "edge", "volume", "open_interest"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["_signal"]
        & frame["date"].notna()
        & frame["clv_cents"].notna()
        & frame["realized_profit_per_share"].notna()
        & frame["price_cents"].notna()
    ].copy()
    if "edge" not in frame.columns:
        frame["edge"] = np.nan
    frame["month"] = frame["date"].dt.to_period("M")
    return frame.reset_index(drop=True)


def _apply_policy(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    side = frame["_side"].astype(str).str.upper()
    allowed = side.isin(policy["allowed_sides"])
    min_yes_edge = float(policy["min_yes_edge"])
    if np.isposinf(min_yes_edge):
        return allowed & side.ne("YES")
    if np.isneginf(min_yes_edge):
        return allowed
    edge = pd.to_numeric(frame["edge"], errors="coerce")
    yes_ok = side.ne("YES") | edge.ge(min_yes_edge)
    return allowed & yes_ok.fillna(False)


def _metrics(frame: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy_name,
            "signals": 0,
            "months": 0,
            "avg_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "avg_profit_per_share": 0.0,
            "positive_month_share": 0.0,
            "yes_signals": 0,
            "no_signals": 0,
            "status": "not_ready",
            "score": -999.0,
        }
    monthly = (
        frame.assign(month=frame["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(avg_profit_per_share=("realized_profit_per_share", "mean"), avg_clv_cents=("clv_cents", "mean"))
    )
    positive_month_share = float((monthly["avg_clv_cents"] > 0).mean()) if not monthly.empty else 0.0
    avg_clv = float(frame["clv_cents"].mean())
    positive_clv = float((frame["clv_cents"] > 0).mean())
    avg_profit = float(frame["realized_profit_per_share"].mean())
    signals = int(len(frame))
    months = int(frame["month"].nunique())
    status = (
        "research_candidate"
        if signals >= 100 and months >= 3 and avg_clv > 0 and positive_clv >= 0.50 and avg_profit > 0
        else "watchlist"
        if signals >= 50 and avg_clv > 0 and avg_profit > 0
        else "not_ready"
    )
    score = float(avg_profit * 0.40 + avg_clv / 100.0 * 0.25 + positive_clv * 0.20 + positive_month_share * 0.15)
    return {
        "policy": policy_name,
        "signals": signals,
        "months": months,
        "avg_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv,
        "avg_profit_per_share": avg_profit,
        "positive_month_share": positive_month_share,
        "yes_signals": int(frame["_side"].eq("YES").sum()),
        "no_signals": int(frame["_side"].eq("NO").sum()),
        "status": status,
        "score": score,
    }


def _evaluate_policies(frame: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
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
    return output.sort_values(["_rank", "score", "signals"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)


def run_side_suppression_research(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_train_months: int = 2,
    min_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate side-suppression policies in-sample and with monthly walk-forward selection."""

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
        if test.empty:
            continue
        if train_months < int(min_train_months):
            test = test.copy()
            test["side_suppression_signal"] = False
            test["side_suppression_policy"] = "skipped_insufficient_prior_months"
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
            test["side_suppression_signal"] = False
            test["side_suppression_policy"] = "skipped_no_prior_policy"
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
        test["side_suppression_signal"] = mask
        test["side_suppression_policy"] = best_policy_name
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
                "test_avg_clv_cents": float(metrics["avg_clv_cents"]),
                "test_positive_clv_rate": float(metrics["positive_clv_rate"]),
                "test_avg_profit_per_share": float(metrics["avg_profit_per_share"]),
                "test_yes_signals": int(metrics["yes_signals"]),
                "test_no_signals": int(metrics["no_signals"]),
            }
        )

    validated = pd.concat(validated_frames, ignore_index=True, sort=False) if validated_frames else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    selected_validated = (
        validated[validated["side_suppression_signal"].fillna(False)].copy()
        if not validated.empty and "side_suppression_signal" in validated.columns
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
        "note": "Research-only side suppression. Do not use as betting logic unless proof gates pass out of sample.",
    }
    return descriptive, validated.reset_index(drop=True), folds.reset_index(drop=True), summary


def save_side_suppression_outputs(
    descriptive: pd.DataFrame,
    validated: pd.DataFrame,
    folds: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "side_suppression",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(output_root / f"{prefix}_descriptive.csv", index=False)
    validated.to_csv(output_root / f"{prefix}_walk_forward_trades.csv", index=False)
    folds.to_csv(output_root / f"{prefix}_walk_forward_folds.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
