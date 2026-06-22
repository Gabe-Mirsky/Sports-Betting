"""Price-bucket and month-stability sweeps for CLV-filtered signals."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


DEFAULT_PRICE_BREAKS = [0, 25, 40, 55, 70, 85, 100]
WALK_FORWARD_SIGNAL_COLUMNS = [
    "walk_forward_clv_price_signal",
    "walk_forward_clv_price_rule",
    "walk_forward_clv_price_rule_status",
]
WALK_FORWARD_FOLD_COLUMNS = [
    "test_month",
    "status",
    "train_rows",
    "train_months",
    "test_rows",
    "signals",
    "selected_rule",
    "selected_rule_status",
    "train_rules_tested",
    "train_best_positive_month_share",
    "train_best_positive_clv_rate",
    "test_avg_profit_per_share",
    "test_avg_clv_cents",
    "test_positive_clv_rate",
    "test_total_profit_per_share",
    "test_month_rows",
]
MONTHLY_SUMMARY_COLUMNS = [
    "month",
    "rows",
    "avg_clv_cents",
    "positive_clv_rate",
    "avg_profit_per_share",
    "total_profit_per_share",
]


def _selected_rows(rows: pd.DataFrame, signal_column: str, side: str = "YES") -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    required = ["date", signal_column, "price_cents", "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"CLV concentration rows are missing columns: {missing}")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price_cents"] = pd.to_numeric(frame["price_cents"], errors="coerce")
    frame["clv_cents"] = pd.to_numeric(frame["clv_cents"], errors="coerce")
    frame["realized_profit_per_share"] = pd.to_numeric(frame["realized_profit_per_share"], errors="coerce")
    if "clv_filter_side" in frame.columns:
        side_column = frame["clv_filter_side"]
    elif "calibrated_side" in frame.columns:
        side_column = frame["calibrated_side"]
    elif "candidate_side" in frame.columns:
        side_column = frame["candidate_side"]
    else:
        side_column = pd.Series("YES", index=frame.index)
    frame["_side"] = side_column.fillna("").astype(str).str.upper()
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame = frame[
        frame["_signal"]
        & frame["_side"].eq(str(side).upper())
        & frame["date"].notna()
        & frame["price_cents"].notna()
        & frame["clv_cents"].notna()
        & frame["realized_profit_per_share"].notna()
    ].copy()
    return frame.reset_index(drop=True)


def _month_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=MONTHLY_SUMMARY_COLUMNS)
    working = frame.copy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    return (
        working.groupby("month", as_index=False)
        .agg(
            rows=("month", "size"),
            avg_clv_cents=("clv_cents", "mean"),
            positive_clv_rate=("clv_cents", lambda values: float((values > 0).mean())),
            avg_profit_per_share=("realized_profit_per_share", "mean"),
            total_profit_per_share=("realized_profit_per_share", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )


def _price_ranges(price_breaks: list[float]) -> list[tuple[float, float]]:
    ordered = sorted(set(float(value) for value in price_breaks))
    ranges: list[tuple[float, float]] = []
    for start, end in combinations(ordered, 2):
        if start < end:
            ranges.append((start, end))
    return ranges


def _profit_share_by_month(monthly: pd.DataFrame) -> float:
    if monthly.empty or "total_profit_per_share" not in monthly.columns:
        return 0.0
    profits = pd.to_numeric(monthly["total_profit_per_share"], errors="coerce")
    positive_total = profits[profits > 0].sum()
    if positive_total <= 0:
        return 1.0
    return float(profits.max() / positive_total)


def _status(row: dict[str, Any]) -> str:
    if (
        int(row["rows"]) >= 100
        and int(row["months"]) >= 5
        and float(row["positive_month_share"]) >= 0.80
        and float(row["avg_profit_per_share"]) > 0
        and float(row["avg_clv_cents"]) > 0
        and float(row["positive_clv_rate"]) >= 0.50
        and float(row["max_month_profit_share"]) <= 0.60
    ):
        return "stability_candidate"
    if (
        int(row["rows"]) >= 50
        and int(row["months"]) >= 4
        and float(row["positive_month_share"]) >= 0.60
        and float(row["avg_profit_per_share"]) > 0
        and float(row["avg_clv_cents"]) > 0
    ):
        return "watchlist"
    return "not_ready"


def _score(row: dict[str, Any]) -> float:
    return float(
        (float(row["avg_profit_per_share"]) * 0.50)
        + (float(row["avg_clv_cents"]) / 100.0 * 0.20)
        + (float(row["positive_clv_rate"]) * 0.08)
        + (float(row["positive_month_share"]) * 0.08)
        - (max(0.0, 0.60 - float(row["positive_month_share"])) * 0.15)
        - (max(0.0, float(row["max_month_profit_share"]) - 0.60) * 0.10)
    )


def run_clv_price_month_sweep(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
    side: str = "YES",
    price_breaks: list[float] | None = None,
    min_rows: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sweep contiguous price ranges and score monthly CLV/profit stability."""

    selected = _selected_rows(rows, signal_column=signal_column, side=side)
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "rules_tested": 0, "status": "not_ready"}
    breaks = price_breaks or DEFAULT_PRICE_BREAKS
    rule_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for min_price, max_price in _price_ranges(breaks):
        subset = selected[
            selected["price_cents"].ge(min_price)
            & selected["price_cents"].le(max_price)
        ].copy()
        if len(subset) < int(min_rows):
            continue
        monthly = _month_summary(subset)
        positive_months = int((monthly["avg_profit_per_share"] > 0).sum()) if not monthly.empty else 0
        months = int(len(monthly))
        row = {
            "side": str(side).upper(),
            "min_price_cents": float(min_price),
            "max_price_cents": float(max_price),
            "rows": int(len(subset)),
            "months": months,
            "positive_months": positive_months,
            "positive_month_share": float(positive_months / months) if months else 0.0,
            "avg_clv_cents": float(subset["clv_cents"].mean()),
            "positive_clv_rate": float((subset["clv_cents"] > 0).mean()),
            "avg_profit_per_share": float(subset["realized_profit_per_share"].mean()),
            "total_profit_per_share": float(subset["realized_profit_per_share"].sum()),
            "worst_month_profit_per_share": float(monthly["avg_profit_per_share"].min()) if months else 0.0,
            "worst_month_positive_clv_rate": float(monthly["positive_clv_rate"].min()) if months else 0.0,
            "max_month_profit_share": _profit_share_by_month(monthly),
        }
        row["status"] = _status(row)
        row["score"] = _score(row)
        rule_rows.append(row)
        monthly = monthly.copy()
        monthly.insert(0, "max_price_cents", float(max_price))
        monthly.insert(0, "min_price_cents", float(min_price))
        monthly.insert(0, "side", str(side).upper())
        monthly_frames.append(monthly)

    rules = pd.DataFrame(rule_rows)
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    if rules.empty:
        return rules, monthly_all, {"rows": int(len(selected)), "rules_tested": 0, "status": "not_ready"}
    status_rank = {"stability_candidate": 0, "watchlist": 1, "not_ready": 2}
    rules["_status_rank"] = rules["status"].map(status_rank).fillna(99)
    rules = (
        rules.sort_values(
            ["_status_rank", "score", "rows"],
            ascending=[True, False, False],
        )
        .drop(columns=["_status_rank"])
        .reset_index(drop=True)
    )
    best = rules.iloc[0].to_dict()
    summary = {
        "rows": int(len(selected)),
        "rules_tested": int(len(rules)),
        "stability_candidates": int(rules["status"].eq("stability_candidate").sum()),
        "watchlist_rules": int(rules["status"].eq("watchlist").sum()),
        "not_ready_rules": int(rules["status"].eq("not_ready").sum()),
        "best_status": str(best["status"]),
        "best_rule": f"{str(side).upper()} price {float(best['min_price_cents']):.0f}-{float(best['max_price_cents']):.0f}c",
        "best_rule_rows": int(best["rows"]),
        "best_rule_months": int(best["months"]),
        "best_rule_positive_month_share": float(best["positive_month_share"]),
        "best_rule_avg_clv_cents": float(best["avg_clv_cents"]),
        "best_rule_positive_clv_rate": float(best["positive_clv_rate"]),
        "best_rule_avg_profit_per_share": float(best["avg_profit_per_share"]),
        "best_rule_max_month_profit_share": float(best["max_month_profit_share"]),
        "note": "This sweep is descriptive research. It does not prove edge unless proof gates pass out of sample.",
    }
    return rules, monthly_all, summary


def _apply_price_rule(frame: pd.DataFrame, min_price: float, max_price: float) -> pd.Series:
    return frame["price_cents"].ge(float(min_price)) & frame["price_cents"].le(float(max_price))


def run_walk_forward_clv_price_month_validation(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
    side: str = "YES",
    price_breaks: list[float] | None = None,
    min_rows: int = 25,
    min_train_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate price/month rules with expanding monthly walk-forward selection."""

    selected = _selected_rows(rows, signal_column=signal_column, side=side)
    if selected.empty:
        empty_validated = selected.copy()
        for column in WALK_FORWARD_SIGNAL_COLUMNS:
            if column not in empty_validated.columns:
                empty_validated[column] = pd.Series(dtype=object)
        return (
            empty_validated,
            pd.DataFrame(columns=WALK_FORWARD_FOLD_COLUMNS),
            pd.DataFrame(columns=MONTHLY_SUMMARY_COLUMNS + ["positive_profit_month"]),
            {"rows": 0, "folds": 0, "status": "not_ready"},
        )
    selected["_month"] = selected["date"].dt.to_period("M")
    months = sorted(selected["_month"].dropna().unique())
    validated_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for month in months:
        train = selected[selected["_month"] < month].copy()
        test = selected[selected["_month"] == month].copy()
        train_months = int(train["_month"].nunique()) if not train.empty else 0
        if test.empty:
            continue
        if train_months < int(min_train_months):
            test = test.copy()
            test["walk_forward_clv_price_signal"] = False
            test["walk_forward_clv_price_rule"] = "skipped_insufficient_prior_months"
            test["walk_forward_clv_price_rule_status"] = "skipped"
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

        rules, _, train_summary = run_clv_price_month_sweep(
            train,
            signal_column=signal_column,
            side=side,
            price_breaks=price_breaks,
            min_rows=min_rows,
        )
        if rules.empty:
            test = test.copy()
            test["walk_forward_clv_price_signal"] = False
            test["walk_forward_clv_price_rule"] = "skipped_no_prior_rule"
            test["walk_forward_clv_price_rule_status"] = "skipped"
            validated_frames.append(test)
            fold_rows.append(
                {
                    "test_month": str(month),
                    "status": "skipped_no_prior_rule",
                    "train_rows": int(len(train)),
                    "train_months": train_months,
                    "test_rows": int(len(test)),
                    "signals": 0,
                }
            )
            continue

        best = rules.iloc[0].to_dict()
        signal = _apply_price_rule(test, float(best["min_price_cents"]), float(best["max_price_cents"]))
        test_selected = test[signal].copy()
        monthly = _month_summary(test_selected)
        test_profit = (
            float(test_selected["realized_profit_per_share"].mean())
            if len(test_selected)
            else 0.0
        )
        test_clv = float(test_selected["clv_cents"].mean()) if len(test_selected) else 0.0
        test_positive_clv = float((test_selected["clv_cents"] > 0).mean()) if len(test_selected) else 0.0
        test = test.copy()
        test["walk_forward_clv_price_signal"] = signal
        test["walk_forward_clv_price_rule"] = (
            f"{str(side).upper()} price {float(best['min_price_cents']):.0f}-{float(best['max_price_cents']):.0f}c"
        )
        test["walk_forward_clv_price_rule_status"] = str(best["status"])
        validated_frames.append(test)
        fold_rows.append(
            {
                "test_month": str(month),
                "status": "evaluated",
                "train_rows": int(len(train)),
                "train_months": train_months,
                "test_rows": int(len(test)),
                "signals": int(signal.sum()),
                "selected_rule": test["walk_forward_clv_price_rule"].iloc[0],
                "selected_rule_status": str(best["status"]),
                "train_rules_tested": int(train_summary.get("rules_tested", 0) or 0),
                "train_best_positive_month_share": float(
                    train_summary.get("best_rule_positive_month_share", 0.0) or 0.0
                ),
                "train_best_positive_clv_rate": float(train_summary.get("best_rule_positive_clv_rate", 0.0) or 0.0),
                "test_avg_profit_per_share": test_profit,
                "test_avg_clv_cents": test_clv,
                "test_positive_clv_rate": test_positive_clv,
                "test_total_profit_per_share": float(test_selected["realized_profit_per_share"].sum())
                if len(test_selected)
                else 0.0,
                "test_month_rows": int(monthly["rows"].sum()) if not monthly.empty else 0,
            }
        )

    validated = pd.concat(validated_frames, ignore_index=True, sort=False) if validated_frames else pd.DataFrame()
    if "_month" in validated.columns:
        validated = validated.drop(columns=["_month"])
    folds = pd.DataFrame(fold_rows, columns=WALK_FORWARD_FOLD_COLUMNS)
    selected_validated = (
        validated[validated["walk_forward_clv_price_signal"].fillna(False)].copy()
        if not validated.empty and "walk_forward_clv_price_signal" in validated.columns
        else pd.DataFrame()
    )
    monthly = _month_summary(selected_validated)
    if not monthly.empty:
        monthly["positive_profit_month"] = monthly["avg_profit_per_share"] > 0
    positive_months = int((monthly["avg_profit_per_share"] > 0).sum()) if not monthly.empty else 0
    months_seen = int(len(monthly))
    positive_month_share = float(positive_months / months_seen) if months_seen else 0.0
    avg_profit = (
        float(selected_validated["realized_profit_per_share"].mean())
        if len(selected_validated)
        else 0.0
    )
    avg_clv = float(selected_validated["clv_cents"].mean()) if len(selected_validated) else 0.0
    positive_clv_rate = (
        float((selected_validated["clv_cents"] > 0).mean())
        if len(selected_validated)
        else 0.0
    )
    status = "walk_forward_candidate" if (
        len(selected_validated) >= 100
        and months_seen >= 3
        and positive_month_share >= 0.67
        and avg_profit > 0
        and avg_clv > 0
        and positive_clv_rate >= 0.50
    ) else "not_ready"
    summary = {
        "rows": int(len(selected)),
        "folds": int(len(folds)),
        "evaluated_months": int(folds["status"].eq("evaluated").sum()) if not folds.empty else 0,
        "skipped_months": int(folds["status"].astype(str).str.startswith("skipped").sum()) if not folds.empty else 0,
        "signals": int(len(selected_validated)),
        "months": months_seen,
        "positive_months": positive_months,
        "positive_month_share": positive_month_share,
        "avg_profit_per_share": avg_profit,
        "avg_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv_rate,
        "status": status,
        "parlay_ready": False,
        "min_train_months": int(min_train_months),
        "min_rows": int(min_rows),
        "note": "Nested walk-forward validation: each test month uses a price rule selected only from prior months.",
    }
    return validated.reset_index(drop=True), folds.reset_index(drop=True), monthly.reset_index(drop=True), summary


def save_clv_price_month_sweep_outputs(
    rules: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: dict[str, Any],
    rules_path: str | Path,
    monthly_path: str | Path,
    summary_path: str | Path,
) -> None:
    rules_output = Path(rules_path)
    monthly_output = Path(monthly_path)
    summary_output = Path(summary_path)
    rules_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(rules_output, index=False)
    monthly.to_csv(monthly_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_walk_forward_clv_price_month_outputs(
    validated: pd.DataFrame,
    folds: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: dict[str, Any],
    validated_path: str | Path,
    folds_path: str | Path,
    monthly_path: str | Path,
    summary_path: str | Path,
) -> None:
    validated_output = Path(validated_path)
    folds_output = Path(folds_path)
    monthly_output = Path(monthly_path)
    summary_output = Path(summary_path)
    validated_output.parent.mkdir(parents=True, exist_ok=True)
    folds_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(validated_output, index=False)
    folds.to_csv(folds_output, index=False)
    monthly.to_csv(monthly_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
