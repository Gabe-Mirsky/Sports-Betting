"""Defensive non-team filters for CLV-filtered single-game signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


DEFAULT_MIN_PRICE_CENTS = [5.0, 10.0, 15.0]
DEFAULT_MAX_PRICE_CENTS = [40.0, 55.0, 100.0]
DEFAULT_MIN_ROIS = [0.0, 0.5]
DEFAULT_MAX_ROIS = [2.0, 3.0, 5.0, 10.0]
DEFAULT_MAX_VOLUMES = [100.0, 1000.0, 10000.0, np.inf]
MONTHLY_SUMMARY_COLUMNS = [
    "month",
    "rows",
    "avg_clv_cents",
    "positive_clv_rate",
    "avg_profit_per_share",
    "total_profit_per_share",
]
WALK_FORWARD_SIGNAL_COLUMNS = [
    "walk_forward_defensive_signal",
    "walk_forward_defensive_rule",
    "walk_forward_defensive_rule_status",
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
    "train_best_positive_clv_rate",
    "train_best_positive_month_share",
    "test_avg_profit_per_share",
    "test_avg_clv_cents",
    "test_positive_clv_rate",
    "test_total_profit_per_share",
]


def add_defensive_filters(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
    min_price_cents: float = 10.0,
    max_price_cents: float = 100.0,
    min_calibrated_expected_roi: float = 0.0,
    max_calibrated_expected_roi: float = 3.0,
    min_volume: float = 10.0,
    max_volume: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Block decay-prone price, ROI, and liquidity slices before portfolio selection."""

    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "defensive_trades": 0}
    required = [signal_column, "price_cents", "calibrated_expected_roi", "volume", "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Defensive filter rows are missing columns: {missing}")

    frame = rows.copy()
    frame["_base_signal"] = _coerce_bool(frame[signal_column])
    for column in ["price_cents", "calibrated_expected_roi", "volume", "clv_cents", "realized_profit_per_share"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    checks = {
        "price_below_minimum": frame["price_cents"].lt(float(min_price_cents)),
        "price_above_maximum": frame["price_cents"].gt(float(max_price_cents)),
        "calibrated_roi_below_minimum": frame["calibrated_expected_roi"].lt(float(min_calibrated_expected_roi)),
        "calibrated_roi_too_high": frame["calibrated_expected_roi"].ge(float(max_calibrated_expected_roi)),
        "volume_below_minimum": frame["volume"].lt(float(min_volume)),
        "volume_above_maximum": frame["volume"].ge(float(max_volume)),
    }
    pass_filter = frame["_base_signal"].copy()
    for failed in checks.values():
        pass_filter = pass_filter & ~failed.fillna(True)
    frame["defensive_trade"] = pass_filter.fillna(False)

    reasons: list[str] = []
    for index, row in frame.iterrows():
        if not bool(row["_base_signal"]):
            reasons.append("base_signal_false")
            continue
        failed_reasons = [name for name, failed in checks.items() if bool(failed.loc[index])]
        reasons.append("defensive_filter_passed" if not failed_reasons else ",".join(failed_reasons))
    frame["defensive_filter_reason"] = reasons

    base = frame[frame["_base_signal"]].copy()
    selected = frame[frame["defensive_trade"]].copy()
    audit_rows: list[dict[str, Any]] = []
    for reason, group in frame[frame["_base_signal"]].groupby("defensive_filter_reason", dropna=False):
        audit_rows.append(
            {
                "reason": str(reason),
                "rows": int(len(group)),
                "avg_clv_cents": float(group["clv_cents"].mean()) if len(group) else 0.0,
                "positive_clv_rate": float((group["clv_cents"] > 0).mean()) if len(group) else 0.0,
                "avg_profit_per_share": float(group["realized_profit_per_share"].mean()) if len(group) else 0.0,
            }
        )
    audit_columns = ["reason", "rows", "avg_clv_cents", "positive_clv_rate", "avg_profit_per_share"]
    audit = pd.DataFrame(audit_rows, columns=audit_columns)
    if not audit.empty:
        audit = audit.sort_values(["reason"]).reset_index(drop=True)

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        selected_dates = pd.to_datetime(selected["date"], errors="coerce").dropna() if len(selected) else pd.Series(dtype="datetime64[ns]")
    else:
        selected_dates = pd.Series(dtype="datetime64[ns]")

    summary = {
        "rows": int(len(frame)),
        "base_signals": int(len(base)),
        "defensive_trades": int(len(selected)),
        "blocked_trades": int(len(base) - len(selected)),
        "avg_clv_cents": float(selected["clv_cents"].mean()) if len(selected) else 0.0,
        "positive_clv_rate": float((selected["clv_cents"] > 0).mean()) if len(selected) else 0.0,
        "avg_profit_per_share": float(selected["realized_profit_per_share"].mean()) if len(selected) else 0.0,
        "trade_start_date": selected_dates.min().date().isoformat() if not selected_dates.empty else None,
        "trade_end_date": selected_dates.max().date().isoformat() if not selected_dates.empty else None,
        "rules": {
            "min_price_cents": float(min_price_cents),
            "max_price_cents": float(max_price_cents),
            "min_calibrated_expected_roi": float(min_calibrated_expected_roi),
            "max_calibrated_expected_roi": float(max_calibrated_expected_roi),
            "min_volume": float(min_volume),
            "max_volume": float(max_volume),
        },
        "note": "Defensive filters avoid team exclusions and block known decay-prone price, ROI, and liquidity slices.",
    }
    if summary["trade_start_date"] and summary["trade_end_date"]:
        start = summary["trade_start_date"]
        end = summary["trade_end_date"]
        summary["trade_timeline"] = start if start == end else f"{start} to {end}"
    else:
        summary["trade_timeline"] = "n/a"
    return frame.drop(columns=["_base_signal"]), audit, summary


def _selected(frame: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = [signal_column, "date", "price_cents", "calibrated_expected_roi", "volume", "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Defensive sweep rows are missing columns: {missing}")
    rows = frame.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["_base_signal"] = _coerce_bool(rows[signal_column])
    for column in ["price_cents", "calibrated_expected_roi", "volume", "clv_cents", "realized_profit_per_share"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows[
        rows["_base_signal"]
        & rows["date"].notna()
        & rows["price_cents"].notna()
        & rows["calibrated_expected_roi"].notna()
        & rows["volume"].notna()
        & rows["clv_cents"].notna()
        & rows["realized_profit_per_share"].notna()
    ].copy()
    return rows.reset_index(drop=True)


def _month_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=MONTHLY_SUMMARY_COLUMNS)
    rows = frame.copy()
    rows["month"] = rows["date"].dt.to_period("M").astype(str)
    return (
        rows.groupby("month", as_index=False)
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


def _status(metrics: dict[str, Any]) -> str:
    if (
        int(metrics["rows"]) >= 100
        and int(metrics["months"]) >= 4
        and float(metrics["positive_month_share"]) >= 0.75
        and float(metrics["avg_profit_per_share"]) > 0
        and float(metrics["avg_clv_cents"]) > 0
        and float(metrics["positive_clv_rate"]) >= 0.50
    ):
        return "defensive_candidate"
    if (
        int(metrics["rows"]) >= 50
        and int(metrics["months"]) >= 3
        and float(metrics["positive_month_share"]) >= 0.60
        and float(metrics["avg_profit_per_share"]) > 0
        and float(metrics["avg_clv_cents"]) > 0
    ):
        return "watchlist"
    return "not_ready"


def _score(metrics: dict[str, Any]) -> float:
    return float(
        (float(metrics["avg_profit_per_share"]) * 0.45)
        + (float(metrics["avg_clv_cents"]) / 100.0 * 0.20)
        + (float(metrics["positive_clv_rate"]) * 0.12)
        + (float(metrics["positive_month_share"]) * 0.10)
        - max(0.0, 0.50 - float(metrics["positive_clv_rate"])) * 0.15
    )


def _apply_rule(
    frame: pd.DataFrame,
    min_price_cents: float,
    max_price_cents: float,
    min_calibrated_expected_roi: float,
    max_calibrated_expected_roi: float,
    max_volume: float,
    min_volume: float = 10.0,
) -> pd.Series:
    return (
        frame["price_cents"].ge(float(min_price_cents))
        & frame["price_cents"].le(float(max_price_cents))
        & frame["calibrated_expected_roi"].ge(float(min_calibrated_expected_roi))
        & frame["calibrated_expected_roi"].lt(float(max_calibrated_expected_roi))
        & frame["volume"].ge(float(min_volume))
        & frame["volume"].lt(float(max_volume))
    )


def run_defensive_rule_sweep(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
    min_price_values: list[float] | None = None,
    max_price_values: list[float] | None = None,
    min_roi_values: list[float] | None = None,
    max_roi_values: list[float] | None = None,
    max_volume_values: list[float] | None = None,
    min_rows: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Sweep defensive price, ROI, and volume thresholds."""

    base = _selected(rows, signal_column)
    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "rules_tested": 0, "status": "not_ready"}
    rule_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for min_price in min_price_values or DEFAULT_MIN_PRICE_CENTS:
        for max_price in max_price_values or DEFAULT_MAX_PRICE_CENTS:
            if float(min_price) > float(max_price):
                continue
            for min_roi in min_roi_values or DEFAULT_MIN_ROIS:
                for max_roi in max_roi_values or DEFAULT_MAX_ROIS:
                    if float(min_roi) >= float(max_roi):
                        continue
                    for max_volume in max_volume_values or DEFAULT_MAX_VOLUMES:
                        signal = _apply_rule(base, min_price, max_price, min_roi, max_roi, max_volume)
                        selected = base[signal].copy()
                        if len(selected) < int(min_rows):
                            continue
                        monthly = _month_summary(selected)
                        positive_months = int((monthly["avg_profit_per_share"] > 0).sum()) if not monthly.empty else 0
                        months = int(len(monthly))
                        metrics = {
                            "min_price_cents": float(min_price),
                            "max_price_cents": float(max_price),
                            "min_calibrated_expected_roi": float(min_roi),
                            "max_calibrated_expected_roi": float(max_roi),
                            "max_volume": float(max_volume),
                            "rows": int(len(selected)),
                            "months": months,
                            "positive_months": positive_months,
                            "positive_month_share": float(positive_months / months) if months else 0.0,
                            "avg_clv_cents": float(selected["clv_cents"].mean()),
                            "positive_clv_rate": float((selected["clv_cents"] > 0).mean()),
                            "avg_profit_per_share": float(selected["realized_profit_per_share"].mean()),
                            "total_profit_per_share": float(selected["realized_profit_per_share"].sum()),
                        }
                        metrics["status"] = _status(metrics)
                        metrics["score"] = _score(metrics)
                        rule_rows.append(metrics)
                        monthly = monthly.copy()
                        monthly.insert(0, "max_volume", float(max_volume))
                        monthly.insert(0, "max_calibrated_expected_roi", float(max_roi))
                        monthly.insert(0, "min_calibrated_expected_roi", float(min_roi))
                        monthly.insert(0, "max_price_cents", float(max_price))
                        monthly.insert(0, "min_price_cents", float(min_price))
                        monthly_frames.append(monthly)

    rules = pd.DataFrame(rule_rows)
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    if rules.empty:
        return rules, monthly_all, {"rows": int(len(base)), "rules_tested": 0, "status": "not_ready"}
    status_rank = {"defensive_candidate": 0, "watchlist": 1, "not_ready": 2}
    rules["_status_rank"] = rules["status"].map(status_rank).fillna(99)
    rules = rules.sort_values(
        ["_status_rank", "score", "positive_clv_rate", "rows"],
        ascending=[True, False, False, False],
    ).drop(columns=["_status_rank"]).reset_index(drop=True)
    best = rules.iloc[0].to_dict()
    summary = {
        "rows": int(len(base)),
        "rules_tested": int(len(rules)),
        "defensive_candidates": int(rules["status"].eq("defensive_candidate").sum()),
        "watchlist_rules": int(rules["status"].eq("watchlist").sum()),
        "not_ready_rules": int(rules["status"].eq("not_ready").sum()),
        "best_status": str(best["status"]),
        "best_rule": _describe_rule(best),
        "best_rule_rows": int(best["rows"]),
        "best_rule_months": int(best["months"]),
        "best_rule_positive_month_share": float(best["positive_month_share"]),
        "best_rule_avg_clv_cents": float(best["avg_clv_cents"]),
        "best_rule_positive_clv_rate": float(best["positive_clv_rate"]),
        "best_rule_avg_profit_per_share": float(best["avg_profit_per_share"]),
        "note": "In-sample defensive sweep. Use walk-forward validation as the trust gate.",
    }
    return rules, monthly_all, summary


def run_walk_forward_defensive_validation(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
    min_price_values: list[float] | None = None,
    max_price_values: list[float] | None = None,
    min_roi_values: list[float] | None = None,
    max_roi_values: list[float] | None = None,
    max_volume_values: list[float] | None = None,
    min_rows: int = 25,
    min_train_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select defensive rules on prior months and apply them to future months."""

    base = _selected(rows, signal_column)
    if base.empty:
        empty_validated = base.copy()
        if "_base_signal" in empty_validated.columns:
            empty_validated = empty_validated.drop(columns=["_base_signal"])
        for column in WALK_FORWARD_SIGNAL_COLUMNS:
            if column not in empty_validated.columns:
                empty_validated[column] = pd.Series(dtype=object)
        return (
            empty_validated,
            pd.DataFrame(columns=WALK_FORWARD_FOLD_COLUMNS),
            pd.DataFrame(columns=MONTHLY_SUMMARY_COLUMNS),
            {"rows": 0, "status": "not_ready"},
        )
    base["_month"] = base["date"].dt.to_period("M")
    months = sorted(base["_month"].dropna().unique())
    validated_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for month in months:
        train = base[base["_month"] < month].copy()
        test = base[base["_month"] == month].copy()
        train_months = int(train["_month"].nunique()) if not train.empty else 0
        if test.empty:
            continue
        if train_months < int(min_train_months):
            test = test.copy()
            test["walk_forward_defensive_signal"] = False
            test["walk_forward_defensive_rule"] = "skipped_insufficient_prior_months"
            test["walk_forward_defensive_rule_status"] = "skipped"
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
        rules, _, train_summary = run_defensive_rule_sweep(
            train,
            signal_column=signal_column,
            min_price_values=min_price_values,
            max_price_values=max_price_values,
            min_roi_values=min_roi_values,
            max_roi_values=max_roi_values,
            max_volume_values=max_volume_values,
            min_rows=min_rows,
        )
        if rules.empty:
            test = test.copy()
            test["walk_forward_defensive_signal"] = False
            test["walk_forward_defensive_rule"] = "skipped_no_prior_rule"
            test["walk_forward_defensive_rule_status"] = "skipped"
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
        signal = _apply_rule(
            test,
            float(best["min_price_cents"]),
            float(best["max_price_cents"]),
            float(best["min_calibrated_expected_roi"]),
            float(best["max_calibrated_expected_roi"]),
            float(best["max_volume"]),
        )
        selected = test[signal].copy()
        test = test.copy()
        test["walk_forward_defensive_signal"] = signal
        test["walk_forward_defensive_rule"] = _describe_rule(best)
        test["walk_forward_defensive_rule_status"] = str(best["status"])
        validated_frames.append(test)
        fold_rows.append(
            {
                "test_month": str(month),
                "status": "evaluated",
                "train_rows": int(len(train)),
                "train_months": train_months,
                "test_rows": int(len(test)),
                "signals": int(signal.sum()),
                "selected_rule": _describe_rule(best),
                "selected_rule_status": str(best["status"]),
                "train_rules_tested": int(train_summary.get("rules_tested", 0) or 0),
                "train_best_positive_clv_rate": float(train_summary.get("best_rule_positive_clv_rate", 0.0) or 0.0),
                "train_best_positive_month_share": float(train_summary.get("best_rule_positive_month_share", 0.0) or 0.0),
                "test_avg_profit_per_share": float(selected["realized_profit_per_share"].mean()) if len(selected) else 0.0,
                "test_avg_clv_cents": float(selected["clv_cents"].mean()) if len(selected) else 0.0,
                "test_positive_clv_rate": float((selected["clv_cents"] > 0).mean()) if len(selected) else 0.0,
                "test_total_profit_per_share": float(selected["realized_profit_per_share"].sum()) if len(selected) else 0.0,
            }
        )

    validated = pd.concat(validated_frames, ignore_index=True, sort=False) if validated_frames else pd.DataFrame()
    if "_month" in validated.columns:
        validated = validated.drop(columns=["_month"])
    folds = pd.DataFrame(fold_rows, columns=WALK_FORWARD_FOLD_COLUMNS)
    selected_validated = (
        validated[validated["walk_forward_defensive_signal"].fillna(False)].copy()
        if not validated.empty and "walk_forward_defensive_signal" in validated.columns
        else pd.DataFrame()
    )
    monthly = _month_summary(selected_validated)
    positive_months = int((monthly["avg_profit_per_share"] > 0).sum()) if not monthly.empty else 0
    months_seen = int(len(monthly))
    positive_month_share = float(positive_months / months_seen) if months_seen else 0.0
    avg_profit = float(selected_validated["realized_profit_per_share"].mean()) if len(selected_validated) else 0.0
    avg_clv = float(selected_validated["clv_cents"].mean()) if len(selected_validated) else 0.0
    positive_clv_rate = float((selected_validated["clv_cents"] > 0).mean()) if len(selected_validated) else 0.0
    status = "walk_forward_candidate" if (
        len(selected_validated) >= 100
        and months_seen >= 3
        and positive_month_share >= 0.67
        and avg_profit > 0
        and avg_clv > 0
        and positive_clv_rate >= 0.50
    ) else "not_ready"
    summary = {
        "rows": int(len(base)),
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
        "note": "Nested walk-forward defensive validation: each test month uses thresholds selected only from prior months.",
    }
    return validated.reset_index(drop=True), folds.reset_index(drop=True), monthly.reset_index(drop=True), summary


def run_defensive_sample_expansion(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
    min_price_values: list[float] | None = None,
    max_price_values: list[float] | None = None,
    min_roi_values: list[float] | None = None,
    max_roi_values: list[float] | None = None,
    max_volume_values: list[float] | None = None,
    min_train_months: int = 2,
    target_min_signals: int = 100,
    target_max_signals: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate nearby broader defensive rules on walk-forward-eligible months.

    This is a sensitivity report, not the primary nested selector. It asks whether
    broader non-team rules can recover sample size while preserving CLV evidence.
    """

    base = _selected(rows, signal_column)
    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "status": "not_ready"}
    base["_month"] = base["date"].dt.to_period("M")
    months = sorted(base["_month"].dropna().unique())
    eligible_months = []
    for month in months:
        train_months = int(base[base["_month"] < month]["_month"].nunique())
        if train_months >= int(min_train_months):
            eligible_months.append(month)
    eligible = base[base["_month"].isin(eligible_months)].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": int(len(base)), "status": "not_ready"}

    candidate_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for min_price in min_price_values or [10.0, 12.5, 15.0]:
        for max_price in max_price_values or [40.0, 45.0, 50.0, 55.0]:
            if float(min_price) > float(max_price):
                continue
            for min_roi in min_roi_values or [0.0, 0.25, 0.5]:
                for max_roi in max_roi_values or [3.0, 5.0]:
                    if float(min_roi) >= float(max_roi):
                        continue
                    for max_volume in max_volume_values or [1000.0, 10000.0, np.inf]:
                        signal = _apply_rule(eligible, min_price, max_price, min_roi, max_roi, max_volume)
                        selected = eligible[signal].copy()
                        if selected.empty:
                            continue
                        monthly = _month_summary(selected)
                        positive_months = int((monthly["avg_profit_per_share"] > 0).sum()) if not monthly.empty else 0
                        months_seen = int(len(monthly))
                        positive_month_share = float(positive_months / months_seen) if months_seen else 0.0
                        min_month_positive_clv_rate = (
                            float(monthly["positive_clv_rate"].min()) if not monthly.empty else 0.0
                        )
                        min_month_avg_clv_cents = (
                            float(monthly["avg_clv_cents"].min()) if not monthly.empty else 0.0
                        )
                        signals = int(len(selected))
                        avg_profit = float(selected["realized_profit_per_share"].mean())
                        avg_clv = float(selected["clv_cents"].mean())
                        positive_clv_rate = float((selected["clv_cents"] > 0).mean())
                        in_target_range = int(target_min_signals) <= signals <= int(target_max_signals)
                        reaches_min_sample = signals >= int(target_min_signals)
                        clv_safe = (
                            months_seen >= 3
                            and positive_month_share >= 0.67
                            and min_month_positive_clv_rate >= 0.50
                            and min_month_avg_clv_cents > 0
                            and avg_profit > 0
                            and avg_clv > 0
                            and positive_clv_rate >= 0.50
                        )
                        if clv_safe and reaches_min_sample:
                            status = "expanded_sample_candidate"
                        elif clv_safe:
                            status = "undersized_clv_candidate"
                        else:
                            status = "not_ready"
                        distance_from_target = (
                            0
                            if in_target_range
                            else min(abs(signals - int(target_min_signals)), abs(signals - int(target_max_signals)))
                        )
                        metrics = {
                            "min_price_cents": float(min_price),
                            "max_price_cents": float(max_price),
                            "min_calibrated_expected_roi": float(min_roi),
                            "max_calibrated_expected_roi": float(max_roi),
                            "max_volume": float(max_volume),
                            "signals": signals,
                            "months": months_seen,
                            "positive_months": positive_months,
                            "positive_month_share": positive_month_share,
                            "min_month_positive_clv_rate": min_month_positive_clv_rate,
                            "min_month_avg_clv_cents": min_month_avg_clv_cents,
                            "avg_clv_cents": avg_clv,
                            "positive_clv_rate": positive_clv_rate,
                            "avg_profit_per_share": avg_profit,
                            "total_profit_per_share": float(selected["realized_profit_per_share"].sum()),
                            "status": status,
                            "in_target_sample_range": bool(in_target_range),
                            "sample_distance_from_target": int(distance_from_target),
                        }
                        metrics["score"] = _score(
                            {
                                "rows": signals,
                                "months": months_seen,
                                "positive_month_share": positive_month_share,
                                "avg_profit_per_share": avg_profit,
                                "avg_clv_cents": avg_clv,
                                "positive_clv_rate": positive_clv_rate,
                            }
                        ) - (float(distance_from_target) / 1000.0)
                        candidate_rows.append(metrics)
                        monthly = monthly.copy()
                        monthly.insert(0, "max_volume", float(max_volume))
                        monthly.insert(0, "max_calibrated_expected_roi", float(max_roi))
                        monthly.insert(0, "min_calibrated_expected_roi", float(min_roi))
                        monthly.insert(0, "max_price_cents", float(max_price))
                        monthly.insert(0, "min_price_cents", float(min_price))
                        monthly_frames.append(monthly)

    candidates = pd.DataFrame(candidate_rows)
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    if candidates.empty:
        return candidates, monthly_all, {"rows": int(len(base)), "status": "not_ready"}
    status_rank = {"expanded_sample_candidate": 0, "undersized_clv_candidate": 1, "not_ready": 2}
    candidates["_status_rank"] = candidates["status"].map(status_rank).fillna(99)
    candidates = candidates.sort_values(
        ["_status_rank", "in_target_sample_range", "score", "positive_clv_rate", "signals"],
        ascending=[True, False, False, False, False],
    ).drop(columns=["_status_rank"]).reset_index(drop=True)
    best = candidates.iloc[0].to_dict()
    summary = {
        "rows": int(len(base)),
        "eligible_rows": int(len(eligible)),
        "eligible_months": int(len(eligible_months)),
        "rules_tested": int(len(candidates)),
        "expanded_sample_candidates": int(candidates["status"].eq("expanded_sample_candidate").sum()),
        "undersized_clv_candidates": int(candidates["status"].eq("undersized_clv_candidate").sum()),
        "target_min_signals": int(target_min_signals),
        "target_max_signals": int(target_max_signals),
        "best_status": str(best["status"]),
        "best_rule": _describe_rule(best),
        "best_rule_signals": int(best["signals"]),
        "best_rule_positive_month_share": float(best["positive_month_share"]),
        "best_rule_min_month_positive_clv_rate": float(best["min_month_positive_clv_rate"]),
        "best_rule_min_month_avg_clv_cents": float(best["min_month_avg_clv_cents"]),
        "best_rule_avg_clv_cents": float(best["avg_clv_cents"]),
        "best_rule_positive_clv_rate": float(best["positive_clv_rate"]),
        "best_rule_avg_profit_per_share": float(best["avg_profit_per_share"]),
        "status": "sample_expansion_candidate"
        if str(best["status"]) == "expanded_sample_candidate"
        else "not_ready",
        "parlay_ready": False,
        "note": "Sample expansion report: broader fixed defensive rules over walk-forward-eligible months. Use this to choose hypotheses for the nested validator, not to green-light parlays.",
    }
    return candidates, monthly_all, summary


def _describe_rule(row: dict[str, Any]) -> str:
    max_volume = float(row.get("max_volume", np.inf))
    max_volume_text = "inf" if np.isinf(max_volume) else f"{max_volume:.0f}"
    return (
        f"price={float(row.get('min_price_cents', 0.0)):.0f}-{float(row.get('max_price_cents', 100.0)):.0f}c, "
        f"roi={float(row.get('min_calibrated_expected_roi', 0.0)):.1f}-{float(row.get('max_calibrated_expected_roi', 0.0)):.1f}, "
        f"volume<={max_volume_text}"
    )


def save_defensive_filter_outputs(
    filtered: pd.DataFrame,
    audit: pd.DataFrame,
    summary: dict[str, Any],
    filtered_path: str | Path,
    audit_path: str | Path,
    summary_path: str | Path,
) -> None:
    filtered_output = Path(filtered_path)
    audit_output = Path(audit_path)
    summary_output = Path(summary_path)
    filtered_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_output, index=False)
    audit.to_csv(audit_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_defensive_rule_sweep_outputs(
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


def save_walk_forward_defensive_outputs(
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


def save_defensive_sample_expansion_outputs(
    candidates: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: dict[str, Any],
    candidates_path: str | Path,
    monthly_path: str | Path,
    summary_path: str | Path,
) -> None:
    candidates_output = Path(candidates_path)
    monthly_output = Path(monthly_path)
    summary_output = Path(summary_path)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_output, index=False)
    monthly.to_csv(monthly_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
