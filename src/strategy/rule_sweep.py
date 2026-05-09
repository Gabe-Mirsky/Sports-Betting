"""Explore stricter calibrated signal rules before expanding paper trading."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool
from strategy.stability import summarize_signal_stability


DEFAULT_MIN_EDGES = [-0.10, -0.05, 0.0, 0.02, 0.05]
DEFAULT_MIN_EXPECTED_ROIS = [0.0, 0.10, 0.25, 0.50, 0.75]
DEFAULT_MIN_HISTORY_ROWS = [0, 50, 100, 150, 200]
DEFAULT_MIN_PRICE_CENTS = [1, 5, 10]
DEFAULT_MAX_PRICE_CENTS = [90, 95, 99]


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _status_from_summary(summary: dict[str, Any]) -> str:
    signals = int(summary.get("signals", 0) or 0)
    months = int(summary.get("months", 0) or 0)
    positive_month_share = float(summary.get("positive_month_share", 0.0) or 0.0)
    avg_profit = float(summary.get("overall_avg_profit_per_share", 0.0) or 0.0)
    worst_month_profit = float(summary.get("worst_month_avg_profit_per_share", 0.0) or 0.0)

    if (
        signals >= 100
        and months >= 5
        and positive_month_share >= 0.67
        and avg_profit >= 0.02
        and worst_month_profit >= -0.12
    ):
        return "exploratory_candidate"
    if signals >= 50 and months >= 4 and positive_month_share >= 0.60 and avg_profit > 0.0:
        return "watchlist"
    return "not_ready"


def _score_rule(summary: dict[str, Any], avg_expected_roi: float, avg_edge: float) -> float:
    signals = int(summary.get("signals", 0) or 0)
    months = int(summary.get("months", 0) or 0)
    positive_month_share = float(summary.get("positive_month_share", 0.0) or 0.0)
    avg_profit = float(summary.get("overall_avg_profit_per_share", 0.0) or 0.0)
    worst_month_profit = float(summary.get("worst_month_avg_profit_per_share", 0.0) or 0.0)
    signal_penalty = max(0.0, (100 - signals) / 100.0) * 0.05
    month_penalty = max(0.0, (5 - months) / 5.0) * 0.04
    worst_penalty = abs(min(0.0, worst_month_profit)) * 0.20
    roi_bonus = min(max(avg_expected_roi, -1.0), 2.0) * 0.01
    edge_bonus = min(max(avg_edge, -0.25), 0.25) * 0.03
    return float(avg_profit + (positive_month_share * 0.04) + roi_bonus + edge_bonus - signal_penalty - month_penalty - worst_penalty)


def build_rule_signal(
    rows: pd.DataFrame,
    signal_column: str,
    expected_roi_column: str,
    min_edge: float,
    min_expected_roi: float,
    min_history_rows: int,
    min_price_cents: float,
    max_price_cents: float,
    history_column: str = "edge_bin_history_rows",
    secondary_history_column: str | None = "edge_bin_history_rows_blend",
) -> pd.Series:
    """Return rows that pass the base signal and stricter research filters."""

    required = [signal_column, expected_roi_column, "edge", "price_cents"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Signal-rule sweep rows are missing columns: {missing}")

    base_signal = _coerce_bool(rows[signal_column])
    edge = _numeric(rows, "edge")
    expected_roi = _numeric(rows, expected_roi_column)
    price_cents = _numeric(rows, "price_cents")
    history = _numeric(rows, history_column, default=0.0).fillna(0.0)

    rule = (
        base_signal
        & edge.ge(min_edge)
        & expected_roi.ge(min_expected_roi)
        & history.ge(float(min_history_rows))
        & price_cents.ge(min_price_cents)
        & price_cents.le(max_price_cents)
    )
    if secondary_history_column and secondary_history_column in rows.columns:
        secondary_history = _numeric(rows, secondary_history_column, default=0.0).fillna(0.0)
        rule = rule & secondary_history.ge(float(min_history_rows))
    return rule.fillna(False)


def run_signal_rule_sweep(
    rows: pd.DataFrame,
    signal_column: str = "consensus_trade",
    expected_roi_column: str = "consensus_expected_roi",
    min_edges: list[float] | None = None,
    min_expected_rois: list[float] | None = None,
    min_history_rows: list[int] | None = None,
    min_price_cents: list[float] | None = None,
    max_price_cents: list[float] | None = None,
    history_column: str = "edge_bin_history_rows",
    secondary_history_column: str | None = "edge_bin_history_rows_blend",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Try stricter filters and summarize each rule's historical stability."""

    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "rules_tested": 0}

    frame = rows.copy()
    edge = _numeric(frame, "edge")
    expected_roi = _numeric(frame, expected_roi_column)
    frame["_rule_edge"] = edge
    frame["_rule_expected_roi"] = expected_roi

    rule_rows: list[dict[str, Any]] = []
    for min_edge, min_roi, min_history, min_price, max_price in product(
        min_edges or DEFAULT_MIN_EDGES,
        min_expected_rois or DEFAULT_MIN_EXPECTED_ROIS,
        min_history_rows or DEFAULT_MIN_HISTORY_ROWS,
        min_price_cents or DEFAULT_MIN_PRICE_CENTS,
        max_price_cents or DEFAULT_MAX_PRICE_CENTS,
    ):
        if min_price > max_price:
            continue
        signal = build_rule_signal(
            frame,
            signal_column=signal_column,
            expected_roi_column=expected_roi_column,
            min_edge=float(min_edge),
            min_expected_roi=float(min_roi),
            min_history_rows=int(min_history),
            min_price_cents=float(min_price),
            max_price_cents=float(max_price),
            history_column=history_column,
            secondary_history_column=secondary_history_column,
        )
        temp = frame.copy()
        temp["_sweep_signal"] = signal
        monthly, summary = summarize_signal_stability(
            temp,
            signal_column="_sweep_signal",
            expected_roi_column=expected_roi_column,
        )
        signals = int(summary.get("signals", 0) or 0)
        selected = temp[signal].copy()
        avg_edge = float(selected["_rule_edge"].mean()) if signals else 0.0
        avg_expected_roi = float(selected["_rule_expected_roi"].mean()) if signals else 0.0
        status = _status_from_summary(summary)
        score = _score_rule(summary, avg_expected_roi, avg_edge)
        row = {
            "status": status,
            "score": score,
            "min_edge": float(min_edge),
            "min_expected_roi": float(min_roi),
            "min_edge_bin_history_rows": int(min_history),
            "min_price_cents": float(min_price),
            "max_price_cents": float(max_price),
            "signals": signals,
            "timeline": summary.get("timeline", "n/a"),
            "months": int(summary.get("months", 0) or 0),
            "positive_months": int(summary.get("positive_months", 0) or 0),
            "positive_month_share": float(summary.get("positive_month_share", 0.0) or 0.0),
            "overall_win_rate": float(summary.get("overall_win_rate", 0.0) or 0.0),
            "overall_avg_profit_per_share": float(summary.get("overall_avg_profit_per_share", 0.0) or 0.0),
            "worst_month": summary.get("worst_month"),
            "worst_month_avg_profit_per_share": float(summary.get("worst_month_avg_profit_per_share", 0.0) or 0.0),
            "best_month": summary.get("best_month"),
            "best_month_avg_profit_per_share": float(summary.get("best_month_avg_profit_per_share", 0.0) or 0.0),
            "avg_edge": avg_edge,
            "avg_expected_roi": avg_expected_roi,
            "parlay_ready": False,
            "note": "Exploratory in-sample rule; use as a paper-watch hypothesis only.",
        }
        rule_rows.append(row)

    rules = pd.DataFrame(rule_rows)
    if rules.empty:
        return rules, pd.DataFrame(), {"rows": int(len(rows)), "rules_tested": 0}

    status_rank = {"exploratory_candidate": 0, "watchlist": 1, "not_ready": 2}
    rules["_status_rank"] = rules["status"].map(status_rank).fillna(99)
    rules = (
        rules.sort_values(
            ["_status_rank", "score", "signals", "positive_month_share"],
            ascending=[True, False, False, False],
        )
        .drop(columns=["_status_rank"])
        .reset_index(drop=True)
    )
    best = rules.iloc[0].to_dict()
    best_signal = build_rule_signal(
        frame,
        signal_column=signal_column,
        expected_roi_column=expected_roi_column,
        min_edge=float(best["min_edge"]),
        min_expected_roi=float(best["min_expected_roi"]),
        min_history_rows=int(best["min_edge_bin_history_rows"]),
        min_price_cents=float(best["min_price_cents"]),
        max_price_cents=float(best["max_price_cents"]),
        history_column=history_column,
        secondary_history_column=secondary_history_column,
    )
    best_frame = frame.copy()
    best_frame["_sweep_signal"] = best_signal
    best_monthly, _ = summarize_signal_stability(
        best_frame,
        signal_column="_sweep_signal",
        expected_roi_column=expected_roi_column,
    )
    if not best_monthly.empty:
        best_monthly.insert(0, "rule_score", float(best["score"]))
        best_monthly.insert(0, "rule_status", str(best["status"]))
        best_monthly.insert(0, "rule", _describe_rule(best))
    summary = {
        "rows": int(len(rows)),
        "rules_tested": int(len(rules)),
        "exploratory_candidates": int(rules["status"].eq("exploratory_candidate").sum()),
        "watchlist_rules": int(rules["status"].eq("watchlist").sum()),
        "not_ready_rules": int(rules["status"].eq("not_ready").sum()),
        "best_rule": _describe_rule(best),
        "best_rule_params": {
            "min_edge": float(best.get("min_edge", 0.0) or 0.0),
            "min_expected_roi": float(best.get("min_expected_roi", 0.0) or 0.0),
            "min_edge_bin_history_rows": int(best.get("min_edge_bin_history_rows", 0) or 0),
            "min_price_cents": float(best.get("min_price_cents", 0.0) or 0.0),
            "max_price_cents": float(best.get("max_price_cents", 0.0) or 0.0),
        },
        "best_rule_status": str(best.get("status", "n/a")),
        "best_rule_signals": int(best.get("signals", 0) or 0),
        "best_rule_timeline": str(best.get("timeline", "n/a")),
        "best_rule_positive_months": int(best.get("positive_months", 0) or 0),
        "best_rule_months": int(best.get("months", 0) or 0),
        "best_rule_avg_profit_per_share": float(best.get("overall_avg_profit_per_share", 0.0) or 0.0),
        "best_rule_worst_month": best.get("worst_month"),
        "best_rule_worst_month_avg_profit_per_share": float(
            best.get("worst_month_avg_profit_per_share", 0.0) or 0.0
        ),
        "signal_column": signal_column,
        "expected_roi_column": expected_roi_column,
        "note": "This is an in-sample research sweep. It can suggest forward paper-watch filters but does not make any rule parlay-ready.",
    }
    return rules, best_monthly, summary


def run_walk_forward_signal_rule_validation(
    rows: pd.DataFrame,
    signal_column: str = "consensus_trade",
    expected_roi_column: str = "consensus_expected_roi",
    min_edges: list[float] | None = None,
    min_expected_rois: list[float] | None = None,
    min_history_rows: list[int] | None = None,
    min_price_cents: list[float] | None = None,
    max_price_cents: list[float] | None = None,
    history_column: str = "edge_bin_history_rows",
    secondary_history_column: str | None = "edge_bin_history_rows_blend",
    min_train_rows: int = 50,
    min_train_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate signal-rule selection with expanding monthly walk-forward folds."""

    if rows.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "rows": 0,
                "folds": 0,
                "evaluated_months": 0,
                "skipped_months": 0,
                "status": "not_ready",
            },
        )

    frame = rows.copy()
    if "date" not in frame.columns:
        raise ValueError("Walk-forward rule validation requires a date column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["date"]).reset_index(drop=True)
    frame["_rule_month"] = frame["date"].dt.to_period("M")
    months = sorted(frame["_rule_month"].dropna().unique())

    validated_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    skipped_months = 0

    for month in months:
        train = frame[frame["_rule_month"] < month].copy()
        test = frame[frame["_rule_month"] == month].copy()
        if test.empty:
            continue
        train_months = int(train["_rule_month"].nunique()) if not train.empty else 0
        if len(train) < int(min_train_rows) or train_months < int(min_train_months):
            skipped_months += 1
            test = test.copy()
            test["walk_forward_rule_signal"] = False
            test["walk_forward_rule"] = "skipped_insufficient_prior_history"
            test["walk_forward_rule_status"] = "skipped"
            test["walk_forward_rule_score"] = np.nan
            validated_frames.append(test)
            fold_rows.append(
                {
                    "test_month": str(month),
                    "status": "skipped_insufficient_prior_history",
                    "train_rows": int(len(train)),
                    "train_months": train_months,
                    "test_rows": int(len(test)),
                    "signals": 0,
                }
            )
            continue

        rules, _, train_summary = run_signal_rule_sweep(
            train,
            signal_column=signal_column,
            expected_roi_column=expected_roi_column,
            min_edges=min_edges,
            min_expected_rois=min_expected_rois,
            min_history_rows=min_history_rows,
            min_price_cents=min_price_cents,
            max_price_cents=max_price_cents,
            history_column=history_column,
            secondary_history_column=secondary_history_column,
        )
        if rules.empty:
            skipped_months += 1
            test = test.copy()
            test["walk_forward_rule_signal"] = False
            test["walk_forward_rule"] = "skipped_no_rule_selected"
            test["walk_forward_rule_status"] = "skipped"
            test["walk_forward_rule_score"] = np.nan
            validated_frames.append(test)
            fold_rows.append(
                {
                    "test_month": str(month),
                    "status": "skipped_no_rule_selected",
                    "train_rows": int(len(train)),
                    "train_months": train_months,
                    "test_rows": int(len(test)),
                    "signals": 0,
                }
            )
            continue

        best = rules.iloc[0].to_dict()
        signal = build_rule_signal(
            test,
            signal_column=signal_column,
            expected_roi_column=expected_roi_column,
            min_edge=float(best["min_edge"]),
            min_expected_roi=float(best["min_expected_roi"]),
            min_history_rows=int(best["min_edge_bin_history_rows"]),
            min_price_cents=float(best["min_price_cents"]),
            max_price_cents=float(best["max_price_cents"]),
            history_column=history_column,
            secondary_history_column=secondary_history_column,
        )
        test = test.copy()
        test["walk_forward_rule_signal"] = signal
        test["walk_forward_rule"] = _describe_rule(best)
        test["walk_forward_rule_status"] = str(best.get("status", "n/a"))
        test["walk_forward_rule_score"] = float(best.get("score", 0.0) or 0.0)
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
                "selected_rule_status": str(best.get("status", "n/a")),
                "selected_rule_score": float(best.get("score", 0.0) or 0.0),
                "train_rules_tested": int(train_summary.get("rules_tested", 0) or 0),
                "train_best_rule_signals": int(best.get("signals", 0) or 0),
                "train_best_rule_timeline": str(best.get("timeline", "n/a")),
            }
        )

    validated = pd.concat(validated_frames, ignore_index=True, sort=False) if validated_frames else pd.DataFrame()
    if "_rule_month" in validated.columns:
        validated = validated.drop(columns=["_rule_month"])
    folds = pd.DataFrame(fold_rows)
    monthly, stability_summary = summarize_signal_stability(
        validated,
        signal_column="walk_forward_rule_signal",
        expected_roi_column=expected_roi_column,
    ) if not validated.empty else (pd.DataFrame(), {"rows": 0})

    raw_status = _status_from_summary(stability_summary)
    validation_status = "walk_forward_candidate" if raw_status == "exploratory_candidate" else raw_status
    summary = {
        **stability_summary,
        "folds": int(len(folds)),
        "evaluated_months": int(folds["status"].eq("evaluated").sum()) if not folds.empty else 0,
        "skipped_months": int(skipped_months),
        "min_train_rows": int(min_train_rows),
        "min_train_months": int(min_train_months),
        "status": validation_status,
        "parlay_ready": False,
        "signal_column": signal_column,
        "expected_roi_column": expected_roi_column,
        "note": (
            "Nested walk-forward rule validation: each test month uses a rule chosen only from prior months. "
            "This replaces in-sample sweep results as the trust gate for forward use."
        ),
    }
    return validated.reset_index(drop=True), folds.reset_index(drop=True), monthly, summary


def _describe_rule(row: dict[str, Any]) -> str:
    return (
        f"edge>={float(row.get('min_edge', 0.0)):.2f}, "
        f"roi>={float(row.get('min_expected_roi', 0.0)):.2f}, "
        f"history>={int(row.get('min_edge_bin_history_rows', 0) or 0)}, "
        f"price={float(row.get('min_price_cents', 0.0)):.0f}-{float(row.get('max_price_cents', 0.0)):.0f}c"
    )


def save_signal_rule_sweep_outputs(
    rules: pd.DataFrame,
    best_monthly: pd.DataFrame,
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
    best_monthly.to_csv(monthly_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_walk_forward_rule_validation_outputs(
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
