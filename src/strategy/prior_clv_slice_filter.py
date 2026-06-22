"""Walk-forward CLV slice filters selected from prior months only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


POLICIES: dict[str, list[str]] = {
    "side": ["side"],
    "side_price": ["side", "price_bucket"],
    "side_edge": ["side", "edge_bucket"],
    "side_price_edge": ["side", "price_bucket", "edge_bucket"],
    "side_liquidity": ["side", "liquidity_bucket"],
}


def _side(frame: pd.DataFrame) -> pd.Series:
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
    required = ["date", signal_column, "price_cents", "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Prior CLV slice rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["side"] = _side(frame)
    for column in ["price_cents", "clv_cents", "realized_profit_per_share", "edge", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["_signal"]
        & frame["date"].notna()
        & frame["price_cents"].notna()
        & frame["clv_cents"].notna()
        & frame["realized_profit_per_share"].notna()
    ].copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    frame["_month_period"] = frame["date"].dt.to_period("M")
    frame["positive_clv"] = frame["clv_cents"] > 0
    frame["profitable"] = frame["realized_profit_per_share"] > 0
    frame["price_bucket"] = pd.cut(
        frame["price_cents"],
        bins=[0, 10, 20, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    frame["edge_bucket"] = pd.cut(
        frame.get("edge", pd.Series(np.nan, index=frame.index)),
        bins=[-np.inf, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<=2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    frame["liquidity_bucket"] = pd.cut(
        frame.get("volume", pd.Series(np.nan, index=frame.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
    return frame.reset_index(drop=True)


def _group_stats(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_columns, dropna=False, observed=False)
        .agg(
            train_rows=("clv_cents", "size"),
            train_months=("month", "nunique"),
            train_avg_clv_cents=("clv_cents", "mean"),
            train_positive_clv_rate=("positive_clv", "mean"),
            train_avg_profit_per_share=("realized_profit_per_share", "mean"),
        )
        .reset_index()
    )
    grouped["train_score"] = (
        grouped["train_avg_profit_per_share"] * 0.50
        + grouped["train_avg_clv_cents"] / 100.0 * 0.25
        + grouped["train_positive_clv_rate"] * 0.10
    )
    return grouped.sort_values(["train_score", "train_rows"], ascending=[False, False]).reset_index(drop=True)


def _matching_keys(frame: pd.DataFrame, stats: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    if frame.empty or stats.empty:
        return pd.Series(False, index=frame.index)
    keys = set(tuple(str(value) for value in row) for row in stats[group_columns].itertuples(index=False, name=None))
    return frame[group_columns].astype(str).apply(lambda row: tuple(row.values) in keys, axis=1)


def _summarize_selected(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "signals": 0,
            "months": 0,
            "avg_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "avg_profit_per_share": 0.0,
            "profit_rate": 0.0,
            "positive_month_share": 0.0,
        }
    monthly_profit = frame.groupby("month", observed=False)["realized_profit_per_share"].mean()
    return {
        "signals": int(len(frame)),
        "months": int(frame["month"].nunique()),
        "avg_clv_cents": float(frame["clv_cents"].mean()),
        "positive_clv_rate": float(frame["positive_clv"].mean()),
        "avg_profit_per_share": float(frame["realized_profit_per_share"].mean()),
        "profit_rate": float(frame["profitable"].mean()),
        "positive_month_share": float((monthly_profit > 0).mean()) if len(monthly_profit) else 0.0,
    }


def run_prior_clv_slice_filter(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_train_rows_values: list[int] | None = None,
    min_positive_clv_values: list[float] | None = None,
    min_train_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run expanding-month CLV slice filters across simple pregame buckets."""

    prepared = _prepare(rows, signal_column=signal_column)
    if prepared.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "no_signals", "signals": 0}

    min_rows_options = min_train_rows_values or [15, 25, 40]
    min_positive_options = min_positive_clv_values or [0.30, 0.40, 0.50]
    months = sorted(prepared["_month_period"].dropna().unique())
    all_policy_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for policy_name, group_columns in POLICIES.items():
        for min_rows in min_rows_options:
            for min_positive_clv in min_positive_options:
                policy_id = f"{policy_name}|rows>={min_rows}|pos_clv>={min_positive_clv:.2f}"
                selected_frames: list[pd.DataFrame] = []
                for month in months:
                    train = prepared[prepared["_month_period"] < month].copy()
                    test = prepared[prepared["_month_period"] == month].copy()
                    train_months = int(train["_month_period"].nunique()) if not train.empty else 0
                    if test.empty or train_months < int(min_train_months):
                        fold_rows.append(
                            {
                                "policy": policy_id,
                                "test_month": str(month),
                                "status": "skipped_insufficient_prior_months",
                                "train_rows": int(len(train)),
                                "train_months": train_months,
                                "test_rows": int(len(test)),
                                "signals": 0,
                            }
                        )
                        continue
                    stats = _group_stats(train, group_columns)
                    accepted = stats[
                        stats["train_rows"].ge(int(min_rows))
                        & stats["train_months"].ge(int(min_train_months))
                        & stats["train_avg_clv_cents"].gt(0)
                        & stats["train_positive_clv_rate"].ge(float(min_positive_clv))
                        & stats["train_avg_profit_per_share"].gt(0)
                    ].copy()
                    mask = _matching_keys(test, accepted, group_columns)
                    selected = test[mask].copy()
                    if not selected.empty:
                        selected["prior_clv_policy"] = policy_id
                        selected["prior_clv_group_columns"] = "|".join(group_columns)
                        selected_frames.append(selected)
                    fold_summary = _summarize_selected(selected)
                    fold_rows.append(
                        {
                            "policy": policy_id,
                            "test_month": str(month),
                            "status": "evaluated",
                            "train_rows": int(len(train)),
                            "train_months": train_months,
                            "test_rows": int(len(test)),
                            "accepted_groups": int(len(accepted)),
                            **fold_summary,
                        }
                    )
                policy_rows = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
                if not policy_rows.empty:
                    all_policy_rows.append(policy_rows)

    selected_all = pd.concat(all_policy_rows, ignore_index=True, sort=False) if all_policy_rows else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    summaries: list[dict[str, Any]] = []
    if not folds.empty:
        evaluated = folds[folds["status"].eq("evaluated")].copy()
        for policy, group in evaluated.groupby("policy", dropna=False, observed=False):
            selected = selected_all[selected_all["prior_clv_policy"].eq(policy)] if not selected_all.empty else pd.DataFrame()
            row = {"policy": str(policy), "evaluated_months": int(group["test_month"].nunique())}
            row.update(_summarize_selected(selected))
            row["score"] = float(
                row["avg_profit_per_share"] * 0.50
                + row["avg_clv_cents"] / 100.0 * 0.25
                + row["positive_clv_rate"] * 0.10
                + row["positive_month_share"] * 0.05
            )
            row["status"] = (
                "candidate"
                if row["signals"] >= 100
                and row["months"] >= 3
                and row["avg_profit_per_share"] > 0
                and row["avg_clv_cents"] > 0
                and row["positive_clv_rate"] >= 0.50
                and row["positive_month_share"] >= 0.67
                else "not_ready"
            )
            summaries.append(row)
    summary_table = pd.DataFrame(summaries)
    if not summary_table.empty:
        summary_table = summary_table.sort_values(["status", "score", "signals"], ascending=[True, False, False])
    best = summary_table.iloc[0].to_dict() if not summary_table.empty else {}
    summary = {
        "status": str(best.get("status", "not_ready")),
        "policies_tested": int(len(summary_table)),
        "candidate_policies": int(summary_table["status"].eq("candidate").sum()) if not summary_table.empty else 0,
        "best_policy": str(best.get("policy", "n/a")),
        "best_signals": int(best.get("signals", 0) or 0),
        "best_avg_clv_cents": float(best.get("avg_clv_cents", 0.0) or 0.0),
        "best_positive_clv_rate": float(best.get("positive_clv_rate", 0.0) or 0.0),
        "best_avg_profit_per_share": float(best.get("avg_profit_per_share", 0.0) or 0.0),
        "best_positive_month_share": float(best.get("positive_month_share", 0.0) or 0.0),
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "note": "Research-only prior-slice filter. Promote nothing unless full proof gates improve out of sample.",
    }
    return selected_all.reset_index(drop=True), summary_table.reset_index(drop=True), folds.reset_index(drop=True), summary


def save_prior_clv_slice_filter(
    selected: pd.DataFrame,
    summary_table: pd.DataFrame,
    folds: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "prior_clv_slice_filter",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_root / f"{prefix}_signals.csv", index=False)
    summary_table.to_csv(output_root / f"{prefix}_policies.csv", index=False)
    folds.to_csv(output_root / f"{prefix}_folds.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
