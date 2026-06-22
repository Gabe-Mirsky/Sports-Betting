"""Diagnostics for month-to-month CLV decay in selected signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


MONTHLY_COLUMNS = [
    "month",
    "rows",
    "avg_clv_cents",
    "median_clv_cents",
    "positive_clv_rate",
    "avg_profit_per_share",
    "total_profit_per_share",
]
SEGMENT_DELTA_COLUMNS = [
    "segment_type",
    "segment",
    "first_month",
    "last_month",
    "first_rows",
    "last_rows",
    "positive_clv_rate_change",
    "avg_clv_cents_change",
    "avg_profit_per_share_change",
    "last_positive_clv_rate",
    "last_avg_clv_cents",
    "last_avg_profit_per_share",
]
NEGATIVE_CLV_COLUMNS = [
    "date",
    "game_id",
    "market_ticker",
    "home_team_abbr",
    "away_team_abbr",
    "yes_team_abbr",
    "price_cents",
    "edge",
    "calibrated_expected_roi",
    "volume",
    "open_interest",
    "clv_cents",
    "realized_profit_per_share",
    "month",
    "price_bucket",
    "liquidity_bucket",
]


def _selected(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    required = ["date", signal_column, "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"CLV decay rows are missing columns: {missing}")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["clv_cents"] = pd.to_numeric(frame["clv_cents"], errors="coerce")
    frame["realized_profit_per_share"] = pd.to_numeric(frame["realized_profit_per_share"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame = frame[frame["_signal"] & frame["date"].notna() & frame["clv_cents"].notna()].copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    for column in ["price_cents", "edge", "calibrated_expected_roi", "volume", "open_interest"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def _summary(frame: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        columns = list(group_columns or []) + [
            "rows",
            "avg_clv_cents",
            "median_clv_cents",
            "positive_clv_rate",
            "avg_profit_per_share",
            "total_profit_per_share",
        ]
        return pd.DataFrame(columns=columns)
    group_columns = group_columns or []
    grouped = frame.groupby(group_columns, dropna=False, observed=False) if group_columns else [(None, frame)]
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        row = {
            "rows": int(len(group)),
            "avg_clv_cents": float(group["clv_cents"].mean()),
            "median_clv_cents": float(group["clv_cents"].median()),
            "positive_clv_rate": float((group["clv_cents"] > 0).mean()),
            "avg_profit_per_share": float(group["realized_profit_per_share"].mean()),
            "total_profit_per_share": float(group["realized_profit_per_share"].sum()),
        }
        for optional in ["price_cents", "edge", "calibrated_expected_roi", "volume", "open_interest"]:
            if optional in group.columns:
                row[f"avg_{optional}"] = float(pd.to_numeric(group[optional], errors="coerce").mean())
        if group_columns:
            keys = key if isinstance(key, tuple) else (key,)
            for column, value in zip(group_columns, keys):
                row[column] = str(value)
        rows.append(row)
    output = pd.DataFrame(rows)
    if group_columns and not output.empty:
        metric_columns = [column for column in output.columns if column not in group_columns]
        output = output[group_columns + metric_columns]
    return output


def _bucketize(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["price_bucket"] = pd.cut(
        working.get("price_cents", pd.Series(np.nan, index=working.index)),
        bins=[0, 10, 15, 20, 25, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    working["edge_bucket"] = pd.cut(
        working.get("edge", pd.Series(np.nan, index=working.index)),
        bins=[-np.inf, 0.0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<=0", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    working["roi_bucket"] = pd.cut(
        working.get("calibrated_expected_roi", pd.Series(np.nan, index=working.index)),
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
        labels=["<=0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"],
    ).astype(str)
    working["liquidity_bucket"] = pd.cut(
        working.get("volume", pd.Series(np.nan, index=working.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
    return working


def _month_delta(monthly: pd.DataFrame) -> dict[str, Any]:
    if monthly.empty or len(monthly) < 2:
        return {}
    ordered = monthly.sort_values("month").reset_index(drop=True)
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    return {
        "first_month": str(first["month"]),
        "last_month": str(last["month"]),
        "first_positive_clv_rate": float(first["positive_clv_rate"]),
        "last_positive_clv_rate": float(last["positive_clv_rate"]),
        "positive_clv_rate_change": float(last["positive_clv_rate"] - first["positive_clv_rate"]),
        "first_avg_clv_cents": float(first["avg_clv_cents"]),
        "last_avg_clv_cents": float(last["avg_clv_cents"]),
        "avg_clv_cents_change": float(last["avg_clv_cents"] - first["avg_clv_cents"]),
        "first_avg_profit_per_share": float(first["avg_profit_per_share"]),
        "last_avg_profit_per_share": float(last["avg_profit_per_share"]),
        "avg_profit_per_share_change": float(last["avg_profit_per_share"] - first["avg_profit_per_share"]),
    }


def _segment_deltas(segment_monthly: pd.DataFrame, segment_column: str, min_rows: int = 5) -> pd.DataFrame:
    if segment_monthly.empty:
        return pd.DataFrame(columns=SEGMENT_DELTA_COLUMNS)
    rows: list[dict[str, Any]] = []
    for segment, group in segment_monthly.groupby(segment_column, dropna=False, observed=False):
        ordered = group.sort_values("month").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        if int(first["rows"]) < min_rows or int(last["rows"]) < min_rows:
            continue
        rows.append(
            {
                "segment_type": segment_column,
                "segment": str(segment),
                "first_month": str(first["month"]),
                "last_month": str(last["month"]),
                "first_rows": int(first["rows"]),
                "last_rows": int(last["rows"]),
                "positive_clv_rate_change": float(last["positive_clv_rate"] - first["positive_clv_rate"]),
                "avg_clv_cents_change": float(last["avg_clv_cents"] - first["avg_clv_cents"]),
                "avg_profit_per_share_change": float(last["avg_profit_per_share"] - first["avg_profit_per_share"]),
                "last_positive_clv_rate": float(last["positive_clv_rate"]),
                "last_avg_clv_cents": float(last["avg_clv_cents"]),
                "last_avg_profit_per_share": float(last["avg_profit_per_share"]),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=SEGMENT_DELTA_COLUMNS)
    return output.sort_values(["positive_clv_rate_change", "avg_clv_cents_change"]).reset_index(drop=True)


def build_clv_decay_audit(
    rows: pd.DataFrame,
    signal_column: str = "walk_forward_clv_price_signal",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build month and segment reports explaining CLV hit-rate decay."""

    selected = _bucketize(_selected(rows, signal_column=signal_column))
    if selected.empty:
        return {
            "monthly": pd.DataFrame(columns=MONTHLY_COLUMNS),
            "by_month_price_bucket": pd.DataFrame(columns=["month", "price_bucket", *MONTHLY_COLUMNS[1:]]),
            "by_month_edge_bucket": pd.DataFrame(columns=["month", "edge_bucket", *MONTHLY_COLUMNS[1:]]),
            "by_month_roi_bucket": pd.DataFrame(columns=["month", "roi_bucket", *MONTHLY_COLUMNS[1:]]),
            "by_month_liquidity_bucket": pd.DataFrame(columns=["month", "liquidity_bucket", *MONTHLY_COLUMNS[1:]]),
            "negative_clv_rows": pd.DataFrame(columns=NEGATIVE_CLV_COLUMNS),
            "decay_drivers": pd.DataFrame(columns=SEGMENT_DELTA_COLUMNS),
        }, {"rows": 0, "status": "no_selected_rows"}

    monthly = _summary(selected, ["month"]).sort_values("month").reset_index(drop=True)
    reports: dict[str, pd.DataFrame] = {
        "monthly": monthly,
        "by_month_price_bucket": _summary(selected, ["month", "price_bucket"]),
        "by_month_edge_bucket": _summary(selected, ["month", "edge_bucket"]),
        "by_month_roi_bucket": _summary(selected, ["month", "roi_bucket"]),
        "by_month_liquidity_bucket": _summary(selected, ["month", "liquidity_bucket"]),
    }
    if "yes_team_abbr" in selected.columns:
        team_month = _summary(selected, ["month", "yes_team_abbr"])
        reports["by_month_team"] = team_month.sort_values(
            ["month", "rows", "positive_clv_rate"],
            ascending=[True, False, True],
        )
    negative = selected[selected["clv_cents"] <= 0].copy()
    reports["negative_clv_rows"] = negative[
        [column for column in NEGATIVE_CLV_COLUMNS if column in negative.columns]
    ].sort_values(["month", "clv_cents"]).reset_index(drop=True)

    driver_frames = []
    for report_name, segment_column in [
        ("by_month_price_bucket", "price_bucket"),
        ("by_month_edge_bucket", "edge_bucket"),
        ("by_month_roi_bucket", "roi_bucket"),
        ("by_month_liquidity_bucket", "liquidity_bucket"),
    ]:
        deltas = _segment_deltas(reports[report_name], segment_column)
        if not deltas.empty:
            driver_frames.append(deltas)
    reports["decay_drivers"] = (
        pd.concat(driver_frames, ignore_index=True).sort_values(
            ["positive_clv_rate_change", "avg_clv_cents_change"],
        ).reset_index(drop=True)
        if driver_frames
        else pd.DataFrame()
    )

    delta = _month_delta(monthly)
    summary = {
        "rows": int(len(selected)),
        "months": int(selected["month"].nunique()),
        "avg_clv_cents": float(selected["clv_cents"].mean()),
        "positive_clv_rate": float((selected["clv_cents"] > 0).mean()),
        "avg_profit_per_share": float(selected["realized_profit_per_share"].mean()),
        "negative_clv_rows": int(len(negative)),
        "status": "decay_detected"
        if float(delta.get("positive_clv_rate_change", 0.0)) < 0
        else "no_decay_detected",
        **delta,
        "note": "Decay is measured from first evaluated selected month to last evaluated selected month.",
    }
    return reports, summary


def save_clv_decay_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "clv_decay",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
