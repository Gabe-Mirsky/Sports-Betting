"""Audit month-specific failures in defensive walk-forward signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


SCHEDULE_CONTEXT_COLUMNS = [
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_is_back_to_back",
    "away_is_back_to_back",
    "yes_rest_days",
    "opponent_rest_days",
    "yes_is_back_to_back",
    "opponent_is_back_to_back",
]
SUMMARY_COLUMNS = [
    "rows",
    "avg_clv_cents",
    "positive_clv_rate",
    "avg_profit_per_share",
    "total_profit_per_share",
    "loss_rate",
]
DETAIL_COLUMNS = [
    "date",
    "game_id",
    "market_ticker",
    "home_team_abbr",
    "away_team_abbr",
    "yes_team_abbr",
    "yes_location",
    "price_cents",
    "calibrated_expected_roi",
    "edge",
    "volume",
    "open_interest",
    "clv_cents",
    "realized_profit_per_share",
    "price_bucket",
    "roi_bucket",
    "edge_bucket",
    "liquidity_bucket",
]


def _selected(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    if rows.empty:
        frame = rows.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["month"] = pd.Series(dtype=object)
        frame["is_failure_month"] = pd.Series(dtype=bool)
        return frame
    required = ["date", signal_column, "clv_cents", "realized_profit_per_share", "price_cents"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Defensive failure audit rows are missing columns: {missing}")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    for column in [
        "clv_cents",
        "realized_profit_per_share",
        "price_cents",
        "edge",
        "calibrated_expected_roi",
        "volume",
        "open_interest",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["_signal"] & frame["date"].notna()].copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    frame["is_failure_month"] = frame["month"].eq("")
    return frame.reset_index(drop=True)


def _bucketize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["price_bucket"] = pd.cut(
        output.get("price_cents", pd.Series(np.nan, index=output.index)),
        bins=[0, 10, 15, 20, 25, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    output["roi_bucket"] = pd.cut(
        output.get("calibrated_expected_roi", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
        labels=["<=0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"],
    ).astype(str)
    output["edge_bucket"] = pd.cut(
        output.get("edge", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, 0.0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<=0", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    output["liquidity_bucket"] = pd.cut(
        output.get("volume", pd.Series(np.nan, index=output.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
    if "yes_team_abbr" in output.columns and "home_team_abbr" in output.columns:
        output["yes_location"] = np.where(
            output["yes_team_abbr"].astype(str).eq(output["home_team_abbr"].astype(str)),
            "home",
            "away",
        )
    return output


def _summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(group_columns) + SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        row.update(
            {
                "rows": int(len(group)),
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "positive_clv_rate": float((group["clv_cents"] > 0).mean()),
                "avg_profit_per_share": float(group["realized_profit_per_share"].mean()),
                "total_profit_per_share": float(group["realized_profit_per_share"].sum()),
                "loss_rate": float((group["realized_profit_per_share"] < 0).mean()),
            }
        )
        for column in ["price_cents", "edge", "calibrated_expected_roi", "volume", "open_interest"]:
            if column in group.columns:
                row[f"avg_{column}"] = float(group[column].mean())
        rows.append(row)
    output = pd.DataFrame(rows)
    metric_columns = [column for column in output.columns if column not in group_columns]
    return output[group_columns + metric_columns].reset_index(drop=True)


def _compare_failure_month(frame: pd.DataFrame, failure_month: str, group_column: str, min_rows: int) -> pd.DataFrame:
    if frame.empty or group_column not in frame.columns:
        return pd.DataFrame()
    grouped = _summary(frame, ["month", group_column])
    if grouped.empty:
        return grouped
    failure = grouped[grouped["month"].eq(failure_month)].copy()
    other = frame[~frame["month"].eq(failure_month)].copy()
    other_summary = _summary(other, [group_column])
    if failure.empty or other_summary.empty:
        return pd.DataFrame()
    merged = failure.merge(other_summary, on=group_column, how="inner", suffixes=("_failure_month", "_other_months"))
    merged = merged[
        (pd.to_numeric(merged["rows_failure_month"], errors="coerce") >= int(min_rows))
        & (pd.to_numeric(merged["rows_other_months"], errors="coerce") >= int(min_rows))
    ].copy()
    if merged.empty:
        return merged
    merged["profit_delta"] = merged["avg_profit_per_share_failure_month"] - merged["avg_profit_per_share_other_months"]
    merged["clv_rate_delta"] = merged["positive_clv_rate_failure_month"] - merged["positive_clv_rate_other_months"]
    merged["avg_clv_delta"] = merged["avg_clv_cents_failure_month"] - merged["avg_clv_cents_other_months"]
    return merged.sort_values(["profit_delta", "clv_rate_delta"]).reset_index(drop=True)


def _load_schedule_context(path: str | Path | None) -> tuple[pd.DataFrame, str]:
    if path is None:
        return pd.DataFrame(), "not_requested"
    schedule_path = Path(path)
    if not schedule_path.exists():
        return pd.DataFrame(), "missing_file"
    try:
        if schedule_path.suffix.lower() == ".csv":
            frame = pd.read_csv(schedule_path, dtype={"game_id": str})
        elif schedule_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(schedule_path)
        else:
            return pd.DataFrame(), "unsupported_file_type"
    except Exception as exc:  # pragma: no cover - depends on optional parquet engines
        return pd.DataFrame(), f"read_failed:{type(exc).__name__}"
    keep = ["game_id", "game_date", "home_team_abbr", "away_team_abbr"] + [
        column for column in SCHEDULE_CONTEXT_COLUMNS if column in frame.columns
    ]
    keep = [column for column in keep if column in frame.columns]
    if not {"game_id"}.issubset(keep):
        return pd.DataFrame(), "missing_join_key"
    return frame[keep].drop_duplicates("game_id"), "loaded"


def build_defensive_failure_audit(
    rows: pd.DataFrame,
    signal_column: str = "walk_forward_defensive_signal",
    failure_month: str = "2026-03",
    schedule_context_path: str | Path | None = None,
    min_segment_rows: int = 3,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Explain why the defensive walk-forward strategy failed in one month."""

    selected = _bucketize(_selected(rows, signal_column=signal_column))
    schedule_context, schedule_status = _load_schedule_context(schedule_context_path)
    schedule_columns: list[str] = []
    if not schedule_context.empty:
        schedule_columns = [column for column in SCHEDULE_CONTEXT_COLUMNS if column in schedule_context.columns]
        selected = selected.merge(schedule_context[["game_id"] + schedule_columns], on="game_id", how="left")
    available_schedule_columns = [column for column in SCHEDULE_CONTEXT_COLUMNS if column in selected.columns]

    reports: dict[str, pd.DataFrame] = {
        "monthly": _summary(selected, ["month"]) if not selected.empty else pd.DataFrame(),
    }
    for segment in ["price_bucket", "roi_bucket", "edge_bucket", "liquidity_bucket", "yes_location"]:
        if segment in selected.columns:
            reports[f"by_month_{segment}"] = _summary(selected, ["month", segment])
            reports[f"failure_vs_other_{segment}"] = _compare_failure_month(
                selected,
                failure_month,
                segment,
                min_rows=min_segment_rows,
            )
    if "yes_team_abbr" in selected.columns:
        reports["by_month_team"] = _summary(selected, ["month", "yes_team_abbr"])
        reports["failure_vs_other_team"] = _compare_failure_month(
            selected,
            failure_month,
            "yes_team_abbr",
            min_rows=min_segment_rows,
        )
    for column in available_schedule_columns:
        selected[column] = selected[column].astype(str)
        reports[f"by_month_{column}"] = _summary(selected, ["month", column])
        reports[f"failure_vs_other_{column}"] = _compare_failure_month(
            selected,
            failure_month,
            column,
            min_rows=min_segment_rows,
        )
    failure_rows = selected[selected["month"].eq(failure_month)].copy()
    failure_rows = failure_rows.sort_values(["realized_profit_per_share", "clv_cents"]).reset_index(drop=True)
    detail_columns = DETAIL_COLUMNS + available_schedule_columns
    reports["failure_month_rows"] = failure_rows[[column for column in detail_columns if column in failure_rows.columns]]

    monthly = reports["monthly"]
    failure_summary = monthly[monthly["month"].eq(failure_month)].iloc[0].to_dict() if not monthly.empty and monthly["month"].eq(failure_month).any() else {}
    other_rows = selected[~selected["month"].eq(failure_month)].copy()
    summary = {
        "rows": int(len(selected)),
        "failure_month": failure_month,
        "failure_month_rows": int(len(failure_rows)),
        "failure_month_avg_profit_per_share": float(failure_summary.get("avg_profit_per_share", 0.0) or 0.0),
        "failure_month_positive_clv_rate": float(failure_summary.get("positive_clv_rate", 0.0) or 0.0),
        "failure_month_avg_clv_cents": float(failure_summary.get("avg_clv_cents", 0.0) or 0.0),
        "other_month_rows": int(len(other_rows)),
        "other_month_avg_profit_per_share": float(other_rows["realized_profit_per_share"].mean()) if len(other_rows) else 0.0,
        "other_month_positive_clv_rate": float((other_rows["clv_cents"] > 0).mean()) if len(other_rows) else 0.0,
        "other_month_avg_clv_cents": float(other_rows["clv_cents"].mean()) if len(other_rows) else 0.0,
        "schedule_context_status": schedule_status,
        "available_schedule_context_columns": available_schedule_columns,
        "note": "Team slices are diagnostic only. Prefer price/ROI/liquidity rules unless team effects survive walk-forward validation.",
    }
    return reports, summary


def save_defensive_failure_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "defensive_failure",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
