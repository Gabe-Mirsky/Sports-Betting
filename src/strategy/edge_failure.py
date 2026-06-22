"""Summarize why calibrated single-game edges are not yet proven."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


SEGMENT_COLUMNS = [
    "side",
    "price_bucket",
    "edge_bucket",
    "roi_bucket",
    "liquidity_bucket",
    "month",
    "yes_location",
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
    required = ["date", signal_column, "price_cents", "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Edge failure rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["side"] = _side_series(frame)
    for column in [
        "price_cents",
        "clv_cents",
        "realized_profit_per_share",
        "edge",
        "calibrated_expected_roi",
        "volume",
        "open_interest",
        "calibrated_win_rate",
    ]:
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
    frame["positive_clv"] = frame["clv_cents"] > 0
    frame["profitable"] = frame["realized_profit_per_share"] > 0
    frame["price_bucket"] = pd.cut(
        frame["price_cents"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    frame["edge_bucket"] = pd.cut(
        frame.get("edge", pd.Series(np.nan, index=frame.index)),
        bins=[-np.inf, -0.10, -0.05, 0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<-10%", "-10--5%", "-5-0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    frame["roi_bucket"] = pd.cut(
        frame.get("calibrated_expected_roi", pd.Series(np.nan, index=frame.index)),
        bins=[-np.inf, 0, 0.25, 0.5, 1, 2, 3, np.inf],
        labels=["<=0", "0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", "3+"],
    ).astype(str)
    frame["liquidity_bucket"] = pd.cut(
        frame.get("volume", pd.Series(np.nan, index=frame.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
    if "yes_team_abbr" in frame.columns and "home_team_abbr" in frame.columns:
        frame["yes_location"] = np.where(
            frame["yes_team_abbr"].astype(str).eq(frame["home_team_abbr"].astype(str)),
            "home",
            "away",
        )
    return frame.reset_index(drop=True)


def _segment_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    columns = group_columns + [
        "rows",
        "avg_clv_cents",
        "positive_clv_rate",
        "avg_profit_per_share",
        "profit_rate",
        "avg_price_cents",
        "avg_edge",
        "avg_calibrated_roi",
        "avg_volume",
        "failure_score",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=False):
        keys = key if isinstance(key, tuple) else (key,)
        avg_clv = float(group["clv_cents"].mean())
        positive_clv_rate = float(group["positive_clv"].mean())
        avg_profit = float(group["realized_profit_per_share"].mean())
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        row.update(
            {
                "rows": int(len(group)),
                "avg_clv_cents": avg_clv,
                "positive_clv_rate": positive_clv_rate,
                "avg_profit_per_share": avg_profit,
                "profit_rate": float(group["profitable"].mean()),
                "avg_price_cents": float(group["price_cents"].mean()),
                "avg_edge": float(group["edge"].mean()) if "edge" in group.columns else 0.0,
                "avg_calibrated_roi": float(group["calibrated_expected_roi"].mean())
                if "calibrated_expected_roi" in group.columns
                else 0.0,
                "avg_volume": float(group["volume"].mean()) if "volume" in group.columns else 0.0,
            }
        )
        row["failure_score"] = float(
            max(0.0, -avg_profit) * 2.0
            + max(0.0, -avg_clv / 100.0)
            + max(0.0, 0.50 - positive_clv_rate)
            + min(len(group), 100) / 500.0
        )
        rows.append(row)
    output = pd.DataFrame(rows, columns=columns)
    return output.sort_values(["failure_score", "rows"], ascending=[False, False]).reset_index(drop=True)


def build_edge_failure_diagnosis(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_segment_rows: int = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build compact reports identifying the largest CLV/profit failure slices."""

    signals = _prepare(rows, signal_column=signal_column)
    reports: dict[str, pd.DataFrame] = {"signals": signals}
    if signals.empty:
        return reports, {
            "signals": 0,
            "status": "no_signals",
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
        }

    reports["overall_by_side"] = _segment_summary(signals, ["side"])
    for column in SEGMENT_COLUMNS:
        if column in signals.columns:
            reports[f"by_{column}"] = _segment_summary(signals, [column])
    for columns, name in [
        (["side", "price_bucket"], "by_side_price"),
        (["side", "edge_bucket"], "by_side_edge"),
        (["side", "roi_bucket"], "by_side_roi"),
        (["side", "liquidity_bucket"], "by_side_liquidity"),
        (["side", "month"], "by_side_month"),
    ]:
        if all(column in signals.columns for column in columns):
            reports[name] = _segment_summary(signals, columns)

    candidate_frames = []
    for name, frame in reports.items():
        if name == "signals" or frame.empty or "failure_score" not in frame.columns:
            continue
        filtered = frame[pd.to_numeric(frame["rows"], errors="coerce") >= int(min_segment_rows)].copy()
        if filtered.empty:
            continue
        filtered.insert(0, "segment_report", name)
        candidate_frames.append(filtered)
    reports["worst_segments"] = (
        pd.concat(candidate_frames, ignore_index=True, sort=False)
        .sort_values(["failure_score", "rows"], ascending=[False, False])
        .head(50)
        .reset_index(drop=True)
        if candidate_frames
        else pd.DataFrame()
    )

    overall = _segment_summary(signals, ["side"])
    failing_segments = reports["worst_segments"]
    summary = {
        "signals": int(len(signals)),
        "status": "not_proven",
        "avg_clv_cents": float(signals["clv_cents"].mean()),
        "positive_clv_rate": float(signals["positive_clv"].mean()),
        "avg_profit_per_share": float(signals["realized_profit_per_share"].mean()),
        "profit_rate": float(signals["profitable"].mean()),
        "side_count": int(signals["side"].nunique()),
        "worst_segment": str(failing_segments.iloc[0]["segment_report"]) if not failing_segments.empty else "n/a",
        "worst_segment_rows": int(failing_segments.iloc[0]["rows"]) if not failing_segments.empty else 0,
        "worst_segment_failure_score": float(failing_segments.iloc[0]["failure_score"])
        if not failing_segments.empty
        else 0.0,
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "diagnosis": (
            "The current calibrated signals still fail the CLV gate. Use worst_segments.csv to pick the next "
            "model/calibration hypothesis; do not turn these slices into betting rules until they pass walk-forward CLV."
        ),
    }
    for _, row in overall.iterrows():
        side = str(row["side"]).lower()
        summary[f"{side}_signals"] = int(row["rows"])
        summary[f"{side}_avg_clv_cents"] = float(row["avg_clv_cents"])
        summary[f"{side}_positive_clv_rate"] = float(row["positive_clv_rate"])
        summary[f"{side}_avg_profit_per_share"] = float(row["avg_profit_per_share"])
    return reports, summary


def save_edge_failure_diagnosis(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "edge_failure",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
