"""Audit whether a pregame snapshot policy creates broad CLV improvement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def _side(frame: pd.DataFrame) -> pd.Series:
    if "side" in frame.columns:
        side = frame["side"]
    elif "candidate_side" in frame.columns:
        side = frame["candidate_side"]
    else:
        side = pd.Series("YES", index=frame.index)
    side = side.fillna("").astype(str).str.upper()
    return side.where(side.isin({"YES", "NO"}), "YES")


def _prepare(rows: pd.DataFrame, signal_column: str = "trade") -> pd.DataFrame:
    required = ["date", signal_column, "price_cents", "clv_cents", "profit"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Snapshot CLV audit rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["side"] = _side(frame)
    for column in ["price_cents", "clv_cents", "profit", "edge", "volume", "open_interest"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["_signal"]
        & frame["date"].notna()
        & frame["price_cents"].notna()
        & frame["clv_cents"].notna()
        & frame["profit"].notna()
    ].copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    frame["positive_clv"] = frame["clv_cents"] > 0
    frame["flat_clv"] = frame["clv_cents"].eq(0)
    frame["profitable"] = frame["profit"] > 0
    frame["price_bucket"] = pd.cut(
        frame["price_cents"],
        bins=[0, 10, 20, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    frame["edge_bucket"] = pd.cut(
        frame.get("edge", pd.Series(np.nan, index=frame.index)),
        bins=[-np.inf, 0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<=0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    frame["liquidity_bucket"] = pd.cut(
        frame.get("volume", pd.Series(np.nan, index=frame.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
    return frame.reset_index(drop=True)


def _summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    columns = group_columns + [
        "rows",
        "avg_clv_cents",
        "median_clv_cents",
        "positive_clv_rate",
        "flat_clv_rate",
        "avg_profit",
        "profit_rate",
        "avg_price_cents",
        "avg_edge",
        "avg_volume",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        row.update(
            {
                "rows": int(len(group)),
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "median_clv_cents": float(group["clv_cents"].median()),
                "positive_clv_rate": float(group["positive_clv"].mean()),
                "flat_clv_rate": float(group["flat_clv"].mean()),
                "avg_profit": float(group["profit"].mean()),
                "profit_rate": float(group["profitable"].mean()),
                "avg_price_cents": float(group["price_cents"].mean()),
                "avg_edge": float(group["edge"].mean()) if "edge" in group.columns else 0.0,
                "avg_volume": float(group["volume"].mean()) if "volume" in group.columns else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(["rows", "avg_clv_cents"], ascending=[False, False])


def build_snapshot_clv_audit(
    rows: pd.DataFrame,
    signal_column: str = "trade",
    concentration_top_n: int = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build reports for a snapshot policy's CLV distribution."""

    signals = _prepare(rows, signal_column=signal_column)
    reports: dict[str, pd.DataFrame] = {
        "signals": signals,
        "by_side": _summary(signals, ["side"]),
    }
    for columns, name in [
        (["side", "month"], "by_side_month"),
        (["side", "price_bucket"], "by_side_price"),
        (["side", "edge_bucket"], "by_side_edge"),
        (["side", "liquidity_bucket"], "by_side_liquidity"),
        (["side", "snapshot_target"], "by_side_entry_snapshot"),
        (["side", "clv_reference_snapshot"], "by_side_reference_snapshot"),
    ]:
        if all(column in signals.columns for column in columns):
            reports[name] = _summary(signals, columns)

    if signals.empty:
        return reports, {
            "status": "no_signals",
            "signals": 0,
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
        }

    reports["top_positive_clv"] = signals.sort_values("clv_cents", ascending=False).head(25).reset_index(drop=True)
    reports["top_negative_clv"] = signals.sort_values("clv_cents", ascending=True).head(25).reset_index(drop=True)

    positive = signals[signals["clv_cents"] > 0].copy()
    positive_total = float(positive["clv_cents"].sum()) if not positive.empty else 0.0
    top_positive_sum = (
        float(positive.sort_values("clv_cents", ascending=False).head(concentration_top_n)["clv_cents"].sum())
        if positive_total > 0
        else 0.0
    )
    top_positive_share = top_positive_sum / positive_total if positive_total > 0 else 0.0

    positive_months = signals.groupby("month", observed=False)["clv_cents"].mean().gt(0).mean()
    status = "watchlist" if signals["clv_cents"].mean() > 0 and positive_months >= 0.5 else "not_ready"
    if signals["positive_clv"].mean() < 0.35 or signals["profit"].mean() <= 0:
        status = "not_ready"

    summary = {
        "status": status,
        "signals": int(len(signals)),
        "avg_clv_cents": float(signals["clv_cents"].mean()),
        "median_clv_cents": float(signals["clv_cents"].median()),
        "positive_clv_rate": float(signals["positive_clv"].mean()),
        "flat_clv_rate": float(signals["flat_clv"].mean()),
        "avg_profit": float(signals["profit"].mean()),
        "profit_rate": float(signals["profitable"].mean()),
        "positive_month_share": float(positive_months),
        "top_positive_clv_count": int(min(concentration_top_n, len(positive))),
        "top_positive_clv_share": float(top_positive_share),
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "diagnosis": (
            "A snapshot policy needs broad positive CLV, not just a slightly positive average from rare large moves."
        ),
    }
    return reports, summary


def save_snapshot_clv_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "snapshot_clv",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
