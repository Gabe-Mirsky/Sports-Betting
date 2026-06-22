"""Closing-line value summaries for single-game market signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _trade_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "clv_cents" not in trades.columns:
        return pd.DataFrame()
    working = trades.copy()
    if "trade" in working.columns:
        if pd.api.types.is_bool_dtype(working["trade"]):
            working = working[working["trade"]].copy()
        else:
            working = working[working["trade"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    working["clv_cents"] = pd.to_numeric(working["clv_cents"], errors="coerce")
    working = working[working["clv_cents"].notna()].copy()
    if "price_cents" in working.columns:
        working["price_cents"] = pd.to_numeric(working["price_cents"], errors="coerce")
    if "edge" in working.columns:
        working["edge"] = pd.to_numeric(working["edge"], errors="coerce")
    if "volume" in working.columns:
        working["volume"] = pd.to_numeric(working["volume"], errors="coerce")
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
    return working


def _summary(frame: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    if frame.empty:
        columns = ["rows", "avg_clv_cents", "median_clv_cents", "positive_clv_rate"]
        return pd.DataFrame(columns=([group_column] if group_column else []) + columns)
    if group_column is None:
        grouped = [(None, frame)]
    else:
        grouped = list(frame.groupby(group_column, dropna=False, observed=False))

    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        row: dict[str, Any] = {
            "rows": int(len(group)),
            "avg_clv_cents": float(group["clv_cents"].mean()),
            "median_clv_cents": float(group["clv_cents"].median()),
            "positive_clv_rate": float((group["clv_cents"] > 0).mean()),
        }
        if group_column is not None:
            row[group_column] = str(key)
        rows.append(row)
    output = pd.DataFrame(rows)
    if group_column is not None:
        output = output[[group_column, "rows", "avg_clv_cents", "median_clv_cents", "positive_clv_rate"]]
    return output


def build_clv_reports(trades: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return CLV summaries by edge, price, team, season, and liquidity."""

    working = _trade_rows(trades)
    if working.empty:
        empty = _summary(working)
        return {"overall": empty}, {"trades_with_clv": 0}

    reports: dict[str, pd.DataFrame] = {"overall": _summary(working)}
    if "edge" in working.columns and working["edge"].notna().any():
        working["edge_bucket"] = pd.cut(
            working["edge"],
            bins=[-np.inf, 0.0, 0.02, 0.05, 0.08, 0.12, np.inf],
            labels=["<=0", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
        )
        reports["by_edge_bucket"] = _summary(working, "edge_bucket")
    if "price_cents" in working.columns and working["price_cents"].notna().any():
        working["price_bucket"] = pd.cut(
            working["price_cents"],
            bins=[0, 25, 40, 55, 70, 85, 100],
            labels=["0-25", "25-40", "40-55", "55-70", "70-85", "85-100"],
            include_lowest=True,
        )
        reports["by_price_bucket"] = _summary(working, "price_bucket")
    if "yes_team_abbr" in working.columns:
        reports["by_team"] = _summary(working, "yes_team_abbr").sort_values(
            ["rows", "avg_clv_cents"],
            ascending=[False, False],
        )
    if "side" in working.columns:
        reports["by_side"] = _summary(working, "side")
    if "season" in working.columns:
        reports["by_season"] = _summary(working, "season")
    elif "date" in working.columns and working["date"].notna().any():
        working["season"] = working["date"].dt.year
        reports["by_season"] = _summary(working, "season")
    if "volume" in working.columns and working["volume"].notna().any():
        working["liquidity_bucket"] = pd.cut(
            working["volume"],
            bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
            labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
        )
        reports["by_liquidity"] = _summary(working, "liquidity_bucket")

    overall = reports["overall"].iloc[0].to_dict()
    summary = {
        "trades_with_clv": int(overall["rows"]),
        "avg_clv_cents": float(overall["avg_clv_cents"]),
        "median_clv_cents": float(overall["median_clv_cents"]),
        "positive_clv_rate": float(overall["positive_clv_rate"]),
    }
    return reports, summary


def save_clv_reports(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "clv",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
