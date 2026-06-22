"""Attribute calibrated signal results to pregame market movement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def _signal_side(frame: pd.DataFrame) -> pd.Series:
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
    required = [
        "date",
        signal_column,
        "price_cents",
        "clv_reference_price_cents",
        "clv_cents",
        "actual_contract_win",
        "realized_profit_per_share",
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Market movement audit rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["_side"] = _signal_side(frame)
    for column in [
        "price_cents",
        "clv_reference_price_cents",
        "clv_cents",
        "realized_profit_per_share",
        "model_prob",
        "model_yes_prob",
        "market_prob",
        "edge",
        "calibrated_expected_roi",
        "calibrated_expected_profit_per_share",
        "calibrated_win_rate",
        "volume",
        "open_interest",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["actual_contract_win"] = _coerce_bool(frame["actual_contract_win"])
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
    frame["market_move"] = np.select(
        [
            frame["clv_cents"] > 0,
            frame["clv_cents"] < 0,
        ],
        [
            "with_signal",
            "against_signal",
        ],
        default="flat",
    )
    frame["outcome_result"] = np.where(frame["actual_contract_win"], "won_settlement", "lost_settlement")
    frame["movement_outcome"] = frame["market_move"] + "_" + frame["outcome_result"]
    frame["profit_positive"] = frame["realized_profit_per_share"] > 0
    frame["price_bucket"] = pd.cut(
        frame["price_cents"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    frame["edge_bucket"] = pd.cut(
        frame.get("edge", pd.Series(np.nan, index=frame.index)),
        bins=[-np.inf, -0.15, -0.10, -0.05, 0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<-15%", "-15--10%", "-10--5%", "-5-0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
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


def _summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        row.update(
            {
                "rows": int(len(group)),
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "median_clv_cents": float(group["clv_cents"].median()),
                "positive_clv_rate": float((group["clv_cents"] > 0).mean()),
                "win_rate": float(group["actual_contract_win"].mean()),
                "avg_profit_per_share": float(group["realized_profit_per_share"].mean()),
                "total_profit_per_share": float(group["realized_profit_per_share"].sum()),
                "avg_price_cents": float(group["price_cents"].mean()),
                "avg_edge": float(group["edge"].mean()) if "edge" in group.columns else 0.0,
                "avg_calibrated_roi": float(group["calibrated_expected_roi"].mean())
                if "calibrated_expected_roi" in group.columns
                else 0.0,
                "avg_volume": float(group["volume"].mean()) if "volume" in group.columns else 0.0,
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    return output[group_columns + [column for column in output.columns if column not in group_columns]].reset_index(
        drop=True
    )


def build_market_movement_audit(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build reports explaining whether signals beat later pregame prices."""

    signals = _prepare(rows, signal_column=signal_column)
    reports: dict[str, pd.DataFrame] = {
        "signals": signals,
        "overall": _summary(signals, ["_side"]) if not signals.empty else pd.DataFrame(),
    }
    for columns, name in [
        (["_side", "market_move"], "by_side_move"),
        (["_side", "movement_outcome"], "by_side_move_outcome"),
        (["_side", "month"], "by_side_month"),
        (["_side", "price_bucket"], "by_side_price"),
        (["_side", "edge_bucket"], "by_side_edge"),
        (["_side", "roi_bucket"], "by_side_roi"),
        (["_side", "liquidity_bucket"], "by_side_liquidity"),
        (["_side", "clv_reference_snapshot"], "by_side_reference_snapshot"),
    ]:
        if all(column in signals.columns for column in columns):
            reports[name] = _summary(signals, columns)
    if "yes_team_abbr" in signals.columns:
        reports["by_side_team"] = _summary(signals, ["_side", "yes_team_abbr"])
    if "yes_location" in signals.columns:
        reports["by_side_location"] = _summary(signals, ["_side", "yes_location"])

    if signals.empty:
        summary = {
            "signals": 0,
            "status": "no_signals",
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
        }
        return reports, summary

    settlement_winners = signals[signals["actual_contract_win"]].copy()
    clv_winners = signals[signals["clv_cents"] > 0].copy()
    profit_not_clv = signals[(signals["realized_profit_per_share"] > 0) & (signals["clv_cents"] <= 0)].copy()
    clv_not_profit = signals[(signals["clv_cents"] > 0) & (signals["realized_profit_per_share"] <= 0)].copy()
    reports["profit_without_clv"] = profit_not_clv.sort_values(
        ["realized_profit_per_share", "clv_cents"],
        ascending=[False, True],
    ).reset_index(drop=True)
    reports["clv_without_profit"] = clv_not_profit.sort_values(
        ["clv_cents", "realized_profit_per_share"],
        ascending=[False, True],
    ).reset_index(drop=True)

    summary = {
        "signals": int(len(signals)),
        "status": "not_proven",
        "avg_clv_cents": float(signals["clv_cents"].mean()),
        "positive_clv_rate": float((signals["clv_cents"] > 0).mean()),
        "win_rate": float(signals["actual_contract_win"].mean()),
        "avg_profit_per_share": float(signals["realized_profit_per_share"].mean()),
        "profit_without_clv_count": int(len(profit_not_clv)),
        "profit_without_clv_share": float(len(profit_not_clv) / len(signals)),
        "clv_without_profit_count": int(len(clv_not_profit)),
        "clv_without_profit_share": float(len(clv_not_profit) / len(signals)),
        "settlement_winner_positive_clv_rate": float((settlement_winners["clv_cents"] > 0).mean())
        if len(settlement_winners)
        else 0.0,
        "clv_winner_settlement_win_rate": float(clv_winners["actual_contract_win"].mean()) if len(clv_winners) else 0.0,
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "interpretation": (
            "If profit_without_clv_share is high and positive_clv_rate is low, results are outcome variance rather "
            "than repeatable market-beating entries."
        ),
    }
    return reports, summary


def save_market_movement_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "market_movement",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
