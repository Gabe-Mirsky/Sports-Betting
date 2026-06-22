"""Audit NO-side CLV signals for pricing and outcome consistency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _prepare_no_rows(rows: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    required = [
        "date",
        "price_cents",
        "clv_reference_price_cents",
        "clv_cents",
        "realized_profit_per_share",
        signal_column,
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"NO-side audit rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    if "clv_filter_side" in frame.columns:
        frame["_audit_side"] = frame["clv_filter_side"]
    elif "calibrated_side" in frame.columns:
        frame["_audit_side"] = frame["calibrated_side"]
    elif "candidate_side" in frame.columns:
        frame["_audit_side"] = frame["candidate_side"]
    elif "side" in frame.columns:
        frame["_audit_side"] = frame["side"]
    else:
        raise ValueError("NO-side audit rows need one of: clv_filter_side, candidate_side, side")
    frame["_audit_side"] = frame["_audit_side"].fillna("").astype(str).str.upper()
    frame = _numeric(
        frame,
        [
            "price_cents",
            "clv_reference_price_cents",
            "clv_cents",
            "profit",
            "shares",
            "realized_profit_per_share",
            "model_expected_roi",
            "calibrated_expected_roi",
            "edge",
            "volume",
            "open_interest",
            "model_prob",
            "market_prob",
            "yes_bid",
            "yes_ask",
        ],
    )
    frame = frame[frame["_signal"] & frame["_audit_side"].eq("NO") & frame["date"].notna()].copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    frame["entry_no_price_cents"] = frame["price_cents"]
    frame["reference_no_price_cents"] = frame["clv_reference_price_cents"]
    frame["entry_yes_bid_cents"] = 100.0 - frame["entry_no_price_cents"]
    if "yes_bid" in frame.columns and frame["yes_bid"].notna().any():
        frame["entry_yes_bid_cents"] = frame["yes_bid"]
    frame["entry_yes_ask_cents"] = frame.get("yes_ask", pd.Series(np.nan, index=frame.index))
    frame["bid_ask_spread_cents"] = frame["entry_yes_ask_cents"] - frame["entry_yes_bid_cents"]
    frame["computed_no_clv_cents"] = frame["reference_no_price_cents"] - frame["entry_no_price_cents"]
    if "actual_contract_win" in frame.columns:
        frame["actual_contract_win_bool"] = _coerce_bool(frame["actual_contract_win"])
    elif "actual_yes_win" in frame.columns:
        frame["actual_contract_win_bool"] = ~_coerce_bool(frame["actual_yes_win"])
    else:
        frame["actual_contract_win_bool"] = frame["realized_profit_per_share"] > 0

    entry_cost = frame["entry_no_price_cents"] / 100.0
    frame["expected_profit_per_share_from_outcome"] = np.where(
        frame["actual_contract_win_bool"],
        1.0 - entry_cost,
        -entry_cost,
    )
    frame["profit_per_share_diff"] = (
        frame["realized_profit_per_share"] - frame["expected_profit_per_share_from_outcome"]
    )
    frame["positive_clv"] = frame["clv_cents"] > 0
    frame["positive_clv_loss"] = frame["positive_clv"] & (frame["realized_profit_per_share"] < 0)
    frame["large_positive_clv_loss"] = (frame["clv_cents"] >= 10.0) & (frame["realized_profit_per_share"] < 0)
    frame["profit_math_mismatch"] = frame["profit_per_share_diff"].abs() > 0.0001
    frame["missing_entry_or_reference"] = (
        frame["entry_no_price_cents"].isna() | frame["reference_no_price_cents"].isna()
    )
    if "clv_reference_snapshot" in frame.columns:
        frame["reference_not_5m"] = ~frame["clv_reference_snapshot"].astype(str).eq("pregame_5m")
    else:
        frame["reference_not_5m"] = False
    frame["low_volume"] = frame.get("volume", pd.Series(np.nan, index=frame.index)).fillna(0) < 100
    frame["low_open_interest"] = frame.get("open_interest", pd.Series(np.nan, index=frame.index)).fillna(0) < 100
    frame["wide_spread"] = frame["bid_ask_spread_cents"].fillna(np.inf) > 10
    return _bucketize(frame).reset_index(drop=True)


def _bucketize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["entry_price_bucket"] = pd.cut(
        output.get("entry_no_price_cents", pd.Series(np.nan, index=output.index)),
        bins=[0, 10, 20, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    output["yes_market_price_bucket"] = pd.cut(
        output.get("entry_yes_bid_cents", pd.Series(np.nan, index=output.index)),
        bins=[0, 10, 20, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    output["spread_bucket"] = pd.cut(
        output.get("bid_ask_spread_cents", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, 1, 2, 5, 10, np.inf],
        labels=["<=1c", "1-2c", "2-5c", "5-10c", ">10c"],
    ).astype(str)
    output["edge_bucket"] = pd.cut(
        output.get("edge", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, -0.10, -0.05, 0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<-10%", "-10--5%", "-5-0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    output["clv_bucket"] = pd.cut(
        output.get("clv_cents", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, -10, -2, 0, 2, 10, 25, np.inf],
        labels=["<-10c", "-10--2c", "-2-0c", "0-2c", "2-10c", "10-25c", "25c+"],
    ).astype(str)
    output["roi_bucket"] = pd.cut(
        output.get("calibrated_expected_roi", pd.Series(np.nan, index=output.index)),
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
        labels=["<=0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"],
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
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        row.update(
            {
                "rows": int(len(group)),
                "win_rate": float(group["actual_contract_win_bool"].mean()),
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "median_clv_cents": float(group["clv_cents"].median()),
                "positive_clv_rate": float(group["positive_clv"].mean()),
                "avg_profit_per_share": float(group["realized_profit_per_share"].mean()),
                "total_profit_per_share": float(group["realized_profit_per_share"].sum()),
                "positive_clv_loss_rate": float(group["positive_clv_loss"].mean()),
                "large_positive_clv_loss_rate": float(group["large_positive_clv_loss"].mean()),
                "profit_math_mismatch_count": int(group["profit_math_mismatch"].sum()),
                "low_volume_rate": float(group["low_volume"].mean()),
                "wide_spread_rate": float(group["wide_spread"].mean()),
                "reference_not_5m_rate": float(group["reference_not_5m"].mean()),
                "avg_entry_no_price_cents": float(group["entry_no_price_cents"].mean()),
                "avg_entry_yes_bid_cents": float(group["entry_yes_bid_cents"].mean()),
                "avg_bid_ask_spread_cents": float(group["bid_ask_spread_cents"].mean()),
                "avg_reference_no_price_cents": float(group["reference_no_price_cents"].mean()),
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    return output[group_columns + [column for column in output.columns if column not in group_columns]].reset_index(
        drop=True
    )


def build_no_side_audit(
    rows: pd.DataFrame,
    signal_column: str = "clv_filtered_trade",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build diagnostic reports for selected NO-side CLV signals."""

    selected = _prepare_no_rows(rows, signal_column=signal_column)
    reports: dict[str, pd.DataFrame] = {
        "rows": selected,
        "monthly": _summary(selected, ["month"]),
    }
    for column in [
        "entry_price_bucket",
        "yes_market_price_bucket",
        "spread_bucket",
        "edge_bucket",
        "clv_bucket",
        "roi_bucket",
        "liquidity_bucket",
        "clv_reference_snapshot",
        "yes_location",
        "yes_team_abbr",
    ]:
        if column in selected.columns:
            reports[f"by_{column}"] = _summary(selected, [column])

    if selected.empty:
        summary = {
            "selected_no_rows": 0,
            "status": "no_selected_no_rows",
            "note": "No selected NO signals were available for audit.",
        }
        return reports, summary

    positive_clv_losses = selected[selected["positive_clv_loss"]].copy()
    large_positive_clv_losses = selected[selected["large_positive_clv_loss"]].copy()
    reports["positive_clv_losses"] = positive_clv_losses.sort_values(
        ["clv_cents", "realized_profit_per_share"],
        ascending=[False, True],
    ).reset_index(drop=True)
    reports["large_positive_clv_losses"] = large_positive_clv_losses.sort_values(
        ["clv_cents", "realized_profit_per_share"],
        ascending=[False, True],
    ).reset_index(drop=True)
    reports["profit_math_mismatches"] = selected[selected["profit_math_mismatch"]].copy().reset_index(drop=True)

    summary = {
        "selected_no_rows": int(len(selected)),
        "status": "review_required",
        "avg_clv_cents": float(selected["clv_cents"].mean()),
        "median_clv_cents": float(selected["clv_cents"].median()),
        "positive_clv_rate": float(selected["positive_clv"].mean()),
        "win_rate": float(selected["actual_contract_win_bool"].mean()),
        "avg_profit_per_share": float(selected["realized_profit_per_share"].mean()),
        "positive_clv_loss_count": int(selected["positive_clv_loss"].sum()),
        "positive_clv_loss_rate": float(selected["positive_clv_loss"].mean()),
        "large_positive_clv_loss_count": int(selected["large_positive_clv_loss"].sum()),
        "profit_math_mismatch_count": int(selected["profit_math_mismatch"].sum()),
        "missing_entry_or_reference_count": int(selected["missing_entry_or_reference"].sum()),
        "reference_not_5m_count": int(selected["reference_not_5m"].sum()),
        "low_volume_count": int(selected["low_volume"].sum()),
        "low_volume_rate": float(selected["low_volume"].mean()),
        "wide_spread_count": int(selected["wide_spread"].sum()),
        "wide_spread_rate": float(selected["wide_spread"].mean()),
        "interpretation": (
            "High NO CLV is not proof of edge if it frequently settles as a loss, depends on low-volume "
            "markets, or uses a non-5m reference. Profit math mismatches indicate implementation bugs."
        ),
    }
    if summary["profit_math_mismatch_count"] == 0 and summary["missing_entry_or_reference_count"] == 0:
        summary["status"] = "math_consistent_review_economics"
    return reports, summary


def save_no_side_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "no_side_audit",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
