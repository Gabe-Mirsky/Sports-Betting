"""Audit whether model-vs-market residuals translate into real edge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


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
    required = [
        "date",
        signal_column,
        "price_cents",
        "market_prob",
        "model_prob",
        "edge",
        "calibrated_win_rate",
        "actual_contract_win",
        "realized_profit_per_share",
        "clv_cents",
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Residual audit rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["_signal"] = _coerce_bool(frame[signal_column])
    frame["_side"] = _side(frame)
    for column in [
        "price_cents",
        "market_prob",
        "model_prob",
        "model_yes_prob",
        "edge",
        "calibrated_win_rate",
        "calibrated_expected_profit_per_share",
        "calibrated_expected_roi",
        "actual_contract_win",
        "realized_profit_per_share",
        "clv_cents",
        "volume",
        "open_interest",
    ]:
        if column in frame.columns and column != "actual_contract_win":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["actual_contract_win"] = _coerce_bool(frame["actual_contract_win"])
    frame = frame[
        frame["_signal"]
        & frame["date"].notna()
        & frame["price_cents"].notna()
        & frame["market_prob"].notna()
        & frame["model_prob"].notna()
        & frame["calibrated_win_rate"].notna()
        & frame["realized_profit_per_share"].notna()
        & frame["clv_cents"].notna()
    ].copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    frame["contract_cost"] = frame["price_cents"] / 100.0
    frame["raw_residual"] = frame["model_prob"] - frame["market_prob"]
    frame["calibrated_residual"] = frame["calibrated_win_rate"] - frame["contract_cost"]
    if "calibrated_expected_profit_per_share" not in frame.columns:
        frame["calibrated_expected_profit_per_share"] = frame["calibrated_residual"]
    frame["calibration_error"] = frame["actual_contract_win"].astype(float) - frame["calibrated_win_rate"]
    frame["market_error"] = frame["actual_contract_win"].astype(float) - frame["market_prob"]
    frame["model_error"] = frame["actual_contract_win"].astype(float) - frame["model_prob"]
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    frame["price_bucket"] = pd.cut(
        frame["price_cents"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    frame["market_price_zone"] = pd.cut(
        frame["price_cents"],
        bins=[0, 25, 55, 100],
        labels=["cheap", "mid", "expensive"],
        include_lowest=True,
    ).astype(str)
    frame["raw_residual_bucket"] = pd.cut(
        frame["raw_residual"],
        bins=[-np.inf, -0.15, -0.10, -0.05, 0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=["<-15%", "-15--10%", "-10--5%", "-5-0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"],
    ).astype(str)
    frame["calibrated_residual_bucket"] = pd.cut(
        frame["calibrated_residual"],
        bins=[-np.inf, 0, 0.02, 0.05, 0.08, 0.12, 0.20, np.inf],
        labels=["<=0", "0-2%", "2-5%", "5-8%", "8-12%", "12-20%", "20%+"],
    ).astype(str)
    frame["liquidity_bucket"] = pd.cut(
        frame.get("volume", pd.Series(np.nan, index=frame.index)).fillna(0),
        bins=[-np.inf, 10, 100, 1000, 10000, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", "10k+"],
    ).astype(str)
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
                "avg_price_cents": float(group["price_cents"].mean()),
                "avg_market_prob": float(group["market_prob"].mean()),
                "avg_model_prob": float(group["model_prob"].mean()),
                "avg_calibrated_win_rate": float(group["calibrated_win_rate"].mean()),
                "realized_win_rate": float(group["actual_contract_win"].mean()),
                "avg_raw_residual": float(group["raw_residual"].mean()),
                "avg_calibrated_residual": float(group["calibrated_residual"].mean()),
                "avg_calibration_error": float(group["calibration_error"].mean()),
                "avg_market_error": float(group["market_error"].mean()),
                "avg_model_error": float(group["model_error"].mean()),
                "avg_profit_per_share": float(group["realized_profit_per_share"].mean()),
                "total_profit_per_share": float(group["realized_profit_per_share"].sum()),
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "positive_clv_rate": float((group["clv_cents"] > 0).mean()),
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    return output[group_columns + [column for column in output.columns if column not in group_columns]].reset_index(
        drop=True
    )


def build_residual_audit(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build residual-vs-outcome and residual-vs-CLV reports."""

    signals = _prepare(rows, signal_column=signal_column)
    reports: dict[str, pd.DataFrame] = {
        "signals": signals,
        "overall": _summary(signals, ["_side"]) if not signals.empty else pd.DataFrame(),
    }
    for columns, name in [
        (["_side", "calibrated_residual_bucket"], "by_side_calibrated_residual"),
        (["_side", "raw_residual_bucket"], "by_side_raw_residual"),
        (["_side", "market_price_zone"], "by_side_price_zone"),
        (["_side", "price_bucket"], "by_side_price"),
        (["_side", "month"], "by_side_month"),
        (["_side", "liquidity_bucket"], "by_side_liquidity"),
        (["_side", "market_price_zone", "calibrated_residual_bucket"], "by_side_price_zone_calibrated_residual"),
    ]:
        if all(column in signals.columns for column in columns):
            reports[name] = _summary(signals, columns)

    if signals.empty:
        return reports, {
            "signals": 0,
            "status": "no_signals",
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
        }

    high_residual = signals[signals["calibrated_residual"] >= 0.08].copy()
    low_price = signals[signals["market_price_zone"].eq("cheap")].copy()
    summary = {
        "signals": int(len(signals)),
        "status": "not_proven",
        "avg_calibrated_residual": float(signals["calibrated_residual"].mean()),
        "avg_raw_residual": float(signals["raw_residual"].mean()),
        "realized_win_rate": float(signals["actual_contract_win"].mean()),
        "avg_calibrated_win_rate": float(signals["calibrated_win_rate"].mean()),
        "avg_calibration_error": float(signals["calibration_error"].mean()),
        "avg_profit_per_share": float(signals["realized_profit_per_share"].mean()),
        "avg_clv_cents": float(signals["clv_cents"].mean()),
        "positive_clv_rate": float((signals["clv_cents"] > 0).mean()),
        "high_residual_rows": int(len(high_residual)),
        "high_residual_win_rate": float(high_residual["actual_contract_win"].mean()) if len(high_residual) else 0.0,
        "high_residual_avg_profit_per_share": float(high_residual["realized_profit_per_share"].mean())
        if len(high_residual)
        else 0.0,
        "high_residual_positive_clv_rate": float((high_residual["clv_cents"] > 0).mean()) if len(high_residual) else 0.0,
        "cheap_rows": int(len(low_price)),
        "cheap_win_rate": float(low_price["actual_contract_win"].mean()) if len(low_price) else 0.0,
        "cheap_avg_profit_per_share": float(low_price["realized_profit_per_share"].mean()) if len(low_price) else 0.0,
        "cheap_positive_clv_rate": float((low_price["clv_cents"] > 0).mean()) if len(low_price) else 0.0,
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "interpretation": (
            "If calibrated residuals are large but realized win rate and CLV do not rise by bucket, calibration is "
            "manufacturing false edges rather than finding tradable mispricing."
        ),
    }
    return reports, summary


def save_residual_audit(
    reports: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "residual",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(output_root / f"{prefix}_{name}.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_residual_guardrail_sweep(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    min_history_options: list[int] | None = None,
    min_prior_calibration_error_options: list[float] | None = None,
    min_prior_positive_clv_rate_options: list[float] | None = None,
    min_prior_profit_options: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Walk-forward test side/price-zone guardrails against calibrated signals."""

    signals = _prepare(rows, signal_column=signal_column)
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "rules_tested": 0, "status": "not_ready"}
    min_history_options = min_history_options or [10, 20, 30, 40]
    min_prior_calibration_error_options = min_prior_calibration_error_options or [-0.05, -0.10, -0.15, -0.20]
    min_prior_positive_clv_rate_options = min_prior_positive_clv_rate_options or [0.20, 0.25, 0.30]
    min_prior_profit_options = min_prior_profit_options or [0.0]

    rule_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    sort_columns = ["date"] + (["market_ticker"] if "market_ticker" in signals.columns else [])
    sorted_signals = signals.sort_values(sort_columns).reset_index(drop=True)
    for min_history in min_history_options:
        for min_error in min_prior_calibration_error_options:
            for min_pos_clv in min_prior_positive_clv_rate_options:
                for min_profit in min_prior_profit_options:
                    history: dict[tuple[str, str], list[dict[str, float]]] = {}
                    selected_rows: list[pd.Series] = []
                    for _, slate in sorted_signals.groupby(sorted_signals["date"].dt.date, sort=True):
                        for _, row in slate.iterrows():
                            key = (str(row["_side"]), str(row["market_price_zone"]))
                            prior = history.get(key, [])
                            pass_rule = False
                            if len(prior) >= int(min_history):
                                prior_frame = pd.DataFrame(prior)
                                prior_error = float((prior_frame["actual"] - prior_frame["calibrated"]).mean())
                                prior_profit = float(prior_frame["profit"].mean())
                                prior_pos_clv = float((prior_frame["clv"] > 0).mean())
                                pass_rule = (
                                    prior_error >= float(min_error)
                                    and prior_profit >= float(min_profit)
                                    and prior_pos_clv >= float(min_pos_clv)
                                )
                            if pass_rule:
                                selected_rows.append(row)
                        for _, row in slate.iterrows():
                            key = (str(row["_side"]), str(row["market_price_zone"]))
                            history.setdefault(key, []).append(
                                {
                                    "actual": float(row["actual_contract_win"]),
                                    "calibrated": float(row["calibrated_win_rate"]),
                                    "profit": float(row["realized_profit_per_share"]),
                                    "clv": float(row["clv_cents"]),
                                }
                            )

                    selected = pd.DataFrame(selected_rows)
                    if not selected.empty:
                        selected = selected.copy()
                        selected["guardrail_rule"] = (
                            f"history>={min_history},cal_error>={min_error},"
                            f"pos_clv>={min_pos_clv},profit>={min_profit}"
                        )
                        selected_frames.append(selected)
                    months = int(selected["month"].nunique()) if not selected.empty else 0
                    positive_month_share = 0.0
                    if not selected.empty:
                        monthly = selected.groupby("month")["realized_profit_per_share"].mean()
                        positive_month_share = float((monthly > 0).mean()) if len(monthly) else 0.0
                    row = {
                        "min_history": int(min_history),
                        "min_prior_calibration_error": float(min_error),
                        "min_prior_positive_clv_rate": float(min_pos_clv),
                        "min_prior_profit": float(min_profit),
                        "signals": int(len(selected)),
                        "months": months,
                        "positive_month_share": positive_month_share,
                        "avg_profit_per_share": float(selected["realized_profit_per_share"].mean())
                        if len(selected)
                        else 0.0,
                        "avg_clv_cents": float(selected["clv_cents"].mean()) if len(selected) else 0.0,
                        "positive_clv_rate": float((selected["clv_cents"] > 0).mean()) if len(selected) else 0.0,
                        "win_rate": float(selected["actual_contract_win"].mean()) if len(selected) else 0.0,
                    }
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
                    row["score"] = (
                        row["avg_profit_per_share"] * 0.5
                        + row["avg_clv_cents"] / 100.0 * 0.25
                        + row["positive_clv_rate"] * 0.15
                        + row["positive_month_share"] * 0.10
                    )
                    rule_rows.append(row)

    rules = pd.DataFrame(rule_rows).sort_values(
        ["status", "score", "signals"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    selected_all = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    best = rules.iloc[0].to_dict() if not rules.empty else {}
    summary = {
        "rows": int(len(signals)),
        "rules_tested": int(len(rules)),
        "candidates": int(rules["status"].eq("candidate").sum()) if not rules.empty else 0,
        "best_status": str(best.get("status", "not_ready")),
        "best_signals": int(best.get("signals", 0) or 0),
        "best_avg_profit_per_share": float(best.get("avg_profit_per_share", 0.0) or 0.0),
        "best_avg_clv_cents": float(best.get("avg_clv_cents", 0.0) or 0.0),
        "best_positive_clv_rate": float(best.get("positive_clv_rate", 0.0) or 0.0),
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "note": "Guardrails use only prior side/price-zone history. They are meant to test whether overcalibration can be blocked without hindsight.",
    }
    return rules, selected_all, summary
