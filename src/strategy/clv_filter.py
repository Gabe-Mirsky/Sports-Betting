"""CLV-aware filters for calibrated single-game signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


DEFAULT_SIDE_RULES = {
    "YES": {
        "min_history_rows": 50,
        "min_avg_clv_cents": 0.0,
        "min_positive_clv_rate": 0.40,
        "min_avg_profit_per_share": 0.0,
    },
    "NO": {
        "min_history_rows": 50,
        "min_avg_clv_cents": 0.25,
        "min_positive_clv_rate": 0.50,
        "min_avg_profit_per_share": 0.0,
    },
}


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


def _rules(side_rules: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, float]]:
    merged = {side: values.copy() for side, values in DEFAULT_SIDE_RULES.items()}
    for side, overrides in (side_rules or {}).items():
        key = str(side).upper()
        if key not in merged:
            continue
        merged[key].update({name: float(value) for name, value in overrides.items()})
    return merged


def _history_metrics(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            "history_rows": 0,
            "history_avg_clv_cents": np.nan,
            "history_positive_clv_rate": np.nan,
            "history_avg_profit_per_share": np.nan,
        }
    clv_values = pd.to_numeric(pd.Series([row["clv_cents"] for row in history]), errors="coerce")
    profit_values = pd.to_numeric(pd.Series([row["profit"] for row in history]), errors="coerce")
    return {
        "history_rows": int(clv_values.notna().sum()),
        "history_avg_clv_cents": float(clv_values.mean()) if clv_values.notna().any() else np.nan,
        "history_positive_clv_rate": float((clv_values.dropna() > 0).mean()) if clv_values.notna().any() else np.nan,
        "history_avg_profit_per_share": float(profit_values.mean()) if profit_values.notna().any() else np.nan,
    }


def _passes_side_rule(metrics: dict[str, float], rule: dict[str, float]) -> tuple[bool, str]:
    if int(metrics["history_rows"]) < int(rule["min_history_rows"]):
        return False, "insufficient_side_clv_history"
    if float(metrics["history_avg_clv_cents"]) < float(rule["min_avg_clv_cents"]):
        return False, "side_average_clv_below_threshold"
    if float(metrics["history_positive_clv_rate"]) < float(rule["min_positive_clv_rate"]):
        return False, "side_positive_clv_rate_below_threshold"
    if float(metrics["history_avg_profit_per_share"]) < float(rule["min_avg_profit_per_share"]):
        return False, "side_average_profit_below_threshold"
    return True, "clv_filter_passed"


def add_expanding_clv_filter(
    rows: pd.DataFrame,
    signal_column: str = "calibrated_trade",
    side_rules: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add an expanding side-specific CLV gate to calibrated signal rows."""

    if rows.empty:
        return pd.DataFrame(), pd.DataFrame(), {"rows": 0, "clv_filtered_trades": 0}
    required = ["date", signal_column, "clv_cents", "realized_profit_per_share"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"CLV filter rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["date", "market_ticker"]).reset_index(drop=True)
    frame["clv_filter_side"] = _side_series(frame)
    frame["_base_signal"] = _coerce_bool(frame[signal_column])
    frame["clv_cents"] = pd.to_numeric(frame["clv_cents"], errors="coerce")
    frame["realized_profit_per_share"] = pd.to_numeric(frame["realized_profit_per_share"], errors="coerce")
    rules = _rules(side_rules)
    history: dict[str, list[dict[str, float]]] = {"YES": [], "NO": []}
    output_rows: list[dict[str, Any]] = []

    for _, slate in frame.groupby(frame["date"].dt.date, sort=True):
        slate_outputs: list[dict[str, Any]] = []
        for _, row in slate.iterrows():
            side = str(row["clv_filter_side"])
            metrics = _history_metrics(history.get(side, []))
            passes, reason = _passes_side_rule(metrics, rules[side])
            clv_filtered_trade = bool(row["_base_signal"] and passes)
            output = row.to_dict()
            output.update(
                {
                    "clv_filter_side": side,
                    "side_clv_history_rows": metrics["history_rows"],
                    "side_history_avg_clv_cents": metrics["history_avg_clv_cents"],
                    "side_history_positive_clv_rate": metrics["history_positive_clv_rate"],
                    "side_history_avg_profit_per_share": metrics["history_avg_profit_per_share"],
                    "clv_filtered_trade": clv_filtered_trade,
                    "clv_filter_reason": reason if bool(row["_base_signal"]) else "base_signal_false",
                }
            )
            slate_outputs.append(output)
        output_rows.extend(slate_outputs)

        for _, row in slate.iterrows():
            if not bool(row["_base_signal"]):
                continue
            side = str(row["clv_filter_side"])
            if pd.isna(row["clv_cents"]) or pd.isna(row["realized_profit_per_share"]):
                continue
            history.setdefault(side, []).append(
                {
                    "clv_cents": float(row["clv_cents"]),
                    "profit": float(row["realized_profit_per_share"]),
                }
            )

    output = pd.DataFrame(output_rows)
    selected = output[output["clv_filtered_trade"]].copy()
    side_audit_rows: list[dict[str, Any]] = []
    for side in ["YES", "NO"]:
        side_rows = output[output["clv_filter_side"].eq(side)]
        side_selected = selected[selected["clv_filter_side"].eq(side)]
        side_base = side_rows[side_rows["_base_signal"]]
        side_audit_rows.append(
            {
                "side": side,
                "base_signals": int(len(side_base)),
                "clv_filtered_trades": int(len(side_selected)),
                "selected_avg_clv_cents": float(side_selected["clv_cents"].mean()) if len(side_selected) else 0.0,
                "selected_positive_clv_rate": float((side_selected["clv_cents"] > 0).mean()) if len(side_selected) else 0.0,
                "selected_avg_profit_per_share": float(side_selected["realized_profit_per_share"].mean())
                if len(side_selected)
                else 0.0,
                "min_history_rows": int(rules[side]["min_history_rows"]),
                "min_avg_clv_cents": float(rules[side]["min_avg_clv_cents"]),
                "min_positive_clv_rate": float(rules[side]["min_positive_clv_rate"]),
                "min_avg_profit_per_share": float(rules[side]["min_avg_profit_per_share"]),
            }
        )
    side_audit = pd.DataFrame(side_audit_rows)
    dates = pd.to_datetime(selected["date"], errors="coerce").dropna() if not selected.empty else pd.Series(dtype="datetime64[ns]")
    summary = {
        "rows": int(len(output)),
        "base_signals": int(output["_base_signal"].sum()),
        "clv_filtered_trades": int(len(selected)),
        "yes_clv_filtered_trades": int(selected["clv_filter_side"].eq("YES").sum()) if len(selected) else 0,
        "no_clv_filtered_trades": int(selected["clv_filter_side"].eq("NO").sum()) if len(selected) else 0,
        "trade_start_date": dates.min().date().isoformat() if not dates.empty else None,
        "trade_end_date": dates.max().date().isoformat() if not dates.empty else None,
        "avg_clv_cents": float(selected["clv_cents"].mean()) if len(selected) else 0.0,
        "positive_clv_rate": float((selected["clv_cents"] > 0).mean()) if len(selected) else 0.0,
        "avg_profit_per_share": float(selected["realized_profit_per_share"].mean()) if len(selected) else 0.0,
        "side_rules": rules,
        "note": "The CLV gate uses only prior dates by side; same-day and future CLV are not used.",
    }
    if summary["trade_start_date"] and summary["trade_end_date"]:
        start = summary["trade_start_date"]
        end = summary["trade_end_date"]
        summary["trade_timeline"] = start if start == end else f"{start} to {end}"
    else:
        summary["trade_timeline"] = "n/a"
    return output.drop(columns=["_base_signal"]), side_audit, summary


def save_clv_filter_outputs(
    filtered: pd.DataFrame,
    side_audit: pd.DataFrame,
    summary: dict[str, Any],
    filtered_path: str | Path,
    side_audit_path: str | Path,
    summary_path: str | Path,
) -> None:
    filtered_output = Path(filtered_path)
    audit_output = Path(side_audit_path)
    summary_output = Path(summary_path)
    filtered_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_output, index=False)
    side_audit.to_csv(audit_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
