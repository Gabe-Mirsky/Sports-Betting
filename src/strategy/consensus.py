"""Consensus filters that compare raw model calibration with market-blend calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.edge_calibration import _coerce_bool


CONSENSUS_KEYS = ["date", "game_id", "market_ticker"]


def _timeline(values: pd.Series) -> tuple[str | None, str | None, str]:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None, None, "n/a"
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    return start, end, start if start == end else f"{start} to {end}"


def build_consensus_calibrated_edges(
    raw_calibrated: pd.DataFrame,
    market_blend_calibrated: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require both raw and market-blend calibrations to agree on a signal."""

    if raw_calibrated.empty:
        return pd.DataFrame(), {"rows": 0}
    missing_raw = [column for column in [*CONSENSUS_KEYS, "calibrated_trade"] if column not in raw_calibrated.columns]
    missing_blend = [
        column for column in [*CONSENSUS_KEYS, "calibrated_trade"] if column not in market_blend_calibrated.columns
    ]
    if missing_raw:
        raise ValueError(f"Raw calibrated rows are missing consensus columns: {missing_raw}")
    if missing_blend:
        raise ValueError(f"Market-blend calibrated rows are missing consensus columns: {missing_blend}")

    raw = raw_calibrated.copy()
    blend = market_blend_calibrated.copy()
    raw["raw_calibrated_trade"] = _coerce_bool(raw["calibrated_trade"])
    blend["blend_calibrated_trade"] = _coerce_bool(blend["calibrated_trade"])
    blend_columns = [
        *CONSENSUS_KEYS,
        "blend_calibrated_trade",
        "calibrated_expected_roi",
        "calibrated_expected_profit_per_share",
        "calibrated_yes_rate",
        "edge_bin_history_rows",
        "edge_global_history_rows",
        "edge",
        "model_yes_prob",
        "calibration_reason",
    ]
    blend_columns = [column for column in blend_columns if column in blend.columns]
    consensus = raw.merge(
        blend[blend_columns],
        on=CONSENSUS_KEYS,
        how="left",
        suffixes=("", "_blend"),
    )
    consensus["blend_calibrated_trade"] = consensus["blend_calibrated_trade"].fillna(False)
    consensus["consensus_trade"] = consensus["raw_calibrated_trade"] & consensus["blend_calibrated_trade"]
    for column in ["calibrated_expected_roi", "calibrated_expected_roi_blend"]:
        if column in consensus.columns:
            consensus[column] = pd.to_numeric(consensus[column], errors="coerce")
    roi_columns = [
        column for column in ["calibrated_expected_roi", "calibrated_expected_roi_blend"] if column in consensus.columns
    ]
    consensus["consensus_expected_roi"] = consensus[roi_columns].min(axis=1) if roi_columns else 0.0
    if "calibrated_expected_profit_per_share_blend" in consensus.columns:
        consensus["calibrated_expected_profit_per_share_blend"] = pd.to_numeric(
            consensus["calibrated_expected_profit_per_share_blend"],
            errors="coerce",
        )
    if "calibrated_expected_profit_per_share" in consensus.columns:
        consensus["calibrated_expected_profit_per_share"] = pd.to_numeric(
            consensus["calibrated_expected_profit_per_share"],
            errors="coerce",
        )
    profit_columns = [
        column
        for column in ["calibrated_expected_profit_per_share", "calibrated_expected_profit_per_share_blend"]
        if column in consensus.columns
    ]
    consensus["consensus_expected_profit_per_share"] = (
        consensus[profit_columns].min(axis=1) if profit_columns else 0.0
    )

    selected = consensus[consensus["consensus_trade"]].copy()
    start, end, timeline = _timeline(selected["date"]) if not selected.empty else (None, None, "n/a")
    if "actual_yes_win" in selected.columns:
        actual_yes = _coerce_bool(selected["actual_yes_win"])
    else:
        actual_yes = pd.Series(dtype=bool)
    realized_profit = (
        pd.to_numeric(selected["realized_profit_per_share"], errors="coerce")
        if "realized_profit_per_share" in selected.columns
        else pd.Series(dtype=float)
    )
    summary = {
        "rows": int(len(consensus)),
        "raw_calibrated_trades": int(consensus["raw_calibrated_trade"].sum()),
        "market_blend_calibrated_trades": int(consensus["blend_calibrated_trade"].sum()),
        "consensus_trades": int(consensus["consensus_trade"].sum()),
        "trade_start_date": start,
        "trade_end_date": end,
        "trade_timeline": timeline,
        "win_rate": float(actual_yes.mean()) if len(actual_yes) else 0.0,
        "avg_realized_profit_per_share": float(realized_profit.mean()) if len(realized_profit) else 0.0,
        "avg_consensus_expected_roi": float(selected["consensus_expected_roi"].mean()) if len(selected) else 0.0,
        "note": "Consensus requires both raw-model edge calibration and market-blend edge calibration to agree.",
    }
    return consensus.reset_index(drop=True), summary


def save_consensus_outputs(
    consensus: pd.DataFrame,
    summary: dict[str, Any],
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    output = Path(output_path)
    summary_output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
