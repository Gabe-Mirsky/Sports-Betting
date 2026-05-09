"""Research-only parlay correlation diagnostics."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


def _timeline(rows: pd.DataFrame, date_column: str = "date") -> str:
    if rows.empty or date_column not in rows.columns:
        return "n/a"
    dates = pd.to_datetime(rows[date_column], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    return start if start == end else f"{start} to {end}"


def _probability_estimate(frame: pd.DataFrame) -> pd.Series:
    if {"market_prob", "consensus_expected_profit_per_share"}.issubset(frame.columns):
        return (
            pd.to_numeric(frame["market_prob"], errors="coerce")
            + pd.to_numeric(frame["consensus_expected_profit_per_share"], errors="coerce")
        ).clip(0.01, 0.99)
    if {"contract_cost", "consensus_expected_profit_per_share"}.issubset(frame.columns):
        return (
            pd.to_numeric(frame["contract_cost"], errors="coerce")
            + pd.to_numeric(frame["consensus_expected_profit_per_share"], errors="coerce")
        ).clip(0.01, 0.99)
    if "model_yes_prob" in frame.columns:
        return pd.to_numeric(frame["model_yes_prob"], errors="coerce").clip(0.01, 0.99)
    raise ValueError("Parlay research needs a probability column or consensus expected profit column.")


def _price_bucket(market_prob: float) -> str:
    if market_prob >= 0.60:
        return "heavy_favorite"
    if market_prob >= 0.50:
        return "favorite"
    if market_prob >= 0.35:
        return "underdog"
    return "longshot"


def _combo(value_a: str, value_b: str) -> str:
    return "_".join(sorted([str(value_a), str(value_b)]))


def build_parlay_pair_frame(
    rows: pd.DataFrame,
    signal_column: str = "consensus_trade",
    exclude_same_game: bool = True,
) -> pd.DataFrame:
    """Create same-slate two-leg pair rows from selected historical signals."""

    if rows.empty:
        return pd.DataFrame()
    required = ["date", "game_id", signal_column, "actual_yes_win", "yes_team_abbr", "home_team_abbr", "away_team_abbr"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Parlay research rows are missing columns: {missing}")

    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["signal"] = _coerce_bool(frame[signal_column])
    frame["actual_yes_win"] = _coerce_bool(frame["actual_yes_win"])
    if "market_prob" in frame.columns:
        frame["market_prob"] = pd.to_numeric(frame["market_prob"], errors="coerce")
    elif "price_cents" in frame.columns:
        frame["market_prob"] = pd.to_numeric(frame["price_cents"], errors="coerce") / 100.0
    elif "contract_cost" in frame.columns:
        frame["market_prob"] = pd.to_numeric(frame["contract_cost"], errors="coerce")
    else:
        raise ValueError("Parlay research rows need market_prob, price_cents, or contract_cost.")
    frame["estimated_yes_prob"] = _probability_estimate(frame)
    frame = frame.dropna(subset=["market_prob", "estimated_yes_prob"]).copy()
    frame = frame[frame["signal"]].sort_values(["date", "game_id", "market_ticker" if "market_ticker" in frame.columns else "game_id"]).copy()
    if frame.empty:
        return pd.DataFrame()

    pair_rows: list[dict[str, Any]] = []
    for slate_date, slate in frame.groupby(frame["date"].dt.date, sort=True):
        records = slate.to_dict(orient="records")
        for first, second in combinations(records, 2):
            if exclude_same_game and str(first["game_id"]) == str(second["game_id"]):
                continue
            first_is_home = str(first["yes_team_abbr"]) == str(first["home_team_abbr"])
            second_is_home = str(second["yes_team_abbr"]) == str(second["home_team_abbr"])
            first_price_bucket = _price_bucket(float(first["market_prob"]))
            second_price_bucket = _price_bucket(float(second["market_prob"]))
            pair_market_prob = float(first["market_prob"]) * float(second["market_prob"])
            pair_estimated_prob = float(first["estimated_yes_prob"]) * float(second["estimated_yes_prob"])
            pair_win = bool(first["actual_yes_win"] and second["actual_yes_win"])
            pair_rows.append(
                {
                    "date": slate_date.isoformat(),
                    "game_id_1": str(first["game_id"]),
                    "game_id_2": str(second["game_id"]),
                    "market_ticker_1": first.get("market_ticker"),
                    "market_ticker_2": second.get("market_ticker"),
                    "yes_team_abbr_1": first["yes_team_abbr"],
                    "yes_team_abbr_2": second["yes_team_abbr"],
                    "home_team_abbr_1": first["home_team_abbr"],
                    "home_team_abbr_2": second["home_team_abbr"],
                    "away_team_abbr_1": first["away_team_abbr"],
                    "away_team_abbr_2": second["away_team_abbr"],
                    "yes_is_home_1": first_is_home,
                    "yes_is_home_2": second_is_home,
                    "side_combo": _combo("home" if first_is_home else "away", "home" if second_is_home else "away"),
                    "price_bucket_1": first_price_bucket,
                    "price_bucket_2": second_price_bucket,
                    "price_bucket_combo": _combo(first_price_bucket, second_price_bucket),
                    "market_prob_1": float(first["market_prob"]),
                    "market_prob_2": float(second["market_prob"]),
                    "estimated_yes_prob_1": float(first["estimated_yes_prob"]),
                    "estimated_yes_prob_2": float(second["estimated_yes_prob"]),
                    "actual_yes_win_1": bool(first["actual_yes_win"]),
                    "actual_yes_win_2": bool(second["actual_yes_win"]),
                    "pair_win": pair_win,
                    "market_pair_prob_independent": pair_market_prob,
                    "estimated_pair_prob_independent": pair_estimated_prob,
                    "pair_edge_independent": pair_estimated_prob - pair_market_prob,
                    "synthetic_independence_profit_per_dollar": (1.0 / pair_market_prob - 1.0)
                    if pair_win and pair_market_prob > 0
                    else -1.0,
                }
            )
    return pd.DataFrame(pair_rows)


def _correlation(pair_rows: pd.DataFrame) -> float:
    if pair_rows.empty:
        return 0.0
    left = pair_rows["actual_yes_win_1"].astype(float)
    right = pair_rows["actual_yes_win_2"].astype(float)
    if left.nunique() < 2 or right.nunique() < 2:
        return 0.0
    value = float(left.corr(right))
    return 0.0 if np.isnan(value) else value


def summarize_parlay_pairs(pair_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize pair outcomes overall and by simple dependency buckets."""

    if pair_rows.empty:
        return pd.DataFrame(), {
            "pair_rows": 0,
            "parlay_ready": False,
            "status": "blocked_no_pairs",
            "note": "No same-slate signal pairs were available.",
        }

    frame = pair_rows.copy()
    frame["pair_win"] = _coerce_bool(frame["pair_win"])
    for column in [
        "market_pair_prob_independent",
        "estimated_pair_prob_independent",
        "pair_edge_independent",
        "synthetic_independence_profit_per_dollar",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    def summarize_group(group_name: str, group_value: str, group: pd.DataFrame) -> dict[str, Any]:
        corr = _correlation(group)
        return {
            "group": group_name,
            "group_value": group_value,
            "pairs": int(len(group)),
            "slates": int(group["date"].nunique()),
            "pair_win_rate": float(group["pair_win"].mean()),
            "avg_market_pair_prob_independent": float(group["market_pair_prob_independent"].mean()),
            "avg_estimated_pair_prob_independent": float(group["estimated_pair_prob_independent"].mean()),
            "avg_pair_edge_independent": float(group["pair_edge_independent"].mean()),
            "avg_synthetic_profit_per_dollar": float(group["synthetic_independence_profit_per_dollar"].mean()),
            "leg_outcome_correlation": corr,
        }

    rows = [summarize_group("overall", "all_pairs", frame)]
    for column in ["side_combo", "price_bucket_combo"]:
        for value, group in frame.groupby(column, dropna=False):
            rows.append(summarize_group(column, str(value), group))
    report = pd.DataFrame(rows).sort_values(["group", "pairs"], ascending=[True, False]).reset_index(drop=True)

    overall = rows[0]
    status = "correlation_watchlist"
    failed: list[str] = []
    if overall["pairs"] < 100 or overall["slates"] < 30:
        failed.append("too_few_pair_observations")
    if overall["avg_synthetic_profit_per_dollar"] <= 0:
        failed.append("negative_synthetic_pair_profit")
    if abs(overall["leg_outcome_correlation"]) > 0.15:
        failed.append("correlation_too_high_for_naive_parlays")
    failed.append("pair_economics_not_validated_out_of_sample")
    if failed:
        status = "blocked_" + failed[0]
    summary = {
        "pair_rows": int(len(frame)),
        "timeline": _timeline(frame),
        "slates_with_pairs": int(frame["date"].nunique()),
        "pair_win_rate": overall["pair_win_rate"],
        "avg_market_pair_prob_independent": overall["avg_market_pair_prob_independent"],
        "avg_estimated_pair_prob_independent": overall["avg_estimated_pair_prob_independent"],
        "avg_pair_edge_independent": overall["avg_pair_edge_independent"],
        "avg_synthetic_profit_per_dollar": overall["avg_synthetic_profit_per_dollar"],
        "leg_outcome_correlation": overall["leg_outcome_correlation"],
        "pair_economics_validation": "in_sample_only",
        "status": status,
        "failed_checks": failed,
        "parlay_ready": False,
        "note": "Research-only two-leg diagnostics. Parlays remain blocked until individual strategy readiness is positive and pair economics are positive out-of-sample.",
    }
    return report, summary


def apply_strategy_readiness_gate(summary: dict[str, Any], readiness_summary: dict[str, Any]) -> dict[str, Any]:
    """Block parlay research when no individual strategy is parlay-ready."""

    output = dict(summary)
    parlay_ready_count = int(readiness_summary.get("parlay_ready", 0) or 0)
    output["strategy_parlay_ready_count"] = parlay_ready_count
    if parlay_ready_count <= 0:
        failures = list(output.get("failed_checks", []))
        if "no_parlay_ready_individual_strategy" not in failures:
            failures.insert(0, "no_parlay_ready_individual_strategy")
        output["failed_checks"] = failures
        output["status"] = "blocked_strategy_readiness"
        output["parlay_ready"] = False
    return output


def save_parlay_research_outputs(
    pair_rows: pd.DataFrame,
    report: pd.DataFrame,
    summary: dict[str, Any],
    pair_rows_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
) -> None:
    pair_output = Path(pair_rows_path)
    report_output = Path(report_path)
    summary_output = Path(summary_path)
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    pair_rows.to_csv(pair_output, index=False)
    report.to_csv(report_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
