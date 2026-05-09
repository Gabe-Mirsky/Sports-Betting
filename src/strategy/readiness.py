"""Strategy readiness scoring before any parlay research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.stability import summarize_signal_stability


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check(value: float, threshold: float, direction: str = "gte") -> bool:
    return value >= threshold if direction == "gte" else value <= threshold


def evaluate_strategy_readiness(
    name: str,
    monthly: pd.DataFrame,
    stability_summary: dict[str, Any],
    portfolio_summary: dict[str, Any],
    min_signals: int = 100,
    min_months: int = 6,
    min_positive_month_share: float = 0.60,
    min_avg_profit_per_share: float = 0.0,
    min_ending_bankroll: float = 100.0,
    max_drawdown_floor: float = -0.60,
) -> dict[str, Any]:
    """Return one readiness row for a strategy."""

    signals = int(stability_summary.get("signals", 0) or 0)
    months = int(stability_summary.get("months", 0) or 0)
    positive_month_share = float(stability_summary.get("positive_month_share", 0.0) or 0.0)
    avg_profit = float(stability_summary.get("overall_avg_profit_per_share", 0.0) or 0.0)
    ending_bankroll = float(portfolio_summary.get("ending_bankroll", 0.0) or 0.0)
    max_drawdown = float(portfolio_summary.get("max_drawdown", 0.0) or 0.0)
    trade_timeline = str(portfolio_summary.get("trade_timeline") or stability_summary.get("timeline") or "n/a")
    failed: list[str] = []
    if not _check(signals, min_signals):
        failed.append("too_few_signals")
    if not _check(months, min_months):
        failed.append("too_few_months")
    if not _check(positive_month_share, min_positive_month_share):
        failed.append("unstable_monthly_profit")
    if not _check(avg_profit, min_avg_profit_per_share):
        failed.append("negative_average_signal_profit")
    if not _check(ending_bankroll, min_ending_bankroll):
        failed.append("portfolio_lost_money")
    if not _check(max_drawdown, max_drawdown_floor):
        failed.append("drawdown_too_large")

    hard_failures = {"negative_average_signal_profit", "portfolio_lost_money"}
    if hard_failures.intersection(failed):
        status = "not_ready"
    elif failed:
        status = "watchlist"
    else:
        status = "paper_trade_candidate"

    parlay_ready = bool(
        status == "paper_trade_candidate"
        and months >= 9
        and positive_month_share >= 0.75
        and max_drawdown >= -0.25
    )
    if parlay_ready:
        recommendation = "Eligible for future parlay-correlation research, still paper only."
    elif status == "paper_trade_candidate":
        recommendation = "Candidate for individual paper bets only; keep excluding parlays."
    elif status == "watchlist":
        recommendation = "Watchlist only; improve stability before raising size or combining bets."
    else:
        recommendation = "Do not use for new paper-trade selection without changing the rule."

    return {
        "strategy": name,
        "status": status,
        "parlay_ready": parlay_ready,
        "signals": signals,
        "months": months,
        "positive_month_share": positive_month_share,
        "avg_signal_profit_per_share": avg_profit,
        "ending_bankroll": ending_bankroll,
        "total_return_pct": float(portfolio_summary.get("total_return_pct", 0.0) or 0.0),
        "max_drawdown": max_drawdown,
        "trade_timeline": trade_timeline,
        "failed_checks": ",".join(failed) if failed else "",
        "recommendation": recommendation,
    }


def build_strategy_readiness_report(
    specs: list[dict[str, Any]],
    min_signals: int = 100,
    min_months: int = 6,
    min_positive_month_share: float = 0.60,
    min_avg_profit_per_share: float = 0.0,
    min_ending_bankroll: float = 100.0,
    max_drawdown_floor: float = -0.60,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build readiness and monthly stability tables from strategy specs."""

    readiness_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for spec in specs:
        input_path = Path(spec["input_path"])
        if not input_path.exists():
            continue
        rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
        monthly, stability_summary = summarize_signal_stability(
            rows,
            signal_column=str(spec["signal_column"]),
            expected_roi_column=spec.get("expected_roi_column"),
        )
        if not monthly.empty:
            monthly = monthly.copy()
            monthly.insert(0, "strategy", spec["name"])
            monthly_frames.append(monthly)
        portfolio_summary = _read_json(Path(spec["portfolio_summary_path"]))
        readiness_rows.append(
            evaluate_strategy_readiness(
                name=str(spec["name"]),
                monthly=monthly,
                stability_summary=stability_summary,
                portfolio_summary=portfolio_summary,
                min_signals=min_signals,
                min_months=min_months,
                min_positive_month_share=min_positive_month_share,
                min_avg_profit_per_share=min_avg_profit_per_share,
                min_ending_bankroll=min_ending_bankroll,
                max_drawdown_floor=max_drawdown_floor,
            )
        )

    readiness = pd.DataFrame(readiness_rows)
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    summary = {
        "strategies_evaluated": int(len(readiness)),
        "paper_trade_candidates": int(readiness["status"].eq("paper_trade_candidate").sum())
        if not readiness.empty
        else 0,
        "watchlist": int(readiness["status"].eq("watchlist").sum()) if not readiness.empty else 0,
        "not_ready": int(readiness["status"].eq("not_ready").sum()) if not readiness.empty else 0,
        "parlay_ready": int(readiness["parlay_ready"].sum()) if not readiness.empty else 0,
        "thresholds": {
            "min_signals": int(min_signals),
            "min_months": int(min_months),
            "min_positive_month_share": float(min_positive_month_share),
            "min_avg_profit_per_share": float(min_avg_profit_per_share),
            "min_ending_bankroll": float(min_ending_bankroll),
            "max_drawdown_floor": float(max_drawdown_floor),
        },
        "note": "Readiness is for paper-trading research. Parlay-ready should remain zero until correlation modeling exists.",
    }
    return readiness, monthly_all, summary


def save_strategy_readiness_outputs(
    readiness: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: dict[str, Any],
    readiness_path: str | Path,
    monthly_path: str | Path,
    summary_path: str | Path,
) -> None:
    readiness_output = Path(readiness_path)
    monthly_output = Path(monthly_path)
    summary_output = Path(summary_path)
    readiness_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(readiness_output, index=False)
    monthly.to_csv(monthly_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
