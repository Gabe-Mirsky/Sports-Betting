"""Choose the default headline paper-trading result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


HEADLINE_CANDIDATES = [
    {
        "strategy": "consensus_calibrated",
        "label": "Consensus calibrated slate",
        "summary_file": "portfolio_summary_consensus_calibrated.json",
        "trades_file": "portfolio_trades_consensus_calibrated.csv",
        "slates_file": "portfolio_slates_consensus_calibrated.csv",
    },
    {
        "strategy": "market_blend_calibrated",
        "label": "Market-blend calibrated slate",
        "summary_file": "portfolio_summary_market_blend_calibrated.json",
        "trades_file": "portfolio_trades_market_blend_calibrated.csv",
        "slates_file": "portfolio_slates_market_blend_calibrated.csv",
    },
    {
        "strategy": "raw_calibrated",
        "label": "Raw calibrated slate",
        "summary_file": "portfolio_summary_calibrated.json",
        "trades_file": "portfolio_trades_calibrated.csv",
        "slates_file": "portfolio_slates_calibrated.csv",
    },
    {
        "strategy": "raw_edge",
        "label": "Raw edge slate",
        "summary_file": "portfolio_summary.json",
        "trades_file": "portfolio_trades.csv",
        "slates_file": "portfolio_slates.csv",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _readiness_by_strategy(readiness_path: Path) -> dict[str, dict[str, Any]]:
    readiness = _read_csv(readiness_path)
    if readiness.empty or "strategy" not in readiness.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for record in readiness.to_dict(orient="records"):
        rows[str(record.get("strategy", ""))] = record
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _candidate_summary(report_dir: Path, candidate: dict[str, str], readiness: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary_path = report_dir / candidate["summary_file"]
    summary = _read_json(summary_path)
    if not summary:
        return {}

    strategy_name = candidate["strategy"]
    readiness_row = readiness.get(strategy_name, {})
    parlay_ready = _as_bool(readiness_row.get("parlay_ready", False))
    trades_path = report_dir / candidate["trades_file"]
    slates_path = report_dir / candidate["slates_file"]
    num_trades = int(summary.get("num_selected_trades", summary.get("num_trades", 0)) or 0)
    return {
        "headline_strategy": strategy_name,
        "headline_label": candidate["label"],
        "settlement_mode": "slate_settled",
        "source_summary_file": candidate["summary_file"],
        "source_trades_file": candidate["trades_file"],
        "source_slates_file": candidate["slates_file"],
        "source_summary_path": str(summary_path),
        "source_trades_path": str(trades_path),
        "source_slates_path": str(slates_path),
        "starting_bankroll": float(summary.get("starting_bankroll", 100.0) or 100.0),
        "ending_bankroll": float(summary.get("ending_bankroll", summary.get("starting_bankroll", 100.0)) or 100.0),
        "total_return_pct": float(summary.get("total_return_pct", 0.0) or 0.0),
        "max_drawdown": float(summary.get("max_drawdown", 0.0) or 0.0),
        "roi_on_amount_risked": float(summary.get("roi_on_amount_risked", 0.0) or 0.0),
        "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
        "average_edge": float(summary.get("average_edge", 0.0) or 0.0),
        "num_selected_trades": num_trades,
        "num_slates": int(summary.get("num_slates", 0) or 0),
        "trade_timeline": str(summary.get("trade_timeline", "n/a") or "n/a"),
        "readiness_status": str(readiness_row.get("status", "unknown")),
        "readiness_failed_checks": str(readiness_row.get("failed_checks", "")),
        "parlay_ready": parlay_ready,
        "parlays_blocked": not parlay_ready,
    }


def build_headline_backtest_summary(report_dir: str | Path) -> dict[str, Any]:
    """Return the preferred slate-settled headline result from report artifacts."""

    report_path = Path(report_dir)
    readiness = _readiness_by_strategy(report_path / "strategy_readiness.csv")
    candidates: list[dict[str, Any]] = []
    for candidate in HEADLINE_CANDIDATES:
        row = _candidate_summary(report_path, candidate, readiness)
        if row:
            candidates.append(row)

    if not candidates:
        return {
            "status": "unavailable",
            "settlement_mode": "slate_settled",
            "headline_label": "No slate-settled result available",
            "headline_strategy": "",
            "num_selected_trades": 0,
            "trade_timeline": "n/a",
            "parlays_blocked": True,
            "note": "Run the portfolio optimization steps before treating any backtest as the headline result.",
        }

    headline = dict(candidates[0])
    headline["status"] = "available"
    headline["candidate_count"] = len(candidates)
    headline["candidate_strategies"] = [candidate["headline_strategy"] for candidate in candidates]
    headline["note"] = (
        "Headline result uses slate-settled individual paper bets. "
        "Parlays remain blocked unless individual readiness and out-of-sample pair economics both pass."
    )
    return headline


def save_headline_backtest_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
