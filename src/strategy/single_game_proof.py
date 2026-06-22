"""Single-game edge proof gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(name: str, value: Any, threshold: Any, passed: bool, severity: str, details: str) -> dict[str, Any]:
    return {
        "gate": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
        "severity": severity,
        "details": details,
    }


def _true_rows(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    mask = frame[column].astype(str).str.lower().isin({"true", "1", "yes"})
    return frame[mask].copy()


def _max_profit_share(frame: pd.DataFrame, group_column: str) -> float:
    if frame.empty or group_column not in frame.columns or "realized_profit_per_share" not in frame.columns:
        return 0.0
    working = frame.copy()
    working["realized_profit_per_share"] = pd.to_numeric(working["realized_profit_per_share"], errors="coerce")
    profits = working.groupby(group_column, dropna=False)["realized_profit_per_share"].sum()
    total_positive = profits[profits > 0].sum()
    if total_positive <= 0:
        return 1.0
    return float(profits.max() / total_positive)


def _price_bucket_share(frame: pd.DataFrame) -> float:
    if frame.empty or "price_cents" not in frame.columns:
        return 0.0
    working = frame.copy()
    working["price_bucket"] = pd.cut(
        pd.to_numeric(working["price_cents"], errors="coerce"),
        bins=[0, 25, 40, 55, 70, 85, 100],
        include_lowest=True,
    ).astype(str)
    return _max_profit_share(working, "price_bucket")


def build_single_game_proof_report(
    market_truth_summary: dict[str, Any],
    backtest_summary: dict[str, Any],
    clv_summary: dict[str, Any],
    readiness_summary: dict[str, Any],
    readiness: pd.DataFrame,
    calibrated_trades: pd.DataFrame,
    strategy_name: str = "raw_calibrated",
    min_matched_markets: int = 300,
    min_months: int = 6,
    max_profit_share: float = 0.60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build explicit gates before treating single-game bets as proven."""

    gates: list[dict[str, Any]] = []
    matched = int(market_truth_summary.get("matched_game_markets", 0) or 0)
    usable = market_truth_summary.get("usable_price_counts", {}) or {}
    usable_60m = int(usable.get("pregame_60m", 0) or 0)
    usable_30m = int(usable.get("pregame_30m", 0) or 0)
    usable_5m = int(usable.get("pregame_5m", 0) or 0)
    mismatch_rate = (
        float(market_truth_summary.get("ticker_mapping_mismatch_count", 0) or 0) / matched
        if matched
        else 1.0
    )
    wide_spread_rate = float(market_truth_summary.get("wide_spread_count", 0) or 0) / matched if matched else 1.0
    low_liquidity_rate = float(market_truth_summary.get("low_liquidity_count", 0) or 0) / matched if matched else 1.0
    starting_bankroll = float(backtest_summary.get("starting_bankroll", 0.0) or 0.0)
    ending_bankroll = float(backtest_summary.get("ending_bankroll", 0.0) or 0.0)
    avg_clv = float(clv_summary.get("avg_clv_cents", 0.0) or 0.0)
    positive_clv = float(clv_summary.get("positive_clv_rate", 0.0) or 0.0)

    gates.append(
        _gate(
            "historical_market_coverage",
            matched,
            f">= {min_matched_markets}",
            matched >= min_matched_markets,
            "hard",
            "At least 300 matched historical markets are required before edge claims matter.",
        )
    )
    gates.append(
        _gate(
            "usable_pregame_prices",
            min(usable_60m, usable_30m, usable_5m),
            f">= {min_matched_markets} at 60m, 30m, and 5m",
            min(usable_60m, usable_30m, usable_5m) >= min_matched_markets,
            "hard",
            "Backtests need repeated tradable pregame snapshots, not last-price-only rows.",
        )
    )
    gates.append(
        _gate(
            "ticker_team_mapping_quality",
            round(mismatch_rate, 4),
            "<= 0.01",
            mismatch_rate <= 0.01,
            "hard",
            "Wrong team mapping can manufacture fake edge.",
        )
    )
    gates.append(
        _gate(
            "spread_quality",
            round(wide_spread_rate, 4),
            "<= 0.05",
            wide_spread_rate <= 0.05,
            "hard",
            "Wide spreads make historical fills unrealistic.",
        )
    )
    gates.append(
        _gate(
            "liquidity_quality",
            round(low_liquidity_rate, 4),
            "<= 0.25",
            low_liquidity_rate <= 0.25,
            "hard",
            "Low-liquidity markets need stricter sizing and filtering.",
        )
    )
    gates.append(
        _gate(
            "strategy_backtest_profit",
            round(ending_bankroll - starting_bankroll, 2),
            "> 0",
            ending_bankroll > starting_bankroll,
            "hard",
            "The strategy under test must survive realistic bid/ask entry prices.",
        )
    )
    gates.append(
        _gate(
            "average_clv",
            round(avg_clv, 4),
            "> 0",
            avg_clv > 0,
            "hard",
            "A real edge should beat later pregame prices on average.",
        )
    )
    gates.append(
        _gate(
            "positive_clv_rate",
            round(positive_clv, 4),
            "> 0.50",
            positive_clv > 0.50,
            "hard",
            "A repeatable signal should show positive CLV more often than not.",
        )
    )

    raw_readiness = pd.DataFrame()
    if not readiness.empty and "strategy" in readiness.columns:
        raw_readiness = readiness[readiness["strategy"].eq(strategy_name)].copy()
    readiness_status = str(raw_readiness["status"].iloc[0]) if not raw_readiness.empty else "missing"
    readiness_months = int(raw_readiness["months"].iloc[0]) if not raw_readiness.empty else 0
    gates.append(
        _gate(
            "calibrated_strategy_readiness",
            readiness_status,
            "paper_trade_candidate",
            readiness_status == "paper_trade_candidate",
            "hard",
            "Calibrated signals must pass stability and drawdown readiness gates.",
        )
    )
    gates.append(
        _gate(
            "repeatability_months",
            readiness_months,
            f">= {min_months}",
            readiness_months >= min_months,
            "hard",
            "A single partial season is not enough to prove repeatability.",
        )
    )

    if strategy_name == "defensive_clv_filtered":
        signal_column = "defensive_trade"
    elif strategy_name == "clv_filtered_calibrated":
        signal_column = "clv_filtered_trade"
    else:
        signal_column = "calibrated_trade"
    selected = _true_rows(calibrated_trades, signal_column)
    season_share = _max_profit_share(selected, "season")
    team_share = _max_profit_share(selected, "yes_team_abbr")
    price_share = _price_bucket_share(selected)
    gates.append(
        _gate(
            "season_profit_concentration",
            round(season_share, 4),
            f"<= {max_profit_share}",
            season_share <= max_profit_share,
            "hard",
            "No single season should explain most positive calibrated profit.",
        )
    )
    gates.append(
        _gate(
            "team_profit_concentration",
            round(team_share, 4),
            f"<= {max_profit_share}",
            team_share <= max_profit_share,
            "warning",
            "No one team should explain most positive calibrated profit.",
        )
    )
    gates.append(
        _gate(
            "price_bucket_profit_concentration",
            round(price_share, 4),
            f"<= {max_profit_share}",
            price_share <= max_profit_share,
            "warning",
            "No one price range should explain most positive calibrated profit.",
        )
    )

    gate_frame = pd.DataFrame(gates)
    hard_failures = gate_frame[gate_frame["severity"].eq("hard") & ~gate_frame["passed"]]
    warning_failures = gate_frame[gate_frame["severity"].eq("warning") & ~gate_frame["passed"]]
    status = "single_game_edge_proven" if hard_failures.empty else "not_proven"
    summary = {
        "status": status,
        "strategy_under_test": strategy_name,
        "single_game_edge_proven": bool(status == "single_game_edge_proven"),
        "parlay_research_allowed": bool(status == "single_game_edge_proven" and warning_failures.empty),
        "hard_failures": int(len(hard_failures)),
        "warning_failures": int(len(warning_failures)),
        "failed_gates": hard_failures["gate"].tolist(),
        "warning_gates": warning_failures["gate"].tolist(),
        "next_goal": "Prove real, repeatable edge on single-game bets before revisiting parlays.",
        "recommendation": (
            "Continue single-game calibration, CLV, and stability work. Do not optimize parlays."
            if status != "single_game_edge_proven"
            else "Single-game gates passed; next work can evaluate controlled paper trading."
        ),
    }
    return gate_frame, summary


def build_single_game_proof_report_from_files(reports_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    reports = Path(reports_dir)
    readiness_path = reports / "strategy_readiness.csv"
    readiness = pd.read_csv(readiness_path) if readiness_path.exists() else pd.DataFrame()
    strategy_name = "raw_calibrated"
    clv_summary_path = reports / "clv_summary.json"
    portfolio_summary_path = reports / "backtest_summary.json"
    trades_path = reports / "edge_calibrated_trades.csv"
    if not readiness.empty and readiness["strategy"].eq("clv_filtered_calibrated").any():
        strategy_name = "clv_filtered_calibrated"
        clv_summary_path = reports / "clv_filtered_summary.json"
        portfolio_summary_path = reports / "portfolio_summary_clv_filtered.json"
        trades_path = reports / "clv_filtered_trades.csv"
    if not readiness.empty and readiness["strategy"].eq("defensive_clv_filtered").any():
        strategy_name = "defensive_clv_filtered"
        clv_summary_path = reports / "defensive_filter_summary.json"
        portfolio_summary_path = reports / "portfolio_summary_defensive.json"
        trades_path = reports / "defensive_filtered_trades.csv"
    calibrated = pd.read_csv(trades_path, dtype={"game_id": str}) if trades_path.exists() else pd.DataFrame()
    return build_single_game_proof_report(
        market_truth_summary=_read_json(reports / "market_truth_audit_summary.json"),
        backtest_summary=_read_json(portfolio_summary_path),
        clv_summary=_read_json(clv_summary_path),
        readiness_summary=_read_json(reports / "strategy_readiness_summary.json"),
        readiness=readiness,
        calibrated_trades=calibrated,
        strategy_name=strategy_name,
    )


def save_single_game_proof_report(
    gates: pd.DataFrame,
    summary: dict[str, Any],
    gates_path: str | Path,
    summary_path: str | Path,
) -> None:
    gates_output = Path(gates_path)
    summary_output = Path(summary_path)
    gates_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    gates.to_csv(gates_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
