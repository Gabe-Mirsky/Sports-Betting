"""Research sweeps for anchoring model probabilities to market prices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.backtest import run_backtest, summarize_backtest


def _monthly_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    traded = trades[trades["trade"]].copy() if "trade" in trades.columns else pd.DataFrame()
    if traded.empty:
        return {"trade_months": 0, "positive_profit_month_share": 0.0, "positive_clv_month_share": 0.0}
    traded["date"] = pd.to_datetime(traded["date"], errors="coerce")
    traded["month"] = traded["date"].dt.to_period("M").astype(str)
    monthly = traded.groupby("month", observed=False).agg(
        profit=("profit", "sum"),
        avg_clv_cents=("clv_cents", "mean"),
    )
    return {
        "trade_months": int(len(monthly)),
        "positive_profit_month_share": float((monthly["profit"] > 0).mean()) if len(monthly) else 0.0,
        "positive_clv_month_share": float((monthly["avg_clv_cents"] > 0).mean()) if len(monthly) else 0.0,
    }


def _prepare(markets: pd.DataFrame, model_weight: float) -> pd.DataFrame:
    required = ["game_date", "game_id", "market_ticker", "model_yes_prob", "yes_bid", "yes_ask"]
    missing = [column for column in required if column not in markets.columns]
    if missing:
        raise ValueError(f"Market-anchor rows are missing columns: {missing}")

    frame = markets.copy()
    for column in ["model_yes_prob", "yes_bid", "yes_ask"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["model_yes_prob", "yes_bid", "yes_ask"]).copy()
    market_mid_prob = ((frame["yes_bid"] + frame["yes_ask"]) / 200.0).clip(0.0, 1.0)
    frame["raw_model_yes_prob"] = frame["model_yes_prob"]
    frame["market_anchor_mid_prob"] = market_mid_prob
    frame["model_yes_prob"] = (
        market_mid_prob + (frame["raw_model_yes_prob"] - market_mid_prob) * float(model_weight)
    ).clip(0.0, 1.0)
    frame["market_anchor_model_weight"] = float(model_weight)
    return frame.reset_index(drop=True)


def sweep_market_anchor(
    markets: pd.DataFrame,
    bankroll: float = 100.0,
    model_weights: list[float] | None = None,
    edge_thresholds: list[float] | None = None,
    max_bet_fraction: float = 0.03,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sweep market-anchored probability blends and realistic two-sided backtests."""

    model_weights = model_weights or [0.25, 0.40, 0.50, 0.65, 0.80, 1.0]
    edge_thresholds = edge_thresholds or [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    rows: list[dict[str, Any]] = []
    for weight in model_weights:
        prepared = _prepare(markets, model_weight=weight)
        for edge_threshold in edge_thresholds:
            trades = run_backtest(
                prepared,
                starting_bankroll=bankroll,
                edge_threshold=edge_threshold,
                max_bet_fraction=max_bet_fraction,
                allow_no_trades=True,
            )
            summary = summarize_backtest(trades, starting_bankroll=bankroll)
            month_summary = _monthly_metrics(trades)
            row = {
                "model_weight": float(weight),
                "market_weight": float(1.0 - weight),
                "edge_threshold": float(edge_threshold),
                "max_bet_fraction": float(max_bet_fraction),
                "score": float(
                    summary.get("average_profit_per_trade", 0.0) * 0.40
                    + summary.get("average_clv_cents", 0.0) / 100.0 * 0.25
                    + summary.get("positive_clv_rate", 0.0) * 0.15
                    + month_summary.get("positive_profit_month_share", 0.0) * 0.10
                    + month_summary.get("positive_clv_month_share", 0.0) * 0.10
                ),
                **summary,
                **month_summary,
            }
            row["status"] = (
                "candidate"
                if row["num_trades"] >= 100
                and row["average_profit_per_trade"] > 0
                and row["average_clv_cents"] > 0
                and row["positive_clv_rate"] >= 0.50
                and row["positive_profit_month_share"] >= 0.67
                else "watchlist"
                if row["num_trades"] >= 30
                and row["average_profit_per_trade"] > 0
                and row["average_clv_cents"] > 0
                else "not_ready"
            )
            rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return results, {"status": "not_ready", "rules_tested": 0}
    rank = {"candidate": 0, "watchlist": 1, "not_ready": 2}
    results["_rank"] = results["status"].map(rank).fillna(99)
    results = results.sort_values(["_rank", "score", "num_trades"], ascending=[True, False, False]).drop(
        columns=["_rank"]
    ).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    broad = results[pd.to_numeric(results["num_trades"], errors="coerce").ge(100)].copy()
    broad_best = broad.iloc[0].to_dict() if not broad.empty else {}
    summary = {
        "status": str(best.get("status", "not_ready")),
        "rules_tested": int(len(results)),
        "candidate_rules": int(results["status"].eq("candidate").sum()),
        "watchlist_rules": int(results["status"].eq("watchlist").sum()),
        "best_model_weight": float(best.get("model_weight", 0.0) or 0.0),
        "best_market_weight": float(best.get("market_weight", 0.0) or 0.0),
        "best_edge_threshold": float(best.get("edge_threshold", 0.0) or 0.0),
        "best_trades": int(best.get("num_trades", 0) or 0),
        "best_ending_bankroll": float(best.get("ending_bankroll", bankroll) or bankroll),
        "best_average_clv_cents": float(best.get("average_clv_cents", 0.0) or 0.0),
        "best_positive_clv_rate": float(best.get("positive_clv_rate", 0.0) or 0.0),
        "best_average_profit_per_trade": float(best.get("average_profit_per_trade", 0.0) or 0.0),
        "broad_best_model_weight": float(broad_best.get("model_weight", 0.0) or 0.0),
        "broad_best_market_weight": float(broad_best.get("market_weight", 0.0) or 0.0),
        "broad_best_edge_threshold": float(broad_best.get("edge_threshold", 0.0) or 0.0),
        "broad_best_trades": int(broad_best.get("num_trades", 0) or 0),
        "broad_best_ending_bankroll": float(broad_best.get("ending_bankroll", bankroll) or bankroll),
        "broad_best_average_clv_cents": float(broad_best.get("average_clv_cents", 0.0) or 0.0),
        "broad_best_positive_clv_rate": float(broad_best.get("positive_clv_rate", 0.0) or 0.0),
        "broad_best_average_profit_per_trade": float(broad_best.get("average_profit_per_trade", 0.0) or 0.0),
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "note": (
            "Research-only market-anchor sweep. It tests whether shrinking model probabilities toward market mid "
            "reduces overconfidence; do not promote unless proof gates pass."
        ),
    }
    return results, summary


def save_market_anchor_outputs(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "market_anchor",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_root / f"{prefix}_sweep.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
