"""Compare team-only and player-aware walk-forward signals against Kalshi prices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from models.train_model import (  # noqa: E402
    BASELINE_FEATURE_COLUMNS,
    DEFAULT_FEATURE_COLUMNS,
    RICH_TEAM_FORM_FEATURE_COLUMNS,
    available_feature_columns,
)
from models.walk_forward import walk_forward_predict  # noqa: E402
from strategy.backtest import prepare_candlestick_backtest_markets, run_backtest, summarize_backtest  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare team-only and player-aware model edges vs Kalshi prices.")
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--matches-path", default=None)
    parser.add_argument("--prices-path", default=None)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--max-bid-ask-spread-cents", type=float, default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _run_variant(
    name: str,
    modeling: pd.DataFrame,
    feature_columns: list[str],
    matches: pd.DataFrame,
    prices: pd.DataFrame,
    config: Any,
    bankroll: float,
    edge_threshold: float,
    min_volume: float,
    max_bid_ask_spread_cents: float,
    output_dir: Path,
) -> dict[str, Any]:
    predictions, metrics = walk_forward_predict(
        modeling,
        target_column=config.model.target,
        train_start_season=config.model.train_start_season,
        model_type=config.model.model_type,
        random_seed=config.project.random_seed,
        feature_columns=feature_columns,
    )
    predictions = predictions.copy()
    predictions["game_id"] = predictions["game_id"].astype(str)
    predictions["game_date"] = pd.to_datetime(predictions["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    matched, diagnostics = prepare_candlestick_backtest_markets(
        predictions,
        matches,
        prices,
        min_volume=min_volume,
        allowed_price_qualities=[item.strip() for item in str(config.backtest.allowed_price_qualities).split(",")],
        require_bid_ask=bool(config.backtest.require_bid_ask),
        max_candle_interval_minutes=config.backtest.max_candle_interval_minutes,
        max_bid_ask_spread_cents=max_bid_ask_spread_cents,
    )
    if matched.empty:
        trades = pd.DataFrame()
        backtest_summary = {
            "num_trades": 0,
            "ending_bankroll": bankroll,
            "total_return_pct": 0.0,
            "average_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "average_profit_per_trade": 0.0,
        }
    else:
        trades = run_backtest(
            matched,
            starting_bankroll=bankroll,
            edge_threshold=edge_threshold,
            max_bet_fraction=config.strategy.max_bet_fraction,
            min_market_price=config.strategy.min_market_price,
            max_market_price=config.strategy.max_market_price,
            allow_no_trades=config.strategy.allow_no_trades,
        )
        backtest_summary = summarize_backtest(trades, starting_bankroll=bankroll)
        backtest_summary.update(diagnostics)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"player_market_{name}_predictions.csv", index=False)
    matched.to_csv(output_dir / f"player_market_{name}_matched.csv", index=False)
    trades.to_csv(output_dir / f"player_market_{name}_trades.csv", index=False)
    return {
        "feature_count": len(feature_columns),
        "walk_forward": metrics["overall"]["model"],
        "backtest": backtest_summary,
    }


def _delta(player: dict[str, Any], team: dict[str, Any], section: str, key: str) -> float:
    return float(player.get(section, {}).get(key, 0.0) or 0.0) - float(team.get(section, {}).get(key, 0.0) or 0.0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    reports_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "reports"
    output_summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "player_market_edge_comparison.json"
    )
    modeling_path = (
        Path(args.modeling_path)
        if args.modeling_path
        else resolve_project_path(config.data.processed_dir) / "modeling_dataset.parquet"
    )
    matches_path = (
        Path(args.matches_path)
        if args.matches_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_game_market_matches.csv"
    )
    prices_path = (
        Path(args.prices_path)
        if args.prices_path
        else PROJECT_ROOT / "data" / "processed" / "kalshi_pregame_prices.csv"
    )
    min_volume = args.min_volume if args.min_volume is not None else config.backtest.min_volume
    max_bid_ask_spread_cents = (
        args.max_bid_ask_spread_cents
        if args.max_bid_ask_spread_cents is not None
        else config.backtest.max_bid_ask_spread_cents
    )

    modeling = pd.read_parquet(modeling_path)
    matches = pd.read_csv(matches_path, dtype={"game_id": str})
    prices = pd.read_csv(prices_path, dtype={"game_id": str})
    if "game_date" in matches.columns:
        matches["game_date"] = pd.to_datetime(matches["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    team_features = available_feature_columns(modeling, BASELINE_FEATURE_COLUMNS + RICH_TEAM_FORM_FEATURE_COLUMNS)
    player_features = available_feature_columns(modeling, [column for column in DEFAULT_FEATURE_COLUMNS if column in modeling.columns])

    team_only = _run_variant(
        "team_only",
        modeling,
        team_features,
        matches,
        prices,
        config,
        args.bankroll,
        args.edge_threshold,
        min_volume,
        max_bid_ask_spread_cents,
        reports_dir,
    )
    player_aware = _run_variant(
        "player_aware",
        modeling,
        player_features,
        matches,
        prices,
        config,
        args.bankroll,
        args.edge_threshold,
        min_volume,
        max_bid_ask_spread_cents,
        reports_dir,
    )

    deltas = {
        "walk_forward_log_loss": _delta(player_aware, team_only, "walk_forward", "log_loss"),
        "walk_forward_auc": _delta(player_aware, team_only, "walk_forward", "roc_auc"),
        "backtest_return_pct": _delta(player_aware, team_only, "backtest", "total_return_pct"),
        "average_clv_cents": _delta(player_aware, team_only, "backtest", "average_clv_cents"),
        "positive_clv_rate": _delta(player_aware, team_only, "backtest", "positive_clv_rate"),
        "num_trades": _delta(player_aware, team_only, "backtest", "num_trades"),
    }
    decision = "watchlist"
    if deltas["walk_forward_log_loss"] < 0 and deltas["walk_forward_auc"] > 0:
        decision = "keep_player_features_for_modeling"
    if deltas["average_clv_cents"] > 0 and deltas["positive_clv_rate"] > 0:
        decision = "player_features_help_market_edge"
    if deltas["average_clv_cents"] < 0 and deltas["positive_clv_rate"] < 0:
        decision = "player_features_help_predictions_but_not_market_edge"

    payload = {
        "decision": decision,
        "bankroll": float(args.bankroll),
        "edge_threshold": float(args.edge_threshold),
        "min_volume": float(min_volume),
        "max_bid_ask_spread_cents": float(max_bid_ask_spread_cents),
        "team_only": team_only,
        "player_aware": player_aware,
        "player_minus_team_only": deltas,
        "note": "This compares prediction metrics and realistic Kalshi-price backtests. Do not promote a betting rule unless CLV and repeatability improve out of sample.",
    }
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"Decision: {decision}")
    print(f"Player minus team-only log loss: {deltas['walk_forward_log_loss']:+.4f}")
    print(f"Player minus team-only AUC: {deltas['walk_forward_auc']:+.4f}")
    print(f"Player minus team-only average CLV: {deltas['average_clv_cents']:+.2f} cents")
    print(f"Saved market comparison to: {output_summary_path}")


if __name__ == "__main__":
    main()
