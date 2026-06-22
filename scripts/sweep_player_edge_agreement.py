"""Sweep simple player-aware versus team-only agreement filters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy.player_edge_agreement import (  # noqa: E402
    build_player_edge_agreement_report,
    save_player_edge_agreement_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep player-aware/team-only edge agreement policies.")
    parser.add_argument("--player-trades-path", default=None)
    parser.add_argument("--team-trades-path", default=None)
    parser.add_argument("--player-signal-column", default="trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="player_edge_agreement")
    parser.add_argument("--min-train-months", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = PROJECT_ROOT / "data" / "reports"
    player_path = (
        Path(args.player_trades_path)
        if args.player_trades_path
        else reports_dir / "player_market_player_aware_trades.csv"
    )
    team_path = (
        Path(args.team_trades_path)
        if args.team_trades_path
        else reports_dir / "player_market_team_only_trades.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    player_trades = pd.read_csv(player_path, dtype={"game_id": str, "market_ticker": str})
    team_trades = pd.read_csv(team_path, dtype={"game_id": str, "market_ticker": str})
    rows, descriptive, folds, summary = build_player_edge_agreement_report(
        player_trades,
        team_trades,
        player_signal_column=args.player_signal_column,
        min_train_months=args.min_train_months,
    )
    save_player_edge_agreement_report(rows, descriptive, folds, summary, output_dir, prefix=args.prefix)
    print(f"Player/team agreement status: {summary.get('status')}")
    print(f"Signals: {summary.get('signals', 0):,}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Best descriptive policy: {summary.get('descriptive_best_policy', 'n/a')}")
    print(f"Saved player/team agreement reports to: {output_dir}")


if __name__ == "__main__":
    main()
