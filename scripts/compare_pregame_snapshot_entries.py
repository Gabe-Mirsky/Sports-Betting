"""Compare realistic backtests using different pregame candle entry snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.backtest import prepare_candlestick_backtest_markets, run_backtest, summarize_backtest  # noqa: E402


SNAPSHOT_POLICIES = {
    "fixed_60m": ["pregame_60m"],
    "fixed_30m": ["pregame_30m"],
    "best_le_120m": ["pregame_best_le_120m"],
    "current_default": ["pregame_60m", "pregame_30m", "pregame_5m"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare backtests by pregame entry snapshot.")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--matches-path", default=None)
    parser.add_argument("--prices-path", default=None)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--max-bid-ask-spread-cents", type=float, default=10.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="pregame_snapshot_entry")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str, "market_ticker": str})


def _policy_summary(policy: str, summary: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": policy,
        "snapshot_targets": "|".join(str(item) for item in diagnostics.get("preferred_snapshot_targets", [])),
        "markets": int(summary.get("num_markets_seen", 0)),
        "trades": int(summary.get("num_trades", 0)),
        "ending_bankroll": float(summary.get("ending_bankroll", 0.0)),
        "total_return_pct": float(summary.get("total_return_pct", 0.0)),
        "avg_clv_cents": float(summary.get("average_clv_cents", 0.0)),
        "positive_clv_rate": float(summary.get("positive_clv_rate", 0.0)),
        "yes_avg_clv_cents": float(summary.get("yes_average_clv_cents", 0.0)),
        "no_avg_clv_cents": float(summary.get("no_average_clv_cents", 0.0)),
        "no_positive_clv_rate": float(summary.get("no_positive_clv_rate", 0.0)),
        "games_with_usable_pregame_price": int(diagnostics.get("games_with_usable_pregame_price", 0)),
        "price_rows_after_volume_filter": int(diagnostics.get("price_rows_after_volume_filter", 0)),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read(Path(args.predictions_path) if args.predictions_path else reports_dir / "walk_forward_predictions.csv")
    matches = _read(Path(args.matches_path) if args.matches_path else processed_dir / "kalshi_game_market_matches.csv")
    prices = _read(Path(args.prices_path) if args.prices_path else processed_dir / "kalshi_pregame_prices.csv")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for policy, targets in SNAPSHOT_POLICIES.items():
        matched, diagnostics = prepare_candlestick_backtest_markets(
            predictions,
            matches,
            prices,
            min_volume=args.min_volume,
            max_bid_ask_spread_cents=args.max_bid_ask_spread_cents,
            preferred_snapshot_targets=targets,
        )
        if matched.empty:
            trades = pd.DataFrame()
            summary = summarize_backtest(pd.DataFrame(), args.bankroll)
        else:
            trades = run_backtest(
                matched,
                starting_bankroll=args.bankroll,
                edge_threshold=args.edge_threshold,
            )
            summary = summarize_backtest(trades, args.bankroll)
            summary.update(diagnostics)
        rows.append(_policy_summary(policy, summary, diagnostics))
        details[policy] = {"summary": summary, "diagnostics": diagnostics}
        trades.to_csv(output_dir / f"{args.prefix}_{policy}_trades.csv", index=False)

    comparison = pd.DataFrame(rows).sort_values(
        ["avg_clv_cents", "ending_bankroll", "trades"],
        ascending=[False, False, False],
    )
    best = comparison.iloc[0].to_dict() if not comparison.empty else {}
    summary_output = {
        "status": "research_only",
        "best_policy_by_avg_clv": str(best.get("policy", "n/a")),
        "best_avg_clv_cents": float(best.get("avg_clv_cents", 0.0) or 0.0),
        "best_ending_bankroll": float(best.get("ending_bankroll", 0.0) or 0.0),
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "note": "Snapshot selection is research-only. Do not promote a snapshot policy unless downstream proof gates pass.",
        "details": details,
    }
    comparison.to_csv(output_dir / f"{args.prefix}_comparison.csv", index=False)
    (output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary_output, indent=2), encoding="utf-8")
    print(f"Snapshot policies tested: {len(comparison):,}")
    print(f"Best policy by CLV: {summary_output['best_policy_by_avg_clv']}")
    print(f"Best average CLV: {summary_output['best_avg_clv_cents']:+.2f} cents")
    print(f"Best ending bankroll: ${summary_output['best_ending_bankroll']:.2f}")
    print(f"Saved snapshot comparison to: {output_dir / f'{args.prefix}_comparison.csv'}")


if __name__ == "__main__":
    main()
