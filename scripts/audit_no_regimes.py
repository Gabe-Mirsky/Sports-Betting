"""Audit calibrated NO-only signals by market regime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.no_side_audit import build_no_side_audit, save_no_side_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit calibrated NO-only regimes.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--matched-markets-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="no_regime")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _enrich_with_bid_ask(rows: pd.DataFrame, matched_markets_path: Path) -> pd.DataFrame:
    if not matched_markets_path.exists() or rows.empty or "market_ticker" not in rows.columns:
        return rows
    markets = pd.read_csv(matched_markets_path, dtype={"game_id": str, "market_ticker": str})
    keep = [column for column in ["market_ticker", "yes_bid", "yes_ask"] if column in markets.columns]
    if "market_ticker" not in keep:
        return rows
    markets = markets[keep].drop_duplicates("market_ticker")
    overlap = [column for column in ["yes_bid", "yes_ask"] if column in rows.columns]
    base = rows.drop(columns=overlap) if overlap else rows
    return base.merge(markets, on="market_ticker", how="left")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = (
        Path(args.input_path)
        if args.input_path
        else reports_dir / "edge_calibration_price_aware_best_trades.csv"
    )
    matched_markets_path = (
        Path(args.matched_markets_path)
        if args.matched_markets_path
        else reports_dir / "matched_markets.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})
    rows = _enrich_with_bid_ask(rows, matched_markets_path)
    reports, summary = build_no_side_audit(rows, signal_column=args.signal_column)
    save_no_side_audit(reports, summary, output_dir, prefix=args.prefix)
    print(f"NO regime rows: {summary.get('selected_no_rows', 0):,}")
    print(f"Status: {summary.get('status', 'n/a')}")
    print(f"Average CLV: {summary.get('avg_clv_cents', 0.0):+.2f} cents")
    print(f"Positive CLV rate: {summary.get('positive_clv_rate', 0.0):.1%}")
    print(f"Profit/share: {summary.get('avg_profit_per_share', 0.0):+.3f}")
    print(f"Wide spread rate: {summary.get('wide_spread_rate', 0.0):.1%}")
    print(f"Saved NO regime audit to: {output_dir}")


if __name__ == "__main__":
    main()
