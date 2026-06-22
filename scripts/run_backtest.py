"""Run a fake-bankroll backtest against matched Kalshi-style market data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from data.kalshi_client import load_mock_kalshi_markets, match_games_to_markets  # noqa: E402
from data.market_quality import analyze_market_data_quality, save_market_quality_report  # noqa: E402
from data.sportsbook_odds import load_sportsbook_odds  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from reports.summary import format_backtest_results  # noqa: E402
from strategy.backtest import (  # noqa: E402
    prepare_candlestick_backtest_markets,
    prepare_sportsbook_backtest_markets,
    run_backtest,
    save_backtest_outputs,
    summarize_backtest,
)

try:
    from reports.plots import save_edge_distribution, save_equity_curve  # noqa: E402

    PLOTS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional local matplotlib install
    PLOTS_AVAILABLE = False

    def save_edge_distribution(*args: object, **kwargs: object) -> None:
        return None

    def save_equity_curve(*args: object, **kwargs: object) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake-bankroll backtest.")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--edge-threshold", type=float, default=None)
    parser.add_argument("--max-bet-fraction", type=float, default=None)
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--matches-path", default=None)
    parser.add_argument("--prices-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--sportsbook-odds-path", default=None)
    parser.add_argument("--market-source", choices=["auto", "kalshi", "sportsbook"], default="auto")
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--allow-low-quality-prices", action="store_true")
    parser.add_argument("--no-require-bid-ask", action="store_true")
    parser.add_argument("--max-candle-interval-minutes", type=int, default=None)
    parser.add_argument("--max-bid-ask-spread-cents", type=float, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _split_quality_config(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _artifact_record(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "mtime_utc": "",
            "size_bytes": 0,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "mtime_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        "size_bytes": int(stat.st_size),
    }


def _stale_artifact_warnings(reference_time: float, input_paths: list[Path]) -> list[str]:
    warnings = []
    for path in input_paths:
        if not path.exists():
            warnings.append(f"missing_required_artifact:{path}")
        elif path.stat().st_mtime > reference_time:
            warnings.append(f"input_newer_than_output:{path}")
    return warnings


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    starting_bankroll = args.bankroll or config.strategy.starting_bankroll
    edge_threshold = args.edge_threshold if args.edge_threshold is not None else config.strategy.edge_threshold
    max_bet_fraction = (
        args.max_bet_fraction
        if args.max_bet_fraction is not None
        else config.strategy.max_bet_fraction
    )
    min_volume = args.min_volume if args.min_volume is not None else config.backtest.min_volume
    allowed_price_qualities = None if args.allow_low_quality_prices else _split_quality_config(
        config.backtest.allowed_price_qualities
    )
    require_bid_ask = config.backtest.require_bid_ask and not args.no_require_bid_ask
    max_candle_interval_minutes = (
        args.max_candle_interval_minutes
        if args.max_candle_interval_minutes is not None
        else config.backtest.max_candle_interval_minutes
    )
    max_bid_ask_spread_cents = (
        args.max_bid_ask_spread_cents
        if args.max_bid_ask_spread_cents is not None
        else config.backtest.max_bid_ask_spread_cents
    )

    markets_path = (
        Path(args.markets_path)
        if args.markets_path
        else PROJECT_ROOT / "data" / "kalshi" / "markets_mock.csv"
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
    sportsbook_odds_path = (
        Path(args.sportsbook_odds_path)
        if args.sportsbook_odds_path
        else Path(config.data.sportsbook_odds_path)
    )
    if not sportsbook_odds_path.is_absolute():
        sportsbook_odds_path = PROJECT_ROOT / sportsbook_odds_path

    use_sportsbook_mode = args.market_source == "sportsbook" or (
        args.market_source == "auto" and sportsbook_odds_path.exists()
    )
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else PROJECT_ROOT / "data" / "reports" / (
            "all_game_predictions.csv" if use_sportsbook_mode else "model_predictions.csv"
        )
    )

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str})
    diagnostics = None
    quality_report = {"warnings": []}
    unmatched_tickers: list[str] = []
    matched_path = PROJECT_ROOT / "data" / "reports" / "matched_markets.csv"

    if args.market_source == "kalshi" and args.markets_path is not None:
        raise SystemExit("--market-source kalshi uses matched candle files; do not pass --markets-path.")
    if args.market_source == "kalshi":
        missing = [path for path in [matches_path, prices_path] if not path.exists()]
        if missing:
            raise SystemExit(
                "Kalshi backtest mode requires matched markets and pregame prices. Missing: "
                + ", ".join(str(path) for path in missing)
            )

    use_candlestick_mode = (
        not use_sportsbook_mode
        and args.markets_path is None
        and matches_path.exists()
        and prices_path.exists()
    )
    resolved_market_source = "unknown"
    artifact_inputs = [predictions_path]
    if use_sportsbook_mode:
        resolved_market_source = "sportsbook"
        artifact_inputs.append(sportsbook_odds_path)
        sportsbook_odds = load_sportsbook_odds(sportsbook_odds_path)
        matched, diagnostics = prepare_sportsbook_backtest_markets(predictions, sportsbook_odds)
        if matched.empty:
            raise SystemExit(
                "No sportsbook odds matched model predictions. "
                f"Checked: {sportsbook_odds_path}"
            )
    elif use_candlestick_mode:
        resolved_market_source = "kalshi"
        artifact_inputs.extend([matches_path, prices_path])
        matches = pd.read_csv(matches_path, dtype={"game_id": str})
        prices = pd.read_csv(prices_path, dtype={"game_id": str})
        matched, diagnostics = prepare_candlestick_backtest_markets(
            predictions,
            matches,
            prices,
            min_volume=min_volume,
            allowed_price_qualities=allowed_price_qualities,
            require_bid_ask=require_bid_ask,
            max_candle_interval_minutes=max_candle_interval_minutes,
            max_bid_ask_spread_cents=max_bid_ask_spread_cents,
        )
        if matched.empty:
            raise SystemExit("No auto-matched Kalshi markets had usable pregame candle prices.")
    else:
        if args.market_source == "kalshi":
            raise SystemExit("Kalshi backtest mode could not be resolved from candle artifacts.")
        resolved_market_source = "mock_manual"
        artifact_inputs.append(markets_path)
        markets = load_mock_kalshi_markets(markets_path)
        matched = match_games_to_markets(predictions, markets)
        if matched.empty:
            raise SystemExit("No mock/manual Kalshi markets matched model predictions.")
        quality_report = analyze_market_data_quality(markets, matched)
        unmatched_tickers = sorted(
            set(markets["market_ticker"].astype(str)) - set(matched["market_ticker"].astype(str))
        )

    trades = run_backtest(
        matched,
        starting_bankroll=starting_bankroll,
        edge_threshold=edge_threshold,
        max_bet_fraction=max_bet_fraction,
        min_market_price=config.strategy.min_market_price,
        max_market_price=config.strategy.max_market_price,
        allow_no_trades=config.strategy.allow_no_trades,
    )
    summary = summarize_backtest(trades, starting_bankroll=starting_bankroll)
    if diagnostics is not None:
        summary.update(diagnostics)

    suffix = args.output_suffix or (
        "_sportsbook"
        if resolved_market_source == "sportsbook"
        else "_mock"
        if resolved_market_source == "mock_manual"
        else ""
    )
    trades_path = PROJECT_ROOT / "data" / "reports" / f"backtest_trades{suffix}.csv"
    summary_path = PROJECT_ROOT / "data" / "reports" / f"backtest_summary{suffix}.json"
    equity_path = PROJECT_ROOT / "data" / "reports" / f"equity_curve{suffix}.png"
    matching_report_path = PROJECT_ROOT / "data" / "reports" / f"market_matching_report{suffix}.json"
    skipped_report_path = PROJECT_ROOT / "data" / "reports" / f"backtest_skipped_games{suffix}.json"
    quality_report_path = PROJECT_ROOT / "data" / "reports" / f"market_data_quality_report{suffix}.json"
    edge_distribution_path = PROJECT_ROOT / "data" / "reports" / f"edge_distribution{suffix}.png"
    generated_at = pd.Timestamp.now(tz="UTC")
    stale_warnings = _stale_artifact_warnings(generated_at.timestamp(), artifact_inputs)
    selected_snapshot_targets = (
        diagnostics.get("preferred_snapshot_targets", []) if isinstance(diagnostics, dict) else []
    )
    summary.update(
        {
            "market_source": resolved_market_source,
            "requested_market_source": args.market_source,
            "canonical_kalshi_backtest": bool(resolved_market_source == "kalshi" and suffix == ""),
            "price_source": (
                "kalshi_candlesticks_bid_ask"
                if resolved_market_source == "kalshi" and require_bid_ask
                else "kalshi_candlesticks"
                if resolved_market_source == "kalshi"
                else "sportsbook_no_vig_moneyline"
                if resolved_market_source == "sportsbook"
                else "mock_or_manual_market_file"
            ),
            "snapshot_target": ",".join(str(item) for item in selected_snapshot_targets)
            if selected_snapshot_targets
            else "",
            "bid_ask_required": bool(require_bid_ask),
            "no_trades_allowed": bool(config.strategy.allow_no_trades),
            "stale_artifacts_detected": bool(stale_warnings),
            "artifact_warnings": stale_warnings,
            "artifact_inputs": [_artifact_record(path) for path in artifact_inputs],
            "generated_at_utc": generated_at.isoformat(),
        }
    )
    if not trades.empty:
        trades["market_source"] = resolved_market_source
        trades["price_source"] = summary["price_source"]
        trades["canonical_kalshi_backtest"] = bool(summary["canonical_kalshi_backtest"])
        trades["snapshot_target_order"] = summary["snapshot_target"]
    if suffix == "" and resolved_market_source != "kalshi":
        raise SystemExit(
            "Refusing to write unsuffixed canonical backtest outputs from "
            f"{resolved_market_source} mode."
        )
    save_backtest_outputs(trades, summary, trades_path, summary_path)
    save_equity_curve(trades, equity_path)
    save_edge_distribution(trades, edge_distribution_path)
    if suffix:
        matched_path = PROJECT_ROOT / "data" / "reports" / f"matched_markets{suffix}.csv"
    matched.to_csv(matched_path, index=False)
    if use_sportsbook_mode and diagnostics is not None:
        matched_ids = set(matched["game_id"].astype(str))
        skipped = predictions[~predictions["game_id"].astype(str).isin(matched_ids)].copy()
        skipped_payload = {
            "mode": "sportsbook_market_proxy",
            "skipped_games": int(skipped["game_id"].nunique() if "game_id" in skipped.columns else len(skipped)),
            "reason": "missing_moneyline_or_market_proxy",
            "by_dataset_split": diagnostics.get("skipped_games_by_dataset_split", {}),
        }
        skipped_report_path.write_text(json.dumps(skipped_payload, indent=2), encoding="utf-8")
    if diagnostics is None:
        matching_payload = {
            "markets_loaded": int(len(markets)),
            "markets_matched": int(len(matched)),
            "markets_unmatched": int(len(unmatched_tickers)),
            "unmatched_market_tickers": unmatched_tickers,
        }
    else:
        matching_payload = diagnostics
    matching_report_path.write_text(
        json.dumps(matching_payload, indent=2),
        encoding="utf-8",
    )
    if diagnostics is None:
        save_market_quality_report(quality_report, quality_report_path)
    else:
        quality_report = {
            "warnings": [],
            "mode": "candlestick_backtest" if resolved_market_source == "kalshi" else "sportsbook_market_proxy",
            "diagnostics": diagnostics,
            "filters": {
                "allowed_price_qualities": allowed_price_qualities,
                "require_bid_ask": require_bid_ask,
                "min_volume": min_volume,
                "max_candle_interval_minutes": max_candle_interval_minutes,
                "max_bid_ask_spread_cents": max_bid_ask_spread_cents,
            },
        }
        quality_report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")

    print(format_backtest_results(summary))
    if diagnostics is not None:
        print(f"Games available: {diagnostics['games_available']:,}")
        if diagnostics.get("mode") == "sportsbook_market_proxy":
            print(f"Games with sportsbook market proxy: {diagnostics['games_with_sportsbook_odds']:,}")
            print(f"Skipped games without moneyline/market proxy: {diagnostics['skipped_games_due_to_no_sportsbook_odds']:,}")
            print(f"Saved skipped-games report to: {skipped_report_path}")
        else:
            print(f"Games with matched Kalshi market: {diagnostics['games_with_matched_kalshi_market']:,}")
            print(f"Games with usable pregame price: {diagnostics['games_with_usable_pregame_price']:,}")
        print(f"Trades made ({summary.get('trade_timeline', 'n/a')}): {summary['num_trades']:,}")
        print(f"Final bankroll: ${summary['ending_bankroll']:.2f}")
    print(f"Matched markets: {len(matched):,}")
    print(f"Saved trades to: {trades_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved equity curve to: {equity_path}")
    print(f"Saved matched markets to: {matched_path}")
    print(f"Saved matching report to: {matching_report_path}")
    if quality_report["warnings"]:
        print("Market data quality warnings:")
        for warning in quality_report["warnings"]:
            print(f"- {warning}")
    if not PLOTS_AVAILABLE:
        print("Plot generation skipped because matplotlib is not installed in this Python environment.")
    print(f"Saved market quality report to: {quality_report_path}")


if __name__ == "__main__":
    main()
