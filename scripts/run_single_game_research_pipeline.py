"""Run the repeatable single-game edge research pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> str:
    return str(PROJECT_ROOT / "scripts" / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the single-game research path: NBA data, features, models, "
            "Kalshi markets, pregame prices, audit, backtest, and dashboard."
        )
    )
    parser.add_argument("--download", action="store_true", help="Refresh NBA team and player data first.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--kalshi-start-date", default="2023-10-01")
    parser.add_argument("--kalshi-end-date", default=date.today().isoformat())
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--max-spread-cents", type=float, default=10.0)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--skip-market-pull", action="store_true")
    parser.add_argument("--skip-candles", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    python = sys.executable
    commands: list[tuple[str, list[str]]] = []

    if args.download:
        team_command = [
            python,
            _script("download_nba_data.py"),
            "--start-season",
            str(args.start_season),
            "--end-season",
            str(args.end_season),
        ]
        player_command = [
            python,
            _script("download_nba_player_data.py"),
            "--start-season",
            str(args.start_season),
            "--end-season",
            str(args.end_season),
        ]
        if args.force_download:
            team_command.append("--force")
            player_command.append("--force")
        commands.append(("Refresh NBA team data", team_command))
        commands.append(("Refresh NBA player data", player_command))

    commands.extend(
        [
            ("Build leak-safe features", [python, _script("build_features.py")]),
            ("Audit player data coverage", [python, _script("audit_player_data.py")]),
            ("Audit availability input gaps", [python, _script("audit_availability_gaps.py")]),
            ("Train home-win models", [python, _script("train.py")]),
            ("Run walk-forward evaluation", [python, _script("walk_forward.py")]),
            ("Compare player-aware market edge", [python, _script("compare_player_market_edges.py")]),
            ("Sweep player/team edge agreement", [python, _script("sweep_player_edge_agreement.py")]),
            ("Tune calibrated home-win model", [python, _script("tune_model.py")]),
            ("Train margin and total models", [python, _script("train_market_type_models.py")]),
            ("Build home-win ensemble audit", [python, _script("build_home_win_ensemble.py")]),
        ]
    )

    if not args.skip_market_pull:
        commands.append(
            (
                "Backfill raw public Kalshi Sports/NBA markets",
                [
                    python,
                    _script("kalshi_backfill_public_sports.py"),
                    "--log-level",
                    args.log_level,
                ],
            )
        )
        commands.append(
            (
                "Pull and cache Kalshi NBA markets",
                [
                    python,
                    _script("kalshi_backfill_markets.py"),
                    "--start-date",
                    args.kalshi_start_date,
                    "--end-date",
                    args.kalshi_end_date,
                    "--log-level",
                    args.log_level,
                ],
            )
        )
    commands.append(("Match Kalshi markets to NBA games", [python, _script("kalshi_match_games.py")]))
    if not args.skip_candles:
        commands.append(
            (
                "Download candles and extract pregame prices",
                [python, _script("kalshi_download_candles.py"), "--fetch-game-times"],
            )
        )
    commands.append(
        (
            "Build market truth audit",
            [
                python,
                _script("market_truth_audit.py"),
                "--max-spread-cents",
                str(args.max_spread_cents),
                "--min-volume",
                str(args.min_volume),
            ],
        )
    )
    commands.append(("Build Kalshi coverage audit", [python, _script("kalshi_coverage_report.py")]))
    commands.append(
        (
            "Run realistic bid/ask backtest",
            [
                python,
                _script("run_backtest.py"),
                "--predictions-path",
                str(PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"),
                "--market-source",
                "kalshi",
                "--bankroll",
                str(args.bankroll),
                "--edge-threshold",
                str(args.edge_threshold),
                "--min-volume",
                str(args.min_volume),
                "--max-bid-ask-spread-cents",
                str(args.max_spread_cents),
            ],
        )
    )
    commands.append(("Sweep market-anchored probability blends", [python, _script("sweep_market_anchor.py")]))
    commands.append(("Compare pregame snapshot entry policies", [python, _script("compare_pregame_snapshot_entries.py")]))
    commands.append(("Audit best two-hour snapshot CLV", [python, _script("audit_snapshot_clv.py")]))
    commands.append(("Calibrate edge bins", [python, _script("calibrate_edges.py")]))
    commands.append(("Calibrate side/price/edge bins", [python, _script("calibrate_edges_price_aware.py")]))
    commands.append(("Sweep price-aware calibration settings", [python, _script("sweep_price_aware_calibration.py")]))
    commands.append(("Build best price-aware calibration", [python, _script("build_best_price_aware_calibration.py")]))
    commands.append(
        (
            "Sweep calibrated player/team edge agreement",
            [
                python,
                _script("sweep_player_edge_agreement.py"),
                "--player-trades-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
                "--player-signal-column",
                "calibrated_trade",
                "--prefix",
                "player_calibrated_edge_agreement",
            ],
        )
    )
    commands.append(("Audit calibrated residuals", [python, _script("audit_residuals.py")]))
    commands.append(
        (
            "Audit best price-aware residuals",
            [
                python,
                _script("audit_residuals.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
                "--prefix",
                "residual_price_aware_best",
            ],
        )
    )
    commands.append(("Audit market movement attribution", [python, _script("audit_market_movement.py")]))
    commands.append(
        (
            "Audit best price-aware market movement",
            [
                python,
                _script("audit_market_movement.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
                "--prefix",
                "market_movement_price_aware_best",
            ],
        )
    )
    commands.append(
        (
            "Diagnose edge failure drivers",
            [
                python,
                _script("diagnose_edge_failures.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Sweep prior-month CLV slice filters",
            [
                python,
                _script("sweep_prior_clv_slice_filters.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Sweep side-suppression research policies",
            [
                python,
                _script("sweep_side_suppression.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Audit NO-only market regimes",
            [
                python,
                _script("audit_no_regimes.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Audit NO probability calibration vs CLV",
            [
                python,
                _script("audit_no_calibration.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Sweep NO calibration guardrails",
            [
                python,
                _script("sweep_no_calibration_guardrails.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Sweep NO probability shrinkage",
            [
                python,
                _script("sweep_no_shrinkage.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
            ],
        )
    )
    commands.append(
        (
            "Sweep NO player-agreement guardrails",
            [
                python,
                _script("sweep_no_calibration_guardrails.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "player_calibrated_edge_agreement_rows.csv"),
                "--prefix",
                "no_player_agreement_guardrail",
            ],
        )
    )
    commands.append(("Sweep corrected CLV rules", [python, _script("sweep_corrected_clv_rules.py")]))
    commands.append(
        (
            "Sweep corrected best price-aware CLV rules",
            [
                python,
                _script("sweep_corrected_clv_rules.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
                "--prefix",
                "corrected_clv_price_aware_best",
                "--min-rows",
                "10",
                "--min-train-months",
                "2",
            ],
        )
    )
    commands.append(("Sweep residual guardrails", [python, _script("sweep_residual_guardrails.py")]))
    commands.append(
        (
            "Sweep best price-aware residual guardrails",
            [
                python,
                _script("sweep_residual_guardrails.py"),
                "--input-path",
                str(PROJECT_ROOT / "data" / "reports" / "edge_calibration_price_aware_best_trades.csv"),
                "--prefix",
                "residual_guardrail_price_aware_best",
            ],
        )
    )
    commands.append(("Analyze closing-line value", [python, _script("analyze_clv.py")]))
    commands.append(("Build side-specific CLV-filtered strategy", [python, _script("build_clv_filtered_strategy.py")]))
    commands.append(("Sweep CLV price/month stability rules", [python, _script("sweep_clv_price_month_rules.py")]))
    commands.append(
        (
            "Walk-forward validate CLV price/month rules",
            [
                python,
                _script("sweep_clv_price_month_rules.py"),
                "--walk-forward",
            ],
        )
    )
    commands.append(("Analyze CLV decay drivers", [python, _script("analyze_clv_decay.py")]))
    commands.append(("Build defensive CLV-filtered strategy", [python, _script("build_defensive_strategy.py")]))
    commands.append(("Sweep defensive rule thresholds", [python, _script("build_defensive_strategy.py"), "--sweep"]))
    commands.append(
        (
            "Walk-forward validate defensive rules",
            [python, _script("build_defensive_strategy.py"), "--walk-forward"],
        )
    )
    commands.append(
        (
            "Test defensive sample expansion",
            [python, _script("build_defensive_strategy.py"), "--sample-expansion"],
        )
    )
    commands.append(
        (
            "Audit defensive failure month",
            [
                python,
                _script("audit_defensive_failure_month.py"),
                "--schedule-context-path",
                str(PROJECT_ROOT / "data" / "processed" / "modeling_dataset.parquet"),
            ],
        )
    )
    commands.append(
        (
            "Optimize conservative calibrated single-bet slate",
            [
                python,
                _script("optimize_portfolio.py"),
                "--use-calibrated-edges",
                "--bankroll",
                str(args.bankroll),
                "--max-trades-per-slate",
                "3",
                "--max-slate-fraction",
                "0.09",
                "--max-markets-per-game",
                "1",
                "--max-markets-per-team",
                "2",
            ],
        )
    )
    commands.append(
        (
            "Optimize CLV-filtered single-bet slate",
            [
                python,
                _script("optimize_portfolio.py"),
                "--trades-path",
                str(PROJECT_ROOT / "data" / "reports" / "clv_filtered_trades.csv"),
                "--trade-column",
                "clv_filtered_trade",
                "--expected-roi-column",
                "calibrated_expected_roi",
                "--min-edge",
                "-1.0",
                "--bankroll",
                str(args.bankroll),
                "--max-trades-per-slate",
                "3",
                "--max-slate-fraction",
                "0.09",
                "--max-markets-per-game",
                "1",
                "--max-markets-per-team",
                "2",
                "--output-trades-path",
                str(PROJECT_ROOT / "data" / "reports" / "portfolio_trades_clv_filtered.csv"),
                "--output-slates-path",
                str(PROJECT_ROOT / "data" / "reports" / "portfolio_slates_clv_filtered.csv"),
                "--output-summary-path",
                str(PROJECT_ROOT / "data" / "reports" / "portfolio_summary_clv_filtered.json"),
            ],
        )
    )
    commands.append(
        (
            "Optimize defensive CLV-filtered single-bet slate",
            [
                python,
                _script("optimize_portfolio.py"),
                "--trades-path",
                str(PROJECT_ROOT / "data" / "reports" / "defensive_filtered_trades.csv"),
                "--trade-column",
                "defensive_trade",
                "--expected-roi-column",
                "calibrated_expected_roi",
                "--min-edge",
                "-1.0",
                "--bankroll",
                str(args.bankroll),
                "--max-trades-per-slate",
                "3",
                "--max-slate-fraction",
                "0.09",
                "--max-markets-per-game",
                "1",
                "--max-markets-per-team",
                "2",
                "--output-trades-path",
                str(PROJECT_ROOT / "data" / "reports" / "portfolio_trades_defensive.csv"),
                "--output-slates-path",
                str(PROJECT_ROOT / "data" / "reports" / "portfolio_slates_defensive.csv"),
                "--output-summary-path",
                str(PROJECT_ROOT / "data" / "reports" / "portfolio_summary_defensive.json"),
            ],
        )
    )
    commands.append(("Score single-game strategy readiness", [python, _script("strategy_readiness.py")]))
    commands.append(("Build single-game proof gates", [python, _script("single_game_proof.py")]))
    commands.append(
        (
            "Build fair-price single-game signals",
            [
                python,
                _script("build_fair_prices.py"),
                "--edge-threshold",
                str(max(args.edge_threshold - 0.02, 0.0)),
                "--max-spread-cents",
                str(args.max_spread_cents),
                "--min-volume",
                str(args.min_volume),
                "--bankroll",
                str(args.bankroll),
            ],
        )
    )
    commands.append(("Build conservative parlay recommendations", [python, _script("build_parlay_recommendations.py")]))
    commands.append(("Sweep side-specific probability shrinkage", [python, _script("sweep_side_specific_shrinkage.py")]))
    commands.append(("Audit NO settlement calibration", [python, _script("audit_no_settlement_calibration.py")]))
    commands.append(
        (
            "Analyze single-game edge root causes",
            [python, _script("analyze_single_game_edge_diagnostics.py")],
        )
    )
    if not args.skip_dashboard:
        commands.append(("Build dashboard", [python, _script("build_dashboard.py")]))
    return commands


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    print(f"Running {len(commands)} single-game research steps.", flush=True)
    for index, (name, command) in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("\nSingle-game research pipeline complete.", flush=True)
    print(f"Market truth audit: {PROJECT_ROOT / 'data' / 'reports' / 'market_truth_audit.csv'}", flush=True)
    print(f"Dashboard: {PROJECT_ROOT / 'data' / 'reports' / 'dashboard.html'}", flush=True)


if __name__ == "__main__":
    main()
