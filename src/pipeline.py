"""Build command lists for refreshing project outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineOptions:
    python_executable: str
    project_root: Path
    download: bool = False
    start_season: int | None = None
    end_season: int | None = None
    force_download: bool = False
    markets_path: Path | None = None
    bankroll: float = 100.0
    edge_threshold: float = 0.05
    thresholds: str = "0.00,0.02,0.05,0.08,0.10,0.12,0.15"
    skip_features: bool = False
    skip_train: bool = False
    skip_walk_forward: bool = False
    skip_model_tuning: bool = False
    skip_market_type_models: bool = False
    skip_backtest: bool = False
    skip_edge_calibration: bool = False
    skip_market_blend: bool = False
    skip_portfolio: bool = False
    skip_forward: bool = False
    skip_sweep: bool = False
    skip_diagnostics: bool = False
    skip_dashboard: bool = False


def _script(project_root: Path, script_name: str) -> str:
    return str(project_root / "scripts" / script_name)


def build_pipeline_commands(options: PipelineOptions) -> list[tuple[str, list[str]]]:
    """Return named commands for a full local pipeline refresh."""

    python = options.python_executable
    root = options.project_root
    commands: list[tuple[str, list[str]]] = []

    if options.download:
        command = [python, _script(root, "download_nba_data.py")]
        if options.start_season is not None:
            command.extend(["--start-season", str(options.start_season)])
        if options.end_season is not None:
            command.extend(["--end-season", str(options.end_season)])
        if options.force_download:
            command.append("--force")
        commands.append(("Download NBA data", command))

    if not options.skip_features:
        commands.append(("Build features", [python, _script(root, "build_features.py")]))

    if not options.skip_train:
        commands.append(("Train models", [python, _script(root, "train.py")]))

    if not options.skip_walk_forward:
        commands.append(("Walk-forward predictions", [python, _script(root, "walk_forward.py")]))

    if not options.skip_model_tuning:
        commands.append(("Tune home-win model", [python, _script(root, "tune_model.py")]))

    if not options.skip_market_type_models:
        commands.append(("Train spread and total models", [python, _script(root, "train_market_type_models.py")]))

    if not options.skip_walk_forward and not options.skip_model_tuning and not options.skip_market_type_models:
        commands.append(("Build home-win ensemble", [python, _script(root, "build_home_win_ensemble.py")]))

    markets_path = str(
        options.markets_path
        if options.markets_path is not None
        else root / "data" / "kalshi" / "markets_mock.csv"
    )
    walk_forward_predictions = str(root / "data" / "reports" / "walk_forward_predictions.csv")

    if not options.skip_backtest:
        commands.append(
            (
                "Run backtest",
                [
                    python,
                    _script(root, "run_backtest.py"),
                    "--bankroll",
                    str(options.bankroll),
                    "--edge-threshold",
                    str(options.edge_threshold),
                    "--markets-path",
                    markets_path,
                    "--predictions-path",
                    walk_forward_predictions,
                ],
            )
        )

    if not options.skip_market_blend:
        commands.append(
            (
                "Market blend",
                [
                    python,
                    _script(root, "market_blend.py"),
                    "--bankroll",
                    str(options.bankroll),
                    "--edge-threshold",
                    str(options.edge_threshold),
                ],
            )
        )

    if not options.skip_edge_calibration:
        commands.append(("Calibrate edges", [python, _script(root, "calibrate_edges.py")]))

    if not options.skip_portfolio:
        commands.append(("Optimize individual slate", [python, _script(root, "optimize_portfolio.py")]))
        if not options.skip_edge_calibration:
            commands.append(
                (
                    "Optimize calibrated slate",
                    [python, _script(root, "optimize_portfolio.py"), "--use-calibrated-edges"],
                )
            )

    if not options.skip_edge_calibration and not options.skip_market_blend:
        commands.append(
            (
                "Calibrate market-blend edges",
                [
                    python,
                    _script(root, "calibrate_edges.py"),
                    "--trades-path",
                    str(root / "data" / "reports" / "backtest_trades_market_blend.csv"),
                    "--output-calibrated-path",
                    str(root / "data" / "reports" / "edge_calibrated_trades_market_blend.csv"),
                    "--output-bins-path",
                    str(root / "data" / "reports" / "edge_calibration_bins_market_blend.csv"),
                    "--output-summary-path",
                    str(root / "data" / "reports" / "edge_calibration_summary_market_blend.json"),
                    "--output-audit-path",
                    str(root / "data" / "reports" / "edge_calibration_audit_market_blend.csv"),
                    "--output-negative-edge-path",
                    str(root / "data" / "reports" / "edge_calibration_negative_edge_signals_market_blend.csv"),
                    "--output-audit-summary-path",
                    str(root / "data" / "reports" / "edge_calibration_audit_summary_market_blend.json"),
                ],
            )
        )
        commands.append(("Build consensus calibrated edges", [python, _script(root, "build_consensus_edges.py")]))
        commands.append(("Screen robust consensus edges", [python, _script(root, "screen_robust_edges.py")]))
        if not options.skip_portfolio:
            commands.append(
                (
                    "Optimize market-blend calibrated slate",
                    [
                        python,
                        _script(root, "optimize_portfolio.py"),
                        "--use-calibrated-edges",
                        "--trades-path",
                        str(root / "data" / "reports" / "edge_calibrated_trades_market_blend.csv"),
                        "--output-trades-path",
                        str(root / "data" / "reports" / "portfolio_trades_market_blend_calibrated.csv"),
                        "--output-slates-path",
                        str(root / "data" / "reports" / "portfolio_slates_market_blend_calibrated.csv"),
                        "--output-summary-path",
                        str(root / "data" / "reports" / "portfolio_summary_market_blend_calibrated.json"),
                    ],
                )
            )
            commands.append(
                (
                    "Optimize consensus calibrated slate",
                    [
                        python,
                        _script(root, "optimize_portfolio.py"),
                        "--use-calibrated-edges",
                        "--trades-path",
                        str(root / "data" / "reports" / "edge_consensus_calibrated_trades.csv"),
                        "--trade-column",
                        "consensus_trade",
                        "--expected-roi-column",
                        "consensus_expected_roi",
                        "--output-trades-path",
                        str(root / "data" / "reports" / "portfolio_trades_consensus_calibrated.csv"),
                        "--output-slates-path",
                        str(root / "data" / "reports" / "portfolio_slates_consensus_calibrated.csv"),
                        "--output-summary-path",
                        str(root / "data" / "reports" / "portfolio_summary_consensus_calibrated.json"),
                    ],
                )
            )
            commands.append(
                (
                    "Optimize robust consensus slate",
                    [
                        python,
                        _script(root, "optimize_portfolio.py"),
                        "--use-calibrated-edges",
                        "--trades-path",
                        str(root / "data" / "reports" / "edge_robust_consensus_trades.csv"),
                        "--trade-column",
                        "robust_calibrated_trade",
                        "--expected-roi-column",
                        "robust_expected_roi",
                        "--output-trades-path",
                        str(root / "data" / "reports" / "portfolio_trades_robust_consensus.csv"),
                        "--output-slates-path",
                        str(root / "data" / "reports" / "portfolio_slates_robust_consensus.csv"),
                        "--output-summary-path",
                        str(root / "data" / "reports" / "portfolio_summary_robust_consensus.json"),
                    ],
                )
            )
            commands.append(("Analyze consensus stability", [python, _script(root, "analyze_signal_stability.py")]))
            commands.append(
                (
                    "Analyze robust stability",
                    [
                        python,
                        _script(root, "analyze_signal_stability.py"),
                        "--input-path",
                        str(root / "data" / "reports" / "edge_robust_consensus_trades.csv"),
                        "--signal-column",
                        "robust_calibrated_trade",
                        "--expected-roi-column",
                        "robust_expected_roi",
                        "--output-path",
                        str(root / "data" / "reports" / "signal_stability_robust_consensus.csv"),
                        "--output-summary-path",
                        str(root / "data" / "reports" / "signal_stability_robust_consensus_summary.json"),
                    ],
                )
            )
            commands.append(("Assess strategy readiness", [python, _script(root, "strategy_readiness.py")]))
            commands.append(("Build headline slate result", [python, _script(root, "build_headline_backtest.py")]))
            if not options.skip_sweep:
                commands.append(("Sweep signal rules", [python, _script(root, "sweep_signal_rules.py")]))
                commands.append(
                    ("Validate signal rules walk-forward", [python, _script(root, "validate_signal_rules_walk_forward.py")])
                )
            commands.append(("Analyze parlay correlations", [python, _script(root, "analyze_parlay_correlations.py")]))
            commands.append(("Build Kalshi market taxonomy", [python, _script(root, "build_kalshi_market_taxonomy.py")]))
            commands.append(("Audit market-type lines", [python, _script(root, "audit_market_type_lines.py")]))
            commands.append(("Evaluate line markets", [python, _script(root, "evaluate_line_markets.py")]))
            commands.append(("Extract multivariate NBA legs", [python, _script(root, "extract_multivariate_nba_legs.py")]))

    if not options.skip_forward:
        commands.append(("Build forward recommendations", [python, _script(root, "build_forward_recommendations.py")]))

    if not options.skip_sweep:
        commands.append(
            (
                "Sweep thresholds",
                [
                    python,
                    _script(root, "sweep_thresholds.py"),
                    "--thresholds",
                    options.thresholds,
                    "--bankroll",
                    str(options.bankroll),
                    "--markets-path",
                    markets_path,
                    "--predictions-path",
                    walk_forward_predictions,
                ],
            )
        )

    if not options.skip_diagnostics:
        commands.append(("Audit Kalshi vs model", [python, _script(root, "audit_kalshi_vs_model.py")]))
        commands.append(("Analyze results", [python, _script(root, "analyze_results.py")]))
        commands.append(("Security audit", [python, _script(root, "security_audit.py")]))

    if not options.skip_dashboard:
        commands.append(("Build dashboard", [python, _script(root, "build_dashboard.py")]))

    return commands
