"""Refresh features, models, reports, backtests, sweeps, and dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import PipelineOptions, build_pipeline_commands  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full local project pipeline.")
    parser.add_argument("--download", action="store_true", help="Download NBA data before rebuilding.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--markets-path", default=None)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--thresholds", default="0.00,0.02,0.05,0.08,0.10,0.12,0.15")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-model-tuning", action="store_true")
    parser.add_argument("--skip-market-type-models", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-edge-calibration", action="store_true")
    parser.add_argument("--skip-market-blend", action="store_true")
    parser.add_argument("--skip-portfolio", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = PipelineOptions(
        python_executable=sys.executable,
        project_root=PROJECT_ROOT,
        download=args.download,
        start_season=args.start_season,
        end_season=args.end_season,
        force_download=args.force_download,
        markets_path=Path(args.markets_path) if args.markets_path else None,
        bankroll=args.bankroll,
        edge_threshold=args.edge_threshold,
        thresholds=args.thresholds,
        skip_features=args.skip_features,
        skip_train=args.skip_train,
        skip_walk_forward=args.skip_walk_forward,
        skip_model_tuning=args.skip_model_tuning,
        skip_market_type_models=args.skip_market_type_models,
        skip_backtest=args.skip_backtest,
        skip_edge_calibration=args.skip_edge_calibration,
        skip_market_blend=args.skip_market_blend,
        skip_portfolio=args.skip_portfolio,
        skip_forward=args.skip_forward,
        skip_sweep=args.skip_sweep,
        skip_diagnostics=args.skip_diagnostics,
        skip_dashboard=args.skip_dashboard,
    )
    commands = build_pipeline_commands(options)

    print(f"Running {len(commands)} pipeline steps.", flush=True)
    for index, (name, command) in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("\nPipeline complete.", flush=True)
    print(f"Dashboard: {PROJECT_ROOT / 'data' / 'reports' / 'dashboard.html'}", flush=True)


if __name__ == "__main__":
    main()
