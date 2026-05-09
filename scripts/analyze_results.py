"""Generate extra diagnostics from prediction and backtest outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.diagnostics import generate_diagnostics  # noqa: E402
from reports.plots import (  # noqa: E402
    save_edge_bin_plot,
    save_probability_bin_plot,
    save_season_summary_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics from saved outputs.")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--trades-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "reports"
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else output_dir / "walk_forward_predictions.csv"
    )
    trades_path = Path(args.trades_path) if args.trades_path else output_dir / "backtest_trades.csv"

    paths = generate_diagnostics(predictions_path, trades_path, output_dir)

    probability_bins = pd.read_csv(paths["probability_bins"])
    season_summary = pd.read_csv(paths["season_summary"])
    edge_bins = pd.read_csv(paths["edge_bins"])
    save_probability_bin_plot(probability_bins, output_dir / "prediction_probability_bins.png")
    save_season_summary_plot(season_summary, output_dir / "prediction_season_summary.png")
    save_edge_bin_plot(edge_bins, output_dir / "backtest_edge_bins.png")

    print("Generated diagnostics:")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    print(f"- probability_bin_plot: {output_dir / 'prediction_probability_bins.png'}")
    print(f"- season_summary_plot: {output_dir / 'prediction_season_summary.png'}")
    print(f"- edge_bin_plot: {output_dir / 'backtest_edge_bins.png'}")


if __name__ == "__main__":
    main()
