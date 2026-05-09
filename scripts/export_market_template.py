"""Export a manual Kalshi market-entry template from saved predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import build_market_entry_template  # noqa: E402
from logging_setup import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export manual market-entry CSV template.")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--date", default=None, help="Single game date, YYYY-MM-DD.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--yes-side", choices=["home", "away", "both"], default="home")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    default_walk_forward_path = PROJECT_ROOT / "data" / "reports" / "walk_forward_predictions.csv"
    default_single_split_path = PROJECT_ROOT / "data" / "reports" / "model_predictions.csv"
    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else default_walk_forward_path
        if default_walk_forward_path.exists()
        else default_single_split_path
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "data" / "kalshi" / "markets_to_fill.csv"
    )

    predictions = pd.read_csv(predictions_path, dtype={"game_id": str})
    start_date = args.date or args.start_date
    end_date = args.date or args.end_date
    template = build_market_entry_template(
        predictions,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        season=args.season,
        yes_side=args.yes_side,
    )
    if args.limit is not None:
        template = template.head(args.limit)
        template.to_csv(output_path, index=False)

    print(f"Exported {len(template):,} market-entry rows.")
    print(f"Saved template to: {output_path}")
    print("Fill in yes_mid_cents, or yes_bid_cents and yes_ask_cents, before paper trading.")


if __name__ == "__main__":
    main()
