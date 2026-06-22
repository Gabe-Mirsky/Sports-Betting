"""Normalize all free-backfill staging files into one historical game inventory.

Research-only. Reads local staging CSVs only (no network) and writes
``data/processed/historical_game_inventory.csv`` with one row per
canonical_game_key and the four availability flags OR-ed across sources.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logging_setup import setup_logging  # noqa: E402
from data.free_backfill import INVENTORY_PATH, STAGING_DIR, build_historical_game_inventory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the historical game inventory from staging.")
    parser.add_argument("--staging-dir", default=str(STAGING_DIR))
    parser.add_argument("--out-path", default=str(INVENTORY_PATH))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    inventory = build_historical_game_inventory(args.staging_dir, args.out_path)
    print(f"Wrote {len(inventory)} games -> {args.out_path}")
    if not inventory.empty:
        for flag in ("outcome_available", "game_odds_available", "prop_odds_available", "closing_available"):
            print(f"  {flag}: {int(inventory[flag].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
