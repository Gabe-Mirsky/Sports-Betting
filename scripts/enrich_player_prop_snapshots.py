"""Enrich NBA player-prop snapshots with identity + settlement readiness.

Reads the normalized prop snapshots plus the nba_api current games and player
game logs, resolves player_id / team / game context for NBA rows, settles props
where actuals exist, and writes the enriched CSV + match reports.

Research-only: no recommendations, no predictions, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.prop_enrichment import run_prop_enrichment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich NBA player-prop snapshots for settlement.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = run_prop_enrichment(PROJECT_ROOT)

    print(f"Enrichment {summary['enrichment_version']} @ {summary['generated_at_utc']}")
    print(f"  Total snapshots:      {summary['total_snapshots']}")
    print(f"  NBA snapshots:        {summary['nba_snapshots']}")
    print(
        f"  player_id matched:    {summary['player_id_matched']} "
        f"({summary['player_id_match_rate']:.1%})"
    )
    print(
        f"  game_key matched:     {summary['game_key_matched']} "
        f"({summary['game_key_match_rate']:.1%})"
    )
    print(f"  Settlement-ready:     {summary['settlement_ready']}")
    print(f"  Settled:              {summary['settled']}")
    print(f"  Pending result:       {summary['pending']}")
    print(f"  Unmatched players:    {summary['unmatched_player_names']}")
    print(f"  Unmatched game keys:  {summary['unmatched_game_keys']}")
    print(f"  Enriched CSV: {summary['outputs']['enriched_path']}")
    print("Research-only: no recommendations were produced; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
