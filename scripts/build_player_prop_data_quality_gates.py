"""Build the NBA player-prop data quality gates report (research-only).

Evaluates the readiness ladder (not_ready -> collection_ready ->
settlement_ready -> clv_ready -> modeling_experiment_ready) and writes:
  - data/reports/player_prop_data_quality_gates.json
  - data/reports/player_prop_data_quality_gates.md

These gates qualify DATA for research-only modeling experiments. They never
approve betting or parlays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.player_prop_data_quality_gates import write_data_quality_gate_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-prop data quality gates.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = write_data_quality_gate_reports(PROJECT_ROOT)

    print(f"Data quality gates: status = {summary['status']}")
    metrics = summary["metrics"]
    if metrics.get("nba_snapshots"):
        print(f"  NBA snapshots: {metrics['nba_snapshots']}")
        print(f"  Player match rate: {metrics['player_match_rate']:.1%}")
        print(f"  Main-line rate: {metrics['main_line_rate']:.1%}")
        print(f"  Settled props: {metrics['settled_props']}")
        print(f"  CLV markets: {metrics['clv_markets']}")
        print(f"  Closing market rate: {metrics['closing_market_rate']:.1%}")
    if summary["blockers"]:
        print("  Blockers:")
        for blocker in summary["blockers"]:
            print(f"    - {blocker}")
    for key, path in summary["outputs"].items():
        print(f"Wrote: {path}")
    print("Research-only: data gates only; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
