"""Build the player-prop market quality audit reports (research-only).

Reads:
  - data/processed/player_prop_snapshots_enriched.csv (preferred when present)
  - data/processed/player_prop_snapshots_normalized.csv (fallback)

Writes (under data/reports/):
  - player_prop_market_quality_summary.json
  - player_prop_market_quality.md
  - player_prop_line_quality.csv
  - player_prop_likely_main_lines.csv
  - player_prop_possible_alt_lines.csv
  - player_prop_bookmaker_coverage.csv
  - player_prop_closing_snapshot_coverage.csv

No models, no recommendations, no proof-gate or betting changes.
Alternate lines are flagged for review only - nothing is deleted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.player_prop_market_quality import write_market_quality_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player-prop market quality audit reports.")
    parser.add_argument("--normalized", default=None, help="Path to player_prop_snapshots_normalized.csv")
    parser.add_argument("--enriched", default=None, help="Path to player_prop_snapshots_enriched.csv")
    parser.add_argument("--reports-dir", default=None, help="Output directory (default data/reports)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = write_market_quality_reports(
        PROJECT_ROOT,
        normalized_path=args.normalized,
        enriched_path=args.enriched,
        reports_dir=args.reports_dir,
    )

    closing = summary["closing_coverage"]
    print(f"Snapshots audited: {summary['total_snapshots']}")
    print(f"Markets audited: {summary['total_markets_audited']}")
    print("Markets by league:")
    for league, count in summary["markets_by_league"].items():
        print(f"  - {league}: {count}")
    print(f"Likely main lines: {summary['likely_main_lines']} "
          f"(confident: {summary['confident_main_lines']})")
    print(f"Possible alternate-line markets: {summary['possible_alt_line_markets']}")
    print(f"Wide line range warnings: {summary['wide_line_range_markets']}")
    print(f"Missing price warnings: {summary['missing_price_markets']}")
    print("Flag counts:")
    for flag, count in summary["flag_counts"].items():
        print(f"  - {flag}: {count}")
    print(f"Closing-like snapshots: {closing['total_closing_snapshots']}")
    print(f"Markets without closing snapshot: {closing['markets_without_closing']}")
    print(f"CLV readiness: {closing['clv_readiness_verdict']}")
    if summary["nba_best_bookmakers"]:
        best = summary["nba_best_bookmakers"][0]
        print(f"Best NBA bookmaker coverage: {best['bookmaker']} ({best['markets']} markets)")
    print(f"NBA clean enough for modeling later: {summary['nba_clean_enough_for_modeling']}")
    print(f"Verdict: {summary['nba_modeling_verdict']}")
    for key, path in summary["outputs"].items():
        print(f"Wrote: {path}")
    print("Research-only: market quality audit; no models, recommendations, or betting changes.")


if __name__ == "__main__":
    main()
