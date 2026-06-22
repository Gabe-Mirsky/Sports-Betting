"""Build the prop-collection health summary JSON + markdown (research-only).

Reads the run history (data/reports/prop_collection_run_history.jsonl, plus
older run logs), the normalized snapshot CSV, and the current config, then
writes:
  - data/reports/prop_collection_health_summary.json
  - data/reports/prop_collection_health.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.prop_collection import load_prop_collection_config  # noqa: E402
from data.prop_collection_health import write_health_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prop-collection health reports.")
    parser.add_argument("--config", default=None, help="Path to prop_collection.yaml")
    parser.add_argument("--recent-days", type=int, default=3, help="Stale-league threshold in days")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config" / "prop_collection.yaml"
    config = load_prop_collection_config(config_path)
    summary = write_health_reports(config, PROJECT_ROOT, recent_days=args.recent_days)

    runs = summary["runs"]
    print(f"Collection health: {'HEALTHY' if summary['healthy'] else 'UNHEALTHY'}")
    for reason in summary["health_reasons"]:
        print(f"  - {reason}")
    print(f"Last successful collection: {summary['last_successful_collection_utc'] or 'never'}")
    print(
        f"Runs: {runs['total']} total / {runs['successful']} successful / "
        f"{runs['failed']} failed / {runs['skipped']} skipped"
    )
    print(f"Days with no collection: {summary['missed_days_count']}")
    print(f"API key detected: {summary['api_key_detected']}")
    print(f"Likely quota issue: {summary['likely_quota_issue']}")
    print(f"Wrote: {summary['outputs']['summary_json']}")
    print(f"Wrote: {summary['outputs']['summary_md']}")
    print("Research-only: health reporting only; no recommendations.")


if __name__ == "__main__":
    main()
