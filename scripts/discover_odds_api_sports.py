"""List The Odds API's available sports and verify configured sport keys.

Writes data/reports/odds_api_available_sports.json. The /sports endpoint is
free (0 credits), so this is safe to run any time ODDS_API_KEY is set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.prop_collection import load_prop_collection_config  # noqa: E402
from data.odds_api_sports import discover_available_sports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover available Odds API sport keys.")
    parser.add_argument("--config", default=None, help="Path to prop_collection.yaml")
    parser.add_argument("--active-only", action="store_true", help="Only list currently active sports")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config" / "prop_collection.yaml"
    config = load_prop_collection_config(config_path)

    sources_cfg = config.get("sources") or {}
    api_key_env = (sources_cfg.get("odds_api") or {}).get("api_key_env", "ODDS_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"{api_key_env} is not set; cannot query The Odds API.")
        sys.exit(1)

    summary = discover_available_sports(
        config, PROJECT_ROOT, api_key, include_inactive=not args.active_only
    )
    print(f"Sports available: {summary['sports_count']} ({summary['soccer_count']} soccer)")
    for entry in summary["configured_sport_keys"]:
        flag = "ok" if entry["available"] else "NOT FOUND"
        active = "active" if entry["active"] else "inactive/off-season"
        print(f"  {entry['league']:<12} {entry['sport_key']:<28} {flag} ({active})")
    print(f"Wrote: {summary['output_path']}")
    if summary.get("output_md_path"):
        print(f"Wrote: {summary['output_md_path']}")
    if summary.get("configured_not_available"):
        print(f"WARNING: configured but not available: {summary['configured_not_available']}")


if __name__ == "__main__":
    main()
