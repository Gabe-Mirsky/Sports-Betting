"""Profile candidate Kaggle datasets and write inspection reports.

By default this inspects the four known candidate slugs but does NOT download
them (so it is safe offline / without Kaggle credentials). Provide local folders
with --path, or pass --download to fetch slugs via kagglehub when available.

Outputs (under data/reports/):
    kaggle_dataset_profile_summary.json
    kaggle_dataset_file_inventory.csv
    kaggle_dataset_column_inventory.csv
    kaggle_dataset_recommendations.md

Inspection only. Does not change proof gates, betting, or parlays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kaggle_dataset_profiler import profile_datasets, write_reports  # noqa: E402


CANDIDATE_SLUGS = [
    "oliviersportsdata/us-sports-master-historical-closing-odds",
    "austro/beat-the-bookie-worldwide-football-dataset",
    "zachht/wnba-odds-history",
    "ehallmar/nba-historical-stats-and-betting-data",
    "ryancasey2/nba-money-line-betting-model-2025-2026",
    "isfakiqbalchowdhuruy/fifa-mens-world-cup-dataset-1970-2022",
    "clemendes/football-stats-top-5-leagues-2018-to-2025",
    "ritika027/real-time-sports-odds-data-multiple-bookmakers",
    "obiguy/soccer-odds-data",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile candidate Kaggle datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="A Kaggle slug or local folder path. Repeatable. Defaults to the candidate slugs.",
    )
    parser.add_argument("--path", action="append", default=None, help="Alias for --dataset (local folder).")
    parser.add_argument("--download", action="store_true", help="Download slugs via kagglehub when available.")
    parser.add_argument("--local-root", default=None, help="Directory to look for already-downloaded slug folders.")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--sample-rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identifiers = (args.dataset or []) + (args.path or [])
    if not identifiers:
        identifiers = CANDIDATE_SLUGS
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"

    profiles = profile_datasets(
        identifiers,
        download=args.download,
        local_root=args.local_root,
        sample_rows=args.sample_rows,
    )
    paths = write_reports(profiles, reports_dir)

    print(f"Profiled {len(profiles)} dataset(s):")
    for profile in profiles:
        sports = ", ".join(profile.detected_sports) or "unknown"
        print(
            f"  - {profile.identifier}: status={profile.status} sport={sports} "
            f"class={profile.classification} rec={profile.recommendation}"
        )
    print(f"Wrote summary:      {paths['summary']}")
    print(f"Wrote file inv:     {paths['file_inventory']}")
    print(f"Wrote column inv:   {paths['column_inventory']}")
    print(f"Wrote recommend md: {paths['recommendations']}")


if __name__ == "__main__":
    main()
