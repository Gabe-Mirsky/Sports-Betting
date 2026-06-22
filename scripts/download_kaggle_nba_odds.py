"""Download the free Kaggle NBA odds dataset with KaggleHub."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "christophertreasure/nba-odds-data"
SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download free Kaggle NBA odds files for local sportsbook ingestion.")
    parser.add_argument("--dataset", default=DATASET_SLUG)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "raw" / "sportsbook" / "kaggle"))
    return parser.parse_args()


def _looks_like_nba_odds_file(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    text = " ".join(path.parts).lower()
    return "nba" in text or "basketball" in text or "odds" in text


def main() -> int:
    args = parse_args()
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed. Install it with: python -m pip install kagglehub")
        return 1

    dataset_path = Path(kagglehub.dataset_download(args.dataset))
    print(f"Downloaded Kaggle dataset path: {dataset_path}")

    all_files = sorted(path for path in dataset_path.rglob("*") if path.is_file())
    print("Files in downloaded dataset:")
    for path in all_files:
        print(f"- {path}")

    relevant_files = [path for path in all_files if _looks_like_nba_odds_file(path)]
    print("\nUsable-looking NBA odds files:")
    for path in relevant_files:
        print(f"- {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in relevant_files:
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied += 1
        print(f"Copied: {source} -> {destination}")

    print(f"Copied {copied:,} file(s) to: {output_dir}")
    if copied == 0:
        print("WARNING: No CSV/XLS/XLSX files looked like NBA odds files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
