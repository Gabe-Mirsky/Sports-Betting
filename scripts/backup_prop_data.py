"""Create a timestamped backup of the prop-collection data and reports.

Backs up (when present):
  - data/processed/player_prop_snapshots_normalized.csv
  - data/processed/player_prop_snapshots_enriched.csv
  - prop-related reports under data/reports (player_prop_*, prop_collection_*,
    nba_prop_*, next_action_report.*, odds_api_quota_report.*,
    all_sports_prop_readiness.*, full_prop_pipeline_summary.json)
  - run logs: data/logs/prop_collection_runs/ and data/logs/full_prop_pipeline/
  - config/prop_collection.yaml

Output: data/backups/prop_data/YYYYMMDD_HHMMSS/ plus a backup_manifest.json
with per-file sizes and SHA-256 hashes. Never deletes anything; raw data and
alternate lines are untouched. Excludes .venv and cache folders by design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = PROJECT_ROOT / "data" / "backups" / "prop_data"

PROCESSED_FILES = [
    "data/processed/player_prop_snapshots_normalized.csv",
    "data/processed/player_prop_snapshots_enriched.csv",
]
CONFIG_FILES = [
    "config/prop_collection.yaml",
]
REPORT_GLOBS = [
    "player_prop_*.json", "player_prop_*.md", "player_prop_*.csv",
    "prop_collection_*.json", "prop_collection_*.md", "prop_collection_*.jsonl",
    "nba_prop_*.json", "nba_prop_*.md", "nba_prop_*.csv",
    "nba_main_lines_review.csv", "nba_alt_lines_review.csv",
    "nba_bookmaker_comparison.csv", "nba_prop_board_latest.csv",
    "next_action_report.json", "next_action_report.md",
    "odds_api_quota_report.json", "odds_api_quota_report.md",
    "all_sports_prop_readiness.json", "all_sports_prop_readiness.md",
    "full_prop_pipeline_summary.json",
]
LOG_DIRS = [
    "data/logs/prop_collection_runs",
    "data/logs/full_prop_pipeline",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_backup_files() -> list[Path]:
    """Absolute paths of every existing file that belongs in a backup."""
    files: list[Path] = []
    for rel in PROCESSED_FILES + CONFIG_FILES:
        path = PROJECT_ROOT / rel
        if path.exists():
            files.append(path)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    seen: set[Path] = set()
    for pattern in REPORT_GLOBS:
        for path in reports_dir.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    for rel in LOG_DIRS:
        log_dir = PROJECT_ROOT / rel
        if log_dir.exists():
            files.extend(p for p in log_dir.rglob("*") if p.is_file())
    return files


def create_backup(destination: Path, label: str = "manual") -> dict:
    """Copy all prop data files into ``destination`` and write a manifest."""
    files = collect_backup_files()
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    total_bytes = 0
    for source in files:
        rel = source.relative_to(PROJECT_ROOT)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        size = source.stat().st_size
        total_bytes += size
        entries.append({
            "path": str(rel).replace("\\", "/"),
            "size_bytes": size,
            "sha256": _sha256(source),
        })
    manifest = {
        "report": "prop_data_backup_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "project_root": str(PROJECT_ROOT),
        "backup_dir": str(destination),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
        "research_only": True,
        "notes": [
            "Backups never delete anything. Raw data under data/raw/ is large and "
            "append-only; it is not duplicated here (snapshots CSVs are the "
            "normalized source of truth).",
        ],
    }
    (destination / "backup_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up prop collection data and reports.")
    parser.add_argument("--label", default="manual", help="Label stored in the manifest.")
    parser.add_argument("--list", action="store_true", help="List existing backups and exit.")
    args = parser.parse_args()

    if args.list:
        if not BACKUP_ROOT.exists():
            print("No backups yet.")
            return 0
        for entry in sorted(BACKUP_ROOT.iterdir()):
            manifest_path = entry / "backup_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                print(
                    f"{entry.name}: {manifest['file_count']} files, "
                    f"{manifest['total_bytes'] / (1024 * 1024):.1f} MB, label={manifest['label']}"
                )
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / stamp
    manifest = create_backup(destination, label=args.label)

    print(f"Backup created: {destination.relative_to(PROJECT_ROOT)}")
    print(f"  Files: {manifest['file_count']}")
    print(f"  Size:  {manifest['total_bytes'] / (1024 * 1024):.1f} MB")
    print(f"  Manifest: {Path(manifest['backup_dir']).relative_to(PROJECT_ROOT) / 'backup_manifest.json'}")
    print("Restore with: scripts/restore_prop_data_backup.py --backup-path <backup dir> [--dry-run]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
