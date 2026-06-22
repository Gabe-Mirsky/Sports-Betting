"""Restore a prop-data backup created by scripts/backup_prop_data.py.

Safety rules:
  - requires an explicit --backup-path (no implicit "latest")
  - ALWAYS creates a pre-restore backup of the current files before
    overwriting anything (under data/backups/prop_data/prerestore_<stamp>/)
  - --dry-run shows exactly what would change without touching anything
  - verifies the manifest exists and only restores files listed in it
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from backup_prop_data import BACKUP_ROOT, _sha256, create_backup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore a prop data backup.")
    parser.add_argument("--backup-path", required=True,
                        help="Backup folder containing backup_manifest.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be restored without changing anything.")
    args = parser.parse_args()

    backup_dir = Path(args.backup_path)
    if not backup_dir.is_absolute():
        backup_dir = PROJECT_ROOT / backup_dir
    manifest_path = backup_dir / "backup_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no backup_manifest.json in {backup_dir} - not a valid backup.")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    planned = []
    for entry in manifest.get("files", []):
        rel = entry["path"]
        source = backup_dir / rel
        target = PROJECT_ROOT / rel
        if not source.exists():
            planned.append((rel, "MISSING_IN_BACKUP", None))
            continue
        if not target.exists():
            planned.append((rel, "create", source))
        elif _sha256(target) != entry.get("sha256"):
            planned.append((rel, "overwrite", source))
        else:
            planned.append((rel, "identical", None))

    changes = [p for p in planned if p[1] in {"create", "overwrite"}]
    missing = [p for p in planned if p[1] == "MISSING_IN_BACKUP"]

    print(f"Backup: {backup_dir}")
    print(f"  Manifest files: {len(planned)}")
    print(f"  To create:    {sum(1 for p in planned if p[1] == 'create')}")
    print(f"  To overwrite: {sum(1 for p in planned if p[1] == 'overwrite')}")
    print(f"  Identical:    {sum(1 for p in planned if p[1] == 'identical')}")
    if missing:
        print(f"  WARNING: {len(missing)} manifest file(s) missing from the backup folder:")
        for rel, _, _ in missing[:10]:
            print(f"    - {rel}")

    if args.dry_run:
        for rel, action, _ in changes[:50]:
            print(f"  [dry-run] {action}: {rel}")
        print("Dry run: nothing was changed.")
        return 0

    if not changes:
        print("Nothing to restore: current files already match the backup.")
        return 0

    # Pre-restore safety backup of whatever exists right now.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prerestore_dir = BACKUP_ROOT / f"prerestore_{stamp}"
    print(f"Creating pre-restore backup first: {prerestore_dir.relative_to(PROJECT_ROOT)}")
    pre_manifest = create_backup(prerestore_dir, label=f"pre-restore of {backup_dir.name}")
    print(f"  Pre-restore backup: {pre_manifest['file_count']} files saved.")

    restored = 0
    for rel, action, source in changes:
        target = PROJECT_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored += 1
        print(f"  [{action}] {rel}")
    print(f"Restored {restored} file(s) from {backup_dir.name}.")
    print(f"Previous state saved in {prerestore_dir.relative_to(PROJECT_ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
