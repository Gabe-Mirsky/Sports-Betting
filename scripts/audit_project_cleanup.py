"""Run the safe project cleanup audit (audit-only by default).

Default run scans the project and writes reports without touching anything:
  - data/reports/project_cleanup_audit_summary.json
  - data/reports/project_cleanup_candidates.csv
  - data/reports/project_cleanup_audit.md

With --apply, items classified safe_to_delete are MOVED (never deleted) to
data/quarantine/project_cleanup/YYYYMMDD_HHMMSS/. Protected paths (src/,
scripts/, tests/, config/, data/raw/, data/processed/, data/reports/,
README.md, TODO.md, project state/implementation reports, .venv/, .git/, ...)
are never moved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from reports.project_cleanup_audit import write_cleanup_audit_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe project cleanup audit (audit-only by default).")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move safe_to_delete items to data/quarantine/project_cleanup/ (default: audit only)",
    )
    parser.add_argument("--reports-dir", default=None, help="Output directory (default data/reports)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    summary = write_cleanup_audit_reports(PROJECT_ROOT, reports_dir=args.reports_dir, apply=args.apply)

    counts = summary["counts"]
    status_counts = summary["status_counts"]
    print(f"Mode: {'APPLY (quarantine move)' if args.apply else 'AUDIT ONLY (nothing touched)'}")
    print(f"Files scanned: {summary['total_files_scanned']}")
    print(f"Folders scanned: {summary['total_folders_scanned']}")
    print(f"Empty files: {counts['empty_files']}")
    print(f"Empty folders: {counts['empty_folders']}")
    print(f"Duplicate groups: {counts['duplicate_groups']}")
    print(f"Generated cache folders: {counts['generated_cache_folders']}")
    print(f"Large files (>50 MB): {counts['large_files']}")
    print(
        "Status counts: "
        f"safe_to_delete={status_counts.get('safe_to_delete', 0)} "
        f"needs_review={status_counts.get('needs_review', 0)} "
        f"should_keep={status_counts.get('should_keep', 0)}"
    )
    print(f"Estimated cleanup size: {summary['estimated_cleanup_bytes'] / (1024 * 1024):.1f} MB")
    apply_info = summary.get("apply")
    if apply_info:
        print(f"Moved to quarantine: {apply_info['moved_count']} items")
        print(f"Quarantine folder: {apply_info['quarantine_dir'] or '(nothing moved)'}")
        for item in apply_info["skipped"]:
            print(f"  skipped {item['path']}: {item['reason']}")
    print(f"Recommended next action: {summary['recommended_next_action']}")
    for key, path in summary["outputs"].items():
        print(f"Wrote: {path}")
    print("Nothing is ever permanently deleted by this tool; --apply only moves to quarantine.")


if __name__ == "__main__":
    main()
