"""Safe, conservative project cleanup audit.

Scans the project tree and reports empty files/folders, duplicate files,
generated caches, stale temp/debug files, old logs, oversized files, and
possibly-stale generated reports. Every finding is classified as
``safe_to_delete``, ``needs_review``, or ``should_keep``.

Safety rules:
  - Audit-only by default: the first run never deletes or moves anything.
  - ``apply=True`` moves ONLY ``safe_to_delete`` items into a timestamped
    quarantine folder (data/quarantine/project_cleanup/YYYYMMDD_HHMMSS/);
    nothing is ever permanently deleted by this module.
  - Protected paths (source, tests, configs, raw/processed data, reports,
    dashboards, .git, .venv, ...) are never marked safe_to_delete and are
    re-checked at move time as a hard guard.
  - .git and .venv are never scanned for candidates; only their sizes are
    reported.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_TO_DELETE = "safe_to_delete"
NEEDS_REVIEW = "needs_review"
SHOULD_KEEP = "should_keep"

# Directories never scanned for candidates (size-only reporting).
SKIP_SCAN_DIRS = {".git", ".venv", "venv", "node_modules"}

# Generated cache directories: regenerable, safe to quarantine wholesale.
CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".pip_tmp",
}

# Paths (relative to project root, posix-style) that must never be deleted.
PROTECTED_DIRS = (
    "src",
    "scripts",
    "tests",
    "config",
    "data/raw",
    "data/processed",
    "data/reports",
    "data/templates",
    "outputs",
    ".secrets",
)

PROTECTED_FILES = (
    "README.md",
    "TODO.md",
    "config.yaml",
    "requirements.txt",
    ".gitignore",
)

# Root-level report files that must be kept (project state / implementation
# reports, cleanup reports, recommendation snapshots).
PROTECTED_NAME_RE = re.compile(
    r"^(PROJECT_STATE_REPORT|IMPLEMENTATION_REPORT|CLEANUP_REPORT|README|TODO)", re.IGNORECASE
)

SUSPICIOUS_NAME_RE = re.compile(
    r"(?:^|[_\-. ])(temp|tmp|debug|old|backup|copy|scratch|junk)(?:[_\-. 0-9]|$)", re.IGNORECASE
)

LARGE_FILE_BYTES = 50 * 1024 * 1024
OLD_LOG_DAYS = 30
STALE_REPORT_DAYS = 45
MAX_HASH_BYTES = 200 * 1024 * 1024
MAX_DUP_FILES_PER_GROUP = 25
MAX_DUP_FILES_LISTED = 20

QUARANTINE_REL = Path("data") / "quarantine"

AUDIT_OUTPUT_FILES = {
    "summary_json": "project_cleanup_audit_summary.json",
    "candidates_csv": "project_cleanup_candidates.csv",
    "summary_md": "project_cleanup_audit.md",
}


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_protected(rel_posix: str) -> bool:
    """True when a relative path must never be marked safe_to_delete."""

    name = rel_posix.rsplit("/", maxsplit=1)[-1]
    if PROTECTED_NAME_RE.match(name):
        return True
    if rel_posix in PROTECTED_FILES:
        return True
    for protected in PROTECTED_DIRS + (".git", ".venv"):
        if rel_posix == protected or rel_posix.startswith(protected + "/"):
            return True
    return False


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        for filename in filenames:
            try:
                total += (Path(dirpath) / filename).stat().st_size
            except OSError:
                continue
    return total


def _file_md5(path: Path) -> str | None:
    digest = hashlib.md5()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _load_generator_texts(root: Path) -> str:
    """Concatenate scripts/ and src/ python sources to detect regenerable reports."""

    pieces: list[str] = []
    for folder in (root / "scripts", root / "src"):
        if not folder.exists():
            continue
        for path in folder.rglob("*.py"):
            try:
                pieces.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(pieces)


def _candidate(
    rel: str,
    kind: str,
    category: str,
    status: str,
    size: int,
    modified_utc: str | None,
    reason: str,
    duplicate_group: str = "",
) -> dict[str, Any]:
    return {
        "path": rel,
        "kind": kind,
        "category": category,
        "status": status,
        "size_bytes": int(size),
        "modified_utc": modified_utc or "",
        "reason": reason,
        "duplicate_group": duplicate_group,
    }


def run_cleanup_audit(project_root: str | Path, now: datetime | None = None) -> dict[str, Any]:
    """Scan the project and return the audit summary + candidate list (no changes)."""

    root = Path(project_root).resolve()
    now = now or datetime.now(timezone.utc)
    generator_text = _load_generator_texts(root)

    candidates: list[dict[str, Any]] = []
    files_scanned = 0
    folders_scanned = 0
    skipped_dir_sizes: dict[str, int] = {}
    all_files: list[tuple[str, Path, int]] = []  # (rel, path, size)
    dir_children: dict[str, list[tuple[str, int]]] = {}
    largest: list[dict[str, Any]] = []

    quarantine_rel = QUARANTINE_REL.as_posix()
    reports_rel = "data/reports"

    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        current = Path(dirpath)
        rel_dir = "" if current == root else _rel_posix(current, root)

        pruned = []
        for dirname in list(dirnames):
            child = current / dirname
            child_rel = _rel_posix(child, root)
            if dirname in SKIP_SCAN_DIRS and (current == root):
                skipped_dir_sizes[child_rel] = _dir_size(child)
                pruned.append(dirname)
            elif child_rel == quarantine_rel or child_rel.startswith(quarantine_rel + "/"):
                pruned.append(dirname)
            elif dirname in CACHE_DIR_NAMES:
                size = _dir_size(child)
                candidates.append(
                    _candidate(
                        child_rel, "folder", "generated_cache", SAFE_TO_DELETE, size, None,
                        f"generated cache folder ({dirname}); recreated automatically",
                    )
                )
                pruned.append(dirname)
        for dirname in pruned:
            dirnames.remove(dirname)

        folders_scanned += 1
        entries: list[tuple[str, int]] = []
        for filename in filenames:
            path = current / filename
            try:
                stat = path.stat()
            except OSError:
                rel = (rel_dir + "/" if rel_dir else "") + filename
                candidates.append(
                    _candidate(rel, "file", "unreadable", NEEDS_REVIEW, 0, None, "could not stat file (permission denied?)")
                )
                continue
            rel = _rel_posix(path, root)
            files_scanned += 1
            size = stat.st_size
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            age_days = (now - mtime).days
            modified = mtime.isoformat()
            entries.append((rel, size))
            all_files.append((rel, path, size))
            largest.append({"path": rel, "size_bytes": size, "modified_utc": modified})
            protected = is_protected(rel)

            if size == 0:
                status = NEEDS_REVIEW if protected else SAFE_TO_DELETE
                reason = "empty file (0 bytes)"
                if protected:
                    reason += "; under a protected path, review manually"
                candidates.append(_candidate(rel, "file", "empty_file", status, 0, modified, reason))
                continue

            if size > LARGE_FILE_BYTES:
                status = SHOULD_KEEP if protected else NEEDS_REVIEW
                candidates.append(
                    _candidate(
                        rel, "file", "large_file", status, size, modified,
                        f"file larger than 50 MB ({size / (1024 * 1024):.1f} MB)",
                    )
                )

            if SUSPICIOUS_NAME_RE.search(filename) and not protected:
                candidates.append(
                    _candidate(
                        rel, "file", "suspicious_name", NEEDS_REVIEW, size, modified,
                        "name suggests temp/debug/old/backup/copy file; confirm before removing",
                    )
                )

            if rel.startswith("data/logs/") and filename.endswith((".log", ".txt")) and age_days > OLD_LOG_DAYS:
                candidates.append(
                    _candidate(
                        rel, "file", "old_log", NEEDS_REVIEW, size, modified,
                        f"log file older than {OLD_LOG_DAYS} days ({age_days} days old)",
                    )
                )

            if rel.startswith(reports_rel + "/") and age_days > STALE_REPORT_DAYS:
                regenerable = filename in generator_text
                candidates.append(
                    _candidate(
                        rel, "file", "stale_report", NEEDS_REVIEW, size, modified,
                        (
                            f"report not refreshed in {age_days} days; "
                            + (
                                "a script references it so it is regenerable, but keep unless re-built"
                                if regenerable
                                else "no generating script found, keep until reviewed"
                            )
                        ),
                    )
                )

        dir_children[rel_dir] = entries
        if not filenames and not dirnames and not pruned and current != root:
            rel = _rel_posix(current, root)
            status = NEEDS_REVIEW if is_protected(rel) else SAFE_TO_DELETE
            candidates.append(
                _candidate(rel, "folder", "empty_folder", status, 0, None, "empty folder (no files or subfolders)")
            )

    # Folders whose files are all empty (and that have files).
    for rel_dir, entries in dir_children.items():
        if not rel_dir or not entries:
            continue
        if all(size == 0 for _rel, size in entries):
            status = NEEDS_REVIEW
            candidates.append(
                _candidate(
                    rel_dir, "folder", "empty_only_folder", status, 0, None,
                    f"folder contains only empty files ({len(entries)} files, all 0 bytes)",
                )
            )

    # Duplicate detection: group by size first, hash only potential matches.
    by_size: dict[int, list[tuple[str, Path]]] = {}
    for rel, path, size in all_files:
        if 0 < size <= MAX_HASH_BYTES:
            by_size.setdefault(size, []).append((rel, path))
    duplicate_groups: list[dict[str, Any]] = []
    group_index = 0
    for size, group in sorted(by_size.items(), key=lambda kv: -kv[0]):
        if len(group) < 2:
            continue
        by_hash: dict[str, list[str]] = {}
        for rel, path in group:
            digest = _file_md5(path)
            if digest:
                by_hash.setdefault(digest, []).append(rel)
        for digest, paths in by_hash.items():
            if len(paths) < 2:
                continue
            group_index += 1
            group_id = f"dup{group_index:03d}"
            sorted_paths = sorted(paths)
            duplicate_groups.append(
                {
                    "group": group_id,
                    "md5": digest,
                    "size_bytes": size,
                    "file_count": len(sorted_paths),
                    "files": sorted_paths[:MAX_DUP_FILES_LISTED],
                    "files_truncated": max(0, len(sorted_paths) - MAX_DUP_FILES_LISTED),
                }
            )
            if len(sorted_paths) > MAX_DUP_FILES_PER_GROUP:
                # Huge groups (e.g. thousands of identical raw API responses)
                # get one summary row so the CSV stays reviewable.
                parents = {rel.rsplit("/", 1)[0] if "/" in rel else "." for rel in sorted_paths}
                parent = parents.pop() if len(parents) == 1 else "(multiple folders)"
                status = SHOULD_KEEP if all(is_protected(rel) for rel in sorted_paths) else NEEDS_REVIEW
                candidates.append(
                    _candidate(
                        parent, "group", "duplicate", status, size * len(sorted_paths), None,
                        f"{len(sorted_paths)} identical files in this folder (group {group_id}, "
                        f"{size} bytes each); listed once to keep this report readable",
                        duplicate_group=group_id,
                    )
                )
                continue
            for rel in sorted_paths:
                status = SHOULD_KEEP if is_protected(rel) else NEEDS_REVIEW
                candidates.append(
                    _candidate(
                        rel, "file", "duplicate", status, size, None,
                        f"identical content shared by {len(sorted_paths)} files (group {group_id}); keep at least one",
                        duplicate_group=group_id,
                    )
                )

    largest.sort(key=lambda item: -item["size_bytes"])
    top_largest = largest[:20]

    status_counts = {SAFE_TO_DELETE: 0, NEEDS_REVIEW: 0, SHOULD_KEEP: 0}
    category_counts: dict[str, int] = {}
    estimated_cleanup_bytes = 0
    for candidate in candidates:
        status_counts[candidate["status"]] = status_counts.get(candidate["status"], 0) + 1
        category_counts[candidate["category"]] = category_counts.get(candidate["category"], 0) + 1
        if candidate["status"] == SAFE_TO_DELETE:
            estimated_cleanup_bytes += candidate["size_bytes"]

    if status_counts[SAFE_TO_DELETE]:
        next_action = (
            "Review project_cleanup_candidates.csv, then re-run with --apply to move the "
            f"{status_counts[SAFE_TO_DELETE]} safe_to_delete items into data/quarantine/project_cleanup/ "
            "(nothing is permanently deleted)."
        )
    elif status_counts[NEEDS_REVIEW]:
        next_action = "No safe_to_delete items. Manually review the needs_review entries; nothing should be moved automatically."
    else:
        next_action = "Project is clean: no cleanup candidates found."

    return {
        "report": "project_cleanup_audit",
        "generated_at_utc": now.isoformat(),
        "project_root": str(root),
        "audit_only": True,
        "total_files_scanned": files_scanned,
        "total_folders_scanned": folders_scanned,
        "skipped_dirs": {rel: size for rel, size in sorted(skipped_dir_sizes.items())},
        "counts": {
            "empty_files": category_counts.get("empty_file", 0),
            "empty_folders": category_counts.get("empty_folder", 0),
            "empty_only_folders": category_counts.get("empty_only_folder", 0),
            "duplicate_groups": len(duplicate_groups),
            "generated_cache_folders": category_counts.get("generated_cache", 0),
            "large_files": category_counts.get("large_file", 0),
            "suspicious_names": category_counts.get("suspicious_name", 0),
            "old_logs": category_counts.get("old_log", 0),
            "stale_reports": category_counts.get("stale_report", 0),
        },
        "status_counts": status_counts,
        "estimated_cleanup_bytes": estimated_cleanup_bytes,
        "top_largest_files": top_largest,
        "duplicate_groups": duplicate_groups,
        "candidates": candidates,
        "recommended_next_action": next_action,
    }


def apply_quarantine(
    project_root: str | Path,
    candidates: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Move safe_to_delete candidates into a timestamped quarantine folder.

    Never deletes anything. Protected paths are re-checked here as a hard
    guard even if a candidate was mislabeled.
    """

    root = Path(project_root).resolve()
    now = now or datetime.now(timezone.utc)
    quarantine_dir = root / QUARANTINE_REL / "project_cleanup" / now.strftime("%Y%m%d_%H%M%S")

    moved: list[str] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate["status"] != SAFE_TO_DELETE:
            continue
        rel = candidate["path"]
        if is_protected(rel):
            skipped.append({"path": rel, "reason": "protected path; never moved"})
            continue
        source = root / Path(rel)
        if not source.exists():
            skipped.append({"path": rel, "reason": "no longer exists"})
            continue
        destination = quarantine_dir / Path(rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(destination))
        except OSError as error:
            skipped.append({"path": rel, "reason": f"move failed: {error}"})
            continue
        moved.append(rel)

    return {
        "applied": True,
        "quarantine_dir": str(quarantine_dir) if moved else "",
        "moved_count": len(moved),
        "moved": moved,
        "skipped": skipped,
    }


def _fmt_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} GB"


def _build_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    status_counts = summary["status_counts"]
    lines = [
        "# Project Cleanup Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "Audit-only report: nothing has been deleted or moved. Re-run with --apply to move",
        "safe_to_delete items into data/quarantine/project_cleanup/ (never permanent deletion).",
        "",
        "## Totals",
        "",
        f"- Files scanned: {summary['total_files_scanned']}",
        f"- Folders scanned: {summary['total_folders_scanned']}",
        f"- Empty files: {counts['empty_files']}",
        f"- Empty folders: {counts['empty_folders']}",
        f"- Folders with only empty files: {counts['empty_only_folders']}",
        f"- Duplicate groups: {counts['duplicate_groups']}",
        f"- Generated cache folders: {counts['generated_cache_folders']}",
        f"- Large files (>50 MB): {counts['large_files']}",
        f"- Suspicious names (temp/tmp/debug/old/backup/copy): {counts['suspicious_names']}",
        f"- Old logs (>{OLD_LOG_DAYS} days): {counts['old_logs']}",
        f"- Possibly stale reports (>{STALE_REPORT_DAYS} days): {counts['stale_reports']}",
        f"- Estimated cleanup size (safe_to_delete only): {_fmt_bytes(summary['estimated_cleanup_bytes'])}",
        "",
        "## Status Counts",
        "",
        f"- safe_to_delete: {status_counts.get(SAFE_TO_DELETE, 0)}",
        f"- needs_review: {status_counts.get(NEEDS_REVIEW, 0)}",
        f"- should_keep: {status_counts.get(SHOULD_KEEP, 0)}",
        "",
        "## Skipped Directories (size-only, never modified)",
        "",
    ]
    if summary["skipped_dirs"]:
        lines += [f"- {rel}: {_fmt_bytes(size)}" for rel, size in summary["skipped_dirs"].items()]
    else:
        lines.append("- (none)")
    lines += ["", "## Top 20 Largest Files", "", "| path | size | modified |", "| --- | --- | --- |"]
    for item in summary["top_largest_files"]:
        lines.append(f"| {item['path']} | {_fmt_bytes(item['size_bytes'])} | {item['modified_utc']} |")
    lines += ["", "## Duplicate File Groups", ""]
    if summary["duplicate_groups"]:
        for group in summary["duplicate_groups"]:
            lines.append(
                f"- {group['group']} ({group.get('file_count', len(group['files']))} files, "
                f"{_fmt_bytes(group['size_bytes'])} each):"
            )
            lines += [f"  - {path}" for path in group["files"]]
            if group.get("files_truncated"):
                lines.append(f"  - ... and {group['files_truncated']} more identical files")
    else:
        lines.append("- (no duplicate files found)")
    lines += ["", "## Recommended Next Action", "", summary["recommended_next_action"], ""]
    apply_info = summary.get("apply")
    if apply_info:
        lines += [
            "## Apply Result",
            "",
            f"- Moved to quarantine: {apply_info['moved_count']} items",
            f"- Quarantine folder: {apply_info['quarantine_dir'] or '(nothing moved)'}",
            "",
        ]
    return "\n".join(lines)


def write_cleanup_audit_reports(
    project_root: str | Path,
    reports_dir: str | Path | None = None,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the audit, optionally quarantine safe_to_delete items, write reports."""

    import pandas as pd

    root = Path(project_root).resolve()
    out_dir = Path(reports_dir) if reports_dir else root / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_cleanup_audit(root, now=now)
    if apply:
        summary["audit_only"] = False
        summary["apply"] = apply_quarantine(root, summary["candidates"], now=now)

    candidates = summary.pop("candidates")
    columns = ["path", "kind", "category", "status", "size_bytes", "modified_utc", "reason", "duplicate_group"]
    frame = pd.DataFrame(candidates, columns=columns).sort_values(["status", "category", "path"])

    outputs = {key: out_dir / filename for key, filename in AUDIT_OUTPUT_FILES.items()}
    frame.to_csv(outputs["candidates_csv"], index=False)
    summary["candidate_count"] = len(candidates)
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs["summary_md"].write_text(_build_markdown(summary), encoding="utf-8")
    summary["candidates"] = candidates
    return summary
