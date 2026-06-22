from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.project_cleanup_audit import (  # noqa: E402
    AUDIT_OUTPUT_FILES,
    NEEDS_REVIEW,
    SAFE_TO_DELETE,
    is_protected,
    run_cleanup_audit,
    write_cleanup_audit_reports,
)


def _build_fixture_project(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "module.py").write_text("print('keep me')\n", encoding="utf-8")
    (root / "src" / "empty_in_src.py").write_text("", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "gen.py").write_text("# writes my_report.csv\n", encoding="utf-8")
    (root / "README.md").write_text("# readme\n", encoding="utf-8")
    (root / "empty.txt").write_text("", encoding="utf-8")
    (root / "empty_folder").mkdir()
    (root / "dup_a.bin").write_bytes(b"same-bytes-here")
    (root / "dup_b.bin").write_bytes(b"same-bytes-here")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "module.cpython-311.pyc").write_bytes(b"pyc")
    (root / "old_backup_notes.txt").write_text("scratch notes\n", encoding="utf-8")


def _candidate_map(summary: dict) -> dict[str, list[dict]]:
    candidates: dict[str, list[dict]] = {}
    for candidate in summary["candidates"]:
        candidates.setdefault(candidate["path"], []).append(candidate)
    return candidates


class TestCleanupAuditDetection(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _build_fixture_project(self.root)
        self.summary = run_cleanup_audit(self.root)
        self.by_path = _candidate_map(self.summary)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_detects_empty_files(self) -> None:
        empty = [c for c in self.by_path["empty.txt"] if c["category"] == "empty_file"]
        self.assertEqual(len(empty), 1)
        self.assertEqual(empty[0]["status"], SAFE_TO_DELETE)
        self.assertGreaterEqual(self.summary["counts"]["empty_files"], 2)

    def test_detects_empty_folders(self) -> None:
        folder = [c for c in self.by_path["empty_folder"] if c["category"] == "empty_folder"]
        self.assertEqual(len(folder), 1)
        self.assertEqual(folder[0]["status"], SAFE_TO_DELETE)
        self.assertEqual(self.summary["counts"]["empty_folders"], 1)

    def test_detects_duplicate_files(self) -> None:
        self.assertEqual(self.summary["counts"]["duplicate_groups"], 1)
        group = self.summary["duplicate_groups"][0]
        self.assertEqual(group["files"], ["dup_a.bin", "dup_b.bin"])
        dup = [c for c in self.by_path["dup_a.bin"] if c["category"] == "duplicate"]
        self.assertEqual(dup[0]["status"], NEEDS_REVIEW)

    def test_detects_cache_and_suspicious_names(self) -> None:
        cache = [c for c in self.by_path["__pycache__"] if c["category"] == "generated_cache"]
        self.assertEqual(cache[0]["status"], SAFE_TO_DELETE)
        suspicious = [c for c in self.by_path["old_backup_notes.txt"] if c["category"] == "suspicious_name"]
        self.assertEqual(suspicious[0]["status"], NEEDS_REVIEW)

    def test_protects_important_folders(self) -> None:
        self.assertTrue(is_protected("src/module.py"))
        self.assertTrue(is_protected("data/raw/prop_odds/file.json"))
        self.assertTrue(is_protected("data/processed/player_prop_snapshots_normalized.csv"))
        self.assertTrue(is_protected("README.md"))
        self.assertTrue(is_protected("PROJECT_STATE_REPORT_2026-06-09.md"))
        self.assertTrue(is_protected(".venv/lib/site-packages/x.py"))
        # The empty file inside src/ is reported but never safe_to_delete.
        in_src = [c for c in self.by_path["src/empty_in_src.py"] if c["category"] == "empty_file"]
        self.assertEqual(in_src[0]["status"], NEEDS_REVIEW)
        for candidate in self.summary["candidates"]:
            if candidate["path"].startswith(("src/", "scripts/")) or candidate["path"] in {"README.md"}:
                self.assertNotEqual(candidate["status"], SAFE_TO_DELETE, candidate["path"])


class TestCleanupAuditApply(unittest.TestCase):
    def test_default_run_is_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_fixture_project(root)
            summary = write_cleanup_audit_reports(root)

            self.assertTrue(summary["audit_only"])
            self.assertNotIn("apply", summary)
            # Nothing moved or deleted: every fixture path still exists.
            self.assertTrue((root / "empty.txt").exists())
            self.assertTrue((root / "empty_folder").exists())
            self.assertTrue((root / "__pycache__").exists())
            self.assertTrue((root / "dup_a.bin").exists())
            self.assertFalse((root / "data" / "quarantine").exists())
            for filename in AUDIT_OUTPUT_FILES.values():
                self.assertTrue((root / "data" / "reports" / filename).exists(), filename)

    def test_apply_moves_to_quarantine_instead_of_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_fixture_project(root)
            summary = write_cleanup_audit_reports(root, apply=True)

            apply_info = summary["apply"]
            self.assertGreaterEqual(apply_info["moved_count"], 3)  # empty file, empty folder, cache
            quarantine = Path(apply_info["quarantine_dir"])
            self.assertTrue(quarantine.exists())
            self.assertTrue(str(quarantine).startswith(str(root / "data" / "quarantine" / "project_cleanup")))

            # Originals are gone but preserved (moved, not deleted) in quarantine.
            self.assertFalse((root / "empty.txt").exists())
            self.assertTrue((quarantine / "empty.txt").exists())
            self.assertFalse((root / "__pycache__").exists())
            self.assertTrue((quarantine / "__pycache__" / "module.cpython-311.pyc").exists())

            # Protected and needs_review items are untouched.
            self.assertTrue((root / "src" / "module.py").exists())
            self.assertTrue((root / "src" / "empty_in_src.py").exists())
            self.assertTrue((root / "dup_a.bin").exists())
            self.assertTrue((root / "dup_b.bin").exists())
            self.assertTrue((root / "old_backup_notes.txt").exists())
            self.assertTrue((root / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
