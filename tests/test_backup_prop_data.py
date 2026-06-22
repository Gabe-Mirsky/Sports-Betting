"""Tests for the prop-data backup system (temp-dir fixtures only)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import backup_prop_data  # noqa: E402


def _seed_fake_project(root: Path) -> None:
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "reports").mkdir(parents=True)
    (root / "data" / "logs" / "prop_collection_runs").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "data" / "processed" / "player_prop_snapshots_normalized.csv").write_text(
        "league,player_name\nNBA,Test Player\n", encoding="utf-8"
    )
    (root / "data" / "reports" / "player_prop_data_quality_gates.json").write_text(
        json.dumps({"status": "settlement_ready"}), encoding="utf-8"
    )
    (root / "data" / "logs" / "prop_collection_runs" / "run_x.log").write_text(
        "log line\n", encoding="utf-8"
    )
    (root / "config" / "prop_collection.yaml").write_text(
        "research_only: true\n", encoding="utf-8"
    )


class BackupTests(unittest.TestCase):
    def test_backup_copies_files_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _seed_fake_project(root)
            with mock.patch.object(backup_prop_data, "PROJECT_ROOT", root), \
                 mock.patch.object(backup_prop_data, "BACKUP_ROOT", root / "data" / "backups" / "prop_data"):
                destination = root / "data" / "backups" / "prop_data" / "test_stamp"
                manifest = backup_prop_data.create_backup(destination, label="unit-test")

            self.assertEqual(manifest["file_count"], 4)
            self.assertEqual(manifest["label"], "unit-test")
            self.assertTrue((destination / "backup_manifest.json").exists())
            copied = destination / "data" / "processed" / "player_prop_snapshots_normalized.csv"
            self.assertTrue(copied.exists())
            entries = {e["path"] for e in manifest["files"]}
            self.assertIn("config/prop_collection.yaml", entries)
            self.assertIn("data/logs/prop_collection_runs/run_x.log", entries)
            for entry in manifest["files"]:
                self.assertEqual(len(entry["sha256"]), 64)

    def test_backup_never_includes_venv_or_raw(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _seed_fake_project(root)
            (root / ".venv").mkdir()
            (root / ".venv" / "huge.bin").write_text("x" * 10, encoding="utf-8")
            (root / "data" / "raw" / "prop_odds").mkdir(parents=True)
            (root / "data" / "raw" / "prop_odds" / "raw.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(backup_prop_data, "PROJECT_ROOT", root):
                files = backup_prop_data.collect_backup_files()
            as_strings = [str(p) for p in files]
            self.assertFalse(any(".venv" in s for s in as_strings))
            self.assertFalse(any("raw" + "\\" + "prop_odds" in s or "raw/prop_odds" in s for s in as_strings))


if __name__ == "__main__":
    unittest.main()
