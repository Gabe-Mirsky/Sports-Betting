from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from security_audit import scan_for_sensitive_material  # noqa: E402


class TestSecurityAudit(unittest.TestCase):
    def test_scan_reports_secret_like_material_without_value(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_security_audit_bad"
        root.mkdir(parents=True, exist_ok=True)
        try:
            (root / "bad.txt").write_text("-----BEGIN RSA PRIVATE KEY-----\nsecret\n", encoding="utf-8")
            findings, summary = scan_for_sensitive_material(root, skip_dirs=set())
        finally:
            for child in root.glob("*"):
                child.unlink()
            root.rmdir()

        self.assertEqual(summary["status"], "fail")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings.loc[0, "pattern"], "rsa_private_key")
        self.assertNotIn("BEGIN RSA PRIVATE KEY", findings.to_string())

    def test_scan_ignores_secrets_folder(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_security_audit_ignored"
        root.mkdir(parents=True, exist_ok=True)
        try:
            secrets = root / ".secrets"
            secrets.mkdir()
            (secrets / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nsecret\n", encoding="utf-8")
            findings, summary = scan_for_sensitive_material(root)
        finally:
            for child in (root / ".secrets").glob("*"):
                child.unlink()
            (root / ".secrets").rmdir()
            root.rmdir()

        self.assertTrue(findings.empty)
        self.assertEqual(summary["status"], "pass")


if __name__ == "__main__":
    unittest.main()
