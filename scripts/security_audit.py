"""Check that secret-looking material is not in normal project files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from security_audit import save_security_audit_outputs, scan_for_sensitive_material  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local secret hygiene audit without printing secret values.")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    output_path = Path(args.output_path) if args.output_path else reports_dir / "security_audit_findings.csv"
    summary_path = Path(args.summary_path) if args.summary_path else reports_dir / "security_audit_summary.json"
    findings, summary = scan_for_sensitive_material(root)
    save_security_audit_outputs(findings, summary, output_path, summary_path)

    print(f"Security audit status: {summary.get('status', 'n/a')}")
    print(f"Files scanned: {summary.get('files_scanned', 0):,}")
    print(f"Findings outside ignored secret locations: {summary.get('findings', 0):,}")
    print(f"Rotation recommended: {summary.get('rotation_recommended', True)}")
    print(f"Saved findings to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
