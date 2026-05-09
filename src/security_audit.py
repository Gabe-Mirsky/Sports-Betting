"""Local secret hygiene checks that never print secret contents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SECRET_PATTERNS = {
    "rsa_private_key": re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
    "kalshi_api_key_assignment": re.compile(r"KALSHI_(?:API_KEY|API_KEY_ID)\s*=\s*[0-9a-fA-F-]{20,}"),
    "kalshi_access_key_header": re.compile(r"KALSHI-ACCESS-KEY['\"]?\s*[:=]\s*['\"][0-9a-fA-F-]{20,}"),
}

DEFAULT_SKIP_DIRS = {
    ".git",
    ".secrets",
    ".venv",
    ".codex_runtime_pkgs",
    ".pip_tmp",
    "__pycache__",
    ".pytest_cache",
    "data",
}

DEFAULT_SKIP_FILES = {
    ".env",
}

PATTERN_FIXTURE_FILES = {
    "scripts/setup_kalshi_credentials.py",
    "tests/test_security_audit.py",
}


def _is_skipped(path: Path, root: Path, skip_dirs: set[str]) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    relative_text = relative.as_posix()
    if relative_text in PATTERN_FIXTURE_FILES or relative.name in DEFAULT_SKIP_FILES:
        return True
    parts = set(relative.parts)
    if any(part.startswith("tmp") for part in relative.parts):
        return True
    return bool(parts.intersection(skip_dirs))


def scan_for_sensitive_material(
    root: str | Path,
    skip_dirs: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scan text files for secret-like material without returning matched values."""

    root_path = Path(root)
    skips = set(DEFAULT_SKIP_DIRS if skip_dirs is None else skip_dirs)
    findings: list[dict[str, Any]] = []
    files_scanned = 0
    files_skipped = 0
    secret_locations = []

    for path in root_path.rglob("*"):
        if path.is_dir():
            continue
        if _is_skipped(path, root_path, skips):
            files_skipped += 1
            if ".secrets" in path.parts or path.name in DEFAULT_SKIP_FILES:
                secret_locations.append(str(path.relative_to(root_path)))
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".parquet", ".joblib", ".pyc"}:
            files_skipped += 1
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            files_skipped += 1
            continue
        files_scanned += 1
        for pattern_name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    {
                        "file": str(path.relative_to(root_path)),
                        "pattern": pattern_name,
                        "action": "move secret to .secrets/.env and rotate if it was shared",
                    }
                )

    findings_df = pd.DataFrame(findings, columns=["file", "pattern", "action"])
    summary = {
        "files_scanned": int(files_scanned),
        "files_skipped": int(files_skipped),
        "findings": int(len(findings_df)),
        "status": "pass" if findings_df.empty else "fail",
        "ignored_secret_locations_present": sorted(secret_locations),
        "rotation_recommended": True,
        "note": (
            "This scan excludes .env, .secrets, data, virtualenvs, and temp folders. "
            "It never prints secret contents. Rotate any key that was pasted into chat or stored in a non-ignored file."
        ),
    }
    return findings_df, summary


def save_security_audit_outputs(
    findings: pd.DataFrame,
    summary: dict[str, Any],
    findings_path: str | Path,
    summary_path: str | Path,
) -> None:
    findings_output = Path(findings_path)
    summary_output = Path(summary_path)
    findings_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(findings_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
