"""Validate the real-data input files for the no-odds matchup pipeline.

Example
-------
    python scripts/validate_matchup_input_files.py \
        --results-path data/processed/match_results.csv \
        --fixtures-path data/processed/fixtures_today.csv \
        --injuries-path data/processed/injuries.csv

Exits non-zero only on a true FAIL. Weak-but-usable data is reported as a
WARNING and still exits 0 (use --strict to treat warnings as failures).
No odds, closing lines, or market data are required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from quality.matchup_input_validation import (  # noqa: E402
    DEFAULT_ALIASES_PATH,
    STATUS_FAIL,
    build_validation_report,
    render_markdown_report,
    render_text_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate matchup prediction input files.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--fixtures-path", required=True)
    parser.add_argument("--injuries-path", default=None)
    parser.add_argument("--sport", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--aliases-path", default=DEFAULT_ALIASES_PATH)
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    report = build_validation_report(
        results_path=args.results_path,
        fixtures_path=args.fixtures_path,
        injuries_path=args.injuries_path,
        sport=args.sport,
        league=args.league,
        aliases_path=args.aliases_path,
        strict=args.strict,
    )

    text = render_text_report(report)
    print(text)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matchup_input_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "matchup_input_validation_report.md").write_text(
        render_markdown_report(report), encoding="utf-8"
    )

    return 1 if report["overall_status"] == STATUS_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
