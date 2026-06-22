"""Audit free / free-tier data sources for NBA player-prop modeling.

Writes a human-readable markdown report plus machine-readable CSV/JSON tables:

    data/reports/data_source_audit.md
    data/reports/data_source_audit.csv
    data/reports/data_source_field_coverage.csv
    data/reports/data_source_audit_summary.json

This is a planning artifact only. It does not download data, build the model,
change proof gates, or approve any betting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.source_audit import (  # noqa: E402
    build_audit_summary,
    build_field_coverage_table,
    build_source_audit_table,
    render_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NBA player-prop data sources.")
    parser.add_argument("--reports-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_table = build_source_audit_table()
    coverage_table = build_field_coverage_table()
    summary = build_audit_summary()
    markdown = render_markdown_report()

    audit_table.to_csv(reports_dir / "data_source_audit.csv", index=False)
    coverage_table.to_csv(reports_dir / "data_source_field_coverage.csv", index=False)
    (reports_dir / "data_source_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "data_source_audit.md").write_text(markdown, encoding="utf-8")

    print(f"Sources reviewed: {summary['sources_reviewed']}")
    print(f"Fields needed: {summary['fields_needed']}")
    print(f"Player-prop sources: {', '.join(summary['player_prop_sources'])}")
    print(f"Historical-odds sources: {', '.join(summary['historical_odds_sources'])}")
    print(f"Closing-price sources: {', '.join(summary['closing_price_sources'])}")
    print(f"Saved audit to: {reports_dir / 'data_source_audit.md'}")


if __name__ == "__main__":
    main()
