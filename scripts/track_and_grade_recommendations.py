"""Snapshot and grade recommendation outputs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reports.recommendation_tracker import (  # noqa: E402
    build_recommendation_performance_summary,
    grade_parlay_recommendations,
    grade_single_recommendations,
    save_recommendation_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot and grade recommendation outputs.")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--fair-price-signals-path", default=None)
    parser.add_argument("--research-parlay-path", default=None)
    parser.add_argument("--paper-parlay-path", default=None)
    parser.add_argument("--settled-results-path", default=None)
    return parser.parse_args()


def _snapshot_folders(snapshot_dir: Path) -> list[Path]:
    if not snapshot_dir.exists():
        return []
    return sorted(path for path in snapshot_dir.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir) if args.reports_dir else PROJECT_ROOT / "data" / "reports"
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else reports_dir / "recommendation_snapshots"
    run_date = args.run_date or datetime.now().date().isoformat()
    fair_price_path = Path(args.fair_price_signals_path) if args.fair_price_signals_path else reports_dir / "fair_price_signals.csv"
    research_parlay_path = Path(args.research_parlay_path) if args.research_parlay_path else reports_dir / "research_parlay_candidates.csv"
    paper_parlay_path = Path(args.paper_parlay_path) if args.paper_parlay_path else reports_dir / "paper_parlay_candidates.csv"
    settled_results_path = Path(args.settled_results_path) if args.settled_results_path else reports_dir / "backtest_trades.csv"

    manifest = save_recommendation_snapshot(
        fair_price_path,
        research_parlay_path,
        paper_parlay_path,
        snapshot_dir,
        run_date=run_date,
    )

    graded_singles = []
    graded_research_parlays = []
    graded_paper_parlays = []
    for folder in _snapshot_folders(snapshot_dir):
        single_path = folder / fair_price_path.name
        if single_path.exists():
            singles = grade_single_recommendations(single_path, settled_results_path)
            graded_singles.append(singles)
        else:
            singles = pd.DataFrame()
        research_path = folder / research_parlay_path.name
        if research_path.exists():
            graded_research_parlays.append(grade_parlay_recommendations(research_path, singles))
        paper_path = folder / paper_parlay_path.name
        if paper_path.exists():
            graded_paper_parlays.append(grade_parlay_recommendations(paper_path, singles))

    single_frame = pd.concat(graded_singles, ignore_index=True) if graded_singles else pd.DataFrame()
    research_frame = pd.concat(graded_research_parlays, ignore_index=True) if graded_research_parlays else pd.DataFrame()
    paper_frame = pd.concat(graded_paper_parlays, ignore_index=True) if graded_paper_parlays else pd.DataFrame()
    summary = build_recommendation_performance_summary(single_frame, research_frame, paper_frame, reports_dir)

    print(f"Saved recommendation snapshot: {manifest['snapshot_id']}")
    print(f"Snapshot manifest: {manifest['manifest_path']}")
    print(f"Graded single recommendations: {summary.get('graded_rows', 0):,}")
    print(f"Research leans: {summary.get('total_research_leans', 0):,}")
    print(f"Paper candidates: {summary.get('total_paper_candidates', 0):,}")
    print(f"Saved performance summary to: {reports_dir / 'recommendation_performance_summary.json'}")


if __name__ == "__main__":
    main()
