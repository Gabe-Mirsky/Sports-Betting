"""Shared CLI for the free-backfill importer skeletons (research-only).

Every ``scripts/backfill_*`` skeleton is a thin wrapper over this runner so they
behave identically and, crucially, never touch the network on their own. The
real ``fetch()`` step is intentionally disabled until a source list and date
range is approved.

Modes:
    (default)        print the source plan; do nothing else.
    --input PATH     normalize an already-downloaded local CSV to staging (offline).
    --download       print the plan and exit non-zero (network is disabled here).
"""

from __future__ import annotations

import argparse
from typing import Callable, Sequence

import pandas as pd

from logging_setup import setup_logging
from data.free_backfill import STAGING_DIR, run_importer_from_input, write_staging_frame


def run_backfill_cli(
    *,
    source: str,
    sport: str,
    league: str,
    normalize_fn: Callable[[pd.DataFrame], pd.DataFrame],
    planned_sources: Sequence[str],
    date_range: str,
    data_types: Sequence[str],
    license_notes: str,
    fetch_fn: Callable[[argparse.Namespace], pd.DataFrame] | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=f"Free backfill importer: {source} ({league}).")
    parser.add_argument("--input", help="Local raw CSV (already downloaded by hand); normalize to staging offline.")
    parser.add_argument("--league", default=league, help="Override the league label for this run.")
    parser.add_argument("--out-dir", default=str(STAGING_DIR), help="Staging output directory.")
    parser.add_argument("--download", action="store_true",
                        help="Fetch from the (approved) source list and write staging.")
    parser.add_argument("--start-year", type=int, default=None, help="First season/year to fetch.")
    parser.add_argument("--end-year", type=int, default=None, help="Last season/year to fetch.")
    parser.add_argument("--max-events", type=int, default=None, help="Hard cap on quota-metered events (SGO).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    def _print_plan() -> None:
        print(f"[plan] source={source} sport={sport} league={args.league}")
        print(f"[plan] data_types={', '.join(data_types)}")
        print(f"[plan] date_range={date_range}")
        print(f"[plan] license={license_notes}")
        print("[plan] planned sources:")
        for item in planned_sources:
            print(f"        - {item}")

    if args.input:
        path, raw_rows = run_importer_from_input(
            input_path=args.input, normalize_fn=normalize_fn,
            source=source, sport=sport, league=args.league, out_dir=args.out_dir,
        )
        staged = pd.read_csv(path)
        print(f"[staged] {path}  (read {raw_rows} raw rows -> {len(staged)} staged games)")
        print("Research-only: no betting, predictions, parlays, or gate changes.")
        return 0

    if args.download:
        if fetch_fn is None:
            _print_plan()
            print("[blocked] No live fetch is wired for this source yet (needs credentials or a "
                  "per-season API loop). Use --input with a hand-downloaded file.")
            return 2
        print(f"[download] {source} / {args.league} ...")
        raw = fetch_fn(args)
        if raw is None or len(raw) == 0:
            print("[download] no rows fetched.")
            return 1
        path = write_staging_frame(
            normalize_fn(raw), source=source, sport=sport, league=args.league, out_dir=args.out_dir,
        )
        staged = pd.read_csv(path)
        print(f"[staged] {path}  ({len(raw)} raw rows -> {len(staged)} games)")
        print("Research-only: no betting, predictions, parlays, or gate changes.")
        return 0

    _print_plan()
    print("Provide --input <local.csv> to normalize a file, or --download to fetch (when wired).")
    return 0
