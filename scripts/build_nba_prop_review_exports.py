"""Build Excel-friendly NBA prop review CSVs (research-only).

Joins the enriched NBA snapshots with the main-line detection audit and writes
four flat review files designed to be opened directly in Excel:

    data/reports/nba_main_lines_review.csv      one row per market: the likely
                                                main line with latest prices
    data/reports/nba_alt_lines_review.csv       rows quoting non-main (alt) lines
    data/reports/nba_bookmaker_comparison.csv   one row per player/prop/bookmaker
                                                main line, with cross-book spread
    data/reports/nba_prop_board_latest.csv      the latest snapshot of every
                                                NBA market (main + alt rows)

Research-only: descriptive review exports. No recommendations, no bets.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
ENRICHED_PATH = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_enriched.csv"
LINE_QUALITY_PATH = REPORTS_DIR / "player_prop_line_quality.csv"

MARKET_KEYS = ["player_name", "prop_type", "bookmaker", "canonical_game_key"]

REVIEW_COLUMNS = [
    "player_name", "team", "prop_type", "likely_main_line", "line",
    "over_price", "under_price", "bookmaker", "snapshot_time",
    "is_closing_snapshot", "line_quality_label", "is_alt_line",
    "settlement_status", "actual_stat_value", "over_won", "under_won", "push",
    "game_date", "canonical_game_key",
]


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ENRICHED_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    enriched = pd.read_csv(ENRICHED_PATH, low_memory=False)
    nba = enriched[enriched["league"].astype(str).str.upper().eq("NBA")].copy()
    quality = pd.DataFrame()
    if LINE_QUALITY_PATH.exists():
        quality = pd.read_csv(LINE_QUALITY_PATH, low_memory=False)
        if "league" in quality.columns:
            quality = quality[quality["league"].astype(str).eq("NBA")].copy()
    return nba, quality


def annotate(nba: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Attach likely_main_line / line_quality_label and an is_alt_line flag."""
    frame = nba.copy()
    if not quality.empty:
        cols = MARKET_KEYS + ["likely_main_line", "line_quality_label"]
        cols = [c for c in cols if c in quality.columns]
        frame = frame.merge(quality[cols].drop_duplicates(MARKET_KEYS), on=MARKET_KEYS, how="left")
    else:
        frame["likely_main_line"] = pd.NA
        frame["line_quality_label"] = pd.NA
    line = pd.to_numeric(frame["line"], errors="coerce")
    main = pd.to_numeric(frame["likely_main_line"], errors="coerce")
    frame["is_alt_line"] = line.notna() & main.notna() & (line != main)
    frame["snapshot_time"] = frame["snapshot_time"].astype(str)
    return frame


def latest_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Most recent snapshot row per key group (per quoted line)."""
    if frame.empty:
        return frame
    ordered = frame.sort_values("snapshot_time")
    return ordered.groupby(keys + ["line"], dropna=False).tail(1)


def build_main_lines_review(frame: pd.DataFrame) -> pd.DataFrame:
    main = frame[~frame["is_alt_line"]].copy()
    latest = latest_rows(main, MARKET_KEYS)
    columns = [c for c in REVIEW_COLUMNS if c in latest.columns]
    return latest[columns].sort_values(["game_date", "player_name", "prop_type", "bookmaker"])


def build_alt_lines_review(frame: pd.DataFrame) -> pd.DataFrame:
    alt = frame[frame["is_alt_line"]].copy()
    latest = latest_rows(alt, MARKET_KEYS)
    columns = [c for c in REVIEW_COLUMNS if c in latest.columns]
    return latest[columns].sort_values(["game_date", "player_name", "prop_type", "bookmaker", "line"])


def build_bookmaker_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Latest main line per bookmaker with cross-book line spread per market."""
    main = frame[~frame["is_alt_line"]].copy()
    latest = latest_rows(main, MARKET_KEYS)
    if latest.empty:
        return latest
    group_keys = ["player_name", "prop_type", "canonical_game_key"]
    lines = pd.to_numeric(latest["line"], errors="coerce")
    latest = latest.assign(_line_num=lines)
    spread = latest.groupby(group_keys)["_line_num"].agg(
        bookmakers="count", consensus_min_line="min", consensus_max_line="max"
    ).reset_index()
    spread["line_disagreement"] = spread["consensus_max_line"] - spread["consensus_min_line"]
    merged = latest.merge(spread, on=group_keys, how="left").drop(columns=["_line_num"])
    columns = [
        "player_name", "team", "prop_type", "bookmaker", "line",
        "over_price", "under_price", "bookmakers", "consensus_min_line",
        "consensus_max_line", "line_disagreement", "snapshot_time",
        "is_closing_snapshot", "line_quality_label", "game_date", "canonical_game_key",
    ]
    columns = [c for c in columns if c in merged.columns]
    return merged[columns].sort_values(
        ["game_date", "player_name", "prop_type", "bookmaker"]
    )


def build_prop_board(frame: pd.DataFrame) -> pd.DataFrame:
    """Latest snapshot of every NBA market row, main and alt lines included."""
    latest = latest_rows(frame, MARKET_KEYS)
    columns = [c for c in REVIEW_COLUMNS if c in latest.columns]
    return latest[columns].sort_values(
        ["game_date", "player_name", "prop_type", "bookmaker", "is_alt_line", "line"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NBA prop review CSVs.")
    parser.parse_args()

    nba, quality = load_inputs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "nba_main_lines_review.csv": pd.DataFrame(),
        "nba_alt_lines_review.csv": pd.DataFrame(),
        "nba_bookmaker_comparison.csv": pd.DataFrame(),
        "nba_prop_board_latest.csv": pd.DataFrame(),
    }
    if nba.empty:
        print("No NBA snapshots yet; writing empty review files with headers.")
        empty = pd.DataFrame(columns=REVIEW_COLUMNS)
        outputs = {name: empty for name in outputs}
    else:
        frame = annotate(nba, quality)
        outputs["nba_main_lines_review.csv"] = build_main_lines_review(frame)
        outputs["nba_alt_lines_review.csv"] = build_alt_lines_review(frame)
        outputs["nba_bookmaker_comparison.csv"] = build_bookmaker_comparison(frame)
        outputs["nba_prop_board_latest.csv"] = build_prop_board(frame)

    for name, df in outputs.items():
        path = REPORTS_DIR / name
        df.to_csv(path, index=False)
        print(f"Wrote {name}: {len(df)} rows")

    summary = {
        "report": "nba_prop_review_exports",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {name: int(len(df)) for name, df in outputs.items()},
        "research_only": True,
        "approved": False,
    }
    (REPORTS_DIR / "nba_prop_review_exports_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Research-only review exports; no recommendations, approved bets/parlays remain blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
