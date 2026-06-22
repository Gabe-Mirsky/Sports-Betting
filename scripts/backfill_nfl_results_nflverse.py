"""Backfill skeleton: NFL results from nflverse/nflfastR (outcomes + some odds).

Research-only. Downloads NOTHING until approved. Run --input <local.csv> to
normalize an already-downloaded file to staging, or no args to print the plan.

Source: nflverse public data (e.g. games.csv / schedules) published via GitHub
releases. Recent seasons include closing-ish game lines (spread_line, total_line,
home_moneyline, away_moneyline), so this backfills outcomes AND partial game
odds. No player props. nflverse data is CC-BY (attribution).

Data types backfilled: outcome_only + game_odds_recorded (recent seasons).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.free_backfill import http_get, pick_series, save_raw  # noqa: E402
from data.free_backfill_cli import run_backfill_cli  # noqa: E402

NFL_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PLANNED_SOURCES = [NFL_GAMES_URL]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map an nflverse games.csv to the staging partial schema."""

    date = pd.to_datetime(pick_series(raw, ("gameday", "game_date")), errors="coerce")
    home_ml = pick_series(raw, ("home_moneyline",))
    away_ml = pick_series(raw, ("away_moneyline",))
    spread = pick_series(raw, ("spread_line",))
    total = pick_series(raw, ("total_line",))
    out = pd.DataFrame(
        {
            "game_date": date.dt.date.astype("string"),
            "home_team": pick_series(raw, ("home_team",)),
            "away_team": pick_series(raw, ("away_team",)),
            "home_score": pick_series(raw, ("home_score",)),
            "away_score": pick_series(raw, ("away_score",)),
            "home_odds": home_ml,
            "away_odds": away_ml,
        }
    )
    # nflverse schedule lines are closing-ish; flag game-odds closing when present.
    out["game_odds_available"] = home_ml.notna() | away_ml.notna() | spread.notna() | total.notna()
    out["closing_available"] = spread.notna() | total.notna()
    return out


def fetch(args: argparse.Namespace) -> pd.DataFrame:
    """Download the nflverse games.csv (all seasons in one file)."""

    content = http_get(NFL_GAMES_URL)
    save_raw(content, source="nflverse", name="games.csv")
    frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    if args.start_year:
        frame = frame[pd.to_numeric(frame.get("season"), errors="coerce") >= args.start_year]
    if args.end_year:
        frame = frame[pd.to_numeric(frame.get("season"), errors="coerce") <= args.end_year]
    return frame.reset_index(drop=True)


def main() -> int:
    return run_backfill_cli(
        source="nflverse",
        sport="football",
        league="NFL",
        normalize_fn=normalize,
        planned_sources=PLANNED_SOURCES,
        date_range="1999-present (schedules/scores; lines recent seasons)",
        data_types=("outcome_only", "game_odds_recorded", "closing_recorded(game-odds)"),
        license_notes="nflverse data CC-BY; free with attribution.",
        fetch_fn=fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
