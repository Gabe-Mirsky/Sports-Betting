"""Backfill skeleton: SportsGameOdds historical PROP odds (hard-capped).

Research-only. Downloads NOTHING until approved. Run --input <local.csv> to
normalize an already-downloaded SGO export to staging, or no args to print the
plan. This is the ONLY prop-odds backfill path, and it is deliberately small:

  * Uses the EXISTING SportsGameOdds entity quota (1 event ~= 1 entity of the
    monthly budget); a hard --max-events cap protects the budget.
  * Only pulls a historical event when that event actually supports props
    (props reach ~2024-06; close fields ~2025-06). Do not assume older events.
  * Enables no betting, predictions, parlays, or gate changes.

Data types backfilled: prop_odds_recorded (+ closing_recorded where close fields
exist). Game-level odds may also be present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.free_backfill import pick_series  # noqa: E402
from data.free_backfill_cli import run_backfill_cli  # noqa: E402

PLANNED_SOURCES = [
    "SportsGameOdds /events (cheap listing) -> filter to prop-supporting events only",
    "SportsGameOdds historical odds for the capped, approved event list",
    "Cap: --max-events (default tiny); never run a large job",
]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map an SGO events/odds export to the staging partial schema (game level)."""

    date = pd.to_datetime(pick_series(raw, ("game_date", "start_time", "startTime")), errors="coerce")
    has_close = (
        "close_over_price" in raw.columns
        or "closing_price" in raw.columns
        or "is_closing" in raw.columns
    )
    out = pd.DataFrame(
        {
            "game_date": date.dt.date.astype("string"),
            "home_team": pick_series(raw, ("home_team", "homeTeam")),
            "away_team": pick_series(raw, ("away_team", "awayTeam")),
            "home_score": pick_series(raw, ("home_score", "homeScore")),
            "away_score": pick_series(raw, ("away_score", "awayScore")),
        }
    )
    # SGO rows here represent prop-supporting events by construction.
    out["prop_odds_available"] = True
    out["closing_available"] = bool(has_close)
    return out


def main() -> int:
    return run_backfill_cli(
        source="sportsgameodds",
        sport="basketball",
        league="NBA",
        normalize_fn=normalize,
        planned_sources=PLANNED_SOURCES,
        date_range="props ~2024-06 onward; closing fields ~2025-06 onward",
        data_types=("prop_odds_recorded", "closing_recorded"),
        license_notes="SGO API ToS; capped against existing entity quota; only prop-supporting events.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
