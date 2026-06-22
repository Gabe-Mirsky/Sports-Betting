"""Backfill skeleton: NHL results from MoneyPuck / fastRhockey (outcomes only).

Research-only. Downloads NOTHING until approved. Run --input <local.csv> to
normalize an already-downloaded file to staging, or no args to print the plan.

Sources (free):
  * MoneyPuck game CSVs (https://moneypuck.com/data.htm) - personal/non-commercial.
  * fastRhockey / NHL stats API (public) for game results and scores.
Market odds are NOT included (MoneyPuck model odds are not market lines), so this
backfills outcomes only. No player props.

Data types backfilled: outcome_only.
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
    "https://moneypuck.com/data.htm (games CSV)",
    "NHL stats API via fastRhockey (game results / scores)",
]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a MoneyPuck/fastRhockey games export to the staging partial schema."""

    date = pd.to_datetime(pick_series(raw, ("game_date", "gameDate", "date")), errors="coerce")
    return pd.DataFrame(
        {
            "game_date": date.dt.date.astype("string"),
            "home_team": pick_series(raw, ("home_team", "homeTeam", "home")),
            "away_team": pick_series(raw, ("away_team", "awayTeam", "away")),
            "home_score": pick_series(raw, ("home_score", "homeGoals", "home_goals")),
            "away_score": pick_series(raw, ("away_score", "awayGoals", "away_goals")),
        }
    )


def main() -> int:
    return run_backfill_cli(
        source="moneypuck",
        sport="hockey",
        league="NHL",
        normalize_fn=normalize,
        planned_sources=PLANNED_SOURCES,
        date_range="MoneyPuck ~2008-present; NHL API broad",
        data_types=("outcome_only",),
        license_notes="MoneyPuck free for personal/non-commercial with attribution; NHL API public.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
