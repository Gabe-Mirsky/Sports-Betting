"""Backfill skeleton: World Cup historical results (outcomes only).

Research-only. Downloads NOTHING until approved. Run --input <local.csv> to
normalize an already-downloaded file to staging, or no args to print the plan.

Sources (free):
  * Kaggle FIFA World Cup datasets (e.g. martj42 "International football results
    1872-present" results.csv; filter tournament == "FIFA World Cup").
  * TheSportsDB free API events as a fallback (columns strHomeTeam/strAwayTeam/
    intHomeScore/intAwayScore/dateEvent).

Data types backfilled: outcome_only. No odds, no props.
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

# martj42 dataset is mirrored free on GitHub (no Kaggle credentials needed).
WORLD_CUP_URLS = [
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "https://raw.githubusercontent.com/martj42/international_results/main/results.csv",
]
PLANNED_SOURCES = [
    "GitHub: martj42/international_results/results.csv (free mirror of the Kaggle set)",
    "TheSportsDB v1 free API: eventsseason.php / lookupevent.php (fallback)",
]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a martj42 results.csv (or TheSportsDB export) to staging partial."""

    date = pd.to_datetime(pick_series(raw, ("date", "dateEvent")), errors="coerce")
    out = pd.DataFrame(
        {
            "game_date": date.dt.date.astype("string"),
            "home_team": pick_series(raw, ("home_team", "strHomeTeam")),
            "away_team": pick_series(raw, ("away_team", "strAwayTeam")),
            "home_score": pick_series(raw, ("home_score", "intHomeScore")),
            "away_score": pick_series(raw, ("away_score", "intAwayScore")),
        }
    )
    # World Cup finals only (drop qualifiers / other tournaments) when labelled.
    if "tournament" in raw.columns:
        tournament = raw["tournament"].astype(str)
        keep = tournament.str.contains("World Cup", case=False, na=False) & ~tournament.str.contains(
            "qualification", case=False, na=False
        )
        out = out[keep.values].reset_index(drop=True)
    return out


def fetch(args: argparse.Namespace) -> pd.DataFrame:
    """Download the martj42 international results CSV (free GitHub mirror)."""

    for url in WORLD_CUP_URLS:
        try:
            content = http_get(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {url}: {type(exc).__name__}")
            continue
        save_raw(content, source="world_cup_results", name="international_results.csv")
        return pd.read_csv(io.BytesIO(content))
    raise SystemExit("Could not fetch martj42 results.csv from any mirror.")


def main() -> int:
    return run_backfill_cli(
        source="world_cup_results",
        sport="soccer",
        league="WORLD_CUP",
        normalize_fn=normalize,
        planned_sources=PLANNED_SOURCES,
        date_range="1930-2022 (World Cup finals); intl results 1872-present",
        data_types=("outcome_only",),
        license_notes="martj42 dataset is open (CC0-style); TheSportsDB free-tier ToS for the fallback.",
        fetch_fn=fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
