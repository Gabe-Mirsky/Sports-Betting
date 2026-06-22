"""Backfill skeleton: MLB historical results from Retrosheet (outcomes only).

Research-only. Downloads NOTHING until approved. Run --input <local.csv> to
normalize an already-downloaded file to staging, or no args to print the plan.

Source: https://www.retrosheet.org/  game logs (gl1871-present). The raw game
log is a header-less CSV of positional fields; this skeleton accepts either:
  * a pre-parsed CSV with columns date, away_team, home_team, away_score, home_score; or
  * the raw game-log positional layout (date=field0 YYYYMMDD, visitor=field3,
    home=field6, visitor score=field9, home score=field10).

ATTRIBUTION (required by Retrosheet): "The information used here was obtained free
of charge from and is copyrighted by Retrosheet. Interested parties may contact
Retrosheet at www.retrosheet.org."

Data types backfilled: outcome_only. No odds, no props.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.free_backfill import http_get, pick_series, save_raw  # noqa: E402
from data.free_backfill_cli import run_backfill_cli  # noqa: E402

RETROSHEET_NOTICE = (
    "The information used here was obtained free of charge from and is copyrighted "
    "by Retrosheet (www.retrosheet.org)."
)
GAMELOG_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
PLANNED_SOURCES = [GAMELOG_URL]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a Retrosheet game log (parsed or positional) to staging partial."""

    if "date" in raw.columns or "home_team" in raw.columns:
        date = pd.to_datetime(pick_series(raw, ("date",)), errors="coerce", format="mixed")
        home = pick_series(raw, ("home_team",))
        away = pick_series(raw, ("away_team", "visiting_team"))
        home_score = pick_series(raw, ("home_score",))
        away_score = pick_series(raw, ("away_score", "visitor_score"))
    else:
        # Positional game-log layout (no header): cols are integer-indexed.
        date = pd.to_datetime(raw.iloc[:, 0].astype(str), format="%Y%m%d", errors="coerce")
        away = raw.iloc[:, 3]
        home = raw.iloc[:, 6]
        away_score = raw.iloc[:, 9]
        home_score = raw.iloc[:, 10]

    return pd.DataFrame(
        {
            "game_date": pd.to_datetime(date, errors="coerce").dt.date.astype("string"),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
        }
    )


def fetch(args: argparse.Namespace) -> pd.DataFrame:
    """Download Retrosheet game-log zips for a year range; concat positional rows."""

    start = args.start_year or 1990
    end = args.end_year or 2024
    frames = []
    for year in range(start, end + 1):
        url = GAMELOG_URL.format(year=year)
        try:
            content = http_get(url)
        except Exception as exc:  # noqa: BLE001 - some years may be absent
            print(f"  skip {year}: {type(exc).__name__}")
            continue
        save_raw(content, source="retrosheet", name=f"gl{year}.zip")
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as bundle:
                inner = next((n for n in bundle.namelist() if n.upper().endswith(".TXT")), None)
                if inner is None:
                    continue
                with bundle.open(inner) as handle:
                    frame = pd.read_csv(handle, header=None, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  parse-fail {year}: {type(exc).__name__}")
            continue
        frames.append(frame)
        print(f"  {year}: {len(frame)} games")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    print(f"[notice] {RETROSHEET_NOTICE}")
    return run_backfill_cli(
        source="retrosheet",
        sport="baseball",
        league="MLB",
        normalize_fn=normalize,
        planned_sources=PLANNED_SOURCES,
        date_range="1871-present (game logs)",
        data_types=("outcome_only",),
        license_notes=RETROSHEET_NOTICE,
        fetch_fn=fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
