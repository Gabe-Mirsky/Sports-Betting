"""Backfill skeleton: free soccer results + game odds from Football-Data.co.uk.

Research-only. Downloads NOTHING until approved (see scripts/free_backfill CLI):
run with --input <local.csv> to normalize an already-downloaded Football-Data
file to staging, or with no args to print the plan.

Source: https://www.football-data.co.uk/  (free CSVs per league/season, e.g.
``E0.csv`` = English Premier League). Provides full-time scores AND game-level
odds, including CLOSING odds columns (the ``*C*`` variants, e.g. B365CH/B365CD/
B365CA). No player props. Free for personal use; attribution requested.

Data types backfilled: outcome_only + game_odds_recorded (+ game-odds closing).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.free_backfill import (  # noqa: E402
    http_get,
    pick_series,
    pick_valid_odds_series,
    save_raw,
    valid_odds_value_series,
)
from data.free_backfill_cli import run_backfill_cli  # noqa: E402

# Our league label -> Football-Data "Div" code (the per-league/season CSV stem).
LEAGUE_TO_DIV = {
    "EPL": "E0", "EFL_CHAMPIONSHIP": "E1", "LA_LIGA": "SP1", "LA_LIGA2": "SP2",
    "SERIE_A": "I1", "SERIE_B": "I2", "BUNDESLIGA": "D1", "BUNDESLIGA2": "D2",
    "LIGUE_1": "F1", "LIGUE_2": "F2", "EREDIVISIE": "N1", "JUPILER_PRO": "B1",
    "LIGA_PORTUGAL": "P1", "SUPER_LIG": "T1", "SUPER_LEAGUE_GR": "G1",
}
BASE_URL = "https://www.football-data.co.uk/mmz4281"
PLANNED_SOURCES = [f"{BASE_URL}/<season>/<DIV>.csv (e.g. 2324/E0.csv)"]
OPEN_HOME_ODDS = ("B365H", "PSH", "AvgH", "MaxH", "BWH", "IWH", "WHH", "VCH", "LBH", "GBH", "SBH", "SOH", "SYH")
OPEN_DRAW_ODDS = ("B365D", "PSD", "AvgD", "MaxD", "BWD", "IWD", "WHD", "VCD", "LBD", "GBD", "SBD", "SOD", "SYD")
OPEN_AWAY_ODDS = ("B365A", "PSA", "AvgA", "MaxA", "BWA", "IWA", "WHA", "VCA", "LBA", "GBA", "SBA", "SOA", "SYA")
CLOSE_HOME_ODDS = ("B365CH", "PSCH", "AvgCH", "MaxCH", "BWCH", "IWCH", "WHCH", "VCCH")
CLOSE_DRAW_ODDS = ("B365CD", "PSCD", "AvgCD", "MaxCD", "BWCD", "IWCD", "WHCD", "VCCD")
CLOSE_AWAY_ODDS = ("B365CA", "PSCA", "AvgCA", "MaxCA", "BWCA", "IWCA", "WHCA", "VCCA")


def _season_codes(start_year: int, end_year: int) -> list[str]:
    """Football-Data season stems: 1993/94 -> '9394', 2024/25 -> '2425'."""

    codes = []
    for year in range(start_year, end_year + 1):
        codes.append(f"{year % 100:02d}{(year + 1) % 100:02d}")
    return codes


def fetch(args: argparse.Namespace) -> pd.DataFrame:
    """Download every available season CSV for ``args.league`` and concatenate."""

    div = LEAGUE_TO_DIV.get(args.league.upper())
    if not div:
        raise SystemExit(f"Unknown soccer league {args.league!r}; known: {sorted(LEAGUE_TO_DIV)}")
    start = args.start_year or 1993
    end = args.end_year or 2024
    frames = []
    for season in _season_codes(start, end):
        url = f"{BASE_URL}/{season}/{div}.csv"
        try:
            content = http_get(url)
        except Exception as exc:  # noqa: BLE001 - a missing season is normal
            print(f"  skip {season}/{div}: {type(exc).__name__}")
            continue
        save_raw(content, source="football_data", name=f"{div}_{season}.csv")
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip")
        except Exception as exc:  # noqa: BLE001
            print(f"  parse-fail {season}/{div}: {type(exc).__name__}")
            continue
        if not frame.empty:
            frames.append(frame)
            print(f"  {season}/{div}: {len(frame)} rows")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a Football-Data.co.uk CSV to the staging partial schema."""

    date = pd.to_datetime(pick_series(raw, ("Date",)), dayfirst=True, errors="coerce")
    home_open = pick_valid_odds_series(raw, OPEN_HOME_ODDS)
    draw_open = pick_valid_odds_series(raw, OPEN_DRAW_ODDS)
    away_open = pick_valid_odds_series(raw, OPEN_AWAY_ODDS)
    home_close = pick_valid_odds_series(raw, CLOSE_HOME_ODDS)
    draw_close = pick_valid_odds_series(raw, CLOSE_DRAW_ODDS)
    away_close = pick_valid_odds_series(raw, CLOSE_AWAY_ODDS)
    home_odds = home_open.mask(~valid_odds_value_series(home_open), home_close)
    draw_odds = draw_open.mask(~valid_odds_value_series(draw_open), draw_close)
    away_odds = away_open.mask(~valid_odds_value_series(away_open), away_close)
    closing_available = (
        valid_odds_value_series(home_close)
        & valid_odds_value_series(draw_close)
        & valid_odds_value_series(away_close)
    )
    out = pd.DataFrame(
        {
            "game_date": date.dt.date.astype("string"),
            "home_team": pick_series(raw, ("HomeTeam", "Home")),
            "away_team": pick_series(raw, ("AwayTeam", "Away")),
            "home_score": pick_series(raw, ("FTHG", "HG")),
            "away_score": pick_series(raw, ("FTAG", "AG")),
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
        }
    )
    # Game-odds closing snapshot present only when this row has closing 1X2 odds.
    out["closing_available"] = closing_available
    return out


def main() -> int:
    return run_backfill_cli(
        source="football_data",
        sport="soccer",
        league="EPL",
        normalize_fn=normalize,
        planned_sources=PLANNED_SOURCES,
        date_range="1993/94 to present (per league/season CSV)",
        data_types=("outcome_only", "game_odds_recorded", "closing_recorded(game-odds)"),
        license_notes="Free for personal use; attribution requested; official CSV downloads (no scraping).",
        fetch_fn=fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
