"""Free historical backfill source audit (read-only, research-only).

Curated inventory of FREE / already-available sources we can use to backfill
more *games* (outcomes and, where free, game odds) into the Recorded Games
inventory. This is a planning artifact: it downloads nothing and enables no
betting, predictions, parlays, or gate changes. Player-prop odds are explicitly
called out because historical prop odds are mostly NOT free.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_INVENTORY_COLUMNS = [
    "league", "source_name", "source_type", "free_or_quota", "data_type",
    "date_range", "odds_available", "prop_odds_available", "closing_available",
    "license_notes", "importer_needed",
]

# One row per (league family, source). data_type is the PRIMARY thing the source
# backfills for free. odds_available = game-level odds; prop_odds_available =
# player props; closing_available = a line captured near game start.
SOURCE_INVENTORY = [
    {
        "league": "EPL/LA_LIGA/SERIE_A/BUNDESLIGA/LIGUE_1/MLS",
        "source_name": "Football-Data.co.uk",
        "source_type": "CSV download (HTTP)",
        "free_or_quota": "free",
        "data_type": "outcome + game_odds",
        "date_range": "1993/94 to present (varies by league)",
        "odds_available": "yes (1X2, O/U, Asian handicap; multiple books)",
        "prop_odds_available": "no",
        "closing_available": "yes (game-odds closing columns, e.g. *C); no prop closing",
        "license_notes": "Free for personal use; attribution requested; provides CSVs for download (no scraping needed).",
        "importer_needed": "scripts/backfill_free_soccer_results_and_odds.py",
    },
    {
        "league": "WORLD_CUP",
        "source_name": "Kaggle FIFA World Cup datasets (e.g. martj42 results)",
        "source_type": "Kaggle dataset (CSV)",
        "free_or_quota": "free (Kaggle account)",
        "data_type": "outcome",
        "date_range": "1930-2022 (World Cup); intl results 1872-present",
        "odds_available": "no",
        "prop_odds_available": "no",
        "closing_available": "no",
        "license_notes": "Per-dataset license (often CC0/public domain); verify the exact dataset license before use.",
        "importer_needed": "scripts/backfill_world_cup_results.py",
    },
    {
        "league": "WORLD_CUP/soccer",
        "source_name": "TheSportsDB (v1 free API)",
        "source_type": "REST API (free tier, rate-limited)",
        "free_or_quota": "free (test key; quota-limited)",
        "data_type": "outcome (events + scores)",
        "date_range": "broad historical for major events/leagues",
        "odds_available": "no",
        "prop_odds_available": "no",
        "closing_available": "no",
        "license_notes": "Free-tier ToS; attribution; rate limits. Fallback only, used sparingly.",
        "importer_needed": "scripts/backfill_world_cup_results.py (fallback path)",
    },
    {
        "league": "MLB",
        "source_name": "Retrosheet",
        "source_type": "Game logs (downloadable files)",
        "free_or_quota": "free",
        "data_type": "outcome",
        "date_range": "1871-present (game logs)",
        "odds_available": "no",
        "prop_odds_available": "no",
        "closing_available": "no",
        "license_notes": "Free WITH required attribution notice (Retrosheet copyright). Must include the notice on use.",
        "importer_needed": "scripts/backfill_mlb_results_retrosheet.py",
    },
    {
        "league": "NFL",
        "source_name": "nflverse / nflfastR (games.csv schedules)",
        "source_type": "Public data release (CSV/parquet via GitHub)",
        "free_or_quota": "free",
        "data_type": "outcome (+ some game_odds)",
        "date_range": "1999-present (schedules/scores)",
        "odds_available": "partial (recent seasons include spread_line/total_line/moneyline)",
        "prop_odds_available": "no",
        "closing_available": "partial (schedule lines are closing-ish for recent seasons)",
        "license_notes": "nflverse data CC-BY; free with attribution.",
        "importer_needed": "scripts/backfill_nfl_results_nflverse.py",
    },
    {
        "league": "NHL",
        "source_name": "MoneyPuck / fastRhockey (NHL API)",
        "source_type": "CSV download / NHL stats API wrapper",
        "free_or_quota": "free",
        "data_type": "outcome",
        "date_range": "MoneyPuck ~2008-present; NHL API broad",
        "odds_available": "no (model odds only, not market)",
        "prop_odds_available": "no",
        "closing_available": "no",
        "license_notes": "MoneyPuck free for personal/non-commercial with attribution; NHL API public.",
        "importer_needed": "scripts/backfill_nhl_results_moneypuck.py",
    },
    {
        "league": "NBA",
        "source_name": "nba_api + existing local processed files",
        "source_type": "already-available local data",
        "free_or_quota": "free (already collected)",
        "data_type": "outcome (+ historical game_odds locally)",
        "date_range": "2018-present (local parquets) + current season",
        "odds_available": "yes (nba_game_odds_normalized.csv, game level)",
        "prop_odds_available": "recent only (live collection; no historical backfill)",
        "closing_available": "recent only (forward prop collection)",
        "license_notes": "nba_api is unofficial; personal use. No new download needed.",
        "importer_needed": "no (reuse existing files / importers)",
    },
    {
        "league": "NBA/MLB/NHL/NFL (SGO-supported)",
        "source_name": "SportsGameOdds",
        "source_type": "REST API (entity quota)",
        "free_or_quota": "quota (existing entity budget; hard-capped)",
        "data_type": "prop_odds (+ game_odds)",
        "date_range": "props ~2024-06 onward; closing fields ~2025-06 onward",
        "odds_available": "yes",
        "prop_odds_available": "yes (capped; only where the event supports props)",
        "closing_available": "partial (recent window only)",
        "license_notes": "API ToS; 1 event ~= 1 entity of the monthly budget; only /events is cheap. Cap hard.",
        "importer_needed": "scripts/backfill_sgo_historical_props_capped.py",
    },
]


def source_inventory_frame() -> pd.DataFrame:
    return pd.DataFrame(SOURCE_INVENTORY, columns=SOURCE_INVENTORY_COLUMNS)


def _md_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


def build_source_audit_markdown() -> str:
    frame = source_inventory_frame()
    return "\n".join(
        [
            "# Free Historical Backfill - Source Audit",
            "",
            "Research-only planning artifact. Downloads nothing. Enables no betting, "
            "predictions, parlays, or gate changes. The goal is to backfill more **games** "
            "into Recorded Games from free / already-available sources.",
            "",
            "## Four backfill data types",
            "",
            "- **outcome_only** - final score / box score, no market odds.",
            "- **game_odds_recorded** - game-level market odds (moneyline / 1X2 / spread / total).",
            "- **prop_odds_recorded** - player-prop market odds.",
            "- **closing_recorded** - a line/snapshot captured near game start.",
            "",
            "## Source inventory",
            "",
            _md_table(frame),
            "",
            "## What we can backfill for FREE",
            "",
            "- **Outcomes (all target leagues):** soccer leagues + World Cup (Football-Data.co.uk, "
            "Kaggle, TheSportsDB), MLB (Retrosheet), NFL (nflverse), NHL (MoneyPuck/NHL API), "
            "NBA (already local).",
            "- **Game odds (free):** soccer via Football-Data.co.uk (incl. closing columns), NFL "
            "partial via nflverse schedule lines, NBA historical game odds already local.",
            "- **World Cup / soccer events & scores:** free APIs (TheSportsDB) as a fallback to the "
            "CSV/Kaggle sources.",
            "",
            "## What still REQUIRES paid prop-odds data",
            "",
            "- **Historical player-prop odds** are essentially not free. The only free-ish path is "
            "the **SportsGameOdds** capped backfill, and only:",
            "  - where the specific historical event supports props, and",
            "  - within its window (props ~2024-06+, closing fields ~2025-06+).",
            "- **Historical prop CLOSING snapshots** before that window cannot be backfilled for free; "
            "they only accrue going forward via live collection. No gate is loosened to compensate.",
            "",
            "## Dashboard visibility note",
            "",
            "Recorded Games renders per league tab (tabs: NBA, MLB, WNBA, NHL, World Cup). Backfilled "
            "**soccer leagues (EPL/La Liga/etc.) and NFL have no dedicated tab**, so a single "
            "**\"Historical Backfill\" tab** surfaces the full `historical_game_inventory.csv` across "
            "ALL leagues with a per-league x status breakdown. Adding dedicated per-league tabs is a "
            "separate optional change.",
            "",
            "## Approval checklist (before any download)",
            "",
            "1. Confirm the source list above.",
            "2. Confirm per-source date ranges to pull (start small, e.g. one season).",
            "3. Confirm the SGO cap (max events) for the prop backfill.",
            "4. Only then wire each importer's `fetch()` and run a small job.",
            "",
        ]
    )


def write_source_audit(out_dir: Path | str) -> dict[str, Path]:
    """Write the audit markdown + the source inventory CSV; return their paths."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "free_backfill_source_audit.md"
    csv_path = out_dir / "free_backfill_source_inventory.csv"
    md_path.write_text(build_source_audit_markdown(), encoding="utf-8")
    source_inventory_frame().to_csv(csv_path, index=False)
    return {"markdown": md_path, "csv": csv_path}
