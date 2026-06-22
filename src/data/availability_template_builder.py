"""Build manual team-availability templates from upcoming fixtures.

The template is intentionally simple: one editable row per fixture team by
default, with optional placeholder player rows for users who want several
slots ready for manual entry. No odds or market data are read or required.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.validation import require_columns


AVAILABILITY_TEMPLATE_COLUMNS: list[str] = [
    "fixture_id",
    "game_date",
    "team",
    "opponent",
    "sport",
    "league",
    "competition_type",
    "player_name",
    "status",
    "injury_type",
    "position",
    "importance_score",
    "expected_minutes_or_role",
    "last_updated",
    "return_estimate",
    "source",
    "notes",
]


def build_availability_template_from_fixtures(
    fixtures: pd.DataFrame,
    as_of_date: str | date | pd.Timestamp | None = None,
    include_placeholder_players: bool = False,
    players_per_team: int = 1,
) -> pd.DataFrame:
    """Return an editable availability template for every fixture team."""

    if fixtures.empty:
        return pd.DataFrame(columns=AVAILABILITY_TEMPLATE_COLUMNS)

    require_columns(
        fixtures,
        ["fixture_id", "game_date", "team_a", "team_b", "sport", "league", "competition_type"],
        dataframe_name="fixtures",
    )

    if as_of_date is None:
        updated = date.today().isoformat()
    else:
        updated_ts = pd.to_datetime(as_of_date, errors="coerce")
        updated = "" if pd.isna(updated_ts) else updated_ts.date().isoformat()

    slots = max(int(players_per_team or 1), 1) if include_placeholder_players else 1
    rows: list[dict] = []

    for _, fixture in fixtures.iterrows():
        for team_col, opponent_col in (("team_a", "team_b"), ("team_b", "team_a")):
            for idx in range(slots):
                player_name = f"Unknown player {idx + 1}" if include_placeholder_players else ""
                rows.append(
                    {
                        "fixture_id": str(fixture.get("fixture_id", "")),
                        "game_date": _date_text(fixture.get("game_date")),
                        "team": _clean_text(fixture.get(team_col)),
                        "opponent": _clean_text(fixture.get(opponent_col)),
                        "sport": _clean_text(fixture.get("sport")),
                        "league": _clean_text(fixture.get("league")),
                        "competition_type": _clean_text(fixture.get("competition_type")),
                        "player_name": player_name,
                        "status": "unknown",
                        "injury_type": "",
                        "position": "",
                        "importance_score": "",
                        "expected_minutes_or_role": "unknown",
                        "last_updated": updated,
                        "return_estimate": "",
                        "source": "manual",
                        "notes": "",
                    }
                )

    out = pd.DataFrame(rows, columns=AVAILABILITY_TEMPLATE_COLUMNS)
    out = out.drop_duplicates(subset=["fixture_id", "team", "player_name"], keep="first")
    out["_sort_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.sort_values(["_sort_date", "fixture_id", "team", "player_name"], na_position="last")
    out = out.drop(columns=["_sort_date"]).reset_index(drop=True)
    return out[AVAILABILITY_TEMPLATE_COLUMNS]


def write_availability_template(template: pd.DataFrame, output_path: str) -> None:
    """Write the availability template CSV, creating parent folders."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _date_text(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return _clean_text(value)
    return ts.date().isoformat()
