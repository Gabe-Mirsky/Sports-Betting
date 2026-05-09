"""NBA season date helpers."""

from __future__ import annotations

from datetime import date


def current_nba_season_start_year(today: date | None = None) -> int:
    """Return the NBA season start year for a calendar date.

    NBA season labels use the first calendar year in the season. For example,
    games played in May 2026 belong to the 2025-26 season, so this returns 2025.
    """

    current = today or date.today()
    if current.month >= 7:
        return current.year
    return current.year - 1


def nba_season_display_label(season_start_year: int) -> str:
    """Return a friendly season label like 2025-26."""

    return f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
