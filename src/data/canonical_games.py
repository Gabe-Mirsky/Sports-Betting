"""Canonical game key system shared across all data sources.

Builds one stable game key from (sport, league, game_date, home_team, away_team)
so games can be matched across ehallmar, zachht, nba_api, Kalshi, and future
sportsbook player-prop sources that all use different native ids.

Key format (version v1)::

    <sport>|<league>|<YYYY-MM-DD>|<home_abbr>|<away_abbr>
    basketball|NBA|2025-10-21|OKC|HOU

A companion order-independent ``matchup_key`` (teams sorted) supports sources
such as zachht where the home/away orientation is not explicit.

Research-only: ingestion/identity plumbing, no model logic, no proof-gate or
betting changes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .team_aliases import compact_team_text, normalize_team_abbr, CURRENT_TEAM_ABBRS


GAME_KEY_VERSION = "v1"
KEY_SEPARATOR = "|"

# Leagues whose team values run through the NBA alias table.
_NBA_ALIAS_LEAGUES = {"NBA"}

# Pinnacle (zachht) lists US-sport matchups as "away vs home": team1=away, team2=home.
# The current-basketball join verifies this empirically and reports violations.
ZACHHT_TEAM2_IS_HOME = True


def normalize_sport(value: object) -> str:
    """Normalize a sport label: lowercase alphanumeric."""

    return re.sub(r"[^a-z0-9_]", "", str(value).strip().lower().replace(" ", "_"))


def normalize_league(value: object) -> str:
    """Normalize a league label: uppercase alphanumeric (NBA, WNBA, NFL, ...)."""

    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def normalize_game_date(value: object) -> str:
    """Normalize a game date to ISO YYYY-MM-DD; empty string when unparseable."""

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).date().isoformat()


def normalize_team_for_key(value: object, league: str = "NBA") -> str:
    """Normalize a team value to the canonical token used inside game keys.

    NBA values run through the alias table (names, cities, nicknames, alt
    abbreviations -> current 3-letter abbr). Other leagues fall back to compact
    uppercase alphanumeric text, which is stable within a source.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    league_norm = normalize_league(league)
    if league_norm in _NBA_ALIAS_LEAGUES:
        return normalize_team_abbr(value)
    return compact_team_text(value)


def build_canonical_game_key(
    sport: object,
    league: object,
    game_date: object,
    home_team: object,
    away_team: object,
) -> str:
    """Build the canonical (home/away ordered) game key.

    Raises ``ValueError`` when any component is missing or unparseable so bad
    keys never silently enter joins.
    """

    sport_n = normalize_sport(sport)
    league_n = normalize_league(league)
    date_n = normalize_game_date(game_date)
    home_n = normalize_team_for_key(home_team, league_n)
    away_n = normalize_team_for_key(away_team, league_n)
    parts = {"sport": sport_n, "league": league_n, "game_date": date_n, "home_team": home_n, "away_team": away_n}
    missing = [name for name, part in parts.items() if not part]
    if missing:
        raise ValueError(f"Cannot build canonical game key; missing/unparseable: {missing}")
    if home_n == away_n:
        raise ValueError(f"Cannot build canonical game key; home == away ({home_n})")
    return KEY_SEPARATOR.join([sport_n, league_n, date_n, home_n, away_n])


def build_matchup_key(
    sport: object,
    league: object,
    game_date: object,
    team_a: object,
    team_b: object,
) -> str:
    """Order-independent variant of the game key (teams sorted alphabetically)."""

    sport_n = normalize_sport(sport)
    league_n = normalize_league(league)
    date_n = normalize_game_date(game_date)
    a = normalize_team_for_key(team_a, league_n)
    b = normalize_team_for_key(team_b, league_n)
    if not (sport_n and league_n and date_n and a and b):
        raise ValueError("Cannot build matchup key; missing/unparseable components")
    first, second = sorted([a, b])
    return KEY_SEPARATOR.join([sport_n, league_n, date_n, first, second])


def parse_canonical_game_key(key: str) -> dict[str, str]:
    """Split a canonical game key back into its components."""

    parts = str(key).split(KEY_SEPARATOR)
    if len(parts) != 5:
        raise ValueError(f"Not a {GAME_KEY_VERSION} canonical game key: {key!r}")
    return {
        "sport": parts[0],
        "league": parts[1],
        "game_date": parts[2],
        "home_team": parts[3],
        "away_team": parts[4],
    }


def canonical_game_key_series(
    frame: pd.DataFrame,
    sport: str,
    league_col: str,
    date_col: str,
    home_col: str,
    away_col: str,
) -> pd.Series:
    """Vectorized key builder for dataframes; bad rows yield '' instead of raising."""

    if frame.empty:
        return pd.Series(dtype="object")

    sport_n = normalize_sport(sport)
    leagues = frame[league_col].map(normalize_league)
    dates = frame[date_col].map(normalize_game_date)
    homes = [normalize_team_for_key(team, lg) for team, lg in zip(frame[home_col], leagues)]
    aways = [normalize_team_for_key(team, lg) for team, lg in zip(frame[away_col], leagues)]
    keys = []
    for league, date, home, away in zip(leagues, dates, homes, aways):
        if sport_n and league and date and home and away and home != away:
            keys.append(KEY_SEPARATOR.join([sport_n, league, date, home, away]))
        else:
            keys.append("")
    return pd.Series(keys, index=frame.index, dtype="object")


# ---------------------------------------------------------------------------
# zachht per-game collapse (snapshots -> one row per game)
# ---------------------------------------------------------------------------

ZACHHT_GAME_TIMEZONE = "America/New_York"


def estimate_game_date_from_snapshot(snapshot_time: object, tz: str = ZACHHT_GAME_TIMEZONE) -> str:
    """Estimate the (US local) game date from a UTC snapshot timestamp."""

    parsed = pd.to_datetime(snapshot_time, errors="coerce")
    if pd.isna(parsed):
        return ""
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(tz).date().isoformat()


def collapse_zachht_games(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Collapse zachht odds snapshots to one row per game.

    The game date is estimated from the closing (latest) snapshot timestamp
    converted to US/Eastern; zachht has no explicit game date. Home/away is
    assumed from the Pinnacle "away vs home" listing order (team2 = home) and
    flagged as unconfirmed for downstream verification.
    """

    if snapshots.empty:
        return pd.DataFrame()
    frame = snapshots.copy()
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_time"])
    frame = frame.sort_values("snapshot_time").groupby(["league", "game_ref"], as_index=False).last()

    if ZACHHT_TEAM2_IS_HOME:
        home_col, away_col = "team2_abbr", "team1_abbr"
        home_name, away_name = "team2_name", "team1_name"
    else:
        home_col, away_col = "team1_abbr", "team2_abbr"
        home_name, away_name = "team1_name", "team2_name"

    games = pd.DataFrame(
        {
            "source": "zachht",
            "sport": "basketball",
            "league": frame["league"],
            "source_game_id": frame["game_ref"].astype(str),
            "game_date": frame["snapshot_time"].map(estimate_game_date_from_snapshot),
            "home_team_abbr": frame[home_col].astype(str),
            "away_team_abbr": frame[away_col].astype(str),
            "home_team_name": frame.get(home_name),
            "away_team_name": frame.get(away_name),
            "closing_snapshot_time": frame["snapshot_time"],
            "n_snapshots": pd.to_numeric(frame.get("n_snapshots_for_game"), errors="coerce"),
            "game_date_estimated": True,
            "home_away_confident": False,
        }
    )
    games["canonical_game_key"] = canonical_game_key_series(
        games, "basketball", "league", "game_date", "home_team_abbr", "away_team_abbr"
    )
    games["matchup_key"] = [
        _safe_matchup_key("basketball", lg, d, a, b)
        for lg, d, a, b in zip(games["league"], games["game_date"], games["home_team_abbr"], games["away_team_abbr"])
    ]
    return games.reset_index(drop=True)


def _safe_matchup_key(sport: str, league: object, date: object, team_a: object, team_b: object) -> str:
    try:
        return build_matchup_key(sport, league, date, team_a, team_b)
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Canonical games table across all available sources
# ---------------------------------------------------------------------------

_CANONICAL_COLUMNS = [
    "canonical_game_key",
    "matchup_key",
    "key_version",
    "sport",
    "league",
    "game_date",
    "home_team_abbr",
    "away_team_abbr",
    "source",
    "source_game_id",
    "game_date_estimated",
    "home_away_confident",
]


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs) if path.exists() else pd.DataFrame()


def _source_rows(
    frame: pd.DataFrame,
    source: str,
    league: object,
    id_col: str,
    date_col: str = "game_date",
    home_col: str = "home_team_abbr",
    away_col: str = "away_team_abbr",
    date_estimated: bool = False,
    home_away_confident: bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    out = pd.DataFrame(
        {
            "sport": "basketball",
            "league": frame[league] if isinstance(league, str) and league in frame.columns else str(league),
            "game_date": frame[date_col].map(normalize_game_date),
            "home_team_abbr": frame[home_col],
            "away_team_abbr": frame[away_col],
            "source": source,
            "source_game_id": frame[id_col].astype(str),
            "game_date_estimated": date_estimated,
            "home_away_confident": home_away_confident,
        }
    )
    out["canonical_game_key"] = canonical_game_key_series(
        out, "basketball", "league", "game_date", "home_team_abbr", "away_team_abbr"
    )
    out["matchup_key"] = [
        _safe_matchup_key("basketball", lg, d, a, b)
        for lg, d, a, b in zip(out["league"], out["game_date"], out["home_team_abbr"], out["away_team_abbr"])
    ]
    out["key_version"] = GAME_KEY_VERSION
    # Re-normalize the team columns so the table shows the canonical tokens.
    leagues_norm = out["league"].map(normalize_league)
    out["home_team_abbr"] = [normalize_team_for_key(t, lg) for t, lg in zip(out["home_team_abbr"], leagues_norm)]
    out["away_team_abbr"] = [normalize_team_for_key(t, lg) for t, lg in zip(out["away_team_abbr"], leagues_norm)]
    out["league"] = leagues_norm
    return out[_CANONICAL_COLUMNS]


def build_canonical_games_table(processed_dir: str | Path) -> pd.DataFrame:
    """Assemble the cross-source canonical games table from normalized files."""

    processed = Path(processed_dir)
    blocks: list[pd.DataFrame] = []

    ehallmar = _read_csv(
        processed / "nba_games_normalized.csv",
        usecols=["game_id", "game_date", "home_team_abbr", "away_team_abbr"],
    )
    blocks.append(_source_rows(ehallmar, "ehallmar", "NBA", "game_id"))

    current = _read_csv(
        processed / "nba_current_games_normalized.csv",
        usecols=lambda c: c in {"game_id", "game_date", "home_team_abbr", "away_team_abbr"},
    )
    blocks.append(_source_rows(current, "nba_api", "NBA", "game_id"))

    snapshots = _read_csv(processed / "basketball_odds_snapshots_normalized.csv")
    zachht_games = collapse_zachht_games(snapshots)
    if not zachht_games.empty:
        zachht_games["key_version"] = GAME_KEY_VERSION
        blocks.append(zachht_games[_CANONICAL_COLUMNS])

    kalshi = _read_csv(processed / "kalshi_game_market_matches.csv")
    if not kalshi.empty and "match_status" in kalshi.columns:
        status = kalshi["match_status"].astype(str)
        kalshi = kalshi[status.str.contains("match") & ~status.eq("no_match")]
        kalshi = kalshi.drop_duplicates(subset=["game_id"])
        blocks.append(_source_rows(kalshi, "kalshi", "NBA", "game_id"))

    blocks = [block for block in blocks if not block.empty]
    if not blocks:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    table = pd.concat(blocks, ignore_index=True)
    return table.sort_values(["game_date", "canonical_game_key", "source"]).reset_index(drop=True)


def summarize_canonical_games(table: pd.DataFrame) -> dict[str, Any]:
    """Build the canonical-game-key summary report dictionary."""

    valid = table[table["canonical_game_key"].astype(str).ne("")] if not table.empty else table
    invalid_rows = int(len(table) - len(valid)) if not table.empty else 0

    by_source: dict[str, Any] = {}
    duplicates_by_source: dict[str, int] = {}
    if not table.empty:
        for source, group in valid.groupby("source"):
            dates = pd.to_datetime(group["game_date"], errors="coerce")
            by_source[str(source)] = {
                "rows": int(len(group)),
                "unique_keys": int(group["canonical_game_key"].nunique()),
                "date_range": {
                    "min": None if dates.isna().all() else dates.min().date().isoformat(),
                    "max": None if dates.isna().all() else dates.max().date().isoformat(),
                },
                "leagues": {str(k): int(v) for k, v in group["league"].value_counts().items()},
            }
            duplicates_by_source[str(source)] = int(group["canonical_game_key"].duplicated().sum())

    overlaps: dict[str, int] = {}
    if not table.empty:
        keys_by_source = {
            str(source): set(group["canonical_game_key"])
            for source, group in valid.groupby("source")
        }
        names = sorted(keys_by_source)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                overlaps[f"{first}__{second}"] = int(len(keys_by_source[first] & keys_by_source[second]))

    return {
        "report": "canonical_game_key_summary",
        "key_version": GAME_KEY_VERSION,
        "key_format": f"sport{KEY_SEPARATOR}league{KEY_SEPARATOR}YYYY-MM-DD{KEY_SEPARATOR}home_abbr{KEY_SEPARATOR}away_abbr",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": int(len(table)),
        "rows_with_invalid_key": invalid_rows,
        "by_source": by_source,
        "duplicate_keys_within_source": duplicates_by_source,
        "cross_source_key_overlap": overlaps,
        "known_nba_abbreviations": sorted(CURRENT_TEAM_ABBRS),
        "notes": [
            "Canonical key = sport|league|game_date|home_abbr|away_abbr (key version v1).",
            "NBA team tokens use the shared alias table; non-NBA leagues use compact uppercase names.",
            "zachht game dates are estimated from the closing snapshot timestamp (UTC -> US/Eastern).",
            "zachht home/away orientation assumes Pinnacle 'away vs home' listing order (team2 = home).",
            "matchup_key (teams sorted) supports joins when home/away orientation is unknown.",
        ],
        "research_only": True,
        "approved": False,
    }


def build_canonical_games_outputs(
    processed_dir: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    """Write data/processed/canonical_games.csv + the summary JSON report."""

    processed = Path(processed_dir)
    reports = Path(reports_dir)
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    table = build_canonical_games_table(processed)
    table.to_csv(processed / "canonical_games.csv", index=False)
    summary = summarize_canonical_games(table)
    (reports / "canonical_game_key_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
