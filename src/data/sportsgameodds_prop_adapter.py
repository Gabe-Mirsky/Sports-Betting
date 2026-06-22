"""Normalize SportsGameOdds event payloads into the player-prop schema.

Maps the SGO v2 /events response (odds keyed by oddID =
``statID-statEntityID-periodID-betTypeID-sideID``) onto the project's
one-row-per (snapshot, player, game, prop, bookmaker) layout
(``src/data/player_prop_schema.py``).

Rules enforced here:
- Only full-game over/under odds whose statEntityID is a playerID become rows;
  game/team odds (statEntityID in home/away/all) are never labeled player props.
- Over and under sides are merged per (player, stat, bookmaker, line); when a
  book lists different lines for the two sides, each side keeps its own row
  with the missing side left null (counted as one_sided in the stats).
- Alternate/different lines across books are separate rows — never dropped.
- American odds strings are converted to decimal odds at ingestion.

Research-only: no models, no recommendations, no betting changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .player_prop_schema import PLAYER_PROP_SNAPSHOT_COLUMNS
from .prop_collection import (
    GAME_LOCAL_TIMEZONE,
    infer_season_label,
)
from .canonical_games import build_canonical_game_key


SOURCE_NAME = "sportsgameodds"

# statEntityID tokens that mean team/game-level odds, never player props.
TEAM_ENTITY_TOKENS = {"home", "away", "all", "side1", "side2"}

# Full-game period only: the schema's prop types settle against full-game
# actuals, so quarter/half props must not be mislabeled.
GAME_PERIOD_ID = "game"
OVER_UNDER_BET_TYPE = "ou"

# SGO statID -> canonical prop_type (src/data/player_prop_schema.py PROP_TYPES).
# Per-league overrides come from config; these cover the sports we collect.
DEFAULT_STAT_MAP: dict[str, str] = {
    # Basketball (NBA/NCAAB)
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threePointersMade": "threes",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "points+rebounds+assists": "pra",
    "points+rebounds": "points_rebounds",
    "points+assists": "points_assists",
    "rebounds+assists": "rebounds_assists",
    # Baseball (MLB)
    "batting_hits": "hits",
    "batting_homeRuns": "home_runs",
    "batting_totalBases": "total_bases",
    "pitching_strikeouts": "pitcher_strikeouts",
    # Hockey (NHL)
    "goals": "goals",
    "shotsOnGoal": "shots_on_goal",
    # Football (NFL/NCAAF)
    "passing_yards": "pass_yards",
    "passing_touchdowns": "pass_tds",
    "rushing_yards": "rush_yards",
    "receiving_receptions": "receptions",
    "receiving_yards": "receiving_yards",
}


def american_to_decimal(value: Any) -> float | None:
    """Convert an American odds string/number ('+114', '-134') to decimal odds."""

    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.lower() in {"even", "ev"}:
        return 2.0
    try:
        american = float(text.replace("+", ""))
    except ValueError:
        return None
    if american == 0:
        return None
    if american > 0:
        return round(1.0 + american / 100.0, 6)
    return round(1.0 + 100.0 / abs(american), 6)


def _iso_utc(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    # Floor to whole seconds for one uniform ISO format: pandas chokes on a
    # column mixing fractional and non-fractional second strings.
    return pd.Timestamp(parsed).floor("s").isoformat()


def _game_local_date(start_time_utc: Any) -> str:
    parsed = pd.to_datetime(start_time_utc, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).tz_convert(GAME_LOCAL_TIMEZONE).date().isoformat()


def _safe_game_key(sport: str, league: str, game_date: str, home: Any, away: Any) -> str:
    try:
        return build_canonical_game_key(sport, league, game_date, home, away)
    except ValueError:
        return ""


def is_player_prop_odd(odd: dict[str, Any], odd_id: str = "") -> bool:
    """True when an odds object is a full-game player over/under prop.

    Player props are identified by statEntityID/playerID being a real player
    id (never the home/away/all team tokens). Game odds (moneyline, spread,
    team totals) are excluded so they can never be mislabeled as player props.
    """

    if not isinstance(odd, dict):
        return False
    if str(odd.get("betTypeID") or "").strip().lower() != OVER_UNDER_BET_TYPE:
        return False
    if str(odd.get("periodID") or "").strip().lower() != GAME_PERIOD_ID:
        return False
    entity = str(odd.get("statEntityID") or "").strip()
    if not entity and odd_id:
        parts = str(odd_id).split("-")
        entity = parts[1] if len(parts) >= 5 else ""
    if not entity or entity.lower() in TEAM_ENTITY_TOKENS:
        return False
    # playerID, when present, must agree with statEntityID.
    player_id = str(odd.get("playerID") or "").strip()
    if player_id and player_id != entity:
        return False
    return True


def _base_odd_id(odd: dict[str, Any], odd_id: str) -> str:
    """Side-independent market id: statID-statEntityID-periodID-betTypeID."""

    stat = str(odd.get("statID") or "")
    entity = str(odd.get("statEntityID") or "")
    period = str(odd.get("periodID") or "")
    bet_type = str(odd.get("betTypeID") or "")
    if stat and entity and period and bet_type:
        return f"{stat}-{entity}-{period}-{bet_type}"
    parts = str(odd_id).split("-")
    return "-".join(parts[:-1]) if len(parts) >= 5 else str(odd_id)


def normalize_sportsgameodds_event(
    event: dict[str, Any],
    *,
    sport: str,
    league: str,
    stat_map: dict[str, str] | None = None,
    raw_source_file: str = "",
    run_time: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize one SGO event into schema rows + per-event adapter stats.

    Returns ``(frame, stats)`` where stats counts what was kept/skipped:
    ``player_prop_odds, game_odds_skipped, unmapped_stat_skipped, rows,
    one_sided_rows, books_seen``.
    """

    stats = {
        "player_prop_odds": 0,
        "game_odds_skipped": 0,
        "unmapped_stat_skipped": 0,
        "no_line_skipped": 0,
        "rows": 0,
        "one_sided_rows": 0,
        "books_seen": 0,
    }
    if not isinstance(event, dict):
        return pd.DataFrame(columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS)), stats

    mapping = dict(DEFAULT_STAT_MAP)
    mapping.update(stat_map or {})

    event_id = str(event.get("eventID") or "")
    status = event.get("status") or {}
    start_iso = _iso_utc(status.get("startsAt"))
    game_date = _game_local_date(status.get("startsAt"))
    teams = event.get("teams") or {}
    home_team = (teams.get("home") or {})
    away_team = (teams.get("away") or {})
    home_name = ((home_team.get("names") or {}).get("long")) or home_team.get("teamID")
    away_name = ((away_team.get("names") or {}).get("long")) or away_team.get("teamID")
    home_abbr = ((home_team.get("names") or {}).get("short")) or ""
    away_abbr = ((away_team.get("names") or {}).get("short")) or ""
    home_team_id = str(home_team.get("teamID") or "")
    away_team_id = str(away_team.get("teamID") or "")
    game_key = _safe_game_key(sport, league, game_date, home_name, away_name) if game_date else ""
    season = infer_season_label(league, game_date)
    fallback_snapshot = _iso_utc(run_time) if run_time is not None else ""

    players = event.get("players") or {}

    def player_team_context(player_id: str) -> tuple[str, Any, Any, Any]:
        """(display_name, team_abbr, opponent_abbr, home_away) for a playerID."""

        record = players.get(player_id) if isinstance(players, dict) else None
        name = ""
        team = opponent = home_away = pd.NA
        if isinstance(record, dict):
            name = str(record.get("name") or "").strip()
            team_id = str(record.get("teamID") or "")
            if team_id and team_id == home_team_id:
                team, opponent, home_away = home_abbr or pd.NA, away_abbr or pd.NA, "home"
            elif team_id and team_id == away_team_id:
                team, opponent, home_away = away_abbr or pd.NA, home_abbr or pd.NA, "away"
        if not name:
            # Fall back to the playerID slug: FIRSTNAME_LASTNAME_1_NBA.
            tokens = [t for t in player_id.split("_") if t and not t.isdigit()]
            if tokens and tokens[-1].upper() == str(league).upper():
                tokens = tokens[:-1]
            name = " ".join(t.capitalize() for t in tokens)
        return name, team, opponent, home_away

    # group: (player_id, base_odd_id) -> {"over": odd, "under": odd}
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    odds = event.get("odds") or {}
    if isinstance(odds, dict):
        odds_iter = odds.items()
    elif isinstance(odds, list):
        odds_iter = ((str(o.get("oddID") or ""), o) for o in odds if isinstance(o, dict))
    else:
        odds_iter = ()
    for odd_id, odd in odds_iter:
        if not isinstance(odd, dict):
            continue
        if not is_player_prop_odd(odd, odd_id):
            stats["game_odds_skipped"] += 1
            continue
        stats["player_prop_odds"] += 1
        stat_id = str(odd.get("statID") or "").strip()
        if stat_id not in mapping:
            stats["unmapped_stat_skipped"] += 1
            continue
        player_id = str(odd.get("playerID") or odd.get("statEntityID") or "").strip()
        side = str(odd.get("sideID") or "").strip().lower()
        if side not in {"over", "under"}:
            continue
        slot = grouped.setdefault((player_id, _base_odd_id(odd, odd_id)), {})
        slot[side] = odd

    rows: list[dict[str, Any]] = []
    books_seen: set[str] = set()
    for (player_id, base_id), sides in grouped.items():
        over_odd = sides.get("over") or {}
        under_odd = sides.get("under") or {}
        stat_id = str((over_odd or under_odd).get("statID") or "")
        prop_type = mapping.get(stat_id, stat_id)
        player_name, team, opponent, home_away = player_team_context(player_id)
        if not player_name:
            continue

        # bookmaker -> {"over": (line, price, updated), "under": (...)}
        per_book: dict[str, dict[str, tuple[float, float | None, str]]] = {}
        for side_name, odd in (("over", over_odd), ("under", under_odd)):
            for book, quote in (odd.get("byBookmaker") or {}).items():
                if not isinstance(quote, dict):
                    continue
                if quote.get("available") is False:
                    continue
                line = pd.to_numeric(quote.get("overUnder"), errors="coerce")
                if pd.isna(line):
                    stats["no_line_skipped"] += 1
                    continue
                price = american_to_decimal(quote.get("odds"))
                updated = _iso_utc(quote.get("lastUpdatedAt")) or fallback_snapshot
                per_book.setdefault(str(book), {})[side_name] = (float(line), price, updated)

        for book, quotes in per_book.items():
            books_seen.add(book)
            over_quote = quotes.get("over")
            under_quote = quotes.get("under")
            # Same line on both sides -> one merged row; different lines ->
            # one row per side with the other side null (flagged one-sided).
            line_pairs: list[tuple[float, tuple | None, tuple | None]] = []
            if over_quote and under_quote and over_quote[0] == under_quote[0]:
                line_pairs.append((over_quote[0], over_quote, under_quote))
            else:
                if over_quote:
                    line_pairs.append((over_quote[0], over_quote, None))
                if under_quote:
                    line_pairs.append((under_quote[0], None, under_quote))
            for line, over_q, under_q in line_pairs:
                snapshot_candidates = [q[2] for q in (over_q, under_q) if q and q[2]]
                snapshot_iso = max(snapshot_candidates) if snapshot_candidates else fallback_snapshot
                one_sided = (over_q is None) or (under_q is None)
                if one_sided:
                    stats["one_sided_rows"] += 1
                rows.append(
                    {
                        "snapshot_time": snapshot_iso,
                        "sport": sport,
                        "league": league,
                        "season": season,
                        "game_date": game_date,
                        "game_start_time": start_iso,
                        "canonical_game_key": game_key,
                        "player_name": player_name,
                        "player_id": pd.NA,  # nba_api id is filled by enrichment
                        "team": team,
                        "opponent": opponent,
                        "home_away": home_away,
                        "prop_type": prop_type,
                        "line": float(line),
                        "over_price": over_q[1] if over_q and over_q[1] is not None else pd.NA,
                        "under_price": under_q[1] if under_q and under_q[1] is not None else pd.NA,
                        "bookmaker": book,
                        "source": SOURCE_NAME,
                        "market_id": f"{event_id}:{base_id}",
                        "is_closing_snapshot": False,
                        "minutes_to_game_start": pd.NA,
                        "has_result": False,
                        "actual_stat_value": pd.NA,
                        "over_won": pd.NA,
                        "under_won": pd.NA,
                        "raw_source_file": raw_source_file,
                    }
                )

    stats["rows"] = len(rows)
    stats["books_seen"] = len(books_seen)
    frame = pd.DataFrame(rows, columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS))
    return frame, stats


def normalize_sportsgameodds_events(
    events: list[dict[str, Any]],
    *,
    sport: str,
    league: str,
    stat_map: dict[str, str] | None = None,
    raw_source_file: str = "",
    run_time: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize a list of SGO events; aggregates per-event adapter stats."""

    frames: list[pd.DataFrame] = []
    totals: dict[str, int] = {}
    for event in events or []:
        frame, stats = normalize_sportsgameodds_event(
            event,
            sport=sport,
            league=league,
            stat_map=stat_map,
            raw_source_file=raw_source_file,
            run_time=run_time,
        )
        if not frame.empty:
            frames.append(frame)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + int(value)
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS))
    )
    totals["events"] = len(events or [])
    return combined, totals
