"""NBA player-prop snapshot enrichment and settlement readiness (research-only).

Joins normalized player-prop snapshots to the nba_api current games and player
game logs so each NBA snapshot carries a resolved ``player_id``, team context,
and — once the game has actuals — settlement fields (``actual_stat_value``,
``over_won``, ``under_won``, ``push``).

Matching strategy:

1. Game-scoped: (canonical_game_key, normalized player name) against the player
   game logs. This is the strongest match and the only one that settles.
2. Directory fallback: normalized player name against all logged players. A
   unique hit resolves ``player_id`` and the player's latest team; ambiguous
   names (two ids share one normalized name) stay unmatched rather than guess.

Research-only: no recommendations, no predictions, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .canonical_games import parse_canonical_game_key


ENRICHMENT_VERSION = "v1"

# Snapshot prop_type -> player-game-log stat column(s). Combo props sum columns.
SETTLEMENT_STAT_COLUMNS: dict[str, tuple[str, ...]] = {
    "points": ("points",),
    "rebounds": ("rebounds",),
    "assists": ("assists",),
    "threes": ("threes",),
    "steals": ("steals",),
    "blocks": ("blocks",),
    "turnovers": ("turnovers",),
    "points_rebounds_assists": ("points", "rebounds", "assists"),
    "pra": ("points", "rebounds", "assists"),
}

SUPPORTED_PROP_TYPES = tuple(SETTLEMENT_STAT_COLUMNS)

# Per-row settlement_status values, from best to worst.
STATUS_SETTLED = "settled"
STATUS_PENDING = "pending_result"
STATUS_UNSUPPORTED_PROP = "unsupported_prop"
STATUS_AMBIGUOUS_PLAYER = "ambiguous_player"
STATUS_UNMATCHED_PLAYER = "unmatched_player"
STATUS_NOT_NBA = "not_nba"

ENRICHMENT_COLUMNS = (
    "normalized_player_name",
    "player_matched",
    "player_match_method",
    "game_matched",
    "settlement_supported",
    "settlement_ready",
    "settlement_status",
    "push",
)

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_player_name(value: object) -> str:
    """Normalize a player name for matching across sources.

    Lowercase, accents stripped (Dončić -> doncic), punctuation removed
    (De'Aaron -> deaaron), generational suffixes dropped (Jr/Sr/II/III/IV).
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " "))
    tokens = [t for t in text.split() if t and t.rstrip(".") not in _NAME_SUFFIXES]
    return " ".join(tokens)


def _clean_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def build_player_directory(logs: pd.DataFrame) -> dict[str, Any]:
    """Index the player game logs for matching.

    Returns:
      - ``by_game``: (canonical_game_key, normalized name) -> log row dict
      - ``by_name``: normalized name -> sorted list of distinct player_ids
      - ``latest_team``: player_id -> team_abbr from the player's latest game
      - ``game_keys``: set of canonical_game_keys that have logged actuals
    """

    by_game: dict[tuple[str, str], dict[str, Any]] = {}
    by_name: dict[str, set[str]] = {}
    latest: dict[str, tuple[str, str]] = {}  # player_id -> (game_date, team_abbr)

    if logs.empty:
        return {"by_game": by_game, "by_name": {}, "latest_team": {}, "game_keys": set()}

    for row in logs.to_dict("records"):
        name_norm = normalize_player_name(row.get("player_name"))
        player_id = _clean_str(row.get("player_id"))
        game_key = _clean_str(row.get("canonical_game_key"))
        if not name_norm or not player_id:
            continue
        by_name.setdefault(name_norm, set()).add(player_id)
        if game_key:
            by_game[(game_key, name_norm)] = row
        game_date = _clean_str(row.get("game_date"))
        team = _clean_str(row.get("team_abbr"))
        if player_id not in latest or game_date >= latest[player_id][0]:
            latest[player_id] = (game_date, team)

    return {
        "by_game": by_game,
        "by_name": {name: sorted(ids) for name, ids in by_name.items()},
        "latest_team": {pid: team for pid, (_, team) in latest.items()},
        "game_keys": {key for key, _ in by_game},
    }


def compute_actual_stat(log_row: dict[str, Any], prop_type: str) -> float | None:
    """Actual stat value for a prop type from one player-game log row."""

    columns = SETTLEMENT_STAT_COLUMNS.get(str(prop_type).strip().lower())
    if columns is None:
        return None
    total = 0.0
    for column in columns:
        value = pd.to_numeric(pd.Series([log_row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return None
        total += float(value)
    return total


def settle_prop(actual: float, line: float) -> dict[str, bool]:
    """Grade one prop: over wins above the line, under below, push on the line."""

    return {
        "over_won": actual > line,
        "under_won": actual < line,
        "push": actual == line,
    }


def _derive_side_context(game_key: str, team: str) -> dict[str, str]:
    """Derive opponent/home_away from the snapshot game key and a known team."""

    try:
        parts = parse_canonical_game_key(game_key)
    except ValueError:
        return {"opponent": "", "home_away": ""}
    if team == parts["home_team"]:
        return {"opponent": parts["away_team"], "home_away": "home"}
    if team == parts["away_team"]:
        return {"opponent": parts["home_team"], "home_away": "away"}
    return {"opponent": "", "home_away": ""}


def enrich_player_prop_snapshots(
    snapshots: pd.DataFrame,
    games: pd.DataFrame,
    logs: pd.DataFrame,
) -> dict[str, Any]:
    """Enrich NBA prop snapshots with identity, game, and settlement fields.

    Returns ``{"enriched": frame, "summary": dict, "unmatched_players": frame,
    "unmatched_games": frame}``. Non-NBA rows pass through with enrichment
    columns inert (settlement_status='not_nba').
    """

    enriched = snapshots.copy().reset_index(drop=True)
    for column in ENRICHMENT_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = pd.NA
    # Filled columns load as float64 when all-empty; widen so strings/bools fit.
    for column in ("player_id", "team", "opponent", "home_away", "has_result", "over_won", "under_won"):
        if column in enriched.columns:
            enriched[column] = enriched[column].astype("object")
        else:
            enriched[column] = pd.NA

    directory = build_player_directory(logs)
    known_game_keys: set[str] = set()
    if not games.empty and "canonical_game_key" in games.columns:
        known_game_keys = {
            key for key in games["canonical_game_key"].map(_clean_str) if key
        }

    if "league" in enriched.columns:
        league = enriched["league"].map(_clean_str).str.upper()
    else:
        league = pd.Series("", index=enriched.index, dtype="object")
    is_nba = league.eq("NBA")

    unmatched_player_rows: dict[tuple[str, str], dict[str, Any]] = {}
    unmatched_game_rows: dict[str, dict[str, Any]] = {}

    for idx in enriched.index:
        if not bool(is_nba.iloc[idx]):
            enriched.at[idx, "settlement_status"] = STATUS_NOT_NBA
            enriched.at[idx, "player_matched"] = False
            enriched.at[idx, "game_matched"] = False
            enriched.at[idx, "settlement_supported"] = False
            enriched.at[idx, "settlement_ready"] = False
            continue

        row = enriched.loc[idx]
        raw_name = _clean_str(row.get("player_name"))
        name_norm = normalize_player_name(raw_name)
        game_key = _clean_str(row.get("canonical_game_key"))
        prop_type = _clean_str(row.get("prop_type")).lower()
        line = pd.to_numeric(pd.Series([row.get("line")]), errors="coerce").iloc[0]

        enriched.at[idx, "normalized_player_name"] = name_norm
        supported = prop_type in SETTLEMENT_STAT_COLUMNS and pd.notna(line)
        enriched.at[idx, "settlement_supported"] = bool(supported)

        game_matched = bool(game_key) and game_key in known_game_keys
        enriched.at[idx, "game_matched"] = game_matched
        if game_key and not game_matched:
            record = unmatched_game_rows.setdefault(
                game_key,
                {
                    "league": "NBA",
                    "canonical_game_key": game_key,
                    "game_date": _clean_str(row.get("game_date")),
                    "snapshots": 0,
                    "first_snapshot_time": _clean_str(row.get("snapshot_time")),
                    "last_snapshot_time": _clean_str(row.get("snapshot_time")),
                },
            )
            record["snapshots"] += 1
            snap_time = _clean_str(row.get("snapshot_time"))
            record["first_snapshot_time"] = min(record["first_snapshot_time"], snap_time)
            record["last_snapshot_time"] = max(record["last_snapshot_time"], snap_time)

        # 1) Game-scoped log match: identity + actuals from the same game.
        log_row = directory["by_game"].get((game_key, name_norm)) if game_key else None
        candidate_ids = directory["by_name"].get(name_norm, [])

        player_matched = False
        if log_row is not None:
            player_matched = True
            enriched.at[idx, "player_match_method"] = "game_log"
            enriched.at[idx, "player_id"] = _clean_str(log_row.get("player_id"))
            enriched.at[idx, "team"] = _clean_str(log_row.get("team_abbr"))
            enriched.at[idx, "opponent"] = _clean_str(log_row.get("opponent_abbr"))
            is_home = log_row.get("is_home")
            if pd.notna(is_home):
                enriched.at[idx, "home_away"] = (
                    "home" if str(is_home).strip().lower() in {"true", "1", "t", "yes"} else "away"
                )
        elif len(candidate_ids) == 1:
            player_matched = True
            player_id = candidate_ids[0]
            enriched.at[idx, "player_match_method"] = "name_directory"
            enriched.at[idx, "player_id"] = player_id
            team = directory["latest_team"].get(player_id, "")
            if team:
                enriched.at[idx, "team"] = team
                side = _derive_side_context(game_key, team)
                if side["home_away"]:
                    enriched.at[idx, "opponent"] = side["opponent"]
                    enriched.at[idx, "home_away"] = side["home_away"]
        else:
            reason = "ambiguous_name" if len(candidate_ids) > 1 else "no_log_history"
            record = unmatched_player_rows.setdefault(
                (raw_name, reason),
                {
                    "league": "NBA",
                    "player_name": raw_name,
                    "normalized_player_name": name_norm,
                    "reason": reason,
                    "candidate_player_ids": ";".join(candidate_ids),
                    "snapshots": 0,
                },
            )
            record["snapshots"] += 1

        enriched.at[idx, "player_matched"] = player_matched
        settlement_ready = player_matched and supported
        enriched.at[idx, "settlement_ready"] = settlement_ready

        # 2) Settlement: only from a same-game log row with the needed stats.
        actual = compute_actual_stat(log_row, prop_type) if (log_row and supported) else None
        if settlement_ready and actual is not None:
            grade = settle_prop(actual, float(line))
            enriched.at[idx, "has_result"] = True
            enriched.at[idx, "actual_stat_value"] = actual
            enriched.at[idx, "over_won"] = grade["over_won"]
            enriched.at[idx, "under_won"] = grade["under_won"]
            enriched.at[idx, "push"] = grade["push"]
            enriched.at[idx, "settlement_status"] = STATUS_SETTLED
        else:
            enriched.at[idx, "has_result"] = False
            enriched.at[idx, "push"] = pd.NA
            if not player_matched:
                enriched.at[idx, "settlement_status"] = (
                    STATUS_AMBIGUOUS_PLAYER if len(candidate_ids) > 1 else STATUS_UNMATCHED_PLAYER
                )
            elif not supported:
                enriched.at[idx, "settlement_status"] = STATUS_UNSUPPORTED_PROP
            else:
                enriched.at[idx, "settlement_status"] = STATUS_PENDING

    unmatched_players = pd.DataFrame(
        list(unmatched_player_rows.values()),
        columns=[
            "league",
            "player_name",
            "normalized_player_name",
            "reason",
            "candidate_player_ids",
            "snapshots",
        ],
    ).sort_values(["snapshots", "player_name"], ascending=[False, True]).reset_index(drop=True)

    unmatched_games = pd.DataFrame(
        list(unmatched_game_rows.values()),
        columns=[
            "league",
            "canonical_game_key",
            "game_date",
            "snapshots",
            "first_snapshot_time",
            "last_snapshot_time",
        ],
    ).sort_values(["snapshots", "canonical_game_key"], ascending=[False, True]).reset_index(drop=True)

    summary = _summarize(enriched, is_nba, unmatched_players, unmatched_games)
    return {
        "enriched": enriched,
        "summary": summary,
        "unmatched_players": unmatched_players,
        "unmatched_games": unmatched_games,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _summarize(
    enriched: pd.DataFrame,
    is_nba: pd.Series,
    unmatched_players: pd.DataFrame,
    unmatched_games: pd.DataFrame,
) -> dict[str, Any]:
    nba = enriched[is_nba]
    n_nba = int(len(nba))
    player_matched = int(nba["player_matched"].fillna(False).astype(bool).sum())
    game_matched = int(nba["game_matched"].fillna(False).astype(bool).sum())
    ready = int(nba["settlement_ready"].fillna(False).astype(bool).sum())
    settled = int(nba["settlement_status"].eq(STATUS_SETTLED).sum())
    pending = int(nba["settlement_status"].eq(STATUS_PENDING).sum())
    status_counts = {
        status: int(count)
        for status, count in nba["settlement_status"].fillna("(none)").value_counts().items()
    }
    return {
        "report": "player_prop_enrichment_summary",
        "enrichment_version": ENRICHMENT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_snapshots": int(len(enriched)),
        "nba_snapshots": n_nba,
        "player_id_matched": player_matched,
        "player_id_match_rate": _rate(player_matched, n_nba),
        "game_key_matched": game_matched,
        "game_key_match_rate": _rate(game_matched, n_nba),
        "settlement_ready": ready,
        "settled": settled,
        "pending": pending,
        "settlement_status_counts": status_counts,
        "unmatched_player_names": int(len(unmatched_players)),
        "unmatched_game_keys": int(len(unmatched_games)),
        "supported_prop_types": list(SUPPORTED_PROP_TYPES),
        "research_only": True,
        "approved": False,
    }


def run_prop_enrichment(project_root: str | Path) -> dict[str, Any]:
    """Load inputs, enrich, and write the four output files. Returns the summary."""

    root = Path(project_root)
    processed = root / "data" / "processed"
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    def _load(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    snapshots = _load(processed / "player_prop_snapshots_normalized.csv")
    games = _load(processed / "nba_current_games_normalized.csv")
    logs = _load(processed / "nba_current_player_game_logs_normalized.csv")

    result = enrich_player_prop_snapshots(snapshots, games, logs)

    enriched_path = processed / "player_prop_snapshots_enriched.csv"
    result["enriched"].to_csv(enriched_path, index=False)
    result["unmatched_players"].to_csv(reports / "player_prop_unmatched_players.csv", index=False)
    result["unmatched_games"].to_csv(reports / "player_prop_unmatched_games.csv", index=False)

    summary = result["summary"]
    summary["outputs"] = {
        "enriched_path": str(enriched_path.relative_to(root)),
        "summary_path": "data/reports/player_prop_enrichment_summary.json",
        "unmatched_players_path": "data/reports/player_prop_unmatched_players.csv",
        "unmatched_games_path": "data/reports/player_prop_unmatched_games.csv",
    }
    (reports / "player_prop_enrichment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
