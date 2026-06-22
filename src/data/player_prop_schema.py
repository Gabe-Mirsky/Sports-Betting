"""Normalized schema for future player-prop odds snapshots.

Defines the one-row-per (snapshot, player, game, prop, bookmaker) layout that
every future prop source (Kalshi prop tickers, The Odds API player markets,
manual sportsbook captures) must normalize into, plus a validator and an empty
CSV template. Settlement fields (actual_stat_value / over_won / under_won) are
filled later by joining ``canonical_game_key`` + player to the nba_api actuals.

Schema/plumbing only: no model logic, no proof-gate or betting changes. No
prop betting is enabled by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "v1.1"

PROP_TYPES = (
    # Basketball (NBA/WNBA/NCAAB)
    "points",
    "rebounds",
    "assists",
    "threes",
    "steals",
    "blocks",
    "turnovers",
    "pra",  # points + rebounds + assists combos
    "points_rebounds",
    "points_assists",
    "rebounds_assists",
    "double_double",
    "triple_double",
    # Football (NFL)
    "pass_yards",
    "pass_tds",
    "rush_yards",
    "receptions",
    "receiving_yards",
    # Baseball (MLB)
    "hits",
    "home_runs",
    "total_bases",
    "pitcher_strikeouts",
    # Hockey (NHL)
    "goals",
    "shots_on_goal",
    # Soccer (EPL/MLS/La Liga/Serie A/Bundesliga/Ligue 1) — collect_only
    "shots",
    "shots_on_target",
)

HOME_AWAY_VALUES = ("home", "away")


@dataclass(frozen=True)
class PropField:
    name: str
    dtype: str  # pandas-facing dtype family: str | float | int | bool | datetime
    required: bool  # must be non-null on every row (vs nullable until settlement/optional)
    description: str
    allowed: tuple[str, ...] | None = None


PLAYER_PROP_SNAPSHOT_SCHEMA: tuple[PropField, ...] = (
    PropField("snapshot_time", "datetime", True, "UTC time the line was observed (immutable per pull)."),
    PropField("sport", "str", True, "Sport slug, e.g. 'basketball'."),
    PropField("league", "str", True, "League code, e.g. 'NBA' or 'WNBA'."),
    PropField("season", "str", True, "Season label, e.g. '2025-26'."),
    PropField("game_date", "datetime", True, "Local (US/Eastern) game date, YYYY-MM-DD."),
    PropField("game_start_time", "datetime", False, "Scheduled start/tip time (UTC) when the source provides it."),
    PropField("canonical_game_key", "str", True, "sport|league|game_date|home_abbr|away_abbr (key v1)."),
    PropField("player_name", "str", True, "Player display name as published by the source."),
    PropField("player_id", "str", False, "Stable player id (nba_api PLAYER_ID when resolvable)."),
    PropField("team", "str", False, "Player's team (canonical abbreviation); nullable for sources without team mapping."),
    PropField("opponent", "str", False, "Opposing team (canonical abbreviation); nullable for sources without team mapping."),
    PropField("home_away", "str", False, "Whether the player's team is home or away (when known).", HOME_AWAY_VALUES),
    PropField("prop_type", "str", True, "Stat market for the prop.", PROP_TYPES),
    PropField("line", "float", True, "Over/under line value (e.g. 25.5)."),
    PropField("over_price", "float", False, "Price for the over side (decimal odds preferred)."),
    PropField("under_price", "float", False, "Price for the under side (decimal odds preferred)."),
    PropField("bookmaker", "str", True, "Sportsbook / exchange that published the line."),
    PropField("source", "str", True, "Ingestion source: kalshi | odds_api | manual | ..."),
    PropField("market_id", "str", False, "Source-native market id/ticker when available."),
    PropField("is_closing_snapshot", "bool", True, "Whether this is the last snapshot before tip."),
    PropField("minutes_to_game_start", "float", False, "Minutes from snapshot_time to scheduled tip."),
    PropField("has_result", "bool", True, "Whether settlement fields have been filled from actuals."),
    PropField("actual_stat_value", "float", False, "Actual stat from player game logs (settlement)."),
    PropField("over_won", "bool", False, "Settlement: actual > line. Null until has_result."),
    PropField("under_won", "bool", False, "Settlement: actual < line. Null until has_result."),
    PropField("raw_source_file", "str", False, "Project-relative path to the raw payload this row came from."),
)

PLAYER_PROP_SNAPSHOT_COLUMNS: tuple[str, ...] = tuple(f.name for f in PLAYER_PROP_SNAPSHOT_SCHEMA)

_NUMERIC_FIELDS = {f.name for f in PLAYER_PROP_SNAPSHOT_SCHEMA if f.dtype == "float"}
_BOOL_FIELDS = {f.name for f in PLAYER_PROP_SNAPSHOT_SCHEMA if f.dtype == "bool"}
_DATETIME_FIELDS = {f.name for f in PLAYER_PROP_SNAPSHOT_SCHEMA if f.dtype == "datetime"}

_BOOL_TOKENS_TRUE = {"true", "1", "t", "yes"}
_BOOL_TOKENS_FALSE = {"false", "0", "f", "no"}


def empty_player_prop_snapshot_frame() -> pd.DataFrame:
    """Return an empty dataframe with the schema columns in canonical order."""

    return pd.DataFrame(columns=list(PLAYER_PROP_SNAPSHOT_COLUMNS))


def write_player_prop_template(path: str | Path) -> Path:
    """Write the empty player-prop snapshot CSV template (header only)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    empty_player_prop_snapshot_frame().to_csv(target, index=False)
    return target


def _is_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return str(value).strip().lower() in (_BOOL_TOKENS_TRUE | _BOOL_TOKENS_FALSE)


def validate_player_prop_snapshots(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate a player-prop snapshot frame against the v1 schema.

    Returns ``{"valid": bool, "errors": [...], "warnings": [...], "rows": n}``.
    Errors block ingestion; warnings flag suspicious but tolerable content.
    """

    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in PLAYER_PROP_SNAPSHOT_COLUMNS if c not in frame.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    extra = [c for c in frame.columns if c not in PLAYER_PROP_SNAPSHOT_COLUMNS]
    if extra:
        warnings.append(f"extra columns ignored by the schema: {extra}")

    if missing or frame.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": not errors,
            "rows": int(len(frame)),
            "errors": errors,
            "warnings": warnings,
        }

    for field in PLAYER_PROP_SNAPSHOT_SCHEMA:
        column = frame[field.name]
        if field.required:
            null_like = column.isna() | column.astype(str).str.strip().eq("")
            if null_like.any():
                errors.append(f"{field.name}: {int(null_like.sum())} null/empty values in required field")
        if field.name in _NUMERIC_FIELDS:
            coerced = pd.to_numeric(column, errors="coerce")
            bad = column.notna() & coerced.isna()
            if bad.any():
                errors.append(f"{field.name}: {int(bad.sum())} non-numeric values")
        if field.name in _DATETIME_FIELDS:
            coerced = pd.to_datetime(column, errors="coerce")
            bad = column.notna() & coerced.isna()
            if bad.any():
                errors.append(f"{field.name}: {int(bad.sum())} unparseable datetimes")
        if field.name in _BOOL_FIELDS:
            non_null = column.dropna()
            bad_bool = (~non_null.map(_is_bool_like)).sum()
            if bad_bool:
                errors.append(f"{field.name}: {int(bad_bool)} non-boolean values")
        if field.allowed is not None:
            non_null = column.dropna().astype(str).str.strip().str.lower()
            non_null = non_null[non_null.ne("")]
            invalid = sorted(set(non_null) - set(field.allowed))
            if invalid:
                if field.name == "prop_type":
                    warnings.append(f"prop_type: unknown values {invalid} (not in the canonical set)")
                else:
                    errors.append(f"{field.name}: invalid values {invalid}; allowed {list(field.allowed)}")

    # Cross-field consistency: settlement fields only when has_result is true.
    has_result = frame["has_result"].map(lambda v: str(v).strip().lower() in _BOOL_TOKENS_TRUE)
    settled = frame[has_result]
    if not settled.empty:
        missing_actual = settled["actual_stat_value"].isna().sum()
        if missing_actual:
            errors.append(f"actual_stat_value: {int(missing_actual)} rows have has_result=true but no actual")
        line = pd.to_numeric(settled["line"], errors="coerce")
        actual = pd.to_numeric(settled["actual_stat_value"], errors="coerce")
        over = settled["over_won"].map(lambda v: str(v).strip().lower() in _BOOL_TOKENS_TRUE)
        comparable = line.notna() & actual.notna() & (actual != line)
        inconsistent = (over.ne(actual > line) & comparable).sum()
        if inconsistent:
            errors.append(f"over_won: {int(inconsistent)} rows disagree with actual_stat_value vs line")

    if "canonical_game_key" in frame.columns:
        bad_keys = (~frame["canonical_game_key"].astype(str).str.count(r"\|").eq(4)).sum()
        if bad_keys:
            errors.append(f"canonical_game_key: {int(bad_keys)} keys are not 5-part v1 keys")

    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "rows": int(len(frame)),
        "errors": errors,
        "warnings": warnings,
    }


def render_schema_summary_md() -> str:
    """Render the player-prop schema report markdown."""

    lines: list[str] = ["# Player Prop Snapshot Schema (v1)", ""]
    lines.append("_Research-only. Defines the normalized layout for FUTURE player-prop odds snapshots._")
    lines.append("")
    lines.append("One row per **(snapshot_time, player, game, prop_type, bookmaker)**.")
    lines.append("Template: `data/templates/player_prop_snapshot_template.csv`.")
    lines.append("Validator: `src/data/player_prop_schema.py::validate_player_prop_snapshots`.")
    lines.append("")
    lines.append("## Columns")
    lines.append("")
    lines.append("| column | type | required | description |")
    lines.append("| --- | --- | --- | --- |")
    for field in PLAYER_PROP_SNAPSHOT_SCHEMA:
        allowed = f" Allowed: {', '.join(field.allowed)}." if field.allowed else ""
        required = "yes" if field.required else "nullable"
        lines.append(f"| {field.name} | {field.dtype} | {required} | {field.description}{allowed} |")
    lines.append("")
    lines.append("## How the pieces connect")
    lines.append("")
    lines.append("- `canonical_game_key` joins prop snapshots to `data/processed/canonical_games.csv`,")
    lines.append("  the nba_api current games/actuals, and zachht/Kalshi game markets (key v1:")
    lines.append("  `sport|league|game_date|home_abbr|away_abbr`).")
    lines.append("- Settlement: join `canonical_game_key` + `player_id` (or normalized `player_name`)")
    lines.append("  to `data/processed/nba_current_player_game_logs_normalized.csv`; fill")
    lines.append("  `actual_stat_value`, `over_won`, `under_won`, and set `has_result=true`.")
    lines.append("- CLV: keep every pull as an immutable row; the last pre-tip snapshot per market")
    lines.append("  (`is_closing_snapshot=true`, smallest `minutes_to_game_start`) is the closing line.")
    lines.append("")
    lines.append("## Source mapping notes")
    lines.append("")
    lines.append("- **Kalshi prop tickers** (KXNBAPTS/KXNBAREB/...): Yes/No contracts at a threshold;")
    lines.append("  map threshold -> `line`, yes-price -> `over_price`, no-price -> `under_price`,")
    lines.append("  `bookmaker='kalshi'`, ticker -> `market_id`.")
    lines.append("- **The Odds API player markets**: Over/Under per book; one row per bookmaker.")
    lines.append("  No true closing line on the free tier - poll near tip and mark the last pull closing.")
    lines.append("- Prices should be normalized to decimal odds at ingestion when the source uses")
    lines.append("  American odds or cents.")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- No historical player-prop lines exist in any imported dataset yet; this schema is")
    lines.append("  forward-looking collection plumbing.")
    lines.append("- Research-only. No prop betting, approved bets, or parlays are enabled.")
    lines.append("")
    return "\n".join(lines)


def build_player_prop_schema_outputs(
    templates_dir: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    """Write the template CSV + schema markdown report; return a small summary."""

    templates = Path(templates_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    template_path = write_player_prop_template(templates / "player_prop_snapshot_template.csv")
    (reports / "player_prop_schema_summary.md").write_text(render_schema_summary_md(), encoding="utf-8")

    return {
        "report": "player_prop_schema_summary",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": list(PLAYER_PROP_SNAPSHOT_COLUMNS),
        "n_columns": len(PLAYER_PROP_SNAPSHOT_COLUMNS),
        "template_path": str(template_path),
        "prop_types": list(PROP_TYPES),
        "research_only": True,
        "approved": False,
    }
