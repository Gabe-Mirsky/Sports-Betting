"""NBA results refresh + player-prop settlement loop (research-only).

Re-imports the nba_api actuals (optionally re-downloading the raw caches
first), reruns the NBA prop snapshot enrichment, and tracks how settlement
moved: pending before vs after, newly settled rows, and what is still waiting
on unplayed games.

Outputs (in addition to the four enrichment files refreshed by
``run_prop_enrichment``):

- ``data/reports/player_prop_settlement_refresh_summary.json``
- ``data/reports/player_prop_newly_settled.csv``

Research-only: no recommendations, no predictions, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .nba_current_actuals import (
    DEFAULT_MIN_SEASON,
    discover_cached_files,
    import_nba_current_actuals,
)
from .prop_enrichment import STATUS_PENDING, STATUS_SETTLED, run_prop_enrichment


REFRESH_VERSION = "v1"

REFRESH_SUMMARY_FILENAME = "player_prop_settlement_refresh_summary.json"
NEWLY_SETTLED_FILENAME = "player_prop_newly_settled.csv"

# Columns that identify one snapshot row across enrichment reruns. The
# normalized snapshot store is append-only and deduplicated, so this tuple is
# stable between refreshes.
SNAPSHOT_KEY_COLUMNS = (
    "snapshot_time",
    "canonical_game_key",
    "player_name",
    "prop_type",
    "line",
    "bookmaker",
    "source",
    "market_id",
)

NEWLY_SETTLED_COLUMNS = (
    "snapshot_time",
    "canonical_game_key",
    "game_date",
    "player_name",
    "team",
    "prop_type",
    "line",
    "actual_stat_value",
    "over_won",
    "under_won",
    "push",
    "bookmaker",
)


def snapshot_key_series(frame: pd.DataFrame) -> pd.Series:
    """Stable per-row identity key for comparing enrichment runs."""

    if frame.empty:
        return pd.Series(dtype="object")
    parts = []
    for column in SNAPSHOT_KEY_COLUMNS:
        if column in frame.columns:
            parts.append(frame[column].astype(str).str.strip())
        else:
            parts.append(pd.Series("", index=frame.index, dtype="object"))
    return parts[0].str.cat(parts[1:], sep="|")


def refresh_raw_caches(
    raw_dir: str | Path,
    min_season: int = DEFAULT_MIN_SEASON,
    max_season: int | None = None,
) -> None:
    """Re-download the nba_api team + player game-log caches (network)."""

    from .nba_client import download_seasons
    from .player_client import download_player_seasons

    end_season = max_season if max_season is not None else min_season + 1
    for season_type in ("Regular Season", "Playoffs"):
        download_seasons(
            min_season, end_season,
            cache_dir=Path(raw_dir) / "nba",
            season_type=season_type, force=True,
        )
        download_player_seasons(
            min_season, end_season,
            cache_dir=Path(raw_dir) / "nba" / "player",
            season_type=season_type, force=True,
        )


def _status_by_key(enriched: pd.DataFrame) -> dict[str, str]:
    if enriched.empty or "settlement_status" not in enriched.columns:
        return {}
    keys = snapshot_key_series(enriched)
    statuses = enriched["settlement_status"].fillna("").astype(str)
    return dict(zip(keys, statuses))


def _count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].fillna("(missing)").astype(str).replace("", "(missing)").value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _unsettled_games(pending: pd.DataFrame) -> list[dict[str, Any]]:
    """Pending NBA snapshots grouped by game, for the dashboard."""

    if pending.empty or "canonical_game_key" not in pending.columns:
        return []
    games: list[dict[str, Any]] = []
    grouped = pending.groupby(pending["canonical_game_key"].fillna("").astype(str))
    for game_key, rows in grouped:
        dates = rows.get("game_date", pd.Series(dtype="object")).dropna().astype(str)
        games.append(
            {
                "canonical_game_key": game_key or "(missing)",
                "game_date": dates.iloc[0] if not dates.empty else "",
                "pending_snapshots": int(len(rows)),
                "players": int(rows.get("player_name", pd.Series(dtype="object")).nunique()),
            }
        )
    games.sort(key=lambda g: (g["game_date"], g["canonical_game_key"]))
    return games


def run_results_refresh(
    project_root: str | Path,
    download: bool = False,
    min_season: int = DEFAULT_MIN_SEASON,
    max_season: int | None = None,
    downloader: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Refresh NBA actuals, rerun prop enrichment, and track settlement deltas.

    Cache-only by default; ``download=True`` re-fetches the raw nba_api caches
    first (``downloader`` overrides the network fetch, for tests). Returns the
    refresh summary dict, also written to
    ``data/reports/player_prop_settlement_refresh_summary.json``.
    """

    root = Path(project_root)
    raw_dir = root / "data" / "raw"
    processed = root / "data" / "processed"
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    enriched_path = processed / "player_prop_snapshots_enriched.csv"
    before = pd.read_csv(enriched_path) if enriched_path.exists() else pd.DataFrame()
    status_before = _status_by_key(before)
    pending_before = sum(1 for status in status_before.values() if status == STATUS_PENDING)
    settled_before_keys = {key for key, status in status_before.items() if status == STATUS_SETTLED}
    pending_before_keys = {key for key, status in status_before.items() if status == STATUS_PENDING}

    if download:
        (downloader or refresh_raw_caches)(raw_dir, min_season=min_season, max_season=max_season)

    # Rebuild the normalized actuals only when raw caches exist; an empty
    # import would otherwise wipe the current games/logs tables.
    have_caches = bool(discover_cached_files(raw_dir, "team")) and bool(
        discover_cached_files(raw_dir, "player")
    )
    if have_caches:
        import_summary = import_nba_current_actuals(raw_dir, processed, reports, min_season=min_season)
        actuals_import = {
            "status": "ok",
            "games_rows": import_summary["games"]["rows"],
            "player_log_rows": import_summary["player_game_logs"]["rows"],
            "games_date_range": import_summary["games"]["date_range"],
            "player_logs_date_range": import_summary["player_game_logs"]["date_range"],
        }
    else:
        actuals_import = {"status": "skipped_no_raw_caches"}

    enrichment_summary = run_prop_enrichment(root)

    after = pd.read_csv(enriched_path) if enriched_path.exists() else pd.DataFrame()
    after_keys = snapshot_key_series(after)
    if not after.empty and "settlement_status" in after.columns:
        status_after = after["settlement_status"].fillna("").astype(str)
    else:
        status_after = pd.Series(dtype="object")

    settled_mask = status_after.eq(STATUS_SETTLED) if not after.empty else pd.Series(dtype=bool)
    pending_mask = status_after.eq(STATUS_PENDING) if not after.empty else pd.Series(dtype=bool)
    settled_rows = after[settled_mask] if not after.empty else pd.DataFrame()
    pending_rows = after[pending_mask] if not after.empty else pd.DataFrame()

    newly_settled_rows = (
        settled_rows[~after_keys[settled_mask].isin(settled_before_keys)]
        if not settled_rows.empty
        else pd.DataFrame()
    )
    still_pending = (
        int(after_keys[pending_mask].isin(pending_before_keys).sum()) if not pending_rows.empty else 0
    )

    newly_settled_out = pd.DataFrame(
        {
            column: (
                newly_settled_rows[column]
                if column in newly_settled_rows.columns
                else pd.Series(pd.NA, index=newly_settled_rows.index, dtype="object")
            )
            for column in NEWLY_SETTLED_COLUMNS
        }
        if not newly_settled_rows.empty
        else {column: pd.Series(dtype="object") for column in NEWLY_SETTLED_COLUMNS}
    )
    newly_settled_out.to_csv(reports / NEWLY_SETTLED_FILENAME, index=False)

    summary: dict[str, Any] = {
        "report": "player_prop_settlement_refresh_summary",
        "refresh_version": REFRESH_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "download" if download else "cache_only",
        "actuals_import": actuals_import,
        "enrichment": {
            "nba_snapshots": enrichment_summary["nba_snapshots"],
            "player_id_matched": enrichment_summary["player_id_matched"],
            "player_match_rate": enrichment_summary["player_id_match_rate"],
            "game_key_matched": enrichment_summary["game_key_matched"],
            "game_match_rate": enrichment_summary["game_key_match_rate"],
            "settlement_ready": enrichment_summary["settlement_ready"],
        },
        "settlement": {
            "pending_before_refresh": pending_before,
            "pending_after_refresh": int(pending_mask.sum()) if not after.empty else 0,
            "newly_settled": int(len(newly_settled_rows)),
            "still_pending": still_pending,
            "settled_total": int(settled_mask.sum()) if not after.empty else 0,
            "settled_by_prop_type": _count_by(settled_rows, "prop_type"),
            "settled_by_game": _count_by(settled_rows, "canonical_game_key"),
            "newly_settled_by_prop_type": _count_by(newly_settled_rows, "prop_type"),
            "newly_settled_by_game": _count_by(newly_settled_rows, "canonical_game_key"),
            "unsettled_games": _unsettled_games(pending_rows),
        },
        "outputs": {
            "refresh_summary_path": f"data/reports/{REFRESH_SUMMARY_FILENAME}",
            "newly_settled_path": f"data/reports/{NEWLY_SETTLED_FILENAME}",
            "enriched_path": "data/processed/player_prop_snapshots_enriched.csv",
            "enrichment_summary_path": "data/reports/player_prop_enrichment_summary.json",
            "unmatched_players_path": "data/reports/player_prop_unmatched_players.csv",
            "unmatched_games_path": "data/reports/player_prop_unmatched_games.csv",
        },
        "research_only": True,
        "approved": False,
    }
    (reports / REFRESH_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
