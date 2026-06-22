"""NBA near-tip-off prop collection planner (research-only).

Builds a game-aware collection plan for NBA player props: which upcoming NBA
games exist in the collected snapshot store, how far from tip-off each one is,
which target collection windows (24h / 6h / 2h / 60m / 30m / 10m before tip)
already have a snapshot, which windows were missed, when to collect next, and
whether closing-line-value (CLV) measurement will be possible later.

Upcoming games come from the collected prop snapshots themselves (the Odds API
events we already pulled carry ``game_start_time``); the nba_api games table
only contains completed games, so it cannot list future tips.

Planning only: no models, no recommendations, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked. Missed windows are
reported honestly — odds from missed windows are NOT recoverable on the
current Odds API plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PLANNER_VERSION = "v1"

# How long after tip-off a game stays in the plan (so just-started games are
# still reported with their final coverage instead of silently vanishing).
POST_START_GRACE_HOURS = 12.0
# Games older than this are settled history, not planning targets.
LOOKBACK_HOURS = 24.0

# Window statuses.
WINDOW_HIT = "hit"
WINDOW_MISSED = "missed"
WINDOW_OPEN_NOW = "open_now"
WINDOW_UPCOMING = "upcoming"

# Timing classifications for "how close to tip is now".
TIMING_VERY_EARLY = "very_early"
TIMING_EARLY = "early"
TIMING_MID = "mid"
TIMING_LATE = "late"
TIMING_CLOSING_LIKE = "closing_like"
TIMING_POST_START = "post_start"


@dataclass(frozen=True)
class CollectionWindow:
    """One target pre-tip collection window.

    A snapshot "hits" the window when its minutes-to-tip falls in
    ``(band_min_minutes, band_max_minutes]``. The bands partition the pre-tip
    timeline so every snapshot credits exactly one window.
    """

    name: str
    target_minutes_before: float
    band_min_minutes: float
    band_max_minutes: float


COLLECTION_WINDOWS: tuple[CollectionWindow, ...] = (
    CollectionWindow("24h_before", 1440.0, 360.0, 2880.0),
    CollectionWindow("6h_before", 360.0, 120.0, 360.0),
    CollectionWindow("2h_before", 120.0, 60.0, 120.0),
    CollectionWindow("60m_before", 60.0, 30.0, 60.0),
    CollectionWindow("30m_before", 30.0, 10.0, 30.0),
    CollectionWindow("10m_before", 10.0, 0.0, 10.0),
)

# Windows whose snapshots count as "closing-like" for CLV (mirrors the
# collector's closing_window_minutes=60 default).
CLOSING_LIKE_MAX_MINUTES = 60.0

OUTPUT_FILES = {
    "plan_json": "nba_prop_closing_collection_plan.json",
    "plan_md": "nba_prop_closing_collection_plan.md",
    "coverage_csv": "nba_prop_closing_coverage.csv",
    "clv_readiness_json": "nba_prop_clv_readiness_summary.json",
}

COVERAGE_COLUMNS = [
    "canonical_game_key",
    "game",
    "game_start_time",
    "window",
    "window_target_time_utc",
    "window_status",
    "snapshots_in_window",
    "minutes_until_game",
]


def classify_timing(minutes_until_game: float) -> str:
    """Classify how close 'now' is to a game's tip-off."""

    if minutes_until_game <= 0:
        return TIMING_POST_START
    if minutes_until_game <= 60:
        return TIMING_CLOSING_LIKE
    if minutes_until_game <= 120:
        return TIMING_LATE
    if minutes_until_game <= 360:
        return TIMING_MID
    if minutes_until_game <= 1440:
        return TIMING_EARLY
    return TIMING_VERY_EARLY


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _game_label(game_key: str) -> str:
    """Human label 'AWAY @ HOME (date)' from a canonical game key."""

    parts = str(game_key).split("|")
    if len(parts) != 5:
        return str(game_key)
    _, _, game_date, home, away = parts
    return f"{away} @ {home} ({game_date})"


def load_nba_snapshot_games(
    snapshots: pd.DataFrame,
    now: datetime,
    *,
    lookback_hours: float = LOOKBACK_HOURS,
) -> pd.DataFrame:
    """NBA snapshot rows for games starting after ``now - lookback_hours``.

    Adds parsed ``_snap`` / ``_start`` timestamps and ``_minutes_to_tip``
    (snapshot-relative). Rows without a parseable start time are dropped with
    no error: they cannot be planned.
    """

    if snapshots.empty or "league" not in snapshots.columns:
        return pd.DataFrame()
    nba = snapshots[snapshots["league"].astype(str).str.upper().eq("NBA")].copy()
    if nba.empty:
        return nba
    nba["_snap"] = pd.to_datetime(nba.get("snapshot_time"), errors="coerce", utc=True)
    nba["_start"] = pd.to_datetime(nba.get("game_start_time"), errors="coerce", utc=True)
    nba = nba[nba["_start"].notna()]
    if nba.empty:
        return nba
    cutoff = pd.Timestamp(now) - pd.Timedelta(hours=float(lookback_hours))
    nba = nba[nba["_start"] >= cutoff]
    nba["_minutes_to_tip"] = (nba["_start"] - nba["_snap"]).dt.total_seconds() / 60.0
    return nba


def minutes_until_next_nba_game(snapshots: pd.DataFrame, now: datetime) -> float | None:
    """Minutes from ``now`` to the next known NBA tip, from collected snapshots.

    Returns None when no future NBA game is known. Used by the collector to
    decide whether a run should treat NBA closing collection as high priority.
    Only games we already collected snapshots for are visible here — a brand
    new game day needs at least one regular collection run to become known.
    """

    nba = load_nba_snapshot_games(snapshots, now, lookback_hours=0.0)
    if nba.empty:
        return None
    future = nba[nba["_start"] > pd.Timestamp(now)]
    if future.empty:
        return None
    delta = (future["_start"].min() - pd.Timestamp(now)).total_seconds() / 60.0
    return round(float(delta), 2)


def _plan_windows(
    game_rows: pd.DataFrame,
    start: pd.Timestamp,
    minutes_until_game: float,
) -> list[dict[str, Any]]:
    """Per-window status records for one game."""

    minutes = game_rows["_minutes_to_tip"].dropna()
    records: list[dict[str, Any]] = []
    for window in COLLECTION_WINDOWS:
        in_band = minutes[(minutes > window.band_min_minutes) & (minutes <= window.band_max_minutes)]
        target_time = start - pd.Timedelta(minutes=window.target_minutes_before)
        if len(in_band) > 0:
            status = WINDOW_HIT
        elif minutes_until_game <= window.band_min_minutes:
            # The last chance to land a snapshot in this band has passed.
            status = WINDOW_MISSED
        elif minutes_until_game <= window.band_max_minutes:
            status = WINDOW_OPEN_NOW
        else:
            status = WINDOW_UPCOMING
        records.append(
            {
                "window": window.name,
                "target_minutes_before": window.target_minutes_before,
                "window_target_time_utc": target_time.isoformat(),
                "window_status": status,
                "snapshots_in_window": int(len(in_band)),
            }
        )
    return records


def plan_game(game_key: str, game_rows: pd.DataFrame, now: datetime) -> dict[str, Any]:
    """Build the collection plan record for one NBA game."""

    start = game_rows["_start"].iloc[0]
    minutes_until_game = round((start - pd.Timestamp(now)).total_seconds() / 60.0, 2)
    timing = classify_timing(minutes_until_game)
    windows = _plan_windows(game_rows, start, minutes_until_game)

    hit = [w["window"] for w in windows if w["window_status"] == WINDOW_HIT]
    missed = [w["window"] for w in windows if w["window_status"] == WINDOW_MISSED]
    open_now = [w["window"] for w in windows if w["window_status"] == WINDOW_OPEN_NOW]
    upcoming = [w for w in windows if w["window_status"] == WINDOW_UPCOMING]

    collection_needed_now = bool(open_now) and minutes_until_game > 0
    if collection_needed_now:
        next_collection_time: str | None = pd.Timestamp(now).isoformat()
        next_collection_reason = f"window open now: {', '.join(open_now)}"
    elif upcoming:
        next_window = min(upcoming, key=lambda w: w["target_minutes_before"] * -1)
        # Earliest future target time = largest target_minutes_before among upcoming.
        next_collection_time = next_window["window_target_time_utc"]
        next_collection_reason = f"next target window: {next_window['window']}"
    else:
        next_collection_time = None
        next_collection_reason = (
            "game already started" if timing == TIMING_POST_START else "all windows hit or missed"
        )

    minutes = game_rows["_minutes_to_tip"].dropna()
    closing_like = minutes[(minutes > 0) & (minutes <= CLOSING_LIKE_MAX_MINUTES)]
    early = minutes[minutes > CLOSING_LIKE_MAX_MINUTES]
    has_closing_like = len(closing_like) > 0
    has_early = len(early) > 0
    closing_still_reachable = minutes_until_game > 0

    if has_early and has_closing_like:
        clv_possible = True
        clv_reason = "early and closing-like snapshots both exist"
    elif has_early and closing_still_reachable:
        clv_possible = True
        clv_reason = (
            "early snapshots exist; closing-like snapshot still reachable before tip "
            f"({minutes_until_game:.0f} minutes left)"
        )
    elif has_early:
        clv_possible = False
        clv_reason = "no closing-like snapshot was collected before tip; closing line is lost"
    else:
        clv_possible = closing_still_reachable
        clv_reason = (
            "no early snapshot exists yet"
            if closing_still_reachable
            else "no early snapshot and the game already started"
        )

    return {
        "game": _game_label(game_key),
        "canonical_game_key": game_key,
        "game_start_time": start.isoformat(),
        "minutes_until_game": minutes_until_game,
        "timing_classification": timing,
        "snapshots_total": int(len(game_rows)),
        "closing_like_snapshots": int(len(closing_like)),
        "early_snapshots": int(len(early)),
        "windows": windows,
        "windows_hit": hit,
        "windows_missed": missed,
        "windows_open_now": open_now,
        "collection_needed_now": collection_needed_now,
        "next_recommended_collection_time_utc": next_collection_time,
        "next_collection_reason": next_collection_reason,
        "clv_possible": clv_possible,
        "clv_reason": clv_reason,
    }


def build_collection_plan(
    snapshots: pd.DataFrame,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the full NBA near-tip collection plan from collected snapshots."""

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    nba = load_nba_snapshot_games(snapshots, now)
    warnings: list[str] = []
    games: list[dict[str, Any]] = []
    if nba.empty:
        warnings.append(
            "No NBA games with a known start time found in the collected snapshots. "
            "Upcoming games only become visible after at least one collection run touches them; "
            "run scripts/daily_collect_props.py first."
        )
    else:
        for game_key, rows in nba.groupby(nba["canonical_game_key"].astype(str), sort=True):
            games.append(plan_game(str(game_key), rows, now))

    upcoming = [g for g in games if g["timing_classification"] != TIMING_POST_START]
    if games and not upcoming:
        warnings.append(
            "Every known NBA game has already started; no pre-tip collection is possible for them. "
            "New game days appear after the next regular collection run."
        )
    for game in games:
        if game["windows_missed"]:
            warnings.append(
                f"{game['game']}: missed windows {game['windows_missed']} — odds from missed "
                "windows are NOT recoverable on the current Odds API plan."
            )
        if not game["clv_possible"]:
            warnings.append(f"{game['game']}: CLV will NOT be possible ({game['clv_reason']}).")

    needed_now = [g for g in games if g["collection_needed_now"]]
    next_times = [
        g["next_recommended_collection_time_utc"]
        for g in games
        if g["next_recommended_collection_time_utc"]
    ]
    next_collection = min(next_times) if next_times else None

    return {
        "report": "nba_prop_closing_collection_plan",
        "planner_version": PLANNER_VERSION,
        "generated_at_utc": pd.Timestamp(now).isoformat(),
        "windows": [
            {
                "window": w.name,
                "target_minutes_before": w.target_minutes_before,
                "band_minutes": [w.band_min_minutes, w.band_max_minutes],
            }
            for w in COLLECTION_WINDOWS
        ],
        "games": games,
        "games_total": len(games),
        "games_upcoming": len(upcoming),
        "games_needing_collection_now": [g["canonical_game_key"] for g in needed_now],
        "collection_needed_now": bool(needed_now),
        "next_recommended_collection_time_utc": next_collection,
        "minutes_until_next_nba_game": minutes_until_next_nba_game(snapshots, now),
        "warnings": warnings,
        "research_only": True,
        "approved": False,
        "notes": [
            "Planning only: no models, no recommendations.",
            "Approved bets and approved parlays remain blocked.",
            "Missed windows are unrecoverable: The Odds API has no historical odds on the current plan.",
        ],
    }


def build_coverage_frame(plan: dict[str, Any]) -> pd.DataFrame:
    """Flatten the plan to one row per game x window for the coverage CSV."""

    rows: list[dict[str, Any]] = []
    for game in plan.get("games", []):
        for window in game.get("windows", []):
            rows.append(
                {
                    "canonical_game_key": game["canonical_game_key"],
                    "game": game["game"],
                    "game_start_time": game["game_start_time"],
                    "window": window["window"],
                    "window_target_time_utc": window["window_target_time_utc"],
                    "window_status": window["window_status"],
                    "snapshots_in_window": window["snapshots_in_window"],
                    "minutes_until_game": game["minutes_until_game"],
                }
            )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_clv_readiness_summary(
    plan: dict[str, Any],
    snapshots: pd.DataFrame,
    line_quality: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize whether NBA CLV measurement is possible yet.

    ``line_quality`` (from the market-quality audit) adds per-market closing
    and main-line coverage when available; the summary degrades gracefully
    without it.
    """

    games = plan.get("games", [])
    nba_closing = 0
    last_nba_snapshot: str | None = None
    if not snapshots.empty and "league" in snapshots.columns:
        nba = snapshots[snapshots["league"].astype(str).str.upper().eq("NBA")]
        if not nba.empty:
            closing = _truthy(nba.get("is_closing_snapshot", pd.Series(dtype="object")))
            nba_closing = int(closing.sum())
            times = pd.to_datetime(nba.get("snapshot_time"), errors="coerce", utc=True)
            if times.notna().any():
                last_nba_snapshot = times.max().isoformat()

    games_with_closing = [g for g in games if g["closing_like_snapshots"] > 0]
    games_missing_closing = [g for g in games if g["closing_like_snapshots"] == 0]
    clv_capable_now = [
        g for g in games if g["closing_like_snapshots"] > 0 and g["early_snapshots"] > 0
    ]
    clv_possible_later = [g for g in games if g["clv_possible"]]

    market_stats: dict[str, Any] = {
        "nba_markets": None,
        "nba_markets_with_closing": None,
        "nba_closing_market_rate": None,
        "nba_main_line_markets_with_closing": None,
        "nba_main_line_closing_rate": None,
    }
    if line_quality is not None and not line_quality.empty and "league" in line_quality.columns:
        nba_markets = line_quality[line_quality["league"].astype(str).eq("NBA")]
        if not nba_markets.empty:
            has_closing = _truthy(nba_markets["has_closing_snapshot"])
            total = int(len(nba_markets))
            with_closing = int(has_closing.sum())
            main = nba_markets[nba_markets["likely_main_line"].notna()]
            main_with_closing = int(_truthy(main["has_closing_snapshot"]).sum()) if not main.empty else 0
            market_stats = {
                "nba_markets": total,
                "nba_markets_with_closing": with_closing,
                "nba_closing_market_rate": round(with_closing / total, 4) if total else 0.0,
                "nba_main_line_markets_with_closing": main_with_closing,
                "nba_main_line_closing_rate": (
                    round(main_with_closing / len(main), 4) if len(main) else 0.0
                ),
            }

    clv_possible_now = bool(clv_capable_now)
    if clv_possible_now:
        verdict = (
            f"CLV measurement is possible now for {len(clv_capable_now)} game(s): early and "
            "closing-like snapshots both exist."
        )
    elif clv_possible_later:
        verdict = (
            f"CLV is not possible yet ({nba_closing} NBA closing-like snapshots), but is still "
            f"achievable for {len(clv_possible_later)} game(s) if collection runs inside the "
            "closing window (60 minutes before tip)."
        )
    else:
        verdict = (
            "CLV is not possible: no NBA game currently has (or can still get) both an early "
            "and a closing-like snapshot. Keep collecting near tip-off on future game days."
        )

    return {
        "report": "nba_prop_clv_readiness_summary",
        "planner_version": PLANNER_VERSION,
        "generated_at_utc": plan["generated_at_utc"],
        "nba_closing_like_snapshots": nba_closing,
        "last_nba_snapshot_time_utc": last_nba_snapshot,
        "games_in_plan": len(games),
        "games_with_closing_snapshots": [g["canonical_game_key"] for g in games_with_closing],
        "games_missing_closing_snapshots": [g["canonical_game_key"] for g in games_missing_closing],
        "games_clv_capable_now": [g["canonical_game_key"] for g in clv_capable_now],
        "games_clv_possible_later": [g["canonical_game_key"] for g in clv_possible_later],
        "market_closing_coverage": market_stats,
        "next_recommended_collection_time_utc": plan["next_recommended_collection_time_utc"],
        "collection_needed_now": plan["collection_needed_now"],
        "clv_possible_now": clv_possible_now,
        "clv_possible_later": bool(clv_possible_later),
        "verdict": verdict,
        "warnings": plan.get("warnings", []),
        "research_only": True,
        "approved": False,
    }


def _render_plan_markdown(plan: dict[str, Any], readiness: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NBA Prop Closing Collection Plan",
        "",
        f"Generated: {plan['generated_at_utc']}",
        "",
        "_Research-only planning. No models, recommendations, approved bets, or parlays._",
        "",
        "## Target Windows",
        "",
        "| window | target (min before tip) | credit band (min before tip) |",
        "| --- | --- | --- |",
    ]
    for window in plan["windows"]:
        band = window["band_minutes"]
        lines.append(
            f"| {window['window']} | {window['target_minutes_before']:g} | "
            f"({band[0]:g}, {band[1]:g}] |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Games in plan: {plan['games_total']} ({plan['games_upcoming']} upcoming)",
        f"- Collection needed now: {'YES' if plan['collection_needed_now'] else 'no'}",
        f"- Next recommended collection (UTC): {plan['next_recommended_collection_time_utc'] or 'n/a'}",
        f"- Minutes until next NBA tip: {plan['minutes_until_next_nba_game'] if plan['minutes_until_next_nba_game'] is not None else 'unknown'}",
        f"- NBA closing-like snapshots so far: {readiness['nba_closing_like_snapshots']}",
        f"- CLV verdict: {readiness['verdict']}",
        "",
        "## Games",
        "",
    ]
    if not plan["games"]:
        lines.append("(no NBA games with known start times in the snapshot store)")
    for game in plan["games"]:
        lines += [
            f"### {game['game']}",
            "",
            f"- Canonical key: `{game['canonical_game_key']}`",
            f"- Tip (UTC): {game['game_start_time']}",
            f"- Minutes until tip: {game['minutes_until_game']:.0f} ({game['timing_classification']})",
            f"- Snapshots: {game['snapshots_total']} total, "
            f"{game['early_snapshots']} early, {game['closing_like_snapshots']} closing-like",
            f"- Windows hit: {', '.join(game['windows_hit']) or '(none)'}",
            f"- Windows missed: {', '.join(game['windows_missed']) or '(none)'}",
            f"- Windows open now: {', '.join(game['windows_open_now']) or '(none)'}",
            f"- Collect now: {'YES' if game['collection_needed_now'] else 'no'}",
            f"- Next recommended collection (UTC): {game['next_recommended_collection_time_utc'] or 'n/a'}"
            f" — {game['next_collection_reason']}",
            f"- CLV possible: {'YES' if game['clv_possible'] else 'NO'} — {game['clv_reason']}",
            "",
        ]
    if plan["warnings"]:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in plan["warnings"]]
        lines.append("")
    lines += [
        "---",
        "Research-only: this plan schedules data collection. It does not build models, create",
        "recommendations, loosen proof gates, or enable approved bets or parlays.",
        "",
    ]
    return "\n".join(lines)


def write_collection_plan_reports(
    project_root: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the plan from the snapshot store and write all four outputs."""

    root = Path(project_root)
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    snapshots_path = root / "data" / "processed" / "player_prop_snapshots_normalized.csv"
    snapshots = (
        pd.read_csv(snapshots_path, low_memory=False) if snapshots_path.exists() else pd.DataFrame()
    )

    plan = build_collection_plan(snapshots, now=now)

    line_quality_path = reports / "player_prop_line_quality.csv"
    line_quality = (
        pd.read_csv(line_quality_path, low_memory=False) if line_quality_path.exists() else None
    )
    readiness = build_clv_readiness_summary(plan, snapshots, line_quality)
    coverage = build_coverage_frame(plan)

    outputs = {key: reports / filename for key, filename in OUTPUT_FILES.items()}
    outputs["plan_json"].write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    outputs["clv_readiness_json"].write_text(
        json.dumps(readiness, indent=2, default=str), encoding="utf-8"
    )
    coverage.to_csv(outputs["coverage_csv"], index=False)
    outputs["plan_md"].write_text(_render_plan_markdown(plan, readiness), encoding="utf-8")

    plan["outputs"] = {key: str(path.relative_to(root)) for key, path in outputs.items()}
    return plan
