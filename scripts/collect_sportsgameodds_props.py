"""SportsGameOdds player-prop collection (research-only).

Loads config/sportsgameodds.yaml, checks the monthly entity budget via
/account/usage, pulls upcoming events (player props included) for each enabled
league, saves raw responses under data/raw/sportsgameodds/player_props/,
normalizes them with src/data/sportsgameodds_prop_adapter.py, and appends them
with exact-duplicate protection to:

    data/processed/player_prop_snapshots_sportsgameodds.csv   (source-specific)
    data/processed/player_prop_snapshots_normalized.csv       (shared, all sources)

Outputs:
    data/reports/sportsgameodds_collection_summary.json
    data/reports/sportsgameodds_collection.md
    data/reports/sportsgameodds_run_history.jsonl  (append-only)

Quota-careful: 1 event ~= 1 monthly entity (tier cap 2,500/month); the run
skips itself when the remaining budget is below the configured floor.
Research-only: no models, no recommendations, no betting changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.player_prop_schema import validate_player_prop_snapshots  # noqa: E402
from data.prop_collection import (  # noqa: E402
    append_run_history,
    append_snapshots,
    apply_closing_flags,
    load_existing_snapshots,
)
from data.sportsgameodds_client import (  # noqa: E402
    API_KEY_ENV,
    SportsGameOddsClient,
    extract_items,
)
from data.sportsgameodds_prop_adapter import normalize_sportsgameodds_events  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "sportsgameodds.yaml"


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"SportsGameOdds config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SportsGameOdds config must be a mapping: {path}")
    data.setdefault("source", {})
    data.setdefault("defaults", {})
    data.setdefault("quota", {})
    data.setdefault("output", {})
    data.setdefault("leagues", {})
    return data


def monthly_entities(usage: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """(remaining, max) monthly entities from an /account/usage payload."""

    if not isinstance(usage, dict):
        return None, None
    month = (usage.get("rateLimits") or {}).get("per-month") or {}
    max_entities = pd.to_numeric(month.get("max-entities"), errors="coerce")
    current = pd.to_numeric(month.get("current-entities"), errors="coerce")
    if pd.isna(max_entities):
        return None, None  # "unlimited" or absent
    current_val = 0.0 if pd.isna(current) else float(current)
    return float(max_entities) - current_val, float(max_entities)


def write_outputs(summary: dict[str, Any], output_cfg: dict[str, Any]) -> None:
    summary_path = PROJECT_ROOT / output_cfg.get(
        "run_summary_path", "data/reports/sportsgameodds_collection_summary.json"
    )
    md_path = PROJECT_ROOT / output_cfg.get(
        "run_md_path", "data/reports/sportsgameodds_collection.md"
    )
    history_path = PROJECT_ROOT / output_cfg.get(
        "run_history_path", "data/reports/sportsgameodds_run_history.jsonl"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_md(summary), encoding="utf-8")
    append_run_history(
        history_path,
        {
            "run_id": summary.get("run_id"),
            "run_time_utc": summary.get("generated_at_utc"),
            "status": summary.get("status"),
            "snapshots_added_shared": summary.get("totals", {}).get("snapshots_added_shared"),
            "events": summary.get("totals", {}).get("events"),
            "entities_remaining_month": summary.get("quota", {}).get("entities_remaining_after"),
            "leagues": [r.get("league") for r in summary.get("leagues", []) if r.get("status") == "collected"],
        },
    )


def render_md(summary: dict[str, Any]) -> str:
    lines = ["# SportsGameOdds Collection", ""]
    lines.append(f"_Run {summary.get('run_id')} at {summary.get('generated_at_utc')}. Research-only._")
    lines.append("")
    lines.append(f"- Status: **{summary.get('status')}**")
    quota = summary.get("quota") or {}
    if quota:
        lines.append(
            f"- Monthly entities: {quota.get('entities_remaining_before', 'n/a')} remaining before run"
            f" -> {quota.get('entities_remaining_after', 'n/a')} after"
            f" (cap {quota.get('entities_max_month', 'n/a')}, guard floor {quota.get('guard_floor', 'n/a')})"
        )
    totals = summary.get("totals") or {}
    lines.append(f"- Events collected: {totals.get('events', 0)}")
    lines.append(f"- Rows normalized: {totals.get('rows_normalized', 0)}")
    lines.append(f"- Added to source CSV: {totals.get('snapshots_added_source', 0)}")
    lines.append(f"- Added to shared CSV: {totals.get('snapshots_added_shared', 0)}")
    lines.append(f"- One-sided rows (missing side kept+flagged): {totals.get('one_sided_rows', 0)}")
    lines.append(f"- Game odds excluded (never labeled props): {totals.get('game_odds_skipped', 0)}")
    lines.append(f"- Unmapped statIDs skipped: {totals.get('unmapped_stat_skipped', 0)}")
    lines.append("")
    lines.append("## Leagues")
    lines.append("")
    lines.append("| league | status | events | rows | detail |")
    lines.append("| --- | --- | --- | --- | --- |")
    for record in summary.get("leagues", []):
        lines.append(
            f"| {record.get('league')} | {record.get('status')} | {record.get('events', 0)} "
            f"| {record.get('rows', 0)} | {record.get('detail', '')} |"
        )
    validation = summary.get("validation") or {}
    lines.append("")
    lines.append(f"- Schema validation (new rows): valid={validation.get('valid')} "
                 f"errors={validation.get('errors') or []}")
    blockers = summary.get("blockers") or []
    lines.append("")
    lines.append("## Blockers / warnings")
    lines.append("")
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("_Research-only. No approved bets, no approved parlays, no recommendations._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SportsGameOdds player-prop snapshots.")
    parser.add_argument("--config", default=None, help="Path to sportsgameodds.yaml")
    parser.add_argument("--max-events", type=int, default=None, help="Override per-league event cap")
    parser.add_argument("--horizon-hours", type=float, default=None, help="Override event horizon")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only: check key+config (and usage) but collect no events.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else CONFIG_PATH)
    source_cfg = config["source"]
    defaults = config["defaults"]
    quota_cfg = config["quota"]
    output_cfg = config["output"]

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    horizon_hours = float(args.horizon_hours or defaults.get("event_horizon_hours", 36))
    default_max_events = int(args.max_events or defaults.get("max_events_per_run", 5))
    max_leagues = int(defaults.get("max_leagues_per_run", 2) or 0)
    window_minutes = float(defaults.get("closing_window_minutes", 60))
    guard_floor = float(quota_cfg.get("min_remaining_monthly_entities", 0) or 0)

    summary: dict[str, Any] = {
        "report": "sportsgameodds_collection_summary",
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "source": "sportsgameodds",
        "research_only": True,
        "approved": False,
        "leagues": [],
        "totals": {
            "events": 0, "rows_normalized": 0, "snapshots_added_source": 0,
            "snapshots_added_shared": 0, "duplicates_removed_source": 0,
            "duplicates_removed_shared": 0, "one_sided_rows": 0,
            "game_odds_skipped": 0, "unmapped_stat_skipped": 0, "raw_files_saved": 0,
        },
        "quota": {"guard_floor": guard_floor},
        "blockers": [],
    }
    blockers: list[str] = summary["blockers"]

    if not source_cfg.get("enabled", False):
        summary["status"] = "disabled"
        blockers.append("Source disabled in config/sportsgameodds.yaml; no requests made.")
        write_outputs(summary, output_cfg)
        print("SportsGameOdds source disabled; wrote skip summary.")
        return 0

    client = SportsGameOddsClient(base_url=source_cfg.get("base_url", "https://api.sportsgameodds.com/v2"))
    summary["key_detected"] = client.has_key
    if not client.has_key:
        summary["status"] = "no_key"
        blockers.append(f"{API_KEY_ENV} not set; collection skipped gracefully (no requests made).")
        write_outputs(summary, output_cfg)
        print(f"No {API_KEY_ENV}; wrote skip summary.")
        return 0

    # Quota guard: check the monthly entity budget before collecting.
    entities_remaining = entities_max = None
    if quota_cfg.get("check_usage_before_run", True):
        usage_result = client.account_usage()
        usage_items, _ = extract_items(usage_result.get("data"))
        usage = usage_items[0] if usage_items and isinstance(usage_items[0], dict) else None
        entities_remaining, entities_max = monthly_entities(usage)
        summary["quota"]["entities_remaining_before"] = entities_remaining
        summary["quota"]["entities_max_month"] = entities_max
        if not usage_result.get("ok"):
            blockers.append(f"/account/usage failed ({usage_result.get('error')}); "
                            "proceeding cautiously with config caps only.")
        elif (
            guard_floor > 0
            and entities_remaining is not None
            and entities_remaining < guard_floor
        ):
            summary["status"] = "skipped_quota_low"
            blockers.append(
                f"Monthly entity budget low ({entities_remaining:.0f} remaining < floor "
                f"{guard_floor:.0f}); collection skipped to protect closing snapshots."
            )
            write_outputs(summary, output_cfg)
            print("Entity budget below guard floor; collection skipped.")
            return 0

    if args.dry_run:
        summary["status"] = "dry_run"
        enabled = [l for l, c in config["leagues"].items() if c.get("enabled")]
        summary["dry_run_plan"] = {
            "enabled_leagues": enabled,
            "max_events_per_run": default_max_events,
            "horizon_hours": horizon_hours,
        }
        write_outputs(summary, output_cfg)
        print(f"Dry run: would collect {enabled} (<= {default_max_events} events each).")
        return 0

    raw_root = PROJECT_ROOT / output_cfg.get("raw_dir", "data/raw/sportsgameodds/player_props")
    leagues = sorted(
        ((l, c) for l, c in (config["leagues"] or {}).items()),
        key=lambda item: float(item[1].get("priority", 1000)),
    )

    frames: list[pd.DataFrame] = []
    leagues_collected = 0
    time_fmt = "%Y-%m-%dT%H:%M:%SZ"
    for league, league_cfg in leagues:
        record: dict[str, Any] = {"league": league, "sport": league_cfg.get("sport")}
        if not league_cfg.get("enabled"):
            record.update(status="skipped_disabled", events=0, rows=0)
            summary["leagues"].append(record)
            continue
        if max_leagues > 0 and leagues_collected >= max_leagues:
            record.update(status="skipped_league_cap", events=0, rows=0,
                          detail=f"max_leagues_per_run={max_leagues}")
            summary["leagues"].append(record)
            continue
        league_id = str(league_cfg.get("league_id") or league)
        league_max_events = int(league_cfg.get("max_events_per_run", default_max_events))
        result = client.events(
            leagueID=league_id,
            startsAfter=now.strftime(time_fmt),
            startsBefore=(now + timedelta(hours=horizon_hours)).strftime(time_fmt),
            oddsAvailable="true",
            limit=max(1, league_max_events),
        )
        if not result.get("ok"):
            record.update(status="error", events=0, rows=0, detail=str(result.get("error")))
            blockers.append(f"{league}: events request failed ({result.get('error')})")
            summary["leagues"].append(record)
            continue
        events, _ = extract_items(result.get("data"))
        league_raw_dir = raw_root / league
        league_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = league_raw_dir / f"{run_id}__events.json"
        counter = 1
        while raw_path.exists():  # never overwrite old snapshots
            raw_path = league_raw_dir / f"{run_id}__events_{counter}.json"
            counter += 1
        raw_path.write_text(json.dumps(result.get("data"), indent=2, default=str), encoding="utf-8")
        summary["totals"]["raw_files_saved"] += 1

        frame, stats = normalize_sportsgameodds_events(
            events,
            sport=str(league_cfg.get("sport", "")).strip().lower(),
            league=league,
            stat_map=league_cfg.get("stat_map") or None,
            raw_source_file=raw_path.relative_to(PROJECT_ROOT).as_posix(),
            run_time=now,
        )
        frames.append(frame)
        leagues_collected += 1
        record.update(
            status="collected",
            events=int(stats.get("events", 0)),
            rows=int(stats.get("rows", 0)),
            one_sided_rows=int(stats.get("one_sided_rows", 0)),
            game_odds_skipped=int(stats.get("game_odds_skipped", 0)),
            unmapped_stat_skipped=int(stats.get("unmapped_stat_skipped", 0)),
            books_seen=int(stats.get("books_seen", 0)),
            raw_file=raw_path.relative_to(PROJECT_ROOT).as_posix(),
        )
        summary["totals"]["events"] += record["events"]
        summary["totals"]["rows_normalized"] += record["rows"]
        summary["totals"]["one_sided_rows"] += record["one_sided_rows"]
        summary["totals"]["game_odds_skipped"] += record["game_odds_skipped"]
        summary["totals"]["unmapped_stat_skipped"] += record["unmapped_stat_skipped"]
        summary["leagues"].append(record)
        print(f"  {league}: {record['events']} events -> {record['rows']} rows "
              f"({record['books_seen']} books)")

    new = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    if not new.empty:
        new = apply_closing_flags(new, window_minutes)
        validation = validate_player_prop_snapshots(new)
        summary["validation"] = {
            "valid": validation["valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }
        if not validation["valid"]:
            # Never append schema-invalid rows to the shared historical CSV.
            summary["status"] = "validation_failed"
            blockers.append(
                f"New rows failed schema validation ({validation['errors']}); "
                "raw files kept, nothing appended."
            )
            write_outputs(summary, output_cfg)
            print("Validation failed; nothing appended (raw responses preserved).")
            return 1

        # Source-specific CSV (full history for this source).
        source_path = PROJECT_ROOT / output_cfg.get(
            "source_processed_path", "data/processed/player_prop_snapshots_sportsgameodds.csv"
        )
        existing_source = load_existing_snapshots(source_path)
        combined_source, dupes_source = append_snapshots(existing_source, new)
        combined_source = apply_closing_flags(combined_source, window_minutes)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        combined_source.to_csv(source_path, index=False)
        summary["totals"]["snapshots_added_source"] = int(len(combined_source) - len(existing_source))
        summary["totals"]["duplicates_removed_source"] = int(dupes_source)

        # Shared normalized CSV (all sources).
        shared_path = PROJECT_ROOT / output_cfg.get(
            "processed_path", "data/processed/player_prop_snapshots_normalized.csv"
        )
        existing_shared = load_existing_snapshots(shared_path)
        combined_shared, dupes_shared = append_snapshots(existing_shared, new)
        combined_shared = apply_closing_flags(combined_shared, window_minutes)
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        combined_shared.to_csv(shared_path, index=False)
        summary["totals"]["snapshots_added_shared"] = int(len(combined_shared) - len(existing_shared))
        summary["totals"]["duplicates_removed_shared"] = int(dupes_shared)
    else:
        summary["validation"] = {"valid": True, "errors": [], "warnings": ["no new rows"]}

    # Post-run usage (entity cost of this run).
    if quota_cfg.get("check_usage_before_run", True):
        usage_after = client.account_usage()
        after_items, _ = extract_items(usage_after.get("data"))
        usage = after_items[0] if after_items and isinstance(after_items[0], dict) else None
        remaining_after, _ = monthly_entities(usage)
        summary["quota"]["entities_remaining_after"] = remaining_after
        if remaining_after is not None and entities_remaining is not None:
            summary["quota"]["entities_spent_this_run"] = entities_remaining - remaining_after

    collected_any = any(r.get("status") == "collected" for r in summary["leagues"])
    errored = any(r.get("status") == "error" for r in summary["leagues"])
    summary["status"] = "success" if collected_any and not errored else (
        "completed_with_errors" if collected_any else ("failed" if errored else "skipped")
    )
    summary["requests_made"] = client.requests_made
    write_outputs(summary, output_cfg)
    totals = summary["totals"]
    print(f"Run {run_id}: {summary['status']} — {totals['events']} events, "
          f"+{totals['snapshots_added_shared']} shared rows "
          f"({totals['duplicates_removed_shared']} dupes removed)")
    print(f"  Entities remaining (month): {summary['quota'].get('entities_remaining_after', 'n/a')}")
    print("Research-only: no recommendations; approved bets/parlays remain blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
