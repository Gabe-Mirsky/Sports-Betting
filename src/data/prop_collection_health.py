"""Prop-collection health reporting (research-only).

Reads the append-only run history (``data/reports/prop_collection_run_history.jsonl``),
backfills runs that predate the history file from their run logs, and writes a
health summary JSON + markdown report covering run counts, missed days, stale
leagues, API-key/quota signals, and an overall healthy/unhealthy verdict.

Reporting only: no models, no recommendations, no proof-gate or betting changes.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .prop_collection import (
    _OUTPUT_DEFAULTS,
    _CATCH_UP_DEFAULTS,
    load_run_history,
    run_outcome,
)


# Leagues with no snapshots newer than this many days are flagged as stale.
DEFAULT_RECENT_DAYS = 3
# Grace on top of catch_up.max_gap_hours before the last success counts as late.
GAP_GRACE_HOURS = 2.0

_LOG_LEAGUE_RE = re.compile(
    r"^(?P<league>[A-Za-z0-9_]+)(?:/(?P<source>[A-Za-z0-9_]+))?: "
    r"(?P<rest>collected \((?P<snapshots>\d+) snapshots from \d+ events\)|skipped.*|ERROR .*)$"
)


def _parse_run_log(log_path: Path) -> dict[str, Any] | None:
    """Recover a minimal history record from one run log (pre-history runs)."""

    match = re.match(r"run_(?P<run_id>\d{8}T\d{6}Z)\.log$", log_path.name)
    if not match:
        return None
    run_id = match.group("run_id")
    try:
        run_time = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

    league_statuses: dict[str, str] = {}
    snapshots_by_league: dict[str, int] = {}
    errors: list[str] = []
    snapshots_collected = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        # Strip the leading ISO timestamp.
        parts = line.split(" ", 1)
        message = parts[1] if len(parts) == 2 else line
        hit = _LOG_LEAGUE_RE.match(message)
        if not hit or hit.group("league") in {"run", "WARNING"}:
            continue
        league = hit.group("league")
        source = hit.group("source") or "-"
        rest = hit.group("rest")
        if rest.startswith("collected"):
            status = "collected"
            count = int(hit.group("snapshots") or 0)
            snapshots_by_league[league] = snapshots_by_league.get(league, 0) + count
            snapshots_collected += count
        elif rest.startswith("ERROR"):
            status = "error"
            errors.append(f"{league}: {rest}")
        else:
            status = "skipped"
        league_statuses[f"{league}/{source}"] = status

    statuses = [{"status": s} for s in league_statuses.values()]
    return {
        "run_id": run_id,
        "run_time_utc": run_time.isoformat(),
        "outcome": run_outcome(statuses),
        "api_key_detected": None,  # unknown for backfilled runs
        "quota_remaining_requests": None,
        "likely_quota_issue": False,
        "snapshots_collected": snapshots_collected,
        "snapshots_by_league": snapshots_by_league,
        "league_statuses": league_statuses,
        "errors": errors,
        "warnings": [],
        "run_log": log_path.as_posix(),
        "backfilled_from_log": True,
    }


def load_full_run_history(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    """History JSONL plus log-backfilled runs that predate it, sorted by time."""

    root = Path(project_root)
    output_cfg = config.get("output") or {}
    history_path = root / output_cfg.get("run_history_path", _OUTPUT_DEFAULTS["run_history_path"])
    log_dir = root / output_cfg.get("run_log_dir", _OUTPUT_DEFAULTS["run_log_dir"])

    records = load_run_history(history_path)
    known_run_ids = {str(r.get("run_id")) for r in records}
    if log_dir.exists():
        for log_path in sorted(log_dir.glob("run_*.log")):
            record = _parse_run_log(log_path)
            if record and record["run_id"] not in known_run_ids:
                record["run_log"] = log_path.relative_to(root).as_posix()
                records.append(record)
                known_run_ids.add(record["run_id"])
    records.sort(key=lambda r: str(r.get("run_time_utc", "")))
    return records


def _league_sport_map(config: dict[str, Any]) -> dict[str, str]:
    return {
        league: str(cfg.get("sport", "unknown")).strip().lower()
        for league, cfg in (config.get("leagues") or {}).items()
    }


def build_health_summary(
    config: dict[str, Any],
    project_root: str | Path,
    *,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> dict[str, Any]:
    """Build the prop-collection health summary dict."""

    root = Path(project_root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    env = os.environ if env is None else env

    output_cfg = config.get("output") or {}
    processed_path = root / output_cfg.get("processed_path", _OUTPUT_DEFAULTS["processed_path"])
    catch_up_cfg = config.get("catch_up") or {}
    max_gap_hours = float(catch_up_cfg.get("max_gap_hours", _CATCH_UP_DEFAULTS["max_gap_hours"]))

    history = load_full_run_history(config, root)

    def _times(outcome: str | None = None) -> list[pd.Timestamp]:
        out: list[pd.Timestamp] = []
        for record in history:
            if outcome is not None and record.get("outcome") != outcome:
                continue
            parsed = pd.to_datetime(record.get("run_time_utc"), errors="coerce", utc=True)
            if not pd.isna(parsed):
                out.append(pd.Timestamp(parsed))
        return out

    success_times = _times("success")
    failed_times = _times("failed")
    all_times = _times()

    last_success = max(success_times).isoformat() if success_times else None
    last_failure = max(failed_times).isoformat() if failed_times else None

    total_runs = len(history)
    successful_runs = sum(1 for r in history if r.get("outcome") == "success")
    failed_runs = sum(1 for r in history if r.get("outcome") == "failed")
    skipped_runs = sum(1 for r in history if r.get("outcome") == "skipped")

    snapshots_by_run = [
        {
            "run_id": r.get("run_id"),
            "run_time_utc": r.get("run_time_utc"),
            "outcome": r.get("outcome"),
            "snapshots_collected": int(r.get("snapshots_collected", 0) or 0),
            "snapshots_by_league": r.get("snapshots_by_league") or {},
        }
        for r in history
    ]

    # Days with no collection: calendar days (UTC) between the first run and
    # today where no run happened at all.
    days_with_no_collection: list[str] = []
    if all_times:
        run_days = {t.date() for t in all_times}
        day = min(run_days)
        today = now.date()
        while day <= today:
            if day not in run_days:
                days_with_no_collection.append(day.isoformat())
            day = day + timedelta(days=1)

    # Snapshot totals by sport/league come from the normalized CSV (authoritative).
    snapshots_by_sport: dict[str, int] = {}
    snapshots_by_league: dict[str, int] = {}
    latest_snapshot_by_league: dict[str, str] = {}
    if processed_path.exists():
        snaps = pd.read_csv(processed_path, low_memory=False)
        if not snaps.empty:
            sports = snaps.get("sport", pd.Series(dtype="object")).fillna("unknown").astype(str)
            snapshots_by_sport = sports.value_counts().astype(int).to_dict()
            leagues = snaps.get("league", pd.Series(dtype="object")).fillna("unknown").astype(str)
            snapshots_by_league = leagues.value_counts().astype(int).to_dict()
            times = pd.to_datetime(snaps.get("snapshot_time"), errors="coerce", utc=True)
            latest = times.groupby(leagues).max()
            latest_snapshot_by_league = {
                league: ts.isoformat() for league, ts in latest.items() if not pd.isna(ts)
            }

    # Leagues with no recent snapshots (enabled leagues only). Off-season
    # leagues will show up here; that is honest, not a bug.
    cutoff = pd.Timestamp(now) - pd.Timedelta(days=recent_days)
    leagues_with_no_recent_snapshots: list[dict[str, Any]] = []
    for league, league_cfg in (config.get("leagues") or {}).items():
        if not league_cfg.get("enabled"):
            continue
        latest_iso = latest_snapshot_by_league.get(league)
        latest_ts = pd.to_datetime(latest_iso, errors="coerce", utc=True) if latest_iso else None
        if latest_ts is None or pd.isna(latest_ts) or latest_ts < cutoff:
            leagues_with_no_recent_snapshots.append(
                {
                    "league": league,
                    "sport": str(league_cfg.get("sport", "unknown")),
                    "latest_snapshot_utc": latest_iso,
                    "note": "no snapshots yet" if latest_iso is None else f"stale (> {recent_days}d old)",
                }
            )

    sources_cfg = config.get("sources") or {}
    odds_api_cfg = sources_cfg.get("odds_api") or {}
    api_key_env = odds_api_cfg.get("api_key_env", "ODDS_API_KEY")
    api_key_detected = bool(odds_api_cfg.get("api_key") or env.get(api_key_env, ""))

    latest_run = history[-1] if history else None
    likely_quota_issue = bool(latest_run.get("likely_quota_issue")) if latest_run else False
    latest_run_log = str(latest_run.get("run_log")) if latest_run else None
    latest_errors = list(latest_run.get("errors") or []) if latest_run else []
    latest_warnings = list(latest_run.get("warnings") or []) if latest_run else []

    # Healthy verdict, with explicit reasons for every failure mode.
    reasons: list[str] = []
    if not api_key_detected:
        reasons.append(f"{api_key_env} is not set: collection runs but skips every Odds API league.")
    if not history:
        reasons.append("No collection runs recorded yet.")
    else:
        if latest_run and latest_run.get("outcome") == "failed":
            reasons.append("Latest run had league errors.")
        if latest_run and latest_run.get("outcome") == "skipped":
            reasons.append("Latest run collected nothing (all leagues skipped).")
        if not success_times:
            reasons.append("No successful collection run yet.")
        else:
            age_hours = (pd.Timestamp(now) - max(success_times)).total_seconds() / 3600.0
            if age_hours > max_gap_hours + GAP_GRACE_HOURS:
                reasons.append(
                    f"Last successful collection was {age_hours:.1f}h ago "
                    f"(limit {max_gap_hours:.0f}h + {GAP_GRACE_HOURS:.0f}h grace)."
                )
    if likely_quota_issue:
        reasons.append("Latest run shows a likely Odds API quota issue.")
    healthy = not reasons

    return {
        "report": "prop_collection_health",
        "generated_at_utc": now.isoformat(),
        "healthy": healthy,
        "health_reasons": reasons,
        "last_successful_collection_utc": last_success,
        "last_failed_collection_utc": last_failure,
        "runs": {
            "total": total_runs,
            "successful": successful_runs,
            "failed": failed_runs,
            "skipped": skipped_runs,
        },
        "snapshots_by_run": snapshots_by_run,
        "snapshots_by_sport": snapshots_by_sport,
        "snapshots_by_league": snapshots_by_league,
        "days_with_no_collection": days_with_no_collection,
        "missed_days_count": len(days_with_no_collection),
        "leagues_with_no_recent_snapshots": leagues_with_no_recent_snapshots,
        "recent_days_threshold": recent_days,
        "api_key_detected": api_key_detected,
        "likely_quota_issue": likely_quota_issue,
        "latest_run": {
            "run_id": latest_run.get("run_id") if latest_run else None,
            "run_time_utc": latest_run.get("run_time_utc") if latest_run else None,
            "outcome": latest_run.get("outcome") if latest_run else None,
            "snapshots_collected": int(latest_run.get("snapshots_collected", 0) or 0) if latest_run else 0,
        },
        "latest_run_log": latest_run_log,
        "latest_errors": latest_errors,
        "latest_warnings": latest_warnings,
        "honesty_note": (
            "Missed collection windows cannot be backfilled: The Odds API does not "
            "provide historical odds on the current plan. Gaps are permanent."
        ),
        "research_only": True,
        "approved": False,
    }


def render_health_md(summary: dict[str, Any]) -> str:
    """Render the health summary as a small markdown report."""

    runs = summary.get("runs", {})
    latest = summary.get("latest_run", {})
    lines: list[str] = [
        "# Prop Collection Health",
        "",
        f"_Generated {summary.get('generated_at_utc', 'n/a')} (research-only; no recommendations)._",
        "",
        f"## Status: {'HEALTHY' if summary.get('healthy') else 'UNHEALTHY'}",
        "",
    ]
    for reason in summary.get("health_reasons", []):
        lines.append(f"- {reason}")
    if summary.get("healthy"):
        lines.append("- All checks passed.")
    lines += [
        "",
        "## Overview",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| Last successful collection | {summary.get('last_successful_collection_utc') or 'never'} |",
        f"| Last failed collection | {summary.get('last_failed_collection_utc') or 'never'} |",
        f"| Total runs | {runs.get('total', 0)} |",
        f"| Successful runs | {runs.get('successful', 0)} |",
        f"| Failed runs | {runs.get('failed', 0)} |",
        f"| Skipped runs (nothing collected) | {runs.get('skipped', 0)} |",
        f"| Days with no collection | {summary.get('missed_days_count', 0)} |",
        f"| API key detected | {summary.get('api_key_detected')} |",
        f"| Likely quota issue | {summary.get('likely_quota_issue')} |",
        f"| Latest run | {latest.get('run_id') or 'n/a'} ({latest.get('outcome') or 'n/a'}) |",
        f"| Latest run log | {summary.get('latest_run_log') or 'n/a'} |",
        "",
        "## Snapshots By Sport",
        "",
        "| sport | snapshots |",
        "| --- | --- |",
    ]
    for sport, count in sorted(summary.get("snapshots_by_sport", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {sport} | {count} |")
    if not summary.get("snapshots_by_sport"):
        lines.append("| (none) | 0 |")
    lines += [
        "",
        "## Snapshots By Run",
        "",
        "| run | time (UTC) | outcome | snapshots |",
        "| --- | --- | --- | --- |",
    ]
    for record in summary.get("snapshots_by_run", [])[-20:]:
        lines.append(
            f"| {record.get('run_id')} | {record.get('run_time_utc')} | "
            f"{record.get('outcome')} | {record.get('snapshots_collected')} |"
        )
    lines += ["", "## Days With No Collection", ""]
    missed = summary.get("days_with_no_collection", [])
    if missed:
        for day in missed[-30:]:
            lines.append(f"- {day}")
    else:
        lines.append("- None since collection started.")
    lines += ["", "## Leagues With No Recent Snapshots", ""]
    stale = summary.get("leagues_with_no_recent_snapshots", [])
    if stale:
        lines.append(f"(threshold: {summary.get('recent_days_threshold')} days; off-season leagues appear here)")
        lines.append("")
        for entry in stale:
            lines.append(
                f"- {entry.get('league')} ({entry.get('sport')}): {entry.get('note')}"
                + (f", latest {entry.get('latest_snapshot_utc')}" if entry.get("latest_snapshot_utc") else "")
            )
    else:
        lines.append("- All enabled leagues have recent snapshots.")
    lines += ["", "## Latest Errors / Warnings", ""]
    if summary.get("latest_errors") or summary.get("latest_warnings"):
        for err in summary.get("latest_errors", []):
            lines.append(f"- ERROR: {err}")
        for warning in summary.get("latest_warnings", []):
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- None in the latest run.")
    lines += [
        "",
        "## Honesty Note",
        "",
        f"{summary.get('honesty_note', '')}",
        "",
        "_Research-only: no models, no recommendations; approved bets/parlays remain blocked._",
        "",
    ]
    return "\n".join(lines)


def write_health_reports(
    config: dict[str, Any],
    project_root: str | Path,
    *,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> dict[str, Any]:
    """Write prop_collection_health_summary.json + prop_collection_health.md."""

    root = Path(project_root)
    summary = build_health_summary(config, root, now=now, env=env, recent_days=recent_days)
    reports_dir = root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "prop_collection_health_summary.json"
    md_path = reports_dir / "prop_collection_health.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_health_md(summary), encoding="utf-8")
    summary["outputs"] = {
        "summary_json": json_path.relative_to(root).as_posix(),
        "summary_md": md_path.relative_to(root).as_posix(),
    }
    return summary
