"""Unified odds-source usage summary (research-only).

Aggregates, per source: snapshot counts, latest run, quota/usage, recent
errors, reliability, lifecycle status, and the primary/backup source per
league.

Outputs:
    data/reports/odds_source_usage_summary.json
    data/reports/odds_source_usage_summary.md
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_history(path: Path, limit: int = 10) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records[-limit:]


def snapshots_by_source() -> dict[str, dict]:
    if not SNAPSHOTS_PATH.exists():
        return {}
    frame = pd.read_csv(
        SNAPSHOTS_PATH,
        usecols=["source", "league", "snapshot_time", "is_closing_snapshot"],
        low_memory=False,
    )
    out: dict[str, dict] = {}
    for source, group in frame.groupby(frame["source"].astype(str)):
        closing = group["is_closing_snapshot"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"}
        )
        times = pd.to_datetime(group["snapshot_time"], errors="coerce", utc=True)
        by_league = group.groupby(group["league"].astype(str)).size().to_dict()
        out[source] = {
            "snapshots": int(len(group)),
            "closing_snapshots": int(closing.sum()),
            "latest_snapshot_utc": times.max().isoformat() if times.notna().any() else None,
            "by_league": {k: int(v) for k, v in sorted(by_league.items())},
        }
    return out


def build_summary() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    counts = snapshots_by_source()

    odds_api_run = _read_json(REPORTS_DIR / "player_prop_collection_run_summary.json")
    odds_api_quota = _read_json(REPORTS_DIR / "odds_api_quota_report.json")
    odds_api_history = _read_history(REPORTS_DIR / "prop_collection_run_history.jsonl")
    sgo_collect = _read_json(REPORTS_DIR / "sportsgameodds_collection_summary.json")
    sgo_probe = _read_json(REPORTS_DIR / "sportsgameodds_probe_summary.json")
    sgo_history = _read_history(REPORTS_DIR / "sportsgameodds_run_history.jsonl")
    apisports_probe = _read_json(REPORTS_DIR / "apisports_probe_summary.json")

    def _status(source: str) -> str:
        if source == "odds_api":
            if not odds_api_run:
                return "configured"
            return "active"
        if source == "sportsgameodds":
            if not sgo_probe.get("key_detected"):
                return "no_key"
            if counts.get("sportsgameodds", {}).get("snapshots"):
                return "active"
            return "configured"
        if source == "apisports":
            if not apisports_probe.get("key_detected"):
                return "no_key"
            if apisports_probe.get("plan_restriction"):
                return "blocked"
            return "probe_only"
        if source == "kalshi":
            return "configured"  # game markets active; prop tickers not wired
        return "configured"

    odds_api_errors = [
        record for record in (odds_api_run.get("leagues") or [])
        if record.get("status") == "error"
    ]
    sgo_errors = list(sgo_collect.get("blockers") or [])

    sources = {
        "odds_api": {
            "status": _status("odds_api"),
            "snapshots": counts.get("odds_api", {}),
            "latest_run": {
                "run_id": odds_api_run.get("run_id"),
                "status": odds_api_run.get("status"),
            },
            "recent_run_outcomes": [r.get("outcome") or r.get("status") for r in odds_api_history],
            "quota": {
                "remaining_credits": (odds_api_quota.get("usage") or {}).get("quota_remaining"),
                "assumed_monthly": (odds_api_quota.get("usage") or {}).get("assumed_monthly_quota"),
                "risk": (odds_api_quota.get("risk_assessment") or {}).get("risk"),
            },
            "errors": odds_api_errors,
            "reliability": "high (months of scheduled runs)",
        },
        "sportsgameodds": {
            "status": _status("sportsgameodds"),
            "snapshots": counts.get("sportsgameodds", {}),
            "latest_run": {
                "run_id": sgo_collect.get("run_id"),
                "status": sgo_collect.get("status"),
            },
            "recent_run_outcomes": [r.get("status") for r in sgo_history],
            "quota": {
                "monthly_entities_remaining": (sgo_collect.get("quota") or {}).get("entities_remaining_after"),
                "monthly_entities_cap": (sgo_collect.get("quota") or {}).get("entities_max_month"),
                "requests_per_minute_cap": 10,
                "note": "1 event ~= 1 entity (all props+books included); metadata endpoints cost many entities",
            },
            "errors": sgo_errors,
            "reliability": "new (first collection 2026-06-11; probe clean)",
        },
        "apisports": {
            "status": _status("apisports"),
            "snapshots": counts.get("apisports", {"snapshots": 0}),
            "latest_run": {"probe_at": apisports_probe.get("generated_at_utc")},
            "quota": {
                "requests_per_day": ((apisports_probe.get("status_by_api") or {}).get("basketball") or {}).get("requests_limit_day"),
                "plan": ((apisports_probe.get("status_by_api") or {}).get("basketball") or {}).get("plan"),
            },
            "errors": [apisports_probe.get("plan_restriction")] if apisports_probe.get("plan_restriction") else [],
            "reliability": "n/a (probe only)",
            "blocker": apisports_probe.get("plan_restriction"),
        },
        "kalshi": {
            "status": _status("kalshi"),
            "snapshots": counts.get("kalshi", {"snapshots": 0}),
            "note": "game markets + candles active elsewhere in the project; prop tickers not collected yet",
            "errors": [],
        },
    }

    # Primary/backup per league: data-driven (who has rows) + documented intent.
    leagues_seen: set[str] = set()
    for info in counts.values():
        leagues_seen.update((info.get("by_league") or {}).keys())
    per_league: dict[str, dict] = {}
    for league in sorted(leagues_seen):
        oa = (counts.get("odds_api", {}).get("by_league") or {}).get(league, 0)
        sgo = (counts.get("sportsgameodds", {}).get("by_league") or {}).get(league, 0)
        if league == "NBA":
            primary, backup = "odds_api", "sportsgameodds"
            note = ("SGO collects NBA in parallel (entity-cheap). If cross-source agreement "
                    "holds, SGO can take over NBA prop pulls to relieve Odds API credits.")
        elif sgo > 0 and oa == 0:
            primary, backup = "sportsgameodds", "odds_api"
            note = "only SGO has rows"
        elif oa > 0 and sgo == 0:
            primary, backup = "odds_api", ("sportsgameodds" if league in
                {"MLB", "NHL", "NFL", "NCAAB", "MLS", "UEFA_CL", "NCAAF"} else None)
            note = "SGO league available but disabled (entity budget reserved for NBA)" if backup else \
                   "league not on SGO tier (no WNBA/EPL/La Liga/Serie A/Bundesliga/Ligue 1)"
        else:
            primary = "odds_api" if oa >= sgo else "sportsgameodds"
            backup = "sportsgameodds" if primary == "odds_api" else "odds_api"
            note = "both sources have rows"
        per_league[league] = {
            "primary": primary, "backup": backup,
            "odds_api_rows": int(oa), "sportsgameodds_rows": int(sgo),
            "note": note,
        }

    return {
        "report": "odds_source_usage_summary",
        "generated_at_utc": now,
        "research_only": True,
        "approved": False,
        "sources": sources,
        "primary_backup_by_league": per_league,
        "headline": {
            "active_sources": [s for s, v in sources.items() if v.get("status") == "active"],
            "blocked_sources": [s for s, v in sources.items() if v.get("status") == "blocked"],
            "odds_api_quota_risk": (odds_api_quota.get("risk_assessment") or {}).get("risk"),
            "sgo_entities_remaining": (sgo_collect.get("quota") or {}).get("entities_remaining_after"),
        },
    }


def render_md(summary: dict) -> str:
    lines = ["# Odds Source Usage Summary", ""]
    lines.append(f"_Generated {summary['generated_at_utc']}. Research-only._")
    lines.append("")
    headline = summary.get("headline") or {}
    lines.append(f"- Active sources: {', '.join(headline.get('active_sources') or []) or 'none'}")
    lines.append(f"- Blocked sources: {', '.join(headline.get('blocked_sources') or []) or 'none'}")
    lines.append(f"- Odds API quota risk: {headline.get('odds_api_quota_risk')}")
    lines.append(f"- SportsGameOdds monthly entities remaining: {headline.get('sgo_entities_remaining')}")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("| source | status | snapshots | closing | latest snapshot | quota | errors |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, info in (summary.get("sources") or {}).items():
        snaps = info.get("snapshots") or {}
        quota = info.get("quota") or {}
        quota_text = "; ".join(
            f"{k}={v}" for k, v in quota.items() if v is not None and k != "note"
        ) or "n/a"
        errors = info.get("errors") or []
        lines.append(
            f"| {name} | {info.get('status')} | {snaps.get('snapshots', 0)} "
            f"| {snaps.get('closing_snapshots', 0)} | {snaps.get('latest_snapshot_utc', 'n/a')} "
            f"| {quota_text} | {len(errors)} |"
        )
    lines.append("")
    lines.append("## Primary / backup by league")
    lines.append("")
    lines.append("| league | primary | backup | odds_api rows | sgo rows | note |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for league, info in (summary.get("primary_backup_by_league") or {}).items():
        lines.append(
            f"| {league} | {info.get('primary')} | {info.get('backup') or '-'} "
            f"| {info.get('odds_api_rows')} | {info.get('sportsgameodds_rows')} | {info.get('note')} |"
        )
    lines.append("")
    lines.append("_Research-only. Approved bets/parlays remain blocked._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    summary = build_summary()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "odds_source_usage_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (REPORTS_DIR / "odds_source_usage_summary.md").write_text(render_md(summary), encoding="utf-8")
    print("Wrote odds_source_usage_summary.json/.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
