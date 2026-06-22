"""Build the cross-sport collection coverage audit (research-only).

Answers "is the collector actually set up for all five sport groups?" by
joining four sources of truth per league:

  - config/prop_collection.yaml          (configured? enabled? priority? markets?)
  - data/processed/player_prop_snapshots_normalized.csv  (what actually landed)
  - data/reports/prop_collection_run_history.jsonl       (last run statuses)
  - data/reports/odds_api_available_sports.json          (active/off-season)
  - data/raw/prop_odds/<LEAGUE>/                          (raw files saved)

Outputs:
    data/reports/cross_sport_collection_coverage_summary.json
    data/reports/cross_sport_collection_coverage.md
    data/reports/cross_sport_collection_coverage.csv

Statuses: collecting | configured_no_events | configured_skipped_quota |
configured_inactive | not_configured | error

Research-only: a data coverage audit. No models, no recommendations, no bets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "prop_collection.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "prop_odds"

# The five sport groups this project must cover and the leagues expected in each.
EXPECTED_SPORT_GROUPS: dict[str, list[str]] = {
    "basketball": ["NBA", "WNBA", "NCAAB"],
    "baseball": ["MLB"],
    "hockey": ["NHL"],
    "football": ["NFL", "NCAAF"],
    "soccer": ["EPL", "MLS", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1", "UEFA_CL"],
}

QUOTA_SKIP_STATUSES = {"skipped_league_cap", "skipped_quota_low", "skipped_quota_low_priority"}


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def league_status(
    *,
    configured: bool,
    snapshots_last_24h: int,
    last_run_status: str | None,
    sport_active: bool | None,
) -> tuple[str, str]:
    """(status, likely_reason). Pure so tests can hit every branch."""
    if not configured:
        return "not_configured", "league missing from config/prop_collection.yaml"
    if snapshots_last_24h > 0:
        return "collecting", ""
    if last_run_status == "error":
        return "error", "last collection run errored for this league - see the run log"
    if last_run_status == "skipped_no_api_key":
        return "configured_no_events", "ODDS_API_KEY was not visible to the last run"
    if last_run_status in QUOTA_SKIP_STATUSES:
        return (
            "configured_skipped_quota",
            f"skipped by quota guard ({last_run_status}): lower-priority leagues yield "
            "requests first; collects again when quota/league-cap allows",
        )
    if sport_active is False:
        return (
            "configured_inactive",
            "sport is inactive/off-season on The Odds API; events return empty until "
            "the season starts",
        )
    return (
        "configured_no_events",
        "sport is active but no prop-bearing events fell inside the event horizon "
        "(books may not have posted player props yet)",
    )


def build_coverage(
    config: dict,
    snapshots: pd.DataFrame,
    last_run: dict,
    discovery: dict,
    raw_file_counts: dict[str, int],
    now: datetime,
) -> dict:
    """Assemble the coverage summary from pre-loaded inputs. Testable."""
    leagues_cfg = config.get("leagues") or {}
    quota_cfg = config.get("quota") or {}
    defaults = config.get("defaults") or {}
    league_statuses = last_run.get("league_statuses") or {}
    active_by_league = {
        e["league"]: bool(e.get("active"))
        for e in (discovery.get("configured_sport_keys") or [])
    }

    snaps = snapshots if not snapshots.empty else pd.DataFrame(columns=["league"])
    times = (
        pd.to_datetime(snaps.get("snapshot_time"), errors="coerce", utc=True)
        if not snapshots.empty
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    cutoff = now - timedelta(hours=24)

    rows = []
    warnings: list[str] = []
    for sport_group, leagues in EXPECTED_SPORT_GROUPS.items():
        for league in leagues:
            cfg = leagues_cfg.get(league) or {}
            configured = bool(cfg)
            odds_cfg = ((cfg.get("sources") or {}).get("odds_api") or {})
            markets = sorted(set((odds_cfg.get("markets") or {}).values()))

            if not snapshots.empty:
                mask = snaps["league"].astype(str).eq(league)
                total = int(mask.sum())
                league_times = times[mask].dropna()
                last_24h = int((league_times >= cutoff).sum())
                latest = league_times.max().isoformat() if not league_times.empty else None
                prop_types = sorted(
                    snaps.loc[mask, "prop_type"].dropna().astype(str).unique().tolist()
                ) if "prop_type" in snaps.columns else []
                bookmakers = sorted(
                    snaps.loc[mask, "bookmaker"].dropna().astype(str).unique().tolist()
                ) if "bookmaker" in snaps.columns else []
            else:
                total, last_24h, latest, prop_types, bookmakers = 0, 0, None, [], []

            last_status = league_statuses.get(f"{league}/odds_api")
            status, reason = league_status(
                configured=configured,
                snapshots_last_24h=last_24h,
                last_run_status=last_status,
                sport_active=active_by_league.get(league),
            )
            rows.append({
                "sport_group": sport_group,
                "league": league,
                "status": status,
                "likely_reason": reason,
                "configured": configured,
                "enabled": bool(cfg.get("enabled")) if configured else False,
                "collect_only": bool(cfg.get("collect_only")) if configured else None,
                "modeling_priority": bool(cfg.get("modeling_priority")) if configured else None,
                "priority": cfg.get("priority"),
                "sport_key": odds_cfg.get("sport_key"),
                "configured_markets": markets,
                "max_events_per_run": cfg.get(
                    "max_events_per_run", defaults.get("max_events_per_league_per_run")
                ) if configured else None,
                "last_run_status": last_status,
                "quota_blocked_last_run": last_status in QUOTA_SKIP_STATUSES,
                "snapshots_total": total,
                "snapshots_last_24h": last_24h,
                "latest_snapshot_time": latest,
                "prop_types_collected": prop_types,
                "bookmakers_collected": bookmakers,
                "raw_files_saved": raw_file_counts.get(league, 0),
            })

    by_status: dict[str, list[str]] = {}
    for row in rows:
        by_status.setdefault(row["status"], []).append(row["league"])

    # Group-level coverage: a group is "covered" when every expected league is
    # configured; "collecting" when at least one league produced snapshots in 24h.
    groups = []
    for sport_group, leagues in EXPECTED_SPORT_GROUPS.items():
        group_rows = [r for r in rows if r["sport_group"] == sport_group]
        collecting = [r["league"] for r in group_rows if r["status"] == "collecting"]
        not_configured = [r["league"] for r in group_rows if not r["configured"]]
        groups.append({
            "sport_group": sport_group,
            "leagues_expected": leagues,
            "all_configured": not not_configured,
            "collecting_leagues": collecting,
            "missing_from_config": not_configured,
        })
        if sport_group in {"football", "soccer"} and not collecting:
            statuses = {r["league"]: r["status"] for r in group_rows}
            warnings.append(
                f"{sport_group}: no league is currently collecting ({statuses}). "
                "Configured and ready; snapshots will appear when events with player "
                "props enter the horizon."
            )
    for row in rows:
        if row["status"] == "not_configured":
            warnings.append(f"{row['league']} is expected for {row['sport_group']} but missing from config.")
        if row["configured"] and not row["enabled"]:
            warnings.append(f"{row['league']} is configured but disabled.")

    return {
        "report": "cross_sport_collection_coverage",
        "generated_at_utc": now.isoformat(),
        "sport_groups": groups,
        "leagues": rows,
        "leagues_by_status": by_status,
        "quota_guards": {
            "min_remaining_requests": quota_cfg.get("min_remaining_requests"),
            "low_priority_min_remaining": quota_cfg.get("low_priority_min_remaining"),
            "max_leagues_per_run": defaults.get("max_leagues_per_run"),
            "note": (
                "NBA (priority 1, modeling_priority) always collects first. Lower-priority "
                "and collect-only leagues are skipped first under quota pressure or the "
                "per-run league cap, but stay configured so they resume automatically."
            ),
        },
        "warnings": warnings,
        "research_only": True,
        "approved": False,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Cross-Sport Collection Coverage",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Sport groups",
        "",
        "| group | expected leagues | all configured | collecting now |",
        "| --- | --- | --- | --- |",
    ]
    for g in summary["sport_groups"]:
        lines.append(
            f"| {g['sport_group']} | {', '.join(g['leagues_expected'])} | "
            f"{'yes' if g['all_configured'] else 'NO: missing ' + ', '.join(g['missing_from_config'])} | "
            f"{', '.join(g['collecting_leagues']) or '(none)'} |"
        )
    lines += [
        "",
        "## Leagues",
        "",
        "| league | group | status | snapshots | last 24h | latest | prop types | books | priority | quota-blocked |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary["leagues"]:
        lines.append(
            f"| {r['league']} | {r['sport_group']} | {r['status']} | {r['snapshots_total']} | "
            f"{r['snapshots_last_24h']} | {r['latest_snapshot_time'] or '-'} | "
            f"{', '.join(r['prop_types_collected']) or '-'} | {len(r['bookmakers_collected'])} | "
            f"{r['priority'] if r['priority'] is not None else '-'} | "
            f"{'YES' if r['quota_blocked_last_run'] else 'no'} |"
        )
    lines += ["", "## Zero-snapshot leagues (likely reasons)", ""]
    zero = [r for r in summary["leagues"] if r["snapshots_total"] == 0]
    if zero:
        for r in zero:
            lines.append(f"- **{r['league']}** ({r['status']}): {r['likely_reason']}")
    else:
        lines.append("- (none - every league has collected at least one snapshot)")
    lines += ["", "## Quota guards", ""]
    guards = summary["quota_guards"]
    lines += [
        f"- Skip everything below {guards['min_remaining_requests']} requests remaining.",
        f"- Skip collect-only leagues below {guards['low_priority_min_remaining']} remaining.",
        f"- At most {guards['max_leagues_per_run']} leagues collect per run (priority order).",
        f"- {guards['note']}",
        "",
        "## Warnings",
        "",
    ]
    if summary["warnings"]:
        lines += [f"- {w}" for w in summary["warnings"]]
    else:
        lines.append("- (none)")
    lines += [
        "",
        "---",
        "Research-only coverage audit. No models, no recommendations; approved bets/parlays remain blocked.",
        "",
    ]
    return "\n".join(lines)


def count_raw_files() -> dict[str, int]:
    counts: dict[str, int] = {}
    if RAW_DIR.exists():
        for league_dir in RAW_DIR.iterdir():
            if league_dir.is_dir():
                counts[league_dir.name] = sum(1 for p in league_dir.rglob("*") if p.is_file())
    return counts


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_last_run(path: Path) -> dict:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the cross-sport collection coverage audit.")
    parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    snapshots = (
        pd.read_csv(SNAPSHOTS_PATH, low_memory=False) if SNAPSHOTS_PATH.exists() else pd.DataFrame()
    )
    last_run = _read_last_run(REPORTS_DIR / "prop_collection_run_history.jsonl")
    discovery = _read_json(REPORTS_DIR / "odds_api_available_sports.json")

    summary = build_coverage(
        config or {}, snapshots, last_run, discovery, count_raw_files(),
        datetime.now(timezone.utc),
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "cross_sport_collection_coverage_summary.json"
    md_path = REPORTS_DIR / "cross_sport_collection_coverage.md"
    csv_path = REPORTS_DIR / "cross_sport_collection_coverage.csv"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    frame = pd.DataFrame(summary["leagues"])
    frame["configured_markets"] = frame["configured_markets"].map(lambda v: "|".join(v))
    frame["prop_types_collected"] = frame["prop_types_collected"].map(lambda v: "|".join(v))
    frame["bookmakers_collected"] = frame["bookmakers_collected"].map(lambda v: "|".join(v))
    frame.to_csv(csv_path, index=False)

    for r in summary["leagues"]:
        print(
            f"  {r['league']:<11} {r['sport_group']:<11} {r['status']:<26} "
            f"total={r['snapshots_total']:<6} last24h={r['snapshots_last_24h']}"
        )
    for w in summary["warnings"]:
        print(f"WARNING: {w}")
    print(f"Wrote: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {csv_path.relative_to(PROJECT_ROOT)}")
    print("Research-only coverage audit; approved bets/parlays remain blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
