"""Build the Odds API quota and cost protection report (research-only).

Uses the per-run history (data/reports/prop_collection_run_history.jsonl) —
which records the x-requests-remaining response header — plus
config/prop_collection.yaml to estimate:

  - API runs today / this month
  - snapshots collected per run and per league
  - estimated requests consumed per run (from quota header deltas)
  - quota remaining and risk of running out before the month ends
  - recommended max runs per day and league priorities

Outputs:
    data/reports/odds_api_quota_report.json
    data/reports/odds_api_quota_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
RUN_HISTORY_PATH = REPORTS_DIR / "prop_collection_run_history.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config" / "prop_collection.yaml"

# The Odds API free tier. If the observed remaining ever exceeds this, the
# plan is bigger and we scale the assumption up to the observed ceiling.
ASSUMED_MONTHLY_QUOTA = 500


def load_runs() -> list[dict]:
    if not RUN_HISTORY_PATH.exists():
        return []
    runs = []
    for line in RUN_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs


def _run_time(run: dict) -> datetime | None:
    raw = run.get("run_time_utc")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def analyze_runs(runs: list[dict], now: datetime) -> dict:
    dated = [(r, _run_time(r)) for r in runs]
    dated = [(r, t) for r, t in dated if t is not None]

    runs_today = [r for r, t in dated if t.date() == now.date()]
    runs_this_month = [r for r, t in dated if (t.year, t.month) == (now.year, now.month)]
    collected_this_month = [r for r in runs_this_month if r.get("outcome") == "success"]

    # Requests per run from consecutive quota-header deltas (same plan cycle:
    # remaining must decrease; an increase means the quota reset).
    quota_points = [
        (t, float(r["quota_remaining_requests"]))
        for r, t in sorted(dated, key=lambda rt: rt[1])
        if r.get("quota_remaining_requests") is not None
    ]
    deltas = []
    for (_, prev), (_, cur) in zip(quota_points, quota_points[1:]):
        if cur < prev:
            deltas.append(prev - cur)
    avg_requests_per_run = round(sum(deltas) / len(deltas), 1) if deltas else None

    quota_remaining = quota_points[-1][1] if quota_points else None
    max_observed_remaining = max((q for _, q in quota_points), default=None)
    monthly_quota = ASSUMED_MONTHLY_QUOTA
    if max_observed_remaining is not None and max_observed_remaining > monthly_quota:
        monthly_quota = int(max_observed_remaining)

    # Snapshots per run / per league (successful runs only).
    snaps_per_run = [int(r.get("snapshots_collected", 0) or 0) for r in collected_this_month]
    avg_snapshots_per_run = round(sum(snaps_per_run) / len(snaps_per_run), 1) if snaps_per_run else 0
    league_totals: dict[str, int] = {}
    league_runs: dict[str, int] = {}
    for r in collected_this_month:
        for league, count in (r.get("snapshots_by_league") or {}).items():
            league_totals[league] = league_totals.get(league, 0) + int(count or 0)
            league_runs[league] = league_runs.get(league, 0) + 1
    leagues_consuming = [
        {
            "league": league,
            "snapshots_this_month": total,
            "runs_collected": league_runs.get(league, 0),
            "avg_snapshots_per_run": round(total / league_runs[league], 1) if league_runs.get(league) else 0,
        }
        for league, total in sorted(league_totals.items(), key=lambda kv: -kv[1])
    ]

    return {
        "runs_today": len(runs_today),
        "runs_this_month": len(runs_this_month),
        "collected_runs_this_month": len(collected_this_month),
        "avg_requests_per_run": avg_requests_per_run,
        "requests_per_run_observations": len(deltas),
        "avg_snapshots_per_run": avg_snapshots_per_run,
        "quota_remaining": quota_remaining,
        "assumed_monthly_quota": monthly_quota,
        "estimated_requests_used": (
            round(monthly_quota - quota_remaining, 1) if quota_remaining is not None else None
        ),
        "leagues_consuming_requests": leagues_consuming,
    }


def assess_risk(usage: dict, config: dict, now: datetime) -> dict:
    remaining = usage["quota_remaining"]
    per_run = usage["avg_requests_per_run"]

    # Days left in the calendar month (the actual Odds API cycle is rolling from
    # signup; calendar month is the conservative documented assumption here).
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1)
    else:
        next_month = now.replace(month=now.month + 1, day=1)
    days_left = max((next_month.date() - now.date()).days, 1)

    if remaining is None or per_run is None:
        return {
            "risk": "unknown",
            "days_left_in_cycle_assumed": days_left,
            "detail": "Not enough quota-header observations yet to estimate burn rate.",
            "recommended_max_runs_per_day": None,
        }

    safe_budget = remaining * 0.9  # keep a 10% reserve
    sustainable_runs_total = safe_budget / per_run
    recommended_per_day = round(sustainable_runs_total / days_left, 1)

    runs_per_day_recent = usage["collected_runs_this_month"] / max(now.day, 1)
    projected_need = runs_per_day_recent * per_run * days_left
    if projected_need > remaining:
        risk = "high"
    elif projected_need > remaining * 0.7:
        risk = "medium"
    else:
        risk = "low"

    quota_cfg = config.get("quota") or {}
    warnings = []
    if recommended_per_day < 1:
        warnings.append(
            f"Sustainable pace is below 1 full multi-league run/day (~{sustainable_runs_total:.0f} runs "
            f"left for {days_left} days). The scheduled every-4-hours + 7 NBA pregame tasks will hit the "
            f"built-in quota floors (skip-all below {quota_cfg.get('min_remaining_requests')}, collect-only "
            f"leagues below {quota_cfg.get('low_priority_min_remaining')}) within days. NBA keeps collecting "
            "longest (modeling priority); to stretch quota, lower max_events_per_league_per_run or disable "
            "inactive leagues in config/prop_collection.yaml."
        )
    return {
        "risk": risk,
        "days_left_in_cycle_assumed": days_left,
        "projected_requests_needed_at_current_pace": round(projected_need, 0),
        "recent_collected_runs_per_day": round(runs_per_day_recent, 2),
        "sustainable_runs_remaining_total": round(sustainable_runs_total, 1),
        "recommended_max_runs_per_day": max(recommended_per_day, 0.0),
        "reserve_kept": "10%",
        "config_min_remaining_guard": quota_cfg.get("min_remaining_requests"),
        "config_low_priority_guard": quota_cfg.get("low_priority_min_remaining"),
        "warnings": warnings,
        "detail": (
            f"~{per_run} requests/run, {remaining:.0f} remaining, {days_left} day(s) assumed left "
            f"in the cycle -> ~{sustainable_runs_total:.0f} full runs total (~{recommended_per_day}/day) "
            "keeps a 10% reserve. Near-tip NBA-prioritized runs cost less than full multi-league runs."
        ),
    }


def recommend_league_priority(usage: dict, config: dict) -> dict:
    leagues_cfg = config.get("leagues") or {}
    activity = {row["league"]: row for row in usage["leagues_consuming_requests"]}
    rows = []
    for league, cfg in leagues_cfg.items():
        if not (cfg or {}).get("enabled", False):
            continue
        act = activity.get(league, {})
        snapshots = act.get("snapshots_this_month", 0)
        rows.append({
            "league": league,
            "configured_priority": (cfg or {}).get("priority"),
            "modeling_priority": bool((cfg or {}).get("modeling_priority")),
            "collect_only": bool((cfg or {}).get("collect_only")),
            "snapshots_this_month": snapshots,
            "recommendation": (
                "keep first (modeling priority)" if (cfg or {}).get("modeling_priority")
                else "keep collecting" if snapshots > 0
                else "inactive - keep capped/last (no snapshots this month)"
            ),
        })
    rows.sort(key=lambda r: (r["configured_priority"] is None, r["configured_priority"]))

    soccer = [r for r in rows if r["league"] in {"EPL", "MLS", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1", "UEFA_CL"}]
    soccer_active = any(r["snapshots_this_month"] > 0 for r in soccer)
    return {
        "league_priority": rows,
        "soccer_should_remain_capped": not soccer_active,
        "soccer_note": (
            "Soccer produced snapshots this month; keep caps but they are earning their requests."
            if soccer_active
            else "All soccer leagues returned 0 snapshots this month (inactive/off-season): keep them "
            "capped at 3 events/run and lowest priority so they are skipped first under quota pressure."
        ),
    }


def sportsgameodds_offload_note() -> dict:
    """Can SportsGameOdds reduce Odds API pressure? (evidence-based note)."""

    try:
        sgo = json.loads(
            (REPORTS_DIR / "sportsgameodds_collection_summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        sgo = {}
    quota = sgo.get("quota") or {}
    collected = any(r.get("status") == "collected" for r in sgo.get("leagues") or [])
    return {
        "sportsgameodds_active": collected,
        "sgo_entities_remaining_month": quota.get("entities_remaining_after"),
        "can_reduce_odds_api_usage": bool(collected),
        "explanation": (
            "SportsGameOdds collects NBA player props at ~1 entity per event with ALL prop "
            "markets and books included in one request, vs The Odds API charging credits per "
            "event per market-region. Once cross-source agreement is verified "
            "(data/reports/cross_source_prop_comparison.md), NBA prop pulls can shift to "
            "SportsGameOdds and the Odds API NBA event cap can be lowered; keep The Odds API "
            "for WNBA/EPL/La Liga/Serie A/Bundesliga/Ligue 1, which the SGO tier lacks."
            if collected else
            "SportsGameOdds has not collected successfully yet; no offload recommended."
        ),
    }


def render_markdown(summary: dict) -> str:
    u = summary["usage"]
    r = summary["risk_assessment"]
    lines = [
        "# Odds API Quota & Cost Protection Report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Usage",
        "",
        f"- Runs today: {u['runs_today']}",
        f"- Runs this month: {u['runs_this_month']} ({u['collected_runs_this_month']} actually collected)",
        f"- Avg snapshots per collected run: {u['avg_snapshots_per_run']}",
        f"- Estimated requests per run: {u['avg_requests_per_run']} (from {u['requests_per_run_observations']} header delta(s))",
        f"- Quota remaining (x-requests-remaining header): {u['quota_remaining']}",
        f"- Assumed monthly quota: {u['assumed_monthly_quota']}",
        f"- Estimated requests used this cycle: {u['estimated_requests_used']}",
        "",
        "## Risk",
        "",
        f"- Risk of running out: **{r['risk'].upper()}**",
        f"- {r.get('detail', '')}",
        f"- Recommended max collection runs/day: {r.get('recommended_max_runs_per_day')} "
        f"(~{r.get('sustainable_runs_remaining_total')} full runs left at the current cost)",
        f"- Config guards: skip-all below {r.get('config_min_remaining_guard')} remaining; "
        f"skip collect-only leagues below {r.get('config_low_priority_guard')}.",
        "",
    ]
    for w in r.get("warnings") or []:
        lines += [f"> **Warning:** {w}", ""]
    lines += [
        "## Leagues consuming requests (this month)",
        "",
        "| league | snapshots | runs | avg/run |",
        "| --- | --- | --- | --- |",
    ]
    for row in u["leagues_consuming_requests"]:
        lines.append(
            f"| {row['league']} | {row['snapshots_this_month']} | {row['runs_collected']} | {row['avg_snapshots_per_run']} |"
        )
    lines += ["", "## Recommended league priority", "", "| league | priority | recommendation |", "| --- | --- | --- |"]
    for row in summary["league_recommendations"]["league_priority"]:
        lines.append(f"| {row['league']} | {row['configured_priority']} | {row['recommendation']} |")
    lines += [
        "",
        f"**Soccer capped:** {'yes - keep capped' if summary['league_recommendations']['soccer_should_remain_capped'] else 'active - caps still recommended'}",
        "",
        summary["league_recommendations"]["soccer_note"],
    ]
    offload = summary.get("sportsgameodds_offload") or {}
    if offload:
        lines += [
            "",
            "## SportsGameOdds offload",
            "",
            f"- SportsGameOdds active: {offload.get('sportsgameodds_active')}",
            f"- Can reduce Odds API usage: {offload.get('can_reduce_odds_api_usage')}",
            f"- SGO monthly entities remaining: {offload.get('sgo_entities_remaining_month')}",
            "",
            str(offload.get("explanation", "")),
        ]
    lines += [
        "",
        "---",
        "Research-only. Quota protection only controls data collection; approved bets and parlays remain blocked.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Odds API quota report.")
    parser.parse_args()

    now = datetime.now(timezone.utc)
    runs = load_runs()
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}

    usage = analyze_runs(runs, now)
    risk = assess_risk(usage, config, now)
    leagues = recommend_league_priority(usage, config)
    offload = sportsgameodds_offload_note()

    summary = {
        "report": "odds_api_quota_report",
        "generated_at_utc": now.isoformat(),
        "usage": usage,
        "risk_assessment": risk,
        "league_recommendations": leagues,
        "sportsgameodds_offload": offload,
        "notes": [
            "Requests/run is estimated from x-requests-remaining header deltas between runs.",
            "The Odds API quota cycle is rolling from signup; calendar month is used as a conservative assumption.",
            "Player-prop event odds cost extra credits vs plain odds requests.",
        ],
        "research_only": True,
        "approved": False,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "odds_api_quota_report.json"
    md_path = REPORTS_DIR / "odds_api_quota_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Quota remaining: {usage['quota_remaining']}  (~{usage['avg_requests_per_run']} req/run)")
    print(f"Risk: {risk['risk']}  Recommended max runs/day: {risk.get('recommended_max_runs_per_day')}")
    print(f"Soccer capped: {leagues['soccer_should_remain_capped']}")
    print(f"Wrote: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    print("Research-only: quota protection only; approved bets/parlays remain blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
