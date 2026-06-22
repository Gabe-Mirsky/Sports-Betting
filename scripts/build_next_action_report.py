"""Build the live "what should I do next?" report (research-only).

Inspects the current project state — collection runs, settlement, closing
snapshots, CLV, data quality gates, scheduled tasks, API key — and writes a
prioritized list of practical next actions:

    data/reports/next_action_report.md
    data/reports/next_action_report.json

Research-only: actions are about data collection, settlement, and reporting.
Nothing here recommends bets; approved bets and parlays remain blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

PY = r".\.venv\Scripts\python.exe"

EXPECTED_TASKS = [
    "Player Prop Collection Every 4 Hours",
    "Player Prop Collection At Login",
    "NBA Pregame Prop Collection 1800ET",
    "NBA Pregame Prop Collection 1830ET",
    "NBA Pregame Prop Collection 1900ET",
    "NBA Pregame Prop Collection 1930ET",
    "NBA Pregame Prop Collection 2030ET",
    "NBA Pregame Prop Collection 2130ET",
    "NBA Pregame Prop Collection 2200ET",
]

# A game is assumed finished this long after tip-off.
GAME_DURATION_HOURS = 3.5

SOCCER_LEAGUES = {"EPL", "MLS", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1", "UEFA_CL"}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_last_run_history(path: Path) -> dict:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _installed_task_names() -> set[str] | None:
    """Names of all scheduled tasks, or None when unqueryable (non-Windows)."""
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    names: set[str] = set()
    for line in result.stdout.splitlines():
        if line.startswith('"'):
            task_path = line.split('","')[0].strip('"')
            names.add(task_path.rsplit("\\", 1)[-1])
    return names


def _action(priority: int, action: str, reason: str, command: str | None = None) -> dict:
    return {"priority": priority, "action": action, "reason": reason, "command": command}


def evaluate_next_actions(state: dict) -> list[dict]:
    """Pure decision logic: project state dict -> prioritized action list."""
    actions: list[dict] = []
    now = state["now_utc"]

    # 1. API key.
    if not state["api_key_present"]:
        actions.append(_action(
            1, "Set the ODDS_API_KEY environment variable",
            "No API key detected: every Odds API league is skipped, so no new snapshots "
            "are being collected.",
            '[Environment]::SetEnvironmentVariable("ODDS_API_KEY", "<your key>", "User")',
        ))

    # 2. Finished games with pending props -> settlement refresh with --download.
    finished_pending = [
        g for g in state["unsettled_games"]
        if g.get("game_start_utc") and g["game_start_utc"] + timedelta(hours=GAME_DURATION_HOURS) < now
    ]
    actuals_stale = state["actuals_max_date"] is not None and any(
        g.get("game_date") and str(g["game_date"]) > state["actuals_max_date"] for g in finished_pending
    )
    if finished_pending and (actuals_stale or state["actuals_max_date"] is None):
        games = ", ".join(str(g.get("label") or g.get("canonical_game_key")) for g in finished_pending[:5])
        actions.append(_action(
            1, "Run settlement refresh with --download",
            f"{state['pending_props']} props are pending and these games have finished: {games}. "
            f"The local nba_api cache only reaches {state['actuals_max_date'] or 'n/a'}, so a "
            "cache-only refresh cannot settle them.",
            f"{PY} scripts\\refresh_nba_results_and_settle_props.py --download",
        ))
    elif finished_pending:
        actions.append(_action(
            2, "Run settlement refresh (cache-only)",
            f"{state['pending_props']} pending props belong to finished games and the actuals "
            "cache looks current; a cache-only refresh should settle them.",
            f"{PY} scripts\\refresh_nba_results_and_settle_props.py",
        ))

    # 3. NBA closing snapshots.
    upcoming = state["upcoming_nba_games"]
    if state["nba_closing_snapshots"] == 0:
        if upcoming:
            nxt = upcoming[0]
            tip = nxt["game_start_utc"]
            collect_at = tip - timedelta(minutes=30)
            actions.append(_action(
                2, f"Run pregame collection near {collect_at.strftime('%Y-%m-%d %H:%M UTC')}",
                f"NBA closing-like snapshots are still 0 and {nxt['label']} tips at "
                f"{tip.strftime('%H:%M UTC')}. A run inside the 60-minute pre-tip window "
                "creates the first NBA closing snapshots. The scheduled NBA pregame tasks "
                "cover the usual evening slots; this is only manual backup.",
                "run_nba_pregame_prop_collection.bat",
            ))
        else:
            actions.append(_action(
                3, "Wait for the next NBA game day (closing snapshots)",
                "NBA closing-like snapshots are 0 and no upcoming NBA game with collected "
                "props is in the collection plan. The scheduled pregame tasks will capture "
                "closing snapshots automatically on the next game day.",
                None,
            ))

    # 4. Scheduled tasks.
    if state["missing_tasks"] is None:
        actions.append(_action(
            4, "Verify scheduled tasks manually",
            "Scheduled tasks could not be queried from this environment.",
            "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\verify_scheduled_tasks.ps1",
        ))
    elif state["missing_tasks"]:
        missing = ", ".join(state["missing_tasks"])
        actions.append(_action(
            2, "Recreate missing scheduled tasks",
            f"These collection tasks are missing: {missing}.",
            "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\setup_nba_pregame_tasks.ps1",
        ))

    # 5. CLV readiness.
    if state["nba_clv_markets"] == 0:
        actions.append(_action(
            3, "NBA CLV is not computable yet",
            "CLV needs an early snapshot AND a closing-like snapshot of the same market. "
            f"NBA has {state['nba_snapshots']} snapshots but {state['nba_closing_snapshots']} "
            "closing-like ones; the missing piece is closing coverage (runs within 60 minutes "
            "of tip-off). Non-NBA CLV already works "
            f"({state['total_clv_markets']} markets, mostly MLB).",
            None,
        ))

    # 6. Settled props exist -> rebuild outcome reports.
    if state["settled_props"] > 0 and state["settled_props"] > state["outcomes_settled_props"]:
        actions.append(_action(
            2, "Rebuild settlement outcome reports",
            f"{state['settled_props']} props are settled but the outcomes report only covers "
            f"{state['outcomes_settled_props']}.",
            f"{PY} scripts\\build_player_prop_settlement_outcomes.py",
        ))

    # 7. Data gates blocking modeling.
    if state["gate_status"] not in {"modeling_experiment_ready"}:
        blocking = state["gate_blockers"][:6]
        reason = (
            f"Data gates are at '{state['gate_status']}'. Blocking checks: "
            + ("; ".join(blocking) if blocking else "(none listed)")
        )
        actions.append(_action(
            3, "Modeling stays blocked by data quality gates", reason, None,
        ))
    else:
        actions.append(_action(
            2, "Baseline modeling experiments can start (research-only)",
            "Data gates report modeling_experiment_ready: enough settled props, CLV pairs, "
            "and closing coverage exist. Build the player-stat baseline next. This does NOT "
            "approve betting.",
            None,
        ))

    # 8. Soccer status.
    if state["soccer_inactive"]:
        actions.append(_action(
            5, "Soccer leagues: collect-only, waiting for active events",
            "All six soccer leagues returned 0 snapshots recently (off-season or no events "
            "in the horizon). They stay collect-only and quota-capped; no action needed.",
            None,
        ))

    actions.sort(key=lambda a: a["priority"])
    return actions


def gather_state() -> dict:
    gates = _read_json(REPORTS_DIR / "player_prop_data_quality_gates.json")
    refresh = _read_json(REPORTS_DIR / "player_prop_settlement_refresh_summary.json")
    clv = _read_json(REPORTS_DIR / "player_prop_clv_summary.json")
    plan = _read_json(REPORTS_DIR / "nba_prop_closing_collection_plan.json")
    outcomes = _read_json(REPORTS_DIR / "player_prop_settlement_outcomes_summary.json")
    last_run = _read_last_run_history(REPORTS_DIR / "prop_collection_run_history.jsonl")

    now = datetime.now(timezone.utc)
    metrics = gates.get("metrics", {}) if isinstance(gates.get("metrics"), dict) else {}
    settlement = refresh.get("settlement", {}) if isinstance(refresh.get("settlement"), dict) else {}

    # Unsettled games with parsed start times (collection plan knows tips).
    plan_games = {g.get("canonical_game_key"): g for g in plan.get("games", []) if isinstance(g, dict)}
    unsettled = []
    for g in settlement.get("unsettled_games", []) or []:
        key = g.get("canonical_game_key")
        start_raw = (plan_games.get(key) or {}).get("game_start_time")
        start = None
        if start_raw:
            try:
                start = datetime.fromisoformat(str(start_raw))
            except ValueError:
                start = None
        unsettled.append({
            "canonical_game_key": key,
            "label": (plan_games.get(key) or {}).get("game") or key,
            "game_date": g.get("game_date"),
            "pending_snapshots": g.get("pending_snapshots"),
            "game_start_utc": start,
        })

    upcoming = []
    for g in plan.get("games", []):
        start_raw = g.get("game_start_time")
        if not start_raw:
            continue
        try:
            start = datetime.fromisoformat(str(start_raw))
        except ValueError:
            continue
        if start > now:
            upcoming.append({"label": g.get("game"), "game_start_utc": start})
    upcoming.sort(key=lambda g: g["game_start_utc"])

    actuals_max = (
        ((refresh.get("actuals_import") or {}).get("games_date_range") or {}).get("max")
    )

    installed = _installed_task_names()
    missing_tasks = None if installed is None else [t for t in EXPECTED_TASKS if t not in installed]

    by_league = last_run.get("snapshots_by_league") or {}
    soccer_inactive = bool(by_league) and all(
        int(by_league.get(lg, 0) or 0) == 0 for lg in SOCCER_LEAGUES if lg in by_league
    )

    api_key_present = bool(os.environ.get("ODDS_API_KEY")) or bool(last_run.get("api_key_detected"))

    return {
        "now_utc": now,
        "api_key_present": api_key_present,
        "pending_props": int(settlement.get("pending_after_refresh", 0) or 0),
        "settled_props": int(metrics.get("settled_props", 0) or 0),
        "outcomes_settled_props": int(outcomes.get("settled_props", 0) or 0),
        "unsettled_games": unsettled,
        "upcoming_nba_games": upcoming,
        "actuals_max_date": actuals_max,
        "nba_snapshots": int(metrics.get("nba_snapshots", 0) or 0),
        "nba_closing_snapshots": int(metrics.get("closing_like_snapshots", 0) or 0),
        "nba_clv_markets": int(clv.get("nba_markets_with_clv", 0) or 0),
        "total_clv_markets": int(clv.get("markets_with_clv", 0) or 0),
        "gate_status": str(gates.get("status", "unknown")),
        "gate_blockers": [str(b) for b in (gates.get("blockers") or [])],
        "missing_tasks": missing_tasks,
        "soccer_inactive": soccer_inactive,
        "last_run_id": last_run.get("run_id"),
        "last_run_outcome": last_run.get("outcome"),
        "quota_remaining": last_run.get("quota_remaining_requests"),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Next Action Report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "_Research-only: these are data-pipeline actions. No bets, no recommendations;",
        "approved bets and approved parlays remain blocked._",
        "",
        f"## Do this next: {summary['next_action']['action']}",
        "",
        summary["next_action"]["reason"],
        "",
    ]
    if summary["next_action"].get("command"):
        lines += ["```", summary["next_action"]["command"], "```", ""]
    lines += ["## All actions (by priority)", ""]
    for a in summary["actions"]:
        lines.append(f"### P{a['priority']}: {a['action']}")
        lines.append("")
        lines.append(a["reason"])
        if a.get("command"):
            lines += ["", "```", a["command"], "```"]
        lines.append("")
    s = summary["state"]
    lines += [
        "## State snapshot",
        "",
        f"- NBA snapshots: {s['nba_snapshots']}",
        f"- NBA closing-like snapshots: {s['nba_closing_snapshots']}",
        f"- Pending props: {s['pending_props']}",
        f"- Settled props: {s['settled_props']}",
        f"- NBA CLV markets: {s['nba_clv_markets']} (all leagues: {s['total_clv_markets']})",
        f"- Data gate status: {s['gate_status']}",
        f"- API key present: {s['api_key_present']}",
        f"- Quota remaining (last run): {s['quota_remaining']}",
        f"- Missing scheduled tasks: {s['missing_tasks'] if s['missing_tasks'] else 'none' if s['missing_tasks'] is not None else 'unknown'}",
        "",
        "---",
        "Research-only. Approved bets and approved parlays remain blocked.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next-action report.")
    parser.parse_args()

    state = gather_state()
    actions = evaluate_next_actions(state)

    json_state = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in state.items()
    }
    json_state["unsettled_games"] = [
        {**g, "game_start_utc": g["game_start_utc"].isoformat() if g.get("game_start_utc") else None}
        for g in state["unsettled_games"]
    ]
    json_state["upcoming_nba_games"] = [
        {**g, "game_start_utc": g["game_start_utc"].isoformat()}
        for g in state["upcoming_nba_games"]
    ]

    summary = {
        "report": "next_action_report",
        "generated_at_utc": state["now_utc"].isoformat(),
        "next_action": actions[0] if actions else _action(9, "Nothing to do", "All clear.", None),
        "actions": actions,
        "state": json_state,
        "research_only": True,
        "approved": False,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "next_action_report.json"
    md_path = REPORTS_DIR / "next_action_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Next action: {summary['next_action']['action']}")
    if summary["next_action"].get("command"):
        print(f"  Command: {summary['next_action']['command']}")
    print(f"  ({len(actions)} prioritized actions total)")
    print(f"Wrote: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    print("Research-only: no recommendations; approved bets/parlays remain blocked.")


if __name__ == "__main__":
    main()
