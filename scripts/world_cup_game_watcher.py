"""Awake-time World Cup (FIFA) odds watcher (research-only automation).

Runs cheaply every ~10 minutes (via Task Scheduler). Each tick lists World Cup
events with the FREE /events endpoint (0 credits), labels them with the generic
event planner, and decides per match:

  * CLOSING_SNAPSHOT -> match kicks off within 60 min: pull a closing-like
                        game-market snapshot (guarded, capped).
  * EARLY_SNAPSHOT   -> match is 24-48h out: optionally pull an opening line
                        (only when --early is passed).
  * POSTGAME_RESULTS -> match ended recently: run refresh_world_cup_results.py.
  * SKIP             -> nothing due (no credit-spending call).

Duplicate protection: data/logs/world_cup_watcher/run_log.jsonl keyed by
(event_id, action); done after one success, failures retried up to --max-attempts.
A strict quota guard inside the collector protects the NBA Odds API budget.

This watcher runs NO model predictions and NO betting/parlay logic. It is
completely separate from the NBA watcher, which is left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.event_planner import (  # noqa: E402
    ACTION_CLOSING, ACTION_EARLY, ACTION_POSTGAME, EventWindows, actions_due, plan_events,
)
from data.watcher_run_log import action_state, append_run_log, load_run_log, needs_action  # noqa: E402
from data.world_cup_collection import (  # noqa: E402
    collect_world_cup_kalshi_markets, fetch_world_cup_events, load_world_cup_config,
    run_world_cup_collection,
)
from data.prop_collection import make_default_fetch_json  # noqa: E402
from data.source_router import SourceRouter, SourceState, load_router_config  # noqa: E402

ROUTER_CONFIG = PROJECT_ROOT / "config" / "source_priority.yaml"
SOURCE_STATE = PROJECT_ROOT / "data" / "reports" / "source_state.json"
ROUTED_LOG = PROJECT_ROOT / "data" / "logs" / "world_cup_watcher" / "routed_fetch.jsonl"
SOURCE_STATUS_JSON = PROJECT_ROOT / "data" / "reports" / "world_cup_source_status.json"
SOURCE_STATUS_MD = PROJECT_ROOT / "data" / "reports" / "world_cup_source_status.md"


def _build_wc_router(now, api_key: str, odds_api_quota):
    """Build the router from the shared source-state file, overriding odds_api
    with the FRESH credit count read from the free /events listing."""
    cfg = load_router_config(ROUTER_CONFIG)
    state_doc = {}
    if SOURCE_STATE.exists():
        try:
            state_doc = json.loads(SOURCE_STATE.read_text(encoding="utf-8"))
        except Exception:
            state_doc = {}
    states: dict[str, SourceState] = {}
    for name in ("odds_api", "sportsgameodds", "apisports", "kalshi"):
        s = (state_doc.get("sources") or {}).get(name, {})
        states[name] = SourceState(
            name=name, key_present=bool(s.get("key_present")),
            quota_remaining=s.get("quota_remaining"),
            last_success_utc=s.get("last_success_utc"),
            last_failure_utc=s.get("last_failure_utc"),
            blocked_reason=s.get("blocked_reason"),
        )
    # Freshest odds_api signal: this run just listed events and read the header.
    states["odds_api"].key_present = bool(api_key)
    if odds_api_quota is not None:
        states["odds_api"].quota_remaining = odds_api_quota
    return SourceRouter(cfg, states, now=now), states

WATCH_DIR = PROJECT_ROOT / "data" / "logs" / "world_cup_watcher"
RUN_LOG = WATCH_DIR / "run_log.jsonl"
HUMAN_LOG = WATCH_DIR / "watcher.log"
LOCK_FILE = WATCH_DIR / "watcher.lock"
STATUS_JSON = PROJECT_ROOT / "data" / "reports" / "world_cup_watcher_status.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "world_cup_collection.yaml"
RESULTS_SCRIPT = PROJECT_ROOT / "scripts" / "refresh_world_cup_results.py"
STALE_LOCK_HOURS = 2.0

# Map planner action -> run-log action label (one success per event per label).
ACTION_LABEL = {ACTION_CLOSING: "closing", ACTION_EARLY: "early", ACTION_POSTGAME: "results"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def human_line(msg: str) -> None:
    stamp = now_utc().strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"{stamp}  {msg}"
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    with HUMAN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


def acquire_lock() -> bool:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            age_h = (now_utc() - datetime.fromisoformat(data["started_utc"])).total_seconds() / 3600.0
            if age_h < STALE_LOCK_HOURS:
                human_line(f"SKIP  another World Cup watcher is running (lock age {age_h:.2f}h); exiting.")
                return False
            human_line(f"NOTE  overriding stale lock (age {age_h:.2f}h).")
        except Exception:
            human_line("NOTE  unreadable lock file; overriding.")
    LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started_utc": now_utc().isoformat()}), encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _log_action(run_id: str, ev: dict[str, Any], label: str, status: str, rc: int, cmd: str, dry: bool) -> dict:
    entry = {
        "watcher_run_id": run_id, "timestamp_utc": now_utc().isoformat(),
        "event_id": str(ev.get("event_id")), "event_label": f"{ev.get('home_team')} v {ev.get('away_team')}",
        "league": "WORLD_CUP", "action": label, "status": status, "return_code": rc,
        "minutes_until_event": ev.get("minutes_until_event"), "command": cmd, "research_only": True,
    }
    if not dry:
        append_run_log(RUN_LOG, entry)
    return entry


def _append_routed(entry: dict) -> None:
    ROUTED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ROUTED_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _write_wc_source_status(summary: dict, states: dict, router) -> None:
    """Write the World Cup source-status report (json + md)."""
    routing = summary.get("routing", {})
    odds_below = routing.get("game_odds", {}).get("reason") == "SKIP_PAID_ODDS"
    doc = {
        "report": "world_cup_source_status",
        "generated_at_utc": now_utc().isoformat(),
        "research_only": True, "approved": False,
        "events_found": summary.get("events_found"),
        "odds_api_credits": summary.get("credits_remaining"),
        "odds_api_below_floor": odds_below,
        "routing": routing,
        "sources": {n: s.to_dict() for n, s in states.items()},
        "can_collect_schedule_without_paid_odds": routing.get("events_schedule", {}).get("selected") is not None,
    }
    SOURCE_STATUS_JSON.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    lines = [
        "# World Cup Source Status",
        f"_Generated {doc['generated_at_utc']} — research-only; no bets/parlays/predictions._",
        "",
        f"- Events found (free schedule): **{doc['events_found']}**",
        f"- Odds API credits: **{doc['odds_api_credits']}** (below floor: **{odds_below}**)",
        "",
        "## Routed source per data type",
        "| Data type | Selected | Reason |",
        "| --- | --- | --- |",
    ]
    for dt, dec in routing.items():
        lines.append(f"| {dt} | {dec.get('selected') or '—'} | {dec.get('reason')} |")
    if odds_below:
        lines += ["", "> ⚠️ **Odds API is below its quota floor**, so paid World Cup odds are not "
                  "collected right now (`SKIP_PAID_ODDS`). The free schedule still updates, and "
                  "collection resumes automatically when credits recover. Prediction-market prices "
                  "(Kalshi) are tracked as a separate category, never mixed into sportsbook odds."]
    lines += ["", "_Research-only. Source routing decides where data comes from; it enables no "
              "betting, parlays, predictions, or recommendations._"]
    SOURCE_STATUS_MD.write_text("\n".join(lines), encoding="utf-8")


def run_results_refresh(dry_run: bool) -> tuple[int, str]:
    python = sys.executable or str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    # --real makes the actual /scores call; the watcher only reaches here when the
    # router has already confirmed Odds API is above its quota floor.
    cmd = [python, str(RESULTS_SCRIPT)] + ([] if dry_run else ["--real"])
    if dry_run:
        return (0, "dry_run: " + " ".join(cmd))
    if not RESULTS_SCRIPT.exists():
        return (2, f"missing {RESULTS_SCRIPT.name} (results refresh not installed yet)")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return (proc.returncode, " ".join(cmd))


def decide_and_run(args: argparse.Namespace) -> dict[str, Any]:
    now = now_utc()
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    config = load_world_cup_config(args.config)
    windows = EventWindows.from_config(config.get("defaults", {}))
    src_cfg = config.get("source", {})
    api_key = src_cfg.get("api_key") or os.environ.get(src_cfg.get("api_key_env", "ODDS_API_KEY"), "")

    summary: dict[str, Any] = {
        "report": "world_cup_watcher_status", "generated_at_utc": now.isoformat(),
        "watcher_run_id": run_id, "dry_run": bool(args.dry_run), "early_enabled": bool(args.early),
        "research_only": True, "approved": False, "key_detected": bool(api_key),
        "events_found": 0, "considered": [], "actions_fired": [],
        "closing_fired": 0, "early_fired": 0, "results_fired": 0,
        "run_log_path": str(RUN_LOG.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "human_log_path": str(HUMAN_LOG.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }

    if not api_key:
        human_line("SKIP     no ODDS_API_KEY in environment; cannot list World Cup events.")
        summary["status"] = "no_key"
        STATUS_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    # --- List events (FREE, 0 credits) -------------------------------------
    fetch = make_default_fetch_json()
    try:
        events = fetch_world_cup_events(api_key, config, fetch)
    except Exception as exc:  # noqa: BLE001
        human_line(f"ERROR    could not list World Cup events: {exc!r}")
        summary["status"] = "events_error"
        summary["errors"] = [repr(exc)]
        STATUS_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary
    summary["events_found"] = len(events)
    summary["credits_remaining"] = getattr(fetch, "quota_remaining", None)

    # --- Route each World Cup data type through the multi-source router -----
    router, rstates = _build_wc_router(now, api_key, summary["credits_remaining"])
    schedule_dec = router.route("WORLD_CUP", "events_schedule")
    odds_dec = router.route("WORLD_CUP", "game_odds")
    results_dec = router.route("WORLD_CUP", "results")
    pred_dec = router.route("WORLD_CUP", "prediction_market_prices")
    summary["routing"] = {
        "events_schedule": schedule_dec.to_dict(),
        "game_odds": odds_dec.to_dict(),
        "results": results_dec.to_dict(),
        "prediction_market_prices": pred_dec.to_dict(),
    }
    human_line(f"ROUTE    schedule={schedule_dec.selected or schedule_dec.reason} | "
               f"odds={odds_dec.selected or odds_dec.reason} | "
               f"results={results_dec.selected or results_dec.reason} | "
               f"prediction_market={pred_dec.selected or pred_dec.reason}")
    _write_wc_source_status(summary, rstates, router)

    # Kalshi prediction-market check (opt-in, separate category, bounded).
    if getattr(args, "kalshi", False) and pred_dec.selected == "kalshi":
        kal = collect_world_cup_kalshi_markets(PROJECT_ROOT, now=now)
        human_line(f"KALSHI   prediction-market check: scanned={kal['markets_scanned']} "
                   f"world_cup_markets={kal['world_cup_markets_found']} (separate category, not sportsbook odds)")
        summary["kalshi"] = {k: kal.get(k) for k in ("markets_scanned", "world_cup_markets_found", "error")}

    planned = plan_events(events, now=now, windows=windows, allow_early=bool(args.early))
    summary["considered"] = [
        {k: p.get(k) for k in ("event_id", "home_team", "away_team", "minutes_until_event",
                               "event_status", "recommended_action")}
        for p in planned
    ]
    due = actions_due(planned)
    history = load_run_log(RUN_LOG)
    state = action_state(history)

    # --- CLOSING / EARLY snapshots (one guarded collection covers all) ------
    snapshot_due: list[dict] = []
    for action in (ACTION_CLOSING, ACTION_EARLY):
        label = ACTION_LABEL[action]
        for p in due.get(action, []):
            if needs_action(state, str(p.get("event_id")), label, args.max_attempts):
                snapshot_due.append({**p, "_label": label})

    if snapshot_due and odds_dec.selected != "odds_api":
        # Router says no safe PAID odds source (e.g. Odds API below quota floor).
        # Free schedule already ran; we just skip the paid pull and let events stay
        # pending so they retry automatically when quota recovers.
        labels = ", ".join(f"{p['home_team']} v {p['away_team']}" for p in snapshot_due)
        human_line(f"SKIP_PAID_ODDS  World Cup odds source unavailable ({odds_dec.reason}); "
                   f"{len(snapshot_due)} match(es) waiting; free schedule still ran: {labels}")
        summary["closing_skipped_reason"] = odds_dec.reason
        _append_routed(router.record_fetch(
            odds_dec, rows=0, success=False,
            detail=f"{odds_dec.reason}: {len(snapshot_due)} WC match(es) not collected"))
        snapshot_due = []  # nothing fired; no dedup entries so they retry later

    if snapshot_due:
        labels = ", ".join(f"{p['home_team']} v {p['away_team']} [{p['_label']}]" for p in snapshot_due)
        human_line(f"COLLECT  {len(snapshot_due)} World Cup snapshot action(s) via {odds_dec.selected}: {labels}")
        coll = run_world_cup_collection(
            config, PROJECT_ROOT, dry_run=args.dry_run, max_events=args.max_events,
            env=os.environ, now=now,
        )
        _append_routed(router.record_fetch(
            odds_dec, rows=coll.get("rows_normalized", 0),
            quota_before=coll.get("credits_remaining_before"),
            quota_after=coll.get("credits_remaining_after"),
            success=coll.get("status") in ("collected", "dry_run_ok"),
            market_type="game_odds"))
        coll_status = coll.get("status")
        ok = coll_status in ("collected", "dry_run_ok")
        status = "dry_run" if args.dry_run else ("success" if ok else "failed")
        human_line(f"COLLECT  collector status={coll_status} -> {status} "
                   f"(events={coll.get('events_found')}, rows={coll.get('rows_normalized')}, "
                   f"credits_after={coll.get('credits_remaining_after')})")
        summary["collection"] = {k: coll.get(k) for k in (
            "status", "events_found", "events_processed", "rows_normalized", "markets_found",
            "bookmakers_found", "player_props_found", "credits_remaining_before",
            "credits_remaining_after", "blockers", "errors")}
        for p in snapshot_due:
            entry = _log_action(run_id, p, p["_label"], status, coll.get("rc", 0) or 0,
                                f"world_cup_collection:{coll_status}", args.dry_run)
            summary["actions_fired"].append(entry)
            if p["_label"] == "closing":
                summary["closing_fired"] += 1
            else:
                summary["early_fired"] += 1

    # --- POSTGAME results ---------------------------------------------------
    results_due = [p for p in due.get(ACTION_POSTGAME, [])
                   if needs_action(state, str(p.get("event_id")), "results", args.max_attempts)]
    if results_due and results_dec.selected != "odds_api" and not args.dry_run:
        # Preferred results source (apisports) is plan-blocked and Odds API /scores
        # is below the quota floor -> skip paid results but keep waiting.
        human_line(f"RESULTS  no safe results source ({results_dec.reason}); "
                   f"{len(results_due)} ended match(es) wait for quota/plan recovery.")
        summary["results_skipped_reason"] = results_dec.reason
        _append_routed(router.record_fetch(results_dec, rows=0, success=False,
                                           market_type="results", detail=results_dec.reason))
        results_due = []

    if results_due:
        labels = ", ".join(f"{p['home_team']} v {p['away_team']}" for p in results_due)
        src = results_dec.selected or "odds_api"
        human_line(f"RESULTS  {len(results_due)} ended World Cup match(es) via {src}: {labels}")
        rc, cmd = run_results_refresh(args.dry_run)
        if rc == 2 and not args.dry_run:
            human_line("RESULTS  refresh_world_cup_results.py not installed yet; will retry next tick.")
        else:
            status = "dry_run" if args.dry_run else ("success" if rc == 0 else "failed")
            human_line(f"RESULTS  command -> rc={rc} status={status}")
            for p in results_due:
                entry = _log_action(run_id, p, "results", status, rc, cmd, args.dry_run)
                summary["actions_fired"].append(entry)
                summary["results_fired"] += 1

    if not snapshot_due and not results_due:
        soonest = min((p["minutes_until_event"] for p in planned
                       if isinstance(p.get("minutes_until_event"), (int, float)) and p["minutes_until_event"] > 0),
                      default=None)
        reason = (f"next kickoff in {soonest:.0f}m" if soonest is not None
                  else "no upcoming World Cup match in window")
        early_note = "" if args.early else " (early snapshots disabled; pass --early to enable)"
        human_line(f"SKIP     nothing due for {len(events)} World Cup event(s); {reason}{early_note}")

    summary["status"] = summary.get("status", "ok")
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Awake-time World Cup odds watcher (research-only).")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--early", action="store_true",
                   help="Also collect early (24-48h) opening-line snapshots (extra credits).")
    p.add_argument("--kalshi", action="store_true",
                   help="Also run a bounded Kalshi prediction-market check (separate category).")
    p.add_argument("--max-events", type=int, default=None, help="Override per-run event cap.")
    p.add_argument("--max-attempts", type=int, default=3, help="Retry cap per (event, action).")
    p.add_argument("--dry-run", action="store_true",
                   help="Decide + list events (free) only; no odds pulls, no run-log writes.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if not acquire_lock():
        return
    try:
        decide_and_run(args)
    except Exception as exc:  # never crash the scheduled task hard
        human_line(f"ERROR    World Cup watcher failed: {exc!r}")
        raise
    finally:
        release_lock()
    print("Research-only: World Cup watcher enables no betting, parlays, predictions, or gate changes.")


if __name__ == "__main__":
    main()
