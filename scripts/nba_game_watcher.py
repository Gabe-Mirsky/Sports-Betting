"""Awake-time NBA game watcher (research-only automation).

Runs cheaply every ~10 minutes (via Task Scheduler). On each tick it looks at the
known NBA games and decides, per game, whether to:

  * PREGAME  -> a game tips off within the next ``--pregame-window-min`` minutes:
               run ``run_nba_pregame_prop_collection.bat`` once to capture a
               closing-like snapshot.
  * SETTLE   -> a game ended recently (started between ``--settle-after-min`` and
               ``--settle-lookback-min`` minutes ago): run
               ``refresh_nba_results_and_settle_props.py --download`` once to pull
               fresh actuals and grade pending props.
  * SKIP     -> nothing is due.

Duplicate protection: every fired action is recorded in an append-only run-log
(``data/logs/nba_watcher/run_log.jsonl``) keyed by (game_id, action). A game's
action is considered done after one SUCCESS; failures are retried up to
``--max-attempts`` times so a transient Odds API 429 or late box score does not
permanently skip the game.

This script is automation only. It enables NO betting, NO parlays, NO model
predictions, and NO approved recommendations, and it does not touch any model
gate. The two actions it runs are exactly the existing research-only collection
and settlement steps.
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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.nba_collection_planner import build_collection_plan  # noqa: E402

WATCH_DIR = PROJECT_ROOT / "data" / "logs" / "nba_watcher"
RUN_LOG = WATCH_DIR / "run_log.jsonl"          # machine-readable dedup log (req #3)
HUMAN_LOG = WATCH_DIR / "watcher.log"          # human-readable activity log (req #8)
LOCK_FILE = WATCH_DIR / "watcher.lock"
STATUS_JSON = PROJECT_ROOT / "data" / "reports" / "nba_watcher_status.json"
SNAPSHOTS = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"
PREGAME_BAT = PROJECT_ROOT / "run_nba_pregame_prop_collection.bat"
REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "refresh_nba_results_and_settle_props.py"

ACTION_PREGAME = "pregame"
ACTION_SETTLE = "settle"
STALE_LOCK_HOURS = 2.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def human_line(msg: str) -> None:
    """Append one timestamped line to the human-readable log and stdout."""
    stamp = now_utc().strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"{stamp}  {msg}"
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    with HUMAN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


def load_run_log() -> list[dict[str, Any]]:
    if not RUN_LOG.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in RUN_LOG.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows


def append_run_log(entry: dict[str, Any]) -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def action_state(history: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    """Count successes/attempts per (game_id, action) from the run-log."""
    state: dict[tuple[str, str], dict[str, int]] = {}
    for row in history:
        key = (str(row.get("game_id")), str(row.get("action")))
        bucket = state.setdefault(key, {"success": 0, "attempts": 0})
        status = str(row.get("status"))
        if status in ("success", "failed"):
            bucket["attempts"] += 1
        if status == "success":
            bucket["success"] += 1
    return state


def needs_action(state: dict, game_id: str, action: str, max_attempts: int) -> bool:
    bucket = state.get((game_id, action))
    if not bucket:
        return True
    if bucket["success"] > 0:
        return False  # already done once -> never repeat
    return bucket["attempts"] < max_attempts  # retry failures up to the cap


# --------------------------------------------------------------------------- #
# Locking (defensive; Task Scheduler also uses MultipleInstances IgnoreNew)
# --------------------------------------------------------------------------- #
def acquire_lock() -> bool:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(data["started_utc"])
            age_h = (now_utc() - started).total_seconds() / 3600.0
            if age_h < STALE_LOCK_HOURS:
                human_line(f"SKIP  another watcher instance is running (lock age {age_h:.2f}h); exiting.")
                return False
            human_line(f"NOTE  overriding stale lock (age {age_h:.2f}h).")
        except Exception:
            human_line("NOTE  unreadable lock file; overriding.")
    LOCK_FILE.write_text(
        json.dumps({"pid": os.getpid(), "started_utc": now_utc().isoformat()}), encoding="utf-8"
    )
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Subprocess actions
# --------------------------------------------------------------------------- #
def run_pregame(dry_run: bool) -> tuple[int, str]:
    cmd = ["cmd", "/c", str(PREGAME_BAT)]
    if dry_run:
        return (0, "dry_run: " + " ".join(cmd))
    if not PREGAME_BAT.exists():
        return (1, f"missing {PREGAME_BAT.name}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return (proc.returncode, " ".join(cmd))


def run_settle(dry_run: bool) -> tuple[int, str]:
    python = sys.executable or str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    cmd = [python, str(REFRESH_SCRIPT), "--download"]
    if dry_run:
        return (0, "dry_run: " + " ".join(cmd))
    if not REFRESH_SCRIPT.exists():
        return (1, f"missing {REFRESH_SCRIPT.name}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return (proc.returncode, " ".join(cmd))


# --------------------------------------------------------------------------- #
# Core tick
# --------------------------------------------------------------------------- #
def decide_and_run(args: argparse.Namespace) -> dict[str, Any]:
    now = now_utc()
    run_id = now.strftime("%Y%m%dT%H%M%SZ")

    snapshots = (
        pd.read_csv(SNAPSHOTS, low_memory=False) if SNAPSHOTS.exists() else pd.DataFrame()
    )
    plan = build_collection_plan(snapshots, now=now)
    games = plan.get("games", [])

    history = load_run_log()
    state = action_state(history)

    pregame_due: list[dict[str, Any]] = []
    settle_due: list[dict[str, Any]] = []
    considered: list[dict[str, Any]] = []

    for g in games:
        gid = str(g.get("canonical_game_key"))
        label = str(g.get("game") or gid)
        mins = g.get("minutes_until_game")
        if mins is None:
            continue
        mins = float(mins)
        # PREGAME: tips within the next pregame-window minutes (small post-tip grace).
        is_pregame = (-args.pregame_grace_min) <= mins <= args.pregame_window_min
        # SETTLE: started long enough ago to be over, but still within the lookback.
        is_settle = (-args.settle_lookback_min) <= mins <= (-args.settle_after_min)
        considered.append({
            "game_id": gid, "game_label": label, "minutes_until_game": round(mins, 1),
            "timing": g.get("timing_classification"),
            "pregame_window": is_pregame, "settle_window": is_settle,
        })
        if is_pregame and needs_action(state, gid, ACTION_PREGAME, args.max_attempts):
            pregame_due.append({"game_id": gid, "game_label": label, "minutes_until_game": mins})
        if is_settle and needs_action(state, gid, ACTION_SETTLE, args.max_attempts):
            settle_due.append({"game_id": gid, "game_label": label, "minutes_until_game": mins})

    actions_fired: list[dict[str, Any]] = []

    # One bat invocation collects every in-window NBA game at once, so fire once
    # and record a run-log entry per game it covers.
    if pregame_due:
        labels = ", ".join(f"{g['game_label']} (T-{g['minutes_until_game']:.0f}m)" for g in pregame_due)
        human_line(f"PREGAME  {len(pregame_due)} game(s) within {args.pregame_window_min}m: {labels}")
        rc, cmd = run_pregame(args.dry_run)
        status = "dry_run" if args.dry_run else ("success" if rc == 0 else "failed")
        human_line(f"PREGAME  command -> rc={rc} status={status} ({'dry-run' if args.dry_run else cmd})")
        for g in pregame_due:
            entry = {
                "watcher_run_id": run_id, "timestamp_utc": now_utc().isoformat(),
                "game_id": g["game_id"], "game_label": g["game_label"], "action": ACTION_PREGAME,
                "status": status, "return_code": rc, "minutes_until_game": round(g["minutes_until_game"], 1),
                "command": cmd, "research_only": True,
            }
            if not args.dry_run:
                append_run_log(entry)
            actions_fired.append(entry)

    if settle_due:
        labels = ", ".join(f"{g['game_label']} (T+{abs(g['minutes_until_game']):.0f}m)" for g in settle_due)
        human_line(f"SETTLE   {len(settle_due)} recently-ended game(s): {labels}")
        rc, cmd = run_settle(args.dry_run)
        status = "dry_run" if args.dry_run else ("success" if rc == 0 else "failed")
        human_line(f"SETTLE   command -> rc={rc} status={status} ({'dry-run' if args.dry_run else cmd})")
        for g in settle_due:
            entry = {
                "watcher_run_id": run_id, "timestamp_utc": now_utc().isoformat(),
                "game_id": g["game_id"], "game_label": g["game_label"], "action": ACTION_SETTLE,
                "status": status, "return_code": rc, "minutes_until_game": round(g["minutes_until_game"], 1),
                "command": cmd, "research_only": True,
            }
            if not args.dry_run:
                append_run_log(entry)
            actions_fired.append(entry)

    if not pregame_due and not settle_due:
        nxt = plan.get("minutes_until_next_nba_game")
        reason = (
            f"no NBA game within {args.pregame_window_min}m and none recently ended"
            + (f"; next tip in {float(nxt):.0f}m" if nxt is not None else "; no upcoming NBA game known")
        )
        human_line(f"SKIP     {reason} ({plan.get('games_total', 0)} known game(s))")

    summary = {
        "report": "nba_watcher_status",
        "generated_at_utc": now.isoformat(),
        "watcher_run_id": run_id,
        "dry_run": bool(args.dry_run),
        "research_only": True,
        "approved": False,
        "params": {
            "pregame_window_min": args.pregame_window_min,
            "pregame_grace_min": args.pregame_grace_min,
            "settle_after_min": args.settle_after_min,
            "settle_lookback_min": args.settle_lookback_min,
            "max_attempts": args.max_attempts,
        },
        "games_known": plan.get("games_total", 0),
        "games_upcoming": plan.get("games_upcoming", 0),
        "minutes_until_next_nba_game": plan.get("minutes_until_next_nba_game"),
        "considered": considered,
        "actions_fired": actions_fired,
        "pregame_fired": len(pregame_due),
        "settle_fired": len(settle_due),
        "run_log_path": str(RUN_LOG.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "human_log_path": str(HUMAN_LOG.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Awake-time NBA game watcher (research-only).")
    p.add_argument("--pregame-window-min", type=float, default=60.0,
                   help="Fire pregame collection when a game tips within this many minutes (req: 60).")
    p.add_argument("--pregame-grace-min", type=float, default=10.0,
                   help="Allow firing up to this many minutes AFTER tip (catch a just-started game).")
    p.add_argument("--settle-after-min", type=float, default=180.0,
                   help="Treat a game as ended once it started at least this many minutes ago.")
    p.add_argument("--settle-lookback-min", type=float, default=420.0,
                   help="Stop considering a game for settlement after this many minutes past tip.")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Max attempts per (game, action) before giving up (retries failures only).")
    p.add_argument("--dry-run", action="store_true",
                   help="Decide and log, but do not run anything or write the dedup run-log.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if not acquire_lock():
        return
    try:
        decide_and_run(args)
    except Exception as exc:  # never let the scheduled task crash hard
        human_line(f"ERROR    watcher failed: {exc!r}")
        raise
    finally:
        release_lock()
    print("Research-only: watcher enables no betting, parlays, predictions, or gate changes.")


if __name__ == "__main__":
    main()
