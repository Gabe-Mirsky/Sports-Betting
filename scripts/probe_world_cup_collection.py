"""Safe, capped World Cup collection probe (research-only).

Two modes:
  --dry-run (default): list events via the FREE /events endpoint (0 credits) and
                       print the plan. Never spends credits.
  --real            : ONE deliberately-minimal odds pull (default: h2h only,
                       1 region = 1 credit) capped to --max-events events. Saves
                       raw + normalized data and writes a probe report.

This probe is a SEPARATE one-time tool. It uses its own relaxed quota floor so a
single 1-credit probe can run when the *production* World Cup floor (100, in
config/world_cup_collection.yaml) would skip. The production watcher/collector
floor is NOT changed. Research-only: no models, recommendations, bets, parlays.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.world_cup_collection import load_world_cup_config, run_world_cup_collection  # noqa: E402

REPORT_JSON = PROJECT_ROOT / "data" / "reports" / "world_cup_collection_probe.json"
REPORT_MD = PROJECT_ROOT / "data" / "reports" / "world_cup_collection_probe.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe capped World Cup collection probe.")
    p.add_argument("--real", action="store_true",
                   help="Perform ONE minimal real odds pull (default is dry-run, free events only).")
    p.add_argument("--max-events", type=int, default=1, help="Cap on events normalized (default 1).")
    p.add_argument("--max-credits", type=int, default=1, help="Hard credit cap for this probe (default 1).")
    p.add_argument("--markets", default="h2h", help="Comma list of Odds API market keys (default h2h).")
    p.add_argument("--config", default=str(PROJECT_ROOT / "config" / "world_cup_collection.yaml"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def render_md(summary: dict) -> str:
    lines = [
        "# World Cup Collection Probe",
        f"_Generated {summary.get('generated_at_utc')} — research-only; no bets/parlays/predictions._",
        "",
        f"- Mode: {'REAL (capped)' if not summary.get('dry_run') else 'dry-run (free events only)'}",
        f"- Status: **{summary.get('status')}**",
        f"- Events found: **{summary.get('events_found')}**",
        f"- Events processed (normalized): {summary.get('events_processed')}",
        f"- Game markets found: {summary.get('game_markets_found')} → {summary.get('markets_found')}",
        f"- Bookmakers found: {summary.get('bookmakers_found')}",
        f"- Player props found: {summary.get('player_props_found')}",
        f"- Rows normalized: {summary.get('rows_normalized')}",
        f"- Estimated credit cost: {summary.get('estimated_credit_cost')}",
        f"- Credits remaining before/after: {summary.get('credits_remaining_before')} / {summary.get('credits_remaining_after')}",
        f"- Raw file: {summary.get('raw_file')}",
        f"- Processed path: {summary.get('processed_path')}",
        f"- Blockers: {summary.get('blockers')}",
        f"- Errors: {summary.get('errors')}",
        "",
    ]
    if summary.get("status") == "skipped_quota":
        lines += ["> The quota guard blocked the real odds pull (credits below the probe floor). "
                  "This is the safe path: the source is proven via the free /events listing above; "
                  "rerun the probe when Odds API credits recover.", ""]
    lines.append("_The production World Cup watcher keeps its strict floor (100 credits); "
                 "this probe used a relaxed one-time floor only for a single minimal pull._")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_world_cup_config(args.config)

    # One-time relaxed floor + hard caps for THIS probe only (config file untouched).
    config.setdefault("quota", {})
    config["quota"]["min_remaining_requests"] = float(args.max_credits)  # only needs its own credits
    config["quota"]["max_credits_per_run"] = int(args.max_credits)
    config["markets"] = {k: config.get("markets", {}).get(k, k) for k in args.markets.split(",") if k}
    config.setdefault("defaults", {})["max_events_per_run"] = int(args.max_events)

    summary = run_world_cup_collection(
        config, PROJECT_ROOT, dry_run=not args.real, max_events=args.max_events,
    )

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render_md(summary), encoding="utf-8")

    print(f"Probe status: {summary.get('status')} | events_found={summary.get('events_found')} | "
          f"rows={summary.get('rows_normalized')} | markets={summary.get('markets_found')} | "
          f"books={summary.get('bookmakers_found')}")
    print(f"Wrote: {REPORT_JSON.relative_to(PROJECT_ROOT)} and {REPORT_MD.relative_to(PROJECT_ROOT)}")
    print("Research-only: probe enables no betting, parlays, predictions, or gate changes.")


if __name__ == "__main__":
    main()
