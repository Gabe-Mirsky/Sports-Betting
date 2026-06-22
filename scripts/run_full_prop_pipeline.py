"""Master daily prop pipeline: collect -> enrich -> settle -> report -> dashboard.

Runs every stage of the research-only prop data pipeline in order, logging each
step, continuing past report failures where safe, and writing a machine-readable
run summary:

    data/reports/full_prop_pipeline_summary.json
    data/logs/full_prop_pipeline/<run_id>/<NN>_<step>.log

Research-only: produces no recommendations and no bets. Approved bets and
approved parlays remain blocked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
LOG_ROOT = PROJECT_ROOT / "data" / "logs" / "full_prop_pipeline"
SUMMARY_PATH = REPORTS_DIR / "full_prop_pipeline_summary.json"

# (step name, script filename, extra args, kind, optional)
#   kind "collection": failure is loudly reported but the pipeline continues —
#       downstream steps still work against previously collected data.
#   kind "core": enrichment/settlement; failure continues but is flagged.
#   kind "report": failure continues (reports are independent of each other).
#   optional steps are skipped silently when the script does not exist yet.
PIPELINE_STEPS: list[tuple[str, str, list[str], str, bool]] = [
    ("collect_props", "daily_collect_props.py", [], "collection", False),
    # SportsGameOdds: cheap usage probe (1 request) then guarded collection.
    # Probe/collection failures are warnings — they never break the pipeline.
    ("probe_sportsgameodds", "probe_sportsgameodds.py", ["--cheap"], "report", True),
    ("collect_sportsgameodds", "collect_sportsgameodds_props.py", [], "collection", True),
    # API-Sports stays probe-only (free plan blocks the current season);
    # re-probe at most daily.
    ("probe_apisports", "probe_apisports.py", ["--max-age-hours", "24"], "report", True),
    ("enrich_snapshots", "enrich_player_prop_snapshots.py", [], "core", False),
    ("refresh_results_cache_only", "refresh_nba_results_and_settle_props.py", [], "core", False),
    ("market_quality", "build_player_prop_market_quality.py", [], "report", False),
    ("manual_review", "build_player_prop_manual_review.py", [], "report", False),
    ("collection_health", "build_prop_collection_health.py", [], "report", False),
    ("nba_collection_plan", "build_nba_collection_plan.py", [], "report", False),
    ("settlement_outcomes", "build_player_prop_settlement_outcomes.py", [], "report", False),
    ("prop_clv", "build_player_prop_clv.py", [], "report", False),
    ("data_quality_gates", "build_player_prop_data_quality_gates.py", [], "report", False),
    ("nba_review_exports", "build_nba_prop_review_exports.py", [], "report", True),
    ("all_sports_readiness", "build_all_sports_prop_readiness.py", [], "report", True),
    ("cross_sport_coverage", "build_cross_sport_collection_coverage.py", [], "report", True),
    ("odds_api_quota", "build_odds_api_quota_report.py", [], "report", True),
    ("odds_source_comparison", "build_odds_source_comparison.py", [], "report", True),
    ("odds_source_usage", "build_odds_source_usage_summary.py", [], "report", True),
    ("cross_source_comparison", "build_cross_source_prop_comparison.py", [], "report", True),
    ("next_action", "build_next_action_report.py", [], "report", True),
    ("dashboard", "build_dashboard.py", [], "report", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full research-only prop pipeline.")
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip the Odds API collection step (rebuild reports from existing data only).",
    )
    parser.add_argument(
        "--step-timeout",
        type=int,
        default=1800,
        help="Per-step timeout in seconds (default %(default)s).",
    )
    return parser.parse_args()


def run_step(
    index: int,
    name: str,
    script: Path,
    extra_args: list[str],
    log_dir: Path,
    timeout: int,
) -> dict:
    log_path = log_dir / f"{index:02d}_{name}.log"
    cmd = [sys.executable, str(script), *extra_args]
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    status = "success"
    returncode: int | None = None
    error = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"# step: {name}\n# cmd: {' '.join(cmd)}\n# started_utc: {started.isoformat()}\n\n")
            log_file.flush()
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            returncode = result.returncode
            if returncode != 0:
                status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        error = f"step exceeded {timeout}s"
    except OSError as exc:
        status = "error"
        error = str(exc)
    duration = round(time.monotonic() - t0, 1)
    return {
        "step": name,
        "script": script.name,
        "status": status,
        "returncode": returncode,
        "started_utc": started.isoformat(),
        "duration_seconds": duration,
        "log": str(log_path.relative_to(PROJECT_ROOT)),
        "error": error,
    }


def collect_context() -> dict:
    """Pull a few headline numbers from the freshest reports for the summary."""
    context: dict = {}
    run_summary_path = REPORTS_DIR / "player_prop_collection_run_summary.json"
    gates_path = REPORTS_DIR / "player_prop_data_quality_gates.json"
    try:
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        totals = run_summary.get("totals", {})
        context["snapshots_total"] = totals.get("snapshots_total")
        context["closing_snapshots_total"] = totals.get("closing_snapshots_total")
        context["last_collection_run_id"] = run_summary.get("run_id")
        context["last_collection_status"] = run_summary.get("status")
    except (OSError, json.JSONDecodeError):
        pass
    try:
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        context["data_gate_status"] = gates.get("status")
        context["nba_settled_props"] = gates.get("metrics", {}).get("settled_props")
        context["nba_clv_markets"] = gates.get("metrics", {}).get("clv_markets")
    except (OSError, json.JSONDecodeError):
        pass
    return context


def main() -> int:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Full prop pipeline run {run_id} (research-only)")
    steps: list[dict] = []
    collection_failed = False

    for index, (name, filename, extra_args, kind, optional) in enumerate(PIPELINE_STEPS, start=1):
        script = SCRIPTS_DIR / filename
        if kind == "collection" and args.skip_collection:
            print(f"  [{index:02d}] {name:<28} SKIPPED (--skip-collection)")
            steps.append({"step": name, "script": filename, "status": "skipped", "reason": "--skip-collection"})
            continue
        if not script.exists():
            if optional:
                print(f"  [{index:02d}] {name:<28} SKIPPED (optional script not present)")
                steps.append({"step": name, "script": filename, "status": "skipped", "reason": "script not present"})
                continue
            print(f"  [{index:02d}] {name:<28} MISSING SCRIPT {filename}")
            steps.append({"step": name, "script": filename, "status": "missing"})
            continue

        print(f"  [{index:02d}] {name:<28} running...", flush=True)
        record = run_step(index, name, script, extra_args, log_dir, args.step_timeout)
        record["kind"] = kind
        steps.append(record)
        marker = "OK" if record["status"] == "success" else record["status"].upper()
        print(f"  [{index:02d}] {name:<28} {marker} ({record['duration_seconds']}s) -> {record['log']}")
        if record["status"] != "success":
            if kind == "collection":
                collection_failed = True
                print(f"       COLLECTION FAILED — downstream reports use previously collected data. See {record['log']}")
            else:
                print(f"       step failed; continuing (reports are independent). See {record['log']}")

    failed = [s for s in steps if s.get("status") not in {"success", "skipped"}]
    overall = "success" if not failed else "completed_with_errors"
    summary = {
        "report": "full_prop_pipeline_summary",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "collection_failed": collection_failed,
        "steps": steps,
        "failed_steps": [s["step"] for s in failed],
        "context": collect_context(),
        "log_dir": str(log_dir.relative_to(PROJECT_ROOT)),
        "research_only": True,
        "approved": False,
        "scope": (
            "Research-only data pipeline run. No recommendations, no bets, no parlays. "
            "Approved betting remains blocked."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nPipeline status: {overall}")
    if failed:
        print(f"  Failed steps: {', '.join(s['step'] for s in failed)}")
    print(f"  Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Logs:    {log_dir.relative_to(PROJECT_ROOT)}")
    print("Research-only: no recommendations were produced; approved bets/parlays remain blocked.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
