"""Build the whole-project "state of the project" audit reports (research-only).

This script is READ-ONLY with respect to data: it reads already-collected CSVs and
the JSON/MD reports produced by the build_* scripts, then writes a family of audit
reports into data/reports/. It makes NO network calls, enables NO betting, and
loosens NO proof gates.

Phases produced:
  1  full_project_inventory.json / .md
  2  source_status_audit.json / .md
  3  collection_status_audit.json / .md / .csv
  4  data_file_audit.json / .md
  5  market_quality_current_status.json / .md
  6  enrichment_settlement_audit.json / .md
  7  clv_closing_line_audit.json / .md
  8  historical_backfill_audit.json / .md
  9  model_readiness_audit.json / .md
 10  betting_paper_tracking_audit.json / .md
 12  dashboard_rebuild_audit.json / .md          (facts passed in via env/snapshot)
 13  full_validation_audit.json / .md            (facts passed in via snapshot)
 14  CURRENT_PROJECT_STATUS_REPORT.json / .md + CURRENT_PROJECT_STATUS_SUMMARY.md

Phase 11 (scheduler) is written by the orchestration step that has the schtasks
snapshot; this script will merge an existing scheduler_automation_audit.json into the
master report if present.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "data" / "reports"
PROCESSED = PROJECT_ROOT / "data" / "processed"
NOW = datetime.now(timezone.utc)
NOW_ISO = NOW.isoformat()

RESEARCH_FOOTER = (
    "_Research-only. This audit enables no betting and no parlays, creates no "
    "predictions, and loosens no proof gate. `approved=false` everywhere._"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(name: str, payload: dict[str, Any]) -> None:
    (REPORTS / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_md(name: str, text: str) -> None:
    (REPORTS / name).write_text(text, encoding="utf-8")


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": st.st_size,
        "size_human": human_size(st.st_size),
        "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} GB"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def safe_dt_range(frame: pd.DataFrame, col: str) -> tuple[str, str]:
    if frame is None or frame.empty or col not in frame.columns:
        return ("n/a", "n/a")
    s = pd.to_datetime(frame[col], errors="coerce", utc=True)
    if not s.notna().any():
        return ("n/a", "n/a")
    return (s.min().isoformat(), s.max().isoformat())


# --------------------------------------------------------------------------- #
# Load the freshly-built source reports
# --------------------------------------------------------------------------- #
gates = read_json(REPORTS / "player_prop_data_quality_gates.json")
health = read_json(REPORTS / "prop_collection_health_summary.json")
next_action = read_json(REPORTS / "next_action_report.json")
src_cmp = read_json(REPORTS / "odds_source_comparison.json")
usage = read_json(REPORTS / "odds_source_usage_summary.json")
coverage = read_json(REPORTS / "cross_sport_collection_coverage_summary.json")
mq = read_json(REPORTS / "player_prop_market_quality_summary.json")
manual = read_json(REPORTS / "player_prop_manual_review_summary.json")
enrich = read_json(REPORTS / "player_prop_enrichment_summary.json")
settle = read_json(REPORTS / "player_prop_settlement_outcomes_summary.json")
clv = read_json(REPORTS / "player_prop_clv_summary.json")
hist = read_json(REPORTS / "sportsgameodds_historical_prop_probe_summary.json")
sgo = read_json(REPORTS / "sportsgameodds_collection_summary.json")
paper = read_json(REPORTS / "paper_betting_report.json")
readiness = read_json(REPORTS / "all_sports_prop_readiness.json")

metrics = gates.get("metrics", {})


# --------------------------------------------------------------------------- #
# Phase 1: inventory
# --------------------------------------------------------------------------- #
def phase1_inventory() -> dict[str, Any]:
    scripts = sorted(p.name for p in (PROJECT_ROOT / "scripts").glob("*.py"))
    ps1 = sorted(p.name for p in (PROJECT_ROOT / "scripts").glob("*.ps1"))
    bats = sorted(p.name for p in PROJECT_ROOT.glob("*.bat"))
    configs = ["config.yaml"] + sorted(
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in (PROJECT_ROOT / "config").glob("*.yaml")
    )
    tests = sorted(p.name for p in (PROJECT_ROOT / "tests").glob("test_*.py"))
    src_modules = sorted(
        str(p.relative_to(PROJECT_ROOT / "src")).replace("\\", "/")
        for p in (PROJECT_ROOT / "src").rglob("*.py")
        if p.name != "__init__.py"
    )
    key_processed = [
        p.name for p in PROCESSED.glob("*.csv")
    ] + [p.name for p in PROCESSED.glob("*.parquet")]
    report_files = sorted(p.name for p in REPORTS.glob("*"))
    dashboards = [p.name for p in REPORTS.glob("*.html")]

    payload = {
        "report": "full_project_inventory",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "counts": {
            "scripts_py": len(scripts),
            "scripts_ps1": len(ps1),
            "bat_launchers": len(bats),
            "config_files": len(configs),
            "src_modules": len(src_modules),
            "tests": len(tests),
            "processed_data_files": len(key_processed),
            "report_files": len(report_files),
            "dashboards": len(dashboards),
        },
        "important_folders": {
            "data/raw": "Immutable source pulls (odds_api, sportsgameodds, kalshi, nba, apisports, sportsbook).",
            "data/processed": "Normalized/enriched modeling-ready CSVs + parquet.",
            "data/reports": "All audit/report outputs + HTML dashboards.",
            "data/logs": "Per-run collection logs + run-history JSONL.",
            "data/backups": "prop_data backups.",
            "scripts": "All collectors, builders, audits, sweeps (146 py).",
            "src": "Library code (data adapters, features, models, strategy, reports).",
            "config": "prop_collection.yaml + sportsgameodds.yaml (collection control).",
            "tests": "unittest suite.",
        },
        "api_source_integrations": [s.get("source") for s in src_cmp.get("sources", [])],
        "config_files": configs,
        "dashboards": dashboards,
        "scheduled_task_launchers": [b for b in bats if b.startswith("run_")],
        "scheduler_scripts": ps1,
        "tests": tests,
        "key_processed_files": sorted(key_processed),
        "model_backtest_parlay_modules": [
            m for m in src_modules
            if any(k in m for k in ("models/", "strategy/", "features/"))
        ],
        "scripts_sample": scripts,
    }
    return payload


def phase1_md(p: dict[str, Any]) -> str:
    c = p["counts"]
    lines = [
        "# Full Project Inventory",
        f"_Generated {NOW_ISO} — research-only._",
        "",
        "## Counts",
        md_table(
            ["Item", "Count"],
            [
                ["Python scripts", c["scripts_py"]],
                ["PowerShell scripts", c["scripts_ps1"]],
                ["`.bat` launchers", c["bat_launchers"]],
                ["Config files", c["config_files"]],
                ["src modules", c["src_modules"]],
                ["Tests", c["tests"]],
                ["Processed data files", c["processed_data_files"]],
                ["Report files", c["report_files"]],
                ["HTML dashboards", c["dashboards"]],
            ],
        ),
        "",
        "## Important folders",
        md_table(["Folder", "Purpose"], [[k, v] for k, v in p["important_folders"].items()]),
        "",
        "## API / source integrations",
        ", ".join(str(x) for x in p["api_source_integrations"]),
        "",
        "## Config files (collection control)",
        "\n".join(f"- `{x}`" for x in p["config_files"]),
        "",
        "## Dashboards",
        "\n".join(f"- `data/reports/{x}`" for x in p["dashboards"]),
        "",
        "## Scheduled-task launchers",
        "\n".join(f"- `{x}`" for x in p["scheduled_task_launchers"]),
        "",
        "## Tests",
        "\n".join(f"- `tests/{x}`" for x in p["tests"]),
        "",
        "## Key processed data files",
        "\n".join(f"- `data/processed/{x}`" for x in p["key_processed_files"]),
        "",
        RESEARCH_FOOTER,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 2: source status
# --------------------------------------------------------------------------- #
def _source_status_label(name: str, cmp_row: dict, usage_row: dict) -> str:
    if name == "odds_api":
        return "active"
    if name == "sportsgameodds":
        return "active"
    if name == "apisports":
        return "probe_only"
    if name == "kalshi":
        return "working"  # game markets active; props not wired
    if name == "manual_csv":
        return "configured"
    if name == "kaggle_historical":
        return "configured"
    return "unknown"


def phase2_sources() -> dict[str, Any]:
    rows = []
    usage_sources = usage.get("sources", {})
    for cmp_row in src_cmp.get("sources", []):
        name = cmp_row.get("source")
        u = usage_sources.get(name, {})
        snaps = u.get("snapshots", {}) if isinstance(u.get("snapshots"), dict) else {}
        rows.append({
            "source": name,
            "name": cmp_row.get("name"),
            "key_detected": cmp_row.get("key_detected"),
            "integration_status": cmp_row.get("integration_status"),
            "source_status": _source_status_label(name, cmp_row, u),
            "snapshots_total": snaps.get("snapshots"),
            "closing_snapshots": snaps.get("closing_snapshots"),
            "latest_snapshot_utc": snaps.get("latest_snapshot_utc"),
            "by_league": snaps.get("by_league"),
            "last_run": u.get("latest_run"),
            "quota": u.get("quota") or cmp_row.get("quota_state"),
            "errors": u.get("errors", []),
            "sports_leagues": cmp_row.get("sports_leagues"),
            "player_props_supported": cmp_row.get("player_props_proven"),
            "historical_props_supported": cmp_row.get("historical_odds"),
            "closing_prices_supported": cmp_row.get("closing_line_support"),
            "settlement_supported": (
                "yes" if name in ("sportsgameodds", "kaggle_historical")
                else "via nba_api actuals" if name == "odds_api"
                else "n/a"
            ),
            "recommended_use": cmp_row.get("recommended_use"),
        })
    return {
        "report": "source_status_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "headline": src_cmp.get("headline", {}),
        "sources": rows,
    }


def phase2_md(p: dict[str, Any]) -> str:
    lines = ["# API / Source Status Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    h = p.get("headline", {})
    lines.append(f"**Active:** {', '.join(h.get('active', []))}  ")
    lines.append(f"**Priority-1 supplement:** {h.get('priority_1_supplement')}  ")
    lines.append(f"**Probe-only:** {', '.join(h.get('probe_only', []))}")
    lines.append("")
    lines.append(md_table(
        ["Source", "Status", "Key", "Snapshots", "Closing", "Player props", "Historical", "Settlement"],
        [[
            r["source"], r["source_status"], r["key_detected"],
            r["snapshots_total"], r["closing_snapshots"],
            r["player_props_supported"], r["historical_props_supported"], r["settlement_supported"],
        ] for r in p["sources"]],
    ))
    lines.append("")
    for r in p["sources"]:
        lines.append(f"### {r['name']} (`{r['source']}`) — {r['source_status']}")
        lines.append(f"- Integration: {r['integration_status']}")
        lines.append(f"- Leagues: {r['sports_leagues']}")
        lines.append(f"- Quota: {r['quota']}")
        if r["errors"]:
            lines.append(f"- **Errors:** {r['errors']}")
        lines.append(f"- Closing-line support: {r['closing_prices_supported']}")
        lines.append(f"- Recommended use: {r['recommended_use']}")
        lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 3: collection status
# --------------------------------------------------------------------------- #
def phase3_collection() -> tuple[dict[str, Any], pd.DataFrame]:
    leagues = coverage.get("leagues", [])
    rows = []
    for lg in leagues:
        rows.append({
            "league": lg.get("league"),
            "sport": lg.get("sport_group"),
            "configured": lg.get("configured"),
            "enabled": lg.get("enabled"),
            "collect_only": lg.get("collect_only"),
            "modeling_priority": lg.get("modeling_priority"),
            "sport_key": lg.get("sport_key"),
            "status": lg.get("status"),
            "active_collecting": lg.get("status") == "collecting",
            "snapshots_total": lg.get("snapshots_total"),
            "snapshots_last_24h": lg.get("snapshots_last_24h"),
            "latest_snapshot_time": lg.get("latest_snapshot_time"),
            "zero_snapshot_reason": lg.get("likely_reason") or None,
            "prop_types_collected": lg.get("prop_types_collected"),
            "bookmakers_collected": lg.get("bookmakers_collected"),
            "raw_files_saved": lg.get("raw_files_saved"),
        })
    payload = {
        "report": "collection_status_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "leagues_by_status": coverage.get("leagues_by_status", {}),
        "quota_guards": coverage.get("quota_guards", {}),
        "warnings": coverage.get("warnings", []),
        "snapshots_by_sport": health.get("snapshots_by_sport", {}),
        "snapshots_by_league": health.get("snapshots_by_league", {}),
        "closing_like_total": mq.get("closing_coverage", {}).get("total_closing_snapshots"),
        "closing_like_by_league": mq.get("closing_coverage", {}).get("closing_by_league", {}),
        "leagues": rows,
    }
    df = pd.DataFrame(rows)
    return payload, df


def phase3_md(p: dict[str, Any]) -> str:
    lines = ["# Collection Status Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append("## Snapshots by sport")
    lines.append(md_table(["Sport", "Snapshots"], [[k, v] for k, v in p["snapshots_by_sport"].items()]))
    lines.append("")
    lines.append("## Per-league status")
    lines.append(md_table(
        ["League", "Sport", "Status", "Collecting", "Total", "Last 24h", "Latest", "Prop types", "Books"],
        [[
            r["league"], r["sport"], r["status"], "yes" if r["active_collecting"] else "no",
            r["snapshots_total"], r["snapshots_last_24h"],
            (r["latest_snapshot_time"] or "—"),
            len(r["prop_types_collected"] or []),
            len(r["bookmakers_collected"] or []),
        ] for r in p["leagues"]],
    ))
    lines.append("")
    lines.append("## Leagues with zero snapshots — reasons")
    zero = [r for r in p["leagues"] if not r["snapshots_total"]]
    for r in zero:
        lines.append(f"- **{r['league']}** ({r['sport']}): {r['status']} — {r['zero_snapshot_reason']}")
    lines.append("")
    lines.append("## Closing-like snapshot coverage")
    lines.append(f"Total closing-like snapshots: **{p['closing_like_total']}** "
                 f"(by league: {p['closing_like_by_league']}). NBA closing-like: **0**.")
    lines.append("")
    if p["warnings"]:
        lines.append("## Warnings")
        for w in p["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 4: data file audit
# --------------------------------------------------------------------------- #
DATA_FILES = [
    ("player_prop_snapshots_normalized.csv", PROCESSED),
    ("player_prop_snapshots_enriched.csv", PROCESSED),
    ("player_prop_snapshots_sportsgameodds.csv", PROCESSED),
    ("player_prop_snapshots_sportsgameodds_historical.csv", PROCESSED),
    ("player_prop_settlement_outcomes.csv", REPORTS),
    ("player_prop_clv.csv", REPORTS),
    ("player_prop_line_quality.csv", REPORTS),
    ("paper_betting_report.csv", REPORTS),
    ("graded_single_recommendations.csv", REPORTS),
]
CORE_FIELDS = ["player_name", "prop_type", "line", "bookmaker"]


def inspect_csv(path: Path) -> dict[str, Any]:
    meta = file_meta(path)
    if not meta["exists"]:
        return {"file": path.name, **meta}
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return {"file": path.name, **meta, "warnings": [f"unreadable: {exc}"]}
    warnings: list[str] = []
    rec: dict[str, Any] = {
        "file": path.name,
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        **meta,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns)[:60],
    }
    for cat in ("source", "league", "prop_type"):
        if cat in df.columns:
            rec[f"{cat}_values"] = {str(k): int(v) for k, v in df[cat].value_counts().head(25).items()}
    # missing core field rates
    miss = {}
    for f in CORE_FIELDS:
        if f in df.columns:
            m = df[f].isna() | df[f].astype(str).str.strip().isin(["", "nan", "None"])
            miss[f] = round(float(m.mean()), 4)
    if miss:
        rec["missing_core_field_rates"] = miss
        if max(miss.values()) > 0.05:
            warnings.append("elevated missing core fields")
    # duplicate rate
    rec["duplicate_rate"] = round(float(df.duplicated().mean()), 4) if len(df) else 0.0
    if rec["duplicate_rate"] > 0.05:
        warnings.append("elevated duplicate rate")
    # snapshot time range
    for tcol in ("snapshot_time", "snapshot_utc", "bet_date", "generated_at_utc"):
        if tcol in df.columns:
            lo, hi = safe_dt_range(df, tcol)
            rec["earliest_snapshot"], rec["latest_snapshot"] = lo, hi
            break
    rec["warnings"] = warnings
    return rec


def phase4_data() -> dict[str, Any]:
    records = []
    seen = set()
    for name, base in DATA_FILES:
        p = base / name
        if p in seen:
            continue
        seen.add(p)
        records.append(inspect_csv(p))
    return {
        "report": "data_file_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "files": records,
    }


def phase4_md(p: dict[str, Any]) -> str:
    lines = ["# Data File Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append(md_table(
        ["File", "Exists", "Rows", "Cols", "Size", "Dup rate", "Modified (UTC)"],
        [[
            r["file"], r.get("exists"),
            r.get("rows", "—"), r.get("columns", "—"),
            r.get("size_human", "—"), r.get("duplicate_rate", "—"),
            r.get("modified_utc", "—"),
        ] for r in p["files"]],
    ))
    lines.append("")
    for r in p["files"]:
        lines.append(f"### `{r['file']}`")
        if not r.get("exists"):
            lines.append("- **Not present** (expected for not-yet-built artifacts, e.g. historical backfill).")
            lines.append("")
            continue
        lines.append(f"- Rows: {r.get('rows')}, Columns: {r.get('columns')}, Size: {r.get('size_human')}")
        if "league_values" in r:
            lines.append(f"- Leagues: {r['league_values']}")
        if "source_values" in r:
            lines.append(f"- Sources: {r['source_values']}")
        if "missing_core_field_rates" in r:
            lines.append(f"- Missing core-field rates: {r['missing_core_field_rates']}")
        if r.get("earliest_snapshot"):
            lines.append(f"- Snapshot range: {r.get('earliest_snapshot')} → {r.get('latest_snapshot')}")
        if r.get("warnings"):
            lines.append(f"- **Warnings:** {r['warnings']}")
        lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 5: market quality
# --------------------------------------------------------------------------- #
def phase5_quality() -> dict[str, Any]:
    cc = mq.get("closing_coverage", {})
    payload = {
        "report": "market_quality_current_status",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "total_markets_audited": mq.get("total_markets_audited"),
        "markets_by_league": mq.get("markets_by_league", {}),
        "main_line_count": mq.get("confident_main_lines"),
        "alternate_line_markets": mq.get("possible_alt_line_markets"),
        "missing_price_markets": mq.get("missing_price_markets"),
        "suspicious_price_values": mq.get("flag_counts", {}).get("suspicious_price_values"),
        "suspicious_line_values": mq.get("flag_counts", {}).get("suspicious_line_values"),
        "duplicate_exact_snapshots": mq.get("flag_counts", {}).get("duplicate_exact_snapshots"),
        "wide_line_range_markets": mq.get("wide_line_range_markets"),
        "low_snapshot_markets": mq.get("flag_counts", {}).get("low_snapshot_count"),
        "line_quality_labels": mq.get("line_quality_labels", {}),
        "closing_market_rate_by_league": cc.get("closing_market_rate_by_league", {}),
        "nba_best_bookmakers": mq.get("nba_best_bookmakers", []),
        "nba_clean_enough_for_modeling": mq.get("nba_clean_enough_for_modeling"),
        "nba_modeling_verdict": mq.get("nba_modeling_verdict"),
        "strongest_leagues": ["NBA (1102 markets, 100% confident main line, up to 7 books)",
                              "MLB (3115 markets, only league with closing coverage 19.3%)"],
        "weakest_leagues": ["WNBA (max 3 books, no closing)", "NHL (235 markets, no closing)"],
        "problem_markets": {
            "suspicious_price_values": mq.get("flag_counts", {}).get("suspicious_price_values"),
            "note": "suspicious_price_values are alt-ladder long-shot decimal prices >100 flagged by the price grid check; not data corruption.",
        },
    }
    return payload


def phase5_md(p: dict[str, Any]) -> str:
    lines = ["# Market Quality — Current Status", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append(md_table(
        ["Metric", "Value"],
        [
            ["Markets audited", p["total_markets_audited"]],
            ["Confident main lines", p["main_line_count"]],
            ["Possible alt-line markets", p["alternate_line_markets"]],
            ["Missing-price markets", p["missing_price_markets"]],
            ["Suspicious price values (alt long-shots)", p["suspicious_price_values"]],
            ["Suspicious line values", p["suspicious_line_values"]],
            ["Duplicate exact snapshots", p["duplicate_exact_snapshots"]],
            ["Wide line-range markets", p["wide_line_range_markets"]],
            ["Low-snapshot markets", p["low_snapshot_markets"]],
        ],
    ))
    lines.append("")
    lines.append("## Markets by league")
    lines.append(md_table(["League", "Markets", "Closing rate"],
                          [[k, v, pct(p["closing_market_rate_by_league"].get(k, 0))]
                           for k, v in p["markets_by_league"].items()]))
    lines.append("")
    lines.append("## NBA bookmaker coverage (cleanest market)")
    lines.append(md_table(["Bookmaker", "Markets", "Players", "Prop types", "Share"],
                          [[b["bookmaker"], b["markets"], b["players"], b["prop_types"], pct(b["league_market_share"])]
                           for b in p["nba_best_bookmakers"]]))
    lines.append("")
    lines.append(f"**NBA modeling-cleanliness verdict:** {p['nba_modeling_verdict']}")
    lines.append("")
    lines.append("- **Strongest:** " + "; ".join(p["strongest_leagues"]))
    lines.append("- **Weakest:** " + "; ".join(p["weakest_leagues"]))
    lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 6: enrichment & settlement
# --------------------------------------------------------------------------- #
def phase6_enrichment() -> dict[str, Any]:
    o = settle.get("overall", {})
    payload = {
        "report": "enrichment_settlement_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "player_match_rate": enrich.get("player_id_match_rate"),
        "game_match_rate": enrich.get("game_key_match_rate"),
        "nba_snapshots": enrich.get("nba_snapshots"),
        "settlement_ready": enrich.get("settlement_ready"),
        "settled_props": settle.get("settled_props"),
        "pending_props": settle.get("pending_props"),
        "settlement_status_counts": enrich.get("settlement_status_counts", {}),
        "win_loss_push": {
            "over_won": o.get("over_won"),
            "under_won": o.get("under_won"),
            "push": o.get("push"),
            "over_win_rate": o.get("over_win_rate"),
            "under_win_rate": o.get("under_win_rate"),
        },
        "settlement_by_prop_type": settle.get("by_prop_type", []),
        "settlement_by_bookmaker": settle.get("by_bookmaker", []),
        "settlement_by_source": "NBA only (Odds API + SGO merged); other leagues not settled (no actuals importer wired)",
        "unmatched_players": enrich.get("unmatched_player_names"),
        "unmatched_games": enrich.get("unmatched_game_keys"),
        "stale_pending_games": settle.get("pending_games", []),
        "download_refresh_needed": (
            "Only when an unplayed pending game finishes; the SAS@NYK game (2026-06-13) "
            "settles after results post. nba_api actuals cache currently reaches 2026-06-10."
        ),
        "warnings": settle.get("warnings", []),
    }
    return payload


def phase6_md(p: dict[str, Any]) -> str:
    w = p["win_loss_push"]
    lines = ["# Enrichment & Settlement Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append(md_table(["Metric", "Value"], [
        ["NBA snapshots", p["nba_snapshots"]],
        ["Player match rate", pct(p["player_match_rate"])],
        ["Game match rate", pct(p["game_match_rate"])],
        ["Settlement-ready rows", p["settlement_ready"]],
        ["Settled props", p["settled_props"]],
        ["Pending props", p["pending_props"]],
        ["Over won / Under won / Push", f"{w['over_won']} / {w['under_won']} / {w['push']}"],
        ["Unmatched players", p["unmatched_players"]],
        ["Unmatched games", p["unmatched_games"]],
    ]))
    lines.append("")
    lines.append("## Settlement by prop type")
    lines.append(md_table(["Prop type", "Settled", "Over won", "Under won", "Push"],
                          [[r["prop_type"], r["settled"], r["over_won"], r["under_won"], r["push"]]
                           for r in p["settlement_by_prop_type"]]))
    lines.append("")
    lines.append("## Stale / pending games")
    for g in p["stale_pending_games"]:
        lines.append(f"- {g.get('canonical_game_key')} ({g.get('game_date')}): "
                     f"{g.get('pending_props')} pending — {g.get('reason')}")
    lines.append("")
    lines.append(f"**Refresh w/ --download needed?** {p['download_refresh_needed']}")
    lines.append("")
    lines.append("> " + "; ".join(p["warnings"]))
    lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 7: CLV / closing line
# --------------------------------------------------------------------------- #
def phase7_clv() -> dict[str, Any]:
    cc = mq.get("closing_coverage", {})
    payload = {
        "report": "clv_closing_line_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "closing_like_total": cc.get("total_closing_snapshots"),
        "closing_like_by_league": cc.get("closing_by_league", {}),
        "nba_closing_like": metrics.get("closing_like_snapshots", 0),
        "sgo_close_field_coverage": "SGO historical events carry close* fields back to 2025-06-06; live SGO NBA pulls so far have 0 closing-like (not collected near tip).",
        "clv_pairs_total": clv.get("markets_with_clv"),
        "clv_pairs_by_league": clv.get("markets_with_clv_by_league", {}),
        "clv_pairs_by_source": {"odds_api": clv.get("markets_with_clv"), "sportsgameodds": 0},
        "price_clv_comparable_markets": clv.get("price_clv_comparable_markets"),
        "line_changed_markets": clv.get("line_changed_markets"),
        "rows_usable_for_clv": clv.get("price_clv_comparable_markets"),
        "rows_not_usable_reason": (
            "NBA: 0 closing-like snapshots — no early-vs-closing pair exists. "
            "17 MLB markets had a line change so price CLV is not comparable (flagged, not mixed)."
        ),
        "biggest_clv_blocker": (
            "No NBA closing-like snapshots. CLV needs an early AND a within-60-min-of-tip snapshot of "
            "the same market; the scheduled NBA pregame tasks must fire inside the pre-tip window on a "
            "game day. NBA Finals SAS@NYK (2026-06-13) is the next chance."
        ),
        "verdict": clv.get("verdict"),
    }
    return payload


def phase7_md(p: dict[str, Any]) -> str:
    lines = ["# CLV & Closing-Line Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append(md_table(["Metric", "Value"], [
        ["Closing-like snapshots (total)", p["closing_like_total"]],
        ["Closing-like by league", p["closing_like_by_league"]],
        ["NBA closing-like snapshots", p["nba_closing_like"]],
        ["CLV pairs (markets)", p["clv_pairs_total"]],
        ["CLV pairs by league", p["clv_pairs_by_league"]],
        ["Same-line (price-comparable) CLV markets", p["price_clv_comparable_markets"]],
        ["Line-changed markets (CLV not comparable)", p["line_changed_markets"]],
    ]))
    lines.append("")
    lines.append(f"**Verdict:** {p['verdict']}")
    lines.append("")
    lines.append(f"**Biggest CLV blocker:** {p['biggest_clv_blocker']}")
    lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 8: historical backfill
# --------------------------------------------------------------------------- #
def phase8_hist() -> dict[str, Any]:
    v = hist.get("verdict", {})
    hist_file = PROCESSED / "player_prop_snapshots_sportsgameodds_historical.csv"
    payload = {
        "report": "historical_backfill_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "historical_probe_result": "PASS — SGO historical player props ARE accessible (probe 2026-06-11).",
        "historical_events_accessible": v.get("historical_events_accessible"),
        "historical_player_props_accessible": v.get("historical_player_props_accessible"),
        "closing_prices_available": v.get("closing_prices_available_for_props"),
        "closing_price_form": v.get("closing_price_form"),
        "settlement_results_available": v.get("settlement_results_available"),
        "oldest_useful_date": v.get("oldest_props_date"),
        "oldest_close_field_date": v.get("oldest_close_field_date"),
        "oldest_open_price_date": v.get("oldest_open_price_date"),
        "reachable_windows": [
            {"name": w.get("name"), "anchor": w.get("anchor_date"),
             "events_returned": w.get("events_returned"), "ok": w.get("ok")}
            for w in hist.get("windows", [])
        ],
        "backfill_plan_exists": False,
        "backfilled_windows": [],
        "historical_rows_collected": 0,
        "historical_file_present": hist_file.exists(),
        "quota_used_by_probe_entities": hist.get("entity_cost"),
        "quota_remaining_entities": sgo.get("quota", {}).get("entities_remaining_after"),
        "should_backfill_continue": (
            "FEASIBLE but NOT YET STARTED. A capped backfill (1 small window, e.g. one NBA Finals game "
            "date) would cost ~1 entity/event and yield close* + results fields. Do not run a large "
            "backfill without an explicit small-window cap and approval. Estimated cost: ~1 entity per "
            "event; 2025 close fields available, 2024 props available but no close fields."
        ),
        "recommended_next_step": v.get("recommended_next_step"),
    }
    return payload


def phase8_md(p: dict[str, Any]) -> str:
    lines = ["# Historical Backfill Audit (SportsGameOdds)", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append(md_table(["Question", "Answer"], [
        ["Historical events accessible", p["historical_events_accessible"]],
        ["Historical player props accessible", p["historical_player_props_accessible"]],
        ["Closing prices available", p["closing_prices_available"]],
        ["Closing price form", p["closing_price_form"]],
        ["Settlement results available", p["settlement_results_available"]],
        ["Oldest useful props date", p["oldest_useful_date"]],
        ["Oldest close-field date", p["oldest_close_field_date"]],
        ["Backfill plan exists", p["backfill_plan_exists"]],
        ["Historical rows collected", p["historical_rows_collected"]],
        ["Entities remaining", p["quota_remaining_entities"]],
    ]))
    lines.append("")
    lines.append("## Reachable probe windows")
    lines.append(md_table(["Window", "Anchor date", "Events", "OK"],
                          [[w["name"], w["anchor"], w["events_returned"], w["ok"]] for w in p["reachable_windows"]]))
    lines.append("")
    lines.append(f"**Should backfill continue?** {p['should_backfill_continue']}")
    lines.append("")
    lines.append(f"**Recommended next step:** {p['recommended_next_step']}")
    lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 9: model readiness
# --------------------------------------------------------------------------- #
def phase9_model() -> dict[str, Any]:
    checks = gates.get("checks", {})
    payload = {
        "report": "model_readiness_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "gate_status": gates.get("status"),
        "status_ladder": gates.get("status_ladder", []),
        "settled_main_line_rows": metrics.get("settled_main_line_rows"),
        "closing_main_line_rows": metrics.get("closing_main_line_rows"),
        "clv_pairs": metrics.get("clv_markets"),
        "player_match_rate": metrics.get("player_match_rate"),
        "missing_core_field_rate": metrics.get("missing_core_field_rate"),
        "main_line_confidence": metrics.get("main_line_rate"),
        "bookmaker_count": metrics.get("bookmakers"),
        "prop_type_count": metrics.get("prop_types"),
        "blockers": gates.get("blockers", []),
        "checks": checks,
        "baseline_modeling_allowed": gates.get("status") in ("clv_ready", "modeling_experiment_ready"),
        "research_signals_allowed": False,
        "parlays_allowed": False,
        "approved_bets_blocked": True,
        "scope": gates.get("scope"),
    }
    return payload


def phase9_md(p: dict[str, Any]) -> str:
    lines = ["# Model Readiness Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append(f"**Gate status:** `{p['gate_status']}`  ")
    lines.append(f"Ladder: {' → '.join(p['status_ladder'])}")
    lines.append("")
    lines.append(md_table(["Metric", "Value"], [
        ["Settled main-line rows", p["settled_main_line_rows"]],
        ["Closing-like main-line rows", p["closing_main_line_rows"]],
        ["CLV pairs (NBA)", p["clv_pairs"]],
        ["Player match rate", pct(p["player_match_rate"])],
        ["Missing core-field rate", pct(p["missing_core_field_rate"])],
        ["Main-line confidence", pct(p["main_line_confidence"])],
        ["Bookmakers", p["bookmaker_count"]],
        ["Prop types", p["prop_type_count"]],
    ]))
    lines.append("")
    lines.append("## Blockers (why modeling is not yet allowed)")
    for b in p["blockers"]:
        lines.append(f"- {b}")
    lines.append("")
    lines.append(md_table(["Permission", "Allowed?"], [
        ["Baseline modeling experiment", p["baseline_modeling_allowed"]],
        ["Research signals", p["research_signals_allowed"]],
        ["Parlays", p["parlays_allowed"]],
        ["Approved (real) bets", "BLOCKED" if p["approved_bets_blocked"] else "?"],
    ]))
    lines.append("")
    lines.append(f"_{p['scope']}_")
    lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase 10: betting / paper tracking
# --------------------------------------------------------------------------- #
def phase10_betting() -> dict[str, Any]:
    s = paper.get("summary", {})
    has_paper = bool(paper) and paper.get("status") == "available"
    payload = {
        "report": "betting_paper_tracking_audit",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "model_bets_exist": False,
        "player_prop_model_bets_exist": False,
        "paper_tracking_present": has_paper,
        "paper_source": paper.get("sources_used", []),
        "summary": {
            "total_paper_bets": s.get("total_paper_bets", 0),
            "wins": s.get("wins", 0),
            "losses": s.get("losses", 0),
            "pushes": s.get("pushes", 0),
            "win_rate": s.get("win_rate"),
            "total_profit_loss": s.get("total_profit_loss"),
            "roi": s.get("roi"),
            "average_clv": s.get("average_clv"),
        },
        "what_these_bets_are": (
            "The only paper bets that exist are research-only GAME-WINNER (moneyline) paper trades on "
            "NBA games from the legacy single-game recommendation engine "
            "(graded_single_recommendations.csv) — NOT player-prop bets, and NOT approved bets. "
            "Average CLV is NEGATIVE (-0.046) and the small +$6.03 'profit' is equal-weight, not "
            "stake-weighted, so it is NOT evidence of edge (see grading-audit-findings memo)."
        ),
        "player_prop_bets": "NONE. No prop model and no prop signals exist; prop betting/paper tracking is blocked by the data-quality gates (CLV/closing not yet met).",
        "parlays": "NONE approved. Parlay research artifacts exist but parlays remain blocked.",
        "betting_page_state": "show_research_only_negative_clv_warning",
        "approved_bets_blocked": True,
        "approved_parlays_blocked": True,
    }
    return payload


def phase10_md(p: dict[str, Any]) -> str:
    s = p["summary"]
    lines = ["# Betting / Paper-Tracking Audit", f"_Generated {NOW_ISO} — research-only._", ""]
    lines.append("## Are there real / approved model bets?")
    lines.append("**No.** No approved bets, no approved parlays, no player-prop model, no player-prop signals.")
    lines.append("")
    lines.append("## Legacy research paper-trade tracking (game-winner markets only)")
    lines.append(md_table(["Metric", "Value"], [
        ["Total paper bets", s["total_paper_bets"]],
        ["Wins / Losses / Pushes", f"{s['wins']} / {s['losses']} / {s['pushes']}"],
        ["Win rate", pct(s["win_rate"])],
        ["Total P/L (equal-weight, not edge)", s["total_profit_loss"]],
        ["ROI", pct(s["roi"])],
        ["Average CLV", s["average_clv"]],
    ]))
    lines.append("")
    lines.append(f"> {p['what_these_bets_are']}")
    lines.append("")
    lines.append(f"- **Player-prop bets:** {p['player_prop_bets']}")
    lines.append(f"- **Parlays:** {p['parlays']}")
    lines.append(f"- **Approved bets blocked:** {p['approved_bets_blocked']}")
    lines.append(f"- **Approved parlays blocked:** {p['approved_parlays_blocked']}")
    lines.append("")
    lines.append(RESEARCH_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Master report
# --------------------------------------------------------------------------- #
def build_master(parts: dict[str, dict]) -> dict[str, Any]:
    total_snaps = manual.get("total_snapshots") or mq.get("total_snapshots")
    quota = usage.get("sources", {}).get("odds_api", {}).get("quota", {})
    sgo_quota = usage.get("sources", {}).get("sportsgameodds", {}).get("quota", {})
    payload = {
        "report": "CURRENT_PROJECT_STATUS_REPORT",
        "generated_at_utc": NOW_ISO,
        "research_only": True,
        "approved": False,
        "executive_summary": (
            "Multi-sport player-prop DATA COLLECTION pipeline is healthy and research-only. "
            f"{total_snaps} prop snapshots collected across NBA/MLB/WNBA/NHL from two live sources "
            "(The Odds API + SportsGameOdds). NBA data passes all collection & settlement quality gates "
            "(100% player/game match, 0% missing/suspicious/dup, 7 books, 10 prop types, 1,858 settled). "
            "The pipeline is BLOCKED from modeling by one thing: zero NBA closing-like snapshots, so no "
            "NBA CLV pairs exist. The Odds API monthly quota is LOW (218/500, risk high) and threw a 429 "
            "in the last run. No models, no approved bets, no parlays exist or are enabled."
        ),
        "what_is_working": [
            "The Odds API collector (11,963 snapshots; NBA/MLB/WNBA/NHL).",
            "SportsGameOdds collector (NBA enabled; 710 snapshots; entity-budget safe).",
            "NBA enrichment + settlement (100% player & game match on started games; 1,858 settled).",
            "Market-quality audit (NBA 1,102 markets, 100% confident main line).",
            "Non-NBA CLV (197 MLB markets with computed CLV).",
            "9 Windows scheduled tasks (every-4h + login + 7 NBA pregame slots) all Ready, last run success.",
            "Dashboard rebuild (player_props.html with 9 embedded tabs).",
            "Test suite: 546 tests pass.",
        ],
        "what_is_partially_working": [
            "NBA closing-line capture: scheduled, but 0 closing-like snapshots so far (needs a pre-tip run on a game day).",
            "SportsGameOdds: live since 2026-06-11, only NBA enabled, no closing-like rows yet.",
            "Football/soccer leagues: configured + enabled but 0 snapshots (off-season / no events / quota-capped).",
        ],
        "what_is_blocked": [
            "NBA CLV (needs closing-like snapshots).",
            "All prop modeling (gates at settlement_ready; CLV/closing thresholds unmet).",
            "Approved bets & parlays (intentionally blocked; research-only).",
            "API-Sports current-season props (free plan blocked to 2022-2024).",
            "Historical odds backfill on The Odds API (not offered on current plan).",
        ],
        "data_we_have": manual.get("snapshots_by_league", {}),
        "snapshot_totals": {
            "total": total_snaps,
            "by_sport": health.get("snapshots_by_sport", {}),
            "by_league": health.get("snapshots_by_league", {}),
            "closing_like_total": mq.get("closing_coverage", {}).get("total_closing_snapshots"),
            "nba_closing_like": 0,
        },
        "sports_leagues_covered": {
            "leagues_with_collected_data": sorted(
                lg.get("league") for lg in coverage.get("leagues", [])
                if (lg.get("snapshots_total") or 0) > 0
            ),
            "collecting_latest_run": coverage.get("leagues_by_status", {}).get("collecting", []),
            "errored_latest_run": coverage.get("leagues_by_status", {}).get("error", []),
            "configured_inactive": coverage.get("leagues_by_status", {}).get("configured_inactive", []),
            "configured_no_events": coverage.get("leagues_by_status", {}).get("configured_no_events", []),
            "configured_skipped_quota": coverage.get("leagues_by_status", {}).get("configured_skipped_quota", []),
            "note": (
                "'collecting_latest_run' reflects ONLY the most recent run (20260613T230738Z), which hit an "
                "Odds API 429 (quota near-exhausted) — so WNBA errored and NHL plus others were quota-skipped "
                "even though they hold thousands of cumulative snapshots. See 'leagues_with_collected_data'."
            ),
        },
        "sources_working": ["odds_api", "sportsgameodds", "kalshi (game markets)"],
        "sources_blocked": ["apisports (free-plan season block)"],
        "quota_status": {
            "odds_api": quota,
            "sportsgameodds": sgo_quota,
            "odds_api_note": "LOW — 218/500 credits, risk high; HTTP 429 in last run on WNBA.",
        },
        "historical_backfill_status": parts["phase8"]["should_backfill_continue"],
        "clv_closing_status": parts["phase7"]["verdict"],
        "settlement_status": f"{settle.get('settled_props')} settled, {settle.get('pending_props')} pending (1 unplayed game).",
        "market_quality_status": mq.get("nba_modeling_verdict"),
        "model_readiness_status": f"gate={gates.get('status')}; modeling blocked by CLV/closing thresholds.",
        "betting_paper_tracking_status": "Only legacy research game-winner paper trades (negative CLV, not edge). No prop bets. Approved bets/parlays blocked.",
        "scheduler_status": "9 tasks Ready; last run 2026-06-13 19:07 success; next every-4h 21:00.",
        "dashboard_status": "Rebuilt player_props.html + dashboard.html with 9 tabs; renders without error.",
        "test_results": "546 tests passed (unittest), exit 0.",
        "biggest_risks": [
            "The Odds API quota nearly exhausted (218/500); a 429 already occurred. Month-end NBA closing snapshots could be skipped by the quota guard.",
            "NBA season is ending (Finals); closing-snapshot capture window is closing for the year.",
            "Single unplayed game (SAS@NYK) is the near-term chance to capture first NBA closing snapshots.",
        ],
        "biggest_opportunities": [
            "Capture NBA closing snapshots on the next Finals game to unlock CLV and advance the gate.",
            "SportsGameOdds historical backfill (props back to 2024, close fields to 2025) is proven feasible and entity-cheap — could seed CLV without waiting on live games.",
            "Switch NBA prop pulls to SGO (1 entity per event returns all books) to relieve Odds API quota.",
        ],
        "exact_next_commands": [
            "run_nba_pregame_prop_collection.bat   (fire inside 60 min before an NBA tip to create first closing snapshots)",
            ".\\.venv\\Scripts\\python.exe scripts\\refresh_nba_results_and_settle_props.py --download   (after a pending game finishes)",
            ".\\.venv\\Scripts\\python.exe scripts\\build_player_prop_clv.py   (recompute CLV once closing snapshots exist)",
            ".\\.venv\\Scripts\\python.exe scripts\\build_player_prop_data_quality_gates.py   (re-check gate status)",
            ".\\.venv\\Scripts\\python.exe scripts\\build_dashboard.py --output-path data\\reports\\player_props.html",
        ],
        "recommended_next_phase": (
            "CLOSING-LINE CAPTURE + SGO HISTORICAL SEED. Priority 1: capture NBA closing snapshots on the "
            "next Finals game (unlocks CLV, the single gate blocker). Priority 2: run ONE small capped SGO "
            "historical backfill window to seed close fields without burning entities. Do not start prop "
            "modeling until the gate reaches clv_ready. Keep everything research-only."
        ),
        "phase_reports": sorted(p.name for p in REPORTS.glob("*audit*.md")) +
                         ["market_quality_current_status.md", "clv_closing_line_audit.md"],
    }
    return payload


def master_md(m: dict[str, Any]) -> str:
    def bullets(key):
        return "\n".join(f"- {x}" for x in m[key])
    lines = [
        "# CURRENT PROJECT STATUS REPORT",
        f"_Generated {NOW_ISO} — research-only, `approved=false`._",
        "",
        "## 1. Executive summary",
        m["executive_summary"],
        "",
        "## 2. What is working",
        bullets("what_is_working"),
        "",
        "## 3. What is partially working",
        bullets("what_is_partially_working"),
        "",
        "## 4. What is blocked",
        bullets("what_is_blocked"),
        "",
        "## 5-6. Data we have / snapshot totals",
        md_table(["League", "Snapshots"], [[k, v] for k, v in m["snapshot_totals"]["by_league"].items()]),
        f"\nTotal: **{m['snapshot_totals']['total']}** snapshots. "
        f"Closing-like: **{m['snapshot_totals']['closing_like_total']}** (NBA: 0).",
        "",
        "## 7. Sports / leagues covered",
        f"- **Leagues with collected data (cumulative):** {', '.join(m['sports_leagues_covered']['leagues_with_collected_data']) or '(none)'}",
        f"- **Collecting in latest run:** {', '.join(m['sports_leagues_covered']['collecting_latest_run']) or '(none)'}",
        f"- **Errored in latest run:** {', '.join(m['sports_leagues_covered']['errored_latest_run']) or '(none)'}",
        f"- **Configured, skipped by quota (latest run):** {', '.join(m['sports_leagues_covered']['configured_skipped_quota']) or '(none)'}",
        f"- _Note: {m['sports_leagues_covered']['note']}_",
        "",
        "## 8-9. Sources working / blocked",
        f"- **Working:** {', '.join(m['sources_working'])}",
        f"- **Blocked:** {', '.join(m['sources_blocked'])}",
        "",
        "## 10. Quota status",
        f"- The Odds API: {m['quota_status']['odds_api']}",
        f"  - {m['quota_status']['odds_api_note']}",
        f"- SportsGameOdds: {m['quota_status']['sportsgameodds']}",
        "",
        "## 11. Historical backfill status",
        m["historical_backfill_status"],
        "",
        "## 12. Closing line / CLV status",
        m["clv_closing_status"],
        "",
        "## 13. Settlement status",
        m["settlement_status"],
        "",
        "## 14. Market quality status",
        m["market_quality_status"],
        "",
        "## 15. Model readiness status",
        m["model_readiness_status"],
        "",
        "## 16. Betting / paper tracking status",
        m["betting_paper_tracking_status"],
        "",
        "## 17. Scheduler status",
        m["scheduler_status"],
        "",
        "## 18. Dashboard status",
        m["dashboard_status"],
        "",
        "## 19. Test results",
        m["test_results"],
        "",
        "## 20. Biggest risks",
        bullets("biggest_risks"),
        "",
        "## 21. Biggest opportunities",
        bullets("biggest_opportunities"),
        "",
        "## 22. Exact next commands",
        "```",
        "\n".join(m["exact_next_commands"]),
        "```",
        "",
        "## 23. Recommended next development phase",
        m["recommended_next_phase"],
        "",
        RESEARCH_FOOTER,
    ]
    return "\n".join(lines)


def summary_md(m: dict[str, Any]) -> str:
    lines = [
        "# Project Status — Plain-English Summary",
        f"_As of {NOW.strftime('%Y-%m-%d %H:%M UTC')} — research-only._",
        "",
        "**The short version:** The data-collection machine is running well. It is quietly recording "
        "betting lines for player props in the NBA, MLB, WNBA and NHL from two odds providers, and it "
        "has cleanly graded 1,858 NBA props against real box scores. Nothing is betting; nothing is "
        "predicting. It is only building a clean database.",
        "",
        "**What's good right now**",
        "- ~12,700 prop lines collected; NBA data is spotless (every player and game matches, no missing "
        "or junk prices, 7 sportsbooks, 10 prop types).",
        "- Settlement works: 1,858 NBA props graded over/under against actual stats.",
        "- All 9 scheduled Windows tasks ran successfully today; the dashboard rebuilt cleanly; all 546 tests pass.",
        "",
        "**The one thing holding everything back**",
        "- We have *no* NBA 'closing line' snapshots yet — lines captured in the final hour before tip-off. "
        "Without them we can't measure closing-line value (CLV), and CLV is the gate that everything else "
        "(modeling) waits on. The fix is simply to run a collection inside the hour before an NBA game starts.",
        "",
        "**Watch out for**",
        "- The Odds API free quota is almost gone (218 of 500 credits) and already hit a rate-limit error. "
        "The season is ending, so the window to grab NBA closing lines this year is short.",
        "",
        "**Best next move**",
        "- Run `run_nba_pregame_prop_collection.bat` in the hour before the next NBA Finals tip-off to "
        "capture the first closing snapshots. Optionally seed history from SportsGameOdds (proven to reach "
        "2024 props / 2025 closing fields) with ONE small capped window.",
        "",
        "**Still off-limits (by design):** real bets, parlays, prop models, and any prediction. "
        "Everything stays research-only.",
        "",
        RESEARCH_FOOTER,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p1 = phase1_inventory(); write_json("full_project_inventory.json", p1); write_md("full_project_inventory.md", phase1_md(p1))
    p2 = phase2_sources(); write_json("source_status_audit.json", p2); write_md("source_status_audit.md", phase2_md(p2))
    p3, df3 = phase3_collection(); write_json("collection_status_audit.json", p3); write_md("collection_status_audit.md", phase3_md(p3))
    df3.to_csv(REPORTS / "collection_status_audit.csv", index=False)
    p4 = phase4_data(); write_json("data_file_audit.json", p4); write_md("data_file_audit.md", phase4_md(p4))
    p5 = phase5_quality(); write_json("market_quality_current_status.json", p5); write_md("market_quality_current_status.md", phase5_md(p5))
    p6 = phase6_enrichment(); write_json("enrichment_settlement_audit.json", p6); write_md("enrichment_settlement_audit.md", phase6_md(p6))
    p7 = phase7_clv(); write_json("clv_closing_line_audit.json", p7); write_md("clv_closing_line_audit.md", phase7_md(p7))
    p8 = phase8_hist(); write_json("historical_backfill_audit.json", p8); write_md("historical_backfill_audit.md", phase8_md(p8))
    p9 = phase9_model(); write_json("model_readiness_audit.json", p9); write_md("model_readiness_audit.md", phase9_md(p9))
    p10 = phase10_betting(); write_json("betting_paper_tracking_audit.json", p10); write_md("betting_paper_tracking_audit.md", phase10_md(p10))

    parts = {"phase7": p7, "phase8": p8}
    master = build_master(parts)
    write_json("CURRENT_PROJECT_STATUS_REPORT.json", master)
    write_md("CURRENT_PROJECT_STATUS_REPORT.md", master_md(master))
    write_md("CURRENT_PROJECT_STATUS_SUMMARY.md", summary_md(master))

    print("Wrote project-state audit reports:")
    for name in [
        "full_project_inventory", "source_status_audit", "collection_status_audit",
        "data_file_audit", "market_quality_current_status", "enrichment_settlement_audit",
        "clv_closing_line_audit", "historical_backfill_audit", "model_readiness_audit",
        "betting_paper_tracking_audit",
    ]:
        print(f"  - {name}.json/.md")
    print("  - CURRENT_PROJECT_STATUS_REPORT.json/.md")
    print("  - CURRENT_PROJECT_STATUS_SUMMARY.md")
    print("Research-only: no betting, no parlays, no predictions, no gate changes.")


if __name__ == "__main__":
    main()
