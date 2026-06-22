"""Odds source comparison report (research-only).

One row per candidate odds source (The Odds API, SportsGameOdds, API-Sports,
Kalshi, manual CSV, Kaggle/historical) with live evidence pulled from probe
summaries, collection summaries, and the normalized snapshot CSV.

Outputs:
    data/reports/odds_source_comparison.json
    data/reports/odds_source_comparison.md
    data/reports/odds_source_adapter_plan.csv

Research-only: no models, no recommendations to bet, no proof-gate changes.
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


def _snapshot_counts() -> dict[str, dict]:
    if not SNAPSHOTS_PATH.exists():
        return {}
    frame = pd.read_csv(SNAPSHOTS_PATH, usecols=["source", "league", "is_closing_snapshot"], low_memory=False)
    out: dict[str, dict] = {}
    for source, group in frame.groupby(frame["source"].astype(str)):
        closing = group["is_closing_snapshot"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"}
        )
        out[source] = {
            "snapshots": int(len(group)),
            "closing_snapshots": int(closing.sum()),
            "leagues": sorted(group["league"].astype(str).unique().tolist()),
        }
    return out


def build_rows() -> list[dict]:
    env = os.environ
    sgo_probe = _read_json(REPORTS_DIR / "sportsgameodds_probe_summary.json")
    sgo_collect = _read_json(REPORTS_DIR / "sportsgameodds_collection_summary.json")
    apisports_probe = _read_json(REPORTS_DIR / "apisports_probe_summary.json")
    odds_api_summary = _read_json(REPORTS_DIR / "player_prop_collection_run_summary.json")
    odds_api_quota = _read_json(REPORTS_DIR / "odds_api_quota_report.json")
    counts = _snapshot_counts()

    odds_api_counts = counts.get("odds_api", {})
    sgo_counts = counts.get("sportsgameodds", {})

    sgo_quota = sgo_collect.get("quota", {}) if isinstance(sgo_collect, dict) else {}
    sgo_usage = sgo_probe.get("usage", {}) if isinstance(sgo_probe, dict) else {}
    sgo_limits = (sgo_usage or {}).get("rateLimits", {})
    apisports_status = (apisports_probe or {}).get("status_by_api", {})

    rows = [
        {
            "source": "odds_api",
            "name": "The Odds API",
            "key_detected": bool(env.get("ODDS_API_KEY")) or bool(odds_api_summary),
            "integration_status": "active_collector",
            "player_props_proven": bool(odds_api_counts.get("snapshots")),
            "sports_leagues": "NBA, WNBA, MLB, NFL, NHL, NCAAB, NCAAF, EPL, MLS, La Liga, Serie A, Bundesliga, Ligue 1, UEFA CL",
            "historical_odds": False,
            "closing_line_support": "poll-near-tip only (no true closing line on free tier)",
            "usage_endpoint": "x-requests-remaining response header",
            "free_tier_limits": "~500 credits/month; prop markets cost extra credits per region",
            "quota_state": f"{odds_api_quota.get('usage', {}).get('quota_remaining', 'n/a')} credits remaining",
            "fields_available": "player, prop, line, over/under price per book; NO team mapping, NO opening line",
            "recommended_use": "Active multi-sport prop collector; reduce league breadth under quota pressure",
            "integration_priority": 1,
            "reduces_odds_api_usage": "n/a (it IS the pressured source)",
            "evidence": f"{odds_api_counts.get('snapshots', 0)} snapshots in normalized CSV",
        },
        {
            "source": "sportsgameodds",
            "name": "SportsGameOdds",
            "key_detected": bool(sgo_probe.get("key_detected")) or bool(env.get("SPORTSGAMEODDS_API_KEY")),
            "integration_status": "active_collector (NBA enabled 2026-06-11)",
            "player_props_proven": bool(sgo_probe.get("player_prop_markets_visible")) or bool(sgo_counts.get("snapshots")),
            "sports_leagues": ", ".join(sgo_probe.get("leagues_available", [])) or "NBA, MLB, MLS, NCAAB, NCAAF, NFL, NHL, UEFA CL",
            "historical_odds": "opening lines included per market (openBookOdds); finished events queryable",
            "closing_line_support": "poll-near-tip + opening line fields; results on event objects",
            "usage_endpoint": "/account/usage (entities + request rate limits)",
            "free_tier_limits": (
                f"tier {sgo_usage.get('tier', 'amateur')}: 10 req/min, "
                f"{(sgo_limits.get('per-month') or {}).get('max-entities', 2500)} entities/month; "
                "1 event ~= 1 entity (ALL its props included)"
            ),
            "quota_state": f"{sgo_quota.get('entities_remaining_after', 'n/a')} monthly entities remaining",
            "fields_available": (
                "player+team mapping, per-book line/price, consensus+fair odds, opening line, "
                "opposingOddID, event results, venue"
            ),
            "recommended_use": "Priority-1 SUPPLEMENT for NBA player props; expand leagues deliberately within entity budget",
            "integration_priority": 1,
            "reduces_odds_api_usage": "YES — one event request returns every prop+book (1 entity); could replace Odds API NBA prop pulls",
            "evidence": f"probe 2026-06-11 OK; {sgo_counts.get('snapshots', 0)} snapshots collected",
        },
        {
            "source": "apisports",
            "name": "API-Sports",
            "key_detected": bool(apisports_probe.get("key_detected")) or bool(env.get("APISPORTS_API_KEY")),
            "integration_status": "probe_only",
            "player_props_proven": str(apisports_probe.get("player_props_available", "unknown")),
            "sports_leagues": "basketball/nba/football(soccer)/baseball/hockey APIs exist; CURRENT SEASON BLOCKED on free plan",
            "historical_odds": "2022-2024 seasons only on free plan",
            "closing_line_support": "unproven",
            "usage_endpoint": "/status per sport API (daily request counts)",
            "free_tier_limits": (
                f"plan {((apisports_status.get('basketball') or {}).get('plan')) or 'Free'}: "
                f"{((apisports_status.get('basketball') or {}).get('requests_limit_day')) or 100} req/day/API; "
                "free plan limited to 2022-2024 seasons"
            ),
            "quota_state": f"{((apisports_status.get('basketball') or {}).get('requests_today'))} requests today",
            "fields_available": "bet-type catalog includes Player Points/Rebounds/Assists/Threes; live values unproven",
            "recommended_use": "Probe/backup only. Re-probe on a paid plan if current-season props are ever needed",
            "integration_priority": 5,
            "reduces_odds_api_usage": "NO (current season blocked on free plan)",
            "evidence": str(apisports_probe.get("plan_restriction") or "probe pending"),
        },
        {
            "source": "kalshi",
            "name": "Kalshi",
            "key_detected": True,  # public endpoints used throughout the project
            "integration_status": "active for game markets; prop tickers (KXNBAPTS/...) discovered, collector not wired",
            "player_props_proven": "prop ticker taxonomy exists; no prop snapshots collected yet",
            "sports_leagues": "NBA (game + some player thresholds), other sports via series discovery",
            "historical_odds": "candle history downloadable per market",
            "closing_line_support": "true closing via candles",
            "usage_endpoint": "none needed (public data, rate-limited)",
            "free_tier_limits": "public API rate limits",
            "quota_state": "n/a",
            "fields_available": "yes/no threshold contracts; threshold->line, yes->over, no->under",
            "recommended_use": "Game markets active; prop-ticker collection is a future supplement (exchange prices, true closing)",
            "integration_priority": 3,
            "reduces_odds_api_usage": "PARTIAL (NBA points/rebounds thresholds only)",
            "evidence": "kalshi taxonomy + candle infrastructure in repo",
        },
        {
            "source": "manual_csv",
            "name": "Manual CSV capture",
            "key_detected": "n/a",
            "integration_status": "template ready (data/templates/player_prop_snapshot_template.csv)",
            "player_props_proven": "n/a (manual entry)",
            "sports_leagues": "any",
            "historical_odds": "manual",
            "closing_line_support": "manual",
            "usage_endpoint": "n/a",
            "free_tier_limits": "n/a",
            "quota_state": "n/a",
            "fields_available": "whatever is captured by hand",
            "recommended_use": "Spot-check/backfill tool only; too slow for systematic collection",
            "integration_priority": 6,
            "reduces_odds_api_usage": "NO",
            "evidence": "schema template exists",
        },
        {
            "source": "kaggle_historical",
            "name": "Kaggle / manual historical datasets",
            "key_detected": "n/a",
            "integration_status": "imported (ehallmar actuals/odds, zachht game-odds CLV)",
            "player_props_proven": False,
            "sports_leagues": "NBA history",
            "historical_odds": "game odds yes; PLAYER PROP LINES ABSENT in every audited dataset",
            "closing_line_support": "zachht has game closing odds",
            "usage_endpoint": "n/a",
            "free_tier_limits": "n/a",
            "quota_state": "n/a",
            "fields_available": "game results, game odds, player actuals; no prop lines",
            "recommended_use": "Settlement actuals + game-odds research only; never a prop-line source",
            "integration_priority": 4,
            "reduces_odds_api_usage": "NO",
            "evidence": "kaggle profiler + import audits in data/reports",
        },
    ]
    return rows


def render_md(rows: list[dict], generated: str) -> str:
    lines = ["# Odds Source Comparison", ""]
    lines.append(f"_Generated {generated}. Research-only._")
    lines.append("")
    lines.append("Verdict: **SportsGameOdds is the priority-1 supplement for player props** "
                 "(probe + first collection succeeded 2026-06-11; 1 event = 1 entity with all "
                 "props/books included). **API-Sports stays probe-only** (free plan blocks the "
                 "current season). The Odds API remains the active multi-league collector.")
    lines.append("")
    for row in rows:
        lines.append(f"## {row['name']} (`{row['source']}`)")
        lines.append("")
        for key, value in row.items():
            if key in {"source", "name"}:
                continue
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    lines.append("_No betting is enabled by this report. Approved bets/parlays remain blocked._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    generated = datetime.now(timezone.utc).isoformat()
    rows = build_rows()
    summary = {
        "report": "odds_source_comparison",
        "generated_at_utc": generated,
        "research_only": True,
        "approved": False,
        "sources": rows,
        "headline": {
            "priority_1_supplement": "sportsgameodds",
            "probe_only": ["apisports"],
            "active": ["odds_api", "sportsgameodds", "kalshi(game markets)"],
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "odds_source_comparison.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (REPORTS_DIR / "odds_source_comparison.md").write_text(
        render_md(rows, generated), encoding="utf-8"
    )
    plan = pd.DataFrame(rows)[
        ["source", "name", "integration_status", "player_props_proven",
         "integration_priority", "recommended_use", "reduces_odds_api_usage"]
    ].sort_values("integration_priority")
    plan.to_csv(REPORTS_DIR / "odds_source_adapter_plan.csv", index=False)
    print(f"Wrote odds_source_comparison.json/.md + odds_source_adapter_plan.csv ({len(rows)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
