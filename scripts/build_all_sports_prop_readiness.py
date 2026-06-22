"""Build the all-sports prop readiness report (research-only).

For every configured league, reports how far the prop database is from
supporting modeling: snapshots collected, prop types, player/team mapping,
result/settlement support, CLV support, model readiness, and the next adapter
that would have to be built.

Outputs:
    data/reports/all_sports_prop_readiness.json
    data/reports/all_sports_prop_readiness.md

NBA stays the modeling priority. Everything else is collect-only until its
infrastructure (results importer + player mapping) exists.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"

LEAGUES = [
    "NBA", "WNBA", "MLB", "NHL", "NFL", "NCAAB", "NCAAF",
    "EPL", "MLS", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1", "UEFA_CL",
]

# Per-league infrastructure facts. Only NBA has working actuals + player
# mapping (nba_api). CLV support is league-agnostic (early vs closing
# snapshots of the same market), so it is computed from the CLV report.
LEAGUE_INFRA = {
    "NBA": {
        "player_mapping": "implemented (nba_api player ids, 100% match in production)",
        "settlement": "implemented (nba_api game + player logs, settled props live)",
        "next_adapter": "none for data; next milestone is closing-snapshot coverage for CLV",
    },
    "WNBA": {
        "player_mapping": "not implemented",
        "settlement": "not implemented",
        "next_adapter": "WNBA player game-log importer (stats.wnba.com via nba_api-style client) + player-name matcher",
    },
    "MLB": {
        "player_mapping": "not implemented",
        "settlement": "not implemented",
        "next_adapter": "MLB StatsAPI results importer (free) + batter/pitcher name matcher",
    },
    "NHL": {
        "player_mapping": "not implemented",
        "settlement": "not implemented",
        "next_adapter": "NHL API player game-log importer + name matcher",
    },
    "NFL": {
        "player_mapping": "not implemented",
        "settlement": "not implemented",
        "next_adapter": "nflverse weekly player-stats importer + name matcher (season starts Sep)",
    },
    "NCAAB": {
        "player_mapping": "not implemented",
        "settlement": "not implemented",
        "next_adapter": "college box-score source (e.g. sportsdataverse) + large-roster name matcher",
    },
    "NCAAF": {
        "player_mapping": "not implemented",
        "settlement": "not implemented",
        "next_adapter": "college football stats source (cfbd/sportsdataverse) + large-roster name matcher (season starts Aug/Sep)",
    },
}
SOCCER_INFRA = {
    "player_mapping": "not implemented",
    "settlement": "not implemented",
    "next_adapter": "football-data / FBref shots+assists importer + player-name matcher (no soccer models planned)",
}


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def build_readiness() -> dict:
    snaps = pd.read_csv(SNAPSHOTS_PATH, low_memory=False) if SNAPSHOTS_PATH.exists() else pd.DataFrame()
    clv = {}
    clv_path = REPORTS_DIR / "player_prop_clv_summary.json"
    if clv_path.exists():
        try:
            clv = json.loads(clv_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            clv = {}
    gates = {}
    gates_path = REPORTS_DIR / "player_prop_data_quality_gates.json"
    if gates_path.exists():
        try:
            gates = json.loads(gates_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gates = {}

    clv_by_league = clv.get("markets_with_clv_by_league") or {}

    rows = []
    for league in LEAGUES:
        league_snaps = (
            snaps[snaps["league"].astype(str).eq(league)] if not snaps.empty else pd.DataFrame()
        )
        n = int(len(league_snaps))
        prop_types = (
            sorted(league_snaps["prop_type"].dropna().astype(str).unique().tolist())
            if n else []
        )
        closing = (
            int(_truthy(league_snaps.get("is_closing_snapshot", pd.Series(dtype="object"))).sum())
            if n else 0
        )
        clv_markets = int(clv_by_league.get(league, 0) or 0)
        infra = LEAGUE_INFRA.get(league, SOCCER_INFRA)

        if league == "NBA":
            model_readiness = (
                f"modeling priority - data gates: {gates.get('status', 'unknown')}"
            )
        elif n == 0:
            model_readiness = "collect-only - no snapshots yet (inactive/off-season)"
        else:
            model_readiness = "collect-only - infrastructure not built (by design)"

        clv_support = (
            f"working ({clv_markets} markets with CLV)" if clv_markets
            else "supported by design (needs early + closing snapshots of the same market)"
            if n else "no data yet"
        )

        rows.append({
            "league": league,
            "snapshots_collected": n,
            "closing_like_snapshots": closing,
            "prop_types_collected": prop_types,
            "player_mapping_status": infra["player_mapping"],
            "settlement_support_status": infra["settlement"],
            "clv_support_status": clv_support,
            "model_readiness_status": model_readiness,
            "next_required_adapter": infra["next_adapter"],
        })

    return {
        "report": "all_sports_prop_readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modeling_priority_league": "NBA",
        "leagues": rows,
        "policy": (
            "NBA is the only modeling-priority league. All other leagues remain "
            "collect-only (building the historical snapshot database) until their "
            "results importer and player mapping exist. Soccer has no models planned."
        ),
        "research_only": True,
        "approved": False,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# All-Sports Prop Readiness",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        summary["policy"],
        "",
        "| league | snapshots | closing | prop types | player mapping | settlement | CLV | model readiness |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["leagues"]:
        lines.append(
            f"| {row['league']} | {row['snapshots_collected']} | {row['closing_like_snapshots']} | "
            f"{', '.join(row['prop_types_collected']) or '-'} | {row['player_mapping_status']} | "
            f"{row['settlement_support_status']} | {row['clv_support_status']} | "
            f"{row['model_readiness_status']} |"
        )
    lines += ["", "## Next required adapter per league", ""]
    for row in summary["leagues"]:
        lines.append(f"- **{row['league']}**: {row['next_required_adapter']}")
    lines += [
        "",
        "---",
        "Research-only. Approved bets and approved parlays remain blocked for every league.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the all-sports prop readiness report.")
    parser.parse_args()

    summary = build_readiness()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "all_sports_prop_readiness.json"
    md_path = REPORTS_DIR / "all_sports_prop_readiness.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    for row in summary["leagues"]:
        print(
            f"  {row['league']:<10} snapshots={row['snapshots_collected']:<6} "
            f"closing={row['closing_like_snapshots']:<5} {row['model_readiness_status']}"
        )
    print(f"Wrote: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    print("Research-only: NBA stays modeling priority; everything else collect-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
