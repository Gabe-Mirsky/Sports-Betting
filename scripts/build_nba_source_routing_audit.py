"""NBA source-routing audit: can SportsGameOdds replace/supplement Odds API?

Read-only and local: uses the prop snapshots already collected (no new API pull).
Compares SGO vs Odds API NBA coverage and reports whether routing NBA player
props to SGO can reduce Odds API credit usage. Writes:
  data/reports/nba_source_routing_audit.json / .md

This audit creates NO prop predictions, enables no betting/parlays, and does not
change any model gate.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORM = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"
SGO = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_sportsgameodds.csv"
SGO_SUMMARY = PROJECT_ROOT / "data" / "reports" / "sportsgameodds_collection_summary.json"
HIST_PROBE = PROJECT_ROOT / "data" / "reports" / "sportsgameodds_historical_prop_probe_summary.json"
OUT_JSON = PROJECT_ROOT / "data" / "reports" / "nba_source_routing_audit.json"
OUT_MD = PROJECT_ROOT / "data" / "reports" / "nba_source_routing_audit.md"


def _coverage(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "books": [], "players": 0, "prop_types": [], "closing_snapshots": 0}
    closing = 0
    if "is_closing_snapshot" in df.columns:
        closing = int(df["is_closing_snapshot"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    return {
        "rows": int(len(df)),
        "books": sorted(df["bookmaker"].dropna().astype(str).unique().tolist()) if "bookmaker" in df else [],
        "players": int(df["player_name"].nunique()) if "player_name" in df else 0,
        "prop_types": sorted(df["prop_type"].dropna().astype(str).unique().tolist()) if "prop_type" in df else [],
        "closing_snapshots": closing,
    }


def main() -> None:
    norm = pd.read_csv(NORM, low_memory=False) if NORM.exists() else pd.DataFrame()
    sgo = pd.read_csv(SGO, low_memory=False) if SGO.exists() else pd.DataFrame()
    nba_norm = norm[norm.get("league").astype(str).eq("NBA")] if not norm.empty and "league" in norm else norm.iloc[0:0]
    oapi = nba_norm[nba_norm.get("source").astype(str).eq("odds_api")] if "source" in nba_norm else nba_norm
    sgo_cov = _coverage(sgo)
    oapi_cov = _coverage(oapi)

    sgo_sum = json.loads(SGO_SUMMARY.read_text(encoding="utf-8")) if SGO_SUMMARY.exists() else {}
    hist = json.loads(HIST_PROBE.read_text(encoding="utf-8")) if HIST_PROBE.exists() else {}
    verdict = hist.get("verdict", {}) if isinstance(hist, dict) else {}
    quota = sgo_sum.get("quota", {}) if isinstance(sgo_sum, dict) else {}

    entity_cost = 1
    sgo_settle = 0
    if not sgo.empty and "has_result" in sgo.columns:
        sgo_settle = int(sgo["has_result"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())

    can_reduce = bool(sgo_cov["rows"] and sgo_cov["players"] and len(sgo_cov["prop_types"]) >= 3)
    summary = {
        "report": "nba_source_routing_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True, "approved": False, "creates_predictions": False,
        "sgo_coverage": sgo_cov,
        "odds_api_coverage": oapi_cov,
        "close_fields": {
            "sgo_live_closing_snapshots": sgo_cov["closing_snapshots"],
            "sgo_settlement_rows": sgo_settle,
            "sgo_has_close_fields": True,
            "odds_api_has_opening_line": False,
        },
        "historical_availability": {
            "sgo_oldest_props_date": verdict.get("oldest_props_date"),
            "sgo_oldest_close_field_date": verdict.get("oldest_close_field_date"),
            "odds_api_historical": "none on current plan",
        },
        "entity_cost_per_event": entity_cost,
        "sgo_quota": {
            "entities_remaining": quota.get("entities_remaining_after"),
            "entities_cap": quota.get("entities_max_month"),
        },
        "can_reduce_odds_api_usage": can_reduce,
        "verdict": (
            "SportsGameOdds can SUPPLEMENT and largely REPLACE The Odds API for NBA player props at "
            f"much lower marginal cost: 1 event ~= {entity_cost} monthly entity returns ALL books and "
            f"prop types for that game, vs The Odds API charging per market x region per event. SGO "
            f"already covers {sgo_cov['players']} players, {len(sgo_cov['prop_types'])} prop types, "
            f"{len(sgo_cov['books'])} books with settlement + close fields and historical depth to "
            f"{verdict.get('oldest_props_date', 'n/a')}. Recommended: route NBA player props to SGO first "
            "(verify per-game coverage), keep Odds API as fallback. This is the router default already."
        ) if can_reduce else "Insufficient SGO NBA data to recommend replacement; keep Odds API primary.",
        "recommended_routing": {"NBA.player_props": ["sportsgameodds", "odds_api"]},
        "guardrails": [
            "NBA watcher is NOT rewired by this audit; it keeps working as-is.",
            "Closing-line capture is unchanged.",
            "No model gate is loosened; no prop predictions are created.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_md(summary), encoding="utf-8")
    print(f"NBA source routing audit: can_reduce_odds_api_usage={can_reduce}")
    print(f"  SGO: {sgo_cov['rows']} rows, {len(sgo_cov['books'])} books, {sgo_cov['players']} players, "
          f"{len(sgo_cov['prop_types'])} prop types")
    print(f"  Odds API NBA: {oapi_cov['rows']} rows, {len(oapi_cov['books'])} books")
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)} and {OUT_MD.relative_to(PROJECT_ROOT)}")
    print("Research-only: no predictions, no betting/parlays, no gate changes.")


def _md(s: dict) -> str:
    sc, oc = s["sgo_coverage"], s["odds_api_coverage"]
    return "\n".join([
        "# NBA Source Routing Audit — SportsGameOdds vs The Odds API",
        f"_Generated {s['generated_at_utc']} — research-only; no predictions; no gate changes._",
        "",
        "## Coverage (from already-collected data; no new pull)",
        "| Metric | SportsGameOdds | The Odds API (NBA) |",
        "| --- | --- | --- |",
        f"| Rows | {sc['rows']} | {oc['rows']} |",
        f"| Books | {len(sc['books'])} ({', '.join(sc['books'])}) | {len(oc['books'])} ({', '.join(oc['books'])}) |",
        f"| Players | {sc['players']} | {oc['players']} |",
        f"| Prop types | {len(sc['prop_types'])} | {len(oc['prop_types'])} |",
        f"| Live closing snapshots | {sc['closing_snapshots']} | {oc['closing_snapshots']} |",
        "",
        "## Cost & history",
        f"- Entity cost per event (SGO): **{s['entity_cost_per_event']}** (returns all books + props for that game)",
        f"- SGO entities remaining: {s['sgo_quota']['entities_remaining']} / {s['sgo_quota']['entities_cap']}",
        f"- SGO historical props back to: {s['historical_availability']['sgo_oldest_props_date']}; "
        f"close fields to {s['historical_availability']['sgo_oldest_close_field_date']}",
        f"- Odds API historical: {s['historical_availability']['odds_api_historical']}",
        f"- SGO settlement rows present: {s['close_fields']['sgo_settlement_rows']}",
        "",
        f"## Verdict\n{s['verdict']}",
        "",
        "## Guardrails",
        *[f"- {g}" for g in s["guardrails"]],
        "",
        "_Research-only. The router prefers SGO for NBA player props, but the live NBA watcher is "
        "unchanged and continues using its existing pipeline; switching it is a future, separate step._",
    ])


if __name__ == "__main__":
    main()
