"""World Cup results fetch + easy-market settlement (research-only).

Settles ONLY simple, unambiguous game/team markets against final scores:
  * match_winner_1x2 (1X2 / moneyline): home / away / draw
  * total_goals (totals): over / under / push
  * both_teams_to_score (BTTS): yes / no
  * team_total_* (team totals): over / under / push, when the team is named
Player-prop settlement is intentionally NOT attempted (no player stat coverage).

This module runs no models and produces no recommendations or bets. Settlement
results are research labels (won/lost/push), never wagers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data.prop_collection import make_default_fetch_json, _odds_api_url
from data.world_cup_collection import _sport_key

SETTLEABLE_MARKETS = ("match_winner_1x2", "total_goals", "both_teams_to_score")


def fetch_world_cup_scores(
    api_key: str,
    config: dict[str, Any],
    fetch_json: Callable[[str], Any],
    *,
    days_from: int = 1,
) -> Any:
    """Fetch recent/live scores. Odds API /scores costs ~1-2 credits."""
    src = config.get("source", {})
    base_url = src.get("base_url", "https://api.the-odds-api.com/v4")
    params = {"apiKey": api_key, "daysFrom": int(days_from)}
    url = _odds_api_url(base_url, f"/sports/{_sport_key(config)}/scores", params)
    return fetch_json(url)


def parse_scores_payload(payload: Any) -> dict[str, dict[str, Any]]:
    """Map an Odds API /scores payload to {event_id: {teams, scores, completed}}."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(payload, list):
        return out
    for ev in payload:
        if not isinstance(ev, dict):
            continue
        scores = {}
        for s in ev.get("scores") or []:
            name = s.get("name")
            try:
                scores[name] = int(s.get("score"))
            except (TypeError, ValueError):
                continue
        home, away = ev.get("home_team"), ev.get("away_team")
        out[str(ev.get("id") or "")] = {
            "home_team": home, "away_team": away,
            "home_score": scores.get(home), "away_score": scores.get(away),
            "completed": bool(ev.get("completed")),
        }
    return out


def settle_market(
    market_type: str,
    outcome_name: str,
    line: Any,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> str:
    """Return 'won' / 'lost' / 'push' / 'unsupported' for one market outcome."""
    if home_score is None or away_score is None:
        return "unsupported"
    name = str(outcome_name or "").strip()
    mt = str(market_type or "")

    if mt == "match_winner_1x2":
        if name.lower() == "draw":
            return "won" if home_score == away_score else "lost"
        if name == home_team:
            return "won" if home_score > away_score else "lost"
        if name == away_team:
            return "won" if away_score > home_score else "lost"
        return "unsupported"

    if mt == "total_goals":
        try:
            ln = float(line)
        except (TypeError, ValueError):
            return "unsupported"
        total = home_score + away_score
        side = name.lower()
        if side.startswith("over"):
            return "push" if total == ln else ("won" if total > ln else "lost")
        if side.startswith("under"):
            return "push" if total == ln else ("won" if total < ln else "lost")
        return "unsupported"

    if mt == "both_teams_to_score":
        both = home_score > 0 and away_score > 0
        side = name.lower()
        if side in ("yes", "y"):
            return "won" if both else "lost"
        if side in ("no", "n"):
            return "won" if not both else "lost"
        return "unsupported"

    return "unsupported"


def settle_snapshots(snapshots: pd.DataFrame, scores: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Grade each settleable snapshot row against final scores.

    Returns one row per (event, market, bookmaker, outcome) with a `result`
    column. Player-prop rows (if any) and unsupported markets are labeled
    'unsupported' and excluded from win/loss tallies by callers.
    """
    if snapshots.empty:
        return snapshots.assign(result=pd.Series(dtype="object"))
    rows = []
    for _, r in snapshots.iterrows():
        eid = str(r.get("event_id"))
        sc = scores.get(eid)
        if not sc or not sc.get("completed") or sc.get("home_score") is None:
            result = "pending"
        else:
            result = settle_market(
                r.get("market_type"), r.get("outcome_name"), r.get("line"),
                r.get("home_team"), r.get("away_team"),
                sc.get("home_score"), sc.get("away_score"),
            )
        rows.append({**r.to_dict(), "result": result,
                     "home_score": (scores.get(eid) or {}).get("home_score"),
                     "away_score": (scores.get(eid) or {}).get("away_score")})
    return pd.DataFrame(rows)


def run_world_cup_results_refresh(
    config: dict[str, Any],
    project_root: str | Path,
    *,
    dry_run: bool = False,
    days_from: int = 1,
    env: dict[str, str] | None = None,
    fetch_json: Callable[[str], Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch scores (guarded) and settle easy World Cup markets."""
    root = Path(project_root)
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    src_cfg = config.get("source", {})
    out_cfg = config.get("output", {})
    quota_cfg = config.get("quota", {})
    min_remaining = float(quota_cfg.get("results_min_remaining", quota_cfg.get("min_remaining_requests", 100)) or 0)
    api_key = src_cfg.get("api_key") or env.get(src_cfg.get("api_key_env", "ODDS_API_KEY"), "")

    summary: dict[str, Any] = {
        "report": "world_cup_results_summary", "generated_at_utc": now.isoformat(),
        "research_only": True, "approved": False, "dry_run": bool(dry_run),
        "key_detected": bool(api_key), "status": "unknown",
        "events_with_scores": 0, "completed_events": 0,
        "settled_rows": 0, "won": 0, "lost": 0, "push": 0, "unsupported": 0, "pending": 0,
        "by_market_type": {}, "credits_remaining_after": None, "blockers": [], "errors": [],
    }

    processed_path = root / out_cfg.get("processed_path", "data/processed/world_cup_odds_snapshots_normalized.csv")
    snapshots = pd.read_csv(processed_path, low_memory=False) if processed_path.exists() else pd.DataFrame()
    summary["snapshot_rows"] = int(len(snapshots))

    if dry_run:
        summary["status"] = "dry_run_ok"
        summary["blockers"].append("dry-run: no /scores call made (no credits spent)")
        return summary
    if not api_key:
        summary["status"] = "no_key"
        summary["blockers"].append("ODDS_API_KEY not set")
        return summary

    if fetch_json is None:
        fetch_json = make_default_fetch_json()
    try:
        payload = fetch_world_cup_scores(api_key, config, fetch_json, days_from=days_from)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "scores_error"
        summary["errors"].append(repr(exc))
        return summary
    summary["credits_remaining_after"] = getattr(fetch_json, "quota_remaining", None)

    scores = parse_scores_payload(payload)
    summary["events_with_scores"] = len(scores)
    summary["completed_events"] = sum(1 for v in scores.values() if v.get("completed"))

    settled = settle_snapshots(snapshots, scores)
    if not settled.empty and "result" in settled.columns:
        counts = settled["result"].value_counts().to_dict()
        summary["settled_rows"] = int(sum(counts.get(k, 0) for k in ("won", "lost", "push")))
        for k in ("won", "lost", "push", "unsupported", "pending"):
            summary[k] = int(counts.get(k, 0))
        graded = settled[settled["result"].isin(["won", "lost", "push"])]
        summary["by_market_type"] = {
            str(mt): int(n) for mt, n in graded["market_type"].value_counts().items()
        }
        out_path = root / "data" / "processed" / "world_cup_settlement_outcomes.csv"
        graded.to_csv(out_path, index=False)
        summary["settlement_path"] = str(out_path.relative_to(root)).replace("\\", "/")

    summary["status"] = "settled" if summary["settled_rows"] else "no_settlements"
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "world_cup_results_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
