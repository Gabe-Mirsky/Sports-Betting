"""Small, safe SportsGameOdds API probe (research-only).

Makes a handful of cheap requests — account usage, sports, leagues, a small
markets sample, and one tiny NBA events query — saves every raw response under
data/raw/sportsgameodds/probe/, and writes:

    data/reports/sportsgameodds_probe_summary.json
    data/reports/sportsgameodds_probe.md

Quota-careful by design: no broad collection, hard caps on every request.
No models, no recommendations, no betting changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.sportsgameodds_client import (  # noqa: E402
    API_KEY_ENV,
    SportsGameOddsClient,
    extract_items,
    save_raw_payload,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sportsgameodds" / "probe"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SUMMARY_PATH = REPORTS_DIR / "sportsgameodds_probe_summary.json"
MD_PATH = REPORTS_DIR / "sportsgameodds_probe.md"

TARGET_LEAGUES = ("NBA", "NFL", "MLB", "NHL", "WNBA", "EPL", "MLS")

# oddID side tokens that indicate player props when statEntityID is a player id.
_TEAM_ENTITY_TOKENS = {"home", "away", "all", "side1", "side2"}


def _step(client: SportsGameOddsClient, name: str, result: dict, steps: list[dict]) -> dict:
    """Record one probe step (raw saving + compact status)."""

    raw_path = None
    if result.get("data") is not None:
        raw_path = save_raw_payload(result["data"], RAW_DIR, name)
    record = {
        "step": name,
        "ok": bool(result.get("ok")),
        "status": result.get("status"),
        "error": result.get("error"),
        "attempts": result.get("attempts"),
        "headers": result.get("headers") or {},
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)) if raw_path else None,
    }
    steps.append(record)
    flag = "OK" if record["ok"] else f"FAIL ({record['error']})"
    print(f"  [{name}] {flag}")
    return record


def _looks_like_player_entity(stat_entity: str) -> bool:
    token = str(stat_entity or "").strip()
    return bool(token) and token.lower() not in _TEAM_ENTITY_TOKENS


def analyze_events(events: list) -> dict:
    """Inspect a tiny events sample: fields, odds shape, player-prop visibility."""

    info: dict = {
        "events_returned": len(events),
        "event_fields": [],
        "odds_entries_sampled": 0,
        "player_prop_odds_seen": 0,
        "game_odds_seen": 0,
        "sample_player_prop_oddids": [],
        "sample_game_oddids": [],
        "odds_fields": [],
        "has_players_map": False,
        "has_results": False,
        "stat_ids_seen": [],
    }
    stat_ids: set[str] = set()
    odds_fields: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        info["event_fields"] = sorted(set(info["event_fields"]) | set(event.keys()))
        if isinstance(event.get("players"), dict) and event["players"]:
            info["has_players_map"] = True
        if event.get("results"):
            info["has_results"] = True
        odds = event.get("odds")
        if not isinstance(odds, dict):
            continue
        for odd_id, odd in odds.items():
            info["odds_entries_sampled"] += 1
            if isinstance(odd, dict):
                odds_fields |= set(odd.keys())
                stat_entity = odd.get("statEntityID") or ""
                stat_id = odd.get("statID") or ""
                if stat_id:
                    stat_ids.add(str(stat_id))
            else:
                stat_entity = ""
            parts = str(odd_id).split("-")
            entity_from_id = parts[1] if len(parts) >= 5 else ""
            entity = str(stat_entity or entity_from_id)
            if _looks_like_player_entity(entity):
                info["player_prop_odds_seen"] += 1
                if len(info["sample_player_prop_oddids"]) < 8:
                    info["sample_player_prop_oddids"].append(str(odd_id))
            else:
                info["game_odds_seen"] += 1
                if len(info["sample_game_oddids"]) < 5:
                    info["sample_game_oddids"].append(str(odd_id))
    info["odds_fields"] = sorted(odds_fields)
    info["stat_ids_seen"] = sorted(stat_ids)[:40]
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the SportsGameOdds API (small, quota-safe).")
    parser.add_argument("--skip-events", action="store_true", help="Skip the tiny NBA events query.")
    parser.add_argument("--events-limit", type=int, default=2, help="Max NBA events to request (default 2).")
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="Usage-only probe (1 request, ~0 entities) for scheduled pipeline runs. "
             "The full probe costs ~470 monthly entities (metadata endpoints) and should "
             "only be run manually.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    client = SportsGameOddsClient()
    steps: list[dict] = []
    blockers: list[str] = []

    summary: dict = {
        "report": "sportsgameodds_probe_summary",
        "generated_at_utc": now.isoformat(),
        "key_detected": client.has_key,
        "key_env_var": API_KEY_ENV,
        "research_only": True,
        "approved": False,
    }

    if not client.has_key:
        blockers.append(f"{API_KEY_ENV} is not set; probe skipped gracefully (no requests made).")
        summary.update({
            "key_works": False,
            "integration_can_proceed": False,
            "blockers": blockers,
            "steps": [],
        })
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        MD_PATH.write_text(render_md(summary), encoding="utf-8")
        print(f"No {API_KEY_ENV} detected; wrote skip report.")
        return 0

    if args.cheap:
        # Usage-only refresh: keeps the rich full-probe summary intact and just
        # updates key/quota state (safe for every scheduled pipeline run).
        usage_result = client.account_usage()
        usage_items, _ = extract_items(usage_result.get("data"))
        existing = {}
        if SUMMARY_PATH.exists():
            try:
                existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing.update({
            "report": "sportsgameodds_probe_summary",
            "probe_mode": "cheap_usage_refresh",
            "usage_checked_at_utc": now.isoformat(),
            "key_detected": True,
            "key_works": bool(usage_result.get("ok")),
            "usage": usage_items[0] if usage_items and isinstance(usage_items[0], dict) else existing.get("usage"),
            "research_only": True,
            "approved": False,
        })
        existing.setdefault("generated_at_utc", now.isoformat())
        SUMMARY_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        MD_PATH.write_text(render_md(existing), encoding="utf-8")
        print(f"Cheap usage refresh: key_works={existing['key_works']} (1 request)")
        return 0

    print("Probing SportsGameOdds (small, quota-safe)...")

    # 1. Account usage (cheap; proves the key works and reports quota).
    usage_result = client.account_usage()
    _step(client, "account_usage", usage_result, steps)
    usage_items, _ = extract_items(usage_result.get("data"))
    summary["key_works"] = bool(usage_result.get("ok"))
    summary["usage"] = usage_items[0] if usage_items and isinstance(usage_items[0], dict) else None
    summary["rate_limit_headers"] = usage_result.get("headers") or {}
    if not usage_result.get("ok"):
        # A failing usage endpoint does not always mean a dead key; try /sports
        # before declaring a blocker.
        print("  account/usage failed; trying /sports before concluding the key is bad.")

    # 2. Sports.
    sports_result = client.sports()
    _step(client, "sports", sports_result, steps)
    sports_items, _ = extract_items(sports_result.get("data"))
    sport_ids = sorted(
        {str(s.get("sportID")) for s in sports_items if isinstance(s, dict) and s.get("sportID")}
    )
    summary["sports_available"] = sport_ids
    summary["n_sports"] = len(sport_ids)
    if sports_result.get("ok") and not summary.get("key_works"):
        summary["key_works"] = True

    # 3. Leagues.
    leagues_result = client.leagues()
    _step(client, "leagues", leagues_result, steps)
    league_items, _ = extract_items(leagues_result.get("data"))
    league_ids = sorted(
        {str(l.get("leagueID")) for l in league_items if isinstance(l, dict) and l.get("leagueID")}
    )
    summary["leagues_available"] = league_ids
    summary["n_leagues"] = len(league_ids)
    summary["target_league_availability"] = {
        league: league in league_ids for league in TARGET_LEAGUES
    }

    # 4. Markets metadata (one small page; classification data for the adapter).
    markets_result = client.markets()
    _step(client, "markets", markets_result, steps)
    market_items, _ = extract_items(markets_result.get("data"))
    summary["markets_endpoint_ok"] = bool(markets_result.get("ok"))
    summary["n_markets_sampled"] = len(market_items)

    # 5. Stats metadata for NBA (helps map statID -> prop_type).
    stats_result = client.stats(leagueID="NBA")
    _step(client, "stats_nba", stats_result, steps)
    stats_items, _ = extract_items(stats_result.get("data"))
    summary["stats_endpoint_ok"] = bool(stats_result.get("ok"))
    summary["n_nba_stats_sampled"] = len(stats_items)

    # 6. Tiny NBA events query (the real player-prop visibility test).
    if args.skip_events:
        summary["events_probe"] = {"skipped": True}
    else:
        events_result = client.events(
            leagueID="NBA",
            startsAfter=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            startsBefore=(now + timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            oddsAvailable="true",
            limit=max(1, min(int(args.events_limit), 3)),
        )
        _step(client, "events_nba", events_result, steps)
        events_items, _ = extract_items(events_result.get("data"))
        events_info = analyze_events(events_items)
        # Off-season fallback: if no upcoming NBA events have odds, look at
        # recent finished events instead (proves fields/props/results shape).
        if events_result.get("ok") and not events_items:
            fallback = client.events(
                leagueID="NBA",
                startsAfter=(now - timedelta(days=21)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                startsBefore=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                limit=max(1, min(int(args.events_limit), 3)),
            )
            _step(client, "events_nba_recent", fallback, steps)
            fallback_items, _ = extract_items(fallback.get("data"))
            if fallback_items:
                events_info = analyze_events(fallback_items)
                events_info["window"] = "recent_past_21d"
        events_info["ok"] = bool(events_result.get("ok"))
        summary["events_probe"] = events_info
        summary["player_prop_markets_visible"] = events_info.get("player_prop_odds_seen", 0) > 0

    summary["requests_made"] = client.requests_made
    summary["last_rate_limit_headers"] = dict(client.last_headers)

    # Verdict.
    key_works = bool(summary.get("key_works"))
    props_visible = bool(summary.get("player_prop_markets_visible"))
    events_ok = bool((summary.get("events_probe") or {}).get("ok"))
    if not key_works:
        blockers.append("API key did not authenticate against /account/usage or /sports.")
    if key_works and not events_ok and not args.skip_events:
        blockers.append("/events query failed; player-prop visibility unproven.")
    if key_works and events_ok and not props_visible:
        blockers.append(
            "Events returned no player-prop-shaped oddIDs in the tiny sample; "
            "may be off-season timing — re-probe near game days before enabling collection."
        )
    summary["integration_can_proceed"] = key_works and events_ok and props_visible
    summary["blockers"] = blockers
    summary["steps"] = steps

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    MD_PATH.write_text(render_md(summary), encoding="utf-8")
    print(f"\nProbe complete: key_works={key_works} props_visible={props_visible} "
          f"requests_made={client.requests_made}")
    print(f"  Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Report:  {MD_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def render_md(summary: dict) -> str:
    lines = ["# SportsGameOdds Probe", ""]
    lines.append(f"_Generated {summary.get('generated_at_utc')}. Research-only; no betting enabled._")
    lines.append("")
    lines.append(f"- Key detected: **{summary.get('key_detected')}** (env `{summary.get('key_env_var')}`)")
    lines.append(f"- Key works: **{summary.get('key_works')}**")
    lines.append(f"- Integration can proceed: **{summary.get('integration_can_proceed')}**")
    lines.append(f"- Requests made by this probe: {summary.get('requests_made', 0)}")
    usage = summary.get("usage")
    if usage:
        lines.append("")
        lines.append("## Usage / quota")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(usage, indent=2, default=str)[:2000])
        lines.append("```")
    headers = summary.get("last_rate_limit_headers") or {}
    if headers:
        lines.append(f"- Rate-limit headers: `{headers}`")
    if summary.get("sports_available") is not None:
        lines.append("")
        lines.append("## Coverage")
        lines.append("")
        lines.append(f"- Sports ({summary.get('n_sports', 0)}): {', '.join(summary.get('sports_available') or []) or 'none returned'}")
        lines.append(f"- Leagues ({summary.get('n_leagues', 0)}): {', '.join((summary.get('leagues_available') or [])[:40]) or 'none returned'}")
        target = summary.get("target_league_availability") or {}
        if target:
            lines.append("")
            lines.append("| league | available |")
            lines.append("| --- | --- |")
            for league, ok in target.items():
                lines.append(f"| {league} | {'yes' if ok else 'no'} |")
    events_probe = summary.get("events_probe") or {}
    if events_probe and not events_probe.get("skipped"):
        lines.append("")
        lines.append("## NBA events probe")
        lines.append("")
        lines.append(f"- Events returned: {events_probe.get('events_returned', 0)}")
        lines.append(f"- Odds entries sampled: {events_probe.get('odds_entries_sampled', 0)}")
        lines.append(f"- Player-prop-shaped odds: {events_probe.get('player_prop_odds_seen', 0)}")
        lines.append(f"- Game-level odds: {events_probe.get('game_odds_seen', 0)}")
        lines.append(f"- Players map present: {events_probe.get('has_players_map')}")
        lines.append(f"- Results present: {events_probe.get('has_results')}")
        if events_probe.get("sample_player_prop_oddids"):
            lines.append(f"- Sample player-prop oddIDs: `{events_probe['sample_player_prop_oddids']}`")
        if events_probe.get("event_fields"):
            lines.append(f"- Event fields: `{events_probe['event_fields']}`")
        if events_probe.get("odds_fields"):
            lines.append(f"- Odds-object fields: `{events_probe['odds_fields']}`")
        if events_probe.get("stat_ids_seen"):
            lines.append(f"- statIDs seen: `{events_probe['stat_ids_seen']}`")
    blockers = summary.get("blockers") or []
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("_Research-only. No approved bets, no approved parlays, no live recommendations._")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
