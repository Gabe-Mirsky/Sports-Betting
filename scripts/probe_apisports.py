"""Small, safe API-Sports probe (research-only; probe-only source).

Checks whether the API-Sports key works, which sport APIs the plan covers,
whether odds endpoints exist, and — the key question — whether PLAYER PROPS
are available anywhere. Makes <= 8 cheap requests, saves raw responses under
data/raw/apisports/probe/, and writes:

    data/reports/apisports_probe_summary.json
    data/reports/apisports_probe.md

No collector is built unless player props are clearly proven. Research-only.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.apisports_client import (  # noqa: E402
    API_KEY_ENV,
    ApiSportsClient,
    extract_response,
)
from data.sportsgameodds_client import save_raw_payload  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "apisports" / "probe"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SUMMARY_PATH = REPORTS_DIR / "apisports_probe_summary.json"
MD_PATH = REPORTS_DIR / "apisports_probe.md"

PLAYER_TOKENS = ("player", "anytime", "scorer", "to score", "assists", "rebounds",
                 "strikeout", "passing", "rushing", "receiving", "shots on")


def _record(name: str, result: dict, steps: list[dict]) -> dict:
    raw_path = None
    if result.get("data") is not None:
        raw_path = save_raw_payload(result["data"], RAW_DIR, name)
    record = {
        "step": name,
        "ok": bool(result.get("ok")),
        "status": result.get("status"),
        "error": result.get("error"),
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)) if raw_path else None,
    }
    steps.append(record)
    print(f"  [{name}] {'OK' if record['ok'] else 'FAIL (' + str(record['error'])[:120] + ')'}")
    return record


def _candidate_nba_dates(max_dates: int = 3) -> list[str]:
    """Dates likely to have an NBA game: next future game_date from our own
    snapshots first, then today/tomorrow (US/Eastern) as fallbacks."""

    candidates: list[str] = []
    snapshots = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"
    try:
        import pandas as pd

        frame = pd.read_csv(snapshots, usecols=["league", "game_date"], low_memory=False)
        today = pd.Timestamp.now(tz="America/New_York").date().isoformat()
        dates = (
            frame.loc[frame["league"].astype(str).eq("NBA"), "game_date"]
            .astype(str).str.slice(0, 10)
        )
        future = sorted({d for d in dates if d >= today})
        candidates.extend(future[:2])
    except Exception:
        pass
    from datetime import date, timedelta as _td

    today_local = date.today()
    for offset in (0, 1):
        value = (today_local + _td(days=offset)).isoformat()
        if value not in candidates:
            candidates.append(value)
    return candidates[:max_dates]


def _player_prop_bets(bets: list) -> list[str]:
    """Bet-type names that look like player props."""

    hits: list[str] = []
    for bet in bets:
        name = str((bet or {}).get("name", "")).strip()
        if name and any(token in name.lower() for token in PLAYER_TOKENS):
            hits.append(name)
    return hits


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe the API-Sports APIs (small, quota-safe).")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=0,
        help="Skip the probe when the existing summary is fresher than this many hours "
             "(0 = always probe). Lets the scheduled pipeline re-probe at most daily.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    if args.max_age_hours > 0 and SUMMARY_PATH.exists():
        try:
            existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            generated = datetime.fromisoformat(str(existing.get("generated_at_utc")))
            age_hours = (now - generated).total_seconds() / 3600.0
            if age_hours < args.max_age_hours:
                print(f"API-Sports probe summary is {age_hours:.1f}h old "
                      f"(< {args.max_age_hours:.0f}h); skipping re-probe.")
                return 0
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    client = ApiSportsClient()
    steps: list[dict] = []

    summary: dict = {
        "report": "apisports_probe_summary",
        "generated_at_utc": now.isoformat(),
        "key_detected": client.has_key,
        "key_env_var": API_KEY_ENV,
        "auth_header": "x-apisports-key",
        "research_only": True,
        "approved": False,
    }

    if not client.has_key:
        summary.update({
            "key_works": False,
            "player_props_available": "unknown",
            "useful_for_project": False,
            "recommended_next_action": f"Set {API_KEY_ENV}; probe skipped gracefully (no requests made).",
            "steps": [],
        })
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        MD_PATH.write_text(render_md(summary), encoding="utf-8")
        print(f"No {API_KEY_ENV} detected; wrote skip report.")
        return 0

    print("Probing API-Sports (small, quota-safe)...")

    # 1-2. /status on the two basketball-relevant APIs (plan + daily usage).
    statuses: dict[str, dict] = {}
    for api in ("basketball", "nba"):
        result = client.status(api)
        _record(f"status_{api}", result, steps)
        payload = result.get("data") or {}
        response = payload.get("response") if isinstance(payload, dict) else {}
        if isinstance(response, dict) and response:
            statuses[api] = {
                "plan": ((response.get("subscription") or {}).get("plan")),
                "active": ((response.get("subscription") or {}).get("active")),
                "requests_today": ((response.get("requests") or {}).get("current")),
                "requests_limit_day": ((response.get("requests") or {}).get("limit_day")),
            }
        elif result.get("ok"):
            statuses[api] = {"raw": True}
    summary["status_by_api"] = statuses
    summary["key_works"] = any(s.get("active") for s in statuses.values()) or bool(statuses)

    # 3. Basketball odds bet types: do any look like player props?
    bets_result = client.request("basketball", "bets")
    _record("basketball_bets", bets_result, steps)
    basketball_bets = extract_response(bets_result.get("data"))
    basketball_player_bets = _player_prop_bets(basketball_bets)
    summary["basketball_bet_types"] = len(basketball_bets)
    summary["basketball_player_prop_bet_types"] = basketball_player_bets

    # 4. Soccer (football v3) bet types — the richest API-Sports odds catalog.
    soccer_bets_result = client.request("football", "odds/bets")
    _record("football_odds_bets", soccer_bets_result, steps)
    soccer_bets = extract_response(soccer_bets_result.get("data"))
    soccer_player_bets = _player_prop_bets(soccer_bets)
    summary["football_bet_types"] = len(soccer_bets)
    summary["football_player_prop_bet_types"] = soccer_player_bets

    # 5. NBA API odds? The v2 NBA API has games/stats; odds support is the
    # question. Hit a cheap odds-looking endpoint and accept failure as data.
    nba_odds_result = client.request("nba", "odds", {"league": "standard"})
    _record("nba_odds", nba_odds_result, steps)
    summary["nba_odds_endpoint_ok"] = bool(nba_odds_result.get("ok"))

    # 6-7. The decisive test: do actual NBA game odds include player-prop
    # bets, or is the catalog aspirational? Find one NBA game (league=12 on
    # v1.basketball), then pull its odds and scan the bet names.
    live_player_bets: list[str] = []
    game_odds_checked = False
    game_id = None
    plan_restriction = None
    for date in _candidate_nba_dates():
        games_result = client.request(
            "basketball", "games", {"league": 12, "season": "2025-2026", "date": date}
        )
        _record(f"basketball_games_{date}", games_result, steps)
        error_text = str(games_result.get("error") or "")
        if "plan" in error_text.lower():
            plan_restriction = error_text
            break  # same answer for every date; stop burning requests
        games = extract_response(games_result.get("data"))
        if games:
            game_id = (games[0] or {}).get("id")
            summary["nba_game_found"] = {"date": date, "game_id": game_id}
            break
    summary["plan_restriction"] = plan_restriction
    if game_id:
        odds_result = client.request("basketball", "odds", {"game": game_id})
        _record("basketball_game_odds", odds_result, steps)
        game_odds_checked = bool(odds_result.get("ok"))
        for entry in extract_response(odds_result.get("data")):
            for bookmaker in (entry or {}).get("bookmakers", []) or []:
                live_player_bets.extend(
                    _player_prop_bets((bookmaker or {}).get("bets", []) or [])
                )
        live_player_bets = sorted(set(live_player_bets))
    summary["live_game_odds_checked"] = game_odds_checked
    summary["live_player_prop_bets"] = live_player_bets

    player_prop_bet_names = basketball_player_bets + soccer_player_bets
    summary["odds_available"] = bool(
        (bets_result.get("ok") and basketball_bets) or (soccer_bets_result.get("ok") and soccer_bets)
    )
    # Proven only when a real game's odds carry player-prop bets. A catalog
    # listing alone is "catalog_only" — never treated as proof.
    if live_player_bets:
        summary["player_props_available"] = True
    elif plan_restriction:
        summary["player_props_available"] = "blocked_free_plan_no_current_season"
    elif player_prop_bet_names and not game_odds_checked:
        summary["player_props_available"] = "catalog_only_unproven"
    elif player_prop_bet_names and game_odds_checked:
        summary["player_props_available"] = "catalog_only_not_in_live_odds"
    else:
        summary["player_props_available"] = (
            "false_in_probe" if summary["odds_available"] else "unknown"
        )
    # Useful even without props: game odds, results, stats as enrichment backup.
    summary["useful_for_project"] = bool(summary.get("key_works")) and not plan_restriction
    if live_player_bets:
        action = ("Player props PROVEN in live game odds — a limited collector may be designed, "
                  "but check daily request budget (100/day free) and line/price field shapes first.")
    elif plan_restriction:
        action = ("BLOCKED for current-season collection: the Free plan only serves seasons "
                  "2022-2024, so current games/odds/props are inaccessible. Player-prop bet "
                  "types exist in the catalog, so a PAID plan could be re-probed later. "
                  "Keep as probe-only; do not build a collector.")
    elif player_prop_bet_names:
        action = ("Keep as probe/backup only: player-prop bet types exist in the odds catalog but "
                  "did not appear in live game odds during this probe. Re-probe near tip-off "
                  "before considering any collector.")
    elif summary["odds_available"]:
        action = ("Keep as probe/backup only: odds endpoints work but no player-prop bet types "
                  "were visible. Useful as a results/schedule/game-odds backup, not for props.")
    else:
        action = "Keep as probe-only; odds endpoints were not usable in this probe."
    summary["recommended_next_action"] = action
    summary["requests_made"] = client.requests_made
    summary["steps"] = steps

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    MD_PATH.write_text(render_md(summary), encoding="utf-8")
    print(f"\nProbe complete: key_works={summary.get('key_works')} "
          f"player_props={summary['player_props_available']} requests={client.requests_made}")
    print(f"  Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def render_md(summary: dict) -> str:
    lines = ["# API-Sports Probe", ""]
    lines.append(f"_Generated {summary.get('generated_at_utc')}. Research-only; probe-only source._")
    lines.append("")
    lines.append(f"- Key detected: **{summary.get('key_detected')}** (env `{summary.get('key_env_var')}`, header `{summary.get('auth_header')}`)")
    lines.append(f"- Key works: **{summary.get('key_works')}**")
    lines.append(f"- Odds available: **{summary.get('odds_available')}**")
    lines.append(f"- Player props available: **{summary.get('player_props_available')}**")
    lines.append(f"- Useful for project: **{summary.get('useful_for_project')}**")
    lines.append(f"- Requests made: {summary.get('requests_made', 0)}")
    lines.append("")
    statuses = summary.get("status_by_api") or {}
    if statuses:
        lines.append("## Plan / usage by API")
        lines.append("")
        lines.append("| api | plan | active | requests today | daily limit |")
        lines.append("| --- | --- | --- | --- | --- |")
        for api, info in statuses.items():
            lines.append(
                f"| {api} | {info.get('plan', 'n/a')} | {info.get('active', 'n/a')} "
                f"| {info.get('requests_today', 'n/a')} | {info.get('requests_limit_day', 'n/a')} |"
            )
        lines.append("")
    lines.append("## Player-prop bet types found")
    lines.append("")
    basketball = summary.get("basketball_player_prop_bet_types") or []
    soccer = summary.get("football_player_prop_bet_types") or []
    if basketball or soccer:
        for name in basketball:
            lines.append(f"- basketball: {name}")
        for name in soccer:
            lines.append(f"- football(soccer): {name}")
    else:
        lines.append(f"- None visible (basketball bet types: {summary.get('basketball_bet_types', 0)}, "
                     f"soccer bet types: {summary.get('football_bet_types', 0)}).")
    lines.append("")
    lines.append("## Live game odds check")
    lines.append("")
    if summary.get("plan_restriction"):
        lines.append(f"- **PLAN BLOCKED**: {summary['plan_restriction']}")
    game_found = summary.get("nba_game_found")
    lines.append(f"- NBA game found: {game_found or 'no'}")
    lines.append(f"- Live odds checked: {summary.get('live_game_odds_checked')}")
    live = summary.get("live_player_prop_bets") or []
    if live:
        lines.append(f"- Player-prop bets IN LIVE ODDS ({len(live)}): {', '.join(live[:20])}")
    else:
        lines.append("- No player-prop bets appeared in the live odds sample.")
    lines.append("")
    lines.append("## Recommended next action")
    lines.append("")
    lines.append(f"- {summary.get('recommended_next_action')}")
    lines.append("")
    lines.append("_Research-only. No collector will be built unless player props are clearly proven._")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
