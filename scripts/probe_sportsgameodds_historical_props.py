"""SportsGameOdds HISTORICAL player-prop probe (research-only, quota-safe).

Question: can SportsGameOdds serve older-season / historical NBA player prop
lines, open/close prices, and settled outcomes — on the current (Amateur) plan?

Method (cheapest possible, no assumptions):
  1. /account/usage before  -> baseline per-month entity count.
  2. A short ladder of tiny /events queries (limit=1 each) at windows that are
     guaranteed to contain completed NBA games (June anchors = NBA Finals):
     this week, earlier this season, ~1y, ~2y, ~4y, ~10y back. Each request
     asks for finalized events with includeOpenCloseOdds=true; if the API
     rejects a param with HTTP 400 the probe drops it and remembers.
  3. /account/usage after   -> exact entity cost of the whole probe.

Every raw response is archived under data/raw/sportsgameodds/historical_probe/
(append-only, collision-safe names — never overwrites existing snapshots).

Outputs:
    data/reports/sportsgameodds_historical_prop_probe_summary.json
    data/reports/sportsgameodds_historical_prop_probe.md

Research-only: no models, no recommendations, no bets, no gate changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sportsgameodds" / "historical_probe"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SUMMARY_PATH = REPORTS_DIR / "sportsgameodds_historical_prop_probe_summary.json"
MD_PATH = REPORTS_DIR / "sportsgameodds_historical_prop_probe.md"

# oddID side tokens that indicate team/game odds rather than player props.
_TEAM_ENTITY_TOKENS = {"home", "away", "all", "side1", "side2"}

# Optional /events params we try first and drop individually on HTTP 400.
_OPTIONAL_PARAMS = ("finalized", "includeOpenCloseOdds")

# Keys on odds / per-bookmaker objects that would indicate settlement info.
_SETTLEMENT_KEY_TOKENS = ("score", "result", "settled", "won", "outcome", "grade")

# Per-minute request cap on the Amateur tier is 10; pace below it.
DEFAULT_PAUSE_SECONDS = 6.5


def _utc_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_windows(now: datetime) -> list[dict]:
    """Ladder of small past windows, newest first.

    June anchors are intentional: every offset lands in (or just after) an NBA
    Finals window, so a 7-day lookback nearly always contains completed games.
    """

    def window(name: str, anchor: datetime, days: int = 7) -> dict:
        return {
            "name": name,
            "starts_after": _iso_z(anchor - timedelta(days=days)),
            "starts_before": _iso_z(anchor),
            "anchor_date": _utc_date(anchor),
        }

    return [
        window("this_week_completed", now, days=4),
        window("earlier_this_season_~5mo", now - timedelta(days=150), days=4),
        window("one_year_back", now - timedelta(days=365)),
        window("two_years_back", now - timedelta(days=730)),
        window("four_years_back", now - timedelta(days=1461)),
        window("ten_years_back", now - timedelta(days=3653)),
    ]


def looks_like_player_entity(stat_entity: str) -> bool:
    token = str(stat_entity or "").strip()
    return bool(token) and token.lower() not in _TEAM_ENTITY_TOKENS


def is_player_prop_odd(odd_id: str, odd: dict | None) -> bool:
    """playerID is the strongest signal; fall back to oddID entity parsing."""

    if isinstance(odd, dict):
        if odd.get("playerID"):
            return True
        entity = odd.get("statEntityID")
        if entity:
            return looks_like_player_entity(str(entity))
    parts = str(odd_id).split("-")
    return looks_like_player_entity(parts[1] if len(parts) >= 5 else "")


def _trim_scalars(obj: dict, max_items: int = 40) -> dict:
    out = {}
    for key, value in obj.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        if len(out) >= max_items:
            break
    return out


def _sample_prop_odd(odd: dict, max_books: int = 2) -> dict:
    """Compact evidence sample of one player-prop odds object."""

    sample = _trim_scalars(odd)
    by_book = odd.get("byBookmaker")
    if isinstance(by_book, dict) and by_book:
        sample["byBookmaker"] = {
            book: _trim_scalars(data) if isinstance(data, dict) else data
            for book, data in list(by_book.items())[:max_books]
        }
    return sample


def analyze_event(event: dict) -> dict:
    """Inspect one event for props, open/close prices, books, and settlement."""

    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    info_obj = event.get("info") if isinstance(event.get("info"), dict) else {}
    starts_at = status.get("startsAt") or event.get("startsAt") or info_obj.get("startsAt")
    teams = event.get("teams") if isinstance(event.get("teams"), dict) else {}

    def team_name(side: str) -> str | None:
        side_obj = teams.get(side)
        if isinstance(side_obj, dict):
            names = side_obj.get("names")
            if isinstance(names, dict):
                return names.get("short") or names.get("medium") or names.get("long")
            return side_obj.get("teamID")
        return None

    results = event.get("results")
    info: dict = {
        "event_id": event.get("eventID"),
        "league_id": event.get("leagueID"),
        "starts_at": starts_at,
        "game_date": str(starts_at)[:10] if starts_at else None,
        "home_team": team_name("home"),
        "away_team": team_name("away"),
        "status_flags": {
            k: v for k, v in status.items()
            if isinstance(v, (bool, str, int, float)) and k != "displayLong"
        },
        "has_results": bool(results),
        "results_keys_sample": sorted(results.keys())[:10] if isinstance(results, dict) else [],
        "n_odds": 0,
        "n_player_prop_odds": 0,
        "n_game_odds": 0,
        "player_props_with_open_price": 0,
        "player_props_with_close_field": 0,
        "player_props_with_final_book_price": 0,
        "player_props_with_score": 0,
        "odds_field_names": [],
        "open_field_names": [],
        "close_field_names": [],
        "settlement_field_names": [],
        "timestamp_field_names": [],
        "bookmakers_seen": [],
        "per_book_field_names": [],
        "per_book_open_fields": [],
        "per_book_close_fields": [],
        "per_book_timestamp_fields": [],
        "n_props_with_bybookmaker": 0,
        "sample_player_prop": None,
    }

    odds = event.get("odds")
    if not isinstance(odds, dict):
        return info

    odds_fields: set[str] = set()
    books: set[str] = set()
    per_book_fields: set[str] = set()
    for odd_id, odd in odds.items():
        info["n_odds"] += 1
        if not isinstance(odd, dict):
            continue
        odds_fields |= set(odd.keys())
        is_prop = is_player_prop_odd(odd_id, odd)
        if not is_prop:
            info["n_game_odds"] += 1
            continue
        info["n_player_prop_odds"] += 1
        if info["sample_player_prop"] is None:
            info["sample_player_prop"] = _sample_prop_odd(odd)
        if any(odd.get(f) not in (None, "") for f in
               ("openBookOdds", "openBookOverUnder", "openBookSpread", "openFairOdds")):
            info["player_props_with_open_price"] += 1
        if any(str(k).lower().startswith("close") and odd.get(k) not in (None, "")
               for k in odd):
            info["player_props_with_close_field"] += 1
        if any(odd.get(f) not in (None, "") for f in ("bookOdds", "bookOverUnder")):
            info["player_props_with_final_book_price"] += 1
        if odd.get("score") not in (None, ""):
            info["player_props_with_score"] += 1
        by_book = odd.get("byBookmaker")
        if isinstance(by_book, dict) and by_book:
            info["n_props_with_bybookmaker"] += 1
            for book, data in by_book.items():
                books.add(str(book))
                if isinstance(data, dict):
                    per_book_fields |= set(data.keys())

    def matching(fields: set[str], predicate) -> list[str]:
        return sorted(f for f in fields if predicate(str(f).lower()))

    info["odds_field_names"] = sorted(odds_fields)
    info["open_field_names"] = matching(odds_fields, lambda f: f.startswith("open"))
    info["close_field_names"] = matching(odds_fields, lambda f: f.startswith("close"))
    info["settlement_field_names"] = matching(
        odds_fields, lambda f: any(tok in f for tok in _SETTLEMENT_KEY_TOKENS)
    )
    info["timestamp_field_names"] = matching(
        odds_fields, lambda f: "updated" in f or f.endswith("at") or "time" in f
    )
    info["bookmakers_seen"] = sorted(books)[:15]
    info["per_book_field_names"] = sorted(per_book_fields)
    info["per_book_open_fields"] = matching(per_book_fields, lambda f: f.startswith("open"))
    info["per_book_close_fields"] = matching(per_book_fields, lambda f: f.startswith("close"))
    info["per_book_timestamp_fields"] = matching(
        per_book_fields, lambda f: "updated" in f or f.endswith("at") or "time" in f
    )
    return info


def build_verdict(window_results: list[dict], tier: str | None,
                  entity_cost: int | None) -> dict:
    """Aggregate the per-window analyses into the report's hard answers."""

    successful = [w for w in window_results if w.get("ok") and w.get("events_returned", 0) > 0]
    with_props = [w for w in successful if (w.get("event") or {}).get("n_player_prop_odds", 0) > 0]
    # Only windows older than the current week count as "historical".
    historical = [w for w in successful if not w.get("is_current_week")]
    historical_props = [w for w in with_props if not w.get("is_current_week")]

    def any_event(windows: list[dict], key: str) -> bool:
        return any((w.get("event") or {}).get(key, 0) > 0 for w in windows)

    open_available = any_event(with_props, "player_props_with_open_price")
    explicit_close = any_event(with_props, "player_props_with_close_field")
    final_book_price = any_event(with_props, "player_props_with_final_book_price")
    settlement = (
        any((w.get("event") or {}).get("has_results") for w in successful)
        or any_event(successful, "player_props_with_score")
    )

    blocked_statuses = {401, 402, 403}
    tier_block_tokens = ("tier", "plan", "upgrade", "subscription", "permission")
    free_tier_blocked = any(
        (w.get("status") in blocked_statuses)
        or any(tok in str(w.get("error") or "").lower() for tok in tier_block_tokens)
        for w in window_results
    )

    oldest = None
    for w in historical_props or historical:
        date = (w.get("event") or {}).get("game_date")
        if date and (oldest is None or date < oldest):
            oldest = date

    def oldest_date(windows: list[dict], key: str) -> str | None:
        dates = [
            (w.get("event") or {}).get("game_date")
            for w in windows
            if (w.get("event") or {}).get(key, 0) > 0 and (w.get("event") or {}).get("game_date")
        ]
        return min(dates) if dates else None

    verdict = {
        "historical_events_accessible": bool(historical),
        "historical_player_props_accessible": bool(historical_props),
        "open_close_prices_available": open_available and (explicit_close or final_book_price),
        "closing_prices_available_for_props": explicit_close or final_book_price,
        "closing_price_form": (
            "explicit close* fields" if explicit_close
            else "final bookOdds/bookOverUnder on finalized events" if final_book_price
            else None
        ),
        "settlement_results_available": settlement,
        "free_tier_blocked": free_tier_blocked,
        "plan_tier": tier,
        "oldest_successful_game_date": oldest,
        # Fidelity horizons: how far back each data tier was actually observed.
        "oldest_event_date": min(
            (d for d in ((w.get("event") or {}).get("game_date") for w in successful) if d),
            default=None,
        ),
        "oldest_props_date": oldest_date(successful, "n_player_prop_odds"),
        "oldest_close_field_date": oldest_date(successful, "player_props_with_close_field"),
        "oldest_open_price_date": oldest_date(successful, "player_props_with_open_price"),
        "entity_cost_estimate": entity_cost,
    }
    if historical_props:
        verdict["recommended_next_step"] = (
            "Historical player props ARE accessible. Next: design a quota-guarded "
            "backfill (NBA only, one season, small date ranges) — see the backfill "
            "plan section. Do NOT bulk-import yet."
        )
    elif historical and not historical_props:
        verdict["recommended_next_step"] = (
            "Historical events load but contain no player-prop odds in the sample. "
            "Keep SportsGameOdds for current/future collection; do not plan a prop "
            "backfill from this source."
        )
    elif free_tier_blocked:
        verdict["recommended_next_step"] = (
            "Historical access appears blocked at the current plan tier. Keep "
            "SportsGameOdds for current/future collection only."
        )
    else:
        verdict["recommended_next_step"] = (
            "No historical events were returned (empty/failed responses). Keep "
            "SportsGameOdds for current/future collection; re-probe before any "
            "backfill plans."
        )
    return verdict


def render_md(summary: dict) -> str:
    verdict = summary.get("verdict") or {}
    lines = ["# SportsGameOdds Historical Player Prop Probe", ""]
    lines.append(
        f"_Generated {summary.get('generated_at_utc')}. Research-only; no recommendations, "
        "no predictions, no gate changes._"
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for key in (
        "historical_events_accessible",
        "historical_player_props_accessible",
        "open_close_prices_available",
        "closing_prices_available_for_props",
        "settlement_results_available",
        "free_tier_blocked",
    ):
        lines.append(f"- {key}: **{bool(verdict.get(key))}**")
    lines.append(f"- closing_price_form: {verdict.get('closing_price_form')}")
    lines.append(f"- plan_tier: {verdict.get('plan_tier')}")
    lines.append(f"- oldest_successful_game_date: **{verdict.get('oldest_successful_game_date')}**")
    lines.append("- fidelity horizons (oldest date each tier was observed): "
                 f"events `{verdict.get('oldest_event_date')}`, "
                 f"player props `{verdict.get('oldest_props_date')}`, "
                 f"close* fields `{verdict.get('oldest_close_field_date')}`, "
                 f"open prices `{verdict.get('oldest_open_price_date')}`")
    lines.append(f"- entity_cost_estimate: **{verdict.get('entity_cost_estimate')}** "
                 f"entities for the whole probe ({summary.get('requests_made', 0)} requests)")
    lines.append(f"- recommended next step: {verdict.get('recommended_next_step')}")
    if summary.get("rejected_params"):
        lines.append(f"- params rejected by the API (HTTP 400): `{summary['rejected_params']}`")
    lines.append("")

    lines.append("## Window ladder")
    lines.append("")
    lines.append("| window | range (UTC) | ok | events | game date | matchup | props | open | close-field | final book price | results |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for w in summary.get("windows") or []:
        ev = w.get("event") or {}
        matchup = (
            f"{ev.get('away_team')}@{ev.get('home_team')}"
            if ev.get("home_team") or ev.get("away_team") else "-"
        )
        ok_cell = "yes" if w.get("ok") else f"NO ({w.get('error')})"
        lines.append(
            f"| {w.get('name')} | {str(w.get('starts_after'))[:10]} → {str(w.get('starts_before'))[:10]} "
            f"| {ok_cell} "
            f"| {w.get('events_returned', 0)} | {ev.get('game_date') or '-'} | {matchup} "
            f"| {ev.get('n_player_prop_odds', 0)} | {ev.get('player_props_with_open_price', 0)} "
            f"| {ev.get('player_props_with_close_field', 0)} | {ev.get('player_props_with_final_book_price', 0)} "
            f"| {'yes' if ev.get('has_results') else 'no'} |"
        )
    lines.append("")

    for title, detail in (
        ("## Field evidence — oldest event with props", summary.get("field_evidence")),
        ("## Field evidence — newest finalized event (full fidelity)",
         summary.get("field_evidence_newest")),
    ):
        if detail:
            lines.append(title)
            lines.append("")
            for label, value in detail.items():
                lines.append(f"- {label}: `{value}`")
            lines.append("")

    sample = summary.get("sample_player_prop")
    if sample:
        lines.append("## Sample historical player-prop odds object (trimmed)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(sample, indent=2, default=str)[:3000])
        lines.append("```")
        lines.append("")

    if verdict.get("historical_player_props_accessible"):
        lines.append("## Careful backfill plan (PROPOSAL ONLY — not implemented, not approved)")
        lines.append("")
        lines.append("1. **One league first:** NBA only.")
        lines.append("2. **One season first:** 2025-26 (newest complete coverage; verify density before older seasons).")
        lines.append("3. **Small date ranges:** 3–7 day windows, `limit<=10` per page, one window per run.")
        lines.append("4. **Quota guard:** read /account/usage first; abort if projected cost would push "
                      "per-month entities above a hard reserve (e.g. keep >=500 entities free for daily collection). "
                      "~1 entity per event means a 1,230-game season costs roughly half the 2,500/month Amateur quota — "
                      "spread across months or upgrade before full-season pulls.")
        lines.append("5. **Raw archive:** append-only JSON under `data/raw/sportsgameodds/historical/NBA/` "
                      "(same never-overwrite naming as current collection).")
        lines.append("6. **Normalized table:** separate historical prop snapshot table "
                      "(`data/props/sportsgameodds_historical_props.csv`), never merged into live snapshot files.")
        lines.append("7. **Source labels:** `source=sportsgameodds_historical` + `snapshot_type=closing_historical` "
                      "so CLV/settlement research can never confuse backfill with live-collected closing lines.")
        lines.append("8. **Dashboard report:** add a historical-coverage section (windows pulled, events, props, quota used).")
        lines.append("")
        lines.append("Proof gates unchanged: backfilled odds are research data, not signals; "
                      "baseline/signals/parlays remain blocked on live closing/CLV evidence.")
    else:
        lines.append("## Blocker")
        lines.append("")
        lines.append(f"- {verdict.get('recommended_next_step')}")
    lines.append("")

    lines.append("## Probe steps")
    lines.append("")
    for step in summary.get("steps") or []:
        flag = "OK" if step.get("ok") else f"FAIL ({step.get('error')})"
        lines.append(f"- `{step.get('step')}` — {flag} (raw: `{step.get('raw_file')}`)")
    lines.append("")
    lines.append("_Research-only. No approved bets, no approved parlays, no live recommendations._")
    lines.append("")
    return "\n".join(lines)


def _step(name: str, result: dict, steps: list[dict]) -> dict:
    raw_file = None
    if result.get("data") is not None:
        raw_path = save_raw_payload(result["data"], RAW_DIR, name)
        try:
            raw_file = str(raw_path.relative_to(PROJECT_ROOT))
        except ValueError:  # RAW_DIR redirected outside the project (tests)
            raw_file = str(raw_path)
    record = {
        "step": name,
        "ok": bool(result.get("ok")),
        "status": result.get("status"),
        "error": result.get("error"),
        "attempts": result.get("attempts"),
        "raw_file": raw_file,
    }
    steps.append(record)
    flag = "OK" if record["ok"] else f"FAIL ({record['error']})"
    print(f"  [{name}] {flag}")
    return record


def _month_entities(usage: dict | None) -> int | None:
    try:
        return int(usage["rateLimits"]["per-month"]["current-entities"])
    except (KeyError, TypeError, ValueError):
        return None


def run_probe(client: SportsGameOddsClient, *, now: datetime, pause: float,
              max_windows: int, limit: int, sleep=time.sleep) -> dict:
    """Execute the ladder; returns the full summary dict (no files written)."""

    steps: list[dict] = []
    summary: dict = {
        "report": "sportsgameodds_historical_prop_probe_summary",
        "generated_at_utc": now.isoformat(),
        "key_detected": client.has_key,
        "key_env_var": API_KEY_ENV,
        "research_only": True,
        "approved": False,
    }

    usage_before = client.account_usage()
    _step("usage_before", usage_before, steps)
    usage_items, _ = extract_items(usage_before.get("data"))
    usage_obj = usage_items[0] if usage_items and isinstance(usage_items[0], dict) else None
    tier = (usage_obj or {}).get("tier")
    entities_before = _month_entities(usage_obj)
    summary["tier"] = tier
    summary["entities_before"] = entities_before

    rejected_params: list[str] = []
    window_results: list[dict] = []
    windows = build_windows(now)[: max(1, max_windows)]
    for index, window in enumerate(windows):
        sleep(pause)
        params = {
            "leagueID": "NBA",
            "startsAfter": window["starts_after"],
            "startsBefore": window["starts_before"],
            "limit": max(1, min(limit, 2)),
        }
        for optional in _OPTIONAL_PARAMS:
            if optional not in rejected_params:
                params[optional] = "true"

        result = client.events(**params)
        # HTTP 400 -> drop optional params one at a time and retry (once each).
        while (not result.get("ok") and result.get("status") == 400
               and any(p in params for p in _OPTIONAL_PARAMS)):
            for optional in _OPTIONAL_PARAMS:
                if optional in params:
                    params.pop(optional)
                    rejected_params.append(optional)
                    break
            sleep(pause)
            result = client.events(**params)

        record = _step(f"events_{window['name']}", result, steps)
        items, _ = extract_items(result.get("data"))
        events = [e for e in items if isinstance(e, dict)]
        window_entry = {
            **window,
            "ok": record["ok"],
            "status": record["status"],
            "error": record["error"],
            "events_returned": len(events),
            "is_current_week": index == 0,
            "event": analyze_event(events[0]) if events else None,
        }
        window_results.append(window_entry)

    sleep(pause)
    usage_after = client.account_usage()
    _step("usage_after", usage_after, steps)
    after_items, _ = extract_items(usage_after.get("data"))
    after_obj = after_items[0] if after_items and isinstance(after_items[0], dict) else None
    entities_after = _month_entities(after_obj)
    summary["entities_after"] = entities_after

    entity_cost = None
    if entities_before is not None and entities_after is not None:
        entity_cost = max(0, entities_after - entities_before)
    summary["entity_cost"] = entity_cost
    summary["requests_made"] = client.requests_made
    summary["rejected_params"] = rejected_params
    summary["windows"] = window_results
    summary["steps"] = steps
    summary["verdict"] = build_verdict(window_results, tier, entity_cost)

    # Evidence from the oldest historical event that had player props.
    evidence_windows = [
        w for w in window_results
        if (w.get("event") or {}).get("n_player_prop_odds", 0) > 0
    ]
    def _evidence(event: dict) -> dict:
        return {
            "game_date": event.get("game_date"),
            "open_fields_on_odds": event.get("open_field_names"),
            "close_fields_on_odds": event.get("close_field_names"),
            "settlement_fields_on_odds": event.get("settlement_field_names"),
            "timestamp_fields_on_odds": event.get("timestamp_field_names"),
            "bookmakers_seen": event.get("bookmakers_seen"),
            "per_book_fields": event.get("per_book_field_names"),
            "per_book_open_fields": event.get("per_book_open_fields"),
            "per_book_close_fields": event.get("per_book_close_fields"),
            "per_book_timestamp_fields": event.get("per_book_timestamp_fields"),
            "results_keys_sample": event.get("results_keys_sample"),
        }

    if evidence_windows:
        oldest_event = evidence_windows[-1]["event"]
        summary["field_evidence"] = _evidence(oldest_event)
        if len(evidence_windows) > 1:
            summary["field_evidence_newest"] = _evidence(evidence_windows[0]["event"])
        summary["sample_player_prop"] = oldest_event.get("sample_player_prop")

    # Flatten the verdict booleans to the top level for easy machine reads.
    summary.update({k: v for k, v in summary["verdict"].items()})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe SportsGameOdds for HISTORICAL NBA player props (tiny, quota-safe)."
    )
    parser.add_argument("--max-windows", type=int, default=6,
                        help="How many past windows to test, newest first (default 6).")
    parser.add_argument("--limit", type=int, default=1,
                        help="Events per request (default 1, hard cap 2).")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SECONDS,
                        help="Seconds between requests (per-minute limit is 10).")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    client = SportsGameOddsClient()

    if not client.has_key:
        summary = {
            "report": "sportsgameodds_historical_prop_probe_summary",
            "generated_at_utc": now.isoformat(),
            "key_detected": False,
            "key_env_var": API_KEY_ENV,
            "research_only": True,
            "approved": False,
            "requests_made": 0,
            "windows": [],
            "steps": [],
            "verdict": build_verdict([], None, None),
        }
        summary["verdict"]["recommended_next_step"] = (
            f"{API_KEY_ENV} is not set; probe skipped gracefully (no requests made)."
        )
        summary.update(summary["verdict"])
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        MD_PATH.write_text(render_md(summary), encoding="utf-8")
        print(f"No {API_KEY_ENV} detected; wrote skip report.")
        return 0

    print("Probing SportsGameOdds historical NBA player props (tiny, quota-safe)...")
    summary = run_probe(
        client,
        now=now,
        pause=max(0.0, args.pause),
        max_windows=args.max_windows,
        limit=args.limit,
    )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    MD_PATH.write_text(render_md(summary), encoding="utf-8")
    verdict = summary["verdict"]
    print(
        f"\nProbe complete: historical_events={verdict['historical_events_accessible']} "
        f"historical_props={verdict['historical_player_props_accessible']} "
        f"open_close={verdict['open_close_prices_available']} "
        f"entity_cost={verdict['entity_cost_estimate']} "
        f"requests={summary.get('requests_made')}"
    )
    print(f"  Summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Report:  {MD_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
