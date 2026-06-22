"""Probe all data sources safely and write a shared source-state snapshot.

Default is --dry-run: only checks API-key presence from the environment (keys are
never printed) and writes data/reports/source_state.json with quota unknown. No
network calls.

--real performs CHEAP/FREE probes only:
  * odds_api      : GET /sports (0 credits) -> reads remaining-credits header
  * sportsgameodds: GET /account/usage (costs requests, not entities)
  * apisports     : GET /status + ONE fixtures probe (league=1, season=2026) to
                    detect a free-plan season block for the World Cup
  * kalshi        : small public /markets read (no key needed)

The state file feeds the router and the source-health report. Enables no betting,
parlays, predictions, or recommendations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.source_router import load_router_config  # noqa: E402

STATE_PATH = PROJECT_ROOT / "data" / "reports" / "source_state.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "source_priority.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_number(obj: Any, *keywords: str) -> float | None:
    """Recursively find the first numeric value whose key contains all keywords."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if all(w in lk for w in keywords) and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            found = _find_number(v, *keywords)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_number(v, *keywords)
            if found is not None:
                return found
    return None


def probe_odds_api(env: dict[str, str], real: bool) -> dict[str, Any]:
    from data.prop_collection import make_default_fetch_json, _odds_api_url
    key = env.get("ODDS_API_KEY", "")
    st: dict[str, Any] = {"name": "odds_api", "key_present": bool(key), "quota_remaining": None,
                          "last_success_utc": None, "last_failure_utc": None, "blocked_reason": None,
                          "used_for": "World Cup odds, NBA player props/closing (paid)"}
    if not real or not key:
        return st
    try:
        fetch = make_default_fetch_json()
        url = _odds_api_url("https://api.the-odds-api.com/v4", "/sports", {"apiKey": key})
        fetch(url)  # /sports is free (0 credits)
        st["quota_remaining"] = getattr(fetch, "quota_remaining", None)
        st["last_success_utc"] = _now()
    except Exception as exc:  # noqa: BLE001
        st["last_failure_utc"] = _now()
        st["blocked_reason"] = f"probe_error:{type(exc).__name__}"
    return st


def probe_sportsgameodds(env: dict[str, str], real: bool) -> dict[str, Any]:
    key = env.get("SPORTSGAMEODDS_API_KEY", "")
    st: dict[str, Any] = {"name": "sportsgameodds", "key_present": bool(key), "quota_remaining": None,
                          "last_success_utc": None, "last_failure_utc": None, "blocked_reason": None,
                          "used_for": "NBA player props (1 entity/event), closing lines"}
    if not real or not key:
        return st
    try:
        from data.sportsgameodds_client import SportsGameOddsClient
        usage = SportsGameOddsClient(env=env).account_usage()
        # SGO /account/usage is double-wrapped: data.data.rateLimits.per-month.{max-entities,current-entities}
        inner = ((usage.get("data") or {}).get("data") or {}) if isinstance(usage, dict) else {}
        rate = inner.get("rateLimits") or (usage.get("rateLimits") if isinstance(usage, dict) else {}) or {}
        month = rate.get("per-month") or {}
        try:
            mx = float(month.get("max-entities"))
            cur = float(month.get("current-entities") or 0)
            st["quota_remaining"] = mx - cur
        except (TypeError, ValueError):
            st["quota_remaining"] = None
        st["last_success_utc"] = _now()
    except Exception as exc:  # noqa: BLE001
        st["last_failure_utc"] = _now()
        st["blocked_reason"] = f"probe_error:{type(exc).__name__}"
    return st


def probe_apisports(env: dict[str, str], real: bool) -> dict[str, Any]:
    key = env.get("APISPORTS_API_KEY", "")
    st: dict[str, Any] = {"name": "apisports", "key_present": bool(key), "quota_remaining": None,
                          "last_success_utc": None, "last_failure_utc": None, "blocked_reason": None,
                          "used_for": "World Cup schedule/results, player stats (no sportsbook odds)"}
    if not real or not key:
        return st
    try:
        from data.apisports_client import ApiSportsClient, extract_response
        client = ApiSportsClient(env=env)
        status = client.status("football")
        if status.get("ok"):
            st["last_success_utc"] = _now()
            st["quota_remaining"] = _find_number(status.get("data"), "limit", "day")
        # World Cup season-block probe: league=1 season=2026 fixtures.
        fixtures = client.request("football", "fixtures", {"league": 1, "season": 2026})
        errors = (fixtures.get("data") or {}).get("errors") if isinstance(fixtures.get("data"), dict) else None
        resp = extract_response(fixtures.get("data"))
        if errors and ("plan" in str(errors).lower() or "season" in str(errors).lower()):
            st["blocked_reason"] = f"plan_blocked_world_cup_2026:{errors}"
        elif not resp and not status.get("ok"):
            st["blocked_reason"] = "apisports_unreachable_or_unauthorized"
        st["wc_fixtures_2026_returned"] = len(resp)
    except Exception as exc:  # noqa: BLE001
        st["last_failure_utc"] = _now()
        st["blocked_reason"] = f"probe_error:{type(exc).__name__}"
    return st


def probe_kalshi(env: dict[str, str], real: bool) -> dict[str, Any]:
    st: dict[str, Any] = {"name": "kalshi", "key_present": bool(env.get("KALSHI_API_KEY")),
                          "quota_remaining": None, "last_success_utc": None, "last_failure_utc": None,
                          "blocked_reason": None, "used_for": "prediction-market prices (separate; never sportsbook odds)"}
    if not real:
        return st
    try:
        from data.kalshi_client import KalshiAPIClient
        client = KalshiAPIClient.from_env() if hasattr(KalshiAPIClient, "from_env") else KalshiAPIClient()
        markets = client.get_markets({"limit": 1, "status": "open"})
        st["last_success_utc"] = _now()
        st["note"] = f"public markets reachable (sample rows={len(markets)})"
    except Exception as exc:  # noqa: BLE001
        st["last_failure_utc"] = _now()
        st["blocked_reason"] = f"public_read_failed:{type(exc).__name__}"
    return st


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe all data sources (safe).")
    p.add_argument("--real", action="store_true", help="Do cheap/free network probes (default: dry-run).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    setup_logging(args.log_level)
    env = dict(os.environ)
    real = bool(args.real)

    sources = {
        "odds_api": probe_odds_api(env, real),
        "sportsgameodds": probe_sportsgameodds(env, real),
        "apisports": probe_apisports(env, real),
        "kalshi": probe_kalshi(env, real),
    }
    payload = {
        "report": "source_state", "generated_at_utc": _now(), "dry_run": not real,
        "research_only": True, "approved": False, "sources": sources,
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Source probe ({'REAL' if real else 'dry-run'}) -> {STATE_PATH.relative_to(PROJECT_ROOT)}")
    for name, s in sources.items():
        print(f"  {name:15s} key={s['key_present']!s:5s} quota={s['quota_remaining']} "
              f"blocked={s['blocked_reason']}")
    print("Research-only: keys never printed; no betting/parlays/predictions/recommendations.")


if __name__ == "__main__":
    main()
