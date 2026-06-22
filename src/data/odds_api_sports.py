"""The Odds API sport-key discovery (research-only).

Lists every sport The Odds API exposes (one cheap /v4/sports request), writes
``data/reports/odds_api_available_sports.json``, and cross-checks the sport
keys configured in ``config/prop_collection.yaml`` against the live list so a
typo'd or out-of-season sport key is visible before it wastes prop credits.
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .prop_collection import make_default_fetch_json


REPORT_FILENAME = "odds_api_available_sports.json"
REPORT_MD_FILENAME = "odds_api_available_sports.md"

# Odds API "group" names for the five sport groups this project collects.
RELEVANT_GROUPS = {"Basketball", "Baseball", "Ice Hockey", "American Football", "Soccer"}


def configured_sport_keys(config: dict[str, Any]) -> dict[str, str]:
    """Map league -> configured odds_api sport_key (only when present)."""

    keys: dict[str, str] = {}
    for league, league_cfg in (config.get("leagues") or {}).items():
        sport_key = ((league_cfg.get("sources") or {}).get("odds_api") or {}).get("sport_key")
        if sport_key:
            keys[league] = str(sport_key)
    return keys


def discover_available_sports(
    config: dict[str, Any],
    project_root: str | Path,
    api_key: str,
    *,
    fetch_json: Callable[[str], Any] | None = None,
    now: datetime | None = None,
    include_inactive: bool = True,
) -> dict[str, Any]:
    """Fetch /v4/sports, write the report JSON, and return the summary."""

    root = Path(project_root)
    now = now or datetime.now(timezone.utc)
    fetch = fetch_json or make_default_fetch_json()

    sources_cfg = config.get("sources") or {}
    base_url = (sources_cfg.get("odds_api") or {}).get("base_url", "https://api.the-odds-api.com/v4")
    params = {"apiKey": api_key}
    if include_inactive:
        params["all"] = "true"
    url = f"{base_url.rstrip('/')}/sports?{urllib.parse.urlencode(params)}"

    payload = fetch(url)
    sports = payload if isinstance(payload, list) else []
    soccer = [s for s in sports if str(s.get("key", "")).startswith("soccer_")]
    available_keys = {str(s.get("key", "")) for s in sports}
    active_keys = {str(s.get("key", "")) for s in sports if s.get("active")}

    configured = configured_sport_keys(config)
    configured_status = [
        {
            "league": league,
            "sport_key": sport_key,
            "available": sport_key in available_keys,
            "active": sport_key in active_keys,
        }
        for league, sport_key in configured.items()
    ]

    matching_leagues = [e["league"] for e in configured_status if e["available"]]
    configured_not_available = [e["league"] for e in configured_status if not e["available"]]
    configured_key_set = set(configured.values())
    supported_not_configured = [
        {
            "sport_key": str(s.get("key", "")),
            "title": s.get("title"),
            "group": s.get("group"),
            "active": bool(s.get("active")),
        }
        for s in sports
        if str(s.get("group", "")) in RELEVANT_GROUPS
        and str(s.get("key", "")) not in configured_key_set
        and not s.get("has_outrights")
    ]

    summary = {
        "report": "odds_api_available_sports",
        "generated_at_utc": now.isoformat(),
        "sports_count": len(sports),
        "soccer_count": len(soccer),
        "quota_remaining_requests": getattr(fetch, "quota_remaining", None),
        "configured_sport_keys": configured_status,
        "matching_leagues": matching_leagues,
        "configured_not_available": configured_not_available,
        "supported_not_configured": supported_not_configured,
        "soccer_sports": soccer,
        "sports": sports,
        "research_only": True,
        "approved": False,
        "notes": [
            "The /sports endpoint costs 0 credits when called with an API key.",
            "active=false sports are usually off-season; their events return empty.",
            "Discovery only: no models, no recommendations.",
        ],
    }

    report_path = root / "data" / "reports" / REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path = root / "data" / "reports" / REPORT_MD_FILENAME
    md_path.write_text(render_sports_markdown(summary), encoding="utf-8")
    summary["output_path"] = report_path.relative_to(root).as_posix()
    summary["output_md_path"] = md_path.relative_to(root).as_posix()
    return summary


def render_sports_markdown(summary: dict[str, Any]) -> str:
    """Human-readable companion to the discovery JSON."""

    lines = [
        "# Odds API Available Sports",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"- Sports available: {summary['sports_count']} ({summary['soccer_count']} soccer)",
        f"- Quota remaining: {summary.get('quota_remaining_requests')}",
        "",
        "## Configured project leagues",
        "",
        "| league | sport_key | available | active |",
        "| --- | --- | --- | --- |",
    ]
    for entry in summary["configured_sport_keys"]:
        lines.append(
            f"| {entry['league']} | {entry['sport_key']} | "
            f"{'yes' if entry['available'] else 'NO - check the key'} | "
            f"{'active' if entry['active'] else 'inactive/off-season'} |"
        )
    lines += [
        "",
        f"**Matching leagues** (configured and available): {', '.join(summary['matching_leagues']) or '(none)'}",
        "",
        f"**Configured but NOT available** (typo or removed sport): "
        f"{', '.join(summary['configured_not_available']) or '(none)'}",
        "",
        "## Supported but not configured (five relevant sport groups)",
        "",
        "| sport_key | title | group | active |",
        "| --- | --- | --- | --- |",
    ]
    for s in summary["supported_not_configured"]:
        lines.append(
            f"| {s['sport_key']} | {s['title']} | {s['group']} | "
            f"{'active' if s['active'] else 'inactive'} |"
        )
    lines += [
        "",
        "---",
        "Research-only discovery. No models, no recommendations; approved bets/parlays remain blocked.",
        "",
    ]
    return "\n".join(lines)
