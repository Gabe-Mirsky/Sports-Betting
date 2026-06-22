"""Build the daily multi-source health report (research-only).

Reads data/reports/source_state.json (from probe_all_sources.py) and
config/source_priority.yaml, then writes:
  data/reports/source_health_summary.json
  data/reports/source_health_summary.md

Shows: sources available/blocked, keys detected (booleans only, never the key),
quotas remaining, latest success/failure, best source per data type, and the
World Cup / NBA source status. No betting, parlays, predictions, recommendations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.source_router import (  # noqa: E402
    SourceState, best_source_by_data_type, load_router_config,
)

STATE_PATH = PROJECT_ROOT / "data" / "reports" / "source_state.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "source_priority.yaml"
OUT_JSON = PROJECT_ROOT / "data" / "reports" / "source_health_summary.json"
OUT_MD = PROJECT_ROOT / "data" / "reports" / "source_health_summary.md"


def _states_from_file(state_doc: dict) -> dict[str, SourceState]:
    out = {}
    for name, s in (state_doc.get("sources") or {}).items():
        out[name] = SourceState(
            name=name, key_present=bool(s.get("key_present")),
            quota_remaining=s.get("quota_remaining"),
            last_success_utc=s.get("last_success_utc"),
            last_failure_utc=s.get("last_failure_utc"),
            blocked_reason=s.get("blocked_reason"),
        )
    return out


def main() -> None:
    setup_logging("INFO")
    config = load_router_config(CONFIG_PATH)
    state_doc = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"sources": {}}
    states = _states_from_file(state_doc)
    floors = config.get("quota_floors", {})

    sources_out = {}
    available, blocked = [], []
    for name, st in states.items():
        floor = floors.get(name)
        below = (st.quota_remaining is not None and floor is not None and st.quota_remaining < float(floor))
        is_blocked = bool(st.blocked_reason) or (st.last_failure_utc and not st.last_success_utc)
        status = "blocked" if is_blocked else ("available" if (st.key_present or not config.get("sources", {}).get(name, {}).get("key_required", True)) else "no_key")
        sources_out[name] = {
            "status": status,
            "key_detected": st.key_present,
            "quota_remaining": st.quota_remaining,
            "quota_floor": floor,
            "below_floor": below,
            "last_success_utc": st.last_success_utc,
            "last_failure_utc": st.last_failure_utc,
            "blocked_reason": st.blocked_reason,
            "kind": config.get("sources", {}).get(name, {}).get("kind"),
            "used_for": (state_doc.get("sources", {}).get(name, {}) or {}).get("used_for"),
        }
        (blocked if is_blocked else available).append(name)

    best = best_source_by_data_type(config, states)
    summary = {
        "report": "source_health_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True, "approved": False,
        "state_generated_at_utc": state_doc.get("generated_at_utc"),
        "state_was_dry_run": state_doc.get("dry_run"),
        "sources_available": available,
        "sources_blocked": blocked,
        "sources": sources_out,
        "best_source_by_data_type": best,
        "world_cup_source_status": best.get("WORLD_CUP", {}),
        "nba_source_status": best.get("NBA", {}),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(_render_md(summary), encoding="utf-8")

    print(f"Source health: available={available} blocked={blocked}")
    print(f"Wrote {OUT_JSON.relative_to(PROJECT_ROOT)} and {OUT_MD.relative_to(PROJECT_ROOT)}")
    print("Research-only: no keys printed; no betting/parlays/predictions/recommendations.")


def _render_md(s: dict) -> str:
    def badge(v):
        return {"available": "🟢 available", "blocked": "🔴 blocked", "no_key": "⚪ no key"}.get(v, v)
    lines = [
        "# Source Health Summary",
        f"_Generated {s['generated_at_utc']} — research-only. Keys are detected as booleans only; never printed._",
        "",
        f"- State snapshot: {s.get('state_generated_at_utc')} (dry_run={s.get('state_was_dry_run')})",
        f"- Available: {', '.join(s['sources_available']) or '—'}",
        f"- Blocked: {', '.join(s['sources_blocked']) or '—'}",
        "",
        "## Sources",
        "| Source | Status | Key | Quota | Floor | Below floor | Last success | Last failure | Used for |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, v in s["sources"].items():
        lines.append(
            f"| {name} | {badge(v['status'])} | {v['key_detected']} | {v['quota_remaining']} | "
            f"{v['quota_floor']} | {v['below_floor']} | {v['last_success_utc'] or '—'} | "
            f"{v['last_failure_utc'] or '—'} | {v['used_for'] or ''} |"
        )
    lines += ["", "## Best source by data type", ""]
    for scope, table in s["best_source_by_data_type"].items():
        lines.append(f"**{scope}**")
        for dt, src in table.items():
            lines.append(f"- {dt}: `{src}`")
        lines.append("")
    lines += [
        "## World Cup source status",
        "\n".join(f"- {k}: `{v}`" for k, v in s["world_cup_source_status"].items()) or "- n/a",
        "",
        "## NBA source status",
        "\n".join(f"- {k}: `{v}`" for k, v in s["nba_source_status"].items()) or "- n/a",
        "",
        "_Research-only. Source routing decides where research data is fetched; it enables no "
        "betting, parlays, predictions, or recommendations and changes no model gate._",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
