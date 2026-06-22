"""Multi-source data router (research-only).

Decides which data source (odds_api / sportsgameodds / apisports / kalshi) should
serve a given (scope, data_type) request, based on capability, key availability,
quota vs floor, recent failures and freshness. The decision logic is PURE: it
takes a snapshot of per-source state and returns a RouteDecision. Network probing
lives in scripts/probe_all_sources.py.

The router enables NO betting, parlays, predictions, or recommendations. When no
source is safe it returns a SKIP decision with a clear reason instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DATA_TYPES = (
    "events_schedule", "game_odds", "player_props", "results",
    "player_stats", "closing_lines", "prediction_market_prices",
)

SKIP_PAID_ODDS = "SKIP_PAID_ODDS"
SKIP_NO_SAFE_SOURCE = "SKIP_NO_SAFE_SOURCE"
PAID_DATA_TYPES = {"game_odds", "player_props", "results", "closing_lines"}


@dataclass
class SourceState:
    """Live snapshot of one source (built by the prober or by tests)."""

    name: str
    key_present: bool = False
    quota_remaining: float | None = None
    last_success_utc: str | None = None
    last_failure_utc: str | None = None
    blocked_reason: str | None = None  # e.g. plan-blocked season

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "key_present": self.key_present,
            "quota_remaining": self.quota_remaining, "last_success_utc": self.last_success_utc,
            "last_failure_utc": self.last_failure_utc, "blocked_reason": self.blocked_reason,
        }


@dataclass
class RouteDecision:
    scope: str
    data_type: str
    market_type: str | None
    selected: str | None
    skipped: list[dict[str, str]] = field(default_factory=list)
    reason: str = "ok"
    candidates: list[str] = field(default_factory=list)
    research_only: bool = True

    @property
    def ok(self) -> bool:
        return self.selected is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope, "data_type": self.data_type, "market_type": self.market_type,
            "selected": self.selected, "skipped": self.skipped, "reason": self.reason,
            "candidates": self.candidates, "research_only": True,
        }


def load_router_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _hours_since(iso: str | None, now: datetime) -> float | None:
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 3600.0


class SourceRouter:
    def __init__(
        self,
        config: dict[str, Any],
        states: dict[str, SourceState] | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config or {}
        self.sources = self.config.get("sources", {})
        self.routes = self.config.get("routes", {})
        self.quota_floors = self.config.get("quota_floors", {})
        self.cooldown_h = float(self.config.get("recent_failure_cooldown_hours", 1) or 0)
        self.states = states or {}
        self.now = now or datetime.now(timezone.utc)

    # -- helpers ---------------------------------------------------------- #
    def _candidates(self, scope: str, data_type: str) -> list[str]:
        scoped = (self.routes.get(scope) or {}).get(data_type)
        if scoped:
            return list(scoped)
        return list((self.routes.get("DEFAULT") or {}).get(data_type) or [])

    def _state(self, src: str) -> SourceState:
        return self.states.get(src) or SourceState(name=src)

    def _is_free(self, src: str, data_type: str) -> bool:
        return data_type in (self.sources.get(src, {}).get("free_data_types") or [])

    def _reason_source_unusable(self, src: str, data_type: str) -> str | None:
        """Return a skip reason if this source cannot serve data_type now, else None."""
        cap = self.sources.get(src)
        if not cap:
            return "unknown_source"
        if data_type not in (cap.get("supports") or []):
            return "unsupported_data_type"
        st = self._state(src)
        if st.blocked_reason:
            return f"blocked:{st.blocked_reason}"
        if cap.get("key_required", True) and not st.key_present:
            return "no_key"
        # recent-failure cooldown
        fail_age = _hours_since(st.last_failure_utc, self.now)
        if fail_age is not None and self.cooldown_h and fail_age < self.cooldown_h:
            if (_hours_since(st.last_success_utc, self.now) or 1e9) > (fail_age or 0):
                return f"recent_failure({fail_age:.1f}h<{self.cooldown_h:g}h)"
        # quota floor for paid data types only
        if not self._is_free(src, data_type):
            floor = self.quota_floors.get(src)
            if floor is not None and st.quota_remaining is not None and st.quota_remaining < float(floor):
                return f"below_quota_floor({st.quota_remaining:g}<{floor:g})"
        return None

    # -- public ----------------------------------------------------------- #
    def route(self, scope: str, data_type: str, market_type: str | None = None) -> RouteDecision:
        if data_type not in DATA_TYPES:
            return RouteDecision(scope, data_type, market_type, None,
                                 reason=f"unknown_data_type:{data_type}")
        candidates = self._candidates(scope, data_type)
        decision = RouteDecision(scope, data_type, market_type, None, candidates=list(candidates))
        if not candidates:
            decision.reason = SKIP_NO_SAFE_SOURCE
            return decision
        for src in candidates:
            why = self._reason_source_unusable(src, data_type)
            if why is None:
                decision.selected = src
                decision.reason = "ok"
                return decision
            decision.skipped.append({"source": src, "reason": why})
        # nothing usable -> classify the skip
        all_paid_blocked = data_type in PAID_DATA_TYPES and all(
            ("below_quota_floor" in s["reason"] or s["reason"] == "no_key" or s["reason"].startswith("blocked"))
            for s in decision.skipped
        )
        decision.reason = SKIP_PAID_ODDS if all_paid_blocked else SKIP_NO_SAFE_SOURCE
        return decision

    def record_fetch(
        self,
        decision: RouteDecision,
        *,
        event_id: str | None = None,
        game_id: str | None = None,
        market_type: str | None = None,
        rows: int = 0,
        quota_before: float | None = None,
        quota_after: float | None = None,
        success: bool = False,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Build a routed-fetch log entry (caller appends it to a JSONL log)."""
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": decision.scope,
            "data_type": decision.data_type,
            "source_selected": decision.selected,
            "sources_skipped": decision.skipped,
            "reason": detail or decision.reason,
            "event_id": event_id or game_id,
            "market_type": market_type or decision.market_type,
            "rows": int(rows),
            "quota_before": quota_before,
            "quota_after": quota_after,
            "success": bool(success),
            "research_only": True,
        }


def best_source_by_data_type(
    config: dict[str, Any], states: dict[str, SourceState], now: datetime | None = None,
) -> dict[str, dict[str, str | None]]:
    """For each scope+data_type, return the currently-selected source (or skip)."""
    router = SourceRouter(config, states, now=now)
    out: dict[str, dict[str, str | None]] = {}
    for scope, table in (config.get("routes") or {}).items():
        out[scope] = {}
        for data_type in table:
            d = router.route(scope, data_type)
            out[scope][data_type] = d.selected or d.reason
    return out
