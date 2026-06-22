"""Tests for the Odds API quota report logic (synthetic run-history fixtures)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_odds_api_quota_report import (  # noqa: E402
    analyze_runs,
    assess_risk,
    recommend_league_priority,
)


NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


def _run(run_time: str, remaining: float | None, outcome: str = "success",
         by_league: dict | None = None) -> dict:
    return {
        "run_id": run_time.replace("-", "").replace(":", ""),
        "run_time_utc": run_time,
        "outcome": outcome,
        "quota_remaining_requests": remaining,
        "snapshots_collected": sum((by_league or {}).values()),
        "snapshots_by_league": by_league or {},
    }


def _runs_fixture() -> list[dict]:
    return [
        _run("2026-06-10T09:00:00+00:00", 417.0, by_league={"NBA": 400, "MLB": 800, "EPL": 0}),
        _run("2026-06-10T13:00:00+00:00", 368.0, by_league={"NBA": 450, "MLB": 850, "EPL": 0}),
        _run("2026-06-11T09:00:00+00:00", 318.0, by_league={"NBA": 460, "MLB": 900, "EPL": 0}),
        _run("2026-06-11T10:00:00+00:00", None, outcome="skipped"),
    ]


CONFIG = {
    "quota": {"min_remaining_requests": 25, "low_priority_min_remaining": 50},
    "leagues": {
        "NBA": {"enabled": True, "priority": 1, "modeling_priority": True, "collect_only": False},
        "MLB": {"enabled": True, "priority": 3, "modeling_priority": False, "collect_only": True},
        "EPL": {"enabled": True, "priority": 7, "modeling_priority": False, "collect_only": True},
    },
}


class AnalyzeRunsTests(unittest.TestCase):
    def test_usage_counts_and_request_estimate(self) -> None:
        usage = analyze_runs(_runs_fixture(), NOW)
        self.assertEqual(usage["runs_today"], 2)
        self.assertEqual(usage["runs_this_month"], 4)
        self.assertEqual(usage["collected_runs_this_month"], 3)
        # Deltas: 417->368 (49) and 368->318 (50) => 49.5 requests per run.
        self.assertEqual(usage["avg_requests_per_run"], 49.5)
        self.assertEqual(usage["quota_remaining"], 318.0)
        leagues = {row["league"]: row for row in usage["leagues_consuming_requests"]}
        self.assertEqual(leagues["MLB"]["snapshots_this_month"], 2550)

    def test_quota_reset_does_not_produce_negative_delta(self) -> None:
        runs = _runs_fixture() + [_run("2026-06-11T11:00:00+00:00", 500.0)]
        usage = analyze_runs(runs, NOW)
        # The 318 -> 500 jump (reset) must not be counted as usage.
        self.assertEqual(usage["avg_requests_per_run"], 49.5)

    def test_empty_history(self) -> None:
        usage = analyze_runs([], NOW)
        self.assertEqual(usage["runs_this_month"], 0)
        self.assertIsNone(usage["avg_requests_per_run"])


class RiskTests(unittest.TestCase):
    def test_risk_unknown_without_observations(self) -> None:
        risk = assess_risk(analyze_runs([], NOW), CONFIG, NOW)
        self.assertEqual(risk["risk"], "unknown")

    def test_risk_assessed_with_recommendation(self) -> None:
        risk = assess_risk(analyze_runs(_runs_fixture(), NOW), CONFIG, NOW)
        self.assertIn(risk["risk"], {"low", "medium", "high"})
        self.assertIsNotNone(risk["recommended_max_runs_per_day"])
        self.assertEqual(risk["config_min_remaining_guard"], 25)


class LeaguePriorityTests(unittest.TestCase):
    def test_soccer_capped_when_inactive(self) -> None:
        usage = analyze_runs(_runs_fixture(), NOW)
        rec = recommend_league_priority(usage, CONFIG)
        self.assertTrue(rec["soccer_should_remain_capped"])
        by_league = {row["league"]: row for row in rec["league_priority"]}
        self.assertIn("keep first", by_league["NBA"]["recommendation"])
        self.assertIn("inactive", by_league["EPL"]["recommendation"])
        self.assertIn("keep collecting", by_league["MLB"]["recommendation"])


if __name__ == "__main__":
    unittest.main()
