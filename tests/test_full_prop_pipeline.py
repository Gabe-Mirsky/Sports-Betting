"""Tests for the master pipeline plumbing (synthetic pipeline-summary fixture).

Does NOT run any collection or hit any API: it checks the step plan and the
summary-context extraction against fake report files.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_full_prop_pipeline as pipeline  # noqa: E402


class StepPlanTests(unittest.TestCase):
    def test_step_order_collect_first_dashboard_last(self) -> None:
        names = [step[0] for step in pipeline.PIPELINE_STEPS]
        self.assertEqual(names[0], "collect_props")
        self.assertEqual(names[-1], "dashboard")
        # Settlement refresh must run before the reports that consume it.
        self.assertLess(names.index("refresh_results_cache_only"), names.index("settlement_outcomes"))
        self.assertLess(names.index("prop_clv"), names.index("data_quality_gates"))
        # The next-action report needs every other report fresh first.
        self.assertLess(names.index("data_quality_gates"), names.index("next_action"))

    def test_every_step_script_exists_or_is_optional(self) -> None:
        for name, filename, _args, _kind, optional in pipeline.PIPELINE_STEPS:
            script = PROJECT_ROOT / "scripts" / filename
            if not optional:
                self.assertTrue(script.exists(), f"required step {name} missing {filename}")


class ContextTests(unittest.TestCase):
    def test_collect_context_reads_fixture_reports(self) -> None:
        # Synthetic pipeline-summary inputs: a collection run summary + gates.
        with tempfile.TemporaryDirectory() as folder:
            reports = Path(folder)
            (reports / "player_prop_collection_run_summary.json").write_text(
                json.dumps({
                    "run_id": "fixture_run",
                    "status": "success",
                    "totals": {"snapshots_total": 7013, "closing_snapshots_total": 526},
                }),
                encoding="utf-8",
            )
            (reports / "player_prop_data_quality_gates.json").write_text(
                json.dumps({
                    "status": "settlement_ready",
                    "metrics": {"settled_props": 1858, "clv_markets": 0},
                }),
                encoding="utf-8",
            )
            with mock.patch.object(pipeline, "REPORTS_DIR", reports):
                context = pipeline.collect_context()

        self.assertEqual(context["snapshots_total"], 7013)
        self.assertEqual(context["closing_snapshots_total"], 526)
        self.assertEqual(context["last_collection_run_id"], "fixture_run")
        self.assertEqual(context["data_gate_status"], "settlement_ready")
        self.assertEqual(context["nba_settled_props"], 1858)

    def test_collect_context_handles_missing_reports(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(pipeline, "REPORTS_DIR", Path(folder)):
                context = pipeline.collect_context()
        self.assertEqual(context, {})


if __name__ == "__main__":
    unittest.main()
