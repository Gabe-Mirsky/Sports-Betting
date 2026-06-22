from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from quality.matchup_input_validation import (  # noqa: E402
    STATUS_FAIL,
    STATUS_WARNING,
    build_validation_report,
)

TODAY = pd.Timestamp("2026-06-20")


def _valid_results() -> pd.DataFrame:
    rows = []
    teams = ["Japan", "Tunisia", "South Korea", "Australia"]
    gid = 0
    start = pd.Timestamp("2025-01-01")
    for rnd in range(4):
        for i, a in enumerate(teams):
            b = teams[(i + 1 + rnd) % len(teams)]
            gid += 1
            rows.append(
                {
                    "game_id": f"g{gid}",
                    "date": (start + pd.Timedelta(days=gid * 3)).date().isoformat(),
                    "team_a": a,
                    "team_b": b,
                    "team_a_score": (gid % 3),
                    "team_b_score": ((gid + 1) % 3),
                    "sport": "soccer",
                    "league": "international",
                    "competition_type": "friendly",
                    "neutral_site": 0,
                    "team_a_home_flag": 1,
                    "team_b_home_flag": 0,
                }
            )
    return pd.DataFrame(rows)


def _valid_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fixture_id": "fx1", "game_date": "2026-06-20", "team_a": "Japan", "team_b": "Tunisia",
             "neutral_site": 1, "sport": "soccer", "league": "international_friendly",
             "competition_type": "friendly", "venue": "Neutral", "status": "scheduled"},
        ]
    )


def _valid_injuries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Tunisia", "player_name": "Striker", "status": "out", "injury_type": "hamstring",
             "position": "FW", "importance_score": 0.9, "expected_minutes_or_role": "star",
             "last_updated": "2026-06-19T12:00:00", "return_estimate": "2026-07-01"},
        ]
    )


class ValidationHarness(unittest.TestCase):
    def _run(self, results=None, fixtures=None, injuries="default", **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_path = tmp_path / "results.csv"
            (results if results is not None else _valid_results()).to_csv(results_path, index=False)
            fixtures_path = tmp_path / "fixtures.csv"
            (fixtures if fixtures is not None else _valid_fixtures()).to_csv(fixtures_path, index=False)

            injuries_path = None
            if isinstance(injuries, pd.DataFrame):
                injuries_path = tmp_path / "injuries.csv"
                injuries.to_csv(injuries_path, index=False)
            elif isinstance(injuries, str) and injuries == "default":
                injuries_path = tmp_path / "injuries.csv"
                _valid_injuries().to_csv(injuries_path, index=False)

            return build_validation_report(
                results_path=results_path,
                fixtures_path=fixtures_path,
                injuries_path=injuries_path,
                aliases_path=None,
                today=TODAY,
                **kwargs,
            )


class TestValidationReport(ValidationHarness):
    def test_valid_files_pass(self) -> None:
        report = self._run()
        self.assertNotEqual(report["overall_status"], STATUS_FAIL)
        self.assertTrue(report["recommendation"]["safe_to_backtest"])
        self.assertTrue(report["recommendation"]["safe_to_predict"])

    def test_missing_injuries_warns_not_fail(self) -> None:
        report = self._run(injuries=None)
        self.assertNotEqual(report["overall_status"], STATUS_FAIL)
        self.assertFalse(report["injuries"]["present"])
        self.assertTrue(any("No injuries file" in w for w in report["injuries"]["warnings"]))

    def test_duplicate_game_id_fails(self) -> None:
        results = _valid_results()
        results.loc[results.index[1], "game_id"] = results.loc[results.index[0], "game_id"]
        report = self._run(results=results)
        self.assertEqual(report["overall_status"], STATUS_FAIL)
        self.assertTrue(any("duplicate" in i.lower() for i in report["results"]["issues"]))

    def test_all_invalid_dates_fail(self) -> None:
        results = _valid_results()
        results["date"] = "not-a-date"
        report = self._run(results=results)
        self.assertEqual(report["overall_status"], STATUS_FAIL)

    def test_unknown_injury_status_warns(self) -> None:
        injuries = _valid_injuries()
        injuries.loc[0, "status"] = "banana"
        report = self._run(injuries=injuries)
        self.assertNotEqual(report["overall_status"], STATUS_FAIL)
        self.assertTrue(any("not recognised" in w for w in report["injuries"]["warnings"]))

    def test_fixture_team_missing_history_warns(self) -> None:
        fixtures = _valid_fixtures()
        fixtures.loc[0, "team_b"] = "Neverland"  # no history
        report = self._run(fixtures=fixtures)
        self.assertNotEqual(report["overall_status"], STATUS_FAIL)
        self.assertIn("Neverland", report["team_matching"]["fixture_teams_missing_history"])

    def test_future_fixture_with_final_score_warns(self) -> None:
        fixtures = _valid_fixtures()
        fixtures["team_a_score"] = 2  # upcoming game should not be scored
        fixtures["team_b_score"] = 1
        report = self._run(fixtures=fixtures)
        self.assertTrue(any("final score" in w for w in report["fixtures"]["warnings"]))

    def test_strict_escalates_warnings_to_fail(self) -> None:
        # A clean-but-warning case (no injuries) becomes FAIL under --strict.
        report = self._run(injuries=None, strict=True)
        self.assertEqual(report["overall_status"], STATUS_FAIL)

    def test_too_few_games_warns(self) -> None:
        results = _valid_results().head(4)
        report = self._run(results=results)
        self.assertEqual(report["overall_status"], STATUS_WARNING)
        self.assertTrue(any("historical games" in w for w in report["results"]["warnings"]))


if __name__ == "__main__":
    unittest.main()
