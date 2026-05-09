from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import PipelineOptions, build_pipeline_commands  # noqa: E402


class TestPipeline(unittest.TestCase):
    def test_default_pipeline_commands_use_walk_forward_predictions(self) -> None:
        commands = build_pipeline_commands(
            PipelineOptions(
                python_executable="python",
                project_root=PROJECT_ROOT,
            )
        )
        names = [name for name, _ in commands]
        self.assertEqual(
            names,
            [
                "Build features",
                "Train models",
                "Walk-forward predictions",
                "Tune home-win model",
                "Train spread and total models",
                "Build home-win ensemble",
                "Run backtest",
                "Market blend",
                "Calibrate edges",
                "Optimize individual slate",
                "Optimize calibrated slate",
                "Calibrate market-blend edges",
                "Build consensus calibrated edges",
                "Screen robust consensus edges",
                "Optimize market-blend calibrated slate",
                "Optimize consensus calibrated slate",
                "Optimize robust consensus slate",
                "Analyze consensus stability",
                "Analyze robust stability",
                "Assess strategy readiness",
                "Build headline slate result",
                "Sweep signal rules",
                "Validate signal rules walk-forward",
                "Analyze parlay correlations",
                "Build Kalshi market taxonomy",
                "Audit market-type lines",
                "Evaluate line markets",
                "Extract multivariate NBA legs",
                "Build forward recommendations",
                "Sweep thresholds",
                "Audit Kalshi vs model",
                "Analyze results",
                "Security audit",
                "Build dashboard",
            ],
        )
        backtest_command = dict(commands)["Run backtest"]
        self.assertIn("walk_forward_predictions.csv", " ".join(backtest_command))
        calibrated_command = dict(commands)["Optimize calibrated slate"]
        self.assertIn("--use-calibrated-edges", calibrated_command)
        blend_calibrated_command = dict(commands)["Optimize market-blend calibrated slate"]
        self.assertIn("edge_calibrated_trades_market_blend.csv", " ".join(blend_calibrated_command))
        consensus_command = dict(commands)["Optimize consensus calibrated slate"]
        self.assertIn("consensus_trade", consensus_command)
        robust_command = dict(commands)["Optimize robust consensus slate"]
        self.assertIn("robust_calibrated_trade", robust_command)

    def test_download_step_is_optional_and_first(self) -> None:
        commands = build_pipeline_commands(
            PipelineOptions(
                python_executable="python",
                project_root=PROJECT_ROOT,
                download=True,
                start_season=2018,
                end_season=2025,
            )
        )
        self.assertEqual(commands[0][0], "Download NBA data")
        self.assertIn("--start-season", commands[0][1])
        self.assertIn("2018", commands[0][1])
        self.assertIn("--end-season", commands[0][1])
        self.assertIn("2025", commands[0][1])

    def test_skip_flags_remove_steps(self) -> None:
        commands = build_pipeline_commands(
            PipelineOptions(
                python_executable="python",
                project_root=PROJECT_ROOT,
                skip_features=True,
                skip_train=True,
                skip_walk_forward=True,
                skip_model_tuning=True,
                skip_market_type_models=True,
                skip_backtest=True,
                skip_edge_calibration=True,
                skip_market_blend=True,
                skip_portfolio=True,
                skip_forward=True,
                skip_sweep=True,
                skip_diagnostics=True,
                skip_dashboard=True,
            )
        )
        self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
