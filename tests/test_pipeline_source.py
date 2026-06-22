from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_fair_prices  # noqa: E402
import build_parlay_recommendations  # noqa: E402
import run_single_game_research_pipeline  # noqa: E402


class TestPipelineSource(unittest.TestCase):
    def test_single_game_pipeline_pins_kalshi_backtest_source(self) -> None:
        args = SimpleNamespace(
            download=False,
            force_download=False,
            start_season=2018,
            end_season=2025,
            kalshi_start_date="2023-10-01",
            kalshi_end_date="2026-06-08",
            bankroll=100.0,
            edge_threshold=0.05,
            max_spread_cents=10.0,
            min_volume=10.0,
            skip_market_pull=True,
            skip_candles=True,
            skip_dashboard=True,
            log_level="INFO",
        )

        commands = run_single_game_research_pipeline.build_commands(args)
        backtest_commands = [command for name, command in commands if name == "Run realistic bid/ask backtest"]

        self.assertEqual(len(backtest_commands), 1)
        command = backtest_commands[0]
        self.assertIn("--market-source", command)
        self.assertEqual(command[command.index("--market-source") + 1], "kalshi")

    def test_fair_price_validation_rejects_non_kalshi_backtest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "backtest_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "market_source": "sportsbook",
                        "canonical_kalshi_backtest": False,
                        "price_source": "sportsbook_no_vig_moneyline",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "non-Kalshi"):
                build_fair_prices.validate_canonical_backtest_summary(summary_path)

    def test_fair_price_validation_accepts_canonical_bid_ask_kalshi_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "backtest_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "market_source": "kalshi",
                        "canonical_kalshi_backtest": True,
                        "price_source": "kalshi_candlesticks_bid_ask",
                        "bid_ask_required": True,
                        "stale_artifacts_detected": False,
                    }
                ),
                encoding="utf-8",
            )

            summary = build_fair_prices.validate_canonical_backtest_summary(summary_path)

            self.assertEqual(summary["market_source"], "kalshi")

    def test_parlay_validation_rejects_unvalidated_fair_price_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "fair_price_summary.json"
            summary_path.write_text(
                json.dumps({"validated_backtest_market_source": "sportsbook"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "validated Kalshi"):
                build_parlay_recommendations.validate_fair_price_summary(summary_path)


if __name__ == "__main__":
    unittest.main()
