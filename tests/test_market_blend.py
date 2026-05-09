from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from models.market_blend import add_market_blended_probabilities  # noqa: E402


class TestMarketBlend(unittest.TestCase):
    def test_blend_uses_prior_dates_only(self) -> None:
        rows = []
        for index, date in enumerate(["2025-01-01", "2025-01-01", "2025-01-02"]):
            rows.append(
                {
                    "game_date": date,
                    "game_id": f"g{index}",
                    "market_ticker": f"M{index}",
                    "model_yes_prob": 0.70 if index != 1 else 0.30,
                    "yes_mid_cents": 60 if index != 1 else 40,
                    "actual_yes_win": index != 1,
                    "is_playoffs": False,
                }
            )
        data = pd.DataFrame(rows)

        blended, metrics = add_market_blended_probabilities(
            data,
            min_train_rows=1,
            use_playoff_features=False,
        )

        self.assertEqual(blended.loc[0, "blend_method"], "warmup_half_model_half_market")
        self.assertEqual(blended.loc[1, "blend_method"], "warmup_half_model_half_market")
        self.assertEqual(blended.loc[2, "blend_method"], "expanding_market_blend")
        self.assertIn("same-date games are not used", metrics["note"])


if __name__ == "__main__":
    unittest.main()
