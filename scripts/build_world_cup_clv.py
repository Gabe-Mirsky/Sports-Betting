"""Build the World Cup CLV summary from collected snapshots (research-only).

Reads data/processed/world_cup_odds_snapshots_normalized.csv and writes
data/reports/world_cup_clv_summary.json/.md plus a pairs CSV. World Cup CLV is
kept entirely separate from NBA model gates. No bets, parlays, or predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from data.world_cup_clv import build_world_cup_clv, render_clv_markdown  # noqa: E402

SNAP = PROJECT_ROOT / "data" / "processed" / "world_cup_odds_snapshots_normalized.csv"
OUT_JSON = PROJECT_ROOT / "data" / "reports" / "world_cup_clv_summary.json"
OUT_MD = PROJECT_ROOT / "data" / "reports" / "world_cup_clv_summary.md"
OUT_PAIRS = PROJECT_ROOT / "data" / "reports" / "world_cup_clv_pairs.csv"


def main() -> None:
    setup_logging("INFO")
    snaps = pd.read_csv(SNAP, low_memory=False) if SNAP.exists() else pd.DataFrame()
    summary = build_world_cup_clv(snaps)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(render_clv_markdown(summary), encoding="utf-8")
    pd.DataFrame(summary.get("pairs", [])).to_csv(OUT_PAIRS, index=False)

    print(f"World Cup CLV: clv_ready={summary['clv_ready']} | outcomes_with_clv={summary['markets_with_clv']} | "
          f"avg_price_clv={summary['avg_price_clv']}")
    print(f"  {summary['verdict']}")
    print(f"Wrote: {OUT_JSON.relative_to(PROJECT_ROOT)}, {OUT_MD.relative_to(PROJECT_ROOT)}")
    print("Research-only: CLV measurement only; isolated from NBA gates; no bets/parlays/predictions.")


if __name__ == "__main__":
    main()
