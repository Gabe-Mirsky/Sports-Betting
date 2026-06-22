"""Run side-corrected CLV rule sweeps for calibrated YES and NO signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from strategy.clv_concentration import (  # noqa: E402
    run_clv_price_month_sweep,
    run_walk_forward_clv_price_month_validation,
    save_clv_price_month_sweep_outputs,
    save_walk_forward_clv_price_month_outputs,
)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected YES/NO CLV rule sweeps from calibrated signals.",
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--signal-column", default="calibrated_trade")
    parser.add_argument("--price-breaks", default="0,5,10,15,20,25,30,40,55,70,85,100")
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-train-months", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="corrected_clv")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _side_payload(summary: dict[str, Any], walk_forward_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "descriptive_status": summary.get("best_status", "not_ready"),
        "descriptive_rows": int(summary.get("rows", 0) or 0),
        "descriptive_rules_tested": int(summary.get("rules_tested", 0) or 0),
        "descriptive_best_rule": summary.get("best_rule", "n/a"),
        "descriptive_best_positive_clv_rate": float(summary.get("best_rule_positive_clv_rate", 0.0) or 0.0),
        "descriptive_best_avg_clv_cents": float(summary.get("best_rule_avg_clv_cents", 0.0) or 0.0),
        "descriptive_best_avg_profit_per_share": float(summary.get("best_rule_avg_profit_per_share", 0.0) or 0.0),
        "walk_forward_status": walk_forward_summary.get("status", "not_ready"),
        "walk_forward_signals": int(walk_forward_summary.get("signals", 0) or 0),
        "walk_forward_positive_clv_rate": float(walk_forward_summary.get("positive_clv_rate", 0.0) or 0.0),
        "walk_forward_avg_clv_cents": float(walk_forward_summary.get("avg_clv_cents", 0.0) or 0.0),
        "walk_forward_avg_profit_per_share": float(walk_forward_summary.get("avg_profit_per_share", 0.0) or 0.0),
        "walk_forward_positive_month_share": float(walk_forward_summary.get("positive_month_share", 0.0) or 0.0),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    input_path = Path(args.input_path) if args.input_path else reports_dir / "edge_calibrated_trades.csv"
    output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    price_breaks = _parse_float_list(args.price_breaks)
    rows = pd.read_csv(input_path, dtype={"game_id": str, "market_ticker": str})

    combined: dict[str, Any] = {
        "input_path": str(input_path),
        "signal_column": args.signal_column,
        "price_breaks": price_breaks,
        "min_rows": int(args.min_rows),
        "min_train_months": int(args.min_train_months),
        "sides": {},
        "single_game_edge_proven": False,
        "parlay_research_allowed": False,
        "note": "These sweeps use side-corrected CLV from calibrated signals. Passing descriptive sweeps is not enough; walk-forward CLV must pass before strategy use.",
    }

    for side in ["YES", "NO"]:
        side_key = side.lower()
        rules, monthly, summary = run_clv_price_month_sweep(
            rows,
            signal_column=args.signal_column,
            side=side,
            price_breaks=price_breaks,
            min_rows=args.min_rows,
        )
        save_clv_price_month_sweep_outputs(
            rules,
            monthly,
            summary,
            output_dir / f"{args.prefix}_{side_key}_sweep.csv",
            output_dir / f"{args.prefix}_{side_key}_sweep_monthly.csv",
            output_dir / f"{args.prefix}_{side_key}_sweep_summary.json",
        )
        validated, folds, walk_monthly, walk_summary = run_walk_forward_clv_price_month_validation(
            rows,
            signal_column=args.signal_column,
            side=side,
            price_breaks=price_breaks,
            min_rows=args.min_rows,
            min_train_months=args.min_train_months,
        )
        save_walk_forward_clv_price_month_outputs(
            validated,
            folds,
            walk_monthly,
            walk_summary,
            output_dir / f"{args.prefix}_{side_key}_walk_forward_trades.csv",
            output_dir / f"{args.prefix}_{side_key}_walk_forward_folds.csv",
            output_dir / f"{args.prefix}_{side_key}_walk_forward_monthly.csv",
            output_dir / f"{args.prefix}_{side_key}_walk_forward_summary.json",
        )
        combined["sides"][side] = _side_payload(summary, walk_summary)

    combined_path = output_dir / f"{args.prefix}_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    for side, payload in combined["sides"].items():
        print(
            f"{side}: descriptive={payload['descriptive_status']} "
            f"walk_forward={payload['walk_forward_status']} "
            f"signals={payload['walk_forward_signals']:,} "
            f"wf_pos_clv={payload['walk_forward_positive_clv_rate']:.1%}"
        )
    print(f"Saved corrected CLV sweep summary to: {combined_path}")


if __name__ == "__main__":
    main()
