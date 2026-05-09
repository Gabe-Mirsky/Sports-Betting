"""Build home-win ensemble diagnostics from saved walk-forward predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logging_setup import setup_logging  # noqa: E402
from models.ensemble import (  # noqa: E402
    build_home_win_ensemble,
    build_static_blend_audit,
    prepare_home_win_ensemble_frame,
    save_home_win_ensemble_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build home-win ensemble diagnostics.")
    parser.add_argument("--base-predictions-path", default=None)
    parser.add_argument("--tuned-predictions-path", default=None)
    parser.add_argument("--margin-predictions-path", default=None)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--output-predictions-path", default=None)
    parser.add_argument("--output-weights-path", default=None)
    parser.add_argument("--output-static-audit-path", default=None)
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    reports_dir = PROJECT_ROOT / "data" / "reports"
    base_path = Path(args.base_predictions_path) if args.base_predictions_path else reports_dir / "walk_forward_predictions.csv"
    tuned_path = (
        Path(args.tuned_predictions_path)
        if args.tuned_predictions_path
        else reports_dir / "tuned_walk_forward_predictions.csv"
    )
    margin_path = (
        Path(args.margin_predictions_path)
        if args.margin_predictions_path
        else reports_dir / "market_type_predictions.csv"
    )
    predictions_path = (
        Path(args.output_predictions_path)
        if args.output_predictions_path
        else reports_dir / "home_win_ensemble_predictions.csv"
    )
    weights_path = (
        Path(args.output_weights_path)
        if args.output_weights_path
        else reports_dir / "home_win_ensemble_weights.csv"
    )
    static_audit_path = (
        Path(args.output_static_audit_path)
        if args.output_static_audit_path
        else reports_dir / "home_win_ensemble_static_audit.csv"
    )
    summary_path = (
        Path(args.output_summary_path)
        if args.output_summary_path
        else reports_dir / "home_win_ensemble_summary.json"
    )

    base = pd.read_csv(base_path, dtype={"game_id": str})
    tuned = pd.read_csv(tuned_path, dtype={"game_id": str})
    margin = pd.read_csv(margin_path, dtype={"game_id": str})
    frame = prepare_home_win_ensemble_frame(base, tuned, margin)
    predictions, weights, summary = build_home_win_ensemble(
        frame,
        weight_step=args.weight_step,
        min_train_rows=args.min_train_rows,
    )
    static_audit = build_static_blend_audit(frame, weight_step=args.weight_step)
    summary["static_audit_best"] = static_audit.head(1).to_dict(orient="records")[0] if not static_audit.empty else {}
    save_home_win_ensemble_outputs(
        predictions,
        weights,
        static_audit,
        summary,
        predictions_path,
        weights_path,
        static_audit_path,
        summary_path,
    )

    ensemble = summary.get("ensemble", {})
    best_component = summary.get("best_component", "n/a")
    print(f"Ensemble rows ({summary.get('timeline', 'n/a')}): {summary.get('rows', 0):,}")
    print(f"Ensemble log loss: {ensemble.get('log_loss', float('nan')):.4f}")
    print(f"Best component: {best_component}")
    print(f"Delta vs best component: {summary.get('log_loss_delta_vs_best_component', float('nan')):.4f}")
    print(f"Adoption status: {summary.get('adoption_status', 'n/a')}")
    print(f"Saved ensemble predictions to: {predictions_path}")
    print(f"Saved ensemble summary to: {summary_path}")


if __name__ == "__main__":
    main()
