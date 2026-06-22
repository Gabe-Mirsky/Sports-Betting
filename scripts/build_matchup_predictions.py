"""Build model-implied matchup predictions for upcoming fixtures (no odds).

Example
-------
    python scripts/build_matchup_predictions.py \
        --results-path data/processed/match_results.csv \
        --fixtures-path data/processed/fixtures_today.csv \
        --injuries-path data/processed/injuries.csv \
        --output-dir data/reports

The output is a set of "model probability" report artifacts – never betting
odds. The pipeline never requires market prices, CLV, or closing lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.fixtures_loader import load_fixtures, normalize_fixtures  # noqa: E402
from data.injuries_loader import load_injuries, normalize_injuries  # noqa: E402
from data.match_results_loader import (  # noqa: E402
    load_match_results,
    normalize_match_results,
)
from data.sport_rules import normalize_sport  # noqa: E402
from features.matchup_features import build_fixture_features, build_training_features  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.matchup_model import (  # noqa: E402
    load_matchup_model,
    predict_matchup_probabilities,
    save_matchup_model,
    train_matchup_model,
)
from quality.matchup_data_quality import assign_prediction_data_quality  # noqa: E402
from reports.matchup_prediction_report import build_today_predictions_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-odds matchup predictions.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--fixtures-path", required=True)
    parser.add_argument("--injuries-path", default=None)
    parser.add_argument("--sport", default=None)
    parser.add_argument("--league", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--aliases-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _filter_league(df: pd.DataFrame, league: str | None) -> pd.DataFrame:
    if not league or "league" not in df.columns:
        return df
    return df[df["league"].astype(str).str.lower() == league.lower()].copy()


def _backtested_sports(output_dir: Path) -> set[str]:
    path = output_dir / "matchup_model_backtest.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    return {normalize_sport(s) for s in (data.get("accuracy_by_sport") or {}).keys()}


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = {"aliases_path": args.aliases_path} if args.aliases_path else {}
    if args.sport:
        config["default_sport"] = normalize_sport(args.sport)

    output_dir = Path(args.output_dir)

    results = normalize_match_results(load_match_results(args.results_path), config)
    fixtures = normalize_fixtures(load_fixtures(args.fixtures_path), config)
    injuries = (
        normalize_injuries(load_injuries(args.injuries_path), config)
        if args.injuries_path
        else None
    )

    # Optional filters.
    if args.sport:
        target = normalize_sport(args.sport)
        results = results[results["sport"] == target].copy()
        fixtures = fixtures[fixtures["sport"] == target].copy()
    results = _filter_league(results, args.league)
    fixtures = _filter_league(fixtures, args.league)

    if fixtures.empty:
        print("No fixtures to predict after filtering. Nothing to do.")
        return

    # As-of cutoff so we only ever learn from games before the prediction date.
    # Without an explicit --as-of-date we fall back to the earliest fixture date,
    # which keeps the features leakage-safe even if the results file happens to
    # contain games scheduled on/after the fixtures.
    cutoff = (
        pd.to_datetime(args.as_of_date)
        if args.as_of_date
        else pd.to_datetime(fixtures["game_date"], errors="coerce").min()
    )
    if pd.notna(cutoff):
        results = results[pd.to_datetime(results["game_date"]) < cutoff].copy()
    if results.empty:
        print("No historical results available to train on. Cannot predict.")
        return

    backtested = _backtested_sports(output_dir)

    # Train (or load) one model per sport present in the fixtures.
    preloaded = None
    if args.model_path and Path(args.model_path).exists():
        preloaded = load_matchup_model(args.model_path)

    all_predictions: list[pd.DataFrame] = []
    for sport in sorted(fixtures["sport"].unique()):
        sport_results = results[results["sport"] == sport]
        sport_fixtures = fixtures[fixtures["sport"] == sport]
        if sport_results.empty:
            print(f"Skipping {sport}: no historical results for this sport.")
            continue

        if preloaded is not None and preloaded.get("sport") == sport:
            bundle = preloaded
        else:
            training = build_training_features(sport_results, injuries, config)
            bundle = train_matchup_model(training, sport, config)
            if args.model_path and len(fixtures["sport"].unique()) == 1:
                save_matchup_model(bundle, args.model_path)

        fixture_features = build_fixture_features(sport_fixtures, sport_results, injuries, config)
        # Carry venue/status through for honest data-quality warnings.
        fixture_features = fixture_features.merge(
            sport_fixtures[["fixture_id", "venue"]], on="fixture_id", how="left"
        )
        fixture_features["data_quality"] = fixture_features.apply(
            assign_prediction_data_quality, axis=1
        )
        fixture_features["model_backtested"] = sport in backtested

        predictions = predict_matchup_probabilities(bundle, fixture_features)
        all_predictions.append(predictions)

    if not all_predictions:
        print("No predictions produced (no sport had both results and fixtures).")
        return

    predictions = pd.concat(all_predictions, ignore_index=True)
    build_today_predictions_report(predictions, output_dir)

    print(f"Built {len(predictions)} matchup prediction(s).")
    print(f"  {output_dir / 'matchup_predictions_today.csv'}")
    print(f"  {output_dir / 'matchup_predictions_today.json'}")
    print("These are model-implied probabilities, not betting odds.")


if __name__ == "__main__":
    main()
