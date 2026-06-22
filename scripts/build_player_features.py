"""Build pregame-safe player availability features for each NBA game."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.player_client import load_raw_player_logs  # noqa: E402
from features.player_features import PLAYER_BASE_FEATURE_COLUMNS, build_player_game_features, normalize_player_logs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pregame-safe player features from free NBA player box score history.")
    parser.add_argument("--games-path", default=str(PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv"))
    parser.add_argument("--player-cache-dir", default=str(PROJECT_ROOT / "data" / "raw" / "nba" / "player"))
    parser.add_argument("--features-output", default=str(PROJECT_ROOT / "outputs" / "player_features_by_game.csv"))
    parser.add_argument("--audit-output", default=str(PROJECT_ROOT / "outputs" / "player_availability_audit.csv"))
    parser.add_argument("--summary-output", default=str(PROJECT_ROOT / "outputs" / "player_features_summary.json"))
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _load_games(path: Path) -> pd.DataFrame:
    games = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    required = ["game_id", "game_date", "season", "home_team_abbr", "away_team_abbr"]
    missing = [column for column in required if column not in games.columns]
    if missing:
        raise SystemExit(f"Games file is missing required columns: {missing}")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.dropna(subset=["game_date"]).copy()
    return games


def _add_team_ids_from_player_logs(games: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_player_logs(player_logs)
    if normalized.empty:
        output = games.copy()
        output["home_team_id"] = pd.NA
        output["away_team_id"] = pd.NA
        return output

    same_game_map = (
        normalized[["game_id", "team_abbr", "team_id"]]
        .dropna()
        .drop_duplicates()
        .assign(team_id=lambda frame: pd.to_numeric(frame["team_id"], errors="coerce"))
    )
    by_team = (
        normalized[["team_abbr", "team_id"]]
        .dropna()
        .drop_duplicates()
        .groupby("team_abbr")["team_id"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0])
        .to_dict()
    )
    output = games.copy()
    home_ids = output[["game_id", "home_team_abbr"]].merge(
        same_game_map.rename(columns={"team_abbr": "home_team_abbr", "team_id": "home_team_id"}),
        on=["game_id", "home_team_abbr"],
        how="left",
    )["home_team_id"]
    away_ids = output[["game_id", "away_team_abbr"]].merge(
        same_game_map.rename(columns={"team_abbr": "away_team_abbr", "team_id": "away_team_id"}),
        on=["game_id", "away_team_abbr"],
        how="left",
    )["away_team_id"]
    output["home_team_id"] = home_ids.fillna(output["home_team_abbr"].map(by_team))
    output["away_team_id"] = away_ids.fillna(output["away_team_abbr"].map(by_team))
    return output


def _uncertainty_label(row: pd.Series, side: str) -> str:
    prior_games = row.get(f"{side}_player_prior_games_last10")
    top8_share = row.get(f"{side}_player_top8_available_last_game_share")
    key_gap = row.get(f"{side}_player_key_absence_minutes_last_game")
    if pd.isna(prior_games) or float(prior_games) < 3:
        return "high"
    if pd.isna(top8_share) or float(top8_share) < 0.5 or (pd.notna(key_gap) and float(key_gap) >= 80):
        return "high"
    if float(top8_share) < 0.75 or (pd.notna(key_gap) and float(key_gap) >= 35):
        return "medium"
    return "low"


def _derive_requested_columns(features: pd.DataFrame) -> pd.DataFrame:
    output = features.copy()
    for side in ["home", "away"]:
        prefix = f"{side}_"
        output[f"{side}_expected_active_players"] = output[f"{prefix}player_active_count_last5"]
        output[f"{side}_expected_top8_rotation"] = output[f"{prefix}player_top8_games_played_share_last10"] * 8.0
        output[f"{side}_expected_top5_minutes_total"] = output[f"{prefix}player_top5_minutes_last10"]
        output[f"{side}_expected_points_from_active_rotation"] = output[f"{prefix}player_top8_points_last10"]
        output[f"{side}_expected_rebounds_from_active_rotation"] = output[f"{prefix}player_top8_reb_last10"]
        output[f"{side}_expected_assists_from_active_rotation"] = output[f"{prefix}player_top8_ast_last10"]
        output[f"{side}_expected_plus_minus_from_active_rotation"] = output[f"{prefix}player_top8_plus_minus_last10"]
        output[f"{side}_missing_key_players_count"] = (
            (1.0 - output[f"{prefix}player_top8_available_last_game_share"]).clip(0, 1) * 8.0
        ).round()
        output[f"{side}_missing_top3_minutes_players_count"] = (
            (1.0 - output[f"{prefix}player_top3_available_last_game_share"]).clip(0, 1) * 3.0
        ).round()
        output[f"{side}_missing_top5_minutes_players_count"] = np.maximum(
            output[f"{side}_missing_top3_minutes_players_count"],
            (output[f"{side}_player_top8_minutes_gap_last_game"] / 28.0).clip(lower=0, upper=5).round(),
        )
        output[f"{side}_projected_rotation_available"] = output[f"{prefix}player_prior_games_last10"].ge(3)
        output[f"{side}_missing_key_player_uncertainty"] = output.apply(_uncertainty_label, axis=1, side=side)

    output["expected_top5_minutes_total_diff"] = output["home_expected_top5_minutes_total"] - output["away_expected_top5_minutes_total"]
    output["missing_key_players_count_diff"] = output["home_missing_key_players_count"] - output["away_missing_key_players_count"]
    output["expected_points_from_active_rotation_diff"] = (
        output["home_expected_points_from_active_rotation"] - output["away_expected_points_from_active_rotation"]
    )
    output["expected_plus_minus_from_active_rotation_diff"] = (
        output["home_expected_plus_minus_from_active_rotation"] - output["away_expected_plus_minus_from_active_rotation"]
    )

    output["player_data_available"] = output["home_player_prior_games_last10"].notna() & output["away_player_prior_games_last10"].notna()
    output["projected_rotation_available"] = output["home_projected_rotation_available"] & output["away_projected_rotation_available"]
    uncertainty_rank = {"low": 0, "medium": 1, "high": 2}
    output["missing_key_player_uncertainty"] = output.apply(
        lambda row: max(
            str(row["home_missing_key_player_uncertainty"]),
            str(row["away_missing_key_player_uncertainty"]),
            key=lambda label: uncertainty_rank.get(label, 2),
        ),
        axis=1,
    )
    return output


def _audit(features: pd.DataFrame, raw_player_logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in features.groupby("season", dropna=False):
        rows.append(
            {
                "season": season,
                "games": int(group["game_id"].nunique()),
                "player_data_available_games": int(group["player_data_available"].sum()),
                "projected_rotation_available_games": int(group["projected_rotation_available"].sum()),
                "player_data_coverage": float(group["player_data_available"].mean()) if len(group) else np.nan,
                "projected_rotation_coverage": float(group["projected_rotation_available"].mean()) if len(group) else np.nan,
                "high_uncertainty_games": int(group["missing_key_player_uncertainty"].eq("high").sum()),
                "medium_uncertainty_games": int(group["missing_key_player_uncertainty"].eq("medium").sum()),
                "low_uncertainty_games": int(group["missing_key_player_uncertainty"].eq("low").sum()),
            }
        )
    audit = pd.DataFrame(rows)
    audit["raw_player_log_rows"] = len(raw_player_logs)
    audit["leakage_check"] = "features_use_only_player_games_before_current_game_date"
    return audit


def main() -> None:
    args = parse_args()
    games = _load_games(Path(args.games_path))
    player_logs = load_raw_player_logs(args.player_cache_dir)
    if player_logs.empty:
        raise SystemExit(f"No player logs found in {args.player_cache_dir}. Run scripts/download_nba_player_data.py first.")

    games_with_ids = _add_team_ids_from_player_logs(games, player_logs)
    missing_team_ids = int(games_with_ids[["home_team_id", "away_team_id"]].isna().any(axis=1).sum())
    if missing_team_ids:
        print(f"WARNING: {missing_team_ids:,} games are missing one or both team IDs for player feature matching.")

    base_features = build_player_game_features(games_with_ids, player_logs)
    features = games[["game_id", "game_date", "season", "home_team_abbr", "away_team_abbr"]].merge(
        base_features,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    features = _derive_requested_columns(features)
    audit = _audit(features, player_logs)

    feature_path = Path(args.features_output)
    audit_path = Path(args.audit_output)
    summary_path = Path(args.summary_output)
    for path in [feature_path, audit_path, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False)
    audit.to_csv(audit_path, index=False)

    summary = {
        "status": "ready",
        "games": int(features["game_id"].nunique()),
        "raw_player_log_rows": int(len(player_logs)),
        "player_feature_columns": len([column for column in features.columns if "player_" in column or "expected_" in column]),
        "player_data_coverage": float(features["player_data_available"].mean()),
        "projected_rotation_coverage": float(features["projected_rotation_available"].mean()),
        "missing_team_id_games": missing_team_ids,
        "leakage_checks": {
            "only_prior_player_games_used": True,
            "current_game_box_score_not_used": True,
            "final_score_not_used_as_player_feature": True,
            "postgame_injury_information_not_used": True,
        },
        "free_data_only": True,
        "source": str(Path(args.player_cache_dir)),
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    print(f"Built player features for {features['game_id'].nunique():,} games.")
    print(f"Player data coverage: {features['player_data_available'].mean() * 100.0:.1f}%")
    print(f"Projected rotation coverage: {features['projected_rotation_available'].mean() * 100.0:.1f}%")
    print(f"Saved player features to: {feature_path}")
    print(f"Saved player availability audit to: {audit_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
