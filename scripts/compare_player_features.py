"""Compare team-only and player-aware models on the same time split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import load_config, resolve_project_path  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from models.train_model import (  # noqa: E402
    BASELINE_FEATURE_COLUMNS,
    DEFAULT_FEATURE_COLUMNS,
    RICH_TEAM_FORM_FEATURE_COLUMNS,
    available_feature_columns,
    train_models,
)
from models.walk_forward import walk_forward_predict  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare team-only and player-aware model features.")
    parser.add_argument("--modeling-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    modeling_path = (
        Path(args.modeling_path)
        if args.modeling_path
        else resolve_project_path(config.data.processed_dir) / "modeling_dataset.parquet"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else PROJECT_ROOT / "data" / "reports" / "player_feature_comparison.json"
    )

    modeling = pd.read_parquet(modeling_path)
    team_only_features = available_feature_columns(
        modeling,
        BASELINE_FEATURE_COLUMNS + RICH_TEAM_FORM_FEATURE_COLUMNS,
    )
    player_aware_features = available_feature_columns(
        modeling,
        [column for column in DEFAULT_FEATURE_COLUMNS if column in modeling.columns],
    )

    _, team_only_metrics, _ = train_models(
        modeling,
        train_start_season=config.model.train_start_season,
        train_end_season=config.model.train_end_season,
        test_season=config.model.test_season,
        random_seed=config.project.random_seed,
        feature_columns=team_only_features,
    )
    _, player_aware_metrics, _ = train_models(
        modeling,
        train_start_season=config.model.train_start_season,
        train_end_season=config.model.train_end_season,
        test_season=config.model.test_season,
        random_seed=config.project.random_seed,
        feature_columns=player_aware_features,
    )
    _, team_only_walk = walk_forward_predict(
        modeling,
        target_column=config.model.target,
        train_start_season=config.model.train_start_season,
        model_type=config.model.model_type,
        random_seed=config.project.random_seed,
        feature_columns=team_only_features,
    )
    _, player_aware_walk = walk_forward_predict(
        modeling,
        target_column=config.model.target,
        train_start_season=config.model.train_start_season,
        model_type=config.model.model_type,
        random_seed=config.project.random_seed,
        feature_columns=player_aware_features,
    )

    team_best = team_only_metrics["best_model"]
    player_best = player_aware_metrics["best_model"]
    team_best_metrics = team_only_metrics["models"][team_best]
    player_best_metrics = player_aware_metrics["models"][player_best]
    deltas = {
        "accuracy": player_best_metrics.get("accuracy", 0) - team_best_metrics.get("accuracy", 0),
        "brier_score": player_best_metrics.get("brier_score", 0) - team_best_metrics.get("brier_score", 0),
        "log_loss": player_best_metrics.get("log_loss", 0) - team_best_metrics.get("log_loss", 0),
        "roc_auc": player_best_metrics.get("roc_auc", 0) - team_best_metrics.get("roc_auc", 0),
    }
    team_walk_metrics = team_only_walk["overall"]["model"]
    player_walk_metrics = player_aware_walk["overall"]["model"]
    walk_deltas = {
        "accuracy": player_walk_metrics.get("accuracy", 0) - team_walk_metrics.get("accuracy", 0),
        "brier_score": player_walk_metrics.get("brier_score", 0) - team_walk_metrics.get("brier_score", 0),
        "log_loss": player_walk_metrics.get("log_loss", 0) - team_walk_metrics.get("log_loss", 0),
        "roc_auc": player_walk_metrics.get("roc_auc", 0) - team_walk_metrics.get("roc_auc", 0),
    }
    payload = {
        "modeling_rows": int(len(modeling)),
        "single_split": {
            "team_only": {
                "best_model": team_best,
                "feature_count": len(team_only_features),
                "metrics": team_best_metrics,
            },
            "player_aware": {
                "best_model": player_best,
                "feature_count": len(player_aware_features),
                "metrics": player_best_metrics,
            },
            "player_minus_team_only": deltas,
        },
        "walk_forward": {
            "model_type": config.model.model_type,
            "num_predictions": player_aware_walk["num_predictions"],
            "team_only": {
                "feature_count": len(team_only_features),
                "metrics": team_walk_metrics,
            },
            "player_aware": {
                "feature_count": len(player_aware_features),
                "metrics": player_walk_metrics,
            },
            "player_minus_team_only": walk_deltas,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"Team-only best {team_best}: log_loss={team_best_metrics.get('log_loss'):.4f}, auc={team_best_metrics.get('roc_auc'):.4f}")
    print(f"Player-aware best {player_best}: log_loss={player_best_metrics.get('log_loss'):.4f}, auc={player_best_metrics.get('roc_auc'):.4f}")
    print(f"Walk-forward team-only: log_loss={team_walk_metrics.get('log_loss'):.4f}, auc={team_walk_metrics.get('roc_auc'):.4f}")
    print(f"Walk-forward player-aware: log_loss={player_walk_metrics.get('log_loss'):.4f}, auc={player_walk_metrics.get('roc_auc'):.4f}")
    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
