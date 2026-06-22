"""Market-specific prediction engines for spreads and totals."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.seasons import TRAIN_START_SEASON
from models.train_model import available_feature_columns


TARGET_MARGIN_COLUMN = "target_home_margin"
TARGET_TOTAL_COLUMN = "target_total_points"
DEFAULT_MARGIN_THRESHOLDS = [-15.5, -10.5, -7.5, -5.5, -3.5, -1.5, 0.0, 1.5, 3.5, 5.5, 7.5, 10.5, 15.5]
DEFAULT_TOTAL_LINES = [200.5, 205.5, 210.5, 215.5, 220.5, 225.5, 230.5, 235.5, 240.5, 245.5]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def normal_cdf(value: float) -> float:
    """Standard normal CDF without requiring scipy."""

    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def probability_margin_exceeds(
    predicted_margin: float,
    residual_std: float,
    margin_threshold: float,
) -> float:
    """Return P(home margin exceeds a threshold)."""

    if residual_std <= 0:
        return float(predicted_margin > margin_threshold)
    z_score = (margin_threshold - predicted_margin) / residual_std
    return 1.0 - normal_cdf(z_score)


def probability_home_spread_covers(
    predicted_margin: float,
    residual_std: float,
    home_spread_line: float,
) -> float:
    """Return P(home team covers a conventional spread line.

    A home spread of -4.5 means the home team must win by more than 4.5.
    A home spread of +4.5 means the home team can lose by fewer than 4.5.
    """

    return probability_margin_exceeds(predicted_margin, residual_std, -home_spread_line)


def probability_total_over(
    predicted_total: float,
    residual_std: float,
    total_line: float,
) -> float:
    """Return P(total points goes over the market line)."""

    if residual_std <= 0:
        return float(predicted_total > total_line)
    z_score = (total_line - predicted_total) / residual_std
    return 1.0 - normal_cdf(z_score)


def prepare_margin_total_dataset(
    modeling_df: pd.DataFrame,
    games_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach final scores and create spread/total targets."""

    output = modeling_df.copy()
    if {"home_points", "away_points"}.issubset(output.columns):
        pass
    elif games_df is not None and {"game_id", "home_points", "away_points"}.issubset(games_df.columns):
        scores = games_df[["game_id", "home_points", "away_points"]].copy()
        scores["game_id"] = scores["game_id"].astype(str)
        output["game_id"] = output["game_id"].astype(str)
        output = output.merge(scores.drop_duplicates("game_id"), on="game_id", how="left")
    else:
        raise ValueError("Final scores are required to train margin and total models.")

    output["home_points"] = pd.to_numeric(output["home_points"], errors="coerce")
    output["away_points"] = pd.to_numeric(output["away_points"], errors="coerce")
    output = output[output["home_points"].notna() & output["away_points"].notna()].copy()
    output[TARGET_MARGIN_COLUMN] = output["home_points"] - output["away_points"]
    output[TARGET_TOTAL_COLUMN] = output["home_points"] + output["away_points"]
    return output.reset_index(drop=True)


def make_regression_model(model_type: str = "ridge", random_seed: int = 42) -> Pipeline:
    """Create a regression model for margin or total points."""

    if model_type == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=random_seed)),
            ]
        )
    if model_type == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.04,
                        l2_regularization=0.05,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown market-type model: {model_type}")


def evaluate_regression(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return regression metrics for a spread/total target."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    residuals = y_true_array - y_pred_array
    return {
        "rows": int(len(y_true_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "mean_error": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
    }


def _fit_predict_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_type: str,
    random_seed: int,
) -> tuple[np.ndarray, float]:
    model = make_regression_model(model_type=model_type, random_seed=random_seed)
    model.fit(train[feature_columns], train[target_column])
    train_predictions = model.predict(train[feature_columns])
    residual_std = float(np.std(train[target_column].to_numpy(dtype=float) - train_predictions, ddof=1))
    predictions = model.predict(test[feature_columns])
    return predictions, residual_std


def walk_forward_margin_total_models(
    modeling_df: pd.DataFrame,
    games_df: pd.DataFrame | None = None,
    train_start_season: int = TRAIN_START_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    model_type: str = "ridge",
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run expanding walk-forward predictions for margin and total points."""

    dataset = prepare_margin_total_dataset(modeling_df, games_df=games_df)
    dataset = dataset.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    features = available_feature_columns(dataset, feature_columns)
    seasons = sorted(int(season) for season in dataset["season"].dropna().unique())
    if first_test_season is None:
        first_test_season = train_start_season + 2
    if last_test_season is None:
        last_test_season = max(seasons)

    prediction_frames: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for test_season in seasons:
        if test_season < first_test_season or test_season > last_test_season:
            continue
        train = dataset[(dataset["season"] >= train_start_season) & (dataset["season"] < test_season)].copy()
        test = dataset[dataset["season"] == test_season].copy()
        if train.empty or test.empty:
            continue

        margin_pred, margin_std = _fit_predict_target(
            train, test, features, TARGET_MARGIN_COLUMN, model_type, random_seed
        )
        total_pred, total_std = _fit_predict_target(
            train, test, features, TARGET_TOTAL_COLUMN, model_type, random_seed
        )

        fold_predictions = test[
            [
                "game_id",
                "game_date",
                "season",
                "season_type",
                "home_team_abbr",
                "away_team_abbr",
                "home_points",
                "away_points",
                TARGET_MARGIN_COLUMN,
                TARGET_TOTAL_COLUMN,
            ]
        ].copy()
        fold_predictions["pred_home_margin"] = margin_pred
        fold_predictions["pred_total_points"] = total_pred
        fold_predictions["margin_residual"] = fold_predictions[TARGET_MARGIN_COLUMN] - margin_pred
        fold_predictions["total_residual"] = fold_predictions[TARGET_TOTAL_COLUMN] - total_pred
        fold_predictions["margin_residual_std_train"] = margin_std
        fold_predictions["total_residual_std_train"] = total_std
        fold_predictions["prob_home_win_from_margin"] = [
            probability_margin_exceeds(pred, margin_std, 0.0) for pred in margin_pred
        ]
        prediction_frames.append(fold_predictions)

        folds.append(
            {
                "test_season": test_season,
                "train_start_season": train_start_season,
                "train_end_season": test_season - 1,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "margin": evaluate_regression(test[TARGET_MARGIN_COLUMN], margin_pred),
                "total": evaluate_regression(test[TARGET_TOTAL_COLUMN], total_pred),
                "train_margin_residual_std": margin_std,
                "train_total_residual_std": total_std,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = {
        "model_type": model_type,
        "features": features,
        "folds": folds,
        "overall": {
            "margin": evaluate_regression(predictions[TARGET_MARGIN_COLUMN], predictions["pred_home_margin"])
            if not predictions.empty
            else {},
            "total": evaluate_regression(predictions[TARGET_TOTAL_COLUMN], predictions["pred_total_points"])
            if not predictions.empty
            else {},
        },
        "num_predictions": int(len(predictions)),
    }
    return predictions, metrics


def train_final_margin_total_models(
    modeling_df: pd.DataFrame,
    games_df: pd.DataFrame | None = None,
    model_type: str = "ridge",
    random_seed: int = 42,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train final margin and total models on all completed games."""

    dataset = prepare_margin_total_dataset(modeling_df, games_df=games_df)
    features = available_feature_columns(dataset, feature_columns)
    margin_model = make_regression_model(model_type=model_type, random_seed=random_seed)
    total_model = make_regression_model(model_type=model_type, random_seed=random_seed)
    margin_model.fit(dataset[features], dataset[TARGET_MARGIN_COLUMN])
    total_model.fit(dataset[features], dataset[TARGET_TOTAL_COLUMN])

    margin_pred = margin_model.predict(dataset[features])
    total_pred = total_model.predict(dataset[features])
    return {
        "model_type": model_type,
        "feature_columns": features,
        "target_columns": [TARGET_MARGIN_COLUMN, TARGET_TOTAL_COLUMN],
        "margin_model": clone(margin_model).fit(dataset[features], dataset[TARGET_MARGIN_COLUMN]),
        "total_model": clone(total_model).fit(dataset[features], dataset[TARGET_TOTAL_COLUMN]),
        "margin_residual_std": float(np.std(dataset[TARGET_MARGIN_COLUMN].to_numpy(dtype=float) - margin_pred, ddof=1)),
        "total_residual_std": float(np.std(dataset[TARGET_TOTAL_COLUMN].to_numpy(dtype=float) - total_pred, ddof=1)),
        "final_fit_rows": int(len(dataset)),
    }


def build_market_type_probability_grid(
    predictions: pd.DataFrame,
    margin_thresholds: list[float] | None = None,
    total_lines: list[float] | None = None,
) -> pd.DataFrame:
    """Create historical binary events for calibrating spread and total engines."""

    if predictions.empty:
        return pd.DataFrame()
    margin_thresholds = margin_thresholds or DEFAULT_MARGIN_THRESHOLDS
    total_lines = total_lines or DEFAULT_TOTAL_LINES
    rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        for threshold in margin_thresholds:
            rows.append(
                {
                    "game_id": row.game_id,
                    "game_date": row.game_date,
                    "season": row.season,
                    "market_type": "spread_handicap",
                    "line_value": threshold,
                    "condition": "home_margin_gt_line",
                    "predicted_prob": probability_margin_exceeds(
                        float(row.pred_home_margin),
                        float(row.margin_residual_std_train),
                        float(threshold),
                    ),
                    "actual_result": float(row.target_home_margin) > float(threshold),
                }
            )
        for total_line in total_lines:
            rows.append(
                {
                    "game_id": row.game_id,
                    "game_date": row.game_date,
                    "season": row.season,
                    "market_type": "total_points_over_under",
                    "line_value": total_line,
                    "condition": "total_points_gt_line",
                    "predicted_prob": probability_total_over(
                        float(row.pred_total_points),
                        float(row.total_residual_std_train),
                        float(total_line),
                    ),
                    "actual_result": float(row.target_total_points) > float(total_line),
                }
            )
    return pd.DataFrame(rows)


def summarize_market_type_calibration(
    probability_grid: pd.DataFrame,
    bin_count: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize calibration by market type and probability bucket."""

    if probability_grid.empty:
        return pd.DataFrame(), {"rows": 0, "market_type_counts": {}}
    working = probability_grid.copy()
    working["actual_result"] = working["actual_result"].astype(int)
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    working["probability_bin"] = pd.cut(
        working["predicted_prob"].clip(0.0, 1.0),
        bins=bins,
        include_lowest=True,
    ).astype(str)
    grouped = (
        working.groupby(["market_type", "probability_bin"], observed=False)
        .agg(
            rows=("actual_result", "size"),
            avg_predicted_prob=("predicted_prob", "mean"),
            observed_rate=("actual_result", "mean"),
        )
        .reset_index()
    )
    grouped["calibration_error"] = grouped["avg_predicted_prob"] - grouped["observed_rate"]
    grouped["abs_calibration_error"] = grouped["calibration_error"].abs()

    summaries = []
    for market_type, frame in working.groupby("market_type"):
        brier = float(np.mean((frame["predicted_prob"].to_numpy() - frame["actual_result"].to_numpy()) ** 2))
        calibration_frame = grouped[grouped["market_type"].eq(market_type)]
        weighted_abs_error = float(
            np.average(
                calibration_frame["abs_calibration_error"],
                weights=calibration_frame["rows"],
            )
        )
        summaries.append(
            {
                "market_type": str(market_type),
                "rows": int(len(frame)),
                "brier_score": brier,
                "weighted_abs_calibration_error": weighted_abs_error,
                "avg_predicted_prob": float(frame["predicted_prob"].mean()),
                "observed_rate": float(frame["actual_result"].mean()),
            }
        )

    return grouped, {
        "rows": int(len(working)),
        "market_type_counts": {
            str(key): int(value) for key, value in working["market_type"].value_counts().to_dict().items()
        },
        "by_market_type": summaries,
        "note": "Spread and total calibration uses historical common line grids until actual Kalshi non-winner lines are available.",
    }


def train_market_type_models_and_save(
    modeling_path: str | Path,
    games_path: str | Path,
    model_output_path: str | Path,
    predictions_output_path: str | Path,
    metrics_output_path: str | Path,
    calibration_output_path: str | Path | None = None,
    calibration_summary_output_path: str | Path | None = None,
    train_start_season: int = TRAIN_START_SEASON,
    first_test_season: int | None = None,
    last_test_season: int | None = None,
    model_type: str = "ridge",
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Train spread/total engines, save predictions, model bundle, and metrics."""

    modeling_df = pd.read_parquet(modeling_path)
    games_df = pd.read_parquet(games_path)
    predictions, metrics = walk_forward_margin_total_models(
        modeling_df=modeling_df,
        games_df=games_df,
        train_start_season=train_start_season,
        first_test_season=first_test_season,
        last_test_season=last_test_season,
        model_type=model_type,
        random_seed=random_seed,
    )
    bundle = train_final_margin_total_models(
        modeling_df=modeling_df,
        games_df=games_df,
        model_type=model_type,
        random_seed=random_seed,
    )

    model_output = Path(model_output_path)
    predictions_output = Path(predictions_output_path)
    metrics_output = Path(metrics_output_path)
    calibration_output = Path(calibration_output_path) if calibration_output_path else None
    calibration_summary_output = Path(calibration_summary_output_path) if calibration_summary_output_path else None
    model_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, model_output)
    predictions.to_csv(predictions_output, index=False)
    probability_grid = build_market_type_probability_grid(predictions)
    calibration, calibration_summary = summarize_market_type_calibration(probability_grid)
    metrics["calibration"] = calibration_summary
    if calibration_output is not None:
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration.to_csv(calibration_output, index=False)
    if calibration_summary_output is not None:
        calibration_summary_output.parent.mkdir(parents=True, exist_ok=True)
        calibration_summary_output.write_text(json.dumps(_json_safe(calibration_summary), indent=2), encoding="utf-8")
    metrics_output.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    return bundle, predictions, metrics
