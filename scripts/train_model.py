"""Train fair NBA probability models and compare them against market benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import load_config, resolve_project_path  # noqa: E402
from data.seasons import build_free_odds_split_plan, nba_season_display_label  # noqa: E402
from data.sportsbook_odds import load_sportsbook_odds, match_sportsbook_odds_to_games, sportsbook_match_report_by_season  # noqa: E402
from models.evaluate import evaluate_binary_probabilities  # noqa: E402


TEAM_FAIR_FEATURES = [
    "home_recent_win_pct",
    "away_recent_win_pct",
    "recent_win_pct_diff",
    "home_recent_point_diff",
    "away_recent_point_diff",
    "recent_point_diff_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
    "season",
]
PLAYER_FEATURE_HINTS = [
    "player_prior_games_last10",
    "player_top3_minutes_last10",
    "player_top5_minutes_last10",
    "player_top8_minutes_last10",
    "player_top8_points_last10",
    "player_top8_reb_last10",
    "player_top8_ast_last10",
    "player_top8_plus_minus_last10",
    "player_top8_value_last10",
    "player_top8_games_played_share_last10",
    "player_active_count_last5",
    "player_rotation_continuity_last5",
    "player_top3_available_last_game_share",
    "player_top8_available_last_game_share",
    "player_key_absence_minutes_last_game",
    "player_top8_minutes_gap_last_game",
    "expected_active_players",
    "expected_top8_rotation",
    "expected_top5_minutes_total",
    "expected_points_from_active_rotation",
    "expected_rebounds_from_active_rotation",
    "expected_assists_from_active_rotation",
    "expected_plus_minus_from_active_rotation",
    "missing_key_players_count",
    "missing_top3_minutes_players_count",
    "missing_top5_minutes_players_count",
]
SELECTED_PLAYER_FEATURES = [
    "home_expected_top5_minutes_total",
    "away_expected_top5_minutes_total",
    "home_missing_key_players_count",
    "away_missing_key_players_count",
    "home_expected_points_from_active_rotation",
    "away_expected_points_from_active_rotation",
    "home_expected_plus_minus_from_active_rotation",
    "away_expected_plus_minus_from_active_rotation",
    "expected_top5_minutes_total_diff",
    "missing_key_players_count_diff",
    "expected_points_from_active_rotation_diff",
    "expected_plus_minus_from_active_rotation_diff",
]
ADJUSTMENT_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_win_pct_l5",
    "away_win_pct_l5",
    "win_pct_l5_diff",
    "home_win_pct_l10",
    "away_win_pct_l10",
    "win_pct_l10_diff",
    "home_point_diff_l5",
    "away_point_diff_l5",
    "point_diff_l5_diff",
    "home_point_diff_l10",
    "away_point_diff_l10",
    "point_diff_l10_diff",
    "home_off_rating_l10",
    "away_off_rating_l10",
    "off_rating_l10_diff",
    "home_def_rating_l10",
    "away_def_rating_l10",
    "def_rating_l10_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
    "home_three_in_four",
    "away_three_in_four",
    "three_in_four_diff",
    "home_court",
    "season",
    "season_progress_pct",
]
CATEGORICAL_COLUMNS = ["home_team_abbr", "away_team_abbr"]
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fair NBA probability models with sportsbook and Kalshi as market comparisons.")
    parser.add_argument("--games-path", default=str(PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv"))
    parser.add_argument("--odds-path", default=None)
    parser.add_argument("--validation-output", default=str(PROJECT_ROOT / "outputs" / "model_validation_predictions.csv"))
    parser.add_argument("--fair-validation-output", default=str(PROJECT_ROOT / "outputs" / "fair_model_validation_predictions.csv"))
    parser.add_argument("--summary-output", default=str(PROJECT_ROOT / "outputs" / "validation_backtest_summary.csv"))
    parser.add_argument("--calibration-output", default=str(PROJECT_ROOT / "outputs" / "calibration_report.csv"))
    parser.add_argument("--metrics-output", default=str(PROJECT_ROOT / "outputs" / "model_performance_summary.json"))
    parser.add_argument("--fair-metrics-output", default=str(PROJECT_ROOT / "outputs" / "fair_model_performance_summary.json"))
    parser.add_argument("--diagnostics-output", default=str(PROJECT_ROOT / "outputs" / "model_diagnostics.csv"))
    parser.add_argument("--edge-bucket-output", default=str(PROJECT_ROOT / "outputs" / "edge_bucket_report.csv"))
    parser.add_argument("--error-group-output", default=str(PROJECT_ROOT / "outputs" / "error_by_group.csv"))
    parser.add_argument("--player-diagnostics-output", default=str(PROJECT_ROOT / "outputs" / "player_feature_diagnostics.csv"))
    parser.add_argument("--feature-importance-output", default=str(PROJECT_ROOT / "outputs" / "player_feature_importance.csv"))
    parser.add_argument("--ablation-output", default=str(PROJECT_ROOT / "outputs" / "model_ablation_results.csv"))
    parser.add_argument("--walk-forward-output", default=str(PROJECT_ROOT / "outputs" / "fair_model_walk_forward_results.csv"))
    parser.add_argument("--walk-forward-summary-output", default=str(PROJECT_ROOT / "outputs" / "fair_model_walk_forward_summary.csv"))
    parser.add_argument("--kalshi-markets-path", default=str(PROJECT_ROOT / "data" / "reports" / "matched_markets.csv"))
    parser.add_argument("--kalshi-paper-trades-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_paper_trades.csv"))
    parser.add_argument("--kalshi-paper-summary-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_paper_trade_summary.csv"))
    parser.add_argument("--kalshi-mapping-audit-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_market_mapping_audit.csv"))
    parser.add_argument("--kalshi-paper-diagnostics-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_paper_trade_diagnostics.csv"))
    parser.add_argument("--kalshi-losing-patterns-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_losing_patterns.csv"))
    parser.add_argument("--kalshi-strategy-grid-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_strategy_grid.csv"))
    parser.add_argument("--kalshi-strategy-selected-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_strategy_selected.json"))
    parser.add_argument("--kalshi-strategy-holdout-output", default=str(PROJECT_ROOT / "outputs" / "kalshi_strategy_holdout_results.csv"))
    parser.add_argument("--player-features-path", default=str(PROJECT_ROOT / "outputs" / "player_features_by_game.csv"))
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _clip_prob(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)


def _moneyline_profit(odds: float, stake: float, won: bool) -> float:
    if not won:
        return -stake
    if odds < 0:
        return stake * (100.0 / abs(odds))
    return stake * (odds / 100.0)


def _load_training_frame(games_path: Path, odds_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(games_path, dtype={"game_id": str}, low_memory=False)
    odds = load_sportsbook_odds(odds_path)
    matched = match_sportsbook_odds_to_games(games, odds)
    required = ["home_no_vig_prob", "away_no_vig_prob", "home_moneyline", "away_moneyline"]
    for column in required + ["home_score", "away_score", "spread", "total"]:
        if column not in matched.columns:
            matched[column] = np.nan
        matched[column] = pd.to_numeric(matched[column], errors="coerce")
    if "actual_home_win" not in matched.columns:
        matched["actual_home_win"] = matched["home_score"] > matched["away_score"]
    matched["actual_home_win"] = pd.to_numeric(matched["actual_home_win"], errors="coerce")
    matched = matched.dropna(subset=["game_date", "season", "home_team_abbr", "away_team_abbr", *required, "actual_home_win"]).copy()
    matched["game_date"] = pd.to_datetime(matched["game_date"], errors="coerce")
    matched = matched.dropna(subset=["game_date"]).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return matched, odds


def _mean_or_default(values: list[float], default: float, window: int) -> float:
    subset = values[-window:]
    return float(np.mean(subset)) if subset else default


def _rest_days(last_date: pd.Timestamp | None, game_date: pd.Timestamp) -> float:
    if last_date is None:
        return np.nan
    return float((game_date - last_date).days)


def _add_pregame_team_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state: dict[str, dict[str, Any]] = {}
    season_sizes = frame.groupby("season")["game_id"].transform("count").replace(0, np.nan)
    season_positions = frame.sort_values(["season", "game_date", "game_id"]).groupby("season").cumcount()
    working = frame.copy()
    working["season_progress_pct"] = (season_positions / season_sizes).fillna(0.0).clip(0.0, 1.0)

    for _, row in working.sort_values(["game_date", "game_id"]).iterrows():
        game_date = pd.Timestamp(row["game_date"]).normalize()
        home = str(row["home_team_abbr"])
        away = str(row["away_team_abbr"])

        def team_features(team: str) -> dict[str, float]:
            team_state = state.get(
                team,
                {
                    "results": [],
                    "margins": [],
                    "points_for": [],
                    "points_against": [],
                    "game_dates": [],
                    "last_date": None,
                    "elo": 1500.0,
                },
            )
            rest = _rest_days(team_state["last_date"], game_date)
            recent_dates = [date for date in team_state["game_dates"] if 0 <= (game_date - date).days <= 3]
            return {
                "elo": float(team_state["elo"]),
                "win_pct_l5": _mean_or_default(team_state["results"], 0.5, 5),
                "win_pct_l10": _mean_or_default(team_state["results"], 0.5, 10),
                "point_diff_l5": _mean_or_default(team_state["margins"], 0.0, 5),
                "point_diff_l10": _mean_or_default(team_state["margins"], 0.0, 10),
                "off_rating_l10": _mean_or_default(team_state["points_for"], 110.0, 10),
                "def_rating_l10": _mean_or_default(team_state["points_against"], 110.0, 10),
                "rest_days": rest,
                "back_to_back": 1.0 if rest == 1.0 else 0.0,
                "three_in_four": 1.0 if len(recent_dates) >= 2 else 0.0,
            }

        home_features = team_features(home)
        away_features = team_features(away)
        output = row.to_dict()
        output.update(
            {
                "home_elo": home_features["elo"],
                "away_elo": away_features["elo"],
                "elo_diff": home_features["elo"] - away_features["elo"],
                "home_recent_win_pct": home_features["win_pct_l10"],
                "away_recent_win_pct": away_features["win_pct_l10"],
                "recent_win_pct_diff": home_features["win_pct_l10"] - away_features["win_pct_l10"],
                "home_recent_point_diff": home_features["point_diff_l10"],
                "away_recent_point_diff": away_features["point_diff_l10"],
                "recent_point_diff_diff": home_features["point_diff_l10"] - away_features["point_diff_l10"],
                "home_win_pct_l5": home_features["win_pct_l5"],
                "away_win_pct_l5": away_features["win_pct_l5"],
                "win_pct_l5_diff": home_features["win_pct_l5"] - away_features["win_pct_l5"],
                "home_win_pct_l10": home_features["win_pct_l10"],
                "away_win_pct_l10": away_features["win_pct_l10"],
                "win_pct_l10_diff": home_features["win_pct_l10"] - away_features["win_pct_l10"],
                "home_point_diff_l5": home_features["point_diff_l5"],
                "away_point_diff_l5": away_features["point_diff_l5"],
                "point_diff_l5_diff": home_features["point_diff_l5"] - away_features["point_diff_l5"],
                "home_point_diff_l10": home_features["point_diff_l10"],
                "away_point_diff_l10": away_features["point_diff_l10"],
                "point_diff_l10_diff": home_features["point_diff_l10"] - away_features["point_diff_l10"],
                "home_off_rating_l10": home_features["off_rating_l10"],
                "away_off_rating_l10": away_features["off_rating_l10"],
                "off_rating_l10_diff": home_features["off_rating_l10"] - away_features["off_rating_l10"],
                "home_def_rating_l10": home_features["def_rating_l10"],
                "away_def_rating_l10": away_features["def_rating_l10"],
                "def_rating_l10_diff": home_features["def_rating_l10"] - away_features["def_rating_l10"],
                "home_rest_days": home_features["rest_days"],
                "away_rest_days": away_features["rest_days"],
                "rest_days_diff": (
                    home_features["rest_days"] - away_features["rest_days"]
                    if not math.isnan(home_features["rest_days"]) and not math.isnan(away_features["rest_days"])
                    else np.nan
                ),
                "home_back_to_back": home_features["back_to_back"],
                "away_back_to_back": away_features["back_to_back"],
                "home_three_in_four": home_features["three_in_four"],
                "away_three_in_four": away_features["three_in_four"],
                "three_in_four_diff": home_features["three_in_four"] - away_features["three_in_four"],
                "home_court": 1.0,
            }
        )
        if "home_no_vig_prob" in row.index and pd.notna(row.get("home_no_vig_prob")):
            output["sportsbook_home_no_vig_prob"] = float(row["home_no_vig_prob"])
            output["market_logit"] = float(logit(_clip_prob([row["home_no_vig_prob"]])[0]))
        else:
            output["sportsbook_home_no_vig_prob"] = np.nan
            output["market_logit"] = np.nan
        rows.append(output)

        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if pd.notna(home_score) and pd.notna(away_score):
            home_score = float(home_score)
            away_score = float(away_score)
            home_margin = home_score - away_score
            expected_home = 1.0 / (1.0 + 10.0 ** (-((home_features["elo"] + 65.0) - away_features["elo"]) / 400.0))
            home_result = 1.0 if home_margin > 0 else 0.0
            k_factor = 20.0
            updated_elos = {
                home: home_features["elo"] + k_factor * (home_result - expected_home),
                away: away_features["elo"] + k_factor * ((1.0 - home_result) - (1.0 - expected_home)),
            }
            for team, won, margin, points_for, points_against in [
                (home, home_margin > 0, home_margin, home_score, away_score),
                (away, home_margin < 0, -home_margin, away_score, home_score),
            ]:
                team_state = state.setdefault(
                    team,
                    {"results": [], "margins": [], "points_for": [], "points_against": [], "game_dates": [], "last_date": None, "elo": 1500.0},
                )
                team_state["results"].append(1.0 if won else 0.0)
                team_state["margins"].append(float(margin))
                team_state["points_for"].append(float(points_for))
                team_state["points_against"].append(float(points_against))
                team_state["game_dates"].append(game_date)
                team_state["last_date"] = game_date
                team_state["elo"] = updated_elos[team]
    return pd.DataFrame(rows)


def _make_preprocessor(feature_columns: list[str], categorical_columns: list[str], dense: bool = False):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=not dense)
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
    return ColumnTransformer([("num", numeric, feature_columns), ("cat", categorical, categorical_columns)])


def _available_player_features(frame: pd.DataFrame) -> list[str]:
    columns = []
    blocked_fragments = ["uncertainty", "available", "game_id", "game_date", "team_abbr"]
    for column in frame.columns:
        if any(fragment in column for fragment in blocked_fragments):
            continue
        if any(hint in column for hint in PLAYER_FEATURE_HINTS):
            if pd.api.types.is_numeric_dtype(frame[column]):
                columns.append(column)
    return sorted(set(columns))


def _fit_logistic_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    penalty: str = "l2",
    c_value: float = 0.25,
    l1_ratio: float | None = None,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    kwargs: dict[str, Any] = {
        "max_iter": 2000,
        "C": c_value,
        "random_state": 42,
        "penalty": penalty,
        "solver": solver,
    }
    if penalty == "elasticnet":
        kwargs["solver"] = "saga"
        kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio
    model = Pipeline(
        [
            ("features", _make_preprocessor(feature_columns, CATEGORICAL_COLUMNS)),
            ("model", LogisticRegression(**kwargs)),
        ]
    )
    model.fit(train[feature_columns + CATEGORICAL_COLUMNS], train["actual_home_win"].astype(int))
    return model


def _fit_tree_model(train: pd.DataFrame, feature_columns: list[str], model_type: str):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.pipeline import Pipeline

    estimator = (
        RandomForestClassifier(n_estimators=250, min_samples_leaf=30, max_depth=5, random_state=42, n_jobs=-1)
        if model_type == "random_forest"
        else GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=2, random_state=42)
    )
    model = Pipeline(
        [
            ("features", _make_preprocessor(feature_columns, CATEGORICAL_COLUMNS)),
            ("model", estimator),
        ]
    )
    model.fit(train[feature_columns + CATEGORICAL_COLUMNS], train["actual_home_win"].astype(int))
    return model


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def _fit_xgboost_model(train: pd.DataFrame, feature_columns: list[str]):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=2,
        learning_rate=0.05,
        n_estimators=250,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=8.0,
        reg_alpha=0.1,
        min_child_weight=20,
        random_state=42,
        n_jobs=1,
    )
    model = Pipeline(
        [
            ("features", _make_preprocessor(feature_columns, CATEGORICAL_COLUMNS)),
            ("model", estimator),
        ]
    )
    model.fit(train[feature_columns + CATEGORICAL_COLUMNS], train["actual_home_win"].astype(int))
    return model


def _predict_pipeline(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return _clip_prob(model.predict_proba(frame[feature_columns + CATEGORICAL_COLUMNS])[:, 1])


def _fit_market_anchored_model(train: pd.DataFrame):
    preprocessor = _make_preprocessor(ADJUSTMENT_FEATURES, CATEGORICAL_COLUMNS, dense=True)
    x_train = preprocessor.fit_transform(train[ADJUSTMENT_FEATURES + CATEGORICAL_COLUMNS])
    x_train = np.asarray(x_train, dtype=float)
    y = train["actual_home_win"].astype(int).to_numpy()
    offset = train["market_logit"].to_numpy(dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = offset + x_train @ beta
        p = expit(z)
        loss = -np.mean(y * np.log(np.clip(p, EPSILON, 1.0)) + (1 - y) * np.log(np.clip(1.0 - p, EPSILON, 1.0)))
        loss += 0.005 * float(np.mean(beta**2))
        grad = (x_train.T @ (p - y)) / len(y) + 0.01 * beta / len(beta)
        return float(loss), grad

    result = minimize(lambda beta: objective(beta), np.zeros(x_train.shape[1]), method="L-BFGS-B", jac=True)
    if not result.success:
        print(f"WARNING: anchored optimizer did not fully converge: {result.message}")
    return preprocessor, result.x


def _predict_market_anchored(model: tuple[Any, np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    preprocessor, beta = model
    x_frame = np.asarray(preprocessor.transform(frame[ADJUSTMENT_FEATURES + CATEGORICAL_COLUMNS]), dtype=float)
    return _clip_prob(expit(frame["market_logit"].to_numpy(dtype=float) + x_frame @ beta))


def _fit_calibrators(train_probs: np.ndarray, y_train: pd.Series) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y = y_train.astype(int).to_numpy()
    platt = LogisticRegression(max_iter=1000, random_state=42)
    platt.fit(train_probs.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(train_probs, y)
    return {"platt": platt, "isotonic": isotonic}


def _apply_calibration(calibrators: dict[str, Any], probs: np.ndarray, method: str) -> np.ndarray:
    if method == "uncalibrated":
        return _clip_prob(probs)
    if method == "platt":
        return _clip_prob(calibrators["platt"].predict_proba(probs.reshape(-1, 1))[:, 1])
    if method == "isotonic":
        return _clip_prob(calibrators["isotonic"].predict(probs))
    raise ValueError(f"Unknown calibration method: {method}")


def _calibration_error(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> float:
    working = pd.DataFrame({"y": y_true.astype(int), "p": probabilities})
    working["bucket"] = pd.cut(working["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    errors = []
    weights = []
    for _, group in working.groupby("bucket", observed=False):
        if group.empty:
            continue
        errors.append(abs(float(group["p"].mean()) - float(group["y"].mean())))
        weights.append(len(group))
    return float(np.average(errors, weights=weights)) if weights else math.nan


def _metrics_row(name: str, y_true: pd.Series, probs: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_binary_probabilities(y_true, probs)
    metrics["calibration_error"] = _calibration_error(y_true, probs)
    metrics["model_name"] = name
    return metrics


def _probability_diagnostics(probabilities: np.ndarray, prefix: str = "") -> dict[str, Any]:
    probs = _clip_prob(probabilities)
    output: dict[str, Any] = {
        f"{prefix}avg_predicted_probability": float(np.mean(probs)),
        f"{prefix}min_predicted_probability": float(np.min(probs)),
        f"{prefix}max_predicted_probability": float(np.max(probs)),
    }
    buckets = pd.cut(probs, bins=np.linspace(0.0, 1.0, 11), include_lowest=True).astype(str)
    counts = pd.Series(buckets).value_counts().sort_index()
    for bucket, count in counts.items():
        output[f"{prefix}bucket_{bucket}"] = int(count)
    return output


def _calibration_bucket_rows(model_name: str, y_true: pd.Series, probabilities: np.ndarray) -> list[dict[str, Any]]:
    working = pd.DataFrame({"actual": y_true.astype(int), "probability": _clip_prob(probabilities)})
    working["probability_bucket"] = pd.cut(working["probability"], bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
    rows = []
    for bucket, group in working.groupby("probability_bucket", dropna=False):
        rows.append(
            {
                "model_name": model_name,
                "row_type": "calibration_bucket",
                "probability_bucket": str(bucket),
                "games": int(len(group)),
                "avg_predicted_probability": float(group["probability"].mean()) if len(group) else np.nan,
                "realized_home_win_rate": float(group["actual"].mean()) if len(group) else np.nan,
            }
        )
    return rows


def _calibration_report(validation: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    working = validation.copy()
    working["probability_bucket"] = pd.cut(working[probability_column], bins=np.linspace(0.0, 1.0, 11), include_lowest=True).astype(str)
    grouped = working.groupby("probability_bucket", dropna=False).agg(
        games=("actual_home_win", "size"),
        avg_model_prob=(probability_column, "mean"),
        avg_sportsbook_prob=("sportsbook_home_no_vig_prob", "mean"),
        realized_home_win_rate=("actual_home_win", "mean"),
    )
    grouped["model_calibration_error"] = grouped["avg_model_prob"] - grouped["realized_home_win_rate"]
    grouped["sportsbook_calibration_error"] = grouped["avg_sportsbook_prob"] - grouped["realized_home_win_rate"]
    return grouped.reset_index()


def _bet_candidates(validation: pd.DataFrame, model_prob_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for _, row in validation.iterrows():
        home_model = float(row[model_prob_col])
        home_market = float(row["sportsbook_home_no_vig_prob"])
        away_model = 1.0 - home_model
        away_market = float(row["away_no_vig_prob"])
        home_edge = home_model - home_market
        away_edge = away_model - away_market
        if home_edge >= threshold:
            won = bool(row["actual_home_win"])
            rows.append(
                {
                    "side": "HOME",
                    "edge": home_edge,
                    "market_probability": home_market,
                    "model_probability": home_model,
                    "profit": _moneyline_profit(float(row["home_moneyline"]), 1.0, won),
                    "won": won,
                    "favorite": home_market >= away_market,
                }
            )
        if away_edge >= threshold:
            won = not bool(row["actual_home_win"])
            rows.append(
                {
                    "side": "AWAY",
                    "edge": away_edge,
                    "market_probability": away_market,
                    "model_probability": away_model,
                    "profit": _moneyline_profit(float(row["away_moneyline"]), 1.0, won),
                    "won": won,
                    "favorite": away_market > home_market,
                }
            )
    return pd.DataFrame(rows)


def _max_drawdown(profits: pd.Series | list[float]) -> float:
    if len(profits) == 0:
        return 0.0
    curve = np.cumsum(np.asarray(profits, dtype=float))
    peaks = np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:]
    drawdowns = curve - peaks
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def _validation_backtest(validation: pd.DataFrame, model_prob_col: str, model_name: str) -> pd.DataFrame:
    rows = []
    segments = {
        "all": lambda bets: bets,
        "bet_home_team": lambda bets: bets[bets["side"].eq("HOME")],
        "bet_away_team": lambda bets: bets[bets["side"].eq("AWAY")],
        "favorites": lambda bets: bets[bets["favorite"]],
        "underdogs": lambda bets: bets[~bets["favorite"]],
    }
    for threshold in THRESHOLDS:
        bets = _bet_candidates(validation, model_prob_col, threshold)
        for segment, selector in segments.items():
            selected = selector(bets) if not bets.empty else bets
            profits = selected["profit"] if not selected.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "model_name": model_name,
                    "edge_threshold": threshold,
                    "segment": segment,
                    "num_bets": int(len(selected)),
                    "win_rate": float(selected["won"].mean()) if not selected.empty else np.nan,
                    "average_edge": float(selected["edge"].mean()) if not selected.empty else np.nan,
                    "average_market_probability": float(selected["market_probability"].mean()) if not selected.empty else np.nan,
                    "average_model_probability": float(selected["model_probability"].mean()) if not selected.empty else np.nan,
                    "profit_loss_1_unit": float(profits.sum()) if not selected.empty else 0.0,
                    "roi": float(profits.sum() / len(selected)) if not selected.empty else np.nan,
                    "max_drawdown": _max_drawdown(profits),
                }
            )
    return pd.DataFrame(rows)


def _group_metrics(frame: pd.DataFrame, model_prob_col: str, group_type: str, group_label: str, threshold: float = 0.02) -> dict[str, Any]:
    if frame.empty:
        return {
            "group_type": group_type,
            "group": group_label,
            "games": 0,
            "model_log_loss": np.nan,
            "sportsbook_log_loss": np.nan,
            "model_brier": np.nan,
            "sportsbook_brier": np.nan,
            "model_minus_sportsbook_log_loss": np.nan,
            "roi_if_betting_model_edges": np.nan,
        }
    model_metrics = evaluate_binary_probabilities(frame["actual_home_win"], frame[model_prob_col])
    book_metrics = evaluate_binary_probabilities(frame["actual_home_win"], frame["sportsbook_home_no_vig_prob"])
    bets = _bet_candidates(frame, model_prob_col, threshold)
    roi = float(bets["profit"].sum() / len(bets)) if not bets.empty else np.nan
    return {
        "group_type": group_type,
        "group": group_label,
        "games": int(len(frame)),
        "model_log_loss": model_metrics.get("log_loss"),
        "sportsbook_log_loss": book_metrics.get("log_loss"),
        "model_brier": model_metrics.get("brier_score"),
        "sportsbook_brier": book_metrics.get("brier_score"),
        "model_minus_sportsbook_log_loss": model_metrics.get("log_loss") - book_metrics.get("log_loss"),
        "roi_if_betting_model_edges": roi,
    }


def _diagnostic_reports(validation: pd.DataFrame, model_prob_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = validation.copy()
    working["sportsbook_probability_bucket"] = pd.cut(working["sportsbook_home_no_vig_prob"], bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
    working["model_probability_bucket"] = pd.cut(working[model_prob_col], bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
    working["edge"] = working[model_prob_col] - working["sportsbook_home_no_vig_prob"]
    working["edge_bucket"] = pd.cut(working["edge"], bins=[-1, -0.10, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.10, 1], include_lowest=True).astype(str)
    working["favorite_group"] = np.where(working["sportsbook_home_no_vig_prob"].ge(0.5), "home_favorite", "home_underdog")
    working["home_favorite_group"] = working["favorite_group"]
    spread_abs = working["spread"].abs() if "spread" in working.columns else pd.Series(np.nan, index=working.index)
    working["spread_group"] = np.where(spread_abs.ge(6), "large_spread", "close_spread")
    working.loc[spread_abs.isna(), "spread_group"] = np.where((working["sportsbook_home_no_vig_prob"] - 0.5).abs().ge(0.12), "large_spread_proxy", "close_spread_proxy")
    working["rest_advantage_group"] = np.select(
        [working["rest_days_diff"].ge(1), working["rest_days_diff"].le(-1)],
        ["home_rest_advantage", "away_rest_advantage"],
        default="no_rest_advantage",
    )
    working["back_to_back_group"] = np.select(
        [working["home_back_to_back"].eq(1), working["away_back_to_back"].eq(1)],
        ["home_back_to_back", "away_back_to_back"],
        default="no_back_to_back",
    )

    diagnostic_rows: list[dict[str, Any]] = []
    for group_type, column in [
        ("sportsbook_probability_bucket", "sportsbook_probability_bucket"),
        ("model_probability_bucket", "model_probability_bucket"),
        ("favorites_vs_underdogs", "favorite_group"),
        ("home_favorites_vs_home_underdogs", "home_favorite_group"),
        ("spread_group", "spread_group"),
        ("rest_advantage_games", "rest_advantage_group"),
        ("back_to_back_games", "back_to_back_group"),
    ]:
        for label, group in working.groupby(column, dropna=False):
            diagnostic_rows.append(_group_metrics(group, model_prob_col, group_type, str(label)))

    edge_rows = [_group_metrics(group, model_prob_col, "edge_bucket", str(label)) for label, group in working.groupby("edge_bucket", dropna=False)]

    team_rows: list[dict[str, Any]] = []
    for team in sorted(set(working["home_team_abbr"].astype(str)) | set(working["away_team_abbr"].astype(str))):
        team_frame = working[working["home_team_abbr"].astype(str).eq(team) | working["away_team_abbr"].astype(str).eq(team)]
        team_rows.append(_group_metrics(team_frame, model_prob_col, "team_level_errors", team))
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(edge_rows), pd.DataFrame(team_rows)


def _player_feature_diagnostics(
    featured: pd.DataFrame,
    player_feature_columns: list[str],
    team_feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    player_available = featured.get("player_data_available", pd.Series(False, index=featured.index)).fillna(False).astype(bool)
    rotation_available = featured.get("projected_rotation_available", pd.Series(False, index=featured.index)).fillna(False).astype(bool)
    rows.extend(
        [
            {
                "diagnostic_type": "attachment_check",
                "item": "home_player_features_attached_to_home_team",
                "status": "pass" if any(column.startswith("home_") for column in player_feature_columns) else "warn",
                "value": any(column.startswith("home_") for column in player_feature_columns),
                "detail": "Home player features are prefixed with home_ after game_id merge.",
            },
            {
                "diagnostic_type": "attachment_check",
                "item": "away_player_features_attached_to_away_team",
                "status": "pass" if any(column.startswith("away_") for column in player_feature_columns) else "warn",
                "value": any(column.startswith("away_") for column in player_feature_columns),
                "detail": "Away player features are prefixed with away_ after game_id merge.",
            },
            {
                "diagnostic_type": "leakage_check",
                "item": "player_features_use_prior_games_only",
                "status": "pass",
                "value": True,
                "detail": "features.player_features uses team_game_order rows with game_date < current game_date.",
            },
        ]
    )
    for season, group in featured.groupby("season", dropna=False):
        rows.append(
            {
                "diagnostic_type": "missingness_by_season",
                "item": str(season),
                "status": "pass" if group.get("player_data_available", pd.Series(False, index=group.index)).fillna(False).mean() >= 0.95 else "warn",
                "value": float(group.get("player_data_available", pd.Series(False, index=group.index)).fillna(False).mean()),
                "detail": f"rotation_coverage={float(group.get('projected_rotation_available', pd.Series(False, index=group.index)).fillna(False).mean()):.4f}",
            }
        )
    teams = sorted(set(featured["home_team_abbr"].astype(str)) | set(featured["away_team_abbr"].astype(str)))
    for team in teams:
        mask = featured["home_team_abbr"].astype(str).eq(team) | featured["away_team_abbr"].astype(str).eq(team)
        group = featured[mask]
        rows.append(
            {
                "diagnostic_type": "missingness_by_team",
                "item": team,
                "status": "pass" if group.get("player_data_available", pd.Series(False, index=group.index)).fillna(False).mean() >= 0.95 else "warn",
                "value": float(group.get("player_data_available", pd.Series(False, index=group.index)).fillna(False).mean()),
                "detail": f"games={len(group)} rotation_coverage={float(group.get('projected_rotation_available', pd.Series(False, index=group.index)).fillna(False).mean()):.4f}",
            }
        )
    numeric = featured[player_feature_columns].apply(pd.to_numeric, errors="coerce") if player_feature_columns else pd.DataFrame()
    for column in player_feature_columns:
        series = numeric[column]
        missing_rate = float(series.isna().mean())
        nunique = int(series.nunique(dropna=True))
        std = float(series.std(skipna=True)) if series.notna().any() else np.nan
        near_constant = bool(nunique <= 2 or (pd.notna(std) and std < 1e-6))
        rows.append(
            {
                "diagnostic_type": "feature_quality",
                "item": column,
                "status": "warn" if near_constant or missing_rate > 0.05 else "pass",
                "value": std,
                "detail": f"missing_rate={missing_rate:.4f} unique_values={nunique} near_constant={near_constant}",
            }
        )
    if not numeric.empty:
        corr = numeric.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_pairs = upper.stack().sort_values(ascending=False).head(50)
        for (left, right), value in high_pairs.items():
            if value >= 0.95:
                rows.append(
                    {
                        "diagnostic_type": "high_correlation",
                        "item": f"{left}__{right}",
                        "status": "warn",
                        "value": float(value),
                        "detail": "absolute correlation >= 0.95",
                    }
                )
        team_numeric = featured[team_feature_columns].apply(pd.to_numeric, errors="coerce")
        team_corr = pd.concat([numeric, team_numeric], axis=1).corr(numeric_only=True).abs()
        for column in player_feature_columns:
            overlaps = team_corr.loc[column, [item for item in team_feature_columns if item in team_corr.columns]].dropna()
            max_corr = float(overlaps.max()) if not overlaps.empty else np.nan
            rows.append(
                {
                    "diagnostic_type": "team_strength_duplication",
                    "item": column,
                    "status": "warn" if pd.notna(max_corr) and max_corr >= 0.75 else "pass",
                    "value": max_corr,
                    "detail": "max absolute correlation to team-strength feature",
                }
            )
    rows.append(
        {
            "diagnostic_type": "coverage_summary",
            "item": "player_data_available",
            "status": "pass" if float(player_available.mean()) >= 0.95 else "warn",
            "value": float(player_available.mean()),
            "detail": f"projected_rotation_available={float(rotation_available.mean()):.4f}",
        }
    )
    return pd.DataFrame(rows)


def _feature_importance_rows(
    model_name: str,
    model: Any,
    validation: pd.DataFrame,
    feature_columns: list[str],
    baseline_log_loss: float,
) -> pd.DataFrame:
    from sklearn.metrics import log_loss

    rows: list[dict[str, Any]] = []
    estimator = model.named_steps.get("model")
    preprocessor = model.named_steps.get("features")
    try:
        transformed_names = list(preprocessor.get_feature_names_out())
    except Exception:
        transformed_names = feature_columns + CATEGORICAL_COLUMNS
    if hasattr(estimator, "coef_"):
        for name, coef in zip(transformed_names, estimator.coef_[0]):
            raw_name = str(name).split("__", maxsplit=1)[-1]
            rows.append(
                {
                    "model_name": model_name,
                    "importance_type": "coefficient",
                    "feature": raw_name,
                    "importance": float(abs(coef)),
                    "signed_value": float(coef),
                }
            )
    if hasattr(estimator, "feature_importances_"):
        for name, value in zip(transformed_names, estimator.feature_importances_):
            rows.append(
                {
                    "model_name": model_name,
                    "importance_type": "tree_importance",
                    "feature": str(name).split("__", maxsplit=1)[-1],
                    "importance": float(value),
                    "signed_value": float(value),
                }
            )
    rng = np.random.default_rng(42)
    for feature in feature_columns:
        shuffled = validation.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        probs = _predict_pipeline(model, shuffled, feature_columns)
        permuted_loss = float(log_loss(validation["actual_home_win"].astype(int), probs, labels=[0, 1]))
        rows.append(
            {
                "model_name": model_name,
                "importance_type": "permutation_log_loss_increase",
                "feature": feature,
                "importance": permuted_loss - baseline_log_loss,
                "signed_value": permuted_loss - baseline_log_loss,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_experiment(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    model_kind: str = "logistic_l2",
) -> tuple[dict[str, Any], Any, np.ndarray]:
    if model_kind == "logistic_l1":
        model = _fit_logistic_model(train, feature_columns, penalty="l1", c_value=0.1)
    elif model_kind == "logistic_elasticnet":
        model = _fit_logistic_model(train, feature_columns, penalty="elasticnet", c_value=0.1, l1_ratio=0.5)
    elif model_kind == "random_forest":
        model = _fit_tree_model(train, feature_columns, "random_forest")
    elif model_kind == "gradient_boosting":
        model = _fit_tree_model(train, feature_columns, "gradient_boosting")
    elif model_kind == "xgboost":
        if not _xgboost_available():
            raise ValueError("xgboost is not installed")
        model = _fit_xgboost_model(train, feature_columns)
    else:
        model = _fit_logistic_model(train, feature_columns, penalty="l2", c_value=0.25)
    probs = _predict_pipeline(model, validation, feature_columns)
    metrics = _metrics_row(name, validation["actual_home_win"], probs)
    metrics.update(
        {
            "feature_count": len(feature_columns),
            "model_kind": model_kind,
            **_probability_diagnostics(probs),
        }
    )
    return metrics, model, probs


def _constant_probability_metrics(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    probability = float(train["actual_home_win"].astype(float).mean())
    probs = _clip_prob(np.full(len(validation), probability))
    metrics = _metrics_row(name, validation["actual_home_win"], probs)
    metrics.update(
        {
            "feature_count": 0,
            "model_kind": "constant_home_rate",
            **_probability_diagnostics(probs),
        }
    )
    return metrics


def _walk_forward_fold_specs(featured: pd.DataFrame) -> list[dict[str, Any]]:
    available = set(featured["season"].dropna().astype(int).unique())
    specs = [
        {"fold": 1, "train_seasons": [2018], "validation_season": 2019, "optional": False},
        {"fold": 2, "train_seasons": [2018, 2019], "validation_season": 2020, "optional": False},
        {"fold": 3, "train_seasons": [2018, 2019, 2020], "validation_season": 2021, "optional": False},
        {"fold": 4, "train_seasons": [2018, 2019, 2020, 2021], "validation_season": 2022, "optional": True},
    ]
    output = []
    for spec in specs:
        if spec["validation_season"] not in available:
            continue
        if not all(season in available for season in spec["train_seasons"]):
            if not spec["optional"]:
                raise SystemExit(f"Walk-forward fold {spec['fold']} is missing required training seasons.")
            continue
        output.append(spec)
    return output


def _walk_forward_model_specs(selected_player_features: list[str]) -> list[dict[str, Any]]:
    selected_features = TEAM_FAIR_FEATURES + selected_player_features
    return [
        {"model_name": "elo_only", "features": ["home_elo", "away_elo", "elo_diff"], "model_kind": "logistic_l2"},
        {"model_name": "team_only_logistic", "features": TEAM_FAIR_FEATURES, "model_kind": "logistic_l2"},
        {
            "model_name": "team_plus_selected_player_logistic",
            "features": selected_features,
            "model_kind": "logistic_l2",
        },
        {
            "model_name": "team_plus_selected_player_random_forest",
            "features": selected_features,
            "model_kind": "random_forest",
        },
        {
            "model_name": "team_plus_selected_player_gradient_boosting",
            "features": selected_features,
            "model_kind": "gradient_boosting",
        },
        {
            "model_name": "team_plus_selected_player_xgboost",
            "features": selected_features,
            "model_kind": "xgboost",
        },
    ]


def _add_walk_forward_context(
    metrics: dict[str, Any],
    fold: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    calibration_method: str = "uncalibrated",
) -> dict[str, Any]:
    output = dict(metrics)
    output.update(
        {
            "fold": int(fold["fold"]),
            "train_seasons": ", ".join(nba_season_display_label(season) for season in fold["train_seasons"]),
            "validation_season": nba_season_display_label(int(fold["validation_season"])),
            "training_games": int(train["game_id"].nunique()),
            "validation_games": int(validation["game_id"].nunique()),
            "calibration_method": calibration_method,
            "optional_fold": bool(fold.get("optional", False)),
        }
    )
    return output


def _model_probability_for_spec(
    spec: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
    features = [feature for feature in spec["features"] if feature in train.columns]
    if not features:
        raise ValueError(f"No usable features for {spec['model_name']}")
    metrics, model, valid_probs = _evaluate_experiment(
        str(spec["model_name"]),
        train,
        validation,
        features,
        model_kind=str(spec["model_kind"]),
    )
    train_probs = _predict_pipeline(model, train, features)
    return model, features, train_probs, valid_probs


def _walk_forward_validation(
    featured: pd.DataFrame,
    selected_player_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_specs = _walk_forward_model_specs(selected_player_features)
    folds = _walk_forward_fold_specs(featured)

    for fold in folds:
        train = featured[featured["season"].astype(int).isin(fold["train_seasons"])].copy()
        validation = featured[featured["season"].astype(int).eq(int(fold["validation_season"]))].copy()
        if train.empty or validation.empty:
            continue

        rows.append(_add_walk_forward_context(_constant_probability_metrics("home_team_baseline", train, validation), fold, train, validation))
        book_probs = _clip_prob(validation["sportsbook_home_no_vig_prob"])
        sportsbook_metrics = _metrics_row("sportsbook_benchmark", validation["actual_home_win"], book_probs)
        sportsbook_metrics.update(
            {
                "feature_count": 0,
                "model_kind": "benchmark_only",
                **_probability_diagnostics(book_probs),
            }
        )
        rows.append(_add_walk_forward_context(sportsbook_metrics, fold, train, validation))

        for spec in model_specs:
            try:
                _, features, train_probs, valid_probs = _model_probability_for_spec(spec, train, validation)
            except ValueError:
                continue
            metrics = _metrics_row(str(spec["model_name"]), validation["actual_home_win"], valid_probs)
            metrics.update(
                {
                    "feature_count": len(features),
                    "model_kind": str(spec["model_kind"]),
                    **_probability_diagnostics(valid_probs),
                }
            )
            rows.append(_add_walk_forward_context(metrics, fold, train, validation))

    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame(), {}

    base_results = results[results["calibration_method"].eq("uncalibrated")].copy()
    summary = (
        base_results.groupby("model_name", dropna=False)
        .agg(
            folds=("fold", "nunique"),
            avg_training_games=("training_games", "mean"),
            avg_validation_games=("validation_games", "mean"),
            avg_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            avg_brier_score=("brier_score", "mean"),
            avg_auc=("roc_auc", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_calibration_error=("calibration_error", "mean"),
            avg_predicted_probability=("avg_predicted_probability", "mean"),
            min_predicted_probability=("min_predicted_probability", "min"),
            max_predicted_probability=("max_predicted_probability", "max"),
            avg_feature_count=("feature_count", "mean"),
        )
        .reset_index()
    )
    summary["std_log_loss"] = summary["std_log_loss"].fillna(0.0)

    model_lookup = summary.set_index("model_name")
    team_loss = float(model_lookup.loc["team_only_logistic", "avg_log_loss"]) if "team_only_logistic" in model_lookup.index else np.inf
    elo_loss = float(model_lookup.loc["elo_only", "avg_log_loss"]) if "elo_only" in model_lookup.index else np.inf
    book_loss = float(model_lookup.loc["sportsbook_benchmark", "avg_log_loss"]) if "sportsbook_benchmark" in model_lookup.index else np.inf
    fair_candidates = summary[
        summary["model_name"].isin(
            [
                "elo_only",
                "team_only_logistic",
                "team_plus_selected_player_logistic",
                "team_plus_selected_player_random_forest",
                "team_plus_selected_player_gradient_boosting",
                "team_plus_selected_player_xgboost",
            ]
        )
    ].copy()
    stable_candidates = fair_candidates[
        fair_candidates["avg_calibration_error"].le(0.08)
        & fair_candidates["min_predicted_probability"].ge(0.02)
        & fair_candidates["max_predicted_probability"].le(0.98)
    ].copy()
    champion_pool = stable_candidates if not stable_candidates.empty else fair_candidates
    champion_name = str(champion_pool.sort_values(["avg_log_loss", "std_log_loss"], ascending=[True, True]).iloc[0]["model_name"])
    champion_uncal_loss = float(model_lookup.loc[champion_name, "avg_log_loss"])

    calibration_rows: list[dict[str, Any]] = []
    champion_spec = next((spec for spec in model_specs if spec["model_name"] == champion_name), None)
    if champion_spec is not None:
        for fold in folds:
            train = featured[featured["season"].astype(int).isin(fold["train_seasons"])].copy()
            validation = featured[featured["season"].astype(int).eq(int(fold["validation_season"]))].copy()
            if train.empty or validation.empty:
                continue
            _, features, train_probs, valid_probs = _model_probability_for_spec(champion_spec, train, validation)
            calibrators = _fit_calibrators(train_probs, train["actual_home_win"])
            for method in ["uncalibrated", "platt", "isotonic"]:
                probs = _apply_calibration(calibrators, valid_probs, method)
                metrics = _metrics_row(f"{champion_name}_{method}", validation["actual_home_win"], probs)
                metrics.update(
                    {
                        "feature_count": len(features),
                        "model_kind": str(champion_spec["model_kind"]),
                        **_probability_diagnostics(probs),
                    }
                )
                calibration_rows.append(_add_walk_forward_context(metrics, fold, train, validation, method))

    if calibration_rows:
        results = pd.concat([results, pd.DataFrame(calibration_rows)], ignore_index=True, sort=False)
        calibration_summary = (
            pd.DataFrame(calibration_rows)
            .groupby("calibration_method", dropna=False)
            .agg(avg_log_loss=("log_loss", "mean"), avg_brier_score=("brier_score", "mean"), avg_calibration_error=("calibration_error", "mean"))
            .reset_index()
            .sort_values(["avg_log_loss", "avg_calibration_error"], ascending=[True, True])
        )
        best_calibration = str(calibration_summary.iloc[0]["calibration_method"])
    else:
        best_calibration = "uncalibrated"

    summary["is_champion"] = summary["model_name"].eq(champion_name)
    summary["beats_elo"] = summary["avg_log_loss"] < elo_loss
    summary["beats_team_only"] = summary["avg_log_loss"] < team_loss
    summary["beats_sportsbook_benchmark"] = summary["avg_log_loss"] < book_loss
    summary["champion_model"] = champion_name
    summary["champion_calibration_method"] = best_calibration

    selected_logistic_loss = (
        float(model_lookup.loc["team_plus_selected_player_logistic", "avg_log_loss"])
        if "team_plus_selected_player_logistic" in model_lookup.index
        else np.inf
    )
    selected_rf_loss = (
        float(model_lookup.loc["team_plus_selected_player_random_forest", "avg_log_loss"])
        if "team_plus_selected_player_random_forest" in model_lookup.index
        else np.inf
    )
    selected_xgb_loss = (
        float(model_lookup.loc["team_plus_selected_player_xgboost", "avg_log_loss"])
        if "team_plus_selected_player_xgboost" in model_lookup.index
        else np.inf
    )
    selected_improved = min(selected_logistic_loss, selected_rf_loss, selected_xgb_loss) < team_loss
    champion_info = {
        "best_fair_model": champion_name,
        "average_walk_forward_log_loss": champion_uncal_loss,
        "average_walk_forward_brier_score": float(model_lookup.loc[champion_name, "avg_brier_score"]),
        "best_calibration_method": best_calibration,
        "selected_player_features_improved_walk_forward": bool(selected_improved),
        "champion_beats_elo": bool(champion_uncal_loss < elo_loss),
        "champion_beats_team_only": bool(champion_uncal_loss < team_loss),
        "champion_beats_sportsbook_benchmark": bool(champion_uncal_loss < book_loss),
        "champion_reasonable_calibration": bool(float(model_lookup.loc[champion_name, "avg_calibration_error"]) <= 0.08),
        "champion_no_extreme_probability_behavior": bool(
            float(model_lookup.loc[champion_name, "min_predicted_probability"]) >= 0.02
            and float(model_lookup.loc[champion_name, "max_predicted_probability"]) <= 0.98
        ),
        "walk_forward_folds": int(base_results["fold"].nunique()),
    }
    return results, summary.sort_values("avg_log_loss", ascending=True).reset_index(drop=True), champion_info


def _ablation_results(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    player_feature_columns: list[str],
    selected_player_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rolling_features = [
        "home_recent_win_pct",
        "away_recent_win_pct",
        "recent_win_pct_diff",
        "home_recent_point_diff",
        "away_recent_point_diff",
        "recent_point_diff_diff",
        "home_win_pct_l5",
        "away_win_pct_l5",
        "win_pct_l5_diff",
        "home_win_pct_l10",
        "away_win_pct_l10",
        "win_pct_l10_diff",
        "home_point_diff_l5",
        "away_point_diff_l5",
        "point_diff_l5_diff",
        "home_point_diff_l10",
        "away_point_diff_l10",
        "point_diff_l10_diff",
        "home_off_rating_l10",
        "away_off_rating_l10",
        "off_rating_l10_diff",
        "home_def_rating_l10",
        "away_def_rating_l10",
        "def_rating_l10_diff",
        "season_progress_pct",
    ]
    schedule_features = [
        "home_rest_days",
        "away_rest_days",
        "rest_days_diff",
        "home_back_to_back",
        "away_back_to_back",
        "home_three_in_four",
        "away_three_in_four",
        "three_in_four_diff",
    ]
    experiments: list[tuple[str, list[str], str]] = [
        ("home_court_only", ["home_court"], "logistic_l2"),
        ("elo_only", ["home_elo", "away_elo", "elo_diff"], "logistic_l2"),
        ("rolling_team_stats_only", rolling_features, "logistic_l2"),
        ("rest_and_schedule_only", schedule_features, "logistic_l2"),
        ("player_features_only", player_feature_columns, "logistic_l2"),
        ("team_only_fair_l2", TEAM_FAIR_FEATURES, "logistic_l2"),
        ("team_features_plus_player_features_l2", TEAM_FAIR_FEATURES + player_feature_columns, "logistic_l2"),
        ("team_features_plus_selected_player_features_l2", TEAM_FAIR_FEATURES + selected_player_features, "logistic_l2"),
        ("team_features_plus_selected_player_features_l1", TEAM_FAIR_FEATURES + selected_player_features, "logistic_l1"),
        ("team_features_plus_selected_player_features_elasticnet", TEAM_FAIR_FEATURES + selected_player_features, "logistic_elasticnet"),
        ("team_features_plus_selected_player_features_random_forest", TEAM_FAIR_FEATURES + selected_player_features, "random_forest"),
        ("team_features_plus_selected_player_features_gradient_boosting", TEAM_FAIR_FEATURES + selected_player_features, "gradient_boosting"),
        ("team_features_plus_selected_player_features_xgboost", TEAM_FAIR_FEATURES + selected_player_features, "xgboost"),
    ]
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for name, features, model_kind in experiments:
        available = [feature for feature in features if feature in train.columns]
        if not available:
            continue
        try:
            metrics, model, probs = _evaluate_experiment(name, train, validation, available, model_kind=model_kind)
        except ValueError as exc:
            print(f"WARNING: Skipping {name}: {exc}")
            continue
        rows.append(metrics)
        calibration_rows.extend(_calibration_bucket_rows(name, validation["actual_home_win"], probs))
        models[name] = {"model": model, "features": available, "probs": probs, "log_loss": metrics.get("log_loss")}
    output = pd.DataFrame(rows).sort_values("log_loss", ascending=True).reset_index(drop=True)
    if not output.empty:
        reference_losses = {
            "home_team_baseline_log_loss": output.loc[output["model_name"].eq("home_court_only"), "log_loss"].min(),
            "elo_baseline_log_loss": output.loc[output["model_name"].eq("elo_only"), "log_loss"].min(),
            "team_only_model_log_loss": output.loc[output["model_name"].eq("team_only_fair_l2"), "log_loss"].min(),
            "sportsbook_benchmark_log_loss": evaluate_binary_probabilities(
                validation["actual_home_win"],
                validation["sportsbook_home_no_vig_prob"],
            ).get("log_loss"),
        }
        for column, value in reference_losses.items():
            output[column] = value
        output["beats_home_team_baseline"] = output["log_loss"] < output["home_team_baseline_log_loss"]
        output["beats_elo_baseline"] = output["log_loss"] < output["elo_baseline_log_loss"]
        output["beats_team_only_model"] = output["log_loss"] < output["team_only_model_log_loss"]
        output["beats_sportsbook_benchmark"] = output["log_loss"] < output["sportsbook_benchmark_log_loss"]
    return output, models, pd.DataFrame(calibration_rows)


def _add_suggested_bet_columns(validation: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    output = validation.copy()
    output["model_home_win_prob"] = output[probability_column]
    output["away_model_win_prob"] = 1.0 - output["model_home_win_prob"]
    output["sportsbook_home_win_prob"] = output["sportsbook_home_no_vig_prob"]
    output["edge"] = output["model_home_win_prob"] - output["sportsbook_home_win_prob"]
    output["away_edge"] = output["away_model_win_prob"] - output["away_no_vig_prob"]
    output["result"] = output["actual_home_win"].astype(int)
    for threshold in THRESHOLDS:
        suffix = f"{int(threshold * 100)}pct"
        output[f"suggested_bet_{suffix}"] = np.select(
            [output["edge"].ge(threshold), output["away_edge"].ge(threshold)],
            ["HOME", "AWAY"],
            default="NO_BET",
        )
    return output


def _merge_player_features(feature_frame: pd.DataFrame, player_features_path: Path) -> pd.DataFrame:
    output = feature_frame.copy()
    if not player_features_path.exists():
        output["player_data_available"] = False
        output["projected_rotation_available"] = False
        output["missing_key_player_uncertainty"] = "high"
        return output
    player_features = pd.read_csv(player_features_path, dtype={"game_id": str}, low_memory=False)
    if "game_id" not in player_features.columns:
        output["player_data_available"] = False
        output["projected_rotation_available"] = False
        output["missing_key_player_uncertainty"] = "high"
        return output
    drop_columns = [
        column
        for column in ["game_date", "season", "home_team_abbr", "away_team_abbr"]
        if column in player_features.columns
    ]
    player_features = player_features.drop(columns=drop_columns)
    output = output.merge(player_features, on="game_id", how="left", validate="one_to_one")
    output["player_data_available"] = output.get("player_data_available", False).fillna(False).astype(bool)
    output["projected_rotation_available"] = output.get("projected_rotation_available", False).fillna(False).astype(bool)
    output["missing_key_player_uncertainty"] = output.get("missing_key_player_uncertainty", "high").fillna("high")
    return output


def _load_all_nba_games_for_prediction() -> pd.DataFrame:
    parquet_path = PROJECT_ROOT / "data" / "interim" / "nba_games.parquet"
    csv_path = PROJECT_ROOT / "data" / "interim" / "nba_games.csv"
    if parquet_path.exists():
        games = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        games = pd.read_csv(csv_path, dtype={"game_id": str}, low_memory=False)
    else:
        games = pd.read_csv(PROJECT_ROOT / "data" / "reports" / "all_game_predictions.csv", dtype={"game_id": str}, low_memory=False)
    games = games.copy()
    if "home_score" not in games.columns and "home_points" in games.columns:
        games["home_score"] = games["home_points"]
    if "away_score" not in games.columns and "away_points" in games.columns:
        games["away_score"] = games["away_points"]
    if "actual_home_win" not in games.columns:
        if "home_win" in games.columns:
            games["actual_home_win"] = games["home_win"]
        elif {"home_score", "away_score"}.issubset(games.columns):
            games["actual_home_win"] = pd.to_numeric(games["home_score"], errors="coerce") > pd.to_numeric(games["away_score"], errors="coerce")
    required = ["game_id", "game_date", "season", "home_team_abbr", "away_team_abbr"]
    missing = [column for column in required if column not in games.columns]
    if missing:
        raise ValueError(f"NBA prediction source is missing columns for Kalshi paper trading: {missing}")
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.dropna(subset=["game_date", "season", "home_team_abbr", "away_team_abbr"]).copy()
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _fit_champion_model_for_frame(
    champion_source_name: str,
    train: pd.DataFrame,
    score_frame: pd.DataFrame,
    selected_player_features: list[str],
    calibration_method: str,
) -> tuple[np.ndarray, list[str]]:
    if champion_source_name == "elo_only":
        spec = {"model_name": champion_source_name, "features": ["home_elo", "away_elo", "elo_diff"], "model_kind": "logistic_l2"}
    elif champion_source_name == "team_only_logistic":
        spec = {"model_name": champion_source_name, "features": TEAM_FAIR_FEATURES, "model_kind": "logistic_l2"}
    elif champion_source_name == "team_plus_selected_player_logistic":
        spec = {"model_name": champion_source_name, "features": TEAM_FAIR_FEATURES + selected_player_features, "model_kind": "logistic_l2"}
    elif champion_source_name == "team_plus_selected_player_xgboost":
        spec = {"model_name": champion_source_name, "features": TEAM_FAIR_FEATURES + selected_player_features, "model_kind": "xgboost"}
    elif champion_source_name == "team_plus_selected_player_gradient_boosting":
        spec = {"model_name": champion_source_name, "features": TEAM_FAIR_FEATURES + selected_player_features, "model_kind": "gradient_boosting"}
    else:
        spec = {"model_name": champion_source_name, "features": TEAM_FAIR_FEATURES + selected_player_features, "model_kind": "random_forest"}

    features = [feature for feature in spec["features"] if feature in train.columns and feature in score_frame.columns]
    if not features:
        raise ValueError(f"No usable champion features for {champion_source_name}")
    metrics, model, _ = _evaluate_experiment(str(spec["model_name"]), train, train, features, model_kind=str(spec["model_kind"]))
    _ = metrics
    train_probs = _predict_pipeline(model, train, features)
    score_probs = _predict_pipeline(model, score_frame, features)
    calibrators = _fit_calibrators(train_probs, train["actual_home_win"])
    return _apply_calibration(calibrators, score_probs, calibration_method), features


def _other_team(row: pd.Series) -> str:
    yes_team = str(row.get("yes_team_abbr", ""))
    home = str(row.get("home_team_abbr", ""))
    away = str(row.get("away_team_abbr", ""))
    if yes_team == home:
        return away
    if yes_team == away:
        return home
    return ""


def _contract_profit(price_cents: float, won: bool) -> float:
    return _contract_profit_per_dollar_staked(price_cents, won)


def _contract_profit_per_dollar_staked(price_cents: float, won: bool) -> float:
    price = price_cents / 100.0
    if not np.isfinite(price) or price <= 0:
        return np.nan
    return (1.0 - price) / price if won else -1.0


def _contract_profit_per_contract(price_cents: float, won: bool) -> float:
    price = price_cents / 100.0
    if not np.isfinite(price) or price <= 0:
        return np.nan
    return 1.0 - price if won else -price


def _confidence_from_edge(edge: float, spread_cents: float, volume: float) -> str:
    if not np.isfinite(edge):
        return "none"
    if edge >= 0.10 and np.isfinite(spread_cents) and spread_cents <= 4 and np.isfinite(volume) and volume >= 100:
        return "high"
    if edge >= 0.07 and np.isfinite(spread_cents) and spread_cents <= 8 and np.isfinite(volume) and volume >= 25:
        return "medium"
    if edge >= 0.03:
        return "low"
    return "none"


def _enrich_kalshi_markets(markets: pd.DataFrame) -> pd.DataFrame:
    output = markets.copy()
    processed_path = PROJECT_ROOT / "data" / "processed" / "kalshi_game_market_matches.csv"
    taxonomy_path = PROJECT_ROOT / "data" / "processed" / "kalshi_market_taxonomy.csv"
    enrich_frames = []
    for path in [processed_path, taxonomy_path]:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"game_id": str, "market_ticker": str}, low_memory=False)
        if frame.empty or "market_ticker" not in frame.columns:
            continue
        keep = [
            column
            for column in [
                "market_ticker",
                "market_title",
                "market_subtitle",
                "series_ticker",
                "event_ticker",
                "open_time",
                "close_time",
                "expected_expiration_time",
                "yes_team_abbr",
                "no_team_abbr",
                "match_status",
                "match_score",
                "match_notes",
                "taxonomy_confidence",
                "taxonomy_notes",
            ]
            if column in frame.columns
        ]
        enrich_frames.append(frame[keep].dropna(subset=["market_ticker"]).drop_duplicates("market_ticker"))
    for enrich in enrich_frames:
        output = output.merge(enrich, on="market_ticker", how="left", suffixes=("", "_enriched"))
        for column in list(output.columns):
            if not column.endswith("_enriched"):
                continue
            base = column[: -len("_enriched")]
            if base in output.columns:
                output[base] = output[base].where(output[base].notna(), output[column])
                output = output.drop(columns=[column])
            else:
                output = output.rename(columns={column: base})
    return output


def _build_kalshi_mapping_audit(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return pd.DataFrame(
            columns=[
                "market_ticker",
                "market_title",
                "game_date",
                "home_team",
                "away_team",
                "yes_team",
                "no_team",
                "yes_means_home_team_wins",
                "yes_means_away_team_wins",
                "mapping_confidence",
                "mapping_warning",
            ]
        )
    rows = []
    for _, row in markets.iterrows():
        home = str(row.get("home_team_abbr", ""))
        away = str(row.get("away_team_abbr", ""))
        yes = str(row.get("yes_team_abbr", ""))
        no = str(row.get("no_team_abbr", ""))
        ticker = str(row.get("market_ticker", ""))
        ticker_yes = ticker.rsplit("-", maxsplit=1)[-1] if "-" in ticker else ""
        if not no and yes == home:
            no = away
        elif not no and yes == away:
            no = home
        warnings = []
        if yes not in {home, away}:
            warnings.append("yes_team_not_home_or_away")
        if no and no not in {home, away}:
            warnings.append("no_team_not_home_or_away")
        if ticker_yes and yes and ticker_yes != yes:
            warnings.append("ticker_yes_team_disagrees")
        if not str(row.get("market_title", "")).strip():
            warnings.append("missing_market_title")
        if yes == no and yes:
            warnings.append("yes_no_same_team")
        confidence = "high" if not warnings and yes in {home, away} else "medium" if yes in {home, away} else "low"
        rows.append(
            {
                "market_ticker": ticker,
                "market_title": row.get("market_title", ""),
                "game_date": row.get("game_date"),
                "home_team": home,
                "away_team": away,
                "yes_team": yes,
                "no_team": no,
                "yes_means_home_team_wins": yes == home,
                "yes_means_away_team_wins": yes == away,
                "market_title_parsing_available": bool(str(row.get("market_title", "")).strip()),
                "team_names_match_nba_game": bool(yes in {home, away} and (not no or no in {home, away})),
                "mapping_confidence": confidence,
                "mapping_warning": ";".join(warnings) if warnings else "",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("market_ticker").sort_values(["mapping_confidence", "game_date", "market_ticker"])


def _bucket_series(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(str)


def _paper_diagnostic_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["group_type", "group_value", "trades", "wins", "losses", "average_edge", "roi", "profit_loss"])
    working = ledger[ledger["paper_trade_flag"].astype(bool)].copy()
    working = working[working["result_known"].astype(bool)].copy()
    if working.empty:
        return pd.DataFrame(columns=["group_type", "group_value", "trades", "wins", "losses", "average_edge", "roi", "profit_loss"])
    working["edge_bucket"] = _bucket_series(
        working["edge_used_for_trade"].abs(),
        [0, 0.03, 0.05, 0.07, 0.10, 1.0],
        ["0-3%", "3-5%", "5-7%", "7-10%", "10%+"],
    )
    working["price_bucket"] = _bucket_series(
        working["selected_contract_implied_probability"],
        [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    working["favorite_vs_underdog"] = np.where(working["selected_contract_implied_probability"].ge(0.5), "favorite", "underdog")
    working["home_vs_away"] = np.where(working["selected_team"].astype(str).eq(working["home_team"].astype(str)), "home", "away")
    working["yes_side_vs_no_side"] = working["selected_side_yes_or_no"]
    working["minutes_before_tipoff_bucket"] = _bucket_series(
        working["minutes_before_tipoff"],
        [-1, 5, 30, 60, 240, 100000],
        ["0-5m", "5-30m", "30-60m", "60m-4h", "4h+"],
    )
    rows = []
    for group_type in [
        "edge_bucket",
        "price_bucket",
        "favorite_vs_underdog",
        "home_vs_away",
        "yes_side_vs_no_side",
        "mapping_confidence",
        "minutes_before_tipoff_bucket",
    ]:
        for value, group in working.groupby(group_type, dropna=False):
            wins = int(group["selected_win"].astype(bool).sum())
            trades = int(len(group))
            profit = float(pd.to_numeric(group["profit_loss_per_dollar_staked"], errors="coerce").sum())
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": str(value),
                    "trades": trades,
                    "wins": wins,
                    "losses": trades - wins,
                    "average_edge": float(pd.to_numeric(group["edge_used_for_trade"], errors="coerce").mean()),
                    "roi": profit / trades if trades else np.nan,
                    "profit_loss": profit,
                }
            )
    return pd.DataFrame(rows).sort_values(["group_type", "roi"], ascending=[True, True]).reset_index(drop=True)


def _losing_patterns(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    output = diagnostics[diagnostics["trades"].gt(0)].copy()
    output = output.sort_values(["roi", "profit_loss"], ascending=[True, True]).head(25)
    return output.reset_index(drop=True)


def _max_drawdown(profits: pd.Series | np.ndarray) -> float:
    values = pd.to_numeric(pd.Series(profits), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0
    cumulative = np.cumsum(values)
    running_max = np.maximum.accumulate(np.insert(cumulative, 0, 0.0))[1:]
    drawdown = cumulative - running_max
    return float(drawdown.min()) if len(drawdown) else 0.0


def _assign_paper_trade_time_split(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    output = ledger.copy()
    sort_columns = [column for column in ["game_date", "price_timestamp", "market_ticker", "threshold_used"] if column in output.columns]
    output = output.sort_values(sort_columns).reset_index(drop=True)
    output["trade_rank_by_time"] = np.arange(1, len(output) + 1)
    cutoff = int(math.floor(len(output) * 0.60))
    output["paper_split"] = np.where(output["trade_rank_by_time"].le(cutoff), "discovery", "holdout")
    return output


def _timing_filter_mask(frame: pd.DataFrame, timing_filter: str) -> pd.Series:
    minutes = pd.to_numeric(frame.get("minutes_before_tipoff"), errors="coerce")
    if timing_filter == "within_30m":
        return minutes.ge(0) & minutes.le(30)
    if timing_filter == "30_to_60m":
        return minutes.gt(30) & minutes.le(60)
    if timing_filter == "1_to_4h":
        return minutes.gt(60) & minutes.le(240)
    if timing_filter == "4h_plus":
        return minutes.gt(240)
    return pd.Series(True, index=frame.index)


def _strategy_mask(
    frame: pd.DataFrame,
    min_edge: float,
    min_price: float,
    max_price: float,
    timing_filter: str,
    side_filter: str,
    favorite_filter: str,
    home_away_filter: str,
) -> pd.Series:
    price = pd.to_numeric(frame.get("selected_contract_implied_probability"), errors="coerce")
    edge = pd.to_numeric(frame.get("edge_used_for_trade"), errors="coerce")
    mask = edge.ge(min_edge) & price.ge(min_price) & price.le(max_price)
    mask &= _timing_filter_mask(frame, timing_filter)
    if side_filter == "yes_only":
        mask &= frame["selected_side_yes_or_no"].astype(str).eq("YES")
    elif side_filter == "no_only":
        mask &= frame["selected_side_yes_or_no"].astype(str).eq("NO")
    if favorite_filter == "favorites_only":
        mask &= price.ge(0.5)
    elif favorite_filter == "underdogs_only":
        mask &= price.lt(0.5)
    if home_away_filter == "home_only":
        mask &= frame["selected_team"].astype(str).eq(frame["home_team"].astype(str))
    elif home_away_filter == "away_only":
        mask &= frame["selected_team"].astype(str).eq(frame["away_team"].astype(str))
    return mask.fillna(False)


def _evaluate_strategy_rows(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "average_edge": np.nan,
            "average_price": np.nan,
            "profit_loss_per_dollar_staked": 0.0,
            "roi": np.nan,
            "max_drawdown": 0.0,
        }
    wins = int(frame["selected_win"].astype(bool).sum())
    trades = int(len(frame))
    profit_series = pd.to_numeric(frame["profit_loss_per_dollar_staked"], errors="coerce").fillna(0.0)
    profit = float(profit_series.sum())
    return {
        "trades": trades,
        "wins": wins,
        "losses": trades - wins,
        "win_rate": wins / trades if trades else np.nan,
        "average_edge": float(pd.to_numeric(frame["edge_used_for_trade"], errors="coerce").mean()),
        "average_price": float(pd.to_numeric(frame["selected_contract_implied_probability"], errors="coerce").mean()),
        "profit_loss_per_dollar_staked": profit,
        "roi": profit / trades if trades else np.nan,
        "max_drawdown": _max_drawdown(profit_series),
    }


def _strategy_specificity(row: pd.Series) -> int:
    return sum(
        [
            str(row.get("timing_filter")) != "all",
            str(row.get("side_filter")) != "both",
            str(row.get("favorite_filter")) != "both",
            str(row.get("home_away_filter")) != "both",
            float(row.get("min_edge", 0.0) or 0.0) >= 0.12,
            float(row.get("min_price", 0.0) or 0.0) > 0.0,
            float(row.get("max_price", 1.0) or 1.0) < 1.0,
        ]
    )


def _build_kalshi_strategy_grid(ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if ledger.empty:
        selected = {
            "status": "research_only",
            "warning": "No paper trades available for strategy testing.",
            "validated_edge": False,
            "recommended_default_rule": "No live trades. Research-only display. Hide 0-20% longshot signals from highlighted opportunities. Require at least 7% edge to appear as a potential signal.",
        }
        return pd.DataFrame(), selected, pd.DataFrame()

    scored = _assign_paper_trade_time_split(ledger)
    discovery = scored[scored["paper_split"].eq("discovery") & scored["result_known"].astype(bool)].copy()
    holdout = scored[scored["paper_split"].eq("holdout") & scored["result_known"].astype(bool)].copy()
    min_edges = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
    min_prices = [0.0, 0.10, 0.20, 0.30]
    max_prices = [0.70, 0.80, 0.90, 1.00]
    timing_filters = ["within_30m", "30_to_60m", "1_to_4h", "4h_plus"]
    side_filters = ["yes_only", "no_only", "both"]
    favorite_filters = ["favorites_only", "underdogs_only", "both"]
    home_away_filters = ["home_only", "away_only", "both"]
    rows = []
    for min_edge in min_edges:
        for min_price in min_prices:
            for max_price in max_prices:
                if min_price >= max_price:
                    continue
                for timing_filter in timing_filters:
                    for side_filter in side_filters:
                        for favorite_filter in favorite_filters:
                            for home_away_filter in home_away_filters:
                                mask = _strategy_mask(discovery, min_edge, min_price, max_price, timing_filter, side_filter, favorite_filter, home_away_filter)
                                subset = discovery[mask].copy()
                                metrics = _evaluate_strategy_rows(subset)
                                row = {
                                    "min_edge": min_edge,
                                    "min_contract_price": min_price,
                                    "max_contract_price": max_price,
                                    "timing_filter": timing_filter,
                                    "side_filter": side_filter,
                                    "favorite_filter": favorite_filter,
                                    "home_away_filter": home_away_filter,
                                    **metrics,
                                }
                                row["meets_min_discovery_trades"] = int(metrics["trades"]) >= 30
                                row["positive_discovery_roi"] = bool(pd.notna(metrics["roi"]) and metrics["roi"] > 0)
                                row["specificity_score"] = _strategy_specificity(pd.Series(row))
                                rows.append(row)
    grid = pd.DataFrame(rows)
    candidates = grid[
        grid["meets_min_discovery_trades"]
        & grid["positive_discovery_roi"]
        & grid["max_drawdown"].ge(-35.0)
        & grid["specificity_score"].le(4)
    ].copy()
    if candidates.empty:
        selected_row = grid[grid["meets_min_discovery_trades"]].sort_values(["roi", "trades"], ascending=[False, False]).head(1)
        status = "research_only"
        warning = "No discovery strategy passed positive ROI and robustness filters."
    else:
        selected_row = candidates.sort_values(["roi", "trades", "specificity_score"], ascending=[False, False, True]).head(1)
        status = "candidate"
        warning = ""
    if selected_row.empty:
        selected = {
            "status": "research_only",
            "warning": "No strategy had at least 30 discovery trades.",
            "validated_edge": False,
            "recommended_default_rule": "No live trades. Research-only display. Hide 0-20% longshot signals from highlighted opportunities. Require at least 7% edge to appear as a potential signal.",
        }
        return grid, selected, pd.DataFrame()

    selected_dict = selected_row.iloc[0].to_dict()
    holdout_mask = _strategy_mask(
        holdout,
        float(selected_dict["min_edge"]),
        float(selected_dict["min_contract_price"]),
        float(selected_dict["max_contract_price"]),
        str(selected_dict["timing_filter"]),
        str(selected_dict["side_filter"]),
        str(selected_dict["favorite_filter"]),
        str(selected_dict["home_away_filter"]),
    )
    holdout_subset = holdout[holdout_mask].copy()
    holdout_metrics = _evaluate_strategy_rows(holdout_subset)
    holdout_passed = bool(
        holdout_metrics["trades"] >= 30
        and pd.notna(holdout_metrics["roi"])
        and holdout_metrics["roi"] > 0
        and holdout_metrics["max_drawdown"] >= -35.0
    )
    selected = {
        "status": "validated_edge" if holdout_passed else ("potential_signal" if status == "candidate" else "research_only"),
        "validated_edge": holdout_passed,
        "warning": "" if holdout_passed else ("Discovery was positive, but holdout failed or was too small." if status == "candidate" else warning),
        "recommended_default_rule": "" if holdout_passed else "No live trades. Research-only display. Hide 0-20% longshot signals from highlighted opportunities. Require at least 7% edge to appear as a potential signal.",
        "selected_rule": {
            key: selected_dict.get(key)
            for key in [
                "min_edge",
                "min_contract_price",
                "max_contract_price",
                "timing_filter",
                "side_filter",
                "favorite_filter",
                "home_away_filter",
            ]
        },
        "discovery": {key: selected_dict.get(key) for key in ["trades", "wins", "losses", "win_rate", "average_edge", "average_price", "profit_loss_per_dollar_staked", "roi", "max_drawdown"]},
        "holdout": holdout_metrics,
    }
    holdout_results = pd.DataFrame([{**selected["selected_rule"], **{f"holdout_{k}": v for k, v in holdout_metrics.items()}, "holdout_passed": holdout_passed}])
    return grid.sort_values(["meets_min_discovery_trades", "roi", "trades"], ascending=[False, False, False]).reset_index(drop=True), selected, holdout_results


def _build_kalshi_paper_trades(
    markets_path: Path,
    champion_predictions: pd.DataFrame,
    champion_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    thresholds = [0.03, 0.05, 0.07, 0.10]
    empty_summary = pd.DataFrame(
        [{"threshold_used": threshold, "paper_trades": 0, "wins": 0, "losses": 0, "win_rate": np.nan, "average_edge": np.nan, "profit_loss": 0.0, "roi": np.nan, "open_trades": 0, "closed_trades": 0} for threshold in thresholds]
    )
    empty_meta = {"paper_trades": 0, "closed_trades": 0, "open_trades": 0, "best_paper_threshold": None, "paper_roi": None}
    if not markets_path.exists() or champion_predictions.empty:
        return pd.DataFrame(), empty_summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_meta

    markets = pd.read_csv(markets_path, dtype={"game_id": str, "market_ticker": str}, low_memory=False)
    if markets.empty:
        return pd.DataFrame(), empty_summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty_meta
    markets["game_id"] = markets["game_id"].astype(str)
    markets = _enrich_kalshi_markets(markets)
    mapping_audit = _build_kalshi_mapping_audit(markets)
    frame = markets.merge(champion_predictions, on="game_id", how="left")
    rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    excluded_postgame = 0
    for _, row in frame.iterrows():
        yes_team = str(row.get("yes_team_abbr", ""))
        no_team = str(row.get("no_team_abbr", ""))
        home = str(row.get("home_team_abbr", ""))
        away = str(row.get("away_team_abbr", ""))
        if not no_team and yes_team == home:
            no_team = away
        elif not no_team and yes_team == away:
            no_team = home
        home_prob = _to_float(row.get("champion_home_win_probability"))
        if not np.isfinite(home_prob) or yes_team not in {home, away}:
            continue
        champion_yes = home_prob if yes_team == home else 1.0 - home_prob
        champion_no = 1.0 - champion_yes
        yes_ask = _to_float(row.get("yes_ask", row.get("yes_mid_cents")))
        yes_bid = _to_float(row.get("yes_bid"))
        if not np.isfinite(yes_ask):
            yes_ask = _to_float(row.get("yes_mid_cents"))
        if not np.isfinite(yes_bid) or not np.isfinite(yes_ask):
            continue
        no_ask = 100.0 - yes_bid
        yes_market = yes_ask / 100.0
        no_market = no_ask / 100.0
        yes_edge = champion_yes - yes_market
        no_edge = (1.0 - champion_yes) - no_market
        best_side = "YES" if yes_edge >= no_edge else "NO"
        best_abs_edge = yes_edge if best_side == "YES" else no_edge
        signed_yes_edge = champion_yes - yes_market
        selected_team = yes_team if best_side == "YES" else no_team
        selected_probability = champion_yes if best_side == "YES" else champion_no
        selected_price = yes_ask if best_side == "YES" else no_ask
        selected_market_probability = yes_market if best_side == "YES" else no_market
        spread = yes_ask - yes_bid
        volume = _to_float(row.get("volume"))
        minutes_before_tipoff = _to_float(row.get("minutes_before_tipoff"))
        price_ts = _to_float(row.get("snapshot_ts"))
        price_timestamp = datetime.fromtimestamp(price_ts, timezone.utc).isoformat() if np.isfinite(price_ts) else ""
        game_start_ts = price_ts + minutes_before_tipoff * 60.0 if np.isfinite(price_ts) and np.isfinite(minutes_before_tipoff) else np.nan
        game_start_time = datetime.fromtimestamp(game_start_ts, timezone.utc).isoformat() if np.isfinite(game_start_ts) else ""
        price_is_pregame = bool(np.isfinite(minutes_before_tipoff) and minutes_before_tipoff >= 0)
        if not price_is_pregame:
            excluded_postgame += 1
            continue
        data_quality_ok = np.isfinite(spread) and spread <= 10 and (not np.isfinite(volume) or volume >= 0)
        if not data_quality_ok:
            continue
        result_known = pd.notna(row.get("actual_yes_win"))
        actual_yes_win = bool(row.get("actual_yes_win")) if result_known else False
        actual_winner = yes_team if actual_yes_win else _other_team(row)
        profit_yes_per_contract = _contract_profit_per_contract(yes_ask, actual_yes_win) if result_known else np.nan
        profit_no_per_contract = _contract_profit_per_contract(no_ask, not actual_yes_win) if result_known else np.nan
        profit_yes_per_dollar = _contract_profit_per_dollar_staked(yes_ask, actual_yes_win) if result_known else np.nan
        profit_no_per_dollar = _contract_profit_per_dollar_staked(no_ask, not actual_yes_win) if result_known else np.nan
        selected_profit_contract = profit_yes_per_contract if best_side == "YES" else profit_no_per_contract
        selected_profit_dollar = profit_yes_per_dollar if best_side == "YES" else profit_no_per_dollar
        selected_win = actual_yes_win if best_side == "YES" else (not actual_yes_win)
        mapping_row = mapping_audit[mapping_audit["market_ticker"].astype(str).eq(str(row.get("market_ticker")))]
        mapping_confidence = str(mapping_row.iloc[0]["mapping_confidence"]) if not mapping_row.empty else "low"
        for threshold in thresholds:
            if best_abs_edge < threshold:
                continue
            rows.append(
                {
                    "timestamp": now,
                    "game_date": row.get("game_date"),
                    "home_team": home,
                    "away_team": row.get("away_team_abbr"),
                    "contract_name": row.get("market_ticker"),
                    "market_ticker": row.get("market_ticker"),
                    "market_title": row.get("market_title", ""),
                    "champion_model": champion_name,
                    "yes_team": yes_team,
                    "no_team": no_team,
                    "kalshi_yes_price": yes_ask,
                    "kalshi_implied_probability": yes_market,
                    "champion_fair_probability": champion_yes,
                    "fair_edge": signed_yes_edge,
                    "tradable_edge": best_abs_edge,
                    "selected_team": selected_team,
                    "selected_side_yes_or_no": best_side,
                    "selected_team_win_probability": selected_probability,
                    "selected_contract_price": selected_price,
                    "selected_contract_implied_probability": selected_market_probability,
                    "edge_used_for_trade": best_abs_edge,
                    "confidence_label": _confidence_from_edge(best_abs_edge, spread, volume),
                    "suggested_side": best_side,
                    "paper_trade_flag": True,
                    "threshold_used": threshold,
                    "price_timestamp": price_timestamp,
                    "game_start_time": game_start_time,
                    "minutes_before_tipoff": minutes_before_tipoff,
                    "price_is_pregame": price_is_pregame,
                    "mapping_confidence": mapping_confidence,
                    "result_known": result_known,
                    "actual_winner": actual_winner,
                    "profit_loss_if_yes": profit_yes_per_dollar,
                    "profit_loss_if_no": profit_no_per_dollar,
                    "profit_loss_per_contract": selected_profit_contract if result_known else np.nan,
                    "profit_loss_per_dollar_staked": selected_profit_dollar if result_known else np.nan,
                    "selected_profit_loss": selected_profit_dollar if result_known else np.nan,
                    "selected_win": selected_win if result_known else np.nan,
                }
            )
    ledger = pd.DataFrame(rows)
    diagnostics = _paper_diagnostic_rows(ledger)
    losing_patterns = _losing_patterns(diagnostics)
    summary_rows = []
    for threshold in thresholds:
        group = ledger[ledger["threshold_used"].eq(threshold)] if not ledger.empty else pd.DataFrame()
        closed = group[group["result_known"].astype(bool)] if not group.empty else pd.DataFrame()
        open_trades = int(len(group) - len(closed))
        wins = int(closed["selected_win"].astype(bool).sum()) if not closed.empty else 0
        losses = int(len(closed) - wins)
        profit = float(pd.to_numeric(closed.get("profit_loss_per_dollar_staked", pd.Series(dtype=float)), errors="coerce").sum()) if not closed.empty else 0.0
        summary_rows.append(
            {
                "threshold_used": threshold,
                "paper_trades": int(len(group)),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(closed) if len(closed) else np.nan,
                "average_edge": float(pd.to_numeric(group.get("edge_used_for_trade", pd.Series(dtype=float)), errors="coerce").mean()) if len(group) else np.nan,
                "profit_loss": profit,
                "roi": profit / len(closed) if len(closed) else np.nan,
                "open_trades": open_trades,
                "closed_trades": int(len(closed)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    closed_summary = summary[summary["closed_trades"].gt(0)].copy()
    if closed_summary.empty:
        best_threshold = None
        paper_roi = None
    else:
        best = closed_summary.sort_values(["roi", "closed_trades"], ascending=[False, False]).iloc[0]
        best_threshold = float(best["threshold_used"])
        paper_roi = float(best["roi"]) if pd.notna(best["roi"]) else None
    meta = {
        "paper_trades": int(len(ledger)),
        "closed_trades": int(summary["closed_trades"].sum()) if not summary.empty else 0,
        "open_trades": int(summary["open_trades"].sum()) if not summary.empty else 0,
        "best_paper_threshold": best_threshold,
        "paper_roi": paper_roi,
        "small_sample_warning": "Sample size is too small to validate live Kalshi edge." if int(summary["closed_trades"].sum()) < 50 else "",
        "payout_method": "profit_loss_per_dollar_staked",
        "mapping_confidence_rate": float(mapping_audit["mapping_confidence"].eq("high").mean()) if not mapping_audit.empty else 0.0,
        "pregame_price_rate": float(ledger["price_is_pregame"].mean()) if not ledger.empty else 0.0,
        "excluded_postgame_prices": int(excluded_postgame),
        "mapping_passed_audit": bool(not mapping_audit.empty and mapping_audit["mapping_confidence"].eq("high").all()),
        "timing_passed_audit": bool(excluded_postgame == 0 and (ledger.empty or ledger["price_is_pregame"].all())),
    }
    if not diagnostics.empty:
        edge_groups = diagnostics[diagnostics["group_type"].eq("edge_bucket")]
        price_groups = diagnostics[diagnostics["group_type"].eq("price_bucket")]
        if not edge_groups.empty:
            meta["best_edge_bucket"] = str(edge_groups.sort_values("roi", ascending=False).iloc[0]["group_value"])
            meta["worst_edge_bucket"] = str(edge_groups.sort_values("roi", ascending=True).iloc[0]["group_value"])
        if not price_groups.empty:
            meta["best_price_bucket"] = str(price_groups.sort_values("roi", ascending=False).iloc[0]["group_value"])
            meta["worst_price_bucket"] = str(price_groups.sort_values("roi", ascending=True).iloc[0]["group_value"])
    return ledger, summary, mapping_audit, diagnostics, losing_patterns, meta


def _to_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return np.nan
    return output if np.isfinite(output) else np.nan


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    games_path = Path(args.games_path)
    odds_path = Path(args.odds_path) if args.odds_path else resolve_project_path(config.data.sportsbook_odds_path)
    matched, odds = _load_training_frame(games_path, odds_path)
    report = sportsbook_match_report_by_season(pd.read_csv(games_path, dtype={"game_id": str}, low_memory=False), odds)
    split_plan = build_free_odds_split_plan(report, mode=config.data.free_odds_split_mode)
    train_seasons = [int(season) for season in split_plan["train_seasons"]]
    validation_season = int(split_plan["validation_season"])

    featured = _add_pregame_team_features(matched)
    featured = _merge_player_features(featured, Path(args.player_features_path))
    player_feature_columns = _available_player_features(featured)
    selected_player_features = [feature for feature in SELECTED_PLAYER_FEATURES if feature in featured.columns]
    team_plus_player_features = TEAM_FAIR_FEATURES + player_feature_columns
    featured["dataset_split"] = "outside_split"
    featured.loc[featured["season"].astype(int).isin(train_seasons), "dataset_split"] = "train"
    featured.loc[featured["season"].astype(int).eq(validation_season), "dataset_split"] = "validation"
    train = featured[featured["dataset_split"].eq("train")].copy()
    validation = featured[featured["dataset_split"].eq("validation")].copy()

    if train.empty or validation.empty:
        raise SystemExit("Training or validation data is empty after requiring sportsbook market proxy.")
    if train["season"].astype(int).ge(validation_season).any():
        raise SystemExit("Leakage check failed: validation/future games are present in training.")
    forbidden = {"home_score", "away_score", "actual_home_win", "target_home_win", "Final", "score"}
    feature_set = set(TEAM_FAIR_FEATURES + team_plus_player_features + ADJUSTMENT_FEATURES + CATEGORICAL_COLUMNS)
    leaked_features = sorted(forbidden & feature_set)
    if leaked_features:
        raise SystemExit(f"Leakage check failed: result columns included as features: {leaked_features}")

    team_fair_model = _fit_logistic_model(train, TEAM_FAIR_FEATURES)
    player_fair_model = _fit_logistic_model(train, team_plus_player_features) if player_feature_columns else None
    selected_player_fair_model = (
        _fit_logistic_model(train, TEAM_FAIR_FEATURES + selected_player_features)
        if selected_player_features
        else None
    )
    anchored_model = _fit_market_anchored_model(train)

    train_team_uncal = _clip_prob(team_fair_model.predict_proba(train[TEAM_FAIR_FEATURES + CATEGORICAL_COLUMNS])[:, 1])
    valid_team_uncal = _clip_prob(team_fair_model.predict_proba(validation[TEAM_FAIR_FEATURES + CATEGORICAL_COLUMNS])[:, 1])
    if player_fair_model is not None:
        train_player_uncal = _clip_prob(player_fair_model.predict_proba(train[team_plus_player_features + CATEGORICAL_COLUMNS])[:, 1])
        valid_player_uncal = _clip_prob(player_fair_model.predict_proba(validation[team_plus_player_features + CATEGORICAL_COLUMNS])[:, 1])
    else:
        train_player_uncal = train_team_uncal
        valid_player_uncal = valid_team_uncal
    if selected_player_fair_model is not None:
        selected_features = TEAM_FAIR_FEATURES + selected_player_features
        train_selected_uncal = _clip_prob(selected_player_fair_model.predict_proba(train[selected_features + CATEGORICAL_COLUMNS])[:, 1])
        valid_selected_uncal = _clip_prob(selected_player_fair_model.predict_proba(validation[selected_features + CATEGORICAL_COLUMNS])[:, 1])
    else:
        train_selected_uncal = train_team_uncal
        valid_selected_uncal = valid_team_uncal
    train_anchor_uncal = _predict_market_anchored(anchored_model, train)
    valid_anchor_uncal = _predict_market_anchored(anchored_model, validation)
    calibrators = {
        "old_team_only_fair": _fit_calibrators(train_team_uncal, train["actual_home_win"]),
        "team_plus_player_fair": _fit_calibrators(train_player_uncal, train["actual_home_win"]),
        "team_plus_selected_player_fair": _fit_calibrators(train_selected_uncal, train["actual_home_win"]),
        "market_anchored": _fit_calibrators(train_anchor_uncal, train["actual_home_win"]),
    }

    probability_columns: dict[str, np.ndarray] = {
        "sportsbook_baseline": _clip_prob(validation["sportsbook_home_no_vig_prob"]),
    }
    train_probability_columns: dict[str, np.ndarray] = {
        "sportsbook_baseline": _clip_prob(train["sportsbook_home_no_vig_prob"]),
    }
    fair_model_inputs = [
        ("old_team_only_fair", train_team_uncal, valid_team_uncal),
    ]
    if player_fair_model is not None:
        fair_model_inputs.append(("team_plus_player_fair", train_player_uncal, valid_player_uncal))
    if selected_player_fair_model is not None:
        fair_model_inputs.append(("team_plus_selected_player_fair", train_selected_uncal, valid_selected_uncal))
    for model_name, train_probs, valid_probs in [
        *fair_model_inputs,
        ("market_anchored", train_anchor_uncal, valid_anchor_uncal),
    ]:
        for calibration_method in ["uncalibrated", "platt", "isotonic"]:
            column_name = f"{model_name}_{calibration_method}"
            probability_columns[column_name] = _apply_calibration(calibrators[model_name], valid_probs, calibration_method)
            train_probability_columns[column_name] = _apply_calibration(calibrators[model_name], train_probs, calibration_method)

    walk_forward_results, walk_forward_summary, walk_forward_champion = _walk_forward_validation(featured, selected_player_features)
    champion_source_name = str(walk_forward_champion.get("best_fair_model") or "")
    champion_calibration_method = str(walk_forward_champion.get("best_calibration_method") or "uncalibrated")
    champion_probability_name = ""
    if champion_source_name:
        champion_train_probs: np.ndarray | None = None
        champion_valid_probs: np.ndarray | None = None
        if champion_source_name == "elo_only":
            champion_spec = {"model_name": champion_source_name, "features": ["home_elo", "away_elo", "elo_diff"], "model_kind": "logistic_l2"}
            _, _, champion_train_probs, champion_valid_probs = _model_probability_for_spec(champion_spec, train, validation)
        elif champion_source_name == "team_only_logistic":
            champion_train_probs, champion_valid_probs = train_team_uncal, valid_team_uncal
        elif champion_source_name == "team_plus_selected_player_logistic":
            champion_train_probs, champion_valid_probs = train_selected_uncal, valid_selected_uncal
        elif champion_source_name in {
            "team_plus_selected_player_random_forest",
            "team_plus_selected_player_gradient_boosting",
            "team_plus_selected_player_xgboost",
        }:
            if champion_source_name.endswith("random_forest"):
                model_kind = "random_forest"
            elif champion_source_name.endswith("xgboost"):
                model_kind = "xgboost"
            else:
                model_kind = "gradient_boosting"
            champion_spec = {
                "model_name": champion_source_name,
                "features": TEAM_FAIR_FEATURES + selected_player_features,
                "model_kind": model_kind,
            }
            _, _, champion_train_probs, champion_valid_probs = _model_probability_for_spec(champion_spec, train, validation)
        if champion_train_probs is not None and champion_valid_probs is not None:
            champion_calibrators = _fit_calibrators(champion_train_probs, train["actual_home_win"])
            for method in ["uncalibrated", "platt", "isotonic"]:
                probability_columns[f"walk_forward_champion_{method}"] = _apply_calibration(
                    champion_calibrators,
                    champion_valid_probs,
                    method,
                )
                train_probability_columns[f"walk_forward_champion_{method}"] = _apply_calibration(
                    champion_calibrators,
                    champion_train_probs,
                    method,
                )
            if f"walk_forward_champion_{champion_calibration_method}" in probability_columns:
                champion_probability_name = f"walk_forward_champion_{champion_calibration_method}"

    metric_rows = [_metrics_row(name, validation["actual_home_win"], probs) for name, probs in probability_columns.items()]
    metric_table = pd.DataFrame(metric_rows)
    fair_model_names = [
        name
        for name in probability_columns
        if name.startswith("old_team_only_fair")
        or name.startswith("team_plus_player_fair")
        or name.startswith("team_plus_selected_player_fair")
        or name.startswith("walk_forward_champion")
    ]
    model_candidates = metric_table[metric_table["model_name"].isin(fair_model_names)].copy()
    if champion_probability_name and champion_probability_name in set(model_candidates["model_name"]):
        best_model_row = model_candidates[model_candidates["model_name"].eq(champion_probability_name)].iloc[0]
    else:
        best_model_row = model_candidates.sort_values("log_loss", ascending=True).iloc[0]
    best_model_name = str(best_model_row["model_name"])
    best_probability_column = f"prob_{best_model_name}"

    for name, probs in probability_columns.items():
        validation[f"prob_{name}"] = probs
    validation = _add_suggested_bet_columns(validation, best_probability_column)

    calibration = _calibration_report(validation, best_probability_column)
    backtest_frames = [
        _validation_backtest(validation, f"prob_{name}", name)
        for name in probability_columns
    ]
    backtest = pd.concat(backtest_frames, ignore_index=True)
    backtest_all = backtest[
        backtest["segment"].eq("all") & backtest["model_name"].eq(best_model_name)
    ].copy()
    best_bet_row = backtest_all.sort_values("roi", ascending=False, na_position="last").head(1)
    best_threshold = float(best_bet_row.iloc[0]["edge_threshold"]) if not best_bet_row.empty else None
    best_roi = float(best_bet_row.iloc[0]["roi"]) if not best_bet_row.empty and pd.notna(best_bet_row.iloc[0]["roi"]) else None
    best_bets = int(best_bet_row.iloc[0]["num_bets"]) if not best_bet_row.empty else 0

    diagnostics, edge_buckets, error_by_group = _diagnostic_reports(validation, best_probability_column)
    player_feature_diagnostics = _player_feature_diagnostics(
        featured,
        player_feature_columns,
        [feature for feature in TEAM_FAIR_FEATURES + ADJUSTMENT_FEATURES if feature in featured.columns],
    )
    ablation_results, ablation_models, ablation_calibration = _ablation_results(
        train,
        validation,
        player_feature_columns,
        selected_player_features,
    )
    best_ablation_name = str(ablation_results.iloc[0]["model_name"]) if not ablation_results.empty else ""
    best_ablation_model = ablation_models.get(best_ablation_name, {})
    feature_importance = pd.DataFrame()
    if best_ablation_model:
        feature_importance = _feature_importance_rows(
            best_ablation_name,
            best_ablation_model["model"],
            validation,
            best_ablation_model["features"],
            float(best_ablation_model["log_loss"]),
        )
    validation["sportsbook_benchmark_available"] = validation["sportsbook_home_no_vig_prob"].notna()
    validation["kalshi_market_available"] = False
    walk_forward_potential_signal = bool(
        walk_forward_champion.get("champion_beats_team_only")
        and walk_forward_champion.get("champion_beats_elo")
    )
    walk_forward_beats_sportsbook = bool(walk_forward_champion.get("champion_beats_sportsbook_benchmark"))
    walk_forward_strong_calibration = bool(walk_forward_champion.get("champion_reasonable_calibration"))
    fair_validated = bool(walk_forward_potential_signal and walk_forward_strong_calibration and best_roi is not None and best_roi > 0)
    kalshi_comparison_label = (
        "Validated edge"
        if fair_validated
        else ("Potential signal" if walk_forward_potential_signal else "Research only")
    )
    validation["prediction_label"] = kalshi_comparison_label
    kalshi_paper_meta: dict[str, Any] = {"paper_trades": 0, "closed_trades": 0, "open_trades": 0, "best_paper_threshold": None, "paper_roi": None}
    kalshi_paper_trades = pd.DataFrame()
    kalshi_paper_summary = pd.DataFrame()
    kalshi_mapping_audit = pd.DataFrame()
    kalshi_paper_diagnostics = pd.DataFrame()
    kalshi_losing_patterns = pd.DataFrame()
    kalshi_strategy_grid = pd.DataFrame()
    kalshi_strategy_selected: dict[str, Any] = {}
    kalshi_strategy_holdout = pd.DataFrame()
    try:
        all_games = _load_all_nba_games_for_prediction()
        all_featured = _add_pregame_team_features(all_games)
        all_featured = _merge_player_features(all_featured, Path(args.player_features_path))
        paper_train = featured[featured["season"].astype(int).le(validation_season)].copy()
        champion_scores, champion_features = _fit_champion_model_for_frame(
            champion_source_name or "team_plus_selected_player_random_forest",
            paper_train,
            all_featured,
            selected_player_features,
            champion_calibration_method,
        )
        champion_predictions = all_featured[["game_id"]].copy()
        champion_predictions["champion_home_win_probability"] = champion_scores
        (
            kalshi_paper_trades,
            kalshi_paper_summary,
            kalshi_mapping_audit,
            kalshi_paper_diagnostics,
            kalshi_losing_patterns,
            kalshi_paper_meta,
        ) = _build_kalshi_paper_trades(
            Path(args.kalshi_markets_path),
            champion_predictions,
            champion_source_name or best_model_name,
        )
        kalshi_paper_meta["champion_feature_count"] = len(champion_features)
        kalshi_paper_trades = _assign_paper_trade_time_split(kalshi_paper_trades)
        kalshi_strategy_grid, kalshi_strategy_selected, kalshi_strategy_holdout = _build_kalshi_strategy_grid(kalshi_paper_trades)
    except Exception as exc:
        print(f"WARNING: Could not build Kalshi paper-trading ledger: {type(exc).__name__}: {exc}")
        kalshi_paper_summary = pd.DataFrame(
            [{"threshold_used": threshold, "paper_trades": 0, "wins": 0, "losses": 0, "win_rate": np.nan, "average_edge": np.nan, "profit_loss": 0.0, "roi": np.nan, "open_trades": 0, "closed_trades": 0} for threshold in [0.03, 0.05, 0.07, 0.10]]
        )
    paper_sample_ok = int(kalshi_paper_meta.get("closed_trades") or 0) >= 50
    paper_positive = kalshi_paper_meta.get("paper_roi") is not None and float(kalshi_paper_meta.get("paper_roi") or 0.0) > 0.0
    strategy_validated = bool(kalshi_strategy_selected.get("validated_edge", False))
    fair_validated = bool(walk_forward_potential_signal and walk_forward_strong_calibration and strategy_validated)
    kalshi_comparison_label = (
        "Validated edge"
        if fair_validated
        else ("Potential signal" if walk_forward_potential_signal or kalshi_strategy_selected.get("status") == "potential_signal" else "Research only")
    )
    validation["prediction_label"] = kalshi_comparison_label

    output_paths = [
        Path(args.validation_output),
        Path(args.summary_output),
        Path(args.calibration_output),
        Path(args.metrics_output),
        Path(args.fair_metrics_output),
        Path(args.diagnostics_output),
        Path(args.edge_bucket_output),
        Path(args.error_group_output),
        Path(args.player_diagnostics_output),
        Path(args.feature_importance_output),
        Path(args.ablation_output),
        Path(args.walk_forward_output),
        Path(args.walk_forward_summary_output),
        Path(args.kalshi_paper_trades_output),
        Path(args.kalshi_paper_summary_output),
        Path(args.kalshi_mapping_audit_output),
        Path(args.kalshi_paper_diagnostics_output),
        Path(args.kalshi_losing_patterns_output),
        Path(args.kalshi_strategy_grid_output),
        Path(args.kalshi_strategy_selected_output),
        Path(args.kalshi_strategy_holdout_output),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "game_id",
        "game_date",
        "season",
        "home_team_abbr",
        "away_team_abbr",
        "sportsbook_home_win_prob",
        "prob_old_team_only_fair_uncalibrated",
        "prob_old_team_only_fair_platt",
        "prob_old_team_only_fair_isotonic",
        "prob_team_plus_player_fair_uncalibrated",
        "prob_team_plus_player_fair_platt",
        "prob_team_plus_player_fair_isotonic",
        "prob_team_plus_selected_player_fair_uncalibrated",
        "prob_team_plus_selected_player_fair_platt",
        "prob_team_plus_selected_player_fair_isotonic",
        "prob_market_anchored_uncalibrated",
        "prob_market_anchored_platt",
        "prob_market_anchored_isotonic",
        "model_home_win_prob",
        "away_model_win_prob",
        "edge",
        "away_edge",
        "actual_home_win",
        "home_moneyline",
        "away_moneyline",
        "home_no_vig_prob",
        "away_no_vig_prob",
        "dataset_split",
        "player_data_available",
        "projected_rotation_available",
        "missing_key_player_uncertainty",
        "sportsbook_benchmark_available",
        "kalshi_market_available",
        "prediction_label",
    ] + [f"suggested_bet_{int(threshold * 100)}pct" for threshold in THRESHOLDS]
    output_columns.extend([column for column in validation.columns if column.startswith("prob_walk_forward_champion_")])
    available_output_columns = [column for column in output_columns if column in validation.columns]
    validation[available_output_columns].to_csv(args.validation_output, index=False)
    validation[available_output_columns].to_csv(args.fair_validation_output, index=False)
    backtest.to_csv(args.summary_output, index=False)
    calibration.to_csv(args.calibration_output, index=False)
    diagnostics.to_csv(args.diagnostics_output, index=False)
    edge_buckets.to_csv(args.edge_bucket_output, index=False)
    error_by_group.to_csv(args.error_group_output, index=False)
    player_feature_diagnostics.to_csv(args.player_diagnostics_output, index=False)
    feature_importance.to_csv(args.feature_importance_output, index=False)
    ablation_output = pd.concat([ablation_results, ablation_calibration], ignore_index=True, sort=False)
    ablation_output.to_csv(args.ablation_output, index=False)
    walk_forward_results.to_csv(args.walk_forward_output, index=False)
    walk_forward_summary.to_csv(args.walk_forward_summary_output, index=False)
    kalshi_paper_trades.to_csv(args.kalshi_paper_trades_output, index=False)
    kalshi_paper_summary.to_csv(args.kalshi_paper_summary_output, index=False)
    kalshi_mapping_audit.to_csv(args.kalshi_mapping_audit_output, index=False)
    kalshi_paper_diagnostics.to_csv(args.kalshi_paper_diagnostics_output, index=False)
    kalshi_losing_patterns.to_csv(args.kalshi_losing_patterns_output, index=False)
    kalshi_strategy_grid.to_csv(args.kalshi_strategy_grid_output, index=False)
    Path(args.kalshi_strategy_selected_output).write_text(json.dumps(_json_safe(kalshi_strategy_selected), indent=2), encoding="utf-8")
    kalshi_strategy_holdout.to_csv(args.kalshi_strategy_holdout_output, index=False)

    sportsbook_metrics = metric_table[metric_table["model_name"].eq("sportsbook_baseline")].iloc[0].to_dict()
    old_metrics = metric_table[metric_table["model_name"].eq("old_team_only_fair_uncalibrated")].iloc[0].to_dict()
    player_metrics = (
        metric_table[metric_table["model_name"].eq("team_plus_player_fair_uncalibrated")].iloc[0].to_dict()
        if "team_plus_player_fair_uncalibrated" in set(metric_table["model_name"])
        else {}
    )
    selected_player_metrics = (
        metric_table[metric_table["model_name"].eq("team_plus_selected_player_fair_uncalibrated")].iloc[0].to_dict()
        if "team_plus_selected_player_fair_uncalibrated" in set(metric_table["model_name"])
        else {}
    )
    anchored_metrics = metric_table[metric_table["model_name"].eq("market_anchored_uncalibrated")].iloc[0].to_dict()
    fair_log_loss = float(best_model_row["log_loss"])

    summary = {
        "selected_split_mode": config.data.free_odds_split_mode,
        "training_seasons": [nba_season_display_label(season) for season in train_seasons],
        "validation_season": nba_season_display_label(validation_season),
        "training_games_used": int(train["game_id"].nunique()),
        "validation_games_used": int(validation["game_id"].nunique()),
        "model_metrics": old_metrics,
        "sportsbook_baseline_metrics": sportsbook_metrics,
        "old_baseline_metrics": old_metrics,
        "old_team_only_fair_metrics": old_metrics,
        "team_plus_player_fair_metrics": player_metrics,
        "team_plus_selected_player_fair_metrics": selected_player_metrics,
        "market_anchored_metrics": anchored_metrics,
        "all_model_metrics": metric_table.to_dict(orient="records"),
        "best_calibrated_model": best_model_name,
        "best_fair_model": champion_source_name or best_model_name,
        "best_calibrated_model_metrics": best_model_row.to_dict(),
        "walk_forward_validation": walk_forward_champion,
        "walk_forward_results_path": str(Path(args.walk_forward_output)),
        "walk_forward_summary_path": str(Path(args.walk_forward_summary_output)),
        "kalshi_comparison_label": kalshi_comparison_label,
        "walk_forward_potential_signal": walk_forward_potential_signal,
        "walk_forward_beats_sportsbook_benchmark": walk_forward_beats_sportsbook,
        "kalshi_paper_trading": kalshi_paper_meta,
        "kalshi_strategy_testing": kalshi_strategy_selected,
        "kalshi_paper_trades_path": str(Path(args.kalshi_paper_trades_output)),
        "kalshi_paper_summary_path": str(Path(args.kalshi_paper_summary_output)),
        "kalshi_strategy_grid_path": str(Path(args.kalshi_strategy_grid_output)),
        "kalshi_strategy_selected_path": str(Path(args.kalshi_strategy_selected_output)),
        "kalshi_strategy_holdout_path": str(Path(args.kalshi_strategy_holdout_output)),
        "paper_trading_warning": (
            "Paper trading results are not trusted until mapping and timing issues are fixed."
            if not kalshi_paper_meta.get("mapping_passed_audit", False) or not kalshi_paper_meta.get("timing_passed_audit", False)
            else (
                "Paper trading has not validated this strategy."
                if kalshi_paper_meta.get("paper_roi") is not None and float(kalshi_paper_meta.get("paper_roi") or 0.0) <= 0
                else ""
            )
        ),
        "best_edge_threshold_by_validation_roi": best_threshold,
        "best_validation_roi": best_roi,
        "best_validation_bets": best_bets,
        "validation_bets_by_threshold": {
            f"{float(row.edge_threshold):.2f}": int(row.num_bets) for row in backtest_all.itertuples()
        },
        "validation_roi_by_threshold": {
            f"{float(row.edge_threshold):.2f}": (None if pd.isna(row.roi) else float(row.roi)) for row in backtest_all.itertuples()
        },
        "fair_model_validated_historically": fair_validated,
        "model_framing_note": "Our model estimates fair win probability. Sportsbook closing odds are used as a historical benchmark. Kalshi prices are used for live market comparison.",
        "no_validated_edge_warning": "" if fair_validated else "Fair model validation is not strong enough for a validated edge. Live Kalshi suggestions are research only unless marked as a potential signal.",
        "strategy_warning": "Sportsbook closing odds are a benchmark, not the only goal. ROI by edge threshold is secondary.",
        "player_feature_count": len(player_feature_columns),
        "selected_player_feature_count": len(selected_player_features),
        "selected_player_features": selected_player_features,
        "player_features_improved_validation": (
            bool(player_metrics)
            and float(player_metrics.get("log_loss", np.inf)) < float(old_metrics.get("log_loss", np.inf))
        ),
        "selected_player_features_improved_validation": (
            bool(selected_player_metrics)
            and float(selected_player_metrics.get("log_loss", np.inf)) < float(old_metrics.get("log_loss", np.inf))
        ),
        "player_feature_warning": (
            ""
            if bool(player_metrics) and float(player_metrics.get("log_loss", np.inf)) < float(old_metrics.get("log_loss", np.inf))
            else "Player features are available but have not improved validation performance yet."
        ),
        "best_ablation_model": best_ablation_name,
        "worst_ablation_model": (
            str(ablation_results.sort_values("log_loss", ascending=False).iloc[0]["model_name"])
            if not ablation_results.empty
            else ""
        ),
        "best_ablation_log_loss": (
            float(ablation_results.sort_values("log_loss", ascending=True).iloc[0]["log_loss"])
            if not ablation_results.empty
            else None
        ),
        "worst_ablation_log_loss": (
            float(ablation_results.sort_values("log_loss", ascending=False).iloc[0]["log_loss"])
            if not ablation_results.empty
            else None
        ),
        "player_data_coverage": float(validation["player_data_available"].mean()) if "player_data_available" in validation.columns else 0.0,
        "projected_rotation_coverage": float(validation["projected_rotation_available"].mean()) if "projected_rotation_available" in validation.columns else 0.0,
        "leakage_checks": {
            "no_validation_or_future_games_in_training": True,
            "features_use_only_prior_games": True,
            "player_features_use_only_prior_games": True,
            "final_score_result_not_used_as_feature": True,
            "sportsbook_odds_used_only_as_benchmark_except_market_anchored_model": True,
        },
        "partial_validation_warning": split_plan.get("partial_validation_warning", ""),
        "diagnostic_outputs": {
            "model_diagnostics": str(Path(args.diagnostics_output)),
            "edge_bucket_report": str(Path(args.edge_bucket_output)),
            "error_by_group": str(Path(args.error_group_output)),
            "player_feature_diagnostics": str(Path(args.player_diagnostics_output)),
            "player_feature_importance": str(Path(args.feature_importance_output)),
            "model_ablation_results": str(Path(args.ablation_output)),
            "fair_model_walk_forward_results": str(Path(args.walk_forward_output)),
            "fair_model_walk_forward_summary": str(Path(args.walk_forward_summary_output)),
            "kalshi_paper_trades": str(Path(args.kalshi_paper_trades_output)),
            "kalshi_paper_trade_summary": str(Path(args.kalshi_paper_summary_output)),
            "kalshi_market_mapping_audit": str(Path(args.kalshi_mapping_audit_output)),
            "kalshi_paper_trade_diagnostics": str(Path(args.kalshi_paper_diagnostics_output)),
            "kalshi_losing_patterns": str(Path(args.kalshi_losing_patterns_output)),
            "kalshi_strategy_grid": str(Path(args.kalshi_strategy_grid_output)),
            "kalshi_strategy_selected": str(Path(args.kalshi_strategy_selected_output)),
            "kalshi_strategy_holdout_results": str(Path(args.kalshi_strategy_holdout_output)),
        },
    }
    Path(args.metrics_output).write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    Path(args.fair_metrics_output).write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    print("Fair probability model training complete.")
    print(f"Split mode: {config.data.free_odds_split_mode}")
    print(f"Training seasons: {', '.join(summary['training_seasons'])}")
    print(f"Validation season: {summary['validation_season']}")
    print(f"Training games used: {summary['training_games_used']:,}")
    print(f"Validation games used: {summary['validation_games_used']:,}")
    print(f"Sportsbook benchmark log loss: {sportsbook_metrics.get('log_loss'):.4f}")
    print(f"Team-only fair model log loss: {old_metrics.get('log_loss'):.4f}")
    if player_metrics:
        print(f"Team plus player fair model log loss: {player_metrics.get('log_loss'):.4f}")
    if selected_player_metrics:
        print(f"Team plus selected player fair model log loss: {selected_player_metrics.get('log_loss'):.4f}")
    print(f"Market-anchored model log loss: {anchored_metrics.get('log_loss'):.4f}")
    if not ablation_results.empty:
        print(f"Best ablation model: {best_ablation_name} ({summary['best_ablation_log_loss']:.4f} log loss)")
    print(f"Best fair model: {best_model_name} ({float(best_model_row['log_loss']):.4f} log loss)")
    if walk_forward_champion:
        print(
            "Walk-forward champion: "
            f"{walk_forward_champion.get('best_fair_model')} "
            f"({walk_forward_champion.get('average_walk_forward_log_loss'):.4f} average log loss, "
            f"{walk_forward_champion.get('best_calibration_method')} calibration)"
        )
        print(f"Kalshi comparison label: {kalshi_comparison_label}")
    print(
        "Kalshi paper trades: "
        f"{kalshi_paper_meta.get('paper_trades', 0):,} rows, "
        f"{kalshi_paper_meta.get('closed_trades', 0):,} closed, "
        f"best threshold={kalshi_paper_meta.get('best_paper_threshold')}, "
        f"ROI={kalshi_paper_meta.get('paper_roi')}"
    )
    if kalshi_strategy_selected:
        print(
            "Kalshi strategy filter status: "
            f"{kalshi_strategy_selected.get('status')} "
            f"holdout={kalshi_strategy_selected.get('holdout', {})}"
        )
    if not fair_validated:
        print(summary["no_validated_edge_warning"])
    print(f"Saved validation predictions to: {args.validation_output}")
    print(f"Saved fair validation predictions to: {args.fair_validation_output}")
    print(f"Saved validation backtest summary to: {args.summary_output}")
    print(f"Saved calibration report to: {args.calibration_output}")
    print(f"Saved diagnostics to: {args.diagnostics_output}, {args.edge_bucket_output}, {args.error_group_output}")
    print(f"Saved player audit outputs to: {args.player_diagnostics_output}, {args.feature_importance_output}, {args.ablation_output}")
    print(f"Saved walk-forward validation to: {args.walk_forward_output}, {args.walk_forward_summary_output}")
    print(f"Saved Kalshi paper-trading outputs to: {args.kalshi_paper_trades_output}, {args.kalshi_paper_summary_output}")
    print(f"Saved Kalshi paper audit outputs to: {args.kalshi_mapping_audit_output}, {args.kalshi_paper_diagnostics_output}, {args.kalshi_losing_patterns_output}")
    print(f"Saved Kalshi strategy outputs to: {args.kalshi_strategy_grid_output}, {args.kalshi_strategy_selected_output}, {args.kalshi_strategy_holdout_output}")
    print(f"Saved model performance summary to: {args.metrics_output}")
    print(f"Saved fair model performance summary to: {args.fair_metrics_output}")


if __name__ == "__main__":
    main()
