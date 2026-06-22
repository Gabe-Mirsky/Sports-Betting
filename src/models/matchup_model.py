"""Train and apply no-odds matchup prediction models.

The model maps the contextual features built by
:mod:`features.matchup_features` to outcome probabilities. It supports:

* two-outcome sports (basketball, baseball, hockey, ...) – binary logistic
  regression for ``team_a`` win vs ``team_b`` win;
* three-outcome sports (soccer, ...) – multinomial logistic regression for
  ``team_a`` win / draw / ``team_b`` win.

Optional ``RandomForestClassifier`` / ``HistGradientBoostingClassifier``
back-ends are available via config, with probability calibration when there is
enough data. The first version stays deliberately simple and reliable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.sport_rules import normalize_sport, sport_allows_draws
from features.matchup_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "matchup_baseline_v1"

# Class encoding shared across the pipeline.
CLASS_TEAM_A_WIN = 0
CLASS_DRAW = 1
CLASS_TEAM_B_WIN = 2
_CLASS_MEANING = {
    CLASS_TEAM_A_WIN: "team_a_win",
    CLASS_DRAW: "draw",
    CLASS_TEAM_B_WIN: "team_b_win",
}

PROB_COLUMNS = ["prob_team_a_win", "prob_draw", "prob_team_b_win"]


def normalize_probability_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return ``df`` with ``columns`` rescaled so each row sums to 1.

    Rows whose probabilities sum to (near) zero are replaced with a uniform
    distribution over the columns, so the output is always a valid simplex.
    """

    out = df.copy()
    block = out[columns].astype(float).clip(lower=0.0)
    totals = block.sum(axis=1)
    safe = totals > 1e-12
    out.loc[safe, columns] = block.loc[safe].div(totals[safe], axis=0)
    out.loc[~safe, columns] = 1.0 / len(columns)
    return out


def _build_estimator(model_type: str, draws_enabled: bool, random_state: int) -> Pipeline:
    """Construct the preprocessing + classifier pipeline."""

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model_type = (model_type or "logistic_regression").lower()
    if model_type in {"random_forest", "rf"}:
        classifier = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=-1
        )
    elif model_type in {"gradient_boosting", "hist_gradient_boosting", "hgb"}:
        classifier = HistGradientBoostingClassifier(random_state=random_state)
    else:
        # LogisticRegression auto-selects a multinomial loss for multiclass y.
        classifier = LogisticRegression(max_iter=1000, C=1.0)

    return Pipeline([("pre", pre), ("clf", classifier)])


def _encode_targets(training_df: pd.DataFrame, draws_enabled: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (rows kept, integer class labels) dropping undetermined games."""

    a = training_df["result_team_a_win"].astype(int)
    draw = training_df["result_draw"].astype(int)
    b = training_df["result_team_b_win"].astype(int)
    determined = (a + draw + b) == 1

    if draws_enabled:
        keep = determined
        y = np.where(
            a[keep] == 1,
            CLASS_TEAM_A_WIN,
            np.where(draw[keep] == 1, CLASS_DRAW, CLASS_TEAM_B_WIN),
        )
    else:
        # Two-outcome sports: keep decisive games only.
        keep = determined & (draw == 0)
        y = np.where(a[keep] == 1, CLASS_TEAM_A_WIN, CLASS_TEAM_B_WIN)

    return training_df[keep], y.astype(int)


def train_matchup_model(
    training_df: pd.DataFrame,
    sport: str,
    config: dict | None = None,
) -> dict:
    """Train a matchup model and return a serializable bundle.

    Parameters
    ----------
    training_df:
        Output of :func:`features.matchup_features.build_training_features`.
    sport:
        Sport key – decides whether a draw class is modeled.
    config:
        Optional overrides: ``model_type`` (``logistic_regression`` (default),
        ``random_forest``, ``gradient_boosting``), ``model_version``,
        ``calibrate`` (bool, default True), ``random_state``,
        plus draw-sport overrides for :func:`data.sport_rules.sport_allows_draws`.
    """

    config = config or {}
    sport = normalize_sport(sport)
    draws_enabled = sport_allows_draws(sport, config)
    random_state = int(config.get("random_state", 42))
    model_type = config.get("model_type", "logistic_regression")

    if training_df.empty:
        raise ValueError("Cannot train a matchup model on an empty training frame.")

    rows, y = _encode_targets(training_df, draws_enabled)
    if len(rows) == 0:
        raise ValueError("No decided games available to train on after filtering.")
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError(
            f"Training data has only one outcome class for sport '{sport}'. "
            "Need at least two outcomes to fit a classifier."
        )

    X = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    estimator = _build_estimator(model_type, draws_enabled, random_state)

    # Probability calibration is opt-in. Logistic regression is already well
    # calibrated, and wrapping a well-separated multiclass model in Platt
    # scaling tends to flatten probabilities toward the base rate – so we keep
    # the reliable default (no calibration) unless the caller asks and the
    # model is a tree-based back-end that genuinely benefits from it.
    calibrate = bool(config.get("calibrate", False))
    class_counts = pd.Series(y).value_counts()
    fitted = _fit_with_optional_calibration(estimator, X, y, calibrate, class_counts, model_type)

    classes = [int(c) for c in fitted.classes_]
    bundle = {
        "model": fitted,
        "sport": sport,
        "draws_enabled": draws_enabled,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes_": classes,
        "class_meaning": {c: _CLASS_MEANING[c] for c in classes},
        "model_type": model_type,
        "model_version": config.get("model_version", DEFAULT_MODEL_VERSION),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(rows)),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }
    logger.info(
        "Trained %s matchup model for %s on %d games (draws=%s).",
        model_type,
        sport,
        len(rows),
        draws_enabled,
    )
    return bundle


def _fit_with_optional_calibration(estimator, X, y, calibrate, class_counts, model_type):
    is_tree = (model_type or "").lower() in {"random_forest", "rf", "gradient_boosting", "hist_gradient_boosting", "hgb"}
    if not calibrate or not is_tree or class_counts.min() < 10 or len(X) < 100:
        return estimator.fit(X, y)
    try:
        from sklearn.calibration import CalibratedClassifierCV

        cv = 3 if class_counts.min() >= 15 else 2
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=cv)
        return calibrated.fit(X, y)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Calibration failed (%s); using uncalibrated model.", exc)
        return estimator.fit(X, y)


def predict_matchup_probabilities(
    model_bundle: dict,
    fixture_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-fixture outcome probabilities and a confidence score.

    Output columns include ``prob_team_a_win``, ``prob_draw``,
    ``prob_team_b_win``, ``predicted_outcome``, ``predicted_side``,
    ``confidence_score``, ``model_version`` and ``feature_snapshot_time``, plus
    all identity/feature/context columns from ``fixture_features_df`` (so the
    explainer and report can read them).
    """

    if fixture_features_df.empty:
        return fixture_features_df.copy()

    model = model_bundle["model"]
    numeric = model_bundle.get("numeric_features", NUMERIC_FEATURES)
    categorical = model_bundle.get("categorical_features", CATEGORICAL_FEATURES)
    classes = model_bundle.get("classes_", [CLASS_TEAM_A_WIN, CLASS_TEAM_B_WIN])
    draws_enabled = bool(model_bundle.get("draws_enabled", False))

    X = fixture_features_df[numeric + categorical]
    proba = model.predict_proba(X)
    class_to_col = {cls: idx for idx, cls in enumerate(classes)}

    out = fixture_features_df.copy()
    out["prob_team_a_win"] = _class_prob(proba, class_to_col, CLASS_TEAM_A_WIN)
    out["prob_draw"] = _class_prob(proba, class_to_col, CLASS_DRAW) if draws_enabled else 0.0
    out["prob_team_b_win"] = _class_prob(proba, class_to_col, CLASS_TEAM_B_WIN)
    out = normalize_probability_columns(out, PROB_COLUMNS)

    out = _add_outcome_and_confidence(out)
    out["model_version"] = model_bundle.get("model_version", DEFAULT_MODEL_VERSION)
    out["feature_snapshot_time"] = datetime.now(timezone.utc).isoformat()
    return out


def _class_prob(proba: np.ndarray, class_to_col: dict, cls: int) -> np.ndarray:
    if cls in class_to_col:
        return proba[:, class_to_col[cls]]
    return np.zeros(proba.shape[0])


def _add_outcome_and_confidence(out: pd.DataFrame) -> pd.DataFrame:
    probs = out[PROB_COLUMNS].to_numpy()
    sides = np.array(["team_a", "draw", "team_b"])
    top_idx = probs.argmax(axis=1)
    out["predicted_side"] = sides[top_idx]

    # Confidence score = separation between the top two probabilities (0..1).
    sorted_probs = np.sort(probs, axis=1)
    out["confidence_score"] = (sorted_probs[:, -1] - sorted_probs[:, -2]).clip(0.0, 1.0)

    team_a = out["team_a"].astype(str) if "team_a" in out else pd.Series("Team A", index=out.index)
    team_b = out["team_b"].astype(str) if "team_b" in out else pd.Series("Team B", index=out.index)
    labels = []
    for side, a, b in zip(out["predicted_side"], team_a, team_b):
        if side == "team_a":
            labels.append(f"{a} win")
        elif side == "team_b":
            labels.append(f"{b} win")
        else:
            labels.append("Draw")
    out["predicted_outcome"] = labels
    return out


def save_matchup_model(model_bundle: dict, path: str | Path) -> None:
    """Persist a model bundle to disk with joblib."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, target)
    logger.info("Saved matchup model bundle to %s", target)


def load_matchup_model(path: str | Path) -> dict:
    """Load a model bundle previously saved with :func:`save_matchup_model`."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model bundle not found: {source}")
    return joblib.load(source)
