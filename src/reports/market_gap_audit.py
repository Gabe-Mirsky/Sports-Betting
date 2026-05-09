"""Audit where Kalshi prices beat the model and where the model beats Kalshi."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = [
    "elo_diff_pre",
    "rest_diff",
    "last_5_win_pct_diff",
    "last_10_win_pct_diff",
    "last_5_point_diff_diff",
    "last_10_point_diff_diff",
    "season_win_pct_diff",
    "season_avg_margin_diff",
    "player_top8_minutes_last10_diff",
    "player_top8_points_last10_diff",
    "player_top8_value_last10_diff",
    "player_key_absence_minutes_last_game_diff",
    "player_top8_minutes_gap_last_game_diff",
    "availability_out_or_doubtful_diff",
    "availability_questionable_or_worse_diff",
    "availability_projected_minutes_lost_diff",
    "availability_status_severity_weighted_diff",
    "home_is_back_to_back",
    "away_is_back_to_back",
]

SEGMENT_COLUMNS = [
    "season_phase",
    "yes_home_away",
    "market_favorite_bucket",
    "model_market_pick_agreement",
    "prob_gap_bucket",
    "market_price_bucket",
    "volume_bucket",
    "yes_rest_context",
    "yes_player_availability_proxy",
    "availability_report_context",
]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _log_loss(actual: pd.Series, probability: pd.Series) -> float:
    y = actual.astype(float).to_numpy()
    p = probability.astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    if len(y) == 0:
        return float("nan")
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _accuracy(actual: pd.Series, probability: pd.Series) -> float:
    if len(actual) == 0:
        return float("nan")
    return float(((probability.astype(float) >= 0.5) == actual.astype(bool)).mean())


def _probability_bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        values.astype(float).clip(lower=bins[0], upper=bins[-1]),
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)


def _volume_bucket(values: pd.Series) -> pd.Series:
    bins = [-0.01, 0, 10, 100, 1000, 10000, float("inf")]
    labels = ["0", "1-10", "10-100", "100-1k", "1k-10k", "10k+"]
    return pd.cut(pd.to_numeric(values, errors="coerce").fillna(0), bins=bins, labels=labels).astype(str)


def _yes_perspective_value(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    yes_is_home = frame["yes_team_abbr"].astype(str).eq(frame["home_team_abbr"].astype(str))
    return values.where(yes_is_home, -values)


def _attach_model_context(rows: pd.DataFrame, modeling: pd.DataFrame | None) -> pd.DataFrame:
    if modeling is None or modeling.empty or "game_id" not in modeling.columns:
        return rows
    context_columns = ["game_id", *[column for column in CONTEXT_COLUMNS if column in modeling.columns]]
    if len(context_columns) == 1:
        return rows
    context = modeling[context_columns].copy()
    context["game_id"] = context["game_id"].astype(str)
    output = rows.merge(context.drop_duplicates("game_id"), on="game_id", how="left")
    for column in CONTEXT_COLUMNS:
        if column not in output.columns or not column.endswith("_diff"):
            continue
        output[f"yes_{column}"] = _yes_perspective_value(output, column)
    if {"home_is_back_to_back", "away_is_back_to_back"}.issubset(output.columns):
        yes_is_home = output["yes_team_abbr"].astype(str).eq(output["home_team_abbr"].astype(str))
        output["yes_is_back_to_back"] = np.where(
            yes_is_home,
            pd.to_numeric(output["home_is_back_to_back"], errors="coerce").fillna(0),
            pd.to_numeric(output["away_is_back_to_back"], errors="coerce").fillna(0),
        )
    return output


def build_market_gap_detail(predictions: pd.DataFrame, modeling: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return row-level model-vs-market diagnostics."""

    if predictions.empty:
        return pd.DataFrame()
    required = {"game_id", "market_ticker", "model_yes_prob", "actual_yes_win"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Market gap audit missing required columns: {sorted(missing)}")

    rows = predictions.copy()
    rows["game_id"] = rows["game_id"].astype(str)
    if "market_yes_prob" not in rows.columns:
        if "yes_ask" in rows.columns:
            rows["market_yes_prob"] = pd.to_numeric(rows["yes_ask"], errors="coerce") / 100.0
        elif "yes_mid_cents" in rows.columns:
            rows["market_yes_prob"] = pd.to_numeric(rows["yes_mid_cents"], errors="coerce") / 100.0
        else:
            raise ValueError("Market gap audit needs market_yes_prob, yes_ask, or yes_mid_cents.")
    if "blended_yes_prob" not in rows.columns:
        rows["blended_yes_prob"] = np.nan

    rows["actual_yes_win"] = _coerce_bool(rows["actual_yes_win"]).astype(int)
    rows["model_yes_prob"] = pd.to_numeric(rows["model_yes_prob"], errors="coerce")
    rows["market_yes_prob"] = pd.to_numeric(rows["market_yes_prob"], errors="coerce")
    rows["blended_yes_prob"] = pd.to_numeric(rows["blended_yes_prob"], errors="coerce")
    rows = rows.dropna(subset=["model_yes_prob", "market_yes_prob", "actual_yes_win"]).copy()

    rows["model_pick_yes"] = rows["model_yes_prob"] >= 0.5
    rows["market_pick_yes"] = rows["market_yes_prob"] >= 0.5
    rows["blend_pick_yes"] = rows["blended_yes_prob"] >= 0.5
    rows["model_correct"] = rows["model_pick_yes"].astype(int).eq(rows["actual_yes_win"])
    rows["market_correct"] = rows["market_pick_yes"].astype(int).eq(rows["actual_yes_win"])
    rows["blend_correct"] = rows["blend_pick_yes"].astype(int).eq(rows["actual_yes_win"])
    rows["model_abs_error"] = (rows["model_yes_prob"] - rows["actual_yes_win"]).abs()
    rows["market_abs_error"] = (rows["market_yes_prob"] - rows["actual_yes_win"]).abs()
    rows["blend_abs_error"] = (rows["blended_yes_prob"] - rows["actual_yes_win"]).abs()
    rows["kalshi_edge_over_model"] = rows["model_abs_error"] - rows["market_abs_error"]
    rows["model_edge_over_kalshi"] = -rows["kalshi_edge_over_model"]
    rows["prob_gap_market_minus_model"] = rows["market_yes_prob"] - rows["model_yes_prob"]
    rows["abs_prob_gap"] = rows["prob_gap_market_minus_model"].abs()
    rows["yes_home_away"] = np.where(
        rows["yes_team_abbr"].astype(str).eq(rows["home_team_abbr"].astype(str)),
        "yes_home",
        "yes_away",
    )
    rows = _attach_model_context(rows, modeling)
    rows["season_phase"] = np.where(
        rows.get("is_playoffs", pd.Series(False, index=rows.index)).astype(bool),
        "playoffs",
        "regular_season",
    )
    rows["market_favorite_bucket"] = np.where(rows["market_yes_prob"] >= 0.5, "market_favorite", "market_underdog")
    rows["model_market_pick_agreement"] = np.where(
        rows["model_pick_yes"].eq(rows["market_pick_yes"]),
        "same_pick",
        "opposite_pick",
    )
    rows["prob_gap_bucket"] = _probability_bucket(
        rows["abs_prob_gap"],
        [0.0, 0.05, 0.10, 0.20, 0.35, 1.0],
        ["0-5pp", "5-10pp", "10-20pp", "20-35pp", "35pp+"],
    )
    rows["market_price_bucket"] = _probability_bucket(
        rows["market_yes_prob"],
        [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0],
        ["0-20c", "20-35c", "35-50c", "50-65c", "65-80c", "80-100c"],
    )
    rows["volume_bucket"] = _volume_bucket(rows["volume"] if "volume" in rows.columns else pd.Series(0, index=rows.index))
    yes_rest = pd.to_numeric(rows.get("yes_rest_diff", pd.Series(np.nan, index=rows.index)), errors="coerce")
    rows["yes_rest_context"] = np.select(
        [yes_rest >= 2, yes_rest <= -2, yes_rest.notna()],
        ["yes_2plus_days_more_rest", "yes_2plus_days_less_rest", "rest_similar"],
        default="rest_unknown",
    )
    yes_minutes_gap = pd.to_numeric(
        rows.get("yes_player_top8_minutes_gap_last_game_diff", pd.Series(np.nan, index=rows.index)),
        errors="coerce",
    )
    rows["yes_player_availability_proxy"] = np.select(
        [yes_minutes_gap >= 20, yes_minutes_gap <= -20, yes_minutes_gap.notna()],
        ["yes_more_rotation_gap", "opponent_more_rotation_gap", "rotation_gap_similar"],
        default="rotation_gap_unknown",
    )
    availability_present = (
        rows.get("home_availability_report_present", pd.Series(0, index=rows.index)).fillna(0).astype(float)
        + rows.get("away_availability_report_present", pd.Series(0, index=rows.index)).fillna(0).astype(float)
    )
    rows["availability_report_context"] = np.where(availability_present > 0, "availability_report_present", "no_availability_report")
    return rows.sort_values(["game_date", "market_ticker"]).reset_index(drop=True)


def _segment_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    actual = frame["actual_yes_win"].astype(int)
    model_prob = frame["model_yes_prob"].astype(float)
    market_prob = frame["market_yes_prob"].astype(float)
    output: dict[str, Any] = {
        "rows": int(len(frame)),
        "avg_model_abs_error": float(frame["model_abs_error"].mean()),
        "avg_market_abs_error": float(frame["market_abs_error"].mean()),
        "avg_kalshi_edge_over_model": float(frame["kalshi_edge_over_model"].mean()),
        "market_beats_model_rate": float((frame["kalshi_edge_over_model"] > 1e-9).mean()),
        "model_beats_market_rate": float((frame["kalshi_edge_over_model"] < -1e-9).mean()),
        "avg_abs_prob_gap": float(frame["abs_prob_gap"].mean()),
        "model_accuracy": _accuracy(actual, model_prob),
        "market_accuracy": _accuracy(actual, market_prob),
        "model_log_loss": _log_loss(actual, model_prob),
        "market_log_loss": _log_loss(actual, market_prob),
    }
    if frame["blended_yes_prob"].notna().any():
        blend = frame["blended_yes_prob"].astype(float)
        output["blend_accuracy"] = _accuracy(actual, blend)
        output["blend_log_loss"] = _log_loss(actual, blend)
    return output


def build_market_gap_segment_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Summarize where market prices outperformed by interpretable segment."""

    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for column in SEGMENT_COLUMNS:
        if column not in detail.columns:
            continue
        for value, frame in detail.groupby(column, dropna=False, observed=False):
            if frame.empty:
                continue
            rows.append({"segment": column, "value": str(value), **_segment_metrics(frame)})
    return (
        pd.DataFrame(rows)
        .sort_values(["avg_kalshi_edge_over_model", "rows"], ascending=[False, False])
        .reset_index(drop=True)
    )


def summarize_market_gap(detail: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    """Create a compact JSON summary for the market gap audit."""

    if detail.empty:
        return {"rows": 0, "status": "empty"}
    timeline = (
        f"{pd.to_datetime(detail['game_date'], errors='coerce').min().date()} "
        f"to {pd.to_datetime(detail['game_date'], errors='coerce').max().date()}"
        if "game_date" in detail.columns
        else "n/a"
    )
    large_segments = segments[segments["rows"] >= 25].head(10) if not segments.empty else pd.DataFrame()
    summary = {
        "rows": int(len(detail)),
        "timeline": timeline,
        "market_beats_model_rows": int((detail["kalshi_edge_over_model"] > 1e-9).sum()),
        "model_beats_market_rows": int((detail["kalshi_edge_over_model"] < -1e-9).sum()),
        "market_beats_model_rate": float((detail["kalshi_edge_over_model"] > 1e-9).mean()),
        "model_beats_market_rate": float((detail["kalshi_edge_over_model"] < -1e-9).mean()),
        "avg_model_abs_error": float(detail["model_abs_error"].mean()),
        "avg_market_abs_error": float(detail["market_abs_error"].mean()),
        "avg_kalshi_edge_over_model": float(detail["kalshi_edge_over_model"].mean()),
        "model_accuracy": _accuracy(detail["actual_yes_win"], detail["model_yes_prob"]),
        "market_accuracy": _accuracy(detail["actual_yes_win"], detail["market_yes_prob"]),
        "model_log_loss": _log_loss(detail["actual_yes_win"], detail["model_yes_prob"]),
        "market_log_loss": _log_loss(detail["actual_yes_win"], detail["market_yes_prob"]),
        "opposite_pick_rows": int(detail["model_market_pick_agreement"].eq("opposite_pick").sum()),
        "opposite_pick_market_accuracy": _accuracy(
            detail.loc[detail["model_market_pick_agreement"].eq("opposite_pick"), "actual_yes_win"],
            detail.loc[detail["model_market_pick_agreement"].eq("opposite_pick"), "market_yes_prob"],
        ),
        "opposite_pick_model_accuracy": _accuracy(
            detail.loc[detail["model_market_pick_agreement"].eq("opposite_pick"), "actual_yes_win"],
            detail.loc[detail["model_market_pick_agreement"].eq("opposite_pick"), "model_yes_prob"],
        ),
        "top_market_advantage_segments_min_25_rows": large_segments.to_dict(orient="records"),
        "note": (
            "Positive avg_kalshi_edge_over_model means Kalshi's pregame probability was closer to the result "
            "than the model. Use these segments to decide which features to improve next."
        ),
    }
    if detail["blended_yes_prob"].notna().any():
        summary["blend_accuracy"] = _accuracy(detail["actual_yes_win"], detail["blended_yes_prob"])
        summary["blend_log_loss"] = _log_loss(detail["actual_yes_win"], detail["blended_yes_prob"])
    return summary


def write_market_gap_audit_outputs(
    predictions: pd.DataFrame,
    modeling: pd.DataFrame | None,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Write row-level, segment-level, and top-gap audit artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail = build_market_gap_detail(predictions, modeling)
    segments = build_market_gap_segment_summary(detail)
    summary = summarize_market_gap(detail, segments)

    detail_path = output / "kalshi_model_gap_audit.csv"
    segments_path = output / "kalshi_model_gap_segments.csv"
    kalshi_best_path = output / "kalshi_beat_model_examples.csv"
    model_best_path = output / "model_beat_kalshi_examples.csv"
    summary_path = output / "kalshi_model_gap_summary.json"

    detail.to_csv(detail_path, index=False)
    segments.to_csv(segments_path, index=False)
    detail.sort_values("kalshi_edge_over_model", ascending=False).head(50).to_csv(kalshi_best_path, index=False)
    detail.sort_values("kalshi_edge_over_model", ascending=True).head(50).to_csv(model_best_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return detail, segments, summary
