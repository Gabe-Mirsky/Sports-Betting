"""Build report artifacts for the matchup-prediction dashboard.

Produces the files the dashboard reads:

    data/reports/matchup_predictions_today.csv
    data/reports/matchup_predictions_today.json
    data/reports/matchup_model_backtest.json
    data/reports/matchup_model_backtest_by_bucket.csv

All outputs are framed as *model probabilities*, never betting odds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.matchup_model import DEFAULT_MODEL_VERSION
from models.prediction_explainer import explain_prediction

# Columns written to the flat CSV (lists are joined with "; ").
_CSV_COLUMNS = [
    "id",
    "sport",
    "league",
    "game_date",
    "team_a",
    "team_b",
    "prob_team_a_win",
    "prob_draw",
    "prob_team_b_win",
    "predicted_outcome",
    "confidence_level",
    "confidence_score",
    "data_quality",
    "key_reasons",
    "main_risks",
    "data_quality_warnings",
    "team_a_availability_present",
    "team_b_availability_present",
    "team_a_availability_source",
    "team_b_availability_source",
    "team_a_availability_manual",
    "team_b_availability_manual",
    "team_a_availability_last_updated",
    "team_b_availability_last_updated",
    "model_version",
]


def _resolve_dir(output_path: str | Path) -> Path:
    """Treat ``output_path`` as a directory (creating it), or use its parent
    when a concrete file path is supplied."""

    path = Path(output_path)
    directory = path.parent if path.suffix else path
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _id_column(df: pd.DataFrame) -> str:
    if "fixture_id" in df.columns:
        return "fixture_id"
    if "game_id" in df.columns:
        return "game_id"
    return ""


def enrich_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Attach confidence/quality/reasons/risks/warnings columns to predictions.

    Idempotent: if the explanation columns already exist they are recomputed so
    the report always reflects the current explainer logic.
    """

    if predictions_df.empty:
        return predictions_df.copy()

    out = predictions_df.copy()
    explanations = out.apply(explain_prediction, axis=1)
    out["confidence_level"] = [e["confidence_level"] for e in explanations]
    out["data_quality"] = [e["data_quality"] for e in explanations]
    out["key_reasons"] = [e["key_reasons"] for e in explanations]
    out["main_risks"] = [e["main_risks"] for e in explanations]
    out["data_quality_warnings"] = [e["data_quality_warnings"] for e in explanations]
    return out


def _records(predictions_df: pd.DataFrame) -> list[dict]:
    id_col = _id_column(predictions_df)
    records = []
    for _, row in predictions_df.iterrows():
        records.append(
            {
                "fixture_id": str(row.get(id_col, "")) if id_col else "",
                "sport": str(row.get("sport", "")),
                "league": str(row.get("league", "")),
                "game_date": _iso(row.get("game_date")),
                "team_a": str(row.get("team_a", "")),
                "team_b": str(row.get("team_b", "")),
                "prob_team_a_win": _round(row.get("prob_team_a_win")),
                "prob_draw": _round(row.get("prob_draw")),
                "prob_team_b_win": _round(row.get("prob_team_b_win")),
                "predicted_outcome": str(row.get("predicted_outcome", "")),
                "confidence_level": str(row.get("confidence_level", "")),
                "confidence_score": _round(row.get("confidence_score")),
                "data_quality": str(row.get("data_quality", "")),
                "key_reasons": list(row.get("key_reasons", []) or []),
                "main_risks": list(row.get("main_risks", []) or []),
                "data_quality_warnings": list(row.get("data_quality_warnings", []) or []),
                "team_a_availability_present": _bool_int(row.get("team_a_availability_present")),
                "team_b_availability_present": _bool_int(row.get("team_b_availability_present")),
                "team_a_availability_source": str(row.get("team_a_availability_source", "")),
                "team_b_availability_source": str(row.get("team_b_availability_source", "")),
                "team_a_availability_manual": _bool_int(row.get("team_a_availability_manual")),
                "team_b_availability_manual": _bool_int(row.get("team_b_availability_manual")),
                "team_a_availability_last_updated": str(row.get("team_a_availability_last_updated", "")),
                "team_b_availability_last_updated": str(row.get("team_b_availability_last_updated", "")),
                "model_version": str(row.get("model_version", DEFAULT_MODEL_VERSION)),
            }
        )
    return records


def _iso(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return ""
        return ts.isoformat()
    except (ValueError, TypeError):
        return str(value)


def _round(value, digits: int = 4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return None


def _bool_int(value) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes", "y"} else 0


def build_today_predictions_report(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write ``matchup_predictions_today.{csv,json}`` into the output directory."""

    directory = _resolve_dir(output_path)
    enriched = enrich_predictions(predictions_df)
    records = _records(enriched)

    json_path = directory / "matchup_predictions_today.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # Flat CSV: join list columns into readable strings.
    csv_rows = []
    for rec in records:
        flat = dict(rec)
        flat["id"] = flat.pop("fixture_id")
        for list_col in ("key_reasons", "main_risks", "data_quality_warnings"):
            flat[list_col] = "; ".join(flat.get(list_col, []))
        csv_rows.append(flat)
    csv_df = pd.DataFrame(csv_rows, columns=_CSV_COLUMNS)
    csv_df.to_csv(directory / "matchup_predictions_today.csv", index=False)


def build_backtest_report(
    metrics: dict,
    buckets_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write ``matchup_model_backtest.json`` and ``..._by_bucket.csv``."""

    directory = _resolve_dir(output_path)
    (directory / "matchup_model_backtest.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if buckets_df is None:
        buckets_df = pd.DataFrame()
    buckets_df.to_csv(directory / "matchup_model_backtest_by_bucket.csv", index=False)
