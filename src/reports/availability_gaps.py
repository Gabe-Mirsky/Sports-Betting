"""Audit missing manual/free availability statuses against the generated template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.injury_availability import normalize_availability_reports
from data.team_aliases import normalize_team_abbr


def _key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce").dt.date.astype(str)
    output["team_abbr"] = output["team_abbr"].map(normalize_team_abbr)
    output["player_name_key"] = output["player_name"].fillna("").astype(str).str.strip().str.lower()
    return output


def build_availability_gap_report(
    template: pd.DataFrame,
    availability: pd.DataFrame,
    high_impact_minutes: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return template rows whose status has not been filled in availability.csv."""

    if template.empty:
        return pd.DataFrame(), {
            "status": "no_template",
            "template_rows": 0,
            "availability_rows": int(len(availability)),
            "missing_rows": 0,
            "high_impact_missing_rows": 0,
        }

    required = {"game_date", "team_abbr", "player_name", "status", "impact_weight"}
    missing = sorted(required - set(template.columns))
    if missing:
        raise ValueError(f"Availability template is missing columns: {missing}")

    template_rows = _key_columns(template)
    template_rows["impact_weight"] = pd.to_numeric(template_rows["impact_weight"], errors="coerce").fillna(0.0)
    if availability.empty:
        completed = pd.DataFrame(columns=["game_date", "team_abbr", "player_name_key", "availability_status"])
    else:
        completed = normalize_availability_reports(availability)
        completed = _key_columns(completed)
        completed = completed[completed["availability_status"].astype(str).str.strip().ne("")]
        completed = completed[["game_date", "team_abbr", "player_name_key", "availability_status"]].drop_duplicates()

    merged = template_rows.merge(
        completed,
        on=["game_date", "team_abbr", "player_name_key"],
        how="left",
    )
    template_status = merged["status"].fillna("").astype(str).str.strip()
    availability_status = merged.get("availability_status", pd.Series("", index=merged.index)).fillna("").astype(str).str.strip()
    missing_status = template_status.eq("") & availability_status.eq("")
    gaps = merged[missing_status].copy()
    sort_columns = [column for column in ["game_date", "game_id", "team_abbr"] if column in gaps.columns]
    if sort_columns:
        gaps = gaps.sort_values([*sort_columns, "impact_weight"], ascending=[True] * len(sort_columns) + [False])
    output_columns = [
        "report_date",
        "game_date",
        "game_id",
        "team_abbr",
        "opponent_abbr",
        "home_away",
        "player_id",
        "player_name",
        "impact_weight",
        "impact_weight_source",
        "impact_prior_games",
        "impact_avg_box_score_value_last10",
    ]
    gaps = gaps[[column for column in output_columns if column in gaps.columns]].reset_index(drop=True)
    high_impact = gaps[pd.to_numeric(gaps["impact_weight"], errors="coerce").fillna(0.0) >= float(high_impact_minutes)]
    games_missing = int(gaps["game_id"].nunique()) if "game_id" in gaps.columns else 0
    status = "needs_availability_input" if len(gaps) else "complete"
    summary = {
        "status": status,
        "template_rows": int(len(template_rows)),
        "availability_rows": int(len(completed)),
        "missing_rows": int(len(gaps)),
        "high_impact_missing_rows": int(len(high_impact)),
        "games_with_missing_statuses": games_missing,
        "high_impact_minutes": float(high_impact_minutes),
        "note": "Fill these statuses from a free/allowed source in data/raw/nba/injuries/availability.csv.",
    }
    return gaps, summary


def save_availability_gap_report(
    gaps: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "availability_gap",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(output_root / f"{prefix}_missing_statuses.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
