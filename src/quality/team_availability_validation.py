"""Validation report for international soccer team availability inputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.injuries_loader import summarize_team_availability
from data.team_name_map import team_match_key


STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"

ALLOWED_STATUSES = {"out", "doubtful", "questionable", "probable", "available", "unknown"}
STATUS_ORDER = ["out", "doubtful", "questionable", "probable", "available", "unknown"]
STALE_AFTER_HOURS = 48.0


def build_team_availability_validation_report(
    fixtures: pd.DataFrame,
    injuries: pd.DataFrame,
    raw_injuries: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    allow_unknown_player_rows: bool = False,
) -> dict[str, Any]:
    """Return a structured validation report for fixture-team availability."""

    as_of = pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else pd.Timestamp.now().normalize()
    issues: list[str] = []
    warnings: list[str] = []

    fixture_teams = _fixture_teams(fixtures)
    fixture_team_keys = {team_match_key(team): team for team in fixture_teams}

    work = injuries.copy() if injuries is not None else pd.DataFrame()
    for column in ["team", "player_name", "status", "importance_score", "last_updated", "source", "notes"]:
        if column not in work.columns:
            work[column] = ""

    work["team"] = work["team"].map(_clean_text)
    work["player_name"] = work["player_name"].map(_clean_text)
    work["status"] = work["status"].map(lambda value: str(value).strip().lower() or "unknown")
    work["last_updated"] = pd.to_datetime(work["last_updated"], errors="coerce")
    work["importance_score"] = pd.to_numeric(work["importance_score"], errors="coerce")

    injury_teams = sorted(t for t in work["team"].dropna().astype(str).unique() if t)
    injury_team_keys = {team_match_key(team): team for team in injury_teams}
    missing_teams = sorted(team for key, team in fixture_team_keys.items() if key not in injury_team_keys)
    teams_not_in_fixtures = sorted(team for key, team in injury_team_keys.items() if key not in fixture_team_keys)
    covered = len(fixture_teams) - len(missing_teams)
    coverage_pct = (covered / len(fixture_teams) * 100.0) if fixture_teams else 0.0

    if missing_teams:
        warnings.append(f"No availability rows for {len(missing_teams)} fixture team(s): {missing_teams}.")
    if teams_not_in_fixtures:
        warnings.append(f"{len(teams_not_in_fixtures)} availability team(s) are not in current fixtures.")

    rows_loaded = int(len(work))
    statuses = work["status"].astype(str).str.lower()
    invalid_statuses = sorted(set(statuses) - ALLOWED_STATUSES)
    invalid_status_rows = int((~statuses.isin(ALLOWED_STATUSES)).sum())
    if invalid_statuses:
        warnings.append(
            f"Strong warning: invalid availability status values found: {invalid_statuses}."
        )
    unknown_status_rows = int(statuses.eq("unknown").sum())
    if unknown_status_rows:
        warnings.append(f"{unknown_status_rows} availability row(s) have unknown status.")

    missing_player_names = int(work["player_name"].eq("").sum())
    unknown_player_rows = int(work["player_name"].str.lower().str.match(r"^unknown player( \d+)?$").sum())
    if missing_player_names:
        warnings.append(f"{missing_player_names} availability row(s) have missing player_name.")
    if unknown_player_rows and not allow_unknown_player_rows:
        warnings.append(f"{unknown_player_rows} availability row(s) use unknown player placeholders.")

    duplicate_team_player_rows = int(
        work.assign(_player_key=work["player_name"].str.lower())
        .duplicated(subset=["team", "_player_key"], keep=False)
        .sum()
    )
    if duplicate_team_player_rows:
        warnings.append(f"{duplicate_team_player_rows} duplicate team/player availability row(s) found.")

    if raw_injuries is not None and not raw_injuries.empty:
        missing_importance_scores = _missing_raw_importance(raw_injuries)
    else:
        missing_importance_scores = int(work["importance_score"].isna().sum())
    if missing_importance_scores:
        warnings.append(f"{missing_importance_scores} availability row(s) are missing importance_score.")

    invalid_date_rows = int(work["last_updated"].isna().sum()) if rows_loaded else 0
    if invalid_date_rows:
        issues.append(f"{invalid_date_rows} availability row(s) have invalid or missing last_updated.")

    stale_mask = pd.Series(False, index=work.index)
    if rows_loaded and pd.notna(as_of):
        age_hours = (as_of - work["last_updated"]).dt.total_seconds() / 3600.0
        stale_mask = work["last_updated"].isna() | (age_hours > STALE_AFTER_HOURS)
    stale_rows = int(stale_mask.sum()) if rows_loaded else 0
    if stale_rows:
        warnings.append(f"{stale_rows} availability row(s) are older than 48 hours or undated.")

    status_counts = {status: int(statuses.eq(status).sum()) for status in STATUS_ORDER}
    summary = summarize_team_availability(work, as_of) if rows_loaded else pd.DataFrame()
    team_rows = _build_team_rows(fixture_teams, work, summary, stale_mask)

    overall = STATUS_FAIL if issues else (STATUS_WARNING if warnings else STATUS_PASS)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "as_of_date": as_of.isoformat() if pd.notna(as_of) else None,
        "coverage": {
            "total_fixture_teams": int(len(fixture_teams)),
            "fixture_teams_with_availability": int(covered),
            "fixture_teams_missing_availability": int(len(missing_teams)),
            "coverage_percentage": round(float(coverage_pct), 2),
            "missing_teams": missing_teams,
        },
        "injury_data": {
            "rows_loaded": rows_loaded,
            "valid_status_rows": int(rows_loaded - invalid_status_rows),
            "invalid_status_rows": invalid_status_rows,
            "invalid_statuses": invalid_statuses,
            "stale_rows_older_than_48h": stale_rows,
            "missing_player_names": missing_player_names,
            "unknown_player_rows": unknown_player_rows,
            "missing_importance_scores": missing_importance_scores,
            "teams_not_found_in_fixtures": teams_not_in_fixtures,
            "duplicate_team_player_rows": duplicate_team_player_rows,
            "status_counts": status_counts,
        },
        "team_rows": team_rows,
        "issues": issues,
        "warnings": warnings,
    }


def render_team_availability_markdown(report: dict[str, Any]) -> str:
    coverage = report.get("coverage", {})
    injury = report.get("injury_data", {})
    lines = [
        "# Team Availability Validation",
        "",
        f"**Overall status: {report.get('overall_status', STATUS_WARNING)}**",
        "",
        "## Fixture Coverage",
        f"- Total fixture teams: {coverage.get('total_fixture_teams', 0)}",
        f"- Teams with availability: {coverage.get('fixture_teams_with_availability', 0)}",
        f"- Missing teams: {coverage.get('fixture_teams_missing_availability', 0)}",
        f"- Coverage percentage: {coverage.get('coverage_percentage', 0)}%",
        f"- Missing team list: {_fmt_list(coverage.get('missing_teams', []))}",
        "",
        "## Injury Data",
        f"- Rows loaded: {injury.get('rows_loaded', 0)}",
        f"- Invalid status rows: {injury.get('invalid_status_rows', 0)}",
        f"- Stale rows older than 48h: {injury.get('stale_rows_older_than_48h', 0)}",
        f"- Missing player names: {injury.get('missing_player_names', 0)}",
        f"- Missing importance scores: {injury.get('missing_importance_scores', 0)}",
        f"- Teams not found in fixtures: {_fmt_list(injury.get('teams_not_found_in_fixtures', []))}",
        f"- Duplicate team/player rows: {injury.get('duplicate_team_player_rows', 0)}",
        f"- Status counts: {injury.get('status_counts', {})}",
        "",
        "## Issues",
        _fmt_list(report.get("issues", [])),
        "",
        "## Warnings",
        _fmt_list(report.get("warnings", [])),
    ]
    return "\n".join(lines) + "\n"


def save_team_availability_validation_report(
    report: dict[str, Any],
    output_dir: str | Path = "data/reports",
) -> tuple[Path, Path]:
    """Write JSON and Markdown validation reports."""

    import json

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "team_availability_validation.json"
    md_path = target / "team_availability_validation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_team_availability_markdown(report), encoding="utf-8")
    return json_path, md_path


def _fixture_teams(fixtures: pd.DataFrame) -> list[str]:
    teams: set[str] = set()
    if fixtures is None or fixtures.empty:
        return []
    for column in ("team_a", "team_b"):
        if column in fixtures.columns:
            teams.update(_clean_text(value) for value in fixtures[column].dropna().tolist())
    return sorted(team for team in teams if team)


def _build_team_rows(
    fixture_teams: list[str],
    injuries: pd.DataFrame,
    summary: pd.DataFrame,
    stale_mask: pd.Series,
) -> list[dict[str, Any]]:
    summary_idx = summary.set_index("team") if summary is not None and not summary.empty else pd.DataFrame()
    injury_groups = injuries.copy()
    injury_groups["_is_stale_row"] = stale_mask if len(stale_mask) == len(injury_groups) else False
    rows: list[dict[str, Any]] = []
    for team in fixture_teams:
        team_rows = injury_groups[injury_groups["team"].map(team_match_key) == team_match_key(team)]
        has = not team_rows.empty
        statuses = team_rows["status"].astype(str).str.lower() if has else pd.Series(dtype=str)
        summary_row = summary_idx.loc[team] if has and team in summary_idx.index else {}
        rows.append(
            {
                "team": team,
                "has_availability_data": bool(has),
                "players_listed": int(len(team_rows)),
                "key_players_out": int(_summary_value(summary_row, "key_players_out", 0)),
                "players_out": int(statuses.eq("out").sum()) if has else 0,
                "questionable_players": int(statuses.eq("questionable").sum()) if has else 0,
                "stale_data_warning": bool(team_rows["_is_stale_row"].any()) if has else False,
                "last_updated": _max_datetime_text(team_rows["last_updated"]) if has else "",
                "source": _join_unique(team_rows.get("source", pd.Series(dtype=str))) if has else "",
                "notes": _join_unique(team_rows.get("notes", pd.Series(dtype=str))) if has else "",
            }
        )
    return rows


def _summary_value(row: Any, key: str, default: Any) -> Any:
    try:
        value = row.get(key, default)
    except AttributeError:
        return default
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return value


def _missing_raw_importance(raw: pd.DataFrame) -> int:
    candidates = [c for c in raw.columns if str(c).strip().lower() in {"importance_score", "importance", "impact"}]
    if not candidates:
        return int(len(raw))
    values = raw[candidates[0]]
    text = values.astype(str).str.strip().str.lower()
    return int((values.isna() | text.isin({"", "nan", "none"})).sum())


def _max_datetime_text(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return ""
    return dates.max().isoformat()


def _join_unique(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _fmt_list(values: list[Any]) -> str:
    return "none" if not values else ", ".join(str(v) for v in values)
