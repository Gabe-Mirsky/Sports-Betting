"""Validate the real-data input files for the no-odds matchup pipeline.

This module reuses the existing loaders (so it checks exactly what they can
read) and adds cross-file checks: team matching, freshness, future-date sanity.
It is intentionally lenient – weak-but-usable data produces WARNINGs, and only
genuinely broken data produces a FAIL. The CLI wrapper lives in
``scripts/validate_matchup_input_files.py``.

No odds, closing lines, or market data are ever required or checked.
"""

from __future__ import annotations

import difflib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.fixtures_loader import (
    FIXTURE_COLUMNS,
    load_fixtures,
    normalize_fixtures,
    validate_fixtures,
)
from data.injuries_loader import (
    load_injuries,
    normalize_injuries,
    summarize_team_availability,
)
from data.match_results_loader import (
    load_match_results,
    normalize_match_results,
    validate_match_results,
)
from data.sport_rules import normalize_sport
from data.team_name_map import load_team_aliases, team_match_key

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"

DEFAULT_ALIASES_PATH = "data/manual/team_aliases_template.csv"

# Statuses the injury loader is expected to surface (post-normalization).
_ALLOWED_INJURY_STATUSES = {
    "out",
    "doubtful",
    "questionable",
    "probable",
    "available",
    "active",
    "unknown",
}

_MIN_GAMES_PREFERRED = 20
_MIN_GAMES_STRONG_WARN = 5
_INJURY_STALE_HOURS = 48.0
_SCORE_LIKE_COLUMNS = {
    "team_a_score",
    "team_b_score",
    "home_score",
    "away_score",
    "home_points",
    "away_points",
    "home_goals",
    "away_goals",
    "score_home",
    "score_away",
}


def _now() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


def _section() -> dict:
    return {"issues": [], "warnings": []}


def _team_set(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    teams: set[str] = set()
    for column in columns:
        if column in frame.columns:
            teams.update(
                str(value).strip()
                for value in frame[column].dropna().unique()
                if str(value).strip()
            )
    return teams


def validate_results_section(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    today: pd.Timestamp,
) -> dict:
    """Validate historical results (already normalized by the loader)."""

    section = _section()
    loader_report = validate_match_results(normalized)
    section["issues"].extend(loader_report.get("issues", []))
    section["warnings"].extend(loader_report.get("warnings", []))

    n_rows = int(len(normalized))
    section["rows_loaded"] = n_rows
    section["teams_found"] = len(_team_set(normalized, ["team_a", "team_b"]))
    section["date_range"] = [loader_report.get("date_min"), loader_report.get("date_max")]
    section["sports"] = sorted(normalized["sport"].dropna().unique().tolist()) if "sport" in normalized else []
    section["leagues"] = sorted(normalized["league"].dropna().unique().tolist()) if "league" in normalized else []

    if n_rows == 0:
        section["issues"].append("No historical results rows loaded.")
    elif n_rows < _MIN_GAMES_STRONG_WARN:
        section["warnings"].append(
            f"Only {n_rows} historical games — far below the {_MIN_GAMES_PREFERRED} preferred; "
            "the model will be very unreliable."
        )
    elif n_rows < _MIN_GAMES_PREFERRED:
        section["warnings"].append(
            f"Only {n_rows} historical games (fewer than the {_MIN_GAMES_PREFERRED} preferred for a useful model)."
        )

    # Blank teams.
    if "team_a" in normalized.columns:
        blank = int((normalized["team_a"].astype(str).str.strip() == "").sum()
                    + (normalized["team_b"].astype(str).str.strip() == "").sum())
        if blank:
            section["issues"].append(f"{blank} result rows have a blank team_a or team_b.")

    # Date sanity: all-unparseable is fatal (can't order games safely); a
    # future date inside historical results is only a warning.
    if "game_date" in normalized.columns and n_rows:
        dates = pd.to_datetime(normalized["game_date"], errors="coerce")
        if dates.isna().all():
            section["issues"].append("All game_date values are unparseable; cannot order games.")
        n_future = int((dates > today).sum())
        if n_future:
            section["warnings"].append(
                f"{n_future} historical result rows are dated in the future "
                "(after today); confirm these are really completed games."
            )

    return section


def validate_fixtures_section(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    today: pd.Timestamp,
) -> dict:
    """Validate upcoming fixtures (already normalized by the loader)."""

    section = _section()
    loader_report = validate_fixtures(normalized)
    section["issues"].extend(loader_report.get("issues", []))
    section["warnings"].extend(loader_report.get("warnings", []))

    n_rows = int(len(normalized))
    section["rows_loaded"] = n_rows
    section["teams"] = sorted(_team_set(normalized, ["team_a", "team_b"]))

    if n_rows == 0:
        section["issues"].append("No fixtures rows loaded.")
        section["upcoming_games"] = 0
        section["date_range"] = [None, None]
        return section

    dates = pd.to_datetime(normalized["game_date"], errors="coerce")
    if dates.isna().all():
        section["issues"].append("All fixture game_date values are unparseable.")
    upcoming = int((dates >= today).sum())
    section["upcoming_games"] = upcoming
    section["date_range"] = [
        dates.min().date().isoformat() if dates.notna().any() else None,
        dates.max().date().isoformat() if dates.notna().any() else None,
    ]

    past = int((dates < today).sum())
    if past:
        section["warnings"].append(
            f"{past} fixtures are dated before today; they look like past games, not upcoming ones."
        )

    # Future fixtures must not carry final scores (check the RAW file).
    score_cols = [c for c in raw.columns if str(c).strip().lower() in _SCORE_LIKE_COLUMNS]
    if score_cols:
        has_scores = raw[score_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        n_scored = int(has_scores.sum())
        if n_scored:
            section["warnings"].append(
                f"{n_scored} fixtures already have final scores ({', '.join(score_cols)}); "
                "upcoming games should not be scored."
            )

    # Status sanity.
    if "status" in normalized.columns:
        ok_status = {"scheduled", "upcoming", "not_started", "pre", "pregame", "fixture", "tbd"}
        bad = normalized[~normalized["status"].astype(str).str.lower().isin(ok_status)]
        if not bad.empty:
            section["warnings"].append(
                f"{len(bad)} fixtures have an unusual status value (expected scheduled/upcoming/not_started)."
            )

    return section


def validate_injuries_section(
    path: str | Path | None,
    raw: pd.DataFrame | None,
    normalized: pd.DataFrame | None,
    fixtures_norm: pd.DataFrame,
    today: pd.Timestamp,
) -> dict:
    """Validate the optional injuries file. Missing => WARNING, never FAIL."""

    section = _section()
    section["present"] = normalized is not None

    if normalized is None:
        section["warnings"].append(
            "No injuries file provided; predictions will run without availability features."
        )
        section["rows_loaded"] = 0
        section["teams"] = []
        section["last_updated_range"] = [None, None]
        return section

    n_rows = int(len(normalized))
    section["rows_loaded"] = n_rows
    section["teams"] = sorted(_team_set(normalized, ["team"]))

    if n_rows == 0:
        section["warnings"].append("Injuries file is empty.")
        section["last_updated_range"] = [None, None]
        return section

    # Blank team / player.
    blank_team = int((normalized["team"].astype(str).str.strip() == "").sum())
    if blank_team:
        section["warnings"].append(f"{blank_team} injury rows have a blank team.")
    blank_player = int((normalized["player_name"].astype(str).str.strip() == "").sum())
    if blank_player:
        section["warnings"].append(f"{blank_player} injury rows have a blank player_name.")

    # Status validity.
    statuses = normalized["status"].astype(str).str.lower()
    unknown_statuses = sorted(set(statuses) - _ALLOWED_INJURY_STATUSES)
    if unknown_statuses:
        section["warnings"].append(
            f"Injury statuses not recognised: {unknown_statuses} (expected out/doubtful/"
            "questionable/probable/available/unknown)."
        )

    # Importance score range (normalized is clipped, so check the raw values).
    if raw is not None:
        raw_imp_col = next(
            (c for c in raw.columns if str(c).strip().lower() in {"importance_score", "importance", "impact"}),
            None,
        )
        if raw_imp_col is not None:
            imp = pd.to_numeric(raw[raw_imp_col], errors="coerce")
            out_of_range = int(((imp < 0) | (imp > 1)).sum())
            if out_of_range:
                section["warnings"].append(
                    f"{out_of_range} importance_score values are outside 0..1 (they were clipped)."
                )

    # Freshness.
    last_updated = pd.to_datetime(normalized["last_updated"], errors="coerce")
    section["last_updated_range"] = [
        last_updated.min().isoformat() if last_updated.notna().any() else None,
        last_updated.max().isoformat() if last_updated.notna().any() else None,
    ]
    if last_updated.notna().any():
        newest = last_updated.max()
        age_hours = (pd.Timestamp(today.tz_localize(None) if today.tzinfo else today) - newest).total_seconds() / 3600.0
        if age_hours > _INJURY_STALE_HOURS:
            section["warnings"].append(
                f"Most recent injury update is {age_hours / 24:.1f} days old (older than 48 hours)."
            )
    else:
        section["warnings"].append("No parseable last_updated timestamps in the injuries file.")

    # Injury data missing for a fixture team.
    fixture_keys = {team_match_key(t) for t in _team_set(fixtures_norm, ["team_a", "team_b"])}
    injury_keys = {team_match_key(t) for t in _team_set(normalized, ["team"])}
    missing = sorted(
        t
        for t in _team_set(fixtures_norm, ["team_a", "team_b"])
        if team_match_key(t) not in injury_keys
    )
    if missing and fixture_keys:
        section["warnings"].append(
            f"No injury data for {len(missing)} fixture team(s): {missing}."
        )

    return section


def validate_team_matching(
    results_norm: pd.DataFrame,
    fixtures_norm: pd.DataFrame,
    injuries_norm: pd.DataFrame | None,
) -> dict:
    """Cross-check team names across results, fixtures, and injuries."""

    section = _section()
    result_teams = _team_set(results_norm, ["team_a", "team_b"])
    fixture_teams = _team_set(fixtures_norm, ["team_a", "team_b"])
    injury_teams = _team_set(injuries_norm, ["team"]) if injuries_norm is not None else set()

    result_keys = {team_match_key(t): t for t in result_teams}
    all_known_keys = set(result_keys) | {team_match_key(t) for t in fixture_teams}

    matched = sorted(t for t in fixture_teams if team_match_key(t) in result_keys)
    missing_history = sorted(t for t in fixture_teams if team_match_key(t) not in result_keys)
    injury_unmatched = sorted(t for t in injury_teams if team_match_key(t) not in all_known_keys)

    section["matched_fixture_teams"] = matched
    section["fixture_teams_missing_history"] = missing_history
    section["injury_teams_not_matched"] = injury_unmatched

    # Alias suggestions for unmatched names (fuzzy against known result teams).
    suggestions: dict[str, list[str]] = {}
    result_team_list = sorted(result_teams)
    for team in missing_history + injury_unmatched:
        close = difflib.get_close_matches(team, result_team_list, n=2, cutoff=0.6)
        if close:
            suggestions[team] = close
    section["alias_suggestions"] = suggestions

    if missing_history:
        section["warnings"].append(
            f"{len(missing_history)} fixture team(s) have no historical results: {missing_history}."
        )
    if injury_unmatched:
        section["warnings"].append(
            f"{len(injury_unmatched)} injury team(s) match no results/fixtures team: {injury_unmatched}."
        )
    if suggestions:
        section["warnings"].append(
            "Alias mappings may be needed: "
            + "; ".join(f"{k} -> {v}" for k, v in suggestions.items())
        )

    return section


def build_validation_report(
    results_path: str | Path,
    fixtures_path: str | Path,
    injuries_path: str | Path | None = None,
    sport: str | None = None,
    league: str | None = None,
    aliases_path: str | Path | None = DEFAULT_ALIASES_PATH,
    strict: bool = False,
    today: pd.Timestamp | None = None,
) -> dict:
    """Run all validation sections and return a structured report dict."""

    today = today if today is not None else _now()
    config: dict = {}
    if aliases_path and Path(aliases_path).exists():
        config["aliases_path"] = str(aliases_path)
    if sport:
        config["default_sport"] = normalize_sport(sport)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strict": bool(strict),
        "inputs": {
            "results_path": str(results_path),
            "fixtures_path": str(fixtures_path),
            "injuries_path": str(injuries_path) if injuries_path else None,
            "aliases_path": config.get("aliases_path"),
        },
    }

    # --- results ---
    try:
        raw_results = load_match_results(results_path)
        results_norm = normalize_match_results(raw_results, config)
    except (FileNotFoundError, ValueError) as exc:
        raw_results = pd.DataFrame()
        results_norm = pd.DataFrame()
        report["results"] = {**_section(), "rows_loaded": 0, "fatal": str(exc)}
        report["results"]["issues"].append(f"Could not load/normalize results: {exc}")

    if "results" not in report:
        if sport and "sport" in results_norm:
            results_norm = results_norm[results_norm["sport"] == normalize_sport(sport)].copy()
        if league and "league" in results_norm:
            results_norm = results_norm[results_norm["league"].astype(str).str.lower() == league.lower()].copy()
        report["results"] = validate_results_section(raw_results, results_norm, today)

    # --- fixtures ---
    try:
        raw_fixtures = load_fixtures(fixtures_path)
        fixtures_norm = normalize_fixtures(raw_fixtures, config)
    except (FileNotFoundError, ValueError) as exc:
        raw_fixtures = pd.DataFrame()
        fixtures_norm = pd.DataFrame(columns=FIXTURE_COLUMNS)
        report["fixtures"] = {**_section(), "rows_loaded": 0, "fatal": str(exc)}
        report["fixtures"]["issues"].append(f"Could not load/normalize fixtures: {exc}")

    if "fixtures" not in report:
        if sport and "sport" in fixtures_norm:
            fixtures_norm = fixtures_norm[fixtures_norm["sport"] == normalize_sport(sport)].copy()
        if league and "league" in fixtures_norm:
            fixtures_norm = fixtures_norm[fixtures_norm["league"].astype(str).str.lower() == league.lower()].copy()
        report["fixtures"] = validate_fixtures_section(raw_fixtures, fixtures_norm, today)

    # --- injuries (optional) ---
    injuries_norm = None
    raw_injuries = None
    if injuries_path and Path(injuries_path).exists():
        try:
            raw_injuries = load_injuries(injuries_path)
            injuries_norm = normalize_injuries(raw_injuries, config)
        except (FileNotFoundError, ValueError) as exc:
            report["injuries"] = {**_section(), "present": True, "rows_loaded": 0}
            report["injuries"]["warnings"].append(f"Could not load/normalize injuries: {exc}")
    if "injuries" not in report:
        report["injuries"] = validate_injuries_section(
            injuries_path, raw_injuries, injuries_norm, fixtures_norm, today
        )

    # --- team matching ---
    report["team_matching"] = validate_team_matching(results_norm, fixtures_norm, injuries_norm)

    # --- overall status + recommendation ---
    all_issues: list[str] = []
    all_warnings: list[str] = []
    for key in ("results", "fixtures", "injuries", "team_matching"):
        all_issues.extend(report[key].get("issues", []))
        all_warnings.extend(report[key].get("warnings", []))

    if all_issues:
        overall = STATUS_FAIL
    elif all_warnings:
        overall = STATUS_FAIL if strict else STATUS_WARNING
    else:
        overall = STATUS_PASS
    report["overall_status"] = overall

    n_results = report["results"].get("rows_loaded", 0)
    n_fixtures = report["fixtures"].get("rows_loaded", 0)
    safe_backtest = (not report["results"].get("issues")) and n_results >= _MIN_GAMES_STRONG_WARN
    safe_predict = safe_backtest and (not report["fixtures"].get("issues")) and n_fixtures >= 1
    main_fix = all_issues[0] if all_issues else (all_warnings[0] if all_warnings else "None — inputs look good.")
    report["recommendation"] = {
        "safe_to_backtest": bool(safe_backtest),
        "safe_to_predict": bool(safe_predict),
        "main_fix": main_fix,
    }
    return report


# --- rendering ------------------------------------------------------------
def _fmt_list(values: list, empty: str = "none") -> str:
    if not values:
        return empty
    return ", ".join(str(v) for v in values)


def render_text_report(report: dict) -> str:
    r = report["results"]
    f = report["fixtures"]
    inj = report["injuries"]
    tm = report["team_matching"]
    rec = report["recommendation"]

    lines = [
        "MATCHUP INPUT VALIDATION REPORT",
        "",
        f"Overall status: {report['overall_status']}",
        "",
        "Historical results:",
        f"- Rows loaded: {r.get('rows_loaded', 0)}",
        f"- Teams found: {r.get('teams_found', 0)}",
        f"- Date range: {_fmt_list(r.get('date_range', []), 'n/a')}",
        f"- Sports: {_fmt_list(r.get('sports', []))}",
        f"- Leagues: {_fmt_list(r.get('leagues', []))}",
        f"- Issues: {_fmt_list(r.get('issues', []), 'none')}",
        f"- Warnings: {_fmt_list(r.get('warnings', []), 'none')}",
        "",
        "Fixtures:",
        f"- Rows loaded: {f.get('rows_loaded', 0)}",
        f"- Upcoming games: {f.get('upcoming_games', 0)}",
        f"- Date range: {_fmt_list(f.get('date_range', []), 'n/a')}",
        f"- Teams: {_fmt_list(f.get('teams', []))}",
        f"- Issues: {_fmt_list(f.get('issues', []), 'none')}",
        f"- Warnings: {_fmt_list(f.get('warnings', []), 'none')}",
        "",
        "Injuries:",
        f"- Rows loaded: {inj.get('rows_loaded', 0)}",
        f"- Teams: {_fmt_list(inj.get('teams', []))}",
        f"- Last updated range: {_fmt_list(inj.get('last_updated_range', []), 'n/a')}",
        f"- Issues: {_fmt_list(inj.get('issues', []), 'none')}",
        f"- Warnings: {_fmt_list(inj.get('warnings', []), 'none')}",
        "",
        "Team matching:",
        f"- Matched fixture teams: {_fmt_list(tm.get('matched_fixture_teams', []))}",
        f"- Fixture teams missing history: {_fmt_list(tm.get('fixture_teams_missing_history', []))}",
        f"- Injury teams not matched: {_fmt_list(tm.get('injury_teams_not_matched', []))}",
        f"- Alias suggestions: {_fmt_list([f'{k} -> {v}' for k, v in tm.get('alias_suggestions', {}).items()])}",
        "",
        "Final recommendation:",
        f"- Safe to run backtest: {'yes' if rec['safe_to_backtest'] else 'no'}",
        f"- Safe to build predictions: {'yes' if rec['safe_to_predict'] else 'no'}",
        f"- Main thing to fix next: {rec['main_fix']}",
    ]
    return "\n".join(lines)


def render_markdown_report(report: dict) -> str:
    text = render_text_report(report)
    # Light markdown: title as heading, keep the rest as a fenced block-free body.
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return f"# Matchup Input Validation Report\n\n**Overall status: {report['overall_status']}**\n\n{body}\n"
