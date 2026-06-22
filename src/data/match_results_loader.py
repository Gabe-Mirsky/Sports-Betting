"""Load and normalize historical match/game results (no odds required).

The matchup prediction pipeline only needs *results* – who played, where, and
the final score. This loader accepts a wide variety of source column names and
produces a single canonical schema the rest of the pipeline relies on.

Canonical (normalized) columns produced by :func:`normalize_match_results`:

    game_id, sport, league, season, game_date,
    team_a, team_b, team_a_score, team_b_score,
    team_a_home_flag, team_b_home_flag, neutral_site,
    result_team_a_win, result_draw, result_team_b_win,
    competition_type

Conventions:
* ``team_a`` is the home team unless the game is at a neutral site.
* For sports without draws, ``result_draw`` is always 0.
* All team names are passed through :func:`team_name_map.resolve_team_name`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from data.sport_rules import normalize_sport, sport_allows_draws
from data.team_name_map import apply_team_aliases, load_team_aliases


MATCH_RESULTS_COLUMNS: list[str] = [
    "game_id",
    "sport",
    "league",
    "season",
    "game_date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "team_a_home_flag",
    "team_b_home_flag",
    "neutral_site",
    "result_team_a_win",
    "result_draw",
    "result_team_b_win",
    "competition_type",
]


# Source-column synonyms -> canonical role. Lookup is case-insensitive.
_COLUMN_SYNONYMS: dict[str, str] = {
    # identifiers
    "game_id": "game_id",
    "match_id": "game_id",
    "id": "game_id",
    "fixture_id": "game_id",
    # context
    "sport": "sport",
    "league": "league",
    "competition": "league",
    "comp": "league",
    "tournament": "league",
    "season": "season",
    "year": "season",
    "game_date": "game_date",
    "date": "game_date",
    "match_date": "game_date",
    "datetime": "game_date",
    "kickoff": "game_date",
    "competition_type": "competition_type",
    "match_type": "competition_type",
    "stage": "competition_type",
    "round": "competition_type",
    # team_a / team_b direct
    "team_a": "team_a",
    "team_b": "team_b",
    "team_a_score": "team_a_score",
    "team_b_score": "team_b_score",
    # home / away style
    "home_team": "home_team",
    "home": "home_team",
    "home_team_name": "home_team",
    "home_team_abbr": "home_team",
    "away_team": "away_team",
    "away": "away_team",
    "away_team_name": "away_team",
    "away_team_abbr": "away_team",
    "home_score": "home_score",
    "home_points": "home_score",
    "home_goals": "home_score",
    "score_home": "home_score",
    "away_score": "away_score",
    "away_points": "away_score",
    "away_goals": "away_score",
    "score_away": "away_score",
    # flags
    "neutral_site": "neutral_site",
    "neutral": "neutral_site",
    "team_a_home_flag": "team_a_home_flag",
    "team_b_home_flag": "team_b_home_flag",
    # explicit winner columns (used only when scores are absent)
    "home_win": "home_win",
    "winner": "winner",
    "result": "result",
}

_TRUE_STRINGS = {"1", "true", "yes", "y", "t"}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in _TRUE_STRINGS


def _build_column_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical role -> actual column name for the columns we recognize."""

    lookup: dict[str, str] = {}
    for column in df.columns:
        role = _COLUMN_SYNONYMS.get(str(column).strip().lower())
        # Do not overwrite an earlier (higher-priority) match for the same role.
        if role and role not in lookup:
            lookup[role] = column
    return lookup


def _make_game_id(row: pd.Series) -> str:
    raw = "|".join(
        str(row.get(field, ""))
        for field in ("game_date", "league", "team_a", "team_b")
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"match_{digest}"


def load_match_results(path: str | Path) -> pd.DataFrame:
    """Load raw match results from a CSV file or a folder of CSV files.

    When ``path`` is a directory, every ``*.csv`` inside it is concatenated.
    The returned frame is intentionally *raw* – call
    :func:`normalize_match_results` to map it onto the canonical schema.
    """

    source = Path(path)
    if source.is_dir():
        frames: list[pd.DataFrame] = []
        for csv_path in sorted(source.glob("*.csv")):
            try:
                frames.append(pd.read_csv(csv_path, low_memory=False))
            except pd.errors.EmptyDataError:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    if not source.exists():
        raise FileNotFoundError(f"Match results file not found: {source}")
    try:
        return pd.read_csv(source, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_match_results(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Map a raw results frame onto the canonical matchup schema.

    Parameters
    ----------
    df:
        Raw results, in any of the supported source layouts.
    config:
        Optional overrides. Supported keys:
        ``default_sport``, ``default_league``, ``default_competition_type``,
        ``aliases_path`` (CSV team alias table), and the draw-sport overrides
        understood by :func:`data.sport_rules.sport_allows_draws`.
    """

    config = config or {}
    if df.empty:
        return pd.DataFrame(columns=MATCH_RESULTS_COLUMNS)

    lookup = _build_column_lookup(df)
    out = pd.DataFrame(index=df.index)

    # --- context columns -------------------------------------------------
    out["sport"] = (
        df[lookup["sport"]] if "sport" in lookup else config.get("default_sport", "soccer")
    )
    out["sport"] = out["sport"].map(normalize_sport)
    out["league"] = (
        df[lookup["league"]] if "league" in lookup else config.get("default_league", "unknown")
    )
    out["competition_type"] = (
        df[lookup["competition_type"]]
        if "competition_type" in lookup
        else config.get("default_competition_type", "unknown")
    )

    out["game_date"] = pd.to_datetime(
        df[lookup["game_date"]] if "game_date" in lookup else pd.NaT,
        errors="coerce",
        utc=False,
    )

    # --- teams + home/away ----------------------------------------------
    neutral = (
        df[lookup["neutral_site"]].map(_coerce_bool)
        if "neutral_site" in lookup
        else pd.Series(False, index=df.index)
    )

    if "team_a" in lookup and "team_b" in lookup:
        out["team_a"] = df[lookup["team_a"]].astype(str)
        out["team_b"] = df[lookup["team_b"]].astype(str)
        out["team_a_score"] = _scores(df, lookup, "team_a_score")
        out["team_b_score"] = _scores(df, lookup, "team_b_score")
        if "team_a_home_flag" in lookup:
            a_home = df[lookup["team_a_home_flag"]].map(_coerce_bool)
        else:
            a_home = ~neutral  # team_a treated as home by convention
        b_home = pd.Series(False, index=df.index) if "team_b_home_flag" not in lookup else df[
            lookup["team_b_home_flag"]
        ].map(_coerce_bool)
    elif "home_team" in lookup and "away_team" in lookup:
        out["team_a"] = df[lookup["home_team"]].astype(str)
        out["team_b"] = df[lookup["away_team"]].astype(str)
        out["team_a_score"] = _scores(df, lookup, "home_score")
        out["team_b_score"] = _scores(df, lookup, "away_score")
        a_home = ~neutral
        b_home = pd.Series(False, index=df.index)
    else:
        raise ValueError(
            "Could not find team columns. Provide either team_a/team_b or "
            "home_team/away_team (case-insensitive)."
        )

    out["neutral_site"] = neutral.astype(int)
    out["team_a_home_flag"] = a_home.where(~neutral, False).astype(int)
    out["team_b_home_flag"] = b_home.where(~neutral, False).astype(int)

    # --- season ----------------------------------------------------------
    if "season" in lookup:
        out["season"] = df[lookup["season"]]
    else:
        out["season"] = out["game_date"].dt.year

    # --- results ---------------------------------------------------------
    out = _derive_results(out, df, lookup, config)

    # --- team-name normalization ----------------------------------------
    aliases = load_team_aliases(config["aliases_path"]) if config.get("aliases_path") else {}
    out = apply_team_aliases(out, ["team_a", "team_b"], aliases)

    # --- game_id ---------------------------------------------------------
    if "game_id" in lookup:
        out["game_id"] = df[lookup["game_id"]].astype(str)
    else:
        out["game_id"] = out.apply(_make_game_id, axis=1)

    # Stable column order, drop fully-duplicate game rows.
    out = out[MATCH_RESULTS_COLUMNS].copy()
    out = out.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return out


def _scores(df: pd.DataFrame, lookup: dict[str, str], role: str) -> pd.Series:
    if role in lookup:
        return pd.to_numeric(df[lookup[role]], errors="coerce")
    return pd.Series(float("nan"), index=df.index, dtype="float64")


def _derive_results(
    out: pd.DataFrame,
    df: pd.DataFrame,
    lookup: dict[str, str],
    config: dict,
) -> pd.DataFrame:
    """Populate result_team_a_win / result_draw / result_team_b_win."""

    a_score = out["team_a_score"]
    b_score = out["team_b_score"]
    have_scores = a_score.notna() & b_score.notna()

    a_win = pd.Series(0, index=out.index, dtype=int)
    draw = pd.Series(0, index=out.index, dtype=int)
    b_win = pd.Series(0, index=out.index, dtype=int)

    # Per-row draw eligibility depends on the sport.
    draw_ok = out["sport"].map(lambda s: sport_allows_draws(s, config))

    # 1) Prefer final scores when available.
    a_win.loc[have_scores & (a_score > b_score)] = 1
    b_win.loc[have_scores & (a_score < b_score)] = 1
    equal = have_scores & (a_score == b_score)
    draw.loc[equal & draw_ok] = 1
    # Equal score in a no-draw sport => result is undetermined (data issue);
    # leave all three at 0 so validation can flag it.

    # 2) Fall back to explicit winner columns where scores are missing.
    missing = ~have_scores
    if missing.any():
        if "home_win" in lookup:
            home_win = df.loc[missing, lookup["home_win"]].map(_coerce_bool)
            # team_a is the home team in the home/away mapping path.
            a_is_home = out.loc[missing, "team_a_home_flag"] == 1
            a_win.loc[missing & a_is_home & home_win.reindex(out.index, fill_value=False)] = 1
            b_win.loc[missing & a_is_home & ~home_win.reindex(out.index, fill_value=False)] = 1
        elif "winner" in lookup:
            winner = df.loc[missing, lookup["winner"]].astype(str).str.strip().str.lower()
            a_name = out.loc[missing, "team_a"].astype(str).str.strip().str.lower()
            b_name = out.loc[missing, "team_b"].astype(str).str.strip().str.lower()
            a_win.loc[missing & (winner == a_name)] = 1
            b_win.loc[missing & (winner == b_name)] = 1
            draw.loc[missing & draw_ok & winner.isin({"draw", "tie", "d"})] = 1

    out["result_team_a_win"] = a_win
    out["result_draw"] = draw
    out["result_team_b_win"] = b_win
    return out


def validate_match_results(df: pd.DataFrame) -> dict:
    """Return a structured validation report for a *normalized* results frame.

    The report never raises; it surfaces issues so the caller can decide what
    to do. Shape::

        {"ok": bool, "n_rows": int, "issues": [...], "warnings": [...],
         "n_sports": int, "n_leagues": int, "date_min": str, "date_max": str}
    """

    issues: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in MATCH_RESULTS_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return {
            "ok": False,
            "n_rows": int(len(df)),
            "issues": issues,
            "warnings": warnings,
        }

    if df.empty:
        issues.append("No rows in results dataset.")

    if df["game_id"].duplicated().any():
        n_dupes = int(df["game_id"].duplicated().sum())
        issues.append(f"{n_dupes} duplicate game_id rows.")

    if df["game_date"].isna().any():
        warnings.append(f"{int(df['game_date'].isna().sum())} rows have an unparseable game_date.")

    result_sum = df[["result_team_a_win", "result_draw", "result_team_b_win"]].sum(axis=1)
    undetermined = int((result_sum == 0).sum())
    if undetermined:
        warnings.append(f"{undetermined} games have no recorded outcome (missing/equal scores).")
    if int((result_sum > 1).sum()):
        issues.append("Some games have more than one outcome flag set.")

    # Draw rows for no-draw sports should not exist.
    for sport, group in df.groupby("sport"):
        if not sport_allows_draws(sport) and int(group["result_draw"].sum()) > 0:
            issues.append(f"Sport '{sport}' has draw outcomes but does not allow draws.")

    dates = pd.to_datetime(df["game_date"], errors="coerce").dropna()
    return {
        "ok": len(issues) == 0,
        "n_rows": int(len(df)),
        "n_sports": int(df["sport"].nunique()),
        "n_leagues": int(df["league"].nunique()),
        "date_min": dates.min().date().isoformat() if not dates.empty else None,
        "date_max": dates.max().date().isoformat() if not dates.empty else None,
        "undetermined_games": undetermined,
        "issues": issues,
        "warnings": warnings,
    }
