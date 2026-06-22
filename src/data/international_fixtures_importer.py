"""Import upcoming international soccer fixtures into the no-odds fixture schema.

The martj42 international_results dataset includes future rows with missing
scores. The results importer drops those rows; this module keeps them as
scheduled fixtures and writes the exact schema consumed by
``data.fixtures_loader``. No odds, prices, CLV, or sportsbook fields are used.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.fixtures_loader import FIXTURE_COLUMNS
from data.international_soccer_importer import DEFAULT_SOURCE_URL
from data.team_name_map import load_team_aliases, resolve_team_name

DEFAULT_LOCAL_PATHS = (
    "data/raw/international_soccer_results_raw.csv",
    "data/raw/free_backfill/world_cup_results/international_results.csv",
)
DEFAULT_OUTPUT_PATH = "data/processed/fixtures_today.csv"
DEFAULT_RAW_OUTPUT_PATH = "data/raw/international_soccer_fixtures_raw.csv"
DEFAULT_ALIASES_PATH = "data/manual/team_aliases_template.csv"

_TRUE_STRINGS = {"1", "true", "yes", "y", "t"}
_COMPLETED_STATUSES = {
    "complete",
    "completed",
    "final",
    "finished",
    "ft",
    "fulltime",
    "full_time",
    "played",
}
_SCORE_COLUMNS = (
    "home_score",
    "away_score",
    "team_a_score",
    "team_b_score",
    "home_goals",
    "away_goals",
    "score_home",
    "score_away",
)


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _coerce_neutral(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(str(value).strip().lower() in _TRUE_STRINGS)


def make_fixture_id(
    game_date: object,
    team_a: object,
    team_b: object,
    competition_type: object,
) -> str:
    """Return a deterministic lowercase fixture id safe for CSV/HTML use."""

    date_text = pd.to_datetime(game_date, errors="coerce")
    date_part = date_text.strftime("%Y-%m-%d") if pd.notna(date_text) else _slug(game_date)
    return "_".join(
        [
            "soccer",
            "international",
            date_part,
            _slug(team_a) or "team_a",
            _slug(team_b) or "team_b",
            _slug(competition_type) or "unknown",
        ]
    )


def load_raw_fixture_source(
    input_path: str | Path | None = None,
    source_url: str | None = None,
    local_defaults: Iterable[str | Path] = DEFAULT_LOCAL_PATHS,
    source_default_url: str = DEFAULT_SOURCE_URL,
) -> tuple[pd.DataFrame, str]:
    """Load raw fixture candidates.

    Resolution order: explicit input path, explicit URL, local martj42 copies,
    then the public martj42 raw GitHub URL.
    """

    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"--input-path not found: {path}")
        return pd.read_csv(path, low_memory=False), f"local file: {path}"

    if source_url:
        return pd.read_csv(source_url, low_memory=False), f"url: {source_url}"

    for candidate in local_defaults:
        path = Path(candidate)
        if path.exists():
            return pd.read_csv(path, low_memory=False), f"local default: {path}"

    try:
        return (
            pd.read_csv(source_default_url, low_memory=False),
            f"default url: {source_default_url}",
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Could not load upcoming international fixtures. Provide a local "
            "martj42 results.csv with --input-path, or pass --source-url. Last "
            f"attempted default URL: {source_default_url}. Error: {exc}"
        ) from exc


def _first_present(df: pd.DataFrame, names: tuple[str, ...], default: object = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _score_present(df: pd.DataFrame) -> pd.Series:
    present = pd.Series(False, index=df.index)
    for column in _SCORE_COLUMNS:
        if column in df.columns:
            present = present | pd.to_numeric(df[column], errors="coerce").notna()
    return present


def _parse_date(value: str | None, fallback: pd.Timestamp | None = None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return fallback
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse date: {value!r}")
    return parsed.normalize()


def normalize_international_fixtures(
    raw: pd.DataFrame,
    *,
    aliases: dict[str, str] | None = None,
    as_of_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    days_ahead: int = 14,
    team_filter: list[str] | None = None,
    include_past_today: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Normalize raw martj42 rows into ``fixtures_loader.FIXTURE_COLUMNS``."""

    rows_read = int(len(raw))
    drop_reasons: dict[str, int] = {}
    warnings: list[str] = []

    def _drop(mask: pd.Series, reason: str, frame: pd.DataFrame) -> pd.DataFrame:
        n = int(mask.sum())
        if n:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + n
        return frame.loc[~mask].copy()

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw fixtures missing required columns: {sorted(missing)}")

    as_of = _parse_date(as_of_date, pd.Timestamp.now().normalize())
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end is None and days_ahead is not None:
        end = as_of + pd.Timedelta(days=int(days_ahead))

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _drop(df["_date"].isna(), "bad_date", df)

    home_blank = df["home_team"].isna() | (df["home_team"].astype(str).str.strip() == "")
    away_blank = df["away_team"].isna() | (df["away_team"].astype(str).str.strip() == "")
    df = _drop(home_blank | away_blank, "missing_team", df)

    df = _drop(_score_present(df), "completed_or_scored", df)

    if "status" in df.columns:
        status = df["status"].fillna("").astype(str).str.strip().str.lower()
        df = _drop(status.isin(_COMPLETED_STATUSES), "completed_status", df)

    if not include_past_today:
        df = _drop(df["_date"] < as_of, "before_as_of_date", df)
    if start is not None:
        df = _drop(df["_date"] < start, "before_start_date", df)
    if end is not None:
        df = _drop(df["_date"] > end, "after_end_date", df)

    if df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS), {
            "rows_read": rows_read,
            "rows_written": 0,
            "rows_dropped": rows_read,
            "drop_reasons": drop_reasons,
            "warnings": warnings + ["No upcoming fixtures remained after filtering."],
            "as_of_date": as_of.date().isoformat(),
            "end_date": end.date().isoformat() if end is not None else None,
        }

    team_a = df["home_team"].map(lambda v: resolve_team_name(v, aliases))
    team_b = df["away_team"].map(lambda v: resolve_team_name(v, aliases))

    if team_filter:
        wanted = {resolve_team_name(t, aliases) for t in team_filter}
        keep = team_a.isin(wanted) | team_b.isin(wanted)
        drop_n = int((~keep).sum())
        if drop_n:
            drop_reasons["team_filter_excluded"] = drop_n
        df = df.loc[keep].copy()
        team_a = team_a.loc[keep]
        team_b = team_b.loc[keep]

    competition = _first_present(df, ("tournament", "competition_type", "competition"), "unknown")
    competition = competition.fillna("unknown").astype(str).str.strip().replace("", "unknown")
    neutral = _first_present(df, ("neutral", "neutral_site"), False).map(_coerce_neutral)
    date_str = df["_date"].dt.strftime("%Y-%m-%d")

    out = pd.DataFrame(
        {
            "sport": "soccer",
            "league": "international",
            "game_date": date_str,
            "team_a": team_a.values,
            "team_b": team_b.values,
            "team_a_home_flag": (1 - neutral).astype(int).values,
            "team_b_home_flag": 0,
            "neutral_site": neutral.astype(int).values,
            "competition_type": competition.values,
            "venue": _first_present(df, ("city", "venue", "stadium"), "").fillna("").astype(str).values,
            "status": "scheduled",
        }
    )
    out["fixture_id"] = [
        make_fixture_id(d, a, b, c)
        for d, a, b, c in zip(
            out["game_date"], out["team_a"], out["team_b"], out["competition_type"]
        )
    ]

    dup = out["fixture_id"].duplicated()
    n_dup = int(dup.sum())
    if n_dup:
        drop_reasons["duplicate_fixture_id"] = n_dup
        out = out.loc[~dup].copy()

    out = out[FIXTURE_COLUMNS].sort_values(["game_date", "fixture_id"]).reset_index(drop=True)
    if out.empty:
        warnings.append("No rows written after duplicate removal/filtering.")

    return out, {
        "rows_read": rows_read,
        "rows_written": int(len(out)),
        "rows_dropped": rows_read - int(len(out)),
        "drop_reasons": drop_reasons,
        "warnings": warnings,
        "as_of_date": as_of.date().isoformat(),
        "end_date": end.date().isoformat() if end is not None else None,
    }


def build_fixture_import_summary(
    fixtures: pd.DataFrame,
    stats: dict,
    source_label: str,
    output_path: str | Path,
    raw_copy_path: str | Path | None = None,
) -> dict:
    teams = (
        pd.concat([fixtures["team_a"], fixtures["team_b"]], ignore_index=True)
        if not fixtures.empty
        else pd.Series(dtype=str)
    )
    dates = pd.to_datetime(fixtures["game_date"], errors="coerce") if not fixtures.empty else pd.Series(dtype="datetime64[ns]")
    competitions = fixtures["competition_type"].value_counts() if not fixtures.empty else pd.Series(dtype=int)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "rows_read": int(stats.get("rows_read", 0)),
        "rows_written": int(stats.get("rows_written", 0)),
        "rows_dropped": int(stats.get("rows_dropped", 0)),
        "drop_reasons": stats.get("drop_reasons", {}),
        "date_range": [
            dates.min().date().isoformat() if dates.notna().any() else None,
            dates.max().date().isoformat() if dates.notna().any() else None,
        ],
        "as_of_date": stats.get("as_of_date"),
        "window_end_date": stats.get("end_date"),
        "num_teams": int(teams.nunique()) if not teams.empty else 0,
        "teams": sorted(str(t) for t in teams.dropna().unique()) if not teams.empty else [],
        "competitions": {str(k): int(v) for k, v in competitions.items()},
        "neutral_site_count": int(fixtures["neutral_site"].sum()) if not fixtures.empty else 0,
        "output_path": str(output_path),
        "raw_copy_path": str(raw_copy_path) if raw_copy_path else None,
        "warnings": stats.get("warnings", []),
    }


def render_fixture_summary_text(summary: dict) -> str:
    teams = summary.get("teams", [])
    competitions = summary.get("competitions", {})
    return "\n".join(
        [
            "INTERNATIONAL FIXTURES IMPORT SUMMARY",
            "",
            f"Source: {summary.get('source')}",
            f"Rows read: {summary.get('rows_read', 0):,}  written: {summary.get('rows_written', 0):,}  dropped: {summary.get('rows_dropped', 0):,}",
            f"Date range: {summary.get('date_range', [None, None])[0]} to {summary.get('date_range', [None, None])[1]}",
            f"As of: {summary.get('as_of_date')}  Window end: {summary.get('window_end_date')}",
            f"Teams: {summary.get('num_teams', 0)}",
            "Competitions: " + (", ".join(f"{k} ({v})" for k, v in competitions.items()) or "none"),
            "Sample teams: " + (", ".join(teams[:12]) if teams else "none"),
            f"Neutral-site fixtures: {summary.get('neutral_site_count', 0):,}",
            "Drop reasons: "
            + (", ".join(f"{k}={v}" for k, v in summary.get("drop_reasons", {}).items()) or "none"),
            f"Output: {summary.get('output_path')}",
        ]
    )
