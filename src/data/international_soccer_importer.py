"""Import real international soccer results into the matchup schema (no odds).

Source: the free/public martj42/international_results dataset (``results.csv``),
which has columns: ``date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral``. No betting odds, prices, or market data
are used anywhere.

The reusable logic lives here; ``scripts/import_international_soccer_results.py``
is a thin CLI wrapper around :func:`normalize_international_results` and
:func:`build_import_summary`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.team_name_map import load_team_aliases, resolve_team_name

# Public raw CSV (used when no local file / input path is supplied).
DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
# A local copy that already ships with this repo (free backfill), tried first
# so the importer works offline.
DEFAULT_LOCAL_PATH = "data/raw/free_backfill/world_cup_results/international_results.csv"
DEFAULT_RAW_OUTPUT_PATH = "data/raw/international_soccer_results_raw.csv"
DEFAULT_ALIASES_PATH = "data/manual/team_aliases_template.csv"

# Output schema: the loader's canonical columns + a few descriptive extras the
# loader safely ignores (venue, country, status) and the pre-computed result
# flags (the loader re-derives these from scores, so they stay consistent).
OUTPUT_COLUMNS: list[str] = [
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
    "venue",
    "country",
    "status",
]

_TRUE_STRINGS = {"1", "true", "yes", "y", "t"}


def _slug(value: object) -> str:
    """Lowercase, alphanumeric-only token safe for a game_id (no spaces/slashes)."""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def make_game_id(
    game_date: str,
    team_a: str,
    team_b: str,
    team_a_score: int,
    team_b_score: int,
    competition_type: str,
) -> str:
    """Deterministic, URL-safe game_id, e.g.
    ``soccer_international_2024-06-11_japan_tunisia_2_1_friendly``."""

    return "_".join(
        [
            "soccer",
            "international",
            str(game_date),
            _slug(team_a),
            _slug(team_b),
            str(int(team_a_score)),
            str(int(team_b_score)),
            _slug(competition_type) or "unknown",
        ]
    )


def _coerce_neutral(series: pd.Series) -> pd.Series:
    return series.map(
        lambda v: 1 if str(v).strip().lower() in _TRUE_STRINGS else 0
    ).astype(int)


def load_raw_source(
    input_path: str | Path | None = None,
    source_url: str | None = None,
    local_default: str | Path | None = DEFAULT_LOCAL_PATH,
    source_default_url: str = DEFAULT_SOURCE_URL,
) -> tuple[pd.DataFrame, str]:
    """Resolve and load the raw results CSV. Returns ``(dataframe, source_label)``.

    Resolution order: explicit ``input_path`` -> explicit ``source_url`` ->
    a local default copy if present -> the public default URL. Raises a clear
    error (mentioning ``--input-path``) if everything fails.
    """

    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"--input-path not found: {path}")
        return pd.read_csv(path, low_memory=False), f"local file: {path}"

    if source_url:
        return pd.read_csv(source_url, low_memory=False), f"url: {source_url}"

    if local_default and Path(local_default).exists():
        return (
            pd.read_csv(local_default, low_memory=False),
            f"local default: {local_default}",
        )

    try:
        return (
            pd.read_csv(source_default_url, low_memory=False),
            f"default url: {source_default_url}",
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Could not download the international results dataset "
            f"({source_default_url}): {exc}. Download results.csv from "
            "https://github.com/martj42/international_results and pass it with "
            "--input-path."
        ) from exc


def normalize_international_results(
    raw: pd.DataFrame,
    *,
    aliases: dict | None = None,
    today: pd.Timestamp | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    include_friendlies: bool = True,
    team_filter: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Normalize the martj42 results frame into the matchup schema + clean it.

    Returns ``(clean_df, stats)`` where ``stats`` records rows read/written/
    dropped, per-reason drop counts, and warnings.
    """

    today = (today or pd.Timestamp.now()).normalize()
    drop_reasons: dict[str, int] = {}
    warnings: list[str] = []
    rows_read = int(len(raw))

    def _drop(mask: pd.Series, reason: str, frame: pd.DataFrame) -> pd.DataFrame:
        n = int(mask.sum())
        if n:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + n
        return frame[~mask].copy()

    df = raw.copy()
    # Be tolerant of column-name casing/whitespace.
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw results missing required columns: {sorted(missing_cols)}")

    # Parse dates.
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _drop(df["_date"].isna(), "bad_date", df)

    # Drop missing teams.
    home_blank = df["home_team"].isna() | (df["home_team"].astype(str).str.strip() == "")
    away_blank = df["away_team"].isna() | (df["away_team"].astype(str).str.strip() == "")
    df = _drop(home_blank | away_blank, "missing_team", df)

    # Numeric scores; drop missing/non-numeric (covers "NA" future fixtures).
    df["_hs"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["_as"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = _drop(df["_hs"].isna() | df["_as"].isna(), "missing_or_bad_score", df)

    # Only completed games up to and including today (no future fixtures).
    df = _drop(df["_date"] > today, "future_or_after_today", df)

    # Optional date-range / year filters.
    if start_date:
        df = _drop(df["_date"] < pd.to_datetime(start_date), "out_of_date_range", df)
    if end_date:
        df = _drop(df["_date"] > pd.to_datetime(end_date), "out_of_date_range", df)
    if min_year is not None:
        df = _drop(df["_date"].dt.year < int(min_year), "out_of_year_range", df)
    if max_year is not None:
        df = _drop(df["_date"].dt.year > int(max_year), "out_of_year_range", df)

    # Friendlies.
    tournament = df.get("tournament", pd.Series("Friendly", index=df.index)).fillna("Friendly")
    if not include_friendlies:
        is_friendly = tournament.astype(str).str.strip().str.lower() == "friendly"
        df = _drop(is_friendly, "friendlies_excluded", df)
        tournament = tournament[df.index]

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {
            "rows_read": rows_read,
            "rows_written": 0,
            "rows_dropped": rows_read,
            "drop_reasons": drop_reasons,
            "warnings": warnings + ["No rows remained after cleaning/filtering."],
        }

    # Normalize team names (alias table first, then built-in rules).
    team_a = df["home_team"].map(lambda v: resolve_team_name(v, aliases))
    team_b = df["away_team"].map(lambda v: resolve_team_name(v, aliases))

    # Optional team filter (match on canonical names).
    if team_filter:
        wanted = {resolve_team_name(t, aliases) for t in team_filter}
        keep = team_a.isin(wanted) | team_b.isin(wanted)
        drop_n = int((~keep).sum())
        if drop_n:
            drop_reasons["team_filter_excluded"] = drop_n
        df = df[keep].copy()
        team_a = team_a[keep]
        team_b = team_b[keep]
        tournament = tournament[keep]

    hs = df["_hs"].astype(int)
    as_ = df["_as"].astype(int)
    neutral = _coerce_neutral(df.get("neutral", pd.Series(0, index=df.index)))
    comp = tournament.astype(str).str.strip().replace("", "unknown")
    date_str = df["_date"].dt.strftime("%Y-%m-%d")

    out = pd.DataFrame(
        {
            "sport": "soccer",
            "league": "international",
            "season": df["_date"].dt.year.astype(int),
            "game_date": date_str,
            "team_a": team_a.values,
            "team_b": team_b.values,
            "team_a_score": hs.values,
            "team_b_score": as_.values,
            "team_a_home_flag": (1 - neutral).values,
            "team_b_home_flag": 0,
            "neutral_site": neutral.values,
            "result_team_a_win": (hs.values > as_.values).astype(int),
            "result_draw": (hs.values == as_.values).astype(int),
            "result_team_b_win": (as_.values > hs.values).astype(int),
            "competition_type": comp.values,
            "venue": df.get("city", pd.Series("", index=df.index)).fillna("").astype(str).values,
            "country": df.get("country", pd.Series("", index=df.index)).fillna("").astype(str).values,
            "status": "completed",
        }
    )

    out["game_id"] = [
        make_game_id(d, a, b, sa, sb, c)
        for d, a, b, sa, sb, c in zip(
            out["game_date"],
            out["team_a"],
            out["team_b"],
            out["team_a_score"],
            out["team_b_score"],
            out["competition_type"],
        )
    ]

    # Remove duplicate game_id rows.
    dup = out["game_id"].duplicated()
    n_dup = int(dup.sum())
    if n_dup:
        drop_reasons["duplicate_game_id"] = n_dup
        out = out[~dup].copy()

    out = out[OUTPUT_COLUMNS].sort_values("game_date").reset_index(drop=True)

    rows_written = int(len(out))
    if rows_written < 20:
        warnings.append(
            f"Only {rows_written} rows written — fewer than 20; the model will be weak."
        )

    stats = {
        "rows_read": rows_read,
        "rows_written": rows_written,
        "rows_dropped": rows_read - rows_written,
        "drop_reasons": drop_reasons,
        "warnings": warnings,
    }
    return out, stats


def build_import_summary(
    clean: pd.DataFrame,
    stats: dict,
    source_label: str,
    output_path: str | Path,
    raw_copy_path: str | Path | None = None,
) -> dict:
    """Build the structured import summary dict (also used for JSON/MD/terminal)."""

    teams = pd.concat([clean["team_a"], clean["team_b"]], ignore_index=True) if not clean.empty else pd.Series(dtype=str)
    top_teams = teams.value_counts().head(20)
    top_tournaments = (
        clean["competition_type"].value_counts().head(20) if not clean.empty else pd.Series(dtype=int)
    )
    dates = pd.to_datetime(clean["game_date"], errors="coerce") if not clean.empty else pd.Series(dtype="datetime64[ns]")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "rows_read": stats.get("rows_read", 0),
        "rows_written": stats.get("rows_written", 0),
        "rows_dropped": stats.get("rows_dropped", 0),
        "drop_reasons": stats.get("drop_reasons", {}),
        "date_range": [
            dates.min().date().isoformat() if dates.notna().any() else None,
            dates.max().date().isoformat() if dates.notna().any() else None,
        ],
        "num_teams": int(teams.nunique()) if not teams.empty else 0,
        "num_competition_types": int(clean["competition_type"].nunique()) if not clean.empty else 0,
        "top_20_teams": {str(k): int(v) for k, v in top_teams.items()},
        "top_20_tournaments": {str(k): int(v) for k, v in top_tournaments.items()},
        "neutral_site_count": int(clean["neutral_site"].sum()) if not clean.empty else 0,
        "draw_count": int(clean["result_draw"].sum()) if not clean.empty else 0,
        "team_a_win_count": int(clean["result_team_a_win"].sum()) if not clean.empty else 0,
        "team_b_win_count": int(clean["result_team_b_win"].sum()) if not clean.empty else 0,
        "output_path": str(output_path),
        "raw_copy_path": str(raw_copy_path) if raw_copy_path else None,
        "warnings": stats.get("warnings", []),
    }


def render_summary_markdown(summary: dict) -> str:
    lines = [
        "# International Soccer Import Summary",
        "",
        f"- Source: {summary['source']}",
        f"- Rows read: {summary['rows_read']:,}",
        f"- Rows written: {summary['rows_written']:,}",
        f"- Rows dropped: {summary['rows_dropped']:,}",
        f"- Date range: {summary['date_range'][0]} to {summary['date_range'][1]}",
        f"- Teams: {summary['num_teams']}",
        f"- Competition types: {summary['num_competition_types']}",
        f"- Neutral-site games: {summary['neutral_site_count']:,}",
        f"- Draws: {summary['draw_count']:,}",
        f"- Home/team_a wins: {summary['team_a_win_count']:,}",
        f"- Away/team_b wins: {summary['team_b_win_count']:,}",
        f"- Output: {summary['output_path']}",
        f"- Raw copy: {summary['raw_copy_path'] or 'not written'}",
        "",
        "## Drop reasons",
    ]
    if summary["drop_reasons"]:
        for reason, count in sorted(summary["drop_reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {reason}: {count:,}")
    else:
        lines.append("- none")

    lines += ["", "## Top 20 teams by match count"]
    for team, count in summary["top_20_teams"].items():
        lines.append(f"- {team}: {count:,}")

    lines += ["", "## Top 20 competition types by match count"]
    for comp, count in summary["top_20_tournaments"].items():
        lines.append(f"- {comp}: {count:,}")

    if summary["warnings"]:
        lines += ["", "## Warnings"]
        lines += [f"- {w}" for w in summary["warnings"]]
    return "\n".join(lines) + "\n"


def render_summary_text(summary: dict) -> str:
    top5 = list(summary["top_20_teams"].items())[:5]
    top5_comp = list(summary["top_20_tournaments"].items())[:5]
    return "\n".join(
        [
            "INTERNATIONAL SOCCER IMPORT SUMMARY",
            "",
            f"Source: {summary['source']}",
            f"Rows read: {summary['rows_read']:,}  written: {summary['rows_written']:,}  dropped: {summary['rows_dropped']:,}",
            f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}",
            f"Teams: {summary['num_teams']}   Competition types: {summary['num_competition_types']}",
            f"Neutral-site: {summary['neutral_site_count']:,}   Draws: {summary['draw_count']:,}   "
            f"Home wins: {summary['team_a_win_count']:,}   Away wins: {summary['team_b_win_count']:,}",
            "Top teams: " + ", ".join(f"{t} ({n})" for t, n in top5),
            "Top competitions: " + ", ".join(f"{t} ({n})" for t, n in top5_comp),
            "Drop reasons: "
            + (", ".join(f"{k}={v}" for k, v in summary["drop_reasons"].items()) or "none"),
            f"Output: {summary['output_path']}",
        ]
    )
