"""Profile candidate Kaggle datasets before committing to using them.

This is a read-only inspection tool. It lists files, loads the ones it can
(csv/parquet/json/xlsx/txt), and reports per-file signals (sport, market type,
odds vs. props, timestamps, results) so a human can decide whether a dataset is
worth wiring into the pipeline. It never downloads unless explicitly asked, and
it does not touch proof gates, betting, or parlays.

Detection is deliberately strict:
  * player props need a player identifier *and* a prop type/line *and* an
    over/under (or yes/no) price - odds alone never imply props.
  * game odds need a team/game identifier *and* a moneyline/spread/total.
  * CLV needs two timestamps or explicit open *and* close prices.
  * settlement grading needs a final result, or enough keys to join to results.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_BASE_EXTENSIONS = (".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls", ".txt")
# Kept for back-compat; base (non-gzip) extensions. Gzipped variants (e.g. .csv.gz)
# are also supported via ``_file_kind``.
SUPPORTED_EXTENSIONS = set(SUPPORTED_BASE_EXTENSIONS)


def _file_kind(name: str) -> tuple[str | None, bool]:
    """Return (base_extension, is_gzip) for a filename, or (None, gz) if unsupported."""

    lower = str(name).lower()
    gz = lower.endswith(".gz")
    core = lower[:-3] if gz else lower
    for ext in SUPPORTED_BASE_EXTENSIONS:
        if core.endswith(ext):
            return ext, gz
    return None, gz


def _is_supported(path: Path) -> bool:
    return _file_kind(path.name)[0] is not None

CLASSIFICATIONS = (
    "player_prop_ready",
    "game_odds_ready",
    "stats_only",
    "odds_without_results",
    "results_without_odds",
    "unusable_or_unknown",
)
RECOMMENDATIONS = ("use_now", "use_later", "manual_review", "skip")

# Box-score / player-game stat tokens used to flag a "stats" dataset.
STAT_TOKENS = {
    "pts", "points", "reb", "rebounds", "trb", "ast", "assists", "stl", "blk",
    "tov", "fg", "fga", "fgm", "fg3m", "fg3a", "min", "minutes", "mp", "usg",
    "yards", "touchdowns", "td", "passing", "rushing", "receiving", "goals",
    "assists", "saves", "era", "strikeouts", "hits", "runs", "rbi",
}


def _norm(name: Any) -> str:
    """Lower-case a column name and collapse punctuation to single spaces."""

    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _toks(normalized: str) -> set[str]:
    return set(normalized.split())


# --- column-level signal detectors -----------------------------------------
def _is_over_price(n: str) -> bool:
    if "turnover" in n or "rollover" in n:
        return False
    if "over" in _toks(n):
        return True
    return "over" in n and any(k in n for k in ("odd", "price", "ml", "payout"))


def _is_under_price(n: str) -> bool:
    if "thunder" in n:
        return False
    if "under" in _toks(n):
        return True
    return "under" in n and any(k in n for k in ("odd", "price", "ml", "payout"))


def _is_yes_no_price(n: str) -> bool:
    has_price = any(k in n for k in ("odd", "price", "ask", "bid"))
    return ("yes" in _toks(n) and has_price) or ("no" in _toks(n) and has_price)


def _is_moneyline(n: str) -> bool:
    toks = _toks(n)
    if "ml" in toks or "h2h" in toks:
        return True
    if any(term in n for term in ("moneyline", "money line", "money_line")):
        return True
    return any(
        side in n
        for side in (
            "home odds", "away odds", "home ml", "away ml", "home moneyline", "away moneyline",
            "odds home", "odds away", "team1 moneyline", "team2 moneyline", "home team odds", "away team odds",
        )
    )


def _is_spread(n: str) -> bool:
    return any(term in n for term in ("spread", "handicap", "puck line", "run line")) or "ats" in _toks(n)


def _is_total(n: str) -> bool:
    if "ou" in _toks(n) or "total" in n:
        return True
    return any(term in n for term in ("over under", "over/under", "game total"))


def _is_opening(n: str) -> bool:
    return "open" in n


def _is_closing(n: str) -> bool:
    return "clos" in n


def _is_timestamp(n: str) -> bool:
    toks = _toks(n)
    if toks & {"ts", "epoch"}:
        return True
    return any(
        term in n
        for term in (
            "timestamp", "datetime", "date", "time", "updated", "captured", "recorded",
            "snapshot", "commence", "kickoff", "tipoff", "scheduled", "created", "pulled", "fetched",
        )
    )


def _is_result(n: str) -> bool:
    toks = _toks(n)
    if toks & {"ftr", "fthg", "ftag", "won", "winner"}:
        return True
    if any(term in n for term in ("home points", "away points", "home goal", "away goal", "goals for", "goals against", "winning margin")):
        return True
    return any(term in n for term in ("result", "final", "score", "outcome", "settle", "actual", "goals"))


def _is_player_name(n: str) -> bool:
    if "player" in n or "athlete" in n or "batter" in n or "pitcher" in n:
        return True
    return n in {"name", "full name", "player name", "first name", "last name"}


def _is_player_id(n: str) -> bool:
    toks = _toks(n)
    return ("player" in toks and "id" in toks) or any(term in n for term in ("player_id", "playerid", "person id", "athlete id"))


def _is_prop_type(n: str) -> bool:
    return any(term in n for term in ("prop", "stat type", "bet type", "market type", "wager type", "selection type"))


def _is_prop_line(n: str) -> bool:
    toks = _toks(n)
    if "line" in toks or "point" in toks or "threshold" in toks:
        return True
    return any(term in n for term in ("prop line", "ou line", "o u line"))


def _is_team_or_game(n: str) -> bool:
    return any(term in n for term in ("team", "home", "away", "game", "match", "opponent", "fixture", "club", "host", "visitor"))


COLUMN_SIGNALS: dict[str, Any] = {
    "player_name": _is_player_name,
    "player_id": _is_player_id,
    "prop_type": _is_prop_type,
    "prop_line": _is_prop_line,
    "over_price": _is_over_price,
    "under_price": _is_under_price,
    "yes_no_price": _is_yes_no_price,
    "moneyline": _is_moneyline,
    "spread": _is_spread,
    "total": _is_total,
    "opening": _is_opening,
    "closing": _is_closing,
    "timestamp": _is_timestamp,
    "result": _is_result,
    "team_or_game": _is_team_or_game,
}


def column_signals(column: str) -> list[str]:
    """Return the list of signal tags a single column name triggers."""

    normalized = _norm(column)
    return [name for name, detector in COLUMN_SIGNALS.items() if detector(normalized)]


@dataclass
class FileProfile:
    """Per-file inspection result."""

    file: str
    relative_path: str
    extension: str
    loaded: bool = False
    load_error: str = ""
    rows: int = 0
    column_count: int = 0
    columns: list[str] = field(default_factory=list)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    detected_sport: str = "unknown"
    detected_market_type: str = "unknown"
    looks_like_game_odds: bool = False
    looks_like_player_props: bool = False
    has_player_names: bool = False
    has_player_id: bool = False
    has_prop_type: bool = False
    has_prop_line: bool = False
    has_over_price: bool = False
    has_under_price: bool = False
    has_yes_no_price: bool = False
    has_moneyline: bool = False
    has_spread: bool = False
    has_total: bool = False
    has_opening_odds: bool = False
    has_closing_odds: bool = False
    has_timestamps: bool = False
    timestamp_column_count: int = 0
    has_final_results: bool = False
    has_team_or_game: bool = False
    has_stats: bool = False
    can_be_used_for_clv: bool = False
    can_be_used_for_settlement: bool = False
    column_signal_map: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Aggregate inspection result for one dataset (folder or slug)."""

    identifier: str
    source: str  # "local_path" | "kaggle_slug"
    resolved_path: str = ""
    status: str = "ok"  # "ok" | "not_downloaded" | "empty" | "error"
    status_detail: str = ""
    detected_sports: list[str] = field(default_factory=list)
    file_count: int = 0
    loaded_file_count: int = 0
    total_rows: int = 0
    classification: str = "unusable_or_unknown"
    recommendation: str = "manual_review"
    recommendation_reason: str = ""
    flags: list[str] = field(default_factory=list)
    files: list[FileProfile] = field(default_factory=list)


# --- loading ----------------------------------------------------------------
def list_dataset_files(root: str | Path) -> list[Path]:
    """Return supported files under ``root`` (recursive), sorted by path."""

    base = Path(root)
    if base.is_file():
        return [base] if _is_supported(base) else []
    if not base.exists():
        return []
    files = [path for path in base.rglob("*") if path.is_file() and _is_supported(path)]
    return sorted(files)


def load_table(path: str | Path, max_rows: int | None = 2000) -> tuple[pd.DataFrame | None, str]:
    """Load a tabular file (incl. gzipped CSV/TSV/JSON). Return (frame_or_None, error)."""

    path = Path(path)
    kind, gz = _file_kind(path.name)
    if kind is None:
        return None, f"unsupported_extension:{path.suffix}"
    compression = "gzip" if gz else "infer"
    try:
        if kind == ".csv":
            frame = pd.read_csv(path, nrows=max_rows, low_memory=False, compression=compression)
            return _retry_alt_delimiter(path, frame, max_rows, compression), ""
        if kind == ".tsv":
            return pd.read_csv(path, nrows=max_rows, low_memory=False, sep="\t", compression=compression), ""
        if kind == ".txt":
            # Unknown delimiter: let pandas sniff, falling back to comma.
            try:
                return pd.read_csv(path, nrows=max_rows, engine="python", sep=None, compression=compression), ""
            except Exception:  # noqa: BLE001
                return pd.read_csv(path, nrows=max_rows, low_memory=False, compression=compression), ""
        if kind == ".parquet":
            frame = pd.read_parquet(path)
            return (frame.head(max_rows) if max_rows else frame), ""
        if kind in {".json", ".jsonl"}:
            frame = pd.read_json(path, lines=kind == ".jsonl", compression=compression)
            return (frame.head(max_rows) if max_rows else frame), ""
        if kind in {".xlsx", ".xls"}:
            try:
                return pd.read_excel(path, nrows=max_rows), ""
            except Exception:  # noqa: BLE001 - some ".xls" exports are really CSV/HTML.
                frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
                return _retry_alt_delimiter(path, frame, max_rows, "infer"), ""
    except Exception as exc:  # noqa: BLE001 - inspection must never crash on one bad file.
        return None, f"{type(exc).__name__}: {exc}"
    return None, f"unsupported_extension:{kind}"


def _retry_alt_delimiter(path: Path, frame: pd.DataFrame, max_rows: int | None, compression: str) -> pd.DataFrame:
    """If a comma read collapsed to one column, retry with ;, tab, or | delimiters."""

    if frame.shape[1] != 1:
        return frame
    header = str(frame.columns[0])
    for sep in (";", "\t", "|"):
        if sep not in header:
            continue
        try:
            alt = pd.read_csv(path, nrows=max_rows, low_memory=False, sep=sep, compression=compression)
        except Exception:  # noqa: BLE001
            continue
        if alt.shape[1] > 1:
            return alt
    return frame


# --- sport / market detection ----------------------------------------------
_SPORT_KEYWORDS = (
    ("wnba", "WNBA"),
    ("nba", "NBA"),
    ("nfl", "NFL"),
    ("mlb", "MLB"),
    ("nhl", "NHL"),
    ("soccer", "soccer"),
    ("premier league", "soccer"),
    ("laliga", "soccer"),
    ("bundesliga", "soccer"),
    ("epl", "soccer"),
    ("uefa", "soccer"),
    ("baseball", "MLB"),
    ("hockey", "NHL"),
    ("basketball", "basketball"),
)


def detect_sport(text_blobs: list[str]) -> str:
    """Detect a sport from filenames, slugs, column names, and sampled values."""

    haystack = " ".join(_norm(blob) for blob in text_blobs if blob)
    for keyword, sport in _SPORT_KEYWORDS:
        if keyword in haystack:
            return sport
    if "football" in haystack:
        return "football_unspecified"
    return "unknown"


def _detect_market_type(profile: FileProfile) -> str:
    if profile.looks_like_player_props:
        return "player_props"
    if profile.looks_like_game_odds:
        return "game_odds"
    odds_signals = any(
        [profile.has_moneyline, profile.has_spread, profile.has_total, profile.has_over_price, profile.has_under_price]
    )
    if odds_signals:
        return "odds_unstructured"
    if profile.has_stats:
        return "stats"
    if profile.has_final_results:
        return "results_or_schedule"
    return "unknown"


def profile_file(path: str | Path, root: str | Path, slug: str = "", sample_rows: int = 5) -> FileProfile:
    """Inspect one file and return a populated :class:`FileProfile`."""

    path = Path(path)
    root = Path(root)
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = path.name
    kind, gz = _file_kind(path.name)
    profile = FileProfile(
        file=path.name,
        relative_path=relative,
        extension=f"{kind or path.suffix.lower()}{'.gz' if gz else ''}",
    )
    frame, error = load_table(path)
    if frame is None:
        profile.load_error = error
        profile.detected_sport = detect_sport([slug, path.name])
        return profile

    profile.loaded = True
    profile.rows = int(len(frame))
    profile.columns = [str(column) for column in frame.columns]
    profile.column_count = len(profile.columns)
    profile.sample_rows = _sample_rows(frame, sample_rows)
    profile.column_signal_map = {column: column_signals(column) for column in profile.columns}

    triggered: dict[str, bool] = {name: False for name in COLUMN_SIGNALS}
    timestamp_columns = 0
    for tags in profile.column_signal_map.values():
        for tag in tags:
            triggered[tag] = True
        if "timestamp" in tags:
            timestamp_columns += 1

    normalized_columns = [_norm(column) for column in profile.columns]
    profile.has_stats = any(_toks(n) & STAT_TOKENS for n in normalized_columns)
    profile.has_player_names = triggered["player_name"]
    profile.has_player_id = triggered["player_id"]
    profile.has_prop_type = triggered["prop_type"]
    profile.has_prop_line = triggered["prop_line"]
    profile.has_over_price = triggered["over_price"]
    profile.has_under_price = triggered["under_price"]
    profile.has_yes_no_price = triggered["yes_no_price"]
    profile.has_moneyline = triggered["moneyline"]
    profile.has_spread = triggered["spread"]
    profile.has_total = triggered["total"]
    profile.has_opening_odds = triggered["opening"]
    profile.has_closing_odds = triggered["closing"]
    profile.has_timestamps = triggered["timestamp"]
    profile.timestamp_column_count = timestamp_columns
    profile.has_final_results = triggered["result"]
    profile.has_team_or_game = triggered["team_or_game"]

    profile.looks_like_player_props = bool(
        (profile.has_player_names or profile.has_player_id)
        and (profile.has_prop_type or profile.has_prop_line)
        and (profile.has_over_price or profile.has_under_price or profile.has_yes_no_price)
    )
    profile.looks_like_game_odds = bool(
        profile.has_team_or_game and (profile.has_moneyline or profile.has_spread or profile.has_total)
    )
    profile.can_be_used_for_clv = bool(
        profile.timestamp_column_count >= 2 or (profile.has_opening_odds and profile.has_closing_odds)
    )
    profile.can_be_used_for_settlement = bool(
        profile.has_final_results or (profile.has_team_or_game and profile.has_timestamps)
    )

    profile.detected_sport = detect_sport([slug, path.name, " ".join(profile.columns), _sample_text(frame)])
    profile.detected_market_type = _detect_market_type(profile)
    return profile


def _sample_rows(frame: pd.DataFrame, count: int) -> list[dict[str, Any]]:
    head = frame.head(count)
    records: list[dict[str, Any]] = []
    for _, row in head.iterrows():
        records.append({str(key): _json_scalar(value) for key, value in row.items()})
    return records


def _sample_text(frame: pd.DataFrame, max_cells: int = 200) -> str:
    object_columns = [column for column in frame.columns if frame[column].dtype == object]
    if not object_columns:
        return ""
    values = frame[object_columns].head(20).to_numpy().ravel()[:max_cells]
    return " ".join(str(value) for value in values)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    return str(value)


# --- classification & recommendation ---------------------------------------
def classify_dataset(files: list[FileProfile]) -> str:
    """Classify a dataset from its per-file signals."""

    loaded = [profile for profile in files if profile.loaded]
    if not loaded:
        return "unusable_or_unknown"
    any_props = any(profile.looks_like_player_props for profile in loaded)
    any_game_odds = any(profile.looks_like_game_odds for profile in loaded)
    any_odds = any_game_odds or any(
        profile.has_moneyline or profile.has_spread or profile.has_total or profile.has_over_price or profile.has_under_price
        for profile in loaded
    )
    any_results = any(profile.has_final_results for profile in loaded)
    any_settlement = any(profile.can_be_used_for_settlement for profile in loaded)
    any_stats = any(profile.has_stats for profile in loaded)

    if any_props:
        return "player_prop_ready" if (any_results or any_settlement) else "odds_without_results"
    if any_game_odds:
        return "game_odds_ready" if (any_results or any_settlement) else "odds_without_results"
    if any_odds:
        return "odds_without_results"
    if any_stats:
        return "stats_only"
    if any_results:
        return "results_without_odds"
    return "unusable_or_unknown"


def _dataset_flags(files: list[FileProfile], classification: str) -> list[str]:
    flags: list[str] = []
    loaded = [profile for profile in files if profile.loaded]
    if any(profile.load_error for profile in files):
        flags.append("some_files_failed_to_load")
    if loaded and not any(profile.has_final_results or profile.can_be_used_for_settlement for profile in loaded):
        flags.append("no_results_for_grading")
    if loaded and not any(profile.can_be_used_for_clv for profile in loaded):
        flags.append("no_clv_support")
    odds_present = any(
        profile.has_moneyline or profile.has_spread or profile.has_total or profile.looks_like_player_props
        for profile in loaded
    )
    if odds_present and "no_results_for_grading" in flags:
        flags.append("odds_present_but_unsettleable")
    if classification == "odds_without_results":
        flags.append("needs_join_to_results")
    return flags


def recommend_dataset(classification: str, sports: list[str], flags: list[str]) -> tuple[str, str]:
    """Map a classification + sport + flags to a recommendation and reason."""

    nba_relevant = any(sport in {"NBA", "WNBA", "basketball"} for sport in sports)
    if classification == "player_prop_ready":
        return "use_now", "Player props with a way to grade outcomes - the highest-value case."
    if classification == "game_odds_ready":
        if nba_relevant:
            return "use_now", "Gradeable game odds for a relevant basketball league."
        return "use_later", "Gradeable game odds, but not an NBA/WNBA league - keep for multi-sport expansion."
    if classification == "odds_without_results":
        return "manual_review", "Odds present but no results/closing structure found; confirm joins before use."
    if classification == "results_without_odds":
        return "use_later", "Results/schedule without odds; useful for settling or features, not pricing."
    if classification == "stats_only":
        return "use_later", "Box-score stats only; useful for features/actuals, not market pricing."
    return "skip", "No usable odds, stats, or results detected."


# --- slug resolution --------------------------------------------------------
def resolve_dataset_path(
    identifier: str,
    download: bool = False,
    local_root: str | Path | None = None,
) -> tuple[str, str, str]:
    """Resolve an identifier to a local path.

    Returns ``(resolved_path, source, status_detail)``. ``source`` is
    ``"local_path"`` or ``"kaggle_slug"``. If a slug is given and ``download`` is
    False (or kagglehub is unavailable), no download happens.
    """

    candidate = Path(identifier)
    if candidate.exists():
        return str(candidate), "local_path", "local_path"

    if local_root is not None:
        slug_dir = Path(local_root) / identifier.replace("/", "__")
        if slug_dir.exists():
            return str(slug_dir), "kaggle_slug", "found_in_local_root"

    looks_like_slug = "/" in identifier and not candidate.is_absolute()
    if not looks_like_slug:
        return "", "local_path", "path_not_found"
    if not download:
        return "", "kaggle_slug", "not_downloaded"

    try:
        import kagglehub  # noqa: PLC0415 - optional dependency, imported on demand.
    except ImportError:
        return "", "kaggle_slug", "kagglehub_not_installed"
    try:
        path = kagglehub.dataset_download(identifier)
    except Exception as exc:  # noqa: BLE001
        return "", "kaggle_slug", f"download_failed:{type(exc).__name__}:{exc}"
    return str(path), "kaggle_slug", "downloaded"


def profile_dataset(
    identifier: str,
    download: bool = False,
    local_root: str | Path | None = None,
    sample_rows: int = 5,
) -> DatasetProfile:
    """Profile a single dataset given a local path or a Kaggle slug."""

    resolved_path, source, status_detail = resolve_dataset_path(identifier, download=download, local_root=local_root)
    profile = DatasetProfile(identifier=identifier, source=source, resolved_path=resolved_path, status_detail=status_detail)

    if not resolved_path:
        profile.status = "not_downloaded" if status_detail in {"not_downloaded", "kagglehub_not_installed"} else "error"
        profile.detected_sports = [detect_sport([identifier])] if detect_sport([identifier]) != "unknown" else []
        profile.classification = "unusable_or_unknown"
        profile.recommendation = "manual_review"
        profile.recommendation_reason = f"Not inspected ({status_detail}). Provide --path or run with --download."
        profile.flags = [status_detail]
        return profile

    files = list_dataset_files(resolved_path)
    profile.file_count = len(files)
    if not files:
        profile.status = "empty"
        profile.recommendation = "skip"
        profile.recommendation_reason = "No supported files (csv/parquet/json/xlsx/txt) found."
        profile.flags = ["no_supported_files"]
        return profile

    profile.files = [profile_file(path, resolved_path, slug=identifier, sample_rows=sample_rows) for path in files]
    profile.loaded_file_count = sum(1 for item in profile.files if item.loaded)
    profile.total_rows = sum(item.rows for item in profile.files)
    profile.detected_sports = sorted({item.detected_sport for item in profile.files if item.detected_sport != "unknown"})
    profile.classification = classify_dataset(profile.files)
    profile.flags = _dataset_flags(profile.files, profile.classification)
    profile.recommendation, profile.recommendation_reason = recommend_dataset(
        profile.classification, profile.detected_sports, profile.flags
    )
    if profile.loaded_file_count == 0:
        profile.status = "error"
    return profile


def profile_datasets(
    identifiers: list[str],
    download: bool = False,
    local_root: str | Path | None = None,
    sample_rows: int = 5,
) -> list[DatasetProfile]:
    return [profile_dataset(identifier, download=download, local_root=local_root, sample_rows=sample_rows) for identifier in identifiers]


# --- report building --------------------------------------------------------
def build_summary(profiles: list[DatasetProfile]) -> dict[str, Any]:
    """Build the JSON summary dictionary."""

    return {
        "report": "kaggle_dataset_profile",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_profiled": len(profiles),
        "datasets": [_dataset_summary(profile) for profile in profiles],
        "research_only": True,
        "approved": False,
        "note": "Inspection only. Detection is conservative; confirm before wiring into the pipeline.",
    }


def _dataset_summary(profile: DatasetProfile) -> dict[str, Any]:
    data = asdict(profile)
    data["files"] = [
        {
            "file": item.file,
            "relative_path": item.relative_path,
            "loaded": item.loaded,
            "load_error": item.load_error,
            "rows": item.rows,
            "column_count": item.column_count,
            "detected_sport": item.detected_sport,
            "detected_market_type": item.detected_market_type,
            "looks_like_player_props": item.looks_like_player_props,
            "looks_like_game_odds": item.looks_like_game_odds,
            "can_be_used_for_clv": item.can_be_used_for_clv,
            "can_be_used_for_settlement": item.can_be_used_for_settlement,
            "columns": item.columns,
            "sample_rows": item.sample_rows,
        }
        for item in profile.files
    ]
    return data


_FILE_INVENTORY_COLUMNS = [
    "dataset",
    "file",
    "relative_path",
    "extension",
    "loaded",
    "load_error",
    "rows",
    "column_count",
    "detected_sport",
    "detected_market_type",
    "looks_like_game_odds",
    "looks_like_player_props",
    "has_player_names",
    "has_player_id",
    "has_prop_type",
    "has_prop_line",
    "has_over_price",
    "has_under_price",
    "has_moneyline",
    "has_spread",
    "has_total",
    "has_opening_odds",
    "has_closing_odds",
    "has_timestamps",
    "has_final_results",
    "can_be_used_for_clv",
    "can_be_used_for_settlement",
]


def build_file_inventory(profiles: list[DatasetProfile]) -> pd.DataFrame:
    """One row per inspected file with all detection booleans."""

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for item in profile.files:
            record = {key: getattr(item, key) for key in _FILE_INVENTORY_COLUMNS if hasattr(item, key)}
            record["dataset"] = profile.identifier
            rows.append(record)
    return pd.DataFrame(rows, columns=_FILE_INVENTORY_COLUMNS)


def build_column_inventory(profiles: list[DatasetProfile]) -> pd.DataFrame:
    """One row per (dataset, file, column) with the signal tags it triggered."""

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for item in profile.files:
            for column in item.columns:
                rows.append(
                    {
                        "dataset": profile.identifier,
                        "file": item.file,
                        "column": column,
                        "normalized": _norm(column),
                        "signals": ", ".join(item.column_signal_map.get(column, [])),
                    }
                )
    return pd.DataFrame(rows, columns=["dataset", "file", "column", "normalized", "signals"])


def render_recommendations_md(profiles: list[DatasetProfile]) -> str:
    """Render the human-readable recommendations report."""

    lines: list[str] = ["# Kaggle Dataset Profiler Report", ""]
    lines.append("_Inspection only. Does not download unless asked, and never changes proof gates or betting._")
    lines.append("")
    lines.append("| dataset | status | sport(s) | classification | recommendation |")
    lines.append("| --- | --- | --- | --- | --- |")
    for profile in profiles:
        sports = ", ".join(profile.detected_sports) or "unknown"
        lines.append(
            f"| {profile.identifier} | {profile.status} | {sports} | {profile.classification} | {profile.recommendation} |"
        )
    lines.append("")

    for profile in profiles:
        lines.append(f"## {profile.identifier}")
        lines.append("")
        lines.append(f"- **Status:** {profile.status} ({profile.status_detail})")
        lines.append(f"- **Resolved path:** {profile.resolved_path or 'n/a'}")
        lines.append(f"- **Detected sport(s):** {', '.join(profile.detected_sports) or 'unknown'}")
        lines.append(f"- **Files:** {profile.file_count} found, {profile.loaded_file_count} loaded, {profile.total_rows:,} rows")
        lines.append(f"- **Classification:** {profile.classification}")
        lines.append(f"- **Recommendation:** {profile.recommendation} - {profile.recommendation_reason}")
        if profile.flags:
            lines.append(f"- **Flags:** {', '.join(profile.flags)}")
        if profile.files:
            lines.append("")
            lines.append("| file | rows | cols | market | props | game odds | clv | settle |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for item in profile.files:
                lines.append(
                    f"| {item.file} | {item.rows} | {item.column_count} | {item.detected_market_type} | "
                    f"{_yn(item.looks_like_player_props)} | {_yn(item.looks_like_game_odds)} | "
                    f"{_yn(item.can_be_used_for_clv)} | {_yn(item.can_be_used_for_settlement)} |"
                )
        lines.append("")
    return "\n".join(lines)


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def write_reports(profiles: list[DatasetProfile], reports_dir: str | Path) -> dict[str, str]:
    """Write all four report artifacts and return their paths."""

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    summary_path = reports / "kaggle_dataset_profile_summary.json"
    file_inventory_path = reports / "kaggle_dataset_file_inventory.csv"
    column_inventory_path = reports / "kaggle_dataset_column_inventory.csv"
    recommendations_path = reports / "kaggle_dataset_recommendations.md"

    summary_path.write_text(json.dumps(build_summary(profiles), indent=2), encoding="utf-8")
    build_file_inventory(profiles).to_csv(file_inventory_path, index=False)
    build_column_inventory(profiles).to_csv(column_inventory_path, index=False)
    recommendations_path.write_text(render_recommendations_md(profiles), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "file_inventory": str(file_inventory_path),
        "column_inventory": str(column_inventory_path),
        "recommendations": str(recommendations_path),
    }
