"""Free historical backfill scaffolding (read-only, research-only).

This module is the shared plumbing for backfilling more *games* into the
Recorded Games inventory from FREE or already-available sources. It enables NO
betting, NO predictions, NO parlays, and touches NO model/readiness gates.

Pipeline shape:

    raw source file ──(importer normalize)──► staging frame
        data/staging/free_backfill/<source>_<league>.csv
                              │
                              ▼  (build_historical_game_inventory)
        data/processed/historical_game_inventory.csv

The four backfill data types are tracked as independent boolean columns so the
dashboard can show exactly what we have per game:

    * outcome_available     - final score / box score exists
    * game_odds_available   - game-level market odds exist (moneyline/1X2/spread/total)
    * prop_odds_available   - player-prop market odds exist
    * closing_available     - a snapshot/line near game start exists

Network fetching lives in the importer skeletons and is intentionally inert
until a source list + date range is approved; this module only normalizes and
inventories local files.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError

import pandas as pd

from data.canonical_games import canonical_game_key_series

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "free_backfill"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "free_backfill"
INVENTORY_PATH = PROJECT_ROOT / "data" / "processed" / "historical_game_inventory.csv"

_USER_AGENT = "Mozilla/5.0 (nba_kalshi_predictor free-backfill; research-only)"

# Columns every staging file carries (one row per game).
STAGING_COLUMNS = [
    "source", "sport", "league", "game_date", "home_team", "away_team",
    "home_score", "away_score", "home_odds", "draw_odds", "away_odds",
    "outcome_available", "game_odds_available", "prop_odds_available",
    "closing_available", "canonical_game_key",
]

# Columns of the normalized inventory consumed by Recorded Games.
INVENTORY_COLUMNS = [
    "canonical_game_key", "league", "game_date", "home_team", "away_team",
    "source", "outcome_available", "game_odds_available", "prop_odds_available",
    "closing_available",
]

_FLAG_COLUMNS = (
    "outcome_available", "game_odds_available", "prop_odds_available", "closing_available"
)
_ODDS_COLUMNS = ("home_odds", "draw_odds", "away_odds")
_TRUE_TOKENS = {"true", "1", "yes", "t", "y"}

# Default sport per league label (importers may override).
SPORT_BY_LEAGUE = {
    "EPL": "soccer", "LA_LIGA": "soccer", "SERIE_A": "soccer", "BUNDESLIGA": "soccer",
    "LIGUE_1": "soccer", "MLS": "soccer", "UEFA_CL": "soccer", "WORLD_CUP": "soccer",
    "MLB": "baseball", "NFL": "football", "NHL": "hockey", "NBA": "basketball",
    "WNBA": "basketball", "NCAAB": "basketball", "NCAAF": "football",
}


def to_bool_series(series: pd.Series) -> pd.Series:
    """Coerce mixed truthy/string values to a clean boolean Series."""

    return series.map(lambda v: str(v).strip().lower() in _TRUE_TOKENS).astype(bool)


def pick_series(frame: pd.DataFrame, candidates: tuple[str, ...], default: object = pd.NA) -> pd.Series:
    """Return the first present column among ``candidates`` (or a default Series)."""

    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def valid_odds_value_series(series: pd.Series) -> pd.Series:
    """True where an odds field contains a usable numeric value."""

    numeric = pd.to_numeric(series, errors="coerce")
    return (
        numeric.notna()
        & numeric.ne(0)
        & numeric.ne(float("inf"))
        & numeric.ne(float("-inf"))
    )


def pick_valid_odds_series(
    frame: pd.DataFrame, candidates: tuple[str, ...], default: object = pd.NA
) -> pd.Series:
    """Return the first row-level valid odds value among candidate columns."""

    result = pd.Series([default] * len(frame), index=frame.index)
    filled = pd.Series(False, index=frame.index)
    for name in candidates:
        if name not in frame.columns:
            continue
        candidate = frame[name]
        use_candidate = ~filled & valid_odds_value_series(candidate)
        if use_candidate.any():
            result.loc[use_candidate] = candidate.loc[use_candidate]
            filled = filled | use_candidate
    return result


def _any_valid_odds(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    present = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            present = present | valid_odds_value_series(frame[column])
    return present


def coerce_staging_frame(
    partial: pd.DataFrame, *, source: str, sport: str, league: str
) -> pd.DataFrame:
    """Turn an importer's partial frame into the full staging schema.

    ``partial`` must provide at least ``game_date``, ``home_team`` and
    ``away_team``. Optional columns: ``home_score``/``away_score`` (drive
    ``outcome_available``), ``home_odds``/``draw_odds``/``away_odds`` (drive
    ``game_odds_available``), and any of the four availability flags to set them
    explicitly. Rows whose canonical key cannot be built are dropped.
    """

    df = partial.copy()
    for required in ("game_date", "home_team", "away_team"):
        if required not in df.columns:
            raise ValueError(f"staging input missing required column: {required!r}")

    df["source"] = source
    df["sport"] = sport
    df["league"] = league

    for column in ("home_score", "away_score", *_ODDS_COLUMNS):
        if column not in df.columns:
            df[column] = pd.NA

    # Infer availability flags when not supplied explicitly.
    if "outcome_available" in df.columns:
        df["outcome_available"] = to_bool_series(df["outcome_available"])
    else:
        df["outcome_available"] = (
            df["home_score"].notna() & df["away_score"].notna()
        )
    if "game_odds_available" in df.columns:
        df["game_odds_available"] = (
            to_bool_series(df["game_odds_available"]) | _any_valid_odds(df, _ODDS_COLUMNS)
        )
    else:
        df["game_odds_available"] = _any_valid_odds(df, _ODDS_COLUMNS)
    df["prop_odds_available"] = (
        to_bool_series(df["prop_odds_available"]) if "prop_odds_available" in df.columns else False
    )
    df["closing_available"] = (
        to_bool_series(df["closing_available"]) if "closing_available" in df.columns else False
    )
    # A closing line must be tied to a row-level price source. Game odds are the
    # normal source; prop odds are the documented closing-only exception.
    df["closing_available"] = df["closing_available"] & (
        df["game_odds_available"] | df["prop_odds_available"]
    )

    df["canonical_game_key"] = canonical_game_key_series(
        df, sport, "league", "game_date", "home_team", "away_team"
    )
    df = df[df["canonical_game_key"].astype(str) != ""].reset_index(drop=True)
    return df.reindex(columns=STAGING_COLUMNS)


def write_staging_frame(
    partial: pd.DataFrame, *, source: str, sport: str, league: str,
    out_dir: Path | str = STAGING_DIR,
) -> Path:
    """Coerce + write a staging file at ``<out_dir>/<source>_<league>.csv``."""

    staged = coerce_staging_frame(partial, source=source, sport=sport, league=league)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}_{league}.csv"
    staged.to_csv(path, index=False)
    return path


def _read_staging_files(staging_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(staging_dir).glob("*.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=STAGING_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_historical_game_inventory(
    staging_dir: Path | str = STAGING_DIR,
    out_path: Path | str = INVENTORY_PATH,
) -> pd.DataFrame:
    """Collapse every staging file into one row per canonical_game_key.

    Availability flags are OR-ed across sources; ``source`` lists every source
    that contributed the game. Read-only aggregation - no betting/prediction.
    """

    raw = _read_staging_files(staging_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if raw.empty or "canonical_game_key" not in raw.columns:
        inventory = pd.DataFrame(columns=INVENTORY_COLUMNS)
        inventory.to_csv(out_path, index=False)
        return inventory

    raw = raw[raw["canonical_game_key"].astype(str).str.len() > 0].copy()
    for flag in _FLAG_COLUMNS:
        raw[flag] = to_bool_series(raw[flag]) if flag in raw.columns else False
    raw["closing_available"] = raw["closing_available"] & (
        raw["game_odds_available"] | raw["prop_odds_available"]
    )

    grouped = raw.groupby("canonical_game_key", as_index=False).agg(
        league=("league", "first"),
        game_date=("game_date", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        source=("source", lambda s: ",".join(sorted({str(v) for v in s if str(v) != "nan"}))),
        outcome_available=("outcome_available", "max"),
        game_odds_available=("game_odds_available", "max"),
        prop_odds_available=("prop_odds_available", "max"),
        closing_available=("closing_available", "max"),
    )
    for flag in _FLAG_COLUMNS:
        grouped[flag] = grouped[flag].astype(bool)

    inventory = grouped.reindex(columns=INVENTORY_COLUMNS)
    inventory = inventory.sort_values(["league", "game_date", "canonical_game_key"]).reset_index(drop=True)
    inventory.to_csv(out_path, index=False)
    return inventory


def load_inventory(path: Path | str = INVENTORY_PATH) -> pd.DataFrame:
    """Read the historical game inventory (empty frame when absent)."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    for flag in _FLAG_COLUMNS:
        if flag in frame.columns:
            frame[flag] = to_bool_series(frame[flag])
    return frame


def run_importer_from_input(
    *, input_path: Path | str, normalize_fn: Callable[[pd.DataFrame], pd.DataFrame],
    source: str, sport: str, league: str, out_dir: Path | str = STAGING_DIR,
) -> tuple[Path, int]:
    """Read a local raw file, normalize it, and write a staging file (offline)."""

    raw = pd.read_csv(input_path, low_memory=False)
    path = write_staging_frame(
        normalize_fn(raw), source=source, sport=sport, league=league, out_dir=out_dir
    )
    return path, int(len(raw))


# --------------------------------------------------------------------------- #
# Live fetch helpers (only used once a source list + date range is approved).
# --------------------------------------------------------------------------- #
def http_get(url: str, *, timeout: int = 30, retries: int = 2) -> bytes:
    """GET a URL with a browser UA. Raises on 4xx; retries transient errors."""

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if 400 <= exc.code < 500:
                raise  # client error (e.g. 404 for a non-existent season); do not retry
            last_error = exc
        except (URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.5 * (attempt + 1))
    raise last_error if last_error else RuntimeError(f"failed to GET {url}")


def save_raw(content: bytes, *, source: str, name: str, raw_dir: Path | str = RAW_DIR) -> Path:
    """Persist a downloaded payload under data/raw/free_backfill/<source>/<name>."""

    folder = Path(raw_dir) / source
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(content)
    return path
