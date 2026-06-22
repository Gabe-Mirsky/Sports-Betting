"""Import free Kaggle NBA odds files into the local sportsbook odds schema."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.team_aliases import normalize_team_name  # noqa: E402


INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sportsbook" / "kaggle"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sportsbook" / "nba_moneyline_odds.csv"
SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx"}
KAGGLE_TEAM_ROW_COLUMNS = {"date", "rot", "vh", "team", "final", "ml"}
KAGGLE_TEAM_GAME_COLUMNS = {
    "date",
    "season",
    "team",
    "home/visitor",
    "opponent",
    "score",
    "opponentscore",
    "moneyline",
    "opponentmoneyline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert free Kaggle NBA odds files into nba_moneyline_odds.csv.")
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    return parser.parse_args()


def infer_season_from_filename(path: Path) -> int | None:
    text = path.stem.lower().replace("_", " ").replace("-", " ")
    range_match = re.search(r"(20\d{2})\s*(?:to|thru|through)?\s*(\d{2})", text)
    if range_match:
        return int(range_match.group(1))
    compact_range = re.search(r"(20\d{2})[-_](\d{2})", path.stem.lower())
    if compact_range:
        return int(compact_range.group(1))
    year_match = re.search(r"(20\d{2})", text)
    if year_match:
        return int(year_match.group(1))
    return None


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [str(column).strip() for column in output.columns]
    return output.dropna(how="all").reset_index(drop=True)


def _read_frames(path: Path) -> list[tuple[str, pd.DataFrame]]:
    if path.suffix.lower() == ".csv":
        return [(path.stem, _clean_columns(pd.read_csv(path, low_memory=False)))]
    sheets = pd.read_excel(path, sheet_name=None)
    return [(str(sheet_name), _clean_columns(frame)) for sheet_name, frame in sheets.items()]


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def _is_kaggle_team_row_format(frame: pd.DataFrame) -> bool:
    return KAGGLE_TEAM_ROW_COLUMNS.issubset(set(_column_lookup(frame)))


def _is_kaggle_team_game_format(frame: pd.DataFrame) -> bool:
    return KAGGLE_TEAM_GAME_COLUMNS.issubset(set(_column_lookup(frame)))


def _parse_moneyline(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"pk", "pick", "nan", "none"}:
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def _parse_score(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def _parse_market_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"pk", "pick", "nan", "none"}:
        return np.nan
    if text.lower().startswith(("o", "u")):
        text = text[1:]
    text = text.replace("½", ".5")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else np.nan


def _parse_game_date(value: Any, season_start: int) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        # Numeric sportsbook dates like 1025 can be interpreted as nanoseconds by pandas.
        if not isinstance(value, (int, float, np.integer, np.floating)) or parsed.year > 1971:
            return parsed.normalize()
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) in {3, 4}:
        month = int(digits[:-2])
        day = int(digits[-2:])
    else:
        slash_match = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})\s*$", text)
        if not slash_match:
            return pd.NaT
        month = int(slash_match.group(1))
        day = int(slash_match.group(2))
    year = season_start if month >= 7 else season_start + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _parse_full_game_date(value: Any) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.normalize() if pd.notna(parsed) else pd.NaT


def _pair_rows(frame: pd.DataFrame, season_start: int, source_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    columns = _column_lookup(frame)
    warnings: list[str] = []
    working = frame.copy()
    working["_vh"] = working[columns["vh"]].astype(str).str.strip().str.upper()
    working = working[working["_vh"].isin(["V", "H"])].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(working) - 1:
        first = working.iloc[index]
        second = working.iloc[index + 1]
        if first["_vh"] == "V" and second["_vh"] == "H":
            away = first
            home = second
            index += 2
        elif first["_vh"] == "H" and second["_vh"] == "V":
            home = first
            away = second
            index += 2
        else:
            warnings.append(f"{source_name}: skipped unpaired row at position {index}.")
            index += 1
            continue

        game_date = _parse_game_date(home[columns["date"]], season_start)
        home_ml = _parse_moneyline(home[columns["ml"]])
        away_ml = _parse_moneyline(away[columns["ml"]])
        if pd.isna(game_date) or pd.isna(home_ml) or pd.isna(away_ml):
            warnings.append(f"{source_name}: skipped pair with missing date or ML at position {index - 2}.")
            continue

        home_close = _parse_market_number(home[columns["close"]]) if "close" in columns else np.nan
        away_close = _parse_market_number(away[columns["close"]]) if "close" in columns else np.nan
        spread = home_close if pd.notna(home_close) and abs(home_close) <= 60 else np.nan
        total_candidates = [value for value in [home_close, away_close] if pd.notna(value) and value >= 100]
        total = total_candidates[0] if total_candidates else np.nan

        rows.append(
            {
                "game_date": game_date.date().isoformat(),
                "home_team": normalize_team_name(home[columns["team"]]),
                "away_team": normalize_team_name(away[columns["team"]]),
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "home_score": _parse_score(home[columns["final"]]),
                "away_score": _parse_score(away[columns["final"]]),
                "sportsbook": "Kaggle christophertreasure/nba-odds-data",
                "timestamp": "",
                "is_closing": True,
                "spread": spread,
                "total": total,
            }
        )
    return rows, warnings


def _import_team_game_rows(frame: pd.DataFrame, source_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    columns = _column_lookup(frame)
    warnings: list[str] = []
    working = frame.copy()
    side = working[columns["home/visitor"]].astype(str).str.strip().str.lower()
    home_rows = working[side.isin(["vs", "home", "h"])].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, home in home_rows.iterrows():
        game_date = _parse_full_game_date(home[columns["date"]])
        home_ml = _parse_moneyline(home[columns["moneyline"]])
        away_ml = _parse_moneyline(home[columns["opponentmoneyline"]])
        if pd.isna(game_date) or pd.isna(home_ml) or pd.isna(away_ml):
            warnings.append(f"{source_name}: skipped home row {index} with missing date or moneyline.")
            continue
        rows.append(
            {
                "game_date": game_date.date().isoformat(),
                "home_team": normalize_team_name(home[columns["team"]]),
                "away_team": normalize_team_name(home[columns["opponent"]]),
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "home_score": _parse_score(home[columns["score"]]),
                "away_score": _parse_score(home[columns["opponentscore"]]),
                "sportsbook": "Kaggle christophertreasure/nba-odds-data",
                "timestamp": "",
                "is_closing": True,
                "spread": _parse_market_number(home[columns["spread"]]) if "spread" in columns else np.nan,
                "total": _parse_market_number(home[columns["total"]]) if "total" in columns else np.nan,
            }
        )
    return rows, warnings


def import_file(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    season_start = infer_season_from_filename(path)
    imported: list[dict[str, Any]] = []
    warnings: list[str] = []
    for sheet_name, frame in _read_frames(path):
        if frame.empty:
            continue
        if _is_kaggle_team_game_format(frame):
            rows, sheet_warnings = _import_team_game_rows(frame, f"{path.name}::{sheet_name}")
            imported.extend(rows)
            warnings.extend(f"WARNING: {warning}" for warning in sheet_warnings)
            continue
        if _is_kaggle_team_row_format(frame):
            if season_start is None:
                warnings.append(f"WARNING: Could not infer season from file name, skipping: {path.name}::{sheet_name}")
                continue
            rows, sheet_warnings = _pair_rows(frame, season_start, f"{path.name}::{sheet_name}")
            imported.extend(rows)
            warnings.extend(f"WARNING: {warning}" for warning in sheet_warnings)
            continue
        warnings.append(f"WARNING: {path.name}::{sheet_name} does not match a supported Kaggle NBA odds format, skipping.")
    return imported, warnings


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    if not input_dir.exists():
        print(f"ERROR: Kaggle odds directory does not exist: {input_dir}")
        return 1

    files = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    print(f"Found {len(files):,} candidate Kaggle odds file(s) in: {input_dir}")
    all_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for path in files:
        rows, warnings = import_file(path)
        all_warnings.extend(warnings)
        for warning in warnings:
            print(warning)
        print(f"Imported {len(rows):,} game row(s) from: {path}")
        all_rows.extend(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        all_rows,
        columns=[
            "game_date",
            "home_team",
            "away_team",
            "home_moneyline",
            "away_moneyline",
            "home_score",
            "away_score",
            "sportsbook",
            "timestamp",
            "is_closing",
            "spread",
            "total",
        ],
    )
    if not output.empty:
        output = output.drop_duplicates(["game_date", "home_team", "away_team", "home_moneyline", "away_moneyline"])
        output = output.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output):,} game row(s) to: {output_path}")
    _print_2022_23_audit(input_dir, files, output, all_warnings)
    if output.empty:
        print("WARNING: No usable Kaggle NBA odds rows were imported.")
        return 1
    return 0


def _season_start_from_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)


def _print_2022_23_audit(input_dir: Path, files: list[Path], output: pd.DataFrame, warnings: list[str]) -> None:
    raw_rows = 0
    for path in files:
        for _, frame in _read_frames(path):
            columns = _column_lookup(frame)
            if "date" not in columns:
                continue
            if _is_kaggle_team_row_format(frame):
                season_start = infer_season_from_filename(path)
                if season_start == 2022:
                    raw_rows += int(len(frame))
                continue
            dates = pd.to_datetime(frame[columns["date"]], errors="coerce")
            season_values = dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)
            raw_rows += int(season_values.eq(2022).sum())

    if output.empty:
        converted_2022 = pd.DataFrame()
    else:
        working = output.copy()
        working["_season"] = _season_start_from_dates(working["game_date"])
        converted_2022 = working[working["_season"].eq(2022)].copy()

    skipped_reasons: dict[str, int] = {}
    for warning in warnings:
        if "skipped" not in warning.lower():
            continue
        reason = warning.split(":", 2)[-1].strip() if ":" in warning else warning
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    print("\n2022-23 Kaggle import audit")
    print("---------------------------")
    print(f"Source directory: {input_dir}")
    print(f"Raw Kaggle rows for 2022-23: {raw_rows:,}")
    print(f"Converted games for 2022-23: {len(converted_2022):,}")
    print(f"Skipped pairs for 2022-23: {sum(skipped_reasons.values()):,}")
    if skipped_reasons:
        print("Skip reasons:")
        for reason, count in sorted(skipped_reasons.items()):
            print(f"- {reason}: {count:,}")
    else:
        print("Skip reasons: none")
    if not converted_2022.empty:
        dates = pd.to_datetime(converted_2022["game_date"], errors="coerce")
        print(f"Date range of imported 2022-23 odds: {dates.min().date()} to {dates.max().date()}")
    else:
        print("Date range of imported 2022-23 odds: n/a")


if __name__ == "__main__":
    raise SystemExit(main())
