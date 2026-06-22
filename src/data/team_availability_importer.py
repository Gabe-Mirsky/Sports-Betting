"""Import manual team availability CSVs into the processed injuries file."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from data.injuries_loader import INJURY_COLUMNS, load_injuries, normalize_injuries


IMPORT_STATUSES = {"out", "doubtful", "questionable", "probable", "available", "unknown"}


@dataclass
class AvailabilityImportSummary:
    rows_read: int = 0
    rows_written: int = 0
    rows_dropped_blank_team: int = 0
    rows_dropped_unknown_player: int = 0
    duplicate_rows_removed: int = 0
    missing_last_updated_filled: int = 0
    statuses_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "rows_dropped_blank_team": self.rows_dropped_blank_team,
            "rows_dropped_unknown_player": self.rows_dropped_unknown_player,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "missing_last_updated_filled": self.missing_last_updated_filled,
            "statuses_found": self.statuses_found,
            "warnings": self.warnings,
        }


def import_team_availability(
    input_paths: str | Path | Iterable[str | Path],
    output_path: str | Path,
    append: bool = False,
    existing_path: str | Path | None = None,
    aliases_path: str | Path | None = None,
    as_of_date: str | date | pd.Timestamp | None = None,
    drop_unknown_player_rows: bool = False,
    keep_unknown_status: bool = True,
) -> tuple[pd.DataFrame, AvailabilityImportSummary]:
    """Normalize one or more manual availability CSVs and write injuries.csv."""

    summary = AvailabilityImportSummary()
    raw = read_availability_inputs(input_paths)
    summary.rows_read = int(len(raw))

    config = {"aliases_path": str(aliases_path)} if aliases_path and Path(aliases_path).exists() else {}
    normalized = normalize_injuries(raw, config)
    normalized = _postprocess_normalized(
        normalized,
        summary,
        as_of_date=as_of_date,
        drop_unknown_player_rows=drop_unknown_player_rows,
        keep_unknown_status=keep_unknown_status,
    )

    if append:
        source = Path(existing_path) if existing_path else Path(output_path)
        if source.exists():
            existing = normalize_injuries(load_injuries(source), config)
            normalized = pd.concat([existing, normalized], ignore_index=True)

    before_dedupe = len(normalized)
    normalized = _dedupe_availability_rows(normalized)
    summary.duplicate_rows_removed = int(before_dedupe - len(normalized))
    summary.rows_written = int(len(normalized))
    summary.statuses_found = sorted(normalized["status"].dropna().astype(str).str.lower().unique().tolist())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return normalized, summary


def read_availability_inputs(input_paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Read CSV files or folders of CSVs into one raw frame."""

    if isinstance(input_paths, (str, Path)):
        paths = [input_paths]
    else:
        paths = list(input_paths)

    frames: list[pd.DataFrame] = []
    for input_path in paths:
        path = Path(input_path)
        if path.is_dir():
            for csv_path in sorted(path.glob("*.csv")):
                frames.append(_read_one_csv(csv_path))
        else:
            frames.append(_read_one_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_one_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Availability input file not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _postprocess_normalized(
    normalized: pd.DataFrame,
    summary: AvailabilityImportSummary,
    as_of_date: str | date | pd.Timestamp | None,
    drop_unknown_player_rows: bool,
    keep_unknown_status: bool,
) -> pd.DataFrame:
    frame = normalized.copy()
    for column in INJURY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame["team"] = frame["team"].map(_clean_text)
    blank_team = frame["team"].eq("")
    summary.rows_dropped_blank_team = int(blank_team.sum())
    if summary.rows_dropped_blank_team:
        summary.warnings.append(f"Dropped {summary.rows_dropped_blank_team} row(s) with blank team.")
    frame = frame[~blank_team].copy()

    frame["player_name"] = frame["player_name"].map(_clean_text)
    unknown_player = frame["player_name"].eq("") | frame["player_name"].str.lower().str.match(r"^unknown player( \d+)?$")
    if drop_unknown_player_rows:
        summary.rows_dropped_unknown_player = int(unknown_player.sum())
        if summary.rows_dropped_unknown_player:
            summary.warnings.append(
                f"Dropped {summary.rows_dropped_unknown_player} blank/unknown player row(s)."
            )
        frame = frame[~unknown_player].copy()
    elif int(unknown_player.sum()):
        summary.warnings.append(f"Kept {int(unknown_player.sum())} blank/unknown player row(s).")

    frame["status"] = frame["status"].map(_normalize_import_status)
    invalid_status = sorted(set(frame["status"].astype(str).str.lower()) - IMPORT_STATUSES)
    if invalid_status:
        summary.warnings.append(f"Unrecognized availability status values retained for validation: {invalid_status}.")
    if not keep_unknown_status:
        unknown = frame["status"].eq("unknown")
        if int(unknown.sum()):
            summary.warnings.append(
                f"Kept {int(unknown.sum())} unknown-status row(s); unknown is a valid placeholder status."
            )

    last_updated = pd.to_datetime(frame["last_updated"], errors="coerce")
    missing_updated = last_updated.isna()
    summary.missing_last_updated_filled = int(missing_updated.sum())
    fill_date = _as_of_date_text(as_of_date)
    frame.loc[missing_updated, "last_updated"] = fill_date
    frame["last_updated"] = pd.to_datetime(frame["last_updated"], errors="coerce")

    importance = pd.to_numeric(frame["importance_score"], errors="coerce")
    fallback = frame["expected_minutes_or_role"].map(_importance_from_role_for_import).astype(float)
    frame["importance_score"] = importance.fillna(fallback).fillna(0.25).clip(0.0, 1.0).astype(float)

    for column in ["injury_type", "position", "expected_minutes_or_role", "return_estimate", "source", "notes"]:
        frame[column] = frame[column].map(_clean_text)
    frame.loc[frame["expected_minutes_or_role"].eq(""), "expected_minutes_or_role"] = "unknown"
    frame.loc[frame["source"].eq(""), "source"] = "manual"

    return frame[INJURY_COLUMNS].copy()


def _dedupe_availability_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame[INJURY_COLUMNS].copy()
    work = frame.copy()
    work["_player_key"] = work["player_name"].astype(str).str.strip().str.lower()
    work["_updated_sort"] = pd.to_datetime(work["last_updated"], errors="coerce")
    work = work.sort_values(["team", "_player_key", "_updated_sort"], na_position="first")
    work = work.drop_duplicates(subset=["team", "_player_key"], keep="last")
    work = work.drop(columns=["_player_key", "_updated_sort"])
    return work.sort_values(["team", "player_name"]).reset_index(drop=True)[INJURY_COLUMNS]


def _normalize_import_status(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown"
    if text in {"gtd", "game-time decision", "game time decision", "day-to-day", "day to day"}:
        return "questionable"
    if text in {"inactive", "ruled out", "sidelined", "unavailable"}:
        return "out"
    if text in {"healthy", "no injury", "fit", "active", "cleared"}:
        return "available"
    return text


def _importance_from_role_for_import(value: object) -> float:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return 0.25
    if any(word in text for word in ("star", "key player", "key", "captain", "talisman")):
        return 1.0
    if "starter" in text or "starting" in text or "first team" in text or "first-team" in text:
        return 0.75
    if "rotation" in text or "squad" in text:
        return 0.4
    if any(word in text for word in ("bench", "depth", "reserve", "backup")):
        return 0.15
    return 0.25


def _as_of_date_text(value: str | date | pd.Timestamp | None) -> str:
    if value is None:
        return date.today().isoformat()
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return date.today().isoformat()
    return ts.date().isoformat()


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text
