"""Leak-safe local injury and availability feature inputs."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from data.team_aliases import normalize_team_abbr
from data.validation import require_columns


AVAILABILITY_SCHEMA_COLUMNS = [
    "report_date",
    "game_date",
    "team_abbr",
    "player_name",
    "status",
]
OPTIONAL_IMPACT_COLUMNS = [
    "impact_weight",
    "expected_minutes",
    "avg_minutes",
    "rotation_minutes",
    "minutes",
]

AVAILABILITY_BASE_FEATURE_COLUMNS = [
    "availability_report_present",
    "availability_reported_players",
    "availability_players_out",
    "availability_players_doubtful",
    "availability_players_questionable",
    "availability_players_probable",
    "availability_players_available",
    "availability_players_unknown",
    "availability_out_or_doubtful",
    "availability_questionable_or_worse",
    "availability_out_weighted",
    "availability_doubtful_weighted",
    "availability_questionable_weighted",
    "availability_questionable_or_worse_weighted",
    "availability_status_severity_weighted",
    "availability_projected_minutes_lost",
]

OUT_STATUSES = {"out", "inactive", "not with team", "suspended"}
DOUBTFUL_STATUSES = {"doubtful"}
QUESTIONABLE_STATUSES = {"questionable", "game time decision", "game-time decision"}
PROBABLE_STATUSES = {"probable"}
AVAILABLE_STATUSES = {"available", "active", "will play", "probable to play"}
STATUS_SEVERITY = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.50,
    "probable": 0.15,
    "unknown": 0.25,
    "available": 0.0,
}


def _normalize_player_name(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def empty_availability_reports() -> pd.DataFrame:
    return pd.DataFrame(columns=AVAILABILITY_SCHEMA_COLUMNS)


def normalize_availability_status(status: object) -> str:
    text = str(status or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    if text in OUT_STATUSES:
        return "out"
    if text in DOUBTFUL_STATUSES:
        return "doubtful"
    if text in QUESTIONABLE_STATUSES:
        return "questionable"
    if text in PROBABLE_STATUSES:
        return "probable"
    if text in AVAILABLE_STATUSES:
        return "available"
    return "unknown" if text else "unknown"


def load_availability_reports(path: str | Path) -> pd.DataFrame:
    """Load optional local availability reports.

    The file is intentionally local and source-agnostic. It can be populated
    from any free source the user is allowed to use.
    """

    source = Path(path)
    if not source.exists():
        return empty_availability_reports()
    reports = pd.read_csv(source)
    return normalize_availability_reports(reports)


def normalize_availability_reports(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return empty_availability_reports()
    require_columns(reports, AVAILABILITY_SCHEMA_COLUMNS, dataframe_name="availability_reports")
    output = reports.copy()
    output["report_date"] = pd.to_datetime(output["report_date"], errors="coerce")
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output["team_abbr"] = output["team_abbr"].map(normalize_team_abbr)
    output["player_name"] = output["player_name"].fillna("").astype(str).str.strip()
    output["availability_status"] = output["status"].map(normalize_availability_status)
    if "impact_weight" not in output.columns:
        impact_source = None
        for column in ["expected_minutes", "avg_minutes", "rotation_minutes", "minutes"]:
            if column in output.columns:
                impact_source = column
                break
        if impact_source:
            output["impact_weight"] = pd.to_numeric(output[impact_source], errors="coerce")
        else:
            output["impact_weight"] = 1.0
    output["impact_weight"] = pd.to_numeric(output["impact_weight"], errors="coerce").fillna(1.0)
    output["impact_weight"] = output["impact_weight"].clip(lower=0.0)
    output["availability_status_severity"] = output["availability_status"].map(STATUS_SEVERITY).fillna(0.25)
    output = output.dropna(subset=["report_date", "game_date"])
    output = output[
        output["team_abbr"].astype(str).str.strip().ne("")
        & output["player_name"].astype(str).str.strip().ne("")
    ].copy()
    return output.sort_values(["game_date", "team_abbr", "player_name", "report_date"]).reset_index(drop=True)


def enrich_availability_reports_with_player_impact(
    reports: pd.DataFrame,
    player_logs: pd.DataFrame,
    lookback_games: int = 10,
) -> pd.DataFrame:
    """Add player impact weights to availability rows from prior player game logs.

    The enrichment is leak-safe: for each availability row, only games before that
    availability row's game date are used. If a report already has a usable
    `impact_weight`, it is preserved.
    """

    if reports.empty or player_logs.empty:
        return reports.copy()

    output = reports.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output["team_abbr"] = output["team_abbr"].map(normalize_team_abbr)
    output["player_name"] = output["player_name"].fillna("").astype(str).str.strip()
    output["_player_key"] = output["player_name"].map(_normalize_player_name)

    try:
        from features.player_features import normalize_player_logs

        normalized_logs = normalize_player_logs(player_logs)
    except Exception:
        normalized_logs = pd.DataFrame()
    if normalized_logs.empty:
        output = output.drop(columns=["_player_key"])
        return output

    logs = normalized_logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs["team_abbr"] = logs["team_abbr"].map(normalize_team_abbr)
    logs["_player_key"] = logs["player_name"].map(_normalize_player_name)
    logs = logs.dropna(subset=["game_date"])
    logs = logs[logs["_player_key"].astype(str).str.strip().ne("")]
    logs = logs.sort_values(["team_abbr", "_player_key", "game_date", "game_id"])
    grouped_logs = {
        key: frame.copy()
        for key, frame in logs.groupby(["team_abbr", "_player_key"], sort=False)
    }

    existing_impact = (
        pd.to_numeric(output["impact_weight"], errors="coerce")
        if "impact_weight" in output.columns
        else pd.Series(pd.NA, index=output.index, dtype="Float64")
    )
    impact_weights: list[float] = []
    impact_sources: list[str] = []
    avg_minutes: list[float] = []
    avg_values: list[float] = []
    games_found: list[int] = []

    for index, row in output.iterrows():
        current_weight = existing_impact.loc[index]
        if pd.notna(current_weight) and float(current_weight) > 0:
            weight = float(current_weight)
            source = "provided_impact_weight"
            recent = pd.DataFrame()
        else:
            game_date = row.get("game_date")
            key = (row.get("team_abbr"), row.get("_player_key"))
            player_history = grouped_logs.get(key, pd.DataFrame())
            if pd.isna(game_date) or player_history.empty:
                recent = pd.DataFrame()
            else:
                recent = player_history[player_history["game_date"] < pd.Timestamp(game_date)].tail(lookback_games)
            if recent.empty:
                weight = 1.0
                source = "fallback_unweighted"
            else:
                weight = float(pd.to_numeric(recent["minutes"], errors="coerce").fillna(0).mean())
                source = f"player_logs_last_{lookback_games}_games_minutes"

        impact_weights.append(weight)
        impact_sources.append(source)
        if recent.empty:
            avg_minutes.append(float("nan"))
            avg_values.append(float("nan"))
            games_found.append(0)
        else:
            avg_minutes.append(float(pd.to_numeric(recent["minutes"], errors="coerce").fillna(0).mean()))
            avg_values.append(float(pd.to_numeric(recent["box_score_value"], errors="coerce").fillna(0).mean()))
            games_found.append(int(len(recent)))

    output["impact_weight"] = impact_weights
    output["impact_weight_source"] = impact_sources
    output["impact_prior_games"] = games_found
    output["impact_avg_minutes_last10"] = avg_minutes
    output["impact_avg_box_score_value_last10"] = avg_values
    return output.drop(columns=["_player_key"])


def build_team_availability_features(games: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Return one availability feature row per team-game using pregame reports only."""

    if reports.empty:
        return pd.DataFrame(columns=["game_id", "team_abbr", *AVAILABILITY_BASE_FEATURE_COLUMNS])
    require_columns(
        games,
        ["game_id", "game_date", "home_team_abbr", "away_team_abbr"],
        dataframe_name="games",
    )
    normalized = normalize_availability_reports(reports)
    if normalized.empty:
        return pd.DataFrame(columns=["game_id", "team_abbr", *AVAILABILITY_BASE_FEATURE_COLUMNS])

    games_copy = games[["game_id", "game_date", "home_team_abbr", "away_team_abbr"]].copy()
    games_copy["game_date"] = pd.to_datetime(games_copy["game_date"], errors="coerce")
    team_games = pd.concat(
        [
            games_copy.rename(columns={"home_team_abbr": "team_abbr"})[["game_id", "game_date", "team_abbr"]],
            games_copy.rename(columns={"away_team_abbr": "team_abbr"})[["game_id", "game_date", "team_abbr"]],
        ],
        ignore_index=True,
    )
    team_games["team_abbr"] = team_games["team_abbr"].map(normalize_team_abbr)
    rows = []
    for team_game in team_games.itertuples(index=False):
        eligible = normalized[
            (normalized["game_date"].dt.normalize() == pd.Timestamp(team_game.game_date).normalize())
            & (normalized["team_abbr"] == team_game.team_abbr)
            & (normalized["report_date"].dt.normalize() <= pd.Timestamp(team_game.game_date).normalize())
        ].copy()
        if eligible.empty:
            feature_values = {column: 0 for column in AVAILABILITY_BASE_FEATURE_COLUMNS}
            rows.append({"game_id": team_game.game_id, "team_abbr": team_game.team_abbr, **feature_values})
            continue
        latest = eligible.sort_values("report_date").drop_duplicates(["player_name"], keep="last")
        counts = latest["availability_status"].value_counts().to_dict()
        out_count = int(counts.get("out", 0))
        doubtful_count = int(counts.get("doubtful", 0))
        questionable_count = int(counts.get("questionable", 0))
        status_weighted = latest.groupby("availability_status")["impact_weight"].sum().to_dict()
        out_weighted = float(status_weighted.get("out", 0.0))
        doubtful_weighted = float(status_weighted.get("doubtful", 0.0))
        questionable_weighted = float(status_weighted.get("questionable", 0.0))
        severity_weighted = float((latest["availability_status_severity"] * latest["impact_weight"]).sum())
        rows.append(
            {
                "game_id": team_game.game_id,
                "team_abbr": team_game.team_abbr,
                "availability_report_present": 1,
                "availability_reported_players": int(len(latest)),
                "availability_players_out": out_count,
                "availability_players_doubtful": doubtful_count,
                "availability_players_questionable": questionable_count,
                "availability_players_probable": int(counts.get("probable", 0)),
                "availability_players_available": int(counts.get("available", 0)),
                "availability_players_unknown": int(counts.get("unknown", 0)),
                "availability_out_or_doubtful": out_count + doubtful_count,
                "availability_questionable_or_worse": out_count + doubtful_count + questionable_count,
                "availability_out_weighted": out_weighted,
                "availability_doubtful_weighted": doubtful_weighted,
                "availability_questionable_weighted": questionable_weighted,
                "availability_questionable_or_worse_weighted": out_weighted + doubtful_weighted + questionable_weighted,
                "availability_status_severity_weighted": severity_weighted,
                "availability_projected_minutes_lost": severity_weighted,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def build_game_availability_features(games: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Return home/away availability features and differences for each game."""

    team_features = build_team_availability_features(games, reports)
    if team_features.empty:
        return pd.DataFrame(columns=["game_id"])

    games_key = games[["game_id", "home_team_abbr", "away_team_abbr"]].copy()
    games_key["home_team_abbr"] = games_key["home_team_abbr"].map(normalize_team_abbr)
    games_key["away_team_abbr"] = games_key["away_team_abbr"].map(normalize_team_abbr)
    home = team_features.rename(columns={column: f"home_{column}" for column in AVAILABILITY_BASE_FEATURE_COLUMNS})
    home = home.rename(columns={"team_abbr": "home_team_abbr"})
    away = team_features.rename(columns={column: f"away_{column}" for column in AVAILABILITY_BASE_FEATURE_COLUMNS})
    away = away.rename(columns={"team_abbr": "away_team_abbr"})
    output = games_key.merge(home, on=["game_id", "home_team_abbr"], how="left")
    output = output.merge(away, on=["game_id", "away_team_abbr"], how="left")
    for prefix in ["home", "away"]:
        for column in AVAILABILITY_BASE_FEATURE_COLUMNS:
            feature = f"{prefix}_{column}"
            if feature in output.columns:
                output[feature] = pd.to_numeric(output[feature], errors="coerce").fillna(0)
    for column in AVAILABILITY_BASE_FEATURE_COLUMNS:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in output.columns and away_column in output.columns:
            output[f"{column}_diff"] = output[home_column] - output[away_column]
    return output.drop(columns=["home_team_abbr", "away_team_abbr"]).reset_index(drop=True)
