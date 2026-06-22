"""Build model-ready matchup features for historical games and future fixtures.

Two public entry points:

* :func:`build_training_features` – one row per *completed* game, with every
  feature computed only from games that finished **before** that game (no
  leakage), plus the three result columns used as the training target.
* :func:`build_fixture_features` – one row per *future* fixture, using the
  terminal state of all completed games before the fixture date.

Both produce the same feature columns (see :data:`NUMERIC_FEATURES` and
:data:`CATEGORICAL_FEATURES`) so a model trained on one scores the other.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from data.injuries_loader import summarize_team_availability
from data.validation import require_columns
from features.team_strength import (
    add_pre_game_elo_features,
    get_latest_team_strength,
)

logger = logging.getLogger(__name__)


# --- feature contract -----------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "team_a_elo",
    "team_b_elo",
    "elo_diff",
    "team_a_recent_win_rate_5",
    "team_b_recent_win_rate_5",
    "team_a_recent_win_rate_10",
    "team_b_recent_win_rate_10",
    "recent_win_rate_diff_5",
    "recent_win_rate_diff_10",
    "team_a_recent_score_for",
    "team_b_recent_score_for",
    "team_a_recent_score_against",
    "team_b_recent_score_against",
    "recent_score_diff",
    "team_a_rest_days",
    "team_b_rest_days",
    "rest_diff",
    "team_a_home_flag",
    "team_b_home_flag",
    "neutral_site",
    "team_a_injury_impact",
    "team_b_injury_impact",
    "injury_impact_diff",
    "team_a_key_players_out",
    "team_b_key_players_out",
    "team_a_games_last_14_days",
    "team_b_games_last_14_days",
    "schedule_congestion_diff",
]

CATEGORICAL_FEATURES: list[str] = ["competition_type"]

# Extra context columns carried alongside the features for the explainer and
# the data-quality gates (not fed to the model directly).
CONTEXT_COLUMNS: list[str] = [
    "team_a_recent_games",
    "team_b_recent_games",
    "min_recent_games",
    "team_a_injury_stale",
    "team_b_injury_stale",
    "injury_data_present",
    "team_a_availability_present",
    "team_b_availability_present",
    "team_a_availability_source",
    "team_b_availability_source",
    "team_a_availability_manual",
    "team_b_availability_manual",
    "team_a_availability_last_updated",
    "team_b_availability_last_updated",
    "team_a_availability_notes",
    "team_b_availability_notes",
]

_BASE_RESULT_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "team_a",
    "team_b",
    "team_a_home_flag",
    "team_b_home_flag",
    "neutral_site",
    "competition_type",
    "result_team_a_win",
    "result_draw",
    "result_team_b_win",
]

_NEUTRAL_WIN_RATE = 0.5
_DEFAULT_REST_DAYS = 7.0
_WINDOW_DAYS = 14


# --- shared helpers -------------------------------------------------------
def _team_long(results_df: pd.DataFrame) -> pd.DataFrame:
    """Explode each game into two team-centric rows (one per side)."""

    base = results_df.copy()
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")
    a_win = base["result_team_a_win"].astype(float)
    b_win = base["result_team_b_win"].astype(float)
    draw = base["result_draw"].astype(float)

    side_a = pd.DataFrame(
        {
            "game_id": base["game_id"].values,
            "sport": base["sport"].values,
            "game_date": base["game_date"].values,
            "side": "a",
            "team": base["team_a"].values,
            "team_score": pd.to_numeric(base.get("team_a_score"), errors="coerce").values,
            "opp_score": pd.to_numeric(base.get("team_b_score"), errors="coerce").values,
            "win_value": (a_win + 0.5 * draw).values,
        }
    )
    side_b = pd.DataFrame(
        {
            "game_id": base["game_id"].values,
            "sport": base["sport"].values,
            "game_date": base["game_date"].values,
            "side": "b",
            "team": base["team_b"].values,
            "team_score": pd.to_numeric(base.get("team_b_score"), errors="coerce").values,
            "opp_score": pd.to_numeric(base.get("team_a_score"), errors="coerce").values,
            "win_value": (b_win + 0.5 * draw).values,
        }
    )
    return pd.concat([side_a, side_b], ignore_index=True)


def _trailing_game_count(dates_ns: np.ndarray, window_days: int = _WINDOW_DAYS) -> np.ndarray:
    """For each game (sorted ascending) count prior games within ``window_days``."""

    days = dates_ns // (10**9 * 86400)
    counts = np.zeros(len(days), dtype=int)
    for i in range(len(days)):
        lo = np.searchsorted(days[:i], days[i] - window_days, side="left")
        counts[i] = i - lo
    return counts


def _add_shifted_form(long_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add leakage-safe rolling form per team (each row uses only prior games)."""

    config = config or {}
    win5 = int(config.get("form_window_short", 5))
    win10 = int(config.get("form_window_long", 10))

    out = long_df.sort_values(["sport", "team", "game_date", "game_id"]).reset_index(drop=True)
    grouped = out.groupby(["sport", "team"], sort=False)

    # The first shifted window of each team is all-NaN by design (no prior
    # games); pandas may emit a benign "Mean of empty slice" warning we fill in.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["recent_win_rate_5"] = grouped["win_value"].transform(
            lambda s: s.shift().rolling(win5, min_periods=1).mean()
        )
        out["recent_win_rate_10"] = grouped["win_value"].transform(
            lambda s: s.shift().rolling(win10, min_periods=1).mean()
        )
        out["recent_score_for"] = grouped["team_score"].transform(
            lambda s: s.shift().rolling(win5, min_periods=1).mean()
        )
        out["recent_score_against"] = grouped["opp_score"].transform(
            lambda s: s.shift().rolling(win5, min_periods=1).mean()
        )
    out["rest_days"] = grouped["game_date"].diff().dt.days
    out["recent_games"] = grouped.cumcount()

    counts = np.zeros(len(out), dtype=int)
    for _, idx in grouped.groups.items():
        rows = out.loc[idx].sort_values(["game_date", "game_id"])
        dates_ns = rows["game_date"].astype("int64").to_numpy()
        counts[rows.index.to_numpy()] = _trailing_game_count(dates_ns)
    out["games_last_14_days"] = counts
    return out


def _fill_form_defaults(df: pd.DataFrame, score_cols: list[str], sport_medians: dict) -> None:
    """In-place neutral fills so early-career rows don't break the model."""

    for col in ("team_a_recent_win_rate_5", "team_b_recent_win_rate_5",
                "team_a_recent_win_rate_10", "team_b_recent_win_rate_10"):
        if col in df:
            df[col] = df[col].fillna(_NEUTRAL_WIN_RATE)
    for col in ("team_a_rest_days", "team_b_rest_days"):
        if col in df:
            df[col] = df[col].fillna(_DEFAULT_REST_DAYS)
    # Robust fill for scores: per-column median, then a global score median,
    # then 0.0 – so a tiny dataset where a whole column is NaN never leaks NaN.
    present_scores = [c for c in score_cols if c in df]
    global_score = pd.concat([df[c] for c in present_scores]).median() if present_scores else float("nan")
    if pd.isna(global_score):
        global_score = 0.0
    for col in present_scores:
        df[col] = df[col].fillna(sport_medians.get(col)).fillna(global_score)


def _empty_injury_features(teams: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": teams.values,
            "team_injury_impact": 0.0,
            "key_players_out": 0,
            "injury_stale": False,
        }
    )


# --- training features ----------------------------------------------------
def build_training_features(
    results_df: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Return one feature row per completed game (no future leakage).

    Historical injury snapshots are rarely available per game, so by default
    training injury features are zero. They are only populated when
    ``config['use_training_injuries']`` is set and ``injuries_df`` carries
    timestamps; otherwise we proceed without them (a warning is logged).
    """

    config = config or {}
    if results_df.empty:
        return pd.DataFrame(columns=_output_columns("game_id"))

    require_columns(results_df, _BASE_RESULT_COLUMNS, dataframe_name="results_df")

    base = results_df.copy()
    base["game_date"] = pd.to_datetime(base["game_date"], errors="coerce")

    # 1) Elo (pre-game).
    elo = add_pre_game_elo_features(base, config.get("elo"))
    elo_cols = elo[["game_id", "team_a_elo", "team_b_elo", "elo_diff"]]

    # 2) Rolling form (leakage-safe).
    long_form = _add_shifted_form(_team_long(base), config)
    wide_form = _pivot_form(long_form)

    out = base[_BASE_RESULT_COLUMNS].merge(elo_cols, on="game_id", how="left")
    out = out.merge(wide_form, on="game_id", how="left")

    # 3) Injuries (training: zeros unless explicitly enabled and timestamped).
    if config.get("use_training_injuries") and injuries_df is not None and not injuries_df.empty:
        out = _attach_training_injuries(out, injuries_df)
    else:
        if injuries_df is not None and not injuries_df.empty:
            logger.info("Injuries provided but training injuries disabled; using zeros.")
        out = _zero_injuries(out)

    out = _finalize(out, base)
    out = out.rename(columns={"game_id": "game_id"})
    return out[_output_columns("game_id")]


def _pivot_form(long_form: pd.DataFrame) -> pd.DataFrame:
    base_features = [
        "recent_win_rate_5",
        "recent_win_rate_10",
        "recent_score_for",
        "recent_score_against",
        "rest_days",
        "games_last_14_days",
        "recent_games",
    ]
    a = long_form[long_form["side"] == "a"].set_index("game_id")[base_features]
    b = long_form[long_form["side"] == "b"].set_index("game_id")[base_features]
    a = a.add_prefix("team_a_")
    b = b.add_prefix("team_b_")
    return a.join(b, how="outer").reset_index()


def _attach_training_injuries(out: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    """As-of injury merge per game date (only used when explicitly enabled)."""

    impacts_a = []
    impacts_b = []
    key_a = []
    key_b = []
    stale_a = []
    stale_b = []
    cache: dict[pd.Timestamp, pd.DataFrame] = {}
    for _, row in out.iterrows():
        date = row["game_date"]
        if date not in cache:
            cache[date] = summarize_team_availability(injuries_df, date)
        summary = cache[date].set_index("team")
        impacts_a.append(_lookup(summary, row["team_a"], "team_injury_impact"))
        impacts_b.append(_lookup(summary, row["team_b"], "team_injury_impact"))
        key_a.append(_lookup(summary, row["team_a"], "key_players_out"))
        key_b.append(_lookup(summary, row["team_b"], "key_players_out"))
        stale_a.append(bool(_lookup(summary, row["team_a"], "injury_data_stale", default=True)))
        stale_b.append(bool(_lookup(summary, row["team_b"], "injury_data_stale", default=True)))
    out["team_a_injury_impact"] = impacts_a
    out["team_b_injury_impact"] = impacts_b
    out["team_a_key_players_out"] = key_a
    out["team_b_key_players_out"] = key_b
    out["team_a_injury_stale"] = stale_a
    out["team_b_injury_stale"] = stale_b
    out["injury_data_present"] = 1
    out["team_a_availability_present"] = 1
    out["team_b_availability_present"] = 1
    out["team_a_availability_source"] = ""
    out["team_b_availability_source"] = ""
    out["team_a_availability_manual"] = False
    out["team_b_availability_manual"] = False
    out["team_a_availability_last_updated"] = ""
    out["team_b_availability_last_updated"] = ""
    out["team_a_availability_notes"] = ""
    out["team_b_availability_notes"] = ""
    return out


def _lookup(summary: pd.DataFrame, team: str, column: str, default: float = 0.0):
    if team in summary.index:
        value = summary.loc[team, column]
        return default if pd.isna(value) else value
    return default


def _zero_injuries(out: pd.DataFrame) -> pd.DataFrame:
    out["team_a_injury_impact"] = 0.0
    out["team_b_injury_impact"] = 0.0
    out["team_a_key_players_out"] = 0
    out["team_b_key_players_out"] = 0
    out["team_a_injury_stale"] = False
    out["team_b_injury_stale"] = False
    out["injury_data_present"] = 0
    out["team_a_availability_present"] = 0
    out["team_b_availability_present"] = 0
    out["team_a_availability_source"] = ""
    out["team_b_availability_source"] = ""
    out["team_a_availability_manual"] = False
    out["team_b_availability_manual"] = False
    out["team_a_availability_last_updated"] = ""
    out["team_b_availability_last_updated"] = ""
    out["team_a_availability_notes"] = ""
    out["team_b_availability_notes"] = ""
    return out


# --- fixture features -----------------------------------------------------
def build_fixture_features(
    fixtures_df: pd.DataFrame,
    results_df: pd.DataFrame,
    injuries_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Return one feature row per fixture using all completed games before it."""

    config = config or {}
    if fixtures_df.empty:
        return pd.DataFrame(columns=_output_columns("fixture_id"))

    require_columns(
        fixtures_df,
        [
            "fixture_id",
            "sport",
            "league",
            "game_date",
            "team_a",
            "team_b",
            "team_a_home_flag",
            "team_b_home_flag",
            "neutral_site",
            "competition_type",
        ],
        dataframe_name="fixtures_df",
    )

    fixtures = fixtures_df.copy()
    fixtures["game_date"] = pd.to_datetime(fixtures["game_date"], errors="coerce")

    # Terminal team state from completed results.
    elo = add_pre_game_elo_features(results_df, config.get("elo")) if not results_df.empty else results_df
    strength = get_latest_team_strength(elo) if not results_df.empty else pd.DataFrame(
        columns=["sport", "team", "elo", "last_game_date", "games_played"]
    )
    state, team_dates = _compute_team_state(results_df, config)

    out = fixtures.copy()
    out["team_a_elo"] = _merge_state(out, strength.set_index(["sport", "team"]), "team_a", "elo", 1500.0)
    out["team_b_elo"] = _merge_state(out, strength.set_index(["sport", "team"]), "team_b", "elo", 1500.0)
    out["elo_diff"] = out["team_a_elo"] - out["team_b_elo"]

    state_idx = state.set_index(["sport", "team"]) if not state.empty else state
    for side in ("team_a", "team_b"):
        out[f"{side}_recent_win_rate_5"] = _merge_state(out, state_idx, side, "recent_win_rate_5", _NEUTRAL_WIN_RATE)
        out[f"{side}_recent_win_rate_10"] = _merge_state(out, state_idx, side, "recent_win_rate_10", _NEUTRAL_WIN_RATE)
        out[f"{side}_recent_score_for"] = _merge_state(out, state_idx, side, "recent_score_for", np.nan)
        out[f"{side}_recent_score_against"] = _merge_state(out, state_idx, side, "recent_score_against", np.nan)
        out[f"{side}_recent_games"] = _merge_state(out, state_idx, side, "n_games", 0).fillna(0).astype(int)

    # Rest + schedule congestion relative to the fixture date.
    out["team_a_rest_days"] = _rest_days(out, state_idx, "team_a")
    out["team_b_rest_days"] = _rest_days(out, state_idx, "team_b")
    out["team_a_games_last_14_days"] = _games_last_14(out, team_dates, "team_a")
    out["team_b_games_last_14_days"] = _games_last_14(out, team_dates, "team_b")

    # Injuries as of each fixture date.
    out = _attach_fixture_injuries(out, injuries_df)

    out = _finalize(out, fixtures, id_col="fixture_id")
    return out[_output_columns("fixture_id")]


def _compute_team_state(results_df: pd.DataFrame, config: dict | None):
    """Terminal rolling-form state per (sport, team) + sorted game dates."""

    config = config or {}
    win5 = int(config.get("form_window_short", 5))
    win10 = int(config.get("form_window_long", 10))
    empty = pd.DataFrame(
        columns=[
            "sport", "team", "recent_win_rate_5", "recent_win_rate_10",
            "recent_score_for", "recent_score_against", "n_games", "last_game_date",
        ]
    )
    if results_df.empty:
        return empty, {}

    long_df = _team_long(results_df).sort_values(["sport", "team", "game_date", "game_id"])
    records = []
    team_dates: dict[tuple, np.ndarray] = {}
    for (sport, team), group in long_df.groupby(["sport", "team"], sort=False):
        wins = group["win_value"].dropna()
        scores_for = group["team_score"].dropna().tail(win5)
        scores_against = group["opp_score"].dropna().tail(win5)
        records.append(
            {
                "sport": sport,
                "team": team,
                "recent_win_rate_5": wins.tail(win5).mean() if len(wins) else np.nan,
                "recent_win_rate_10": wins.tail(win10).mean() if len(wins) else np.nan,
                "recent_score_for": scores_for.mean() if len(scores_for) else np.nan,
                "recent_score_against": scores_against.mean() if len(scores_against) else np.nan,
                "n_games": int(len(group)),
                "last_game_date": group["game_date"].max(),
            }
        )
        team_dates[(sport, team)] = np.sort(
            group["game_date"].dropna().astype("int64").to_numpy() // (10**9 * 86400)
        )
    return pd.DataFrame(records), team_dates


def _merge_state(out, state_idx, side, column, default):
    if state_idx is None or len(state_idx) == 0 or column not in getattr(state_idx, "columns", []):
        return pd.Series(default, index=out.index)
    keys = list(zip(out["sport"], out[side]))
    values = [
        state_idx[column].get(key, default) if key in state_idx.index else default
        for key in keys
    ]
    return pd.Series(values, index=out.index)


def _rest_days(out, state_idx, side):
    if state_idx is None or len(state_idx) == 0 or "last_game_date" not in getattr(state_idx, "columns", []):
        return pd.Series(_DEFAULT_REST_DAYS, index=out.index)
    rest = []
    for _, row in out.iterrows():
        key = (row["sport"], row[side])
        if key in state_idx.index:
            last = state_idx.loc[key, "last_game_date"]
            if pd.notna(last) and pd.notna(row["game_date"]):
                # Clamp to >= 0: a "last game" after the fixture means the
                # results file contains games the fixture should not see.
                delta = max(float((row["game_date"] - last).days), 0.0)
            else:
                delta = _DEFAULT_REST_DAYS
            rest.append(delta)
        else:
            rest.append(_DEFAULT_REST_DAYS)
    return pd.Series(rest, index=out.index)


def _games_last_14(out, team_dates, side):
    counts = []
    for _, row in out.iterrows():
        key = (row["sport"], row[side])
        date = row["game_date"]
        if key in team_dates and pd.notna(date):
            day = int(date.value // (10**9 * 86400))
            arr = team_dates[key]
            lo = np.searchsorted(arr, day - _WINDOW_DAYS, side="left")
            hi = np.searchsorted(arr, day, side="left")  # strictly before fixture day
            counts.append(int(hi - lo))
        else:
            counts.append(0)
    return pd.Series(counts, index=out.index)


def _attach_fixture_injuries(out: pd.DataFrame, injuries_df: pd.DataFrame | None) -> pd.DataFrame:
    if injuries_df is None or injuries_df.empty:
        logger.info("No injuries provided for fixtures; injury features default to 0.")
        return _zero_injuries(out)

    summaries: dict[pd.Timestamp, pd.DataFrame] = {}
    for date in out["game_date"].dropna().unique():
        ts = pd.Timestamp(date)
        summaries[ts] = summarize_team_availability(injuries_df, ts).set_index("team")

    def feat(row, side, column, default=0.0):
        summary = summaries.get(pd.Timestamp(row["game_date"]))
        if summary is None:
            return default
        return _lookup(summary, row[side], column, default=default)

    def present(row, side) -> bool:
        summary = summaries.get(pd.Timestamp(row["game_date"]))
        return bool(summary is not None and row[side] in summary.index)

    def text_feat(row, side, column) -> str:
        value = feat(row, side, column, default="")
        return "" if pd.isna(value) else str(value)

    a_present = out.apply(lambda r: present(r, "team_a"), axis=1)
    b_present = out.apply(lambda r: present(r, "team_b"), axis=1)
    out["team_a_injury_impact"] = out.apply(lambda r: feat(r, "team_a", "team_injury_impact"), axis=1)
    out["team_b_injury_impact"] = out.apply(lambda r: feat(r, "team_b", "team_injury_impact"), axis=1)
    out["team_a_key_players_out"] = out.apply(lambda r: feat(r, "team_a", "key_players_out"), axis=1)
    out["team_b_key_players_out"] = out.apply(lambda r: feat(r, "team_b", "key_players_out"), axis=1)
    out["team_a_injury_stale"] = out.apply(
        lambda r: bool(feat(r, "team_a", "injury_data_stale", default=False)) if present(r, "team_a") else False,
        axis=1,
    )
    out["team_b_injury_stale"] = out.apply(
        lambda r: bool(feat(r, "team_b", "injury_data_stale", default=False)) if present(r, "team_b") else False,
        axis=1,
    )
    out["team_a_availability_present"] = a_present.astype(int)
    out["team_b_availability_present"] = b_present.astype(int)
    out["injury_data_present"] = (a_present & b_present).astype(int)
    out["team_a_availability_source"] = out.apply(lambda r: text_feat(r, "team_a", "availability_sources"), axis=1)
    out["team_b_availability_source"] = out.apply(lambda r: text_feat(r, "team_b", "availability_sources"), axis=1)
    out["team_a_availability_manual"] = out.apply(
        lambda r: bool(feat(r, "team_a", "availability_manual", default=False)) if present(r, "team_a") else False,
        axis=1,
    )
    out["team_b_availability_manual"] = out.apply(
        lambda r: bool(feat(r, "team_b", "availability_manual", default=False)) if present(r, "team_b") else False,
        axis=1,
    )
    out["team_a_availability_last_updated"] = out.apply(lambda r: text_feat(r, "team_a", "last_updated"), axis=1)
    out["team_b_availability_last_updated"] = out.apply(lambda r: text_feat(r, "team_b", "last_updated"), axis=1)
    out["team_a_availability_notes"] = out.apply(lambda r: text_feat(r, "team_a", "availability_notes"), axis=1)
    out["team_b_availability_notes"] = out.apply(lambda r: text_feat(r, "team_b", "availability_notes"), axis=1)
    return out


# --- finalization ---------------------------------------------------------
def _finalize(out: pd.DataFrame, base: pd.DataFrame, id_col: str = "game_id") -> pd.DataFrame:
    """Compute diff features, neutral-fill, and add context/quality columns."""

    # Score-context fills use the per-sport median to stay neutral.
    score_cols = [
        "team_a_recent_score_for",
        "team_b_recent_score_for",
        "team_a_recent_score_against",
        "team_b_recent_score_against",
    ]
    sport_medians = {col: out[col].median() for col in score_cols if col in out}
    _fill_form_defaults(out, score_cols, sport_medians)

    out["recent_win_rate_diff_5"] = out["team_a_recent_win_rate_5"] - out["team_b_recent_win_rate_5"]
    out["recent_win_rate_diff_10"] = out["team_a_recent_win_rate_10"] - out["team_b_recent_win_rate_10"]
    out["recent_score_diff"] = (
        (out["team_a_recent_score_for"] - out["team_a_recent_score_against"])
        - (out["team_b_recent_score_for"] - out["team_b_recent_score_against"])
    )
    out["rest_diff"] = out["team_a_rest_days"] - out["team_b_rest_days"]
    out["injury_impact_diff"] = out["team_a_injury_impact"] - out["team_b_injury_impact"]
    out["schedule_congestion_diff"] = (
        out["team_a_games_last_14_days"] - out["team_b_games_last_14_days"]
    )

    for col in ("team_a_recent_games", "team_b_recent_games"):
        if col not in out:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)
    out["min_recent_games"] = out[["team_a_recent_games", "team_b_recent_games"]].min(axis=1)

    # Ensure flags are numeric.
    for col in ("team_a_home_flag", "team_b_home_flag", "neutral_site"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["team_a_key_players_out"] = pd.to_numeric(out["team_a_key_players_out"], errors="coerce").fillna(0).astype(int)
    out["team_b_key_players_out"] = pd.to_numeric(out["team_b_key_players_out"], errors="coerce").fillna(0).astype(int)

    # Final safety net: no numeric feature should ever be NaN downstream.
    present_numeric = [c for c in NUMERIC_FEATURES if c in out.columns]
    out[present_numeric] = out[present_numeric].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).fillna(0.0)
    return out


def _output_columns(id_col: str) -> list[str]:
    identity = [id_col, "sport", "league", "game_date", "team_a", "team_b", "competition_type"]
    targets = ["result_team_a_win", "result_draw", "result_team_b_win"]
    # Targets only exist for training rows; build_fixture_features won't have them.
    cols = identity + NUMERIC_FEATURES + CATEGORICAL_FEATURES + CONTEXT_COLUMNS
    if id_col == "game_id":
        cols = cols + targets
    # Deduplicate while preserving order (competition_type appears once).
    seen: set[str] = set()
    ordered = []
    for col in cols:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered
