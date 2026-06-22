"""Elo-based team-strength features for the no-odds matchup pipeline.

Works on the canonical results schema produced by
:mod:`data.match_results_loader` (``team_a``/``team_b`` with home flags,
``neutral_site``, scores, and ``result_*`` columns).

Key properties:
* Pre-game ratings are stored *before* each game is used to update them, so the
  features never leak the result of the game they describe.
* Supports a home-court/-field advantage that is disabled at neutral sites.
* Supports draws (needed for soccer) and an optional margin-of-victory boost.
* Ratings are tracked per ``(sport, team)`` so national teams carry one rating
  across friendlies, qualifiers, and tournaments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from data.validation import require_columns


@dataclass(frozen=True)
class EloConfig:
    """Tunable Elo parameters (sensible defaults for most sports)."""

    starting_elo: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    use_margin_of_victory: bool = True

    @classmethod
    def from_dict(cls, config: dict | None) -> "EloConfig":
        config = config or {}
        return cls(
            starting_elo=float(config.get("starting_elo", 1500.0)),
            k_factor=float(config.get("k_factor", 20.0)),
            home_advantage=float(config.get("home_advantage", 60.0)),
            use_margin_of_victory=bool(config.get("use_margin_of_victory", True)),
        )


_ELO_REQUIRED_COLUMNS = [
    "game_id",
    "sport",
    "game_date",
    "team_a",
    "team_b",
    "team_a_home_flag",
    "team_b_home_flag",
    "neutral_site",
    "result_team_a_win",
    "result_draw",
    "result_team_b_win",
]


def expected_score(elo_a: float, elo_b: float) -> float:
    """Return the Elo-expected score for team A (0..1) given adjusted ratings."""

    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _mov_multiplier(point_margin: float, elo_diff_winner: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier.

    ``point_margin`` is the absolute final margin; ``elo_diff_winner`` is the
    winner's pre-game rating minus the loser's. Returns ~1.0 for a one-point
    win between equal teams and grows with the margin while autocorrelation is
    damped for big favorites.
    """

    if point_margin <= 0:
        return 1.0
    return math.log(point_margin + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def calculate_elo_ratings(
    results_df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Return ``results_df`` with pre- and post-game Elo columns added.

    Added columns: ``team_a_elo_pre``, ``team_b_elo_pre``, ``elo_diff_pre``,
    ``elo_expected_a`` (pre-game), ``team_a_elo_post``, ``team_b_elo_post``.
    """

    settings = EloConfig.from_dict(config)
    if results_df.empty:
        empty = results_df.copy()
        for col in (
            "team_a_elo_pre",
            "team_b_elo_pre",
            "elo_diff_pre",
            "elo_expected_a",
            "team_a_elo_post",
            "team_b_elo_post",
        ):
            empty[col] = pd.Series(dtype="float64")
        return empty

    require_columns(results_df, _ELO_REQUIRED_COLUMNS, dataframe_name="results_df")

    frame = results_df.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame = frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings: dict[tuple[str, str], float] = {}
    rows: list[dict[str, float]] = []

    for _, game in frame.iterrows():
        sport = str(game["sport"])
        key_a = (sport, str(game["team_a"]))
        key_b = (sport, str(game["team_b"]))
        elo_a = ratings.get(key_a, settings.starting_elo)
        elo_b = ratings.get(key_b, settings.starting_elo)

        neutral = bool(int(game.get("neutral_site", 0)))
        adj_a = elo_a
        adj_b = elo_b
        if not neutral:
            if int(game.get("team_a_home_flag", 0)) == 1:
                adj_a += settings.home_advantage
            elif int(game.get("team_b_home_flag", 0)) == 1:
                adj_b += settings.home_advantage

        expected_a = expected_score(adj_a, adj_b)

        if int(game.get("result_team_a_win", 0)) == 1:
            actual_a = 1.0
        elif int(game.get("result_draw", 0)) == 1:
            actual_a = 0.5
        elif int(game.get("result_team_b_win", 0)) == 1:
            actual_a = 0.0
        else:
            actual_a = None  # undetermined outcome -> no rating update

        rows.append(
            {
                "team_a_elo_pre": elo_a,
                "team_b_elo_pre": elo_b,
                "elo_diff_pre": elo_a - elo_b,
                "elo_expected_a": expected_a,
            }
        )

        if actual_a is None:
            rows[-1]["team_a_elo_post"] = elo_a
            rows[-1]["team_b_elo_post"] = elo_b
            continue

        multiplier = 1.0
        if settings.use_margin_of_victory:
            a_score = game.get("team_a_score")
            b_score = game.get("team_b_score")
            if pd.notna(a_score) and pd.notna(b_score):
                margin = abs(float(a_score) - float(b_score))
                # Winner's pre-game adjusted-rating edge (>=0 keeps the curve sane).
                if actual_a >= 1.0:
                    elo_diff_winner = adj_a - adj_b
                elif actual_a <= 0.0:
                    elo_diff_winner = adj_b - adj_a
                else:
                    elo_diff_winner = 0.0
                multiplier = _mov_multiplier(margin, max(elo_diff_winner, -100.0))

        change = settings.k_factor * multiplier * (actual_a - expected_a)
        elo_a_post = elo_a + change
        elo_b_post = elo_b - change
        ratings[key_a] = elo_a_post
        ratings[key_b] = elo_b_post
        rows[-1]["team_a_elo_post"] = elo_a_post
        rows[-1]["team_b_elo_post"] = elo_b_post

    elo_frame = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame, elo_frame], axis=1)


def add_pre_game_elo_features(
    results_df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Add model-facing pre-game Elo features: ``team_a_elo``, ``team_b_elo``,
    ``elo_diff`` (plus the underlying pre/post columns)."""

    enriched = calculate_elo_ratings(results_df, config)
    enriched["team_a_elo"] = enriched["team_a_elo_pre"]
    enriched["team_b_elo"] = enriched["team_b_elo_pre"]
    enriched["elo_diff"] = enriched["team_a_elo"] - enriched["team_b_elo"]
    return enriched


def get_latest_team_strength(elo_df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent post-game Elo rating for each ``(sport, team)``.

    Expects a frame produced by :func:`calculate_elo_ratings` /
    :func:`add_pre_game_elo_features`. Output columns: ``sport``, ``team``,
    ``elo``, ``last_game_date``, ``games_played``.
    """

    columns = ["sport", "team", "elo", "last_game_date", "games_played"]
    if elo_df.empty:
        return pd.DataFrame(columns=columns)

    require_columns(
        elo_df,
        ["sport", "game_date", "team_a", "team_b", "team_a_elo_post", "team_b_elo_post"],
        dataframe_name="elo_df",
    )

    side_a = elo_df[["sport", "game_date", "team_a", "team_a_elo_post"]].rename(
        columns={"team_a": "team", "team_a_elo_post": "elo"}
    )
    side_b = elo_df[["sport", "game_date", "team_b", "team_b_elo_post"]].rename(
        columns={"team_b": "team", "team_b_elo_post": "elo"}
    )
    stacked = pd.concat([side_a, side_b], ignore_index=True)
    stacked["game_date"] = pd.to_datetime(stacked["game_date"], errors="coerce")
    stacked = stacked.sort_values("game_date")

    grouped = stacked.groupby(["sport", "team"], dropna=False)
    latest = grouped.agg(
        elo=("elo", "last"),
        last_game_date=("game_date", "max"),
        games_played=("elo", "count"),
    ).reset_index()
    return latest[columns]
