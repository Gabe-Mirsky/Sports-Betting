"""Plain-language catalog of every variable the prediction model considers.

Each entry maps a real model feature (see ``src/features/matchup_features.py``)
to a friendly name and a 5th-grade explanation. ``code`` matches the feature name
used by the model, so the coefficient export can be joined to these descriptions.
"""

from __future__ import annotations

# Groups keep the "How it works" page organised and skimmable.
FEATURE_GROUPS = [
    {
        "group": "Team strength",
        "blurb": "How good each team is overall.",
        "features": [
            {
                "code": "team_a_elo",
                "name": "Team A strength score",
                "desc": "A single number for how strong Team A is, built from its whole history. Bigger means stronger.",
            },
            {
                "code": "team_b_elo",
                "name": "Team B strength score",
                "desc": "The same strength number, but for Team B.",
            },
            {
                "code": "elo_diff",
                "name": "Strength gap",
                "desc": "How much stronger one team is than the other (Team A's score minus Team B's).",
            },
        ],
    },
    {
        "group": "Recent form (how they've been playing lately)",
        "blurb": "Hot or cold streaks in the last few games.",
        "features": [
            {"code": "team_a_recent_win_rate_5", "name": "Team A wins in last 5 games",
             "desc": "How often Team A has won in its last 5 games."},
            {"code": "team_b_recent_win_rate_5", "name": "Team B wins in last 5 games",
             "desc": "How often Team B has won in its last 5 games."},
            {"code": "team_a_recent_win_rate_10", "name": "Team A wins in last 10 games",
             "desc": "How often Team A has won in its last 10 games."},
            {"code": "team_b_recent_win_rate_10", "name": "Team B wins in last 10 games",
             "desc": "How often Team B has won in its last 10 games."},
            {"code": "recent_win_rate_diff_5", "name": "Who's winning more (last 5)",
             "desc": "The difference in recent winning between the two teams over 5 games."},
            {"code": "recent_win_rate_diff_10", "name": "Who's winning more (last 10)",
             "desc": "The difference in recent winning between the two teams over 10 games."},
        ],
    },
    {
        "group": "Scoring",
        "blurb": "How many points each team scores and allows.",
        "features": [
            {"code": "team_a_recent_score_for", "name": "Points Team A usually scores",
             "desc": "The average points Team A has been scoring recently."},
            {"code": "team_b_recent_score_for", "name": "Points Team B usually scores",
             "desc": "The average points Team B has been scoring recently."},
            {"code": "team_a_recent_score_against", "name": "Points Team A usually gives up",
             "desc": "The average points scored against Team A recently."},
            {"code": "team_b_recent_score_against", "name": "Points Team B usually gives up",
             "desc": "The average points scored against Team B recently."},
            {"code": "recent_score_diff", "name": "Scoring edge",
             "desc": "Who outscores their opponents more: each team's points scored minus points allowed, compared."},
        ],
    },
    {
        "group": "Rest and schedule",
        "blurb": "How tired or fresh each team is.",
        "features": [
            {"code": "team_a_rest_days", "name": "Days of rest for Team A",
             "desc": "How many days since Team A's last game. More rest can help."},
            {"code": "team_b_rest_days", "name": "Days of rest for Team B",
             "desc": "How many days since Team B's last game."},
            {"code": "rest_diff", "name": "Rest advantage",
             "desc": "Which team had more rest, and by how much."},
            {"code": "team_a_games_last_14_days", "name": "Team A games in last 2 weeks",
             "desc": "How many games Team A played in the last 14 days. More games can mean more tired."},
            {"code": "team_b_games_last_14_days", "name": "Team B games in last 2 weeks",
             "desc": "How many games Team B played in the last 14 days."},
            {"code": "schedule_congestion_diff", "name": "Who's more tired",
             "desc": "The difference in how busy each team's schedule has been."},
        ],
    },
    {
        "group": "Where the game is played",
        "blurb": "Home, away, or a neutral field.",
        "features": [
            {"code": "team_a_home_flag", "name": "Team A playing at home?",
             "desc": "Yes (1) if Team A is the home team. Home teams often do a little better."},
            {"code": "team_b_home_flag", "name": "Team B playing at home?",
             "desc": "Yes (1) if Team B is the home team."},
            {"code": "neutral_site", "name": "Neutral field?",
             "desc": "Yes (1) if neither team is really at home, like a tournament game."},
        ],
    },
    {
        "group": "Injuries and missing players",
        "blurb": "How much missing players hurt each team.",
        "features": [
            {"code": "team_a_injury_impact", "name": "How much injuries hurt Team A",
             "desc": "A score for how much Team A is hurt by players who are injured or out."},
            {"code": "team_b_injury_impact", "name": "How much injuries hurt Team B",
             "desc": "The same injury score, for Team B."},
            {"code": "injury_impact_diff", "name": "Injury advantage",
             "desc": "Which team is hurt more by injuries, and by how much."},
            {"code": "team_a_key_players_out", "name": "Star players missing for Team A",
             "desc": "How many of Team A's important players are not playing."},
            {"code": "team_b_key_players_out", "name": "Star players missing for Team B",
             "desc": "How many of Team B's important players are not playing."},
        ],
    },
    {
        "group": "Kind of game",
        "blurb": "What type of match it is.",
        "features": [
            {"code": "competition_type", "name": "Type of game",
             "desc": "What kind of game this is (for example a league game, a friendly, or a tournament). Different kinds can play out differently."},
        ],
    },
]


def all_feature_codes() -> list[str]:
    return [f["code"] for g in FEATURE_GROUPS for f in g["features"]]


def feature_lookup() -> dict[str, dict]:
    """Map each feature code to its catalog entry (with group name attached)."""
    out = {}
    for group in FEATURE_GROUPS:
        for feat in group["features"]:
            out[feat["code"]] = {**feat, "group": group["group"]}
    return out
