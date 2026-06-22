"""Sport-level rules shared by the no-odds matchup prediction pipeline.

This module is intentionally tiny and dependency-free so every other matchup
module can import it without creating cycles. It answers two questions:

1. What is the canonical name for a sport string the user typed?
2. Does that sport allow draws (so the model needs a three-way target)?
"""

from __future__ import annotations


# Canonical sport keys used throughout the matchup pipeline.
SOCCER: str = "soccer"

# Map of common spellings/variants to a canonical sport key. Anything not in
# the map is lowercased and returned as-is so new sports still flow through.
_SPORT_ALIASES: dict[str, str] = {
    "soccer": SOCCER,
    "football": SOCCER,  # association football
    "association football": SOCCER,
    "futbol": SOCCER,
    "fussball": SOCCER,
    "fútbol": SOCCER,
    "basketball": "basketball",
    "nba": "basketball",
    "ncaab": "basketball",
    "baseball": "baseball",
    "mlb": "baseball",
    "hockey": "hockey",
    "ice hockey": "hockey",
    "nhl": "hockey",
    "american football": "american_football",
    "nfl": "american_football",
    "ncaaf": "american_football",
}

# Sports whose regulation result can be a draw. These need a three-outcome
# (team_a win / draw / team_b win) model instead of a binary classifier.
_DRAW_SPORTS: set[str] = {SOCCER, "rugby", "cricket"}


def normalize_sport(sport: object) -> str:
    """Return a canonical, lowercase sport key.

    Unknown sports are passed through lowercased and stripped so the pipeline
    keeps working for sports we have not explicitly catalogued yet.
    """

    text = str(sport).strip().lower()
    if not text:
        return ""
    return _SPORT_ALIASES.get(text, text)


def sport_allows_draws(sport: object, config: dict | None = None) -> bool:
    """Return ``True`` when regulation games for this sport can end in a draw.

    A ``config`` dict may override detection, e.g.::

        {"draw_sports": ["soccer", "handball"]}
        {"no_draw_sports": ["hockey"]}
    """

    canonical = normalize_sport(sport)
    config = config or {}

    overrides_no_draw = {normalize_sport(s) for s in config.get("no_draw_sports", [])}
    if canonical in overrides_no_draw:
        return False

    overrides_draw = {normalize_sport(s) for s in config.get("draw_sports", [])}
    if canonical in overrides_draw:
        return True

    return canonical in _DRAW_SPORTS
