"""Recommend parlays from approved predictions, with plain-language reasoning.

A "parlay" wins only if **every** leg wins, so the combined chance is the product
of the legs' individual model probabilities (legs are from different games, so we
treat them as independent). These recommendations are model-implied and are NOT
betting advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

# Only consider picks the model likes at least this much for its own game.
_MIN_LEG_PROB = 0.50
# How many of the best legs to consider when forming combinations.
_LEG_POOL = 8


@dataclass
class RecommendedLeg:
    prediction_id: int
    pick_label: str
    match_label: str
    league: str
    probability: float

    @property
    def probability_pct(self) -> float:
        return round(self.probability * 100, 1)


@dataclass
class RecommendedParlay:
    title: str
    legs: list[RecommendedLeg]
    rationale: list[str]
    combined_probability: float

    @property
    def combined_probability_pct(self) -> float:
        return round(self.combined_probability * 100, 1)

    @property
    def fair_decimal_odds(self) -> float | None:
        if self.combined_probability <= 0:
            return None
        return round(1.0 / self.combined_probability, 2)

    @property
    def fair_american_odds(self) -> str:
        dec = self.fair_decimal_odds
        if not dec or dec <= 1:
            return "n/a"
        if dec >= 2:
            return f"+{round((dec - 1) * 100)}"
        return f"-{round(100 / (dec - 1))}"

    @property
    def prediction_ids(self) -> list[int]:
        return [leg.prediction_id for leg in self.legs]


def _combined(legs: Iterable[RecommendedLeg]) -> float:
    combined = 1.0
    for leg in legs:
        combined *= leg.probability
    return combined


def _leg_from_prediction(prediction) -> RecommendedLeg | None:
    prob = prediction.pick_probability
    if prob is None:
        return None
    return RecommendedLeg(
        prediction_id=prediction.pk,
        pick_label=prediction.predicted_outcome or prediction.matchup,
        match_label=prediction.matchup,
        league=prediction.league or "",
        probability=float(prob),
    )


def recommend_parlays(predictions, max_results: int = 4) -> list[RecommendedParlay]:
    """Return a small, diverse set of recommended parlays with reasoning.

    ``predictions`` is any iterable of ``MatchupPrediction`` instances.
    """

    # Build the pool of eligible legs (strong, confident, one per game).
    eligible = []
    for pred in predictions:
        leg = _leg_from_prediction(pred)
        if leg is None or leg.probability < _MIN_LEG_PROB:
            continue
        if getattr(pred, "confidence_rank", 0) < 1:  # skip "Very low" confidence
            continue
        eligible.append(leg)

    eligible.sort(key=lambda leg: leg.probability, reverse=True)
    pool = eligible[:_LEG_POOL]
    if len(pool) < 2:
        return []

    recs: list[RecommendedParlay] = []
    used_signatures: set[tuple[int, ...]] = set()

    def add(title: str, legs: list[RecommendedLeg], rationale: list[str]) -> None:
        sig = tuple(sorted(leg.prediction_id for leg in legs))
        if len(legs) < 2 or sig in used_signatures:
            return
        used_signatures.add(sig)
        combined = _combined(legs)
        pct = round(combined * 100)
        full = [
            f"This parlay wins only if all {len(legs)} picks win.",
            f"We multiply each pick's chance together, giving about {pct}% combined.",
            "Every leg is from a different game, so one result does not change another.",
        ] + rationale
        recs.append(RecommendedParlay(title=title, legs=legs, rationale=full, combined_probability=combined))

    # 1) Safest double: the two most likely picks.
    add(
        "Safest double",
        pool[:2],
        ["These are the two picks the model is most sure about right now."],
    )

    # 2) Safest treble: the three most likely picks (if available).
    if len(pool) >= 3:
        add(
            "Safest treble",
            pool[:3],
            ["Three strong picks. More legs means a bigger payout but a lower chance to win."],
        )

    # 3) Spread across leagues: top picks from different leagues for variety.
    by_league: dict[str, RecommendedLeg] = {}
    for leg in pool:
        by_league.setdefault(leg.league, leg)
    spread = list(by_league.values())[:2]
    if len(spread) == 2:
        add(
            "Different-leagues double",
            spread,
            ["Picks come from different leagues, so the action is spread around."],
        )

    # 4) Best two-leg combined chance overall (in case the above overlap).
    best_pair = max(combinations(pool, 2), key=_combined)
    add(
        "Highest combined chance",
        list(best_pair),
        ["This pair has the best combined chance to all win among the strong picks."],
    )

    recs.sort(key=lambda r: r.combined_probability, reverse=True)
    return recs[:max_results]
