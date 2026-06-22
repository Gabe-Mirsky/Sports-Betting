"""Conservative two-leg parlay recommendations from proven single-game edges."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PARLAY_OUTPUT_COLUMNS = [
    "rank",
    "status",
    "leg_1_game",
    "leg_1_pick",
    "leg_1_our_odds",
    "leg_1_market_odds",
    "leg_1_edge",
    "leg_2_game",
    "leg_2_pick",
    "leg_2_our_odds",
    "leg_2_market_odds",
    "leg_2_edge",
    "combined_model_probability",
    "combined_market_probability",
    "combined_edge",
    "suggested_stake",
    "potential_profit",
    "risk",
    "main_reason",
    "main_risk",
]

RESEARCH_PARLAY_COLUMNS = [
    "rank",
    "parlay_tier",
    "research_only",
    "approved",
    "legs",
    "number_of_legs",
    "combined_model_probability",
    "estimated_fair_payout",
    "offered_payout",
    "break_even_probability",
    "estimated_ev",
    "average_leg_edge",
    "lowest_leg_edge",
    "correlation_risk",
    "same_game_count",
    "same_team_count",
    "same_player_count",
    "opposing_rebound_conflict",
    "biggest_risk",
    "reason_selected",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _pick_side(row: pd.Series) -> str:
    side = str(row.get("side") or "").strip().upper()
    if side in {"YES", "NO"}:
        return side
    text = str(row.get("recommendation") or "").strip().upper()
    if "BET YES" in text:
        return "YES"
    if "BET NO" in text:
        return "NO"
    return ""


def _leg_from_row(row: pd.Series) -> dict[str, Any] | None:
    side = _pick_side(row)
    if side not in {"YES", "NO"}:
        return None
    model_prob = pd.to_numeric(pd.Series([row.get("calibrated_prob", row.get("model_prob"))]), errors="coerce").iloc[0]
    yes_ask = pd.to_numeric(pd.Series([row.get("market_yes_ask")]), errors="coerce").iloc[0]
    no_ask = pd.to_numeric(pd.Series([row.get("market_no_ask")]), errors="coerce").iloc[0]
    if not np.isfinite(model_prob):
        return None
    if side == "YES":
        our_odds = float(model_prob)
        market_odds = float(yes_ask) / 100.0 if np.isfinite(yes_ask) else np.nan
    else:
        our_odds = float(1.0 - model_prob)
        market_odds = float(no_ask) / 100.0 if np.isfinite(no_ask) else np.nan
    edge = pd.to_numeric(pd.Series([row.get("final_edge", our_odds - market_odds)]), errors="coerce").iloc[0]
    if not np.isfinite(our_odds) or not np.isfinite(market_odds) or not np.isfinite(edge):
        return None
    if our_odds <= 0 or market_odds <= 0:
        return None
    team = str(row.get("yes_team") or row.get("yes_team_abbr") or "").strip()
    pick_team = team if side == "YES" else _opponent_for_no(row, team)
    game = str(row.get("market") or f"{row.get('away_team', '')} at {row.get('home_team', '')}").strip()
    return {
        "game_id": str(row.get("game_id") or ""),
        "market_ticker": str(row.get("market_ticker") or ""),
        "game": game,
        "pick": pick_team or f"{side} {team}".strip(),
        "side": side,
        "our_odds": our_odds,
        "market_odds": market_odds,
        "edge": float(edge),
    }


def _research_leg_from_row(row: pd.Series) -> dict[str, Any] | None:
    side = str(row.get("research_side") or row.get("side") or row.get("ungated_side") or "").strip().upper()
    if side not in {"YES", "NO"}:
        return None
    model_prob = pd.to_numeric(pd.Series([row.get("research_model_probability", row.get("model_prob"))]), errors="coerce").iloc[0]
    market_prob = pd.to_numeric(
        pd.Series([row.get("research_market_implied_probability", row.get("market_implied_probability"))]),
        errors="coerce",
    ).iloc[0]
    price = pd.to_numeric(pd.Series([row.get("research_price", row.get("price"))]), errors="coerce").iloc[0]
    edge = pd.to_numeric(pd.Series([row.get("edge", row.get("final_edge"))]), errors="coerce").iloc[0]
    if not np.isfinite(model_prob) or not np.isfinite(edge):
        return None
    if model_prob <= 0 or model_prob >= 1:
        return None
    if not np.isfinite(market_prob) and np.isfinite(price):
        market_prob = float(price) / 100.0
    if np.isfinite(market_prob) and (market_prob <= 0 or market_prob >= 1):
        market_prob = np.nan
    team = str(row.get("yes_team") or row.get("yes_team_abbr") or "").strip()
    pick_team = team if side == "YES" else _opponent_for_no(row, team)
    player = str(row.get("player_name") or row.get("player") or "").strip()
    game = str(row.get("market") or f"{row.get('away_team', '')} at {row.get('home_team', '')}").strip()
    return {
        "game_id": str(row.get("game_id") or ""),
        "market_ticker": str(row.get("market_ticker") or ""),
        "market": game,
        "side": side,
        "pick": pick_team or f"{side} {team}".strip(),
        "team": pick_team or team,
        "player": player,
        "model_probability": float(model_prob),
        "market_probability": float(market_prob) if np.isfinite(market_prob) else np.nan,
        "price": float(price) if np.isfinite(price) else np.nan,
        "edge": float(edge),
        "tier": str(row.get("recommendation_tier") or ""),
        "text": " ".join(str(row.get(column) or "") for column in ["market", "market_ticker", "reason", "main_reason"]),
    }


def _duplicate_count(values: list[str]) -> int:
    clean = [value for value in values if value]
    return max(0, len(clean) - len(set(clean)))


def _opposing_rebound_conflict(legs: tuple[dict[str, Any], ...]) -> bool:
    rebound_legs = [leg for leg in legs if "rebound" in str(leg.get("text", "")).lower()]
    if len(rebound_legs) < 2:
        return False
    by_player: dict[str, set[str]] = {}
    for leg in rebound_legs:
        player = str(leg.get("player") or "").strip()
        if not player:
            continue
        by_player.setdefault(player, set()).add(str(leg.get("side") or ""))
    return any(len(sides) > 1 for sides in by_player.values())


def _correlation_profile(legs: tuple[dict[str, Any], ...]) -> tuple[str, int, int, int, bool, str]:
    same_game_count = _duplicate_count([str(leg.get("game_id") or "") for leg in legs])
    same_team_count = _duplicate_count([str(leg.get("team") or "") for leg in legs])
    same_player_count = _duplicate_count([str(leg.get("player") or "") for leg in legs])
    rebound_conflict = _opposing_rebound_conflict(legs)
    if same_player_count:
        return "high", same_game_count, same_team_count, same_player_count, rebound_conflict, "same_player_props"
    if rebound_conflict:
        return "high", same_game_count, same_team_count, same_player_count, rebound_conflict, "opposing_rebound_conflict"
    if same_game_count:
        return "high", same_game_count, same_team_count, same_player_count, rebound_conflict, "same_game_legs"
    if same_team_count:
        return "medium", same_game_count, same_team_count, same_player_count, rebound_conflict, "same_team_legs"
    return "unknown", same_game_count, same_team_count, same_player_count, rebound_conflict, "correlation_not_modeled"


def build_research_parlay_candidates(
    fair_price_signals: pd.DataFrame,
    parlay_tier: str,
    max_legs: int = 2,
    max_candidates: int = 50,
    min_average_leg_edge: float = 0.03,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build unapproved research-only parlay candidates from fair-price research tiers."""

    if parlay_tier not in {"research_parlay", "paper_parlay"}:
        raise ValueError("parlay_tier must be research_parlay or paper_parlay")
    if max_legs != 2:
        raise ValueError("Only two-leg research parlays are supported until correlation modeling is added.")
    if fair_price_signals.empty or "recommendation_tier" not in fair_price_signals.columns:
        return pd.DataFrame(columns=RESEARCH_PARLAY_COLUMNS), {
            "parlay_tier": parlay_tier,
            "status": "no_eligible_research_legs",
            "input_rows": int(len(fair_price_signals)),
            "eligible_legs": 0,
            "parlays": 0,
            "research_only": True,
            "approved": False,
        }

    allowed_tiers = {"paper_trade_candidate"} if parlay_tier == "paper_parlay" else {"paper_trade_candidate", "research_lean"}
    frame = fair_price_signals[
        fair_price_signals["recommendation_tier"].fillna("").astype(str).isin(allowed_tiers)
    ].copy()
    legs = [leg for _, row in frame.iterrows() if (leg := _research_leg_from_row(row)) is not None]
    rows: list[dict[str, Any]] = []
    for combo in combinations(legs, max_legs):
        model_probs = [float(leg["model_probability"]) for leg in combo]
        market_probs = [float(leg["market_probability"]) for leg in combo]
        edges = [float(leg["edge"]) for leg in combo]
        average_edge = float(np.mean(edges))
        if average_edge < min_average_leg_edge:
            continue
        combined_model = float(np.prod(model_probs))
        fair_payout = float(1.0 / combined_model) if combined_model > 0 else np.nan
        combined_market = float(np.prod(market_probs)) if all(np.isfinite(market_probs)) else np.nan
        offered_payout = float(1.0 / combined_market) if np.isfinite(combined_market) and combined_market > 0 else np.nan
        estimated_ev = float((combined_model * offered_payout) - 1.0) if np.isfinite(offered_payout) else np.nan
        correlation_risk, same_game_count, same_team_count, same_player_count, rebound_conflict, biggest_risk = (
            _correlation_profile(combo)
        )
        leg_text = " | ".join(
            f"{leg['market']} {leg['side']} @ {leg['price']:.1f}c edge {leg['edge']:.3f}"
            if np.isfinite(float(leg["price"]))
            else f"{leg['market']} {leg['side']} edge {leg['edge']:.3f}"
            for leg in combo
        )
        rows.append(
            {
                "parlay_tier": parlay_tier,
                "research_only": True,
                "approved": False,
                "legs": leg_text,
                "number_of_legs": int(max_legs),
                "combined_model_probability": combined_model,
                "estimated_fair_payout": fair_payout,
                "offered_payout": offered_payout,
                "break_even_probability": combined_market,
                "estimated_ev": estimated_ev,
                "average_leg_edge": average_edge,
                "lowest_leg_edge": float(min(edges)),
                "correlation_risk": correlation_risk,
                "same_game_count": int(same_game_count),
                "same_team_count": int(same_team_count),
                "same_player_count": int(same_player_count),
                "opposing_rebound_conflict": bool(rebound_conflict),
                "biggest_risk": biggest_risk,
                "reason_selected": f"{parlay_tier} built from {', '.join(sorted(allowed_tiers))}; research only, not approved.",
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        summary = {
            "parlay_tier": parlay_tier,
            "status": "no_research_parlays_after_filters",
            "input_rows": int(len(fair_price_signals)),
            "eligible_legs": int(len(legs)),
            "parlays": 0,
            "research_only": True,
            "approved": False,
            "source_recommendation_tiers": sorted(allowed_tiers),
        }
        return pd.DataFrame(columns=RESEARCH_PARLAY_COLUMNS), summary
    output = output.sort_values(
        ["estimated_ev", "average_leg_edge", "combined_model_probability"],
        ascending=[False, False, False],
        na_position="last",
    ).head(max_candidates)
    output.insert(0, "rank", range(1, len(output) + 1))
    output = output.reindex(columns=RESEARCH_PARLAY_COLUMNS)
    summary = {
        "parlay_tier": parlay_tier,
        "status": "research_only_generated",
        "input_rows": int(len(fair_price_signals)),
        "eligible_legs": int(len(legs)),
        "parlays": int(len(output)),
        "research_only": True,
        "approved": False,
        "source_recommendation_tiers": sorted(allowed_tiers),
        "best_estimated_ev": float(pd.to_numeric(output["estimated_ev"], errors="coerce").max()),
        "best_average_leg_edge": float(pd.to_numeric(output["average_leg_edge"], errors="coerce").max()),
        "correlation_risk_counts": output["correlation_risk"].value_counts(dropna=False).to_dict(),
    }
    return output, summary


def _opponent_for_no(row: pd.Series, yes_team: str) -> str:
    home = str(row.get("home_team") or row.get("home_team_abbr") or "").strip()
    away = str(row.get("away_team") or row.get("away_team_abbr") or "").strip()
    if yes_team and home and yes_team == home:
        return away
    if yes_team and away and yes_team == away:
        return home
    return f"NO {yes_team}".strip()


def _risk_label(combined_model_probability: float, combined_edge: float) -> str:
    if combined_model_probability < 0.20:
        return "High"
    if combined_model_probability < 0.35 or combined_edge < 0.05:
        return "Medium"
    return "Lower"


def _stake(bankroll: float, combined_edge: float) -> float:
    if combined_edge < 0.03:
        return 0.0
    if combined_edge >= 0.10:
        return round(bankroll * 0.02, 2)
    return round(bankroll * 0.01, 2)


def build_parlay_recommendations(
    fair_price_signals: pd.DataFrame,
    proof_summary: dict[str, Any] | None = None,
    bankroll: float = 100.0,
    max_legs: int = 2,
    min_leg_edge: float = 0.02,
    min_combined_edge: float = 0.03,
    max_recommendations: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build conservative different-game parlay recommendations.

    This intentionally excludes same-game parlays. Until correlation is explicitly
    modeled, different-game legs are treated as approximately independent.
    """

    proof_summary = proof_summary or {}
    proof_allowed = _as_bool(proof_summary.get("single_game_edge_proven"))
    if max_legs != 2:
        raise ValueError("Only two-leg parlays are supported until correlation modeling is added.")
    if fair_price_signals.empty:
        summary = _summary("blocked_no_single_game_candidates", bankroll, 0, 0, proof_allowed)
        return pd.DataFrame(columns=PARLAY_OUTPUT_COLUMNS), summary

    frame = fair_price_signals.copy()
    if "recommendation" not in frame.columns:
        summary = _summary("blocked_missing_recommendation_column", bankroll, len(frame), 0, proof_allowed)
        return pd.DataFrame(columns=PARLAY_OUTPUT_COLUMNS), summary

    final_edge = frame["final_edge"] if "final_edge" in frame.columns else pd.Series(0, index=frame.index)
    active = frame[
        frame["recommendation"].astype(str).str.startswith("Bet ", na=False)
        & pd.to_numeric(final_edge, errors="coerce").ge(min_leg_edge)
    ].copy()
    if not proof_allowed:
        summary = _summary("blocked_single_game_edge_not_proven", bankroll, len(frame), len(active), proof_allowed)
        return pd.DataFrame(columns=PARLAY_OUTPUT_COLUMNS), summary
    if active.empty:
        summary = _summary("blocked_no_positive_single_game_edges", bankroll, len(frame), 0, proof_allowed)
        return pd.DataFrame(columns=PARLAY_OUTPUT_COLUMNS), summary

    legs = [leg for _, row in active.iterrows() if (leg := _leg_from_row(row)) is not None]
    rows: list[dict[str, Any]] = []
    for leg_1, leg_2 in combinations(legs, 2):
        if leg_1["game_id"] and leg_1["game_id"] == leg_2["game_id"]:
            continue
        combined_model = float(leg_1["our_odds"] * leg_2["our_odds"])
        combined_market = float(leg_1["market_odds"] * leg_2["market_odds"])
        combined_edge = combined_model - combined_market
        if combined_edge < min_combined_edge:
            continue
        stake = _stake(bankroll, combined_edge)
        if stake <= 0 or combined_market <= 0:
            continue
        rows.append(
            {
                "status": "research_only",
                "leg_1_game": leg_1["game"],
                "leg_1_pick": leg_1["pick"],
                "leg_1_our_odds": leg_1["our_odds"],
                "leg_1_market_odds": leg_1["market_odds"],
                "leg_1_edge": leg_1["edge"],
                "leg_2_game": leg_2["game"],
                "leg_2_pick": leg_2["pick"],
                "leg_2_our_odds": leg_2["our_odds"],
                "leg_2_market_odds": leg_2["market_odds"],
                "leg_2_edge": leg_2["edge"],
                "combined_model_probability": combined_model,
                "combined_market_probability": combined_market,
                "combined_edge": combined_edge,
                "suggested_stake": stake,
                "potential_profit": round(stake * ((1.0 / combined_market) - 1.0), 2),
                "risk": _risk_label(combined_model, combined_edge),
                "main_reason": "Two positive-edge picks from different games.",
                "main_risk": "Different-game legs are treated as independent; correlation is not modeled yet.",
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        summary = _summary("blocked_no_parlays_after_filters", bankroll, len(frame), len(active), proof_allowed)
        return pd.DataFrame(columns=PARLAY_OUTPUT_COLUMNS), summary
    output = output.sort_values(["combined_edge", "combined_model_probability"], ascending=[False, False]).head(
        max_recommendations
    )
    output.insert(0, "rank", range(1, len(output) + 1))
    output = output.reindex(columns=PARLAY_OUTPUT_COLUMNS)
    summary = _summary("ready_research_only", bankroll, len(frame), len(active), proof_allowed)
    summary.update(
        {
            "parlays": int(len(output)),
            "best_combined_edge": float(output["combined_edge"].max()),
            "best_combined_model_probability": float(output["combined_model_probability"].max()),
            "assumption": "Same-game parlays excluded; different-game legs treated as approximately independent until correlation is modeled.",
        }
    )
    return output, summary


def _summary(status: str, bankroll: float, input_rows: int, eligible_legs: int, proof_allowed: bool) -> dict[str, Any]:
    return {
        "status": status,
        "bankroll": float(bankroll),
        "input_rows": int(input_rows),
        "eligible_single_game_legs": int(eligible_legs),
        "parlays": 0,
        "single_game_edge_proven": bool(proof_allowed),
        "same_game_parlays_allowed": False,
        "parlay_recommendations_allowed": bool(proof_allowed and eligible_legs >= 2),
        "assumption": "Same-game parlays are excluded until correlation is modeled.",
    }


def save_parlay_recommendations(
    recommendations: pd.DataFrame,
    summary: dict[str, Any],
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    output = Path(output_path)
    summary_output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
