"""Fair-price engine for single-game binary markets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_number(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


def _confidence(final_edge: float, spread_cents: float, volume: float, min_volume: float) -> str:
    if not np.isfinite(final_edge):
        return "none"
    if final_edge >= 0.08 and spread_cents <= 4 and volume >= min_volume * 5:
        return "high"
    if final_edge >= 0.04 and spread_cents <= 8 and volume >= min_volume:
        return "medium"
    if final_edge > 0:
        return "low"
    return "none"


def _side_output(
    side: str,
    calibrated_yes_prob: float,
    yes_bid: float,
    yes_ask: float,
    fee_penalty: float,
    uncertainty_penalty: float,
    spread_penalty_fraction: float,
) -> dict[str, float | str]:
    spread_cents = yes_ask - yes_bid if np.isfinite(yes_bid) and np.isfinite(yes_ask) else float("nan")
    spread_penalty = (spread_cents / 100.0) * spread_penalty_fraction if np.isfinite(spread_cents) else float("nan")
    if side == "YES":
        model_prob = calibrated_yes_prob
        price_cents = yes_ask
    else:
        model_prob = 1.0 - calibrated_yes_prob
        price_cents = 100.0 - yes_bid if np.isfinite(yes_bid) else float("nan")
    market_prob = price_cents / 100.0 if np.isfinite(price_cents) else float("nan")
    gross_edge = model_prob - market_prob if np.isfinite(market_prob) else float("nan")
    final_edge = (
        gross_edge - fee_penalty - spread_penalty - uncertainty_penalty
        if np.isfinite(gross_edge) and np.isfinite(spread_penalty)
        else float("nan")
    )
    return {
        "side": side,
        "model_prob": model_prob,
        "price_cents": price_cents,
        "market_prob": market_prob,
        "gross_edge": gross_edge,
        "fee_adjusted_edge": gross_edge - fee_penalty if np.isfinite(gross_edge) else float("nan"),
        "spread_penalty": spread_penalty,
        "uncertainty_penalty": uncertainty_penalty,
        "final_edge": final_edge,
    }


def build_fair_price_signals(
    markets: pd.DataFrame,
    edge_threshold: float = 0.03,
    max_spread_cents: float = 10.0,
    min_volume: float = 10.0,
    fee_penalty: float = 0.005,
    uncertainty_penalty: float = 0.02,
    spread_penalty_fraction: float = 0.5,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
    research_lean_min_edge: float | None = None,
    paper_trade_min_edge: float = 0.05,
    paper_trade_min_model_probability: float = 0.55,
    paper_trade_min_volume: float | None = None,
    paper_trade_max_spread_cents: float | None = None,
) -> pd.DataFrame:
    """Score both sides of each market with conservative tradable-price penalties."""

    if markets.empty:
        return pd.DataFrame()
    required = ["game_id", "market_ticker", "home_team_abbr", "away_team_abbr", "yes_team_abbr", "model_yes_prob"]
    missing = [column for column in required if column not in markets.columns]
    if missing:
        raise ValueError(f"Fair-price inputs are missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, row in markets.copy().iterrows():
        model_yes_prob = _to_number(row.get("model_yes_prob"))
        calibrated_yes_prob = _to_number(row.get("calibrated_yes_prob", model_yes_prob))
        if not np.isfinite(calibrated_yes_prob):
            calibrated_yes_prob = model_yes_prob
        yes_bid = _to_number(row.get("yes_bid"))
        yes_ask = _to_number(row.get("yes_ask", row.get("yes_mid_cents")))
        if not np.isfinite(yes_ask):
            yes_ask = _to_number(row.get("yes_mid_cents"))
        volume = _to_number(row.get("volume"))
        open_interest = _to_number(row.get("open_interest"))
        spread_cents = yes_ask - yes_bid if np.isfinite(yes_bid) and np.isfinite(yes_ask) else float("nan")

        yes = _side_output(
            "YES",
            calibrated_yes_prob,
            yes_bid,
            yes_ask,
            fee_penalty=fee_penalty,
            uncertainty_penalty=uncertainty_penalty,
            spread_penalty_fraction=spread_penalty_fraction,
        )
        no = _side_output(
            "NO",
            calibrated_yes_prob,
            yes_bid,
            yes_ask,
            fee_penalty=fee_penalty,
            uncertainty_penalty=uncertainty_penalty,
            spread_penalty_fraction=spread_penalty_fraction,
        )
        candidates = [candidate for candidate in [yes, no] if np.isfinite(float(candidate["final_edge"]))]
        best = max(candidates, key=lambda item: float(item["final_edge"])) if candidates else yes
        spread_ok = np.isfinite(spread_cents) and spread_cents <= max_spread_cents
        liquidity_ok = np.isfinite(volume) and volume >= min_volume
        final_edge = float(best["final_edge"])
        edge_ok = np.isfinite(final_edge) and final_edge >= edge_threshold
        research_min_edge = edge_threshold if research_lean_min_edge is None else research_lean_min_edge
        research_ok = spread_ok and liquidity_ok and np.isfinite(final_edge) and final_edge >= research_min_edge
        paper_min_volume = min_volume if paper_trade_min_volume is None else paper_trade_min_volume
        paper_max_spread = max_spread_cents if paper_trade_max_spread_cents is None else paper_trade_max_spread_cents
        best_model_prob = float(best["model_prob"])
        best_market_prob = float(best["market_prob"])
        paper_trade_ok = (
            research_ok
            and final_edge >= paper_trade_min_edge
            and np.isfinite(best_model_prob)
            and best_model_prob >= paper_trade_min_model_probability
            and np.isfinite(best_market_prob)
            and 0.01 <= best_market_prob <= 0.99
            and spread_ok
            and liquidity_ok
            and np.isfinite(spread_cents)
            and spread_cents <= paper_max_spread
            and np.isfinite(volume)
            and volume >= paper_min_volume
        )
        if paper_trade_ok:
            recommendation_tier = "paper_trade_candidate"
        elif research_ok:
            recommendation_tier = "research_lean"
        else:
            recommendation_tier = "none"
        research_side = str(best["side"]) if recommendation_tier != "none" else ""
        research_price = best["price_cents"] if research_side else np.nan
        research_model_prob = best["model_prob"] if research_side else np.nan
        research_market_prob = best["market_prob"] if research_side else np.nan
        research_recommendation = (
            f"{recommendation_tier.replace('_', ' ').title()} {research_side}"
            if research_side
            else "No bet"
        )

        if not spread_ok:
            recommendation = "No bet"
            reason = "bid_ask_spread_too_wide"
            action_side = ""
        elif not liquidity_ok:
            recommendation = "No bet"
            reason = "liquidity_below_minimum"
            action_side = ""
        elif edge_ok:
            action_side = str(best["side"])
            recommendation = f"Bet {action_side}"
            reason = "edge_survives_price_fee_spread_uncertainty_screens"
        else:
            recommendation = "No bet"
            reason = "final_edge_below_threshold"
            action_side = ""

        confidence = _confidence(float(best["final_edge"]), spread_cents, volume, min_volume)
        max_size = starting_bankroll * max_bet_fraction if action_side else 0.0
        confidence_label = confidence if research_side else "none"
        rows.append(
            {
                "game_date": row.get("game_date", row.get("date")),
                "game_id": str(row["game_id"]),
                "market": f"{row.get('away_team_abbr')} at {row.get('home_team_abbr')}",
                "home_team": row.get("home_team_abbr"),
                "away_team": row.get("away_team_abbr"),
                "yes_team": row.get("yes_team_abbr"),
                "market_ticker": row.get("market_ticker"),
                "series_ticker": row.get("series_ticker", ""),
                "model_prob": model_yes_prob,
                "calibrated_prob": calibrated_yes_prob,
                "market_yes_ask": yes_ask,
                "market_no_ask": no["price_cents"],
                "fair_yes_price": calibrated_yes_prob * 100.0,
                "fair_no_price": (1.0 - calibrated_yes_prob) * 100.0,
                "yes_gross_edge": yes["gross_edge"],
                "no_gross_edge": no["gross_edge"],
                "ungated_side": action_side,
                "ungated_recommendation": recommendation,
                "ungated_main_reason": reason,
                "research_side": research_side,
                "research_price": research_price,
                "research_model_probability": research_model_prob,
                "research_market_implied_probability": research_market_prob,
                "research_recommendation": research_recommendation,
                "side": action_side,
                "price": best["price_cents"] if action_side else np.nan,
                "gross_edge": best["gross_edge"],
                "fee_adjusted_edge": best["fee_adjusted_edge"],
                "spread_penalty": best["spread_penalty"],
                "uncertainty_penalty": best["uncertainty_penalty"],
                "final_edge": best["final_edge"],
                "edge": best["final_edge"],
                "market_implied_probability": best["market_prob"],
                "confidence": confidence if action_side else "none",
                "confidence_label": confidence_label,
                "recommendation": recommendation,
                "recommendation_tier": recommendation_tier,
                "max_size": max_size,
                "main_reason": reason,
                "main_risk": "market_price_or_liquidity_quality",
                "blocked_reason": "",
                "parlay_eligibility": "blocked_until_single_game_edge_is_proven",
                "spread": spread_cents,
                "volume": volume,
                "open_interest": open_interest,
                "spread_ok": spread_ok,
                "liquidity_ok": liquidity_ok,
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["rank"] = output["final_edge"].rank(method="first", ascending=False, na_option="bottom").astype(int)
    output = output.sort_values(["recommendation", "final_edge"], ascending=[True, False]).reset_index(drop=True)
    return output[
        [
            "rank",
            "game_date",
            "market",
            "market_ticker",
            "side",
            "price",
            "model_prob",
            "calibrated_prob",
            "market_yes_ask",
            "market_no_ask",
            "fair_yes_price",
            "fair_no_price",
            "gross_edge",
            "fee_adjusted_edge",
            "spread_penalty",
            "uncertainty_penalty",
            "final_edge",
            "confidence",
            "recommendation",
            "max_size",
            "main_reason",
            "main_risk",
            "parlay_eligibility",
            "home_team",
            "away_team",
            "yes_team",
            "game_id",
            "series_ticker",
            "ungated_side",
            "ungated_recommendation",
            "ungated_main_reason",
            "research_side",
            "research_price",
            "research_model_probability",
            "research_market_implied_probability",
            "research_recommendation",
            "market_implied_probability",
            "edge",
            "confidence_label",
            "recommendation_tier",
            "blocked_reason",
            "spread",
            "volume",
            "open_interest",
            "spread_ok",
            "liquidity_ok",
        ]
    ]


def apply_single_game_proof_gate(
    signals: pd.DataFrame,
    single_game_edge_proven: bool,
    proof_status: str = "unknown",
) -> pd.DataFrame:
    """Block action-looking fair-price recommendations until proof gates pass."""

    if signals.empty or single_game_edge_proven:
        output = signals.copy()
        if not output.empty:
            approved_source = output["side"] if "side" in output.columns else output.get("research_side", "")
            approved = approved_source.fillna("").astype(str).ne("")
            output.loc[approved, "recommendation_tier"] = "approved_bet"
            output.loc[approved, "blocked_reason"] = ""
            output["proof_gate_status"] = proof_status
            output["single_game_edge_proven"] = bool(single_game_edge_proven)
        return output
    output = signals.copy()
    if "ungated_side" not in output.columns:
        output["ungated_side"] = output.get("side", "")
    if "ungated_recommendation" not in output.columns:
        output["ungated_recommendation"] = output.get("recommendation", "")
    if "ungated_main_reason" not in output.columns:
        output["ungated_main_reason"] = output.get("main_reason", "")
    if "recommendation_tier" not in output.columns:
        output["recommendation_tier"] = np.where(
            output["ungated_side"].fillna("").astype(str).ne(""),
            "research_lean",
            "none",
        )
    if "research_side" not in output.columns:
        output["research_side"] = output["ungated_side"]
    if "research_price" not in output.columns:
        output["research_price"] = output.get("price", np.nan)
    if "research_model_probability" not in output.columns:
        output["research_model_probability"] = output.get("model_prob", np.nan)
    if "market_implied_probability" not in output.columns:
        output["market_implied_probability"] = output.get("price", np.nan) / 100.0
    if "research_market_implied_probability" not in output.columns:
        output["research_market_implied_probability"] = output["market_implied_probability"]
    if "edge" not in output.columns:
        output["edge"] = output.get("final_edge", np.nan)
    if "confidence_label" not in output.columns:
        output["confidence_label"] = output.get("confidence", "none")
    if "blocked_reason" not in output.columns:
        output["blocked_reason"] = ""
    research_mask = output["recommendation_tier"].isin(["research_lean", "paper_trade_candidate"]) & output[
        "research_side"
    ].fillna("").astype(str).ne("")
    output.loc[research_mask, "main_risk"] = "single_game_edge_not_proven"
    output["side"] = ""
    output["price"] = np.nan
    output.loc[~research_mask, "confidence"] = "none"
    output.loc[research_mask, "confidence"] = output.loc[research_mask, "confidence_label"]
    output["recommendation"] = "No bet"
    output.loc[research_mask, "recommendation"] = (
        output.loc[research_mask, "recommendation_tier"].astype(str).str.replace("_", " ").str.title()
        + " "
        + output.loc[research_mask, "research_side"].astype(str)
    )
    output["max_size"] = 0.0
    output["main_reason"] = "single_game_edge_not_proven"
    output.loc[research_mask, "main_reason"] = output.loc[research_mask, "ungated_main_reason"]
    output["blocked_reason"] = "single_game_edge_not_proven"
    output["parlay_eligibility"] = "blocked_until_single_game_edge_is_proven"
    output["proof_gate_status"] = proof_status
    output["single_game_edge_proven"] = False
    return output


def summarize_fair_price_signals(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"rows": 0, "bets": 0}
    summary = {
        "rows": int(len(signals)),
        "bets": int(signals["side"].astype(str).ne("").sum()),
        "yes_bets": int(signals["side"].astype(str).eq("YES").sum()),
        "no_bets": int(signals["side"].astype(str).eq("NO").sum()),
        "no_bet_rows": int(signals["side"].astype(str).eq("").sum()),
        "average_final_edge": float(pd.to_numeric(signals["final_edge"], errors="coerce").mean()),
        "max_final_edge": float(pd.to_numeric(signals["final_edge"], errors="coerce").max()),
    }
    if "recommendation_tier" in signals.columns:
        tier = signals["recommendation_tier"].fillna("none").astype(str)
        summary["approved_bets_count"] = int(tier.eq("approved_bet").sum())
        summary["paper_trade_candidates_count"] = int(tier.eq("paper_trade_candidate").sum())
        summary["research_leans_count"] = int(tier.eq("research_lean").sum())
        summary["research_only_recommendations_count"] = int(tier.isin(["paper_trade_candidate", "research_lean"]).sum())
    else:
        summary["approved_bets_count"] = int(signals["side"].astype(str).ne("").sum())
        summary["paper_trade_candidates_count"] = 0
        summary["research_leans_count"] = 0
        summary["research_only_recommendations_count"] = 0
    if "blocked_reason" in signals.columns:
        reasons = signals["blocked_reason"].fillna("").astype(str)
        non_empty = reasons[reasons.ne("")]
        summary["blocked_reason"] = str(non_empty.mode().iloc[0]) if not non_empty.empty else ""
    if "ungated_side" in signals.columns:
        summary["ungated_bets"] = int(signals["ungated_side"].fillna("").astype(str).ne("").sum())
        summary["ungated_yes_bets"] = int(signals["ungated_side"].fillna("").astype(str).eq("YES").sum())
        summary["ungated_no_bets"] = int(signals["ungated_side"].fillna("").astype(str).eq("NO").sum())
    if "proof_gate_status" in signals.columns:
        summary["proof_gate_status"] = str(signals["proof_gate_status"].dropna().iloc[0]) if signals["proof_gate_status"].notna().any() else "unknown"
        summary["proof_status"] = summary["proof_gate_status"]
    if "single_game_edge_proven" in signals.columns:
        summary["single_game_edge_proven"] = bool(signals["single_game_edge_proven"].fillna(False).astype(bool).any())
    return summary


def save_fair_price_signals(signals: pd.DataFrame, summary: dict[str, Any], output_path: str | Path, summary_path: str | Path) -> None:
    output = Path(output_path)
    summary_output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
