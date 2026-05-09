"""Forward-looking paper recommendations for upcoming NBA markets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.staking import calculate_flat_fractional_shares


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_pick_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.copy()
    rows["game_id"] = rows["game_id"].astype(str)
    rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
    home_prob = pd.to_numeric(rows["model_home_win_prob"], errors="coerce")
    rows["model_pick_team"] = rows["away_team_abbr"]
    rows.loc[home_prob >= 0.5, "model_pick_team"] = rows.loc[home_prob >= 0.5, "home_team_abbr"]
    rows["model_pick_prob"] = pd.to_numeric(rows["model_away_win_prob"], errors="coerce")
    rows.loc[home_prob >= 0.5, "model_pick_prob"] = pd.to_numeric(
        rows.loc[home_prob >= 0.5, "model_home_win_prob"],
        errors="coerce",
    )
    if "upcoming_status" not in rows.columns:
        rows["upcoming_status"] = "Scheduled"
    return rows


def _best_market_rows(upcoming: pd.DataFrame, suggestions: pd.DataFrame) -> pd.DataFrame:
    if suggestions.empty:
        return pd.DataFrame()
    markets = suggestions.copy()
    markets["game_id"] = markets["game_id"].astype(str)
    markets["edge"] = pd.to_numeric(markets.get("edge"), errors="coerce")
    markets["price_cents"] = pd.to_numeric(markets.get("price_cents"), errors="coerce")
    markets["market_prob"] = pd.to_numeric(markets.get("market_prob"), errors="coerce")
    markets["model_yes_prob"] = pd.to_numeric(markets.get("model_yes_prob"), errors="coerce")
    markets["edge_signal"] = _coerce_bool(markets["trade"]) if "trade" in markets.columns else False
    pick_key = upcoming[["game_id", "model_pick_team"]].rename(columns={"model_pick_team": "yes_team_abbr"})
    pick_markets = markets.merge(pick_key, on=["game_id", "yes_team_abbr"], how="inner")
    best_trade = (
        markets[markets["edge_signal"]]
        .sort_values(["game_id", "edge"], ascending=[True, False])
        .drop_duplicates("game_id", keep="first")
    )
    best_pick = (
        pick_markets.sort_values(["game_id", "edge"], ascending=[True, False])
        .drop_duplicates("game_id", keep="first")
    )
    best = best_pick.set_index("game_id")
    if not best_trade.empty:
        best.update(best_trade.set_index("game_id"))
        missing_trade_games = best_trade[~best_trade["game_id"].isin(best.index)]
        if not missing_trade_games.empty:
            best = pd.concat([best, missing_trade_games.set_index("game_id")], axis=0, sort=False)
    return best.reset_index()


def build_forward_recommendations(
    upcoming_predictions: pd.DataFrame,
    market_suggestions: pd.DataFrame,
    readiness_summary: dict[str, Any] | None = None,
    rule_sweep_summary: dict[str, Any] | None = None,
    rule_validation_summary: dict[str, Any] | None = None,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
    respect_readiness_gate: bool = True,
    as_of_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one forward recommendation row per upcoming game."""

    if upcoming_predictions.empty:
        return pd.DataFrame(), {"rows": 0, "paper_bets": 0, "as_of_date": as_of_date}

    upcoming = _model_pick_rows(upcoming_predictions)
    if as_of_date:
        cutoff = pd.Timestamp(as_of_date).normalize()
        upcoming = upcoming[upcoming["game_date"].dt.normalize() >= cutoff].copy()
    upcoming = upcoming.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    best_markets = _best_market_rows(upcoming, market_suggestions)
    if best_markets.empty:
        rows = upcoming.copy()
    else:
        market_columns = [
            column
            for column in [
                "game_id",
                "market_ticker",
                "event_ticker",
                "yes_team_abbr",
                "model_yes_prob",
                "market_prob",
                "edge",
                "price_cents",
                "price_source",
                "edge_signal",
                "reason",
            ]
            if column in best_markets.columns
        ]
        rows = upcoming.merge(best_markets[market_columns], on="game_id", how="left")

    for column in ["model_home_win_prob", "model_away_win_prob", "model_pick_prob", "model_yes_prob", "market_prob", "edge", "price_cents"]:
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["has_kalshi_odds"] = rows["price_cents"].notna()
    rows["edge_signal"] = rows["edge_signal"].fillna(False) if "edge_signal" in rows.columns else False
    rows["edge_signal"] = _coerce_bool(rows["edge_signal"]) if not pd.api.types.is_bool_dtype(rows["edge_signal"]) else rows["edge_signal"]
    rows["forward_expected_roi"] = pd.NA
    if "edge" in rows.columns and "market_prob" in rows.columns:
        market_prob = pd.to_numeric(rows["market_prob"], errors="coerce")
        rows["forward_expected_roi"] = rows["edge"] / market_prob
        rows.loc[market_prob.le(0) | market_prob.isna(), "forward_expected_roi"] = pd.NA
    rule_summary = rule_sweep_summary or {}
    validation_summary = rule_validation_summary or {}
    validation_status = str(validation_summary.get("status", "missing"))
    validation_allows_rule = validation_status in {"walk_forward_candidate"}
    best_rule_params = rule_summary.get("best_rule_params") or {}
    best_rule_status = str(rule_summary.get("best_rule_status", "n/a"))
    best_rule_text = str(rule_summary.get("best_rule", "n/a"))
    rows["best_sweep_rule"] = best_rule_text
    rows["best_sweep_rule_status"] = best_rule_status if validation_allows_rule else f"blocked_{validation_status}"
    rows["rule_validation_status"] = validation_status
    rows["passes_best_sweep_rule"] = False
    rule_is_forward_usable = (
        validation_allows_rule
        and bool(best_rule_params)
        and int(best_rule_params.get("min_edge_bin_history_rows", 0) or 0) <= 0
    )
    if rule_is_forward_usable:
        min_edge = float(best_rule_params.get("min_edge", 0.0) or 0.0)
        min_expected_roi = float(best_rule_params.get("min_expected_roi", 0.0) or 0.0)
        min_price_cents = float(best_rule_params.get("min_price_cents", 0.0) or 0.0)
        max_price_cents = float(best_rule_params.get("max_price_cents", 100.0) or 100.0)
        passes_rule = (
            rows["has_kalshi_odds"]
            & rows["edge"].ge(min_edge)
            & pd.to_numeric(rows["forward_expected_roi"], errors="coerce").ge(min_expected_roi)
            & rows["price_cents"].ge(min_price_cents)
            & rows["price_cents"].le(max_price_cents)
        )
        rows["passes_best_sweep_rule"] = passes_rule.fillna(False)
    elif validation_allows_rule and best_rule_params:
        rows["best_sweep_rule_status"] = "not_forward_usable"
    paper_candidates = int((readiness_summary or {}).get("paper_trade_candidates", 0) or 0)
    readiness_allows_paper = (paper_candidates > 0) or not respect_readiness_gate
    rows["readiness_gate"] = "paper_allowed" if readiness_allows_paper else "watchlist_only"
    rows["paper_trade"] = rows["edge_signal"] & rows["has_kalshi_odds"] & readiness_allows_paper
    rows["hypothetical_paper_trade"] = rows["edge_signal"] & rows["has_kalshi_odds"]
    rows["paper_shares"] = 0
    rows["paper_amount_risked"] = 0.0
    rows["hypothetical_shares"] = 0
    rows["hypothetical_amount_risked"] = 0.0
    for index, row in rows.iterrows():
        if not bool(row["has_kalshi_odds"]) or pd.isna(row.get("price_cents")):
            continue
        shares = calculate_flat_fractional_shares(
            bankroll=starting_bankroll,
            price_cents=float(row["price_cents"]),
            max_bet_fraction=max_bet_fraction,
        )
        amount = shares * float(row["price_cents"]) / 100.0
        if bool(row["hypothetical_paper_trade"]):
            rows.loc[index, "hypothetical_shares"] = shares
            rows.loc[index, "hypothetical_amount_risked"] = amount
        if bool(row["paper_trade"]):
            rows.loc[index, "paper_shares"] = shares
            rows.loc[index, "paper_amount_risked"] = amount

    rows["recommendation"] = "No Kalshi odds loaded"
    rows.loc[rows["has_kalshi_odds"] & ~rows["edge_signal"], "recommendation"] = "No bet - edge below threshold"
    rows.loc[rows["has_kalshi_odds"] & rows["edge_signal"] & ~rows["paper_trade"], "recommendation"] = (
        "Watchlist only - readiness gate"
    )
    rows.loc[
        rows["has_kalshi_odds"] & rows["edge_signal"] & rows["passes_best_sweep_rule"] & ~rows["paper_trade"],
        "recommendation",
    ] = "Watchlist only - passes best sweep rule"
    rows.loc[rows["paper_trade"], "recommendation"] = "Paper bet"
    rows["recommended_team"] = rows.get("yes_team_abbr", pd.Series(index=rows.index, dtype="object")).fillna(
        rows["model_pick_team"]
    )
    rows["generated_at"] = pd.Timestamp.now().isoformat()
    rows["starting_bankroll"] = float(starting_bankroll)
    rows["max_bet_fraction"] = float(max_bet_fraction)

    output_columns = [
        "generated_at",
        "game_date",
        "game_id",
        "upcoming_status",
        "season_type",
        "home_team_abbr",
        "away_team_abbr",
        "model_home_win_prob",
        "model_away_win_prob",
        "model_pick_team",
        "model_pick_prob",
        "recommended_team",
        "market_ticker",
        "event_ticker",
        "price_source",
        "price_cents",
        "market_prob",
        "model_yes_prob",
        "edge",
        "forward_expected_roi",
        "has_kalshi_odds",
        "edge_signal",
        "passes_best_sweep_rule",
        "best_sweep_rule_status",
        "rule_validation_status",
        "readiness_gate",
        "recommendation",
        "paper_trade",
        "paper_shares",
        "paper_amount_risked",
        "hypothetical_shares",
        "hypothetical_amount_risked",
        "reason",
    ]
    rows = rows[[column for column in output_columns if column in rows.columns]].copy()
    if not rows.empty:
        dates = pd.to_datetime(rows["game_date"], errors="coerce").dropna()
        start_date = dates.min().date().isoformat() if not dates.empty else None
        end_date = dates.max().date().isoformat() if not dates.empty else None
        timeline = start_date if start_date == end_date else f"{start_date} to {end_date}" if start_date else "n/a"
    else:
        timeline = "n/a"
    summary = {
        "rows": int(len(rows)),
        "games": int(rows["game_id"].nunique()) if "game_id" in rows.columns else 0,
        "games_with_kalshi_odds": int(rows.loc[rows["has_kalshi_odds"], "game_id"].nunique())
        if "has_kalshi_odds" in rows.columns
        else 0,
        "edge_signals": int(rows["edge_signal"].sum()) if "edge_signal" in rows.columns else 0,
        "paper_bets": int(rows["paper_trade"].sum()) if "paper_trade" in rows.columns else 0,
        "hypothetical_paper_bets": int(rows["hypothetical_shares"].gt(0).sum())
        if "hypothetical_shares" in rows.columns
        else 0,
        "best_sweep_rule_passes": int(rows["passes_best_sweep_rule"].sum())
        if "passes_best_sweep_rule" in rows.columns
        else 0,
        "best_sweep_rule": best_rule_text,
        "best_sweep_rule_status": best_rule_status,
        "rule_validation_status": validation_status,
        "rule_validation_signals": int(validation_summary.get("signals", 0) or 0),
        "rule_validation_positive_months": int(validation_summary.get("positive_months", 0) or 0),
        "rule_validation_months": int(validation_summary.get("months", 0) or 0),
        "rule_validation_gate_allows_forward": bool(validation_allows_rule),
        "paper_amount_risked": float(rows["paper_amount_risked"].sum()) if "paper_amount_risked" in rows.columns else 0.0,
        "hypothetical_amount_risked": float(rows["hypothetical_amount_risked"].sum())
        if "hypothetical_amount_risked" in rows.columns
        else 0.0,
        "starting_bankroll": float(starting_bankroll),
        "max_bet_fraction": float(max_bet_fraction),
        "readiness_gate": "paper_allowed" if readiness_allows_paper else "watchlist_only",
        "respect_readiness_gate": bool(respect_readiness_gate),
        "timeline": timeline,
        "as_of_date": as_of_date,
        "note": "No real trades are placed. In-sample sweep rules are blocked unless nested walk-forward validation passes.",
    }
    return rows.reset_index(drop=True), summary


def build_forward_recommendations_from_files(
    predictions_path: str | Path,
    suggestions_path: str | Path,
    readiness_summary_path: str | Path,
    rule_sweep_summary_path: str | Path | None,
    rule_validation_summary_path: str | Path | None,
    output_path: str | Path,
    summary_path: str | Path,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
    respect_readiness_gate: bool = True,
    as_of_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions_file = Path(predictions_path)
    suggestions_file = Path(suggestions_path)
    predictions = pd.read_csv(predictions_file, dtype={"game_id": str}) if predictions_file.exists() else pd.DataFrame()
    suggestions = pd.read_csv(suggestions_file, dtype={"game_id": str}) if suggestions_file.exists() else pd.DataFrame()
    readiness = _read_json(Path(readiness_summary_path))
    rule_sweep = _read_json(Path(rule_sweep_summary_path)) if rule_sweep_summary_path else {}
    rule_validation = _read_json(Path(rule_validation_summary_path)) if rule_validation_summary_path else {}
    recommendations, summary = build_forward_recommendations(
        predictions,
        suggestions,
        readiness_summary=readiness,
        rule_sweep_summary=rule_sweep,
        rule_validation_summary=rule_validation,
        starting_bankroll=starting_bankroll,
        max_bet_fraction=max_bet_fraction,
        respect_readiness_gate=respect_readiness_gate,
        as_of_date=as_of_date,
    )
    output = Path(output_path)
    summary_output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return recommendations, summary
