"""Portfolio-level selection for individual paper bets before parlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.backtest import _date_range, _max_drawdown
from strategy.staking import calculate_flat_fractional_shares


def _is_true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def _timeline_label(start_date: str | None, end_date: str | None) -> str:
    if start_date and end_date:
        return start_date if start_date == end_date else f"{start_date} to {end_date}"
    return "n/a"


def prepare_portfolio_candidates(
    trades: pd.DataFrame,
    min_edge: float = 0.05,
    min_expected_roi: float = 0.0,
    trade_column: str = "trade",
    expected_roi_column: str | None = None,
) -> pd.DataFrame:
    """Return candidate rows eligible for slate optimization."""

    if trades.empty:
        return pd.DataFrame()
    required = ["date", "game_id", "market_ticker", "price_cents", "model_yes_prob", "edge", "actual_yes_win"]
    missing = [column for column in required if column not in trades.columns]
    if missing:
        raise ValueError(f"Trade file is missing portfolio columns: {missing}")

    candidates = trades.copy()
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates["price_cents"] = pd.to_numeric(candidates["price_cents"], errors="coerce")
    candidates["model_yes_prob"] = pd.to_numeric(candidates["model_yes_prob"], errors="coerce")
    candidates["edge"] = pd.to_numeric(candidates["edge"], errors="coerce")
    candidates = candidates.dropna(subset=["date", "price_cents", "model_yes_prob", "edge"]).copy()
    if trade_column in candidates.columns:
        candidates = candidates[_is_true(candidates[trade_column])].copy()
    candidates = candidates[candidates["edge"] >= float(min_edge)].copy()
    candidates["contract_cost"] = candidates["price_cents"] / 100.0
    candidates = candidates[(candidates["contract_cost"] > 0) & (candidates["contract_cost"] < 1)].copy()
    candidates["expected_profit_per_share"] = candidates["model_yes_prob"] - candidates["contract_cost"]
    candidates["expected_roi"] = candidates["expected_profit_per_share"] / candidates["contract_cost"]
    if expected_roi_column and expected_roi_column in candidates.columns:
        candidates["selection_expected_roi"] = pd.to_numeric(candidates[expected_roi_column], errors="coerce")
    else:
        candidates["selection_expected_roi"] = candidates["expected_roi"]
    candidates = candidates[candidates["selection_expected_roi"] >= float(min_expected_roi)].copy()
    return candidates.sort_values(["date", "expected_roi", "edge"], ascending=[True, False, False]).reset_index(drop=True)


def optimize_individual_bet_slate(
    trades: pd.DataFrame,
    starting_bankroll: float = 100.0,
    min_edge: float = 0.05,
    min_expected_roi: float = 0.0,
    trade_column: str = "trade",
    expected_roi_column: str | None = None,
    max_bet_fraction: float = 0.03,
    max_slate_fraction: float = 0.12,
    max_trades_per_slate: int = 5,
    max_markets_per_game: int = 1,
    max_markets_per_team: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a constrained daily slate of individual paper bets.

    This intentionally does not build parlays. It limits same-day exposure and
    avoids taking multiple correlated markets from the same game by default.
    """

    candidates = prepare_portfolio_candidates(
        trades,
        min_edge=min_edge,
        min_expected_roi=min_expected_roi,
        trade_column=trade_column,
        expected_roi_column=expected_roi_column,
    )
    bankroll = float(starting_bankroll)
    selected_rows: list[dict[str, Any]] = []
    slate_rows: list[dict[str, Any]] = []

    if candidates.empty:
        summary = {
            "starting_bankroll": float(starting_bankroll),
            "ending_bankroll": float(starting_bankroll),
            "total_return_pct": 0.0,
            "num_candidate_bets": 0,
            "num_selected_trades": 0,
            "trade_start_date": None,
            "trade_end_date": None,
            "trade_timeline": "n/a",
            "min_edge": float(min_edge),
            "min_expected_roi": float(min_expected_roi),
            "trade_column": trade_column,
            "expected_roi_column": expected_roi_column or "expected_roi",
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "average_edge": 0.0,
            "roi_on_amount_risked": 0.0,
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    for slate_date, slate in candidates.groupby(candidates["date"].dt.date, sort=True):
        bankroll_before_slate = bankroll
        remaining_slate_budget = bankroll_before_slate * float(max_slate_fraction)
        selected_count = 0
        selected_game_counts: dict[str, int] = {}
        selected_team_counts: dict[str, int] = {}
        slate_cost = 0.0
        slate_payout = 0.0
        rejected_game_cap = 0
        rejected_team_cap = 0
        rejected_budget_cap = 0

        for _, row in slate.sort_values(["selection_expected_roi", "edge"], ascending=False).iterrows():
            if selected_count >= int(max_trades_per_slate):
                break
            game_id = str(row["game_id"])
            if selected_game_counts.get(game_id, 0) >= int(max_markets_per_game):
                rejected_game_cap += 1
                continue

            involved_teams = [
                str(row.get(column, "")).strip()
                for column in ["home_team_abbr", "away_team_abbr", "yes_team_abbr"]
            ]
            involved_teams = sorted({team for team in involved_teams if team and team.lower() != "nan"})
            if any(selected_team_counts.get(team, 0) >= int(max_markets_per_team) for team in involved_teams):
                rejected_team_cap += 1
                continue

            shares = calculate_flat_fractional_shares(
                bankroll=bankroll_before_slate,
                price_cents=float(row["price_cents"]),
                max_bet_fraction=max_bet_fraction,
            )
            if shares < 1:
                continue
            contract_cost = float(row["contract_cost"])
            affordable_by_slate = int(np.floor(remaining_slate_budget / contract_cost))
            shares = min(shares, affordable_by_slate)
            if shares < 1:
                rejected_budget_cap += 1
                continue

            cost = shares * contract_cost
            actual_yes_win = str(row["actual_yes_win"]).lower() in ["true", "1", "yes"]
            payout = float(shares) if actual_yes_win else 0.0
            profit = payout - cost
            remaining_slate_budget -= cost
            selected_count += 1
            selected_game_counts[game_id] = selected_game_counts.get(game_id, 0) + 1
            for team in involved_teams:
                selected_team_counts[team] = selected_team_counts.get(team, 0) + 1
            slate_cost += cost
            slate_payout += payout
            selected_rows.append(
                {
                    "date": row["date"],
                    "game_id": row["game_id"],
                    "market_ticker": row["market_ticker"],
                    "home_team_abbr": row.get("home_team_abbr", ""),
                    "away_team_abbr": row.get("away_team_abbr", ""),
                    "yes_team_abbr": row.get("yes_team_abbr", ""),
                    "model_yes_prob": row["model_yes_prob"],
                    "market_prob": row.get("market_prob", np.nan),
                    "edge": row["edge"],
                    "expected_roi": row["expected_roi"],
                    "selection_expected_roi": row["selection_expected_roi"],
                    "price_cents": row["price_cents"],
                    "shares": shares,
                    "cost": cost,
                    "payout": payout,
                    "profit": profit,
                    "actual_yes_win": actual_yes_win,
                    "bankroll_before_slate": bankroll_before_slate,
                    "slate_date": pd.Timestamp(slate_date),
                    "involved_team_abbrs": ",".join(involved_teams),
                    "selection_reason": "selected_by_expected_roi_with_slate_caps",
                }
            )

        bankroll = bankroll - slate_cost + slate_payout
        slate_rows.append(
            {
                "date": pd.Timestamp(slate_date),
                "candidate_bets": int(len(slate)),
                "selected_trades": selected_count,
                "bankroll_before": bankroll_before_slate,
                "slate_cost": slate_cost,
                "slate_payout": slate_payout,
                "slate_profit": slate_payout - slate_cost,
                "bankroll_after": bankroll,
                "slate_cost_fraction": slate_cost / bankroll_before_slate if bankroll_before_slate else 0.0,
                "rejected_by_game_cap": rejected_game_cap,
                "rejected_by_team_cap": rejected_team_cap,
                "rejected_by_budget_cap": rejected_budget_cap,
            }
        )

    selected = pd.DataFrame(selected_rows)
    slate_summary = pd.DataFrame(slate_rows)
    if not selected.empty:
        selected = selected.merge(
            slate_summary[["date", "bankroll_after"]],
            on="date",
            how="left",
        )
    selected_start, selected_end = _date_range(selected["date"]) if not selected.empty else (None, None)
    amount_risked = float(selected["cost"].sum()) if not selected.empty else 0.0
    wins = selected[selected["profit"] > 0] if not selected.empty else pd.DataFrame()
    summary = {
        "starting_bankroll": float(starting_bankroll),
        "ending_bankroll": bankroll,
        "total_return_pct": (bankroll / float(starting_bankroll) - 1.0) if starting_bankroll else 0.0,
        "num_candidate_bets": int(len(candidates)),
        "num_selected_trades": int(len(selected)),
        "trade_start_date": selected_start,
        "trade_end_date": selected_end,
        "trade_timeline": _timeline_label(selected_start, selected_end),
        "num_slates": int(len(slate_summary)),
        "max_trades_per_slate": int(max_trades_per_slate),
        "max_slate_fraction": float(max_slate_fraction),
        "max_bet_fraction": float(max_bet_fraction),
        "max_markets_per_game": int(max_markets_per_game),
        "max_markets_per_team": int(max_markets_per_team),
        "min_edge": float(min_edge),
        "min_expected_roi": float(min_expected_roi),
        "trade_column": trade_column,
        "expected_roi_column": expected_roi_column or "expected_roi",
        "avg_selected_trades_per_slate": float(slate_summary["selected_trades"].mean()) if len(slate_summary) else 0.0,
        "avg_slate_cost_fraction": float(slate_summary["slate_cost_fraction"].mean()) if len(slate_summary) else 0.0,
        "rejected_by_game_cap": int(slate_summary["rejected_by_game_cap"].sum()) if len(slate_summary) else 0,
        "rejected_by_team_cap": int(slate_summary["rejected_by_team_cap"].sum()) if len(slate_summary) else 0,
        "rejected_by_budget_cap": int(slate_summary["rejected_by_budget_cap"].sum()) if len(slate_summary) else 0,
        "win_rate": float(len(wins) / len(selected)) if len(selected) else 0.0,
        "average_edge": float(selected["edge"].mean()) if len(selected) else 0.0,
        "average_profit_per_trade": float(selected["profit"].mean()) if len(selected) else 0.0,
        "amount_risked": amount_risked,
        "roi_on_amount_risked": float(selected["profit"].sum() / amount_risked) if amount_risked else 0.0,
        "max_drawdown": _max_drawdown(slate_summary["bankroll_after"]) if not slate_summary.empty else 0.0,
    }
    return selected.reset_index(drop=True), slate_summary.reset_index(drop=True), summary


def save_portfolio_outputs(
    selected_trades: pd.DataFrame,
    slate_summary: pd.DataFrame,
    summary: dict[str, Any],
    trades_path: str | Path,
    slates_path: str | Path,
    summary_path: str | Path,
) -> None:
    trades_output = Path(trades_path)
    slates_output = Path(slates_path)
    summary_output = Path(summary_path)
    trades_output.parent.mkdir(parents=True, exist_ok=True)
    slates_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    selected_trades.to_csv(trades_output, index=False)
    slate_summary.to_csv(slates_output, index=False)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
