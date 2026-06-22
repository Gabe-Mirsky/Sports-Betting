"""Fake-bankroll backtesting for matched Kalshi-style NBA markets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.sportsbook_odds import match_sportsbook_odds_to_games
from strategy.signal import add_two_sided_signals
from strategy.staking import calculate_flat_fractional_shares


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _date_range(values: pd.Series) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _range_label(start_date: str | None, end_date: str | None) -> str:
    if start_date and end_date:
        return start_date if start_date == end_date else f"{start_date} to {end_date}"
    return "n/a"


def _side_metrics(traded: pd.DataFrame, side: str) -> dict[str, Any]:
    side_trades = traded[traded["side"] == side].copy() if "side" in traded.columns else pd.DataFrame()
    if side_trades.empty:
        return {
            f"{side.lower()}_win_rate": 0.0,
            f"{side.lower()}_profit": 0.0,
            f"{side.lower()}_average_profit_per_trade": 0.0,
            f"{side.lower()}_average_clv_cents": 0.0,
            f"{side.lower()}_positive_clv_rate": 0.0,
        }

    output = {
        f"{side.lower()}_win_rate": float((side_trades["profit"] > 0).mean()),
        f"{side.lower()}_profit": float(side_trades["profit"].sum()),
        f"{side.lower()}_average_profit_per_trade": float(side_trades["profit"].mean()),
        f"{side.lower()}_average_clv_cents": 0.0,
        f"{side.lower()}_positive_clv_rate": 0.0,
    }
    if "clv_cents" in side_trades.columns:
        side_clv = pd.to_numeric(side_trades["clv_cents"], errors="coerce").dropna()
        if not side_clv.empty:
            output[f"{side.lower()}_average_clv_cents"] = float(side_clv.mean())
            output[f"{side.lower()}_positive_clv_rate"] = float((side_clv > 0).mean())
    return output


def summarize_backtest(trades: pd.DataFrame, starting_bankroll: float) -> dict[str, Any]:
    """Summarize paper-trading results."""

    traded = trades[trades["trade"]].copy()
    ending_bankroll = float(trades["bankroll_after"].iloc[-1]) if not trades.empty else starting_bankroll
    amount_risked = float(traded["cost"].sum()) if not traded.empty else 0.0
    wins = traded[traded["profit"] > 0]
    side_counts = traded["side"].value_counts().to_dict() if "side" in traded.columns else {}
    market_start_date, market_end_date = _date_range(trades["date"]) if "date" in trades.columns else (None, None)
    trade_start_date, trade_end_date = _date_range(traded["date"]) if "date" in traded.columns else (None, None)

    summary = {
        "starting_bankroll": float(starting_bankroll),
        "ending_bankroll": ending_bankroll,
        "total_return_pct": (ending_bankroll / float(starting_bankroll) - 1.0)
        if starting_bankroll
        else 0.0,
        "num_markets_seen": int(len(trades)),
        "num_trades": int(len(traded)),
        "num_yes_trades": int(side_counts.get("YES", 0)),
        "num_no_trades": int(side_counts.get("NO", 0)),
        "market_start_date": market_start_date,
        "market_end_date": market_end_date,
        "market_timeline": _range_label(market_start_date, market_end_date),
        "trade_start_date": trade_start_date,
        "trade_end_date": trade_end_date,
        "trade_timeline": _range_label(trade_start_date, trade_end_date),
        "win_rate": float(len(wins) / len(traded)) if len(traded) else 0.0,
        "average_edge": float(traded["edge"].mean()) if len(traded) else 0.0,
        "average_profit_per_trade": float(traded["profit"].mean()) if len(traded) else 0.0,
        "max_drawdown": _max_drawdown(trades["bankroll_after"]) if not trades.empty else 0.0,
        "largest_win": float(traded["profit"].max()) if len(traded) else 0.0,
        "largest_loss": float(traded["profit"].min()) if len(traded) else 0.0,
        "roi_on_amount_risked": float(traded["profit"].sum() / amount_risked)
        if amount_risked
        else 0.0,
        "average_clv_cents": float(traded["clv_cents"].mean())
        if len(traded) and "clv_cents" in traded.columns
        else 0.0,
        "median_clv_cents": float(traded["clv_cents"].median())
        if len(traded) and "clv_cents" in traded.columns
        else 0.0,
        "positive_clv_rate": float((traded["clv_cents"] > 0).mean())
        if len(traded) and "clv_cents" in traded.columns
        else 0.0,
    }
    summary.update(_side_metrics(traded, "YES"))
    summary.update(_side_metrics(traded, "NO"))
    return summary


def run_backtest(
    matched_markets_df: pd.DataFrame,
    starting_bankroll: float = 100.0,
    edge_threshold: float = 0.05,
    max_bet_fraction: float = 0.03,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
    allow_no_trades: bool = True,
) -> pd.DataFrame:
    """Run a simple same-day-settlement fake-bankroll backtest."""

    required = [
        "game_date",
        "game_id",
        "market_ticker",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "model_yes_prob",
        "actual_yes_win",
    ]
    missing = [column for column in required if column not in matched_markets_df.columns]
    if missing:
        raise ValueError(f"Matched markets are missing columns: {missing}")

    markets = matched_markets_df.copy()
    markets["game_date"] = pd.to_datetime(markets["game_date"], errors="coerce")
    if "yes_ask" not in markets.columns:
        markets["yes_ask"] = markets.get("yes_mid_cents")
    if "yes_bid" not in markets.columns:
        markets["yes_bid"] = pd.NA
    markets = add_two_sided_signals(
        markets,
        edge_threshold=edge_threshold,
        min_market_price=min_market_price,
        max_market_price=max_market_price,
        allow_no=allow_no_trades,
    )
    markets = markets.sort_values(["game_date", "game_id", "market_ticker"]).reset_index(drop=True)

    bankroll = float(starting_bankroll)
    rows: list[dict[str, Any]] = []

    for _, row in markets.iterrows():
        bankroll_before = bankroll
        shares = 0
        cost = 0.0
        payout = 0.0
        profit = 0.0

        if bool(row["trade"]):
            shares = calculate_flat_fractional_shares(
                bankroll=bankroll,
                price_cents=row["price_cents"],
                max_bet_fraction=max_bet_fraction,
            )
            if shares < 1:
                trade = False
                reason = "insufficient_bankroll_for_one_share"
            else:
                trade = True
                reason = str(row["reason"])
                contract_cost = float(row["price_cents"]) / 100.0
                cost = shares * contract_cost
                side_won = bool(row["actual_yes_win"]) if row["side"] == "YES" else not bool(row["actual_yes_win"])
                payout = float(shares) if side_won else 0.0
                profit = payout - cost
                bankroll = bankroll - cost + payout
        else:
            trade = False
            reason = str(row["reason"])

        clv_reference_price = row.get("clv_reference_price_cents", np.nan)
        clv_reference_snapshot = row.get("clv_reference_snapshot", "")
        candidate_side = str(row.get("candidate_side", row.get("side", ""))).upper()
        if candidate_side == "NO":
            clv_reference_price = row.get("clv_reference_no_price_cents", np.nan)
            clv_reference_snapshot = row.get("clv_reference_no_snapshot", clv_reference_snapshot)
        clv_cents = (
            float(clv_reference_price) - float(row["price_cents"])
            if pd.notna(clv_reference_price) and pd.notna(row["price_cents"])
            else np.nan
        )

        rows.append(
            {
                "date": row["game_date"],
                "game_id": row["game_id"],
                "season": row.get("season", np.nan),
                "season_type": row.get("season_type", ""),
                "market_ticker": row["market_ticker"],
                "home_team_abbr": row["home_team_abbr"],
                "away_team_abbr": row["away_team_abbr"],
                "yes_team_abbr": row["yes_team_abbr"],
                "model_yes_prob": row["model_yes_prob"],
                "model_prob": row["model_prob"],
                "market_prob": row["market_prob"],
                "edge": row["edge"],
                "price_cents": row["price_cents"],
                "trade": trade,
                "side": row["side"] if trade else "",
                "candidate_side": candidate_side,
                "shares": shares,
                "cost": cost,
                "payout": payout,
                "profit": profit,
                "bankroll_before": bankroll_before,
                "bankroll_after": bankroll,
                "actual_yes_win": bool(row["actual_yes_win"]),
                "reason": reason,
                "snapshot_target": row.get("snapshot_target", ""),
                "price_quality": row.get("price_quality", ""),
                "period_interval": row.get("period_interval", np.nan),
                "volume": row.get("volume", np.nan),
                "open_interest": row.get("open_interest", np.nan),
                "clv_reference_price_cents": clv_reference_price,
                "clv_reference_snapshot": clv_reference_snapshot,
                "clv_cents": clv_cents,
            }
        )

    return pd.DataFrame(rows)


def prepare_candlestick_backtest_markets(
    predictions_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    min_volume: float = 0.0,
    allowed_price_qualities: list[str] | tuple[str, ...] | set[str] | None = ("bid_ask_available",),
    require_bid_ask: bool = True,
    max_candle_interval_minutes: int | None = 60,
    max_bid_ask_spread_cents: float | None = 10.0,
    preferred_snapshot_targets: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build run_backtest input from Kalshi matches and extracted candle snapshots."""

    predictions = predictions_df.copy()
    matches = matches_df.copy()
    prices = prices_df.copy()

    predictions["game_id"] = predictions["game_id"].astype(str)
    matches["game_id"] = matches["game_id"].astype(str)
    prices["game_id"] = prices["game_id"].astype(str)

    auto_matches = matches[matches["match_status"] == "auto_matched"].copy()
    review_matches = matches[matches["match_status"] == "needs_review"].copy()
    initial_price_rows = int(len(prices))
    usable_prices = prices[prices["price_quality"] != "missing"].copy()
    non_missing_price_rows = int(len(usable_prices))
    if allowed_price_qualities is not None and "price_quality" in usable_prices.columns:
        allowed = {str(item) for item in allowed_price_qualities}
        usable_prices = usable_prices[usable_prices["price_quality"].astype(str).isin(allowed)].copy()
    after_quality_rows = int(len(usable_prices))

    if max_candle_interval_minutes is not None and "period_interval" in usable_prices.columns:
        usable_prices["period_interval"] = pd.to_numeric(usable_prices["period_interval"], errors="coerce")
        usable_prices = usable_prices[
            usable_prices["period_interval"].notna()
            & (usable_prices["period_interval"] <= int(max_candle_interval_minutes))
        ].copy()
    after_interval_rows = int(len(usable_prices))

    if require_bid_ask:
        for column in ["yes_bid", "yes_ask"]:
            if column not in usable_prices.columns:
                usable_prices[column] = np.nan
            usable_prices[column] = pd.to_numeric(usable_prices[column], errors="coerce")
        usable_prices = usable_prices[
            usable_prices["yes_bid"].notna()
            & usable_prices["yes_ask"].notna()
            & (usable_prices["yes_ask"] >= usable_prices["yes_bid"])
        ].copy()
    after_bid_ask_rows = int(len(usable_prices))

    if max_bid_ask_spread_cents is not None:
        for column in ["yes_bid", "yes_ask"]:
            if column not in usable_prices.columns:
                usable_prices[column] = np.nan
            usable_prices[column] = pd.to_numeric(usable_prices[column], errors="coerce")
        spread = usable_prices["yes_ask"] - usable_prices["yes_bid"]
        usable_prices = usable_prices[spread <= float(max_bid_ask_spread_cents)].copy()
    after_spread_rows = int(len(usable_prices))

    if min_volume > 0 and "volume" in usable_prices.columns:
        usable_prices["volume"] = pd.to_numeric(usable_prices["volume"], errors="coerce").fillna(0)
        usable_prices = usable_prices[usable_prices["volume"] >= float(min_volume)]
    after_volume_rows = int(len(usable_prices))

    target_order = list(preferred_snapshot_targets or ["pregame_60m", "pregame_30m", "pregame_5m"])
    priority = {target: index for index, target in enumerate(target_order)}
    usable_prices["snapshot_priority"] = usable_prices["snapshot_target"].map(priority).fillna(99)
    usable_prices["snapshot_ts"] = pd.to_numeric(usable_prices["snapshot_ts"], errors="coerce")
    chosen_prices = (
        usable_prices.sort_values(["game_id", "market_ticker", "snapshot_priority", "snapshot_ts"])
        .drop_duplicates(subset=["game_id", "market_ticker"], keep="first")
        .copy()
    )

    snapshot_prices = usable_prices.pivot_table(
        index=["game_id", "market_ticker", "series_ticker"],
        columns="snapshot_target",
        values="yes_price",
        aggfunc="first",
    ).reset_index()
    snapshot_prices = snapshot_prices.rename(
        columns={
            "pregame_60m": "pregame_price_60m",
            "pregame_30m": "pregame_price_30m",
            "pregame_5m": "pregame_price_5m",
        }
    )
    snapshot_bids = usable_prices.pivot_table(
        index=["game_id", "market_ticker", "series_ticker"],
        columns="snapshot_target",
        values="yes_bid",
        aggfunc="first",
    ).reset_index()
    snapshot_bids = snapshot_bids.rename(
        columns={
            "pregame_60m": "pregame_yes_bid_60m",
            "pregame_30m": "pregame_yes_bid_30m",
            "pregame_5m": "pregame_yes_bid_5m",
        }
    )

    matched = auto_matches.merge(
        chosen_prices,
        on=["game_id", "market_ticker", "series_ticker"],
        how="left",
        suffixes=("", "_price"),
    )
    if not snapshot_prices.empty:
        matched = matched.merge(
            snapshot_prices,
            on=["game_id", "market_ticker", "series_ticker"],
            how="left",
        )
    if not snapshot_bids.empty:
        matched = matched.merge(
            snapshot_bids,
            on=["game_id", "market_ticker", "series_ticker"],
            how="left",
        )
    backtest_rows = matched.merge(
        predictions,
        on=["game_id", "game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
        suffixes=("", "_prediction"),
    )
    backtest_rows["yes_price"] = pd.to_numeric(backtest_rows["yes_price"], errors="coerce")
    backtest_rows = backtest_rows.dropna(subset=["model_home_win_prob", "model_away_win_prob", "yes_price"]).copy()

    if backtest_rows.empty:
        diagnostics = {
            "mode": "kalshi_candlestick",
            "games_available": int(len(predictions)),
            "games_with_matched_kalshi_market": int(len(auto_matches)),
            "games_with_usable_pregame_price": 0,
            "skipped_games_due_to_no_market": int(max(len(predictions) - len(auto_matches), 0)),
            "skipped_games_due_to_no_price": int(len(auto_matches)),
            "number_of_auto_matched_markets": int(len(auto_matches)),
            "number_of_needs_review_markets": int(len(review_matches)),
            "price_rows_seen": initial_price_rows,
            "price_rows_non_missing": non_missing_price_rows,
            "price_rows_after_quality_filter": after_quality_rows,
            "price_rows_after_interval_filter": after_interval_rows,
            "price_rows_after_bid_ask_filter": after_bid_ask_rows,
            "price_rows_after_spread_filter": after_spread_rows,
            "price_rows_after_volume_filter": after_volume_rows,
            "allowed_price_qualities": sorted(allowed_price_qualities) if allowed_price_qualities else "any_non_missing",
            "require_bid_ask": bool(require_bid_ask),
            "min_volume": float(min_volume),
            "max_candle_interval_minutes": max_candle_interval_minutes,
            "max_bid_ask_spread_cents": max_bid_ask_spread_cents,
            "preferred_snapshot_targets": target_order,
        }
        return pd.DataFrame(), diagnostics

    backtest_rows["model_yes_prob"] = np.where(
        backtest_rows["yes_team_abbr"] == backtest_rows["home_team_abbr"],
        backtest_rows["model_home_win_prob"],
        backtest_rows["model_away_win_prob"],
    )
    outcome_column = "actual_home_win" if "actual_home_win" in backtest_rows.columns else "home_win"
    backtest_rows[outcome_column] = pd.to_numeric(backtest_rows[outcome_column], errors="coerce")
    backtest_rows = backtest_rows.dropna(subset=[outcome_column]).copy()
    actual_home_win = backtest_rows[outcome_column].astype(bool)
    backtest_rows["actual_yes_win"] = np.where(
        backtest_rows["yes_team_abbr"] == backtest_rows["home_team_abbr"],
        actual_home_win,
        ~actual_home_win,
    )
    backtest_rows["yes_mid_cents"] = pd.to_numeric(backtest_rows["yes_price"], errors="coerce")
    backtest_rows["price_source"] = backtest_rows["price_quality"]
    for column in ["pregame_price_60m", "pregame_price_30m", "pregame_price_5m"]:
        if column not in backtest_rows.columns:
            backtest_rows[column] = np.nan
        backtest_rows[column] = pd.to_numeric(backtest_rows[column], errors="coerce")
    for column in ["pregame_yes_bid_60m", "pregame_yes_bid_30m", "pregame_yes_bid_5m"]:
        if column not in backtest_rows.columns:
            backtest_rows[column] = np.nan
        backtest_rows[column] = pd.to_numeric(backtest_rows[column], errors="coerce")
    backtest_rows["clv_reference_price_cents"] = np.where(
        backtest_rows["snapshot_target"].eq("pregame_5m"),
        np.nan,
        backtest_rows["pregame_price_5m"],
    )
    missing_5m = backtest_rows["clv_reference_price_cents"].isna() & ~backtest_rows["snapshot_target"].eq("pregame_30m")
    backtest_rows.loc[missing_5m, "clv_reference_price_cents"] = backtest_rows.loc[missing_5m, "pregame_price_30m"]
    backtest_rows["clv_reference_snapshot"] = np.where(
        backtest_rows["clv_reference_price_cents"].eq(backtest_rows["pregame_price_5m"]),
        "pregame_5m",
        np.where(backtest_rows["clv_reference_price_cents"].notna(), "pregame_30m", ""),
    )
    backtest_rows["clv_reference_yes_bid_cents"] = np.where(
        backtest_rows["clv_reference_snapshot"].eq("pregame_5m"),
        backtest_rows["pregame_yes_bid_5m"],
        np.where(
            backtest_rows["clv_reference_snapshot"].eq("pregame_30m"),
            backtest_rows["pregame_yes_bid_30m"],
            np.nan,
        ),
    )
    backtest_rows["clv_reference_no_price_cents"] = 100.0 - backtest_rows["clv_reference_yes_bid_cents"]
    backtest_rows["clv_reference_no_snapshot"] = backtest_rows["clv_reference_snapshot"]
    backtest_rows["clv_cents"] = backtest_rows["clv_reference_price_cents"] - backtest_rows["yes_mid_cents"]
    if "season_type" in backtest_rows.columns:
        backtest_rows["is_playoffs"] = backtest_rows["season_type"].astype(str).str.contains("Playoffs", case=False, na=False)
    elif "is_playoffs" not in backtest_rows.columns:
        backtest_rows["is_playoffs"] = False

    output_columns = [
        "game_date",
        "game_id",
        "season",
        "season_type",
        "is_playoffs",
        "market_ticker",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "model_yes_prob",
        "yes_mid_cents",
        "yes_bid",
        "yes_ask",
        "actual_yes_win",
        "snapshot_target",
        "snapshot_ts",
        "minutes_before_tipoff",
        "price_quality",
        "period_interval",
        "volume",
        "open_interest",
        "pregame_price_60m",
        "pregame_price_30m",
        "pregame_price_5m",
        "clv_reference_price_cents",
        "clv_reference_snapshot",
        "clv_reference_no_price_cents",
        "clv_reference_no_snapshot",
        "clv_cents",
    ]
    output = backtest_rows[[column for column in output_columns if column in backtest_rows.columns]].copy()
    diagnostics = {
        "mode": "kalshi_candlestick",
        "games_available": int(len(predictions)),
        "games_with_matched_kalshi_market": int(len(auto_matches)),
        "games_with_usable_pregame_price": int(output["game_id"].nunique()),
        "skipped_games_due_to_no_market": int(max(len(predictions) - len(auto_matches), 0)),
        "skipped_games_due_to_no_price": int(max(len(auto_matches) - output["game_id"].nunique(), 0)),
        "number_of_auto_matched_markets": int(len(auto_matches)),
        "number_of_needs_review_markets": int(len(review_matches)),
        "price_rows_seen": initial_price_rows,
        "price_rows_non_missing": non_missing_price_rows,
        "price_rows_after_quality_filter": after_quality_rows,
        "price_rows_after_interval_filter": after_interval_rows,
        "price_rows_after_bid_ask_filter": after_bid_ask_rows,
        "price_rows_after_spread_filter": after_spread_rows,
        "price_rows_after_volume_filter": after_volume_rows,
        "allowed_price_qualities": sorted(allowed_price_qualities) if allowed_price_qualities else "any_non_missing",
        "require_bid_ask": bool(require_bid_ask),
        "min_volume": float(min_volume),
        "max_candle_interval_minutes": max_candle_interval_minutes,
        "max_bid_ask_spread_cents": max_bid_ask_spread_cents,
        "preferred_snapshot_targets": target_order,
    }
    return output.reset_index(drop=True), diagnostics


def prepare_sportsbook_backtest_markets(
    predictions_df: pd.DataFrame,
    sportsbook_odds_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build run_backtest input from historical sportsbook no-vig moneyline odds."""

    predictions = predictions_df.copy()
    matched = match_sportsbook_odds_to_games(predictions, sportsbook_odds_df)
    if matched.empty:
        return pd.DataFrame(), {
            "mode": "sportsbook_market_proxy",
            "games_available": int(len(predictions)),
            "games_with_sportsbook_odds": 0,
            "skipped_games_due_to_no_sportsbook_odds": int(len(predictions)),
        }

    required = ["home_no_vig_prob", "away_no_vig_prob", "model_home_win_prob", "model_away_win_prob"]
    for column in required:
        if column not in matched.columns:
            matched[column] = np.nan
        matched[column] = pd.to_numeric(matched[column], errors="coerce")
    matched = matched.dropna(subset=required).copy()
    if "actual_home_win" not in matched.columns and "home_win" in matched.columns:
        matched["actual_home_win"] = matched["home_win"]
    matched["actual_home_win"] = pd.to_numeric(matched.get("actual_home_win"), errors="coerce")
    matched = matched.dropna(subset=["actual_home_win"]).copy()

    if matched.empty:
        return pd.DataFrame(), {
            "mode": "sportsbook_market_proxy",
            "games_available": int(len(predictions)),
            "games_with_sportsbook_odds": 0,
            "skipped_games_due_to_no_sportsbook_odds": int(len(predictions)),
        }

    output = pd.DataFrame(
        {
            "game_date": matched["game_date"],
            "game_id": matched["game_id"],
            "season": matched.get("season", np.nan),
            "season_type": matched.get("season_type", ""),
            "market_ticker": "SPORTSBOOK-" + matched["game_id"].astype(str) + "-HOME",
            "home_team_abbr": matched["home_team_abbr"],
            "away_team_abbr": matched["away_team_abbr"],
            "yes_team_abbr": matched["home_team_abbr"],
            "model_yes_prob": matched["model_home_win_prob"],
            "yes_mid_cents": matched["home_no_vig_prob"] * 100.0,
            "yes_bid": matched["home_no_vig_prob"] * 100.0,
            "yes_ask": matched["home_no_vig_prob"] * 100.0,
            "actual_yes_win": matched["actual_home_win"].astype(bool),
            "price_quality": "sportsbook_no_vig",
            "market_source": "sportsbook",
            "sportsbook_name": matched.get("sportsbook_name", ""),
            "home_moneyline": matched.get("home_moneyline", np.nan),
            "away_moneyline": matched.get("away_moneyline", np.nan),
            "home_implied_prob": matched.get("home_implied_prob", np.nan),
            "away_implied_prob": matched.get("away_implied_prob", np.nan),
            "home_no_vig_prob": matched["home_no_vig_prob"],
            "away_no_vig_prob": matched["away_no_vig_prob"],
            "spread": matched.get("spread", np.nan),
            "total": matched.get("total", np.nan),
        }
    )
    diagnostics = {
        "mode": "sportsbook_market_proxy",
        "games_available": int(len(predictions)),
        "games_with_sportsbook_odds": int(output["game_id"].nunique()),
        "skipped_games_due_to_no_sportsbook_odds": int(max(len(predictions) - output["game_id"].nunique(), 0)),
        "market_probability_source": "sportsbook_no_vig_moneyline",
    }
    if "dataset_split" in predictions.columns:
        diagnostics["skipped_games_by_dataset_split"] = {
            str(split): int(count)
            for split, count in predictions[
                ~predictions["game_id"].astype(str).isin(output["game_id"].astype(str))
            ].groupby("dataset_split")["game_id"].nunique().items()
        }
    return output.reset_index(drop=True), diagnostics


def save_backtest_outputs(
    trades: pd.DataFrame,
    summary: dict[str, Any],
    trades_path: str | Path,
    summary_path: str | Path,
) -> None:
    """Save backtest trades and summary."""

    trades_path = Path(trades_path)
    summary_path = Path(summary_path)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
