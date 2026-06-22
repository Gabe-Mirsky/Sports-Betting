"""Research filters comparing player-aware and team-only market edges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.edge_calibration import _coerce_bool


POLICIES = [
    "player_all",
    "player_yes",
    "player_no",
    "both_trade",
    "same_side",
    "player_edge_higher",
    "player_edge_higher_yes",
    "player_edge_higher_no",
    "same_side_player_edge_higher",
    "same_side_player_edge_higher_yes",
    "same_side_player_edge_higher_no",
]


def _prepare(
    player_trades: pd.DataFrame,
    team_trades: pd.DataFrame,
    player_signal_column: str = "trade",
) -> pd.DataFrame:
    if player_trades.empty:
        return pd.DataFrame()
    required = [
        "date",
        "game_id",
        "market_ticker",
        player_signal_column,
        "candidate_side",
        "edge",
        "clv_cents",
    ]
    missing = [column for column in required if column not in player_trades.columns]
    if missing:
        raise ValueError(f"Player-aware trades are missing columns: {missing}")
    missing_team = [column for column in ["game_id", "market_ticker", "trade", "candidate_side", "edge"] if column not in team_trades.columns]
    if missing_team:
        raise ValueError(f"Team-only trades are missing columns: {missing_team}")

    player = player_trades.copy()
    team = team_trades.copy()
    player["game_id"] = player["game_id"].astype(str)
    team["game_id"] = team["game_id"].astype(str)
    player["market_ticker"] = player["market_ticker"].astype(str)
    team["market_ticker"] = team["market_ticker"].astype(str)
    keep_team = team[["game_id", "market_ticker", "trade", "candidate_side", "edge"]].rename(
        columns={
            "trade": "team_trade",
            "candidate_side": "team_candidate_side",
            "edge": "team_edge",
        }
    )
    merged = player.merge(keep_team, on=["game_id", "market_ticker"], how="left", validate="one_to_one")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    merged["trade"] = _coerce_bool(merged[player_signal_column])
    merged["team_trade"] = _coerce_bool(merged["team_trade"])
    merged["candidate_side"] = merged["candidate_side"].fillna("").astype(str).str.upper()
    merged["team_candidate_side"] = merged["team_candidate_side"].fillna("").astype(str).str.upper()
    if "profit" not in merged.columns and "realized_profit_per_share" in merged.columns:
        merged["profit"] = merged["realized_profit_per_share"]
    if "profit" not in merged.columns:
        merged["profit"] = 0.0
    for column in ["edge", "team_edge", "profit", "realized_profit_per_share", "clv_cents", "price_cents", "volume"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["same_side"] = merged["candidate_side"].eq(merged["team_candidate_side"])
    merged["edge_delta_vs_team"] = merged["edge"] - merged["team_edge"]
    merged["player_edge_higher"] = merged["edge_delta_vs_team"] > 0
    merged["positive_clv"] = merged["clv_cents"] > 0
    merged["profitable"] = merged["profit"] > 0
    return merged.reset_index(drop=True)


def _policy_mask(frame: pd.DataFrame, policy: str) -> pd.Series:
    base = frame["trade"].astype(bool)
    if policy == "player_all":
        return base
    if policy == "player_yes":
        return base & frame["candidate_side"].eq("YES")
    if policy == "player_no":
        return base & frame["candidate_side"].eq("NO")
    if policy == "both_trade":
        return base & frame["team_trade"].astype(bool)
    if policy == "same_side":
        return base & frame["same_side"].astype(bool)
    if policy == "player_edge_higher":
        return base & frame["player_edge_higher"].astype(bool)
    if policy == "player_edge_higher_yes":
        return base & frame["player_edge_higher"].astype(bool) & frame["candidate_side"].eq("YES")
    if policy == "player_edge_higher_no":
        return base & frame["player_edge_higher"].astype(bool) & frame["candidate_side"].eq("NO")
    if policy == "same_side_player_edge_higher":
        return base & frame["same_side"].astype(bool) & frame["player_edge_higher"].astype(bool)
    if policy == "same_side_player_edge_higher_yes":
        return (
            base
            & frame["same_side"].astype(bool)
            & frame["player_edge_higher"].astype(bool)
            & frame["candidate_side"].eq("YES")
        )
    if policy == "same_side_player_edge_higher_no":
        return (
            base
            & frame["same_side"].astype(bool)
            & frame["player_edge_higher"].astype(bool)
            & frame["candidate_side"].eq("NO")
        )
    raise ValueError(f"Unknown policy: {policy}")


def _summarize_subset(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    rows = frame[_policy_mask(frame, policy)].copy()
    if rows.empty:
        return {
            "policy": policy,
            "signals": 0,
            "months": 0,
            "avg_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "avg_profit": 0.0,
            "profit_rate": 0.0,
            "positive_month_share": 0.0,
            "status": "no_signals",
        }
    monthly = rows.groupby("month", dropna=False).agg(avg_clv_cents=("clv_cents", "mean"), signals=("clv_cents", "size"))
    positive_month_share = float((monthly["avg_clv_cents"] > 0).mean()) if not monthly.empty else 0.0
    avg_clv = float(rows["clv_cents"].mean())
    positive_clv_rate = float(rows["positive_clv"].mean())
    avg_profit = float(rows["profit"].mean())
    status = "watchlist"
    if len(rows) >= 100 and avg_clv > 0 and positive_clv_rate >= 0.45 and positive_month_share >= 0.60:
        status = "promising"
    if avg_clv <= 0 or positive_clv_rate < 0.35:
        status = "not_ready"
    return {
        "policy": policy,
        "signals": int(len(rows)),
        "months": int(rows["month"].nunique()),
        "avg_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv_rate,
        "avg_profit": avg_profit,
        "profit_rate": float(rows["profitable"].mean()),
        "positive_month_share": positive_month_share,
        "status": status,
    }


def build_player_edge_agreement_report(
    player_trades: pd.DataFrame,
    team_trades: pd.DataFrame,
    player_signal_column: str = "trade",
    min_train_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build descriptive and monthly walk-forward policy reports."""

    rows = _prepare(player_trades, team_trades, player_signal_column=player_signal_column)
    if rows.empty:
        summary = {
            "status": "no_rows",
            "signals": 0,
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
        }
        return rows, pd.DataFrame(), pd.DataFrame(), summary

    descriptive = pd.DataFrame([_summarize_subset(rows, policy) for policy in POLICIES]).sort_values(
        ["avg_clv_cents", "positive_clv_rate", "signals"],
        ascending=[False, False, False],
    )

    fold_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    months = sorted(rows["month"].dropna().unique())
    for month in months:
        train_months = [candidate for candidate in months if candidate < month]
        train = rows[rows["month"].isin(train_months)].copy()
        test = rows[rows["month"].eq(month)].copy()
        if len(train_months) < min_train_months or train.empty:
            fold_rows.append(
                {
                    "test_month": month,
                    "status": "skipped_insufficient_prior_months",
                    "train_months": len(train_months),
                    "selected_policy": "",
                    "signals": 0,
                }
            )
            continue
        train_summary = pd.DataFrame([_summarize_subset(train, policy) for policy in POLICIES])
        candidates = train_summary[train_summary["signals"] >= 10].copy()
        if candidates.empty:
            fold_rows.append(
                {
                    "test_month": month,
                    "status": "skipped_no_train_policy",
                    "train_months": len(train_months),
                    "selected_policy": "",
                    "signals": 0,
                }
            )
            continue
        selected = candidates.sort_values(["avg_clv_cents", "positive_clv_rate"], ascending=[False, False]).iloc[0]
        policy = str(selected["policy"])
        test_selected = test[_policy_mask(test, policy)].copy()
        if not test_selected.empty:
            selected_frames.append(test_selected)
        fold_rows.append(
            {
                "test_month": month,
                "status": "evaluated",
                "train_months": len(train_months),
                "selected_policy": policy,
                "train_avg_clv_cents": float(selected["avg_clv_cents"]),
                "train_positive_clv_rate": float(selected["positive_clv_rate"]),
                "signals": int(len(test_selected)),
                "avg_clv_cents": float(test_selected["clv_cents"].mean()) if not test_selected.empty else 0.0,
                "positive_clv_rate": float(test_selected["positive_clv"].mean()) if not test_selected.empty else 0.0,
                "avg_profit": float(test_selected["profit"].mean()) if not test_selected.empty else 0.0,
            }
        )
    folds = pd.DataFrame(fold_rows)
    selected_rows = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    if selected_rows.empty:
        summary = {
            "status": "not_ready",
            "signals": 0,
            "evaluated_months": int(folds["status"].eq("evaluated").sum()) if not folds.empty else 0,
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
            "descriptive_best_policy": str(descriptive.iloc[0]["policy"]) if not descriptive.empty else "",
            "note": "Research-only player/team agreement sweep. No walk-forward-selected signals.",
        }
    else:
        monthly = selected_rows.groupby("month").agg(avg_clv_cents=("clv_cents", "mean"), signals=("clv_cents", "size"))
        avg_clv = float(selected_rows["clv_cents"].mean())
        positive_clv_rate = float(selected_rows["positive_clv"].mean())
        positive_month_share = float((monthly["avg_clv_cents"] > 0).mean()) if not monthly.empty else 0.0
        status = "watchlist"
        if avg_clv <= 0 or positive_clv_rate < 0.35 or positive_month_share < 0.50:
            status = "not_ready"
        summary = {
            "status": status,
            "signals": int(len(selected_rows)),
            "evaluated_months": int(folds["status"].eq("evaluated").sum()) if not folds.empty else 0,
            "avg_clv_cents": avg_clv,
            "positive_clv_rate": positive_clv_rate,
            "avg_profit": float(selected_rows["profit"].mean()),
            "positive_month_share": positive_month_share,
            "descriptive_best_policy": str(descriptive.iloc[0]["policy"]) if not descriptive.empty else "",
            "single_game_edge_proven": False,
            "parlay_research_allowed": False,
            "note": "Research-only player/team agreement sweep. Do not use as betting logic unless proof gates pass.",
        }
    return rows, descriptive, folds, summary


def save_player_edge_agreement_report(
    rows: pd.DataFrame,
    descriptive: pd.DataFrame,
    folds: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "player_edge_agreement",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_root / f"{prefix}_rows.csv", index=False)
    descriptive.to_csv(output_root / f"{prefix}_descriptive.csv", index=False)
    folds.to_csv(output_root / f"{prefix}_walk_forward_folds.csv", index=False)
    (output_root / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
