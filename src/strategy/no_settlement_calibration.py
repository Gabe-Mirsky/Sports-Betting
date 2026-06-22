"""NO-side settlement calibration and suppression research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.staking import calculate_flat_fractional_shares


PRICE_BINS = [0, 10, 20, 30, 40, 55, 70, 85, 100]
PRICE_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-55", "55-70", "70-85", "85-100"]
PROB_BINS = [0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
PROB_LABELS = ["0-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-100%"]
EDGE_BINS = [-np.inf, 0, 0.02, 0.05, 0.08, 0.12, np.inf]
EDGE_LABELS = ["<=0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"]
CLV_BINS = [-np.inf, -10, -2, 0, 2, 10, 25, np.inf]
CLV_LABELS = ["<-10c", "-10--2c", "-2-0c", "0-2c", "2-10c", "10-25c", "25c+"]
LIQUIDITY_BINS = [-np.inf, 10, 100, 1000, 10000, np.inf]
LIQUIDITY_LABELS = ["<10", "10-100", "100-1k", "1k-10k", "10k+"]


@dataclass(frozen=True)
class SuppressionRule:
    group_columns: tuple[str, ...]
    min_prior_rows: int
    min_prior_profit_per_share: float
    min_prior_avg_clv_cents: float
    min_prior_positive_clv_rate: float

    @property
    def name(self) -> str:
        group = "+".join(self.group_columns) if self.group_columns else "overall"
        return (
            f"{group}|rows>={self.min_prior_rows}|profit>={self.min_prior_profit_per_share:.3f}|"
            f"clv>={self.min_prior_avg_clv_cents:.2f}|pos_clv>={self.min_prior_positive_clv_rate:.2f}"
        )


def default_suppression_rules() -> list[SuppressionRule]:
    groups = [
        tuple(),
        ("price_bucket",),
        ("edge_bucket",),
        ("liquidity_bucket",),
        ("price_bucket", "edge_bucket"),
        ("price_bucket", "liquidity_bucket"),
        ("yes_team_abbr",),
    ]
    rules: list[SuppressionRule] = []
    for group in groups:
        for rows in [10, 20, 50]:
            for profit in [-0.02, 0.0, 0.02]:
                for clv in [-0.10, 0.0, 0.10]:
                    for pos_clv in [0.20, 0.30, 0.40]:
                        rules.append(SuppressionRule(group, rows, profit, clv, pos_clv))
    return rules


def break_even_probability(no_buy_price_cents: float) -> float:
    """Return the settlement win rate needed to break even on a NO buy."""

    return float(no_buy_price_cents) / 100.0


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    bucket = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    return bucket.astype("object").where(bucket.notna(), "unknown").astype(str)


def prepare_no_settlement_rows(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize canonical backtest NO trades for settlement calibration."""

    required = ["date", "trade", "side", "price_cents", "model_prob", "market_prob", "actual_yes_win"]
    missing = [column for column in required if column not in trades.columns]
    if missing:
        raise ValueError(f"NO settlement rows are missing columns: {missing}")

    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["trade_bool"] = _bool_series(frame["trade"])
    frame["side_norm"] = frame["side"].fillna("").astype(str).str.upper()
    no_rows = frame[frame["trade_bool"] & frame["side_norm"].eq("NO") & frame["date"].notna()].copy()
    for column in [
        "price_cents",
        "model_prob",
        "market_prob",
        "edge",
        "shares",
        "cost",
        "profit",
        "clv_cents",
        "volume",
        "open_interest",
        "period_interval",
        "bankroll_after",
    ]:
        if column in no_rows.columns:
            no_rows[column] = pd.to_numeric(no_rows[column], errors="coerce")

    no_rows["actual_no_win"] = ~_bool_series(no_rows["actual_yes_win"])
    no_rows["break_even_probability"] = no_rows["price_cents"] / 100.0
    no_rows["predicted_no_probability"] = no_rows["model_prob"]
    no_rows["market_no_probability"] = no_rows["market_prob"]
    no_rows["settlement_edge"] = no_rows["actual_no_win"].astype(float) - no_rows["break_even_probability"]
    no_rows["calibration_error"] = no_rows["actual_no_win"].astype(float) - no_rows["predicted_no_probability"]
    no_rows["profit_per_share"] = np.where(
        no_rows["actual_no_win"],
        1.0 - no_rows["break_even_probability"],
        -no_rows["break_even_probability"],
    )
    if "shares" in no_rows.columns and "profit" in no_rows.columns:
        actual_pps = no_rows["profit"] / no_rows["shares"].replace(0, np.nan)
        no_rows["profit_per_share"] = actual_pps.fillna(no_rows["profit_per_share"])
    no_rows["positive_clv"] = no_rows["clv_cents"] > 0
    no_rows["profitable"] = no_rows["profit"] > 0
    no_rows["positive_clv_loss"] = no_rows["positive_clv"] & ~no_rows["actual_no_win"]
    no_rows["expensive_miss"] = ~no_rows["actual_no_win"] & no_rows["price_cents"].ge(40)
    no_rows["month"] = no_rows["date"].dt.to_period("M").astype(str)
    no_rows["probability_bucket"] = _bucket(no_rows["predicted_no_probability"], PROB_BINS, PROB_LABELS)
    no_rows["price_bucket"] = _bucket(no_rows["price_cents"], PRICE_BINS, PRICE_LABELS)
    no_rows["edge_bucket"] = _bucket(no_rows.get("edge", pd.Series(np.nan, index=no_rows.index)), EDGE_BINS, EDGE_LABELS)
    no_rows["clv_bucket"] = _bucket(no_rows.get("clv_cents", pd.Series(np.nan, index=no_rows.index)), CLV_BINS, CLV_LABELS)
    no_rows["liquidity_bucket"] = _bucket(
        no_rows.get("volume", pd.Series(np.nan, index=no_rows.index)),
        LIQUIDITY_BINS,
        LIQUIDITY_LABELS,
    )
    if "yes_team_abbr" not in no_rows.columns:
        no_rows["yes_team_abbr"] = "unknown"
    return no_rows.sort_values(["date", "game_id", "market_ticker"]).reset_index(drop=True)


def summarize_segments(rows: pd.DataFrame, group_columns: list[str], min_rows: int = 1) -> pd.DataFrame:
    """Summarize settlement, CLV, and break-even economics by segment."""

    output_columns = group_columns + [
        "rows",
        "win_rate",
        "avg_predicted_no_probability",
        "avg_break_even_probability",
        "calibration_error",
        "break_even_edge",
        "profit",
        "avg_profit_per_share",
        "roi_on_amount_risked",
        "avg_clv_cents",
        "positive_clv_rate",
        "positive_clv_loss_count",
        "positive_clv_loss_rate",
        "expensive_miss_count",
        "avg_price_cents",
        "avg_edge",
        "avg_volume",
    ]
    if rows.empty:
        return pd.DataFrame(columns=output_columns)
    groups = [((), rows)] if not group_columns else rows.groupby(group_columns, dropna=False, observed=False)
    records: list[dict[str, Any]] = []
    for key, group in groups:
        if len(group) < min_rows:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        risked = float(group["cost"].sum()) if "cost" in group.columns else float(group["price_cents"].sum() / 100.0)
        win_rate = float(group["actual_no_win"].mean())
        avg_pred = float(group["predicted_no_probability"].mean())
        avg_break_even = float(group["break_even_probability"].mean())
        row = {column: str(value) for column, value in zip(group_columns, keys)}
        row.update(
            {
                "rows": int(len(group)),
                "win_rate": win_rate,
                "avg_predicted_no_probability": avg_pred,
                "avg_break_even_probability": avg_break_even,
                "calibration_error": win_rate - avg_pred,
                "break_even_edge": win_rate - avg_break_even,
                "profit": float(group["profit"].sum()),
                "avg_profit_per_share": float(group["profit_per_share"].mean()),
                "roi_on_amount_risked": float(group["profit"].sum() / risked) if risked else 0.0,
                "avg_clv_cents": float(group["clv_cents"].mean()),
                "positive_clv_rate": float(group["positive_clv"].mean()),
                "positive_clv_loss_count": int(group["positive_clv_loss"].sum()),
                "positive_clv_loss_rate": float(group["positive_clv_loss"].mean()),
                "expensive_miss_count": int(group["expensive_miss"].sum()),
                "avg_price_cents": float(group["price_cents"].mean()),
                "avg_edge": float(group["edge"].mean()) if "edge" in group.columns else np.nan,
                "avg_volume": float(group["volume"].mean()) if "volume" in group.columns else np.nan,
            }
        )
        records.append(row)
    if not records:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(records, columns=output_columns)


def clv_vs_profit(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether NO CLV buckets turned into realized settlement profit."""

    return summarize_segments(rows, ["clv_bucket"], min_rows=1).sort_values("clv_bucket").reset_index(drop=True)


def failure_segments(rows: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    """Rank NO settlement failure segments."""

    frames: list[pd.DataFrame] = []
    for columns in [
        ["probability_bucket"],
        ["price_bucket"],
        ["clv_bucket"],
        ["edge_bucket"],
        ["liquidity_bucket"],
        ["month"],
        ["yes_team_abbr"],
        ["price_bucket", "edge_bucket"],
        ["price_bucket", "liquidity_bucket"],
    ]:
        if not all(column in rows.columns for column in columns):
            continue
        summary = summarize_segments(rows, columns, min_rows=min_rows)
        if summary.empty:
            continue
        summary.insert(0, "segment", "+".join(columns))
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["failure_score"] = (
        np.maximum(0.0, -pd.to_numeric(combined["profit"], errors="coerce").fillna(0.0) / 25.0)
        + np.maximum(0.0, -pd.to_numeric(combined["break_even_edge"], errors="coerce").fillna(0.0))
        + np.maximum(0.0, -pd.to_numeric(combined["calibration_error"], errors="coerce").fillna(0.0))
        + pd.to_numeric(combined["positive_clv_loss_rate"], errors="coerce").fillna(0.0)
        + np.minimum(pd.to_numeric(combined["rows"], errors="coerce").fillna(0.0), 100.0) / 500.0
    )
    combined["recommended_fix"] = np.select(
        [
            combined["segment"].astype(str).str.contains("price_bucket", regex=False),
            combined["segment"].astype(str).str.contains("liquidity_bucket", regex=False),
            combined["segment"].astype(str).str.contains("month", regex=False),
            combined["segment"].astype(str).str.contains("yes_team_abbr", regex=False),
        ],
        [
            "Test prior-period price bucket suppression; do not use full-sample buckets as rules.",
            "Require stronger liquidity history or lower sizing in weak-liquidity buckets.",
            "Investigate regime drift and availability inputs before adding calendar filters.",
            "Treat team concentration as diagnostics only unless it survives prior-period validation.",
        ],
        default="Improve NO calibration before treating CLV as settlement edge.",
    )
    return combined.sort_values(["failure_score", "rows"], ascending=[False, False]).reset_index(drop=True)


def _rule_allows_group(prior: pd.DataFrame, rule: SuppressionRule, row: pd.Series) -> tuple[bool, int]:
    if rule.group_columns:
        mask = pd.Series(True, index=prior.index)
        for column in rule.group_columns:
            mask &= prior[column].astype(str).eq(str(row[column]))
        history = prior[mask]
    else:
        history = prior
    if len(history) < rule.min_prior_rows:
        return False, int(len(history))
    avg_profit = float(history["profit_per_share"].mean())
    avg_clv = float(history["clv_cents"].mean())
    pos_clv = float(history["positive_clv"].mean())
    allowed = (
        avg_profit >= rule.min_prior_profit_per_share
        and avg_clv >= rule.min_prior_avg_clv_cents
        and pos_clv >= rule.min_prior_positive_clv_rate
    )
    return bool(allowed), int(len(history))


def apply_prior_suppression_rule(rows: pd.DataFrame, rule: SuppressionRule) -> pd.DataFrame:
    """Apply a NO suppression rule using only earlier rows for each trade."""

    ordered = rows.sort_values(["date", "game_id", "market_ticker"]).reset_index(drop=True)
    selected: list[dict[str, Any]] = []
    for index, row in ordered.iterrows():
        prior = ordered.iloc[:index]
        allowed, prior_rows = _rule_allows_group(prior, rule, row)
        item = row.to_dict()
        item["suppression_rule"] = rule.name
        item["prior_rows"] = prior_rows
        item["selected_by_rule"] = allowed
        selected.append(item)
    return pd.DataFrame(selected)


def simulate_rule_bankroll(rows: pd.DataFrame, starting_bankroll: float = 100.0, max_bet_fraction: float = 0.03) -> pd.DataFrame:
    """Re-simulate selected NO trades after suppression."""

    bankroll = float(starting_bankroll)
    records: list[dict[str, Any]] = []
    selected = rows[rows["selected_by_rule"].astype(bool)].copy()
    for _, row in selected.sort_values(["date", "game_id", "market_ticker"]).iterrows():
        bankroll_before = bankroll
        shares = calculate_flat_fractional_shares(bankroll, float(row["price_cents"]), max_bet_fraction=max_bet_fraction)
        cost = shares * float(row["price_cents"]) / 100.0
        payout = float(shares) if bool(row["actual_no_win"]) else 0.0
        profit = payout - cost if shares else 0.0
        bankroll = bankroll - cost + payout
        item = row.to_dict()
        item.update({"shares": shares, "cost": cost, "profit": profit, "bankroll_before": bankroll_before, "bankroll_after": bankroll})
        records.append(item)
    return pd.DataFrame(records)


def _max_drawdown(equity: pd.Series, starting_bankroll: float) -> float:
    if equity.empty:
        return 0.0
    full = pd.concat([pd.Series([starting_bankroll]), pd.to_numeric(equity, errors="coerce")], ignore_index=True)
    running = full.cummax()
    return float(((full - running) / running).min())


def summarize_rule_trades(trades: pd.DataFrame, rule: SuppressionRule, starting_bankroll: float = 100.0) -> dict[str, Any]:
    if trades.empty:
        return {
            "rule": rule.name,
            "trade_count": 0,
            "ending_bankroll": starting_bankroll,
            "profit": 0.0,
            "roi_on_amount_risked": 0.0,
            "average_clv_cents": 0.0,
            "positive_clv_rate": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "months": 0,
            "profitable_months": 0,
            "positive_clv_months": 0,
            "repeatability_result": "0/0 profitable months; 0/0 positive-CLV months",
            "overfit_risk": "high",
            "final_status": "rejected",
        }
    risked = float(trades["cost"].sum())
    profit = float(trades["profit"].sum())
    monthly = trades.groupby("month", as_index=False).agg(profit=("profit", "sum"), avg_clv=("clv_cents", "mean"))
    months = int(monthly["month"].nunique())
    profitable_months = int((monthly["profit"] > 0).sum())
    positive_clv_months = int((monthly["avg_clv"] > 0).sum())
    positive_profit = monthly.loc[monthly["profit"] > 0, "profit"].sum()
    max_month_profit_share = float(monthly["profit"].max() / positive_profit) if positive_profit > 0 else 1.0
    positive_clv_rate = float((trades["clv_cents"] > 0).mean())
    avg_clv = float(trades["clv_cents"].mean())
    overfit_risk = "high" if len(trades) < 100 or max_month_profit_share > 0.60 else "medium"
    final_status = (
        "candidate"
        if len(trades) >= 100
        and profit > 0
        and avg_clv > 0
        and positive_clv_rate > 0.50
        and profitable_months >= 3
        and positive_clv_months >= 3
        and max_month_profit_share <= 0.60
        else "watchlist"
        if len(trades) >= 50 and profit > 0 and avg_clv > 0 and profitable_months >= 2
        else "rejected"
    )
    return {
        "rule": rule.name,
        "group_columns": "+".join(rule.group_columns) if rule.group_columns else "overall",
        "min_prior_rows": rule.min_prior_rows,
        "min_prior_profit_per_share": rule.min_prior_profit_per_share,
        "min_prior_avg_clv_cents": rule.min_prior_avg_clv_cents,
        "min_prior_positive_clv_rate": rule.min_prior_positive_clv_rate,
        "trade_count": int(len(trades)),
        "ending_bankroll": float(trades["bankroll_after"].iloc[-1]),
        "profit": profit,
        "roi_on_amount_risked": profit / risked if risked else 0.0,
        "average_profit_per_trade": profit / len(trades),
        "win_rate": float(trades["actual_no_win"].mean()),
        "average_clv_cents": avg_clv,
        "positive_clv_rate": positive_clv_rate,
        "max_drawdown": _max_drawdown(trades["bankroll_after"], starting_bankroll),
        "months": months,
        "profitable_months": profitable_months,
        "positive_clv_months": positive_clv_months,
        "max_month_profit_share": max_month_profit_share,
        "repeatability_result": f"{profitable_months}/{months} profitable months; {positive_clv_months}/{months} positive-CLV months",
        "overfit_risk": overfit_risk,
        "final_status": final_status,
        "research_only": True,
    }


def run_no_suppression_sweep(
    no_rows: pd.DataFrame,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
    rules: list[SuppressionRule] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run simple prior-period-only NO suppression rules."""

    rules = rules or default_suppression_rules()
    summaries: list[dict[str, Any]] = []
    walk_rows: list[pd.DataFrame] = []
    for rule in rules:
        selected_flags = apply_prior_suppression_rule(no_rows, rule)
        simulated = simulate_rule_bankroll(selected_flags, starting_bankroll, max_bet_fraction)
        summaries.append(summarize_rule_trades(simulated, rule, starting_bankroll))
        if not simulated.empty:
            monthly = simulated.groupby("month", as_index=False).agg(
                trade_count=("selected_by_rule", "count"),
                profit=("profit", "sum"),
                average_clv_cents=("clv_cents", "mean"),
                positive_clv_rate=("clv_cents", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
            )
            monthly.insert(0, "rule", rule.name)
            walk_rows.append(monthly)
    sweep = pd.DataFrame(summaries)
    if not sweep.empty:
        rank = {"candidate": 0, "watchlist": 1, "rejected": 2}
        sweep["_rank"] = sweep["final_status"].map(rank).fillna(99)
        sweep = sweep.sort_values(
            ["_rank", "profit", "average_clv_cents", "positive_clv_rate", "trade_count"],
            ascending=[True, False, False, False, False],
        ).drop(columns="_rank").reset_index(drop=True)
    walk = pd.concat(walk_rows, ignore_index=True, sort=False) if walk_rows else pd.DataFrame()
    return sweep, walk


def build_no_settlement_recommendations(summary: dict[str, Any], best_rule: dict[str, Any]) -> str:
    return (
        "# NO Settlement Calibration Audit\n\n"
        "## Verdict\n\n"
        f"- Status: {summary.get('status', 'unknown')}\n"
        f"- NO CLV predicts profit: {summary.get('no_clv_predicts_profit', False)}\n"
        f"- NO overconfident: {summary.get('no_overconfident', False)}\n"
        f"- Best suppression status: {best_rule.get('final_status', 'rejected')}\n"
        f"- Single-game edge proven: {summary.get('single_game_edge_proven', False)}\n"
        f"- Parlays: {summary.get('parlay_status', 'blocked_single_game_edge_not_proven')}\n\n"
        "## Interpretation\n\n"
        f"{summary.get('diagnosis', '')}\n\n"
        "Positive CLV is useful, but it is not enough here because settlement win rate must clear the NO buy "
        "price. Expensive misses and weak positive-CLV frequency keep NO from becoming a proven edge.\n\n"
        "## Best Suppression Rule\n\n"
        f"- Rule: {best_rule.get('rule', 'n/a')}\n"
        f"- Trades: {best_rule.get('trade_count', 0)}\n"
        f"- Profit: {float(best_rule.get('profit', 0.0) or 0.0):+.2f}\n"
        f"- Average CLV: {float(best_rule.get('average_clv_cents', 0.0) or 0.0):+.3f} cents\n"
        f"- Positive CLV rate: {float(best_rule.get('positive_clv_rate', 0.0) or 0.0):.1%}\n"
        f"- Repeatability: {best_rule.get('repeatability_result', 'n/a')}\n"
        f"- Overfit risk: {best_rule.get('overfit_risk', 'high')}\n\n"
        "## Guardrails\n\n"
        "- Research-only output.\n"
        "- No proof gates were loosened.\n"
        "- No fair-price bets or parlays were enabled.\n"
        "- Full-sample bucket summaries are diagnostic only.\n"
    )


def build_no_settlement_calibration_audit(
    trades: pd.DataFrame,
    proof_summary: dict[str, Any] | None = None,
    fair_price_summary: dict[str, Any] | None = None,
    parlay_summary: dict[str, Any] | None = None,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Build all NO settlement calibration reports."""

    proof_summary = proof_summary or {}
    fair_price_summary = fair_price_summary or {}
    parlay_summary = parlay_summary or {}
    no_rows = prepare_no_settlement_rows(trades)
    by_probability = summarize_segments(no_rows, ["probability_bucket"], min_rows=1)
    by_price = summarize_segments(no_rows, ["price_bucket"], min_rows=1)
    by_bucket = pd.concat(
        [
            by_probability.assign(bucket_type="probability_bucket"),
            by_price.assign(bucket_type="price_bucket"),
            summarize_segments(no_rows, ["edge_bucket"], min_rows=1).assign(bucket_type="edge_bucket"),
            summarize_segments(no_rows, ["liquidity_bucket"], min_rows=1).assign(bucket_type="liquidity_bucket"),
        ],
        ignore_index=True,
        sort=False,
    )
    clv_profit = clv_vs_profit(no_rows)
    failures = failure_segments(no_rows)
    sweep, walk = run_no_suppression_sweep(no_rows, starting_bankroll, max_bet_fraction)
    best_rule = sweep.iloc[0].to_dict() if not sweep.empty else {}

    positive_clv_rows = no_rows[no_rows["positive_clv"]].copy()
    positive_clv_profit = float(positive_clv_rows["profit"].sum()) if not positive_clv_rows.empty else 0.0
    no_clv_predicts_profit = bool(positive_clv_profit > 0 and not positive_clv_rows.empty)
    overconfidence = float(no_rows["actual_no_win"].mean() - no_rows["predicted_no_probability"].mean()) if not no_rows.empty else 0.0
    expensive_miss_profit = float(no_rows.loc[no_rows["expensive_miss"], "profit"].sum()) if not no_rows.empty else 0.0
    status = "not_proven"
    if best_rule.get("final_status") == "candidate":
        status = "research_candidate"
    elif best_rule.get("final_status") == "watchlist":
        status = "watchlist"
    diagnosis = (
        "NO shows positive average CLV, but settlement profit is not consistently explained by that CLV. "
        "The NO forecast is overconfident when actual settlement win rate trails predicted NO probability, "
        "and expensive misses can erase small CLV gains."
    )
    summary = {
        "status": status,
        "rows": int(len(no_rows)),
        "avg_predicted_no_probability": float(no_rows["predicted_no_probability"].mean()) if not no_rows.empty else 0.0,
        "actual_no_win_rate": float(no_rows["actual_no_win"].mean()) if not no_rows.empty else 0.0,
        "avg_break_even_probability": float(no_rows["break_even_probability"].mean()) if not no_rows.empty else 0.0,
        "calibration_error": overconfidence,
        "no_overconfident": bool(overconfidence < -0.03),
        "profit": float(no_rows["profit"].sum()) if not no_rows.empty else 0.0,
        "avg_profit_per_share": float(no_rows["profit_per_share"].mean()) if not no_rows.empty else 0.0,
        "avg_clv_cents": float(no_rows["clv_cents"].mean()) if not no_rows.empty else 0.0,
        "positive_clv_rate": float(no_rows["positive_clv"].mean()) if not no_rows.empty else 0.0,
        "positive_clv_rows": int(len(positive_clv_rows)),
        "positive_clv_profit": positive_clv_profit,
        "positive_clv_win_rate": float(positive_clv_rows["actual_no_win"].mean()) if not positive_clv_rows.empty else 0.0,
        "no_clv_predicts_profit": no_clv_predicts_profit,
        "expensive_miss_count": int(no_rows["expensive_miss"].sum()) if not no_rows.empty else 0,
        "expensive_miss_profit": expensive_miss_profit,
        "best_suppression_rule": best_rule,
        "single_game_edge_proven": bool(proof_summary.get("single_game_edge_proven", False)),
        "proof_status": proof_summary.get("status", "unknown"),
        "fair_price_bets": int(fair_price_summary.get("bets", 0) or 0),
        "parlay_status": parlay_summary.get("status", "unknown"),
        "parlay_recommendations_allowed": bool(parlay_summary.get("parlay_recommendations_allowed", False)),
        "research_only": True,
        "diagnosis": diagnosis,
    }
    recommendations = build_no_settlement_recommendations(summary, best_rule)
    return summary, by_bucket, clv_profit, failures, sweep, walk, recommendations


def save_no_settlement_calibration_outputs(
    summary: dict[str, Any],
    by_bucket: pd.DataFrame,
    clv_profit: pd.DataFrame,
    failures: pd.DataFrame,
    sweep: pd.DataFrame,
    walk: pd.DataFrame,
    recommendations: str,
    output_dir: str | Path,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "no_settlement_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    by_bucket.to_csv(output_root / "no_settlement_calibration_by_bucket.csv", index=False)
    clv_profit.to_csv(output_root / "no_clv_vs_profit.csv", index=False)
    failures.to_csv(output_root / "no_settlement_failure_segments.csv", index=False)
    sweep.to_csv(output_root / "no_suppression_rule_sweep.csv", index=False)
    walk.to_csv(output_root / "no_suppression_walk_forward.csv", index=False)
    (output_root / "no_settlement_recommendations.md").write_text(recommendations, encoding="utf-8")
