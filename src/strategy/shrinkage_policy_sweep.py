"""Research-only side-specific probability shrinkage policy sweep."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategy.probability_shrinkage import shrink_probability
from strategy.staking import calculate_flat_fractional_shares
from strategy.uncertainty_penalty import DEFAULT_CONSERVATIVE_PENALTY, apply_prior_penalties


YES_SHRINK_FACTORS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
NO_SHRINK_FACTORS = [0.25, 0.40, 0.50, 0.75, 1.00]
MIN_EDGES = [0.03, 0.05, 0.07, 0.10]
PENALTY_MODES = [
    "none",
    "side-only",
    "side+price_bucket",
    "side+price_bucket+edge_bucket",
    "side+price_bucket+liquidity_bucket",
]
MIN_PRIOR_SAMPLES = [20, 50, 100]

PRICE_BINS = [0, 25, 40, 55, 70, 85, 100]
PRICE_LABELS = ["0-25", "25-40", "40-55", "55-70", "70-85", "85-100"]
EDGE_BINS = [-np.inf, 0.0, 0.02, 0.05, 0.08, 0.12, np.inf]
EDGE_LABELS = ["<=0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"]
LIQUIDITY_BINS = [-np.inf, 10, 100, 1000, 10000, np.inf]
LIQUIDITY_LABELS = ["<10", "10-100", "100-1k", "1k-10k", "10k+"]


@dataclass(frozen=True)
class ShrinkagePolicy:
    yes_shrink_factor: float
    no_shrink_factor: float
    min_edge: float
    uncertainty_penalty_mode: str
    min_prior_samples: int

    @property
    def name(self) -> str:
        return (
            f"yes_{self.yes_shrink_factor:.2f}|no_{self.no_shrink_factor:.2f}|"
            f"edge_{self.min_edge:.2f}|{self.uncertainty_penalty_mode}|prior_{self.min_prior_samples}"
        )


def default_policy_grid() -> list[ShrinkagePolicy]:
    """Return the fixed, interpretable research grid."""

    return [
        ShrinkagePolicy(yes, no, edge, mode, prior)
        for yes in YES_SHRINK_FACTORS
        for no in NO_SHRINK_FACTORS
        for edge in MIN_EDGES
        for mode in PENALTY_MODES
        for prior in MIN_PRIOR_SAMPLES
    ]


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    out = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    return out.astype("object").where(out.notna(), "unknown").astype(str)


def build_candidate_rows(markets: pd.DataFrame) -> pd.DataFrame:
    """Expand matched markets to tradable YES and NO candidate rows."""

    required = ["game_date", "game_id", "market_ticker", "model_yes_prob", "yes_ask", "yes_bid", "actual_yes_win"]
    missing = [column for column in required if column not in markets.columns]
    if missing:
        raise ValueError(f"Matched market rows are missing columns: {missing}")

    frame = markets.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    for column in [
        "model_yes_prob",
        "yes_ask",
        "yes_bid",
        "actual_yes_win",
        "volume",
        "open_interest",
        "clv_reference_price_cents",
        "clv_reference_no_price_cents",
    ]:
        if column in frame.columns and column != "actual_yes_win":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    base_columns = [
        "game_date",
        "game_id",
        "season",
        "season_type",
        "market_ticker",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "model_yes_prob",
        "actual_yes_win",
        "snapshot_target",
        "price_quality",
        "volume",
        "open_interest",
        "clv_reference_price_cents",
        "clv_reference_snapshot",
        "clv_reference_no_price_cents",
        "clv_reference_no_snapshot",
    ]
    yes = frame[[column for column in base_columns if column in frame.columns]].copy()
    yes["side"] = "YES"
    yes["model_prob"] = pd.to_numeric(yes["model_yes_prob"], errors="coerce")
    yes["price_cents"] = pd.to_numeric(frame["yes_ask"], errors="coerce")
    yes["market_prob"] = yes["price_cents"] / 100.0
    yes["contract_won"] = _to_bool(frame["actual_yes_win"])
    yes["clv_reference_price_for_side"] = pd.to_numeric(frame.get("clv_reference_price_cents"), errors="coerce")

    no = frame[[column for column in base_columns if column in frame.columns]].copy()
    no["side"] = "NO"
    no["model_prob"] = 1.0 - pd.to_numeric(no["model_yes_prob"], errors="coerce")
    no["price_cents"] = 100.0 - pd.to_numeric(frame["yes_bid"], errors="coerce")
    no["market_prob"] = no["price_cents"] / 100.0
    no["contract_won"] = ~_to_bool(frame["actual_yes_win"])
    no["clv_reference_price_for_side"] = pd.to_numeric(frame.get("clv_reference_no_price_cents"), errors="coerce")

    candidates = pd.concat([yes, no], ignore_index=True, sort=False)
    candidates = candidates[
        candidates["game_date"].notna()
        & candidates["model_prob"].notna()
        & candidates["market_prob"].notna()
        & candidates["price_cents"].between(5, 95, inclusive="both")
    ].copy()
    candidates["month"] = candidates["game_date"].dt.to_period("M").astype(str)
    candidates["raw_edge"] = candidates["model_prob"] - candidates["market_prob"]
    candidates["price_bucket"] = _bucket(candidates["price_cents"], PRICE_BINS, PRICE_LABELS)
    candidates["edge_bucket"] = _bucket(candidates["raw_edge"], EDGE_BINS, EDGE_LABELS)
    candidates["liquidity_bucket"] = _bucket(candidates.get("volume", pd.Series(np.nan, index=candidates.index)), LIQUIDITY_BINS, LIQUIDITY_LABELS)
    return candidates.sort_values(["game_date", "game_id", "market_ticker", "side"]).reset_index(drop=True)


def apply_shrinkage_policy(candidates: pd.DataFrame, policy: ShrinkagePolicy) -> pd.DataFrame:
    """Apply side-specific shrinkage before uncertainty penalties."""

    output = candidates.copy()
    shrink = np.where(output["side"].eq("NO"), policy.no_shrink_factor, policy.yes_shrink_factor)
    output["shrink_factor"] = shrink
    output["adjusted_probability"] = shrink_probability(output["model_prob"], output["market_prob"], shrink)
    output["adjusted_edge_before_penalty"] = output["adjusted_probability"] - output["market_prob"]
    return output


def _select_current_month(base_rows: pd.DataFrame, current_rows: pd.DataFrame, policy: ShrinkagePolicy) -> pd.DataFrame:
    prior_base = base_rows[
        (base_rows["game_date"] < current_rows["game_date"].min())
        & (base_rows["adjusted_edge_before_penalty"] >= policy.min_edge)
    ].copy()
    current = apply_prior_penalties(
        current_rows,
        prior_base,
        mode=policy.uncertainty_penalty_mode,
        min_prior_samples=policy.min_prior_samples,
        conservative_default=DEFAULT_CONSERVATIVE_PENALTY,
    )
    current["final_edge"] = current["adjusted_edge_before_penalty"] - current["uncertainty_penalty"]
    current = current[current["final_edge"] >= policy.min_edge].copy()
    if current.empty:
        return current
    current = (
        current.sort_values(["game_date", "game_id", "market_ticker", "final_edge"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["game_id", "market_ticker"], keep="first")
        .copy()
    )
    return current


def select_policy_trades(candidates: pd.DataFrame, policy: ShrinkagePolicy) -> pd.DataFrame:
    """Select policy trades using prior-period-only penalties for each evaluation month."""

    base = apply_shrinkage_policy(candidates, policy)
    selected: list[pd.DataFrame] = []
    for month in sorted(base["month"].dropna().unique().tolist()):
        current = base[base["month"].eq(month)].copy()
        if current.empty:
            continue
        selected.append(_select_current_month(base, current, policy))
    if not selected:
        return pd.DataFrame(columns=list(base.columns) + ["uncertainty_penalty", "final_edge"])
    output = pd.concat(selected, ignore_index=True, sort=False)
    output["policy"] = policy.name
    output["yes_shrink_factor"] = policy.yes_shrink_factor
    output["no_shrink_factor"] = policy.no_shrink_factor
    output["min_edge"] = policy.min_edge
    output["uncertainty_penalty_mode"] = policy.uncertainty_penalty_mode
    output["min_prior_samples"] = policy.min_prior_samples
    return output.sort_values(["game_date", "game_id", "market_ticker"]).reset_index(drop=True)


def simulate_bankroll(
    selected: pd.DataFrame,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
) -> pd.DataFrame:
    """Simulate flat fractional fake-bankroll trades for selected research rows."""

    bankroll = float(starting_bankroll)
    rows: list[dict[str, Any]] = []
    selected = selected.sort_values(["game_date", "game_id", "market_ticker"]).reset_index(drop=True)
    for _, row in selected.iterrows():
        bankroll_before = bankroll
        shares = calculate_flat_fractional_shares(
            bankroll=bankroll,
            price_cents=float(row["price_cents"]),
            max_bet_fraction=max_bet_fraction,
        )
        if shares < 1:
            cost = 0.0
            payout = 0.0
            profit = 0.0
            trade = False
        else:
            trade = True
            cost = shares * (float(row["price_cents"]) / 100.0)
            payout = float(shares) if bool(row["contract_won"]) else 0.0
            profit = payout - cost
            bankroll = bankroll - cost + payout
        clv_reference = row.get("clv_reference_price_for_side", np.nan)
        clv_cents = float(clv_reference) - float(row["price_cents"]) if pd.notna(clv_reference) else np.nan
        item = row.to_dict()
        item.update(
            {
                "trade": bool(trade),
                "shares": int(shares),
                "cost": float(cost),
                "payout": float(payout),
                "profit": float(profit),
                "bankroll_before": float(bankroll_before),
                "bankroll_after": float(bankroll),
                "clv_cents": clv_cents,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _max_drawdown(equity: pd.Series, starting_bankroll: float) -> float:
    if equity.empty:
        equity = pd.Series([starting_bankroll], dtype="float64")
    running = equity.cummax()
    drawdown = (equity - running) / running
    return float(drawdown.min()) if not drawdown.empty else 0.0


def summarize_policy_trades(
    trades: pd.DataFrame,
    policy: ShrinkagePolicy,
    starting_bankroll: float = 100.0,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one research policy."""

    baseline = baseline or {}
    traded = trades[trades.get("trade", pd.Series(False, index=trades.index)).astype(bool)].copy() if not trades.empty else pd.DataFrame()
    amount_risked = float(traded["cost"].sum()) if not traded.empty else 0.0
    profit = float(traded["profit"].sum()) if not traded.empty else 0.0
    ending_bankroll = float(trades["bankroll_after"].iloc[-1]) if not trades.empty and "bankroll_after" in trades.columns else float(starting_bankroll)
    by_side = traded.groupby("side") if not traded.empty else {}

    def side_value(side: str, column: str, default: float = 0.0) -> float:
        if traded.empty or side not in traded["side"].values:
            return default
        side_rows = traded[traded["side"].eq(side)]
        if column == "count":
            return float(len(side_rows))
        if column == "positive_clv_rate":
            clv = pd.to_numeric(side_rows["clv_cents"], errors="coerce").dropna()
            return float((clv > 0).mean()) if not clv.empty else 0.0
        return float(pd.to_numeric(side_rows[column], errors="coerce").mean() if column == "clv_cents" else side_rows[column].sum())

    monthly = (
        traded.assign(month=traded["game_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(profit=("profit", "sum"), avg_clv=("clv_cents", "mean"), trades=("trade", "sum"))
        if not traded.empty
        else pd.DataFrame(columns=["month", "profit", "avg_clv", "trades"])
    )
    positive_profit = monthly.loc[monthly["profit"] > 0, "profit"].sum() if not monthly.empty else 0.0
    max_month_profit_share = (
        float(monthly["profit"].max() / positive_profit) if positive_profit and not monthly.empty else 1.0
    )
    average_clv = float(pd.to_numeric(traded["clv_cents"], errors="coerce").mean()) if not traded.empty else 0.0
    positive_clv_rate = float((pd.to_numeric(traded["clv_cents"], errors="coerce") > 0).mean()) if not traded.empty else 0.0
    profitable_months = int((monthly["profit"] > 0).sum()) if not monthly.empty else 0
    positive_clv_months = int((monthly["avg_clv"] > 0).sum()) if not monthly.empty else 0
    month_count = int(monthly["month"].nunique()) if not monthly.empty else 0
    simulated_passes = {
        "strategy_backtest_profit": profit > 0,
        "average_clv": average_clv > 0,
        "positive_clv_rate": positive_clv_rate > 0.50,
        "repeatability_months": month_count >= 6,
        "profitable_month_repeatability": profitable_months >= max(3, int(np.ceil(month_count * 0.5))) if month_count else False,
        "positive_clv_month_repeatability": positive_clv_months >= max(3, int(np.ceil(month_count * 0.5))) if month_count else False,
        "month_profit_concentration": max_month_profit_share <= 0.60,
    }
    proof_gate_simulation_passed = bool(all(simulated_passes.values()))
    improves_clv = average_clv > float(baseline.get("average_clv_cents", -999.0))
    improves_profit = profit > float(baseline.get("profit", -999999.0))
    improves_yes = side_value("YES", "profit") > float(baseline.get("yes_profit", -999999.0))
    enough_trades = len(traded) >= 100
    overfit_risk = (
        "high"
        if len(traded) < 100 or max_month_profit_share > 0.60
        else "medium"
        if positive_clv_rate <= 0.50 or profitable_months < 3
        else "low"
    )
    final_status = (
        "candidate"
        if proof_gate_simulation_passed and enough_trades and overfit_risk == "low"
        else "watchlist"
        if enough_trades and improves_clv and improves_profit and improves_yes and average_clv > 0
        else "rejected"
    )

    return {
        "policy": policy.name,
        "yes_shrink_factor": policy.yes_shrink_factor,
        "no_shrink_factor": policy.no_shrink_factor,
        "min_edge": policy.min_edge,
        "uncertainty_penalty_mode": policy.uncertainty_penalty_mode,
        "min_prior_samples": policy.min_prior_samples,
        "trade_count": int(len(traded)),
        "yes_trade_count": int(side_value("YES", "count")),
        "no_trade_count": int(side_value("NO", "count")),
        "ending_bankroll": ending_bankroll,
        "profit": profit,
        "roi_on_amount_risked": profit / amount_risked if amount_risked else 0.0,
        "average_profit_per_trade": profit / len(traded) if len(traded) else 0.0,
        "win_rate": float((traded["profit"] > 0).mean()) if not traded.empty else 0.0,
        "average_clv_cents": average_clv,
        "positive_clv_rate": positive_clv_rate,
        "yes_profit": side_value("YES", "profit"),
        "yes_average_clv_cents": side_value("YES", "clv_cents"),
        "yes_positive_clv_rate": side_value("YES", "positive_clv_rate"),
        "no_profit": side_value("NO", "profit"),
        "no_average_clv_cents": side_value("NO", "clv_cents"),
        "no_positive_clv_rate": side_value("NO", "positive_clv_rate"),
        "max_drawdown": _max_drawdown(trades["bankroll_after"] if not trades.empty else pd.Series(dtype=float), starting_bankroll),
        "months": month_count,
        "profitable_months": profitable_months,
        "positive_clv_months": positive_clv_months,
        "max_month_profit_share": max_month_profit_share,
        "proof_gate_pass_count_simulated": int(sum(simulated_passes.values())),
        "proof_gate_fail_count_simulated": int(len(simulated_passes) - sum(simulated_passes.values())),
        "proof_gate_simulation_passed": proof_gate_simulation_passed,
        "fair_price_actionable_bets_after_proof_gate": 0,
        "parlay_status": "blocked_single_game_edge_not_proven",
        "repeatability_result": (
            f"{profitable_months}/{month_count} profitable months; {positive_clv_months}/{month_count} positive-CLV months"
        ),
        "overfit_risk": overfit_risk,
        "final_status": final_status,
        "research_only": True,
    }


def run_side_specific_shrinkage_sweep(
    markets: pd.DataFrame,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
    policies: list[ShrinkagePolicy] | None = None,
    baseline: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    """Run all shrinkage policies with prior-period-only uncertainty penalties."""

    candidates = build_candidate_rows(markets)
    policies = policies or default_policy_grid()
    rows: list[dict[str, Any]] = []
    walk_rows: list[pd.DataFrame] = []
    for policy in policies:
        selected = select_policy_trades(candidates, policy)
        trades = simulate_bankroll(selected, starting_bankroll=starting_bankroll, max_bet_fraction=max_bet_fraction)
        summary = summarize_policy_trades(trades, policy, starting_bankroll=starting_bankroll, baseline=baseline)
        rows.append(summary)
        if not trades.empty:
            monthly = (
                trades[trades["trade"].astype(bool)]
                .assign(month=trades["game_date"].dt.to_period("M").astype(str))
                .groupby("month", as_index=False)
                .agg(
                    trade_count=("trade", "sum"),
                    profit=("profit", "sum"),
                    average_clv_cents=("clv_cents", "mean"),
                    positive_clv_rate=("clv_cents", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
                    yes_trade_count=("side", lambda values: int((values == "YES").sum())),
                    no_trade_count=("side", lambda values: int((values == "NO").sum())),
                )
            )
            monthly.insert(0, "policy", policy.name)
            monthly.insert(1, "yes_shrink_factor", policy.yes_shrink_factor)
            monthly.insert(2, "no_shrink_factor", policy.no_shrink_factor)
            monthly.insert(3, "min_edge", policy.min_edge)
            monthly.insert(4, "uncertainty_penalty_mode", policy.uncertainty_penalty_mode)
            monthly.insert(5, "min_prior_samples", policy.min_prior_samples)
            walk_rows.append(monthly)
    sweep = pd.DataFrame(rows)
    if not sweep.empty:
        rank_map = {"candidate": 0, "watchlist": 1, "rejected": 2}
        sweep["_rank"] = sweep["final_status"].map(rank_map).fillna(99)
        sweep = sweep.sort_values(
            [
                "_rank",
                "average_clv_cents",
                "positive_clv_rate",
                "profit",
                "profitable_months",
                "trade_count",
            ],
            ascending=[True, False, False, False, False, False],
        ).drop(columns="_rank").reset_index(drop=True)
    walk_forward = pd.concat(walk_rows, ignore_index=True, sort=False) if walk_rows else pd.DataFrame()
    best = sweep.iloc[0].to_dict() if not sweep.empty else {}
    candidate_count = int(sweep["final_status"].eq("candidate").sum()) if not sweep.empty else 0
    watchlist_count = int(sweep["final_status"].eq("watchlist").sum()) if not sweep.empty else 0
    summary = {
        "status": "candidate_found" if candidate_count else "watchlist_found" if watchlist_count else "no_validated_policy",
        "research_only": True,
        "policies_tested": int(len(sweep)),
        "candidate_policies": candidate_count,
        "watchlist_policies": watchlist_count,
        "best_policy": best,
        "single_game_edge_proven": False,
        "fair_price_actionable_bets_after_proof_gate": 0,
        "parlay_status": "blocked_single_game_edge_not_proven",
        "parlay_recommendations_allowed": False,
        "note": (
            "Research-only side-specific shrinkage sweep. Policies use prior-period-only bucket penalties "
            "and do not modify fair-price recommendations or proof gates."
        ),
    }
    recommendations = build_recommendations_markdown(summary, baseline or {})
    return sweep, walk_forward, summary, recommendations


def build_recommendations_markdown(summary: dict[str, Any], baseline: dict[str, Any]) -> str:
    best = summary.get("best_policy", {}) or {}
    lines = [
        "# Side-Specific Shrinkage Research",
        "",
        "## Verdict",
        "",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Research-only: {summary.get('research_only', True)}",
        f"- Candidate policies: {summary.get('candidate_policies', 0)}",
        f"- Watchlist policies: {summary.get('watchlist_policies', 0)}",
        f"- Single-game edge proven: {summary.get('single_game_edge_proven', False)}",
        f"- Parlays: {summary.get('parlay_status', 'blocked')}",
        "",
        "## Best Policy",
        "",
        f"- Policy: {best.get('policy', 'n/a')}",
        f"- Trades: {best.get('trade_count', 0)}",
        f"- Profit: {float(best.get('profit', 0.0) or 0.0):+.2f}",
        f"- Average CLV: {float(best.get('average_clv_cents', 0.0) or 0.0):+.3f} cents",
        f"- Positive CLV rate: {float(best.get('positive_clv_rate', 0.0) or 0.0):.1%}",
        f"- YES profit: {float(best.get('yes_profit', 0.0) or 0.0):+.2f}",
        f"- NO profit: {float(best.get('no_profit', 0.0) or 0.0):+.2f}",
        f"- Repeatability: {best.get('repeatability_result', 'n/a')}",
        f"- Final status: {best.get('final_status', 'rejected')}",
        "",
        "## Interpretation",
        "",
        (
            "The sweep tests whether model-market disagreement was too aggressive by shrinking probabilities "
            "toward tradable Kalshi prices. A policy is not promoted unless it survives prior-period-only "
            "penalties and improves CLV, profit, drawdown, YES losses, and month-to-month repeatability."
        ),
        "",
    ]
    if best.get("final_status") == "candidate":
        lines.append("The best policy is a research candidate only. It still requires explicit user approval before promotion.")
    elif best.get("final_status") == "watchlist":
        lines.append("The best policy improved some metrics but does not prove repeatable single-game edge.")
    else:
        lines.append("No policy clearly proved repeatable single-game edge.")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Proof gates were not changed.",
            "- Fair-price recommendations remain controlled by `single_game_proof_summary.json`.",
            "- Parlay recommendations remain blocked until single-game edge is proven.",
            "- Full-sample results are diagnostic only.",
        ]
    )
    if baseline:
        lines.extend(
            [
                "",
                "## Baseline",
                "",
                f"- Profit: {float(baseline.get('profit', 0.0) or 0.0):+.2f}",
                f"- Average CLV: {float(baseline.get('average_clv_cents', 0.0) or 0.0):+.3f} cents",
                f"- Positive CLV rate: {float(baseline.get('positive_clv_rate', 0.0) or 0.0):.1%}",
                f"- YES profit: {float(baseline.get('yes_profit', 0.0) or 0.0):+.2f}",
            ]
        )
    return "\n".join(lines) + "\n"


def save_side_specific_shrinkage_outputs(
    sweep: pd.DataFrame,
    walk_forward: pd.DataFrame,
    summary: dict[str, Any],
    recommendations: str,
    output_dir: str | Path,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_root / "side_specific_shrinkage_sweep.csv", index=False)
    walk_forward.to_csv(output_root / "side_specific_shrinkage_walk_forward.csv", index=False)
    (output_root / "side_specific_shrinkage_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output_root / "side_specific_shrinkage_recommendations.md").write_text(recommendations, encoding="utf-8")
