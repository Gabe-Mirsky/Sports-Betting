"""Root-cause diagnostics for current single-game edge failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRICE_BINS = [0, 25, 40, 55, 70, 85, 100]
PRICE_LABELS = ["0-25", "25-40", "40-55", "55-70", "70-85", "85-100"]
EDGE_BINS = [-np.inf, 0.0, 0.02, 0.05, 0.08, 0.12, np.inf]
EDGE_LABELS = ["<=0%", "0-2%", "2-5%", "5-8%", "8-12%", "12%+"]
LIQUIDITY_BINS = [-np.inf, 10, 100, 1000, 10000, np.inf]
LIQUIDITY_LABELS = ["<10", "10-100", "100-1k", "1k-10k", "10k+"]
SPREAD_BINS = [-np.inf, 1, 3, 5, 10, np.inf]
SPREAD_LABELS = ["<=1", "1-3", "3-5", "5-10", "10+"]

DEFAULT_SEGMENTS = [
    ["diagnostic_side"],
    ["price_bucket"],
    ["edge_bucket"],
    ["liquidity_bucket"],
    ["spread_bucket"],
    ["month"],
    ["yes_team_abbr"],
    ["snapshot_bucket"],
    ["confidence_bucket"],
    ["fair_recommendation_bucket"],
    ["proof_gate_bucket"],
    ["diagnostic_side", "edge_bucket"],
    ["diagnostic_side", "price_bucket"],
    ["diagnostic_side", "liquidity_bucket"],
    ["diagnostic_side", "month"],
    ["diagnostic_side", "fair_recommendation_bucket"],
]

WALK_FORWARD_SEGMENTS = [
    ["diagnostic_side"],
    ["diagnostic_side", "price_bucket"],
    ["diagnostic_side", "edge_bucket"],
    ["diagnostic_side", "liquidity_bucket"],
    ["diagnostic_side", "snapshot_bucket"],
    ["diagnostic_side", "fair_recommendation_bucket"],
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    return frame[column].map(_as_bool).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _label_cut(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    bucket = pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    return bucket.astype("object").where(bucket.notna(), "unknown").astype(str)


def _side_series(frame: pd.DataFrame) -> pd.Series:
    side = pd.Series("", index=frame.index, dtype="object")
    for column in ["side", "candidate_side", "calibrated_side", "ungated_side"]:
        if column in frame.columns:
            side = side.where(side.astype(str).str.len() > 0, frame[column])
    side = side.fillna("").astype(str).str.upper()
    return side.where(side.isin({"YES", "NO"}), "unknown")


def _contract_won(frame: pd.DataFrame, side_column: str = "diagnostic_side") -> pd.Series:
    actual_yes = _bool_series(frame, "actual_yes_win")
    side = frame.get(side_column, pd.Series("unknown", index=frame.index)).fillna("").astype(str).str.upper()
    won = pd.Series(np.nan, index=frame.index, dtype="object")
    won = won.where(~side.eq("YES"), actual_yes)
    won = won.where(~side.eq("NO"), ~actual_yes)
    return pd.to_numeric(won, errors="coerce")


def _merge_fair_price_context(trades: pd.DataFrame, fair_price_signals: pd.DataFrame | None) -> pd.DataFrame:
    if fair_price_signals is None or fair_price_signals.empty:
        return trades.copy()

    key_columns = [column for column in ["game_id", "market_ticker"] if column in trades.columns and column in fair_price_signals.columns]
    if not key_columns:
        return trades.copy()

    fair_columns = [
        "game_id",
        "market_ticker",
        "confidence",
        "recommendation",
        "ungated_recommendation",
        "ungated_side",
        "final_edge",
        "gross_edge",
        "fee_adjusted_edge",
        "spread",
        "spread_ok",
        "liquidity_ok",
        "proof_gate_status",
        "single_game_edge_proven",
    ]
    available = [column for column in fair_columns if column in fair_price_signals.columns]
    fair = fair_price_signals[available].drop_duplicates(subset=key_columns).copy()
    rename = {
        "confidence": "fair_confidence",
        "recommendation": "fair_recommendation",
        "final_edge": "fair_final_edge",
        "gross_edge": "fair_gross_edge",
        "fee_adjusted_edge": "fair_fee_adjusted_edge",
        "spread": "fair_spread",
        "spread_ok": "fair_spread_ok",
        "liquidity_ok": "fair_liquidity_ok",
    }
    fair = fair.rename(columns={key: value for key, value in rename.items() if key in fair.columns})
    return trades.merge(fair, on=key_columns, how="left")


def prepare_diagnostics_rows(
    trades: pd.DataFrame,
    fair_price_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach stable diagnostic fields to canonical backtest rows."""

    if trades.empty:
        return pd.DataFrame()

    frame = _merge_fair_price_context(trades, fair_price_signals)
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame["month"] = frame["date"].dt.to_period("M").astype(str).where(frame["date"].notna(), "unknown")
    frame["trade_bool"] = _bool_series(frame, "trade")
    frame["diagnostic_side"] = _side_series(frame)
    frame["contract_won"] = _contract_won(frame)
    frame["profit"] = _num(frame, "profit", 0.0).where(frame["trade_bool"], 0.0)
    frame["cost"] = _num(frame, "cost", 0.0).where(frame["trade_bool"], 0.0)
    frame["shares"] = _num(frame, "shares", 0.0).where(frame["trade_bool"], 0.0)
    frame["price_cents"] = _num(frame, "price_cents")
    frame["edge"] = _num(frame, "edge")
    frame["model_prob"] = _num(frame, "model_prob")
    frame["market_prob"] = _num(frame, "market_prob")
    frame["volume"] = _num(frame, "volume")
    frame["open_interest"] = _num(frame, "open_interest")
    frame["clv_cents"] = _num(frame, "clv_cents")

    frame["price_bucket"] = _label_cut(frame["price_cents"], PRICE_BINS, PRICE_LABELS)
    frame["edge_bucket"] = _label_cut(frame["edge"], EDGE_BINS, EDGE_LABELS)
    frame["liquidity_bucket"] = _label_cut(frame["volume"], LIQUIDITY_BINS, LIQUIDITY_LABELS)
    spread_source = _num(frame, "fair_spread")
    frame["spread_bucket"] = _label_cut(spread_source, SPREAD_BINS, SPREAD_LABELS)
    frame.loc[spread_source.isna(), "spread_bucket"] = "unknown"
    frame["snapshot_bucket"] = frame.get("snapshot_target", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    frame["confidence_bucket"] = frame.get("fair_confidence", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    frame["fair_recommendation_bucket"] = (
        frame.get("ungated_recommendation", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    )
    frame["proof_gate_bucket"] = frame.get("proof_gate_status", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    frame["positive_clv"] = frame["clv_cents"] > 0
    frame["profitable_trade"] = frame["profit"] > 0
    frame["too_small_to_trust"] = False
    return frame.reset_index(drop=True)


def summarize_slice(frame: pd.DataFrame, group_columns: list[str] | None = None, min_rows: int = 30) -> pd.DataFrame:
    """Summarize trade, CLV, calibration, and Brier metrics by a segment."""

    group_columns = group_columns or []
    columns = group_columns + [
        "rows",
        "trade_count",
        "win_rate",
        "profit",
        "amount_risked",
        "roi_on_amount_risked",
        "average_profit_per_trade",
        "average_clv_cents",
        "positive_clv_rate",
        "average_edge",
        "average_model_prob",
        "average_market_prob",
        "calibration_error",
        "brier_score",
        "small_sample_warning",
        "prior_period_check_status",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    groups = [((), frame)] if not group_columns else frame.groupby(group_columns, dropna=False, observed=False)
    rows: list[dict[str, Any]] = []
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        trade_rows = group[group["trade_bool"]].copy()
        outcome_rows = trade_rows[trade_rows["contract_won"].notna() & trade_rows["model_prob"].notna()]
        amount_risked = float(trade_rows["cost"].sum()) if not trade_rows.empty else 0.0
        profit = float(trade_rows["profit"].sum()) if not trade_rows.empty else 0.0
        trade_count = int(len(trade_rows))
        result = {column: str(value) for column, value in zip(group_columns, keys)}
        win_rate = float(trade_rows["profitable_trade"].mean()) if trade_count else np.nan
        contract_win_rate = float(outcome_rows["contract_won"].mean()) if not outcome_rows.empty else np.nan
        avg_model = float(outcome_rows["model_prob"].mean()) if not outcome_rows.empty else np.nan
        brier = (
            float(((outcome_rows["model_prob"] - outcome_rows["contract_won"]) ** 2).mean())
            if not outcome_rows.empty
            else np.nan
        )
        clv_rows = trade_rows[trade_rows["clv_cents"].notna()]
        result.update(
            {
                "rows": int(len(group)),
                "trade_count": trade_count,
                "win_rate": win_rate,
                "profit": profit,
                "amount_risked": amount_risked,
                "roi_on_amount_risked": profit / amount_risked if amount_risked else np.nan,
                "average_profit_per_trade": profit / trade_count if trade_count else np.nan,
                "average_clv_cents": float(clv_rows["clv_cents"].mean()) if not clv_rows.empty else np.nan,
                "positive_clv_rate": float(clv_rows["positive_clv"].mean()) if not clv_rows.empty else np.nan,
                "average_edge": float(group["edge"].mean()) if group["edge"].notna().any() else np.nan,
                "average_model_prob": avg_model,
                "average_market_prob": float(trade_rows["market_prob"].mean()) if trade_rows["market_prob"].notna().any() else np.nan,
                "calibration_error": contract_win_rate - avg_model if pd.notna(contract_win_rate) and pd.notna(avg_model) else np.nan,
                "brier_score": brier,
                "small_sample_warning": bool(trade_count < min_rows),
                "prior_period_check_status": "not_checked",
            }
        )
        rows.append(result)
    return pd.DataFrame(rows, columns=columns)


def build_failure_segments(
    diagnostics: pd.DataFrame,
    min_rows: int = 30,
    segment_sets: list[list[str]] | None = None,
) -> pd.DataFrame:
    """Rank segments by negative CLV, weak positive-CLV frequency, losses, and overconfidence."""

    segment_sets = segment_sets or DEFAULT_SEGMENTS
    outputs: list[pd.DataFrame] = []
    for columns in segment_sets:
        if not all(column in diagnostics.columns for column in columns):
            continue
        summary = summarize_slice(diagnostics, columns, min_rows=min_rows)
        if summary.empty:
            continue
        summary.insert(0, "segment", "+".join(columns))
        outputs.append(summary)
    if not outputs:
        return pd.DataFrame()

    combined = pd.concat(outputs, ignore_index=True, sort=False)
    clv = pd.to_numeric(combined["average_clv_cents"], errors="coerce").fillna(0.0)
    pos_clv = pd.to_numeric(combined["positive_clv_rate"], errors="coerce").fillna(0.0)
    roi = pd.to_numeric(combined["roi_on_amount_risked"], errors="coerce").fillna(0.0)
    calibration = pd.to_numeric(combined["calibration_error"], errors="coerce").fillna(0.0)
    trades = pd.to_numeric(combined["trade_count"], errors="coerce").fillna(0.0)
    combined["failure_score"] = (
        np.maximum(0.0, -clv / 2.0)
        + np.maximum(0.0, 0.50 - pos_clv)
        + np.maximum(0.0, -roi)
        + np.maximum(0.0, -calibration)
        + np.minimum(trades, 100.0) / 500.0
    )
    combined["impact_label"] = np.where(
        combined["small_sample_warning"],
        "research_only_small_sample",
        np.where(combined["average_clv_cents"] < 0, "negative_clv_driver", "profit_or_calibration_driver"),
    )
    combined["affected_trade_count"] = combined["trade_count"]
    combined["analysis_confidence"] = np.where(
        trades >= min_rows,
        "medium",
        np.where(trades >= max(10, min_rows // 3), "low", "very_low"),
    )
    combined.loc[(trades >= min_rows) & (clv < 0) & (pos_clv < 0.35), "analysis_confidence"] = "high"
    combined["proof_gate_connection"] = np.select(
        [
            (clv < 0) & (pos_clv < 0.50) & (roi < 0),
            (clv < 0) & (pos_clv < 0.50),
            roi < 0,
            calibration < 0,
        ],
        [
            "strategy_backtest_profit,average_clv,positive_clv_rate",
            "average_clv,positive_clv_rate",
            "strategy_backtest_profit",
            "calibrated_strategy_readiness",
        ],
        default="repeatability_months",
    )
    combined["recommended_fix"] = np.select(
        [
            combined["segment"].astype(str).str.contains("liquidity", case=False, regex=False),
            combined["segment"].astype(str).str.contains("month", case=False, regex=False),
            combined["segment"].astype(str).str.contains("edge_bucket", case=False, regex=False),
            combined["segment"].astype(str).str.contains("diagnostic_side", case=False, regex=False),
        ],
        [
            "Tighten liquidity sizing and require later-month validation before promotion.",
            "Investigate calendar or availability regime drift; do not add a month filter from full sample only.",
            "Recalibrate high-edge bins and add uncertainty shrinkage before increasing bet count.",
            "Use side-specific calibration and side-specific uncertainty penalties.",
        ],
        default="Treat as root-cause evidence only; require prior-period validation before any rule change.",
    )
    combined["impact_evidence"] = (
        "profit="
        + combined["profit"].map(lambda value: f"{_safe_float(value):+.2f}")
        + "; avg_clv="
        + combined["average_clv_cents"].map(lambda value: f"{_safe_float(value):+.3f}")
        + "; positive_clv="
        + combined["positive_clv_rate"].map(lambda value: f"{_safe_float(value):.1%}")
    )
    combined["prior_period_check_status"] = "research_only_not_walk_forward_validated"
    return combined.sort_values(["failure_score", "trade_count"], ascending=[False, False]).reset_index(drop=True)


def build_walk_forward_slices(
    diagnostics: pd.DataFrame,
    segment_sets: list[list[str]] | None = None,
    min_prior_rows: int = 30,
    min_eval_rows: int = 10,
    min_positive_clv_rate: float = 0.50,
    min_avg_clv_cents: float = 0.0,
) -> pd.DataFrame:
    """Check whether a segment that looked good in prior months also worked later."""

    if diagnostics.empty or "month" not in diagnostics.columns:
        return pd.DataFrame()

    segment_sets = segment_sets or WALK_FORWARD_SEGMENTS
    trade_rows = diagnostics[diagnostics["trade_bool"] & diagnostics["date"].notna()].copy()
    if trade_rows.empty:
        return pd.DataFrame()

    months = sorted(trade_rows["month"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for columns in segment_sets:
        if not all(column in trade_rows.columns for column in columns):
            continue
        for key, _ in trade_rows.groupby(columns, dropna=False, observed=False):
            keys = key if isinstance(key, tuple) else (key,)
            mask = pd.Series(True, index=trade_rows.index)
            for column, value in zip(columns, keys):
                mask &= trade_rows[column].astype(str).eq(str(value))
            segment_rows = trade_rows[mask].copy()
            for month in months:
                prior = segment_rows[segment_rows["month"] < month]
                current = segment_rows[segment_rows["month"].eq(month)]
                if current.empty:
                    continue
                prior_clv = prior["clv_cents"].dropna()
                current_clv = current["clv_cents"].dropna()
                prior_rows = int(len(prior))
                eval_rows = int(len(current))
                prior_pass = (
                    prior_rows >= min_prior_rows
                    and not prior_clv.empty
                    and float(prior_clv.mean()) > min_avg_clv_cents
                    and float((prior_clv > 0).mean()) > min_positive_clv_rate
                )
                eval_pass = (
                    eval_rows >= min_eval_rows
                    and not current_clv.empty
                    and float(current_clv.mean()) > min_avg_clv_cents
                    and float((current_clv > 0).mean()) > min_positive_clv_rate
                )
                row = {
                    "segment": "+".join(columns),
                    "month": month,
                    "prior_rows": prior_rows,
                    "eval_rows": eval_rows,
                    "prior_avg_clv_cents": float(prior_clv.mean()) if not prior_clv.empty else np.nan,
                    "prior_positive_clv_rate": float((prior_clv > 0).mean()) if not prior_clv.empty else np.nan,
                    "eval_avg_clv_cents": float(current_clv.mean()) if not current_clv.empty else np.nan,
                    "eval_positive_clv_rate": float((current_clv > 0).mean()) if not current_clv.empty else np.nan,
                    "eval_profit": float(current["profit"].sum()),
                    "small_sample_warning": bool(prior_rows < min_prior_rows or eval_rows < min_eval_rows),
                    "prior_period_selected": bool(prior_pass),
                    "survived_eval": bool(prior_pass and eval_pass),
                    "status": "validated" if prior_pass and eval_pass else "research_only",
                }
                for column, value in zip(columns, keys):
                    row[column] = str(value)
                rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values(["survived_eval", "prior_period_selected", "eval_rows"], ascending=[False, False, False])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _side_metrics(overall: pd.DataFrame, side: str) -> dict[str, Any]:
    if overall.empty or "diagnostic_side" not in overall.columns:
        return {}
    row = overall[overall["diagnostic_side"].eq(side)]
    if row.empty:
        return {}
    item = row.iloc[0]
    return {
        "trade_count": int(item.get("trade_count", 0) or 0),
        "profit": _safe_float(item.get("profit")),
        "average_clv_cents": _safe_float(item.get("average_clv_cents")),
        "positive_clv_rate": _safe_float(item.get("positive_clv_rate")),
        "roi_on_amount_risked": _safe_float(item.get("roi_on_amount_risked")),
        "calibration_error": _safe_float(item.get("calibration_error")),
    }


def build_recommendations_markdown(
    summary: dict[str, Any],
    failure_segments: pd.DataFrame,
    walk_forward: pd.DataFrame,
) -> str:
    """Build a human-readable root-cause report without promoting betting rules."""

    top = failure_segments.head(8).copy() if not failure_segments.empty else pd.DataFrame()
    validated = int(walk_forward["survived_eval"].sum()) if not walk_forward.empty and "survived_eval" in walk_forward.columns else 0
    lines = [
        "# Single-Game Edge Diagnostics",
        "",
        f"Generated: {summary.get('generated_date', '')}",
        "",
        "## Verdict",
        "",
        f"- Edge proven: {summary.get('single_game_edge_proven', False)}",
        f"- Proof status: {summary.get('proof_status', 'unknown')}",
        f"- Fair-price recommendations: {summary.get('actionable_fair_price_bets', 0)} actionable, blocked while proof fails.",
        f"- Parlay status: {summary.get('parlay_status', 'unknown')}",
        "",
        "## Root Cause",
        "",
        (
            "The market data path is no longer the primary suspect: the canonical backtest is marked Kalshi "
            "bid/ask, the market truth audit has broad usable pregame coverage, and the fair-price layer validates "
            "the canonical source before writing signals. The current failure is strategy quality under tradable "
            "prices: total profit is negative, average CLV is negative, positive CLV frequency is far below 50%, "
            "and repeatability gates fail."
        ),
        "",
        "The most actionable pattern is not a new bet rule. YES trades lose CLV, NO trades show somewhat better CLV "
        "but remain overconfident on settlement outcomes. High-looking edge buckets do not reliably translate into "
        "closing-line value or repeatable profit.",
        "",
        "## Top Failure Segments",
        "",
    ]
    if top.empty:
        lines.append("No segment rows were available.")
    else:
        for _, row in top.iterrows():
            labels = []
            for column in [
                "diagnostic_side",
                "price_bucket",
                "edge_bucket",
                "liquidity_bucket",
                "month",
                "yes_team_abbr",
                "snapshot_bucket",
                "fair_recommendation_bucket",
            ]:
                if column in row and pd.notna(row[column]):
                    labels.append(f"{column}={row[column]}")
            lines.append(
                "- "
                + ", ".join(labels)
                + f": trades={int(row.get('trade_count', 0) or 0)}, "
                + f"profit={_safe_float(row.get('profit')):+.2f}, "
                + f"avg_clv={_safe_float(row.get('average_clv_cents')):+.3f} cents, "
                + f"positive_clv={_safe_float(row.get('positive_clv_rate')):.1%}, "
                + f"status={row.get('prior_period_check_status', 'research_only')}"
            )
    lines.extend(
        [
            "",
            "## Walk-Forward Check",
            "",
            f"Validated prior-period slices: {validated}",
            "",
            "Any segment that only works on the full sample remains research-only. The next implementation should "
            "change calibration and uncertainty penalties, then re-run proof gates, rather than suppressing slices "
            "because they looked bad or good in-sample.",
            "",
            "## Targeted Improvement Plan",
            "",
            "1. Rework side-specific calibration first: shrink NO probabilities where settlement outcomes trail "
            "forecast win rates, and add a stronger YES uncertainty penalty where CLV is repeatedly negative.",
            "2. Add availability and late-scratch inputs before trusting large model-market disagreements.",
            "3. Treat high edge buckets as hypotheses, not automatic bets; require prior-period CLV selection and "
            "later-month validation before any filter is promoted.",
            "4. Keep fair-price output blocked until profit, average CLV, positive CLV rate, readiness, and "
            "repeatability gates all pass together.",
            "5. Keep parlays blocked. No parlay research should be optimized until single-game gates pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_single_game_edge_diagnostics(
    trades: pd.DataFrame,
    fair_price_signals: pd.DataFrame | None = None,
    backtest_summary: dict[str, Any] | None = None,
    clv_summary: dict[str, Any] | None = None,
    proof_summary: dict[str, Any] | None = None,
    proof_gates: pd.DataFrame | None = None,
    parlay_summary: dict[str, Any] | None = None,
    generated_date: str = "",
    min_rows: int = 30,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Build all single-game edge diagnostic artifacts."""

    backtest_summary = backtest_summary or {}
    clv_summary = clv_summary or {}
    proof_summary = proof_summary or {}
    parlay_summary = parlay_summary or {}

    diagnostics = prepare_diagnostics_rows(trades, fair_price_signals)
    overall = summarize_slice(diagnostics, [], min_rows=min_rows)
    by_side = summarize_slice(diagnostics, ["diagnostic_side"], min_rows=min_rows)
    failure_segments = build_failure_segments(diagnostics, min_rows=min_rows)
    walk_forward = build_walk_forward_slices(diagnostics, min_prior_rows=min_rows)
    if not walk_forward.empty and not failure_segments.empty:
        validated_keys = set(
            walk_forward.loc[walk_forward["survived_eval"], "segment"].astype(str).tolist()
        )
        failure_segments["prior_period_check_status"] = np.where(
            failure_segments["segment"].astype(str).isin(validated_keys),
            "has_validated_related_segment",
            "research_only_not_walk_forward_validated",
        )

    overall_row = overall.iloc[0] if not overall.empty else {}
    proof_failed = proof_summary.get("failed_gates", [])
    if proof_gates is not None and not proof_gates.empty and "passed" in proof_gates.columns:
        passed_count = int(proof_gates["passed"].map(_as_bool).sum())
        failed_count = int((~proof_gates["passed"].map(_as_bool)).sum())
    else:
        passed_count = 0
        failed_count = int(proof_summary.get("hard_failures", 0) or 0)

    summary = {
        "generated_date": generated_date,
        "status": "not_proven" if not bool(proof_summary.get("single_game_edge_proven", False)) else "single_game_edge_proven",
        "single_game_edge_proven": bool(proof_summary.get("single_game_edge_proven", False)),
        "proof_status": proof_summary.get("status", "unknown"),
        "proof_gates_passed": passed_count,
        "proof_gates_failed": failed_count,
        "failed_gates": proof_failed,
        "market_source": backtest_summary.get("market_source", "unknown"),
        "price_source": backtest_summary.get("price_source", "unknown"),
        "canonical_kalshi_backtest": bool(backtest_summary.get("canonical_kalshi_backtest", False)),
        "trades": int(overall_row.get("trade_count", 0) or 0),
        "ending_bankroll": _safe_float(backtest_summary.get("ending_bankroll")),
        "starting_bankroll": _safe_float(backtest_summary.get("starting_bankroll")),
        "roi_on_amount_risked": _safe_float(overall_row.get("roi_on_amount_risked")),
        "profit": _safe_float(overall_row.get("profit")),
        "average_profit_per_trade": _safe_float(overall_row.get("average_profit_per_trade")),
        "average_clv_cents": _safe_float(clv_summary.get("avg_clv_cents"), _safe_float(overall_row.get("average_clv_cents"))),
        "positive_clv_rate": _safe_float(
            clv_summary.get("positive_clv_rate"),
            _safe_float(overall_row.get("positive_clv_rate")),
        ),
        "calibration_error": _safe_float(overall_row.get("calibration_error")),
        "brier_score": _safe_float(overall_row.get("brier_score")),
        "yes": _side_metrics(by_side, "YES"),
        "no": _side_metrics(by_side, "NO"),
        "actionable_fair_price_bets": int(
            fair_price_signals["recommendation"].astype(str).str.startswith("Bet ").sum()
        )
        if fair_price_signals is not None and not fair_price_signals.empty and "recommendation" in fair_price_signals.columns
        else 0,
        "ungated_fair_price_bets": int(
            fair_price_signals["ungated_recommendation"].astype(str).str.startswith("Bet ").sum()
        )
        if fair_price_signals is not None and not fair_price_signals.empty and "ungated_recommendation" in fair_price_signals.columns
        else 0,
        "parlay_status": parlay_summary.get("status", "unknown"),
        "parlay_recommendations_allowed": bool(parlay_summary.get("parlay_recommendations_allowed", False)),
        "walk_forward_validated_slices": int(walk_forward["survived_eval"].sum())
        if not walk_forward.empty and "survived_eval" in walk_forward.columns
        else 0,
        "small_sample_segment_rows": int(failure_segments["small_sample_warning"].sum())
        if not failure_segments.empty and "small_sample_warning" in failure_segments.columns
        else 0,
        "root_cause_verdict": (
            "Single-game edge is not proven. Canonical Kalshi bid/ask data is usable, but the strategy fails "
            "profit, CLV frequency, average CLV, calibration/readiness, and repeatability checks."
        ),
        "recommended_next_work": [
            "Recalibrate side-specific probabilities and add uncertainty penalties before changing bet filters.",
            "Add availability inputs to reduce false confidence before games.",
            "Promote no slice filter unless it passes a prior-period and later-month walk-forward check.",
            "Keep fair-price and parlay recommendations blocked until proof gates pass.",
        ],
    }
    if not failure_segments.empty:
        summary["top_failure_drivers"] = failure_segments.head(10).to_dict(orient="records")
    recommendations = build_recommendations_markdown(summary, failure_segments, walk_forward)
    return summary, diagnostics, failure_segments, walk_forward, recommendations


def save_single_game_edge_diagnostics(
    summary: dict[str, Any],
    diagnostics: pd.DataFrame,
    failure_segments: pd.DataFrame,
    walk_forward: pd.DataFrame,
    recommendations_markdown: str,
    output_dir: str | Path,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_root / "single_game_edge_diagnostics.csv", index=False)
    failure_segments.to_csv(output_root / "single_game_edge_failure_segments.csv", index=False)
    walk_forward.to_csv(output_root / "single_game_edge_walk_forward_slices.csv", index=False)
    (output_root / "single_game_edge_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output_root / "single_game_edge_recommendations.md").write_text(
        recommendations_markdown,
        encoding="utf-8",
    )
