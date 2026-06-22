"""World Cup closing-line-value (CLV) measurement (research-only).

Pairs the earliest ("opening") and the closing-like (or latest) snapshot for each
World Cup market outcome and measures price/line movement. CLV here is a research
measurement only: it is NOT evidence of edge, NOT a bet, and is kept completely
separate from the NBA model gates (own output files, no shared thresholds).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

KEY_COLS = ["event_id", "market_type", "bookmaker", "outcome_name"]


def _implied(price: float) -> float | None:
    try:
        p = float(price)
        return 1.0 / p if p > 0 else None
    except (TypeError, ValueError):
        return None


def build_world_cup_clv(snapshots: pd.DataFrame) -> dict[str, Any]:
    """Compute per-outcome opening->closing CLV from collected WC snapshots."""
    summary: dict[str, Any] = {
        "report": "world_cup_clv_summary",
        "research_only": True,
        "approved": False,
        "clv_ready": False,
        "markets_with_clv": 0,
        "total_outcomes": 0,
        "price_clv_comparable": 0,
        "line_changed": 0,
        "avg_price_clv": None,
        "avg_implied_prob_clv": None,
        "by_market_type": {},
        "by_bookmaker": {},
        "pairs": [],
        "isolated_from_nba_gates": True,
        "verdict": "",
        "notes": [
            "CLV compares opening vs closing/latest prices; settlement not required.",
            "Price CLV is only computed when opening and closing quote the same line.",
            "Research-only measurement: World Cup CLV is not evidence of edge and "
            "never feeds NBA model gates.",
        ],
    }
    if snapshots is None or snapshots.empty:
        summary["verdict"] = "No World Cup snapshots collected yet."
        return summary

    df = snapshots.copy()
    df["snapshot_time"] = pd.to_datetime(df.get("snapshot_time"), errors="coerce", utc=True)
    df = df[df["snapshot_time"].notna()]
    if df.empty:
        summary["verdict"] = "No snapshots with a parseable snapshot_time."
        return summary

    pairs: list[dict[str, Any]] = []
    comparable = 0
    line_changed = 0
    price_deltas: list[float] = []
    prob_deltas: list[float] = []

    for key, grp in df.groupby([df[c].astype(str) for c in KEY_COLS], sort=False):
        grp = grp.sort_values("snapshot_time")
        if grp["snapshot_time"].nunique() < 2:
            continue  # need at least two distinct snapshots for CLV
        opening = grp.iloc[0]
        closing_rows = grp[grp.get("is_closing_like").astype(str).str.lower().isin(["true", "1"])]
        closing = closing_rows.iloc[-1] if not closing_rows.empty else grp.iloc[-1]

        line_open, line_close = opening.get("line"), closing.get("line")
        same_line = (pd.isna(line_open) and pd.isna(line_close)) or (line_open == line_close)
        price_clv = None
        prob_clv = None
        if same_line:
            comparable += 1
            try:
                price_clv = float(closing.get("price")) - float(opening.get("price"))
                price_deltas.append(price_clv)
            except (TypeError, ValueError):
                price_clv = None
            io, ic = _implied(opening.get("price")), _implied(closing.get("price"))
            if io is not None and ic is not None:
                prob_clv = io - ic  # positive => price shortened vs open
                prob_deltas.append(prob_clv)
        else:
            line_changed += 1

        pairs.append({
            "event_id": opening.get("event_id"),
            "market_type": opening.get("market_type"),
            "bookmaker": opening.get("bookmaker"),
            "outcome_name": opening.get("outcome_name"),
            "opening_time": opening["snapshot_time"].isoformat(),
            "closing_time": closing["snapshot_time"].isoformat(),
            "opening_price": opening.get("price"),
            "closing_price": closing.get("price"),
            "line_open": line_open,
            "line_close": line_close,
            "line_moved": not same_line,
            "price_clv": price_clv,
            "implied_prob_clv": prob_clv,
        })

    summary["pairs"] = pairs
    summary["markets_with_clv"] = len(pairs)
    summary["total_outcomes"] = int(df.groupby([df[c].astype(str) for c in KEY_COLS]).ngroups)
    summary["price_clv_comparable"] = comparable
    summary["line_changed"] = line_changed
    summary["clv_ready"] = len(pairs) > 0
    if price_deltas:
        summary["avg_price_clv"] = round(sum(price_deltas) / len(price_deltas), 4)
    if prob_deltas:
        summary["avg_implied_prob_clv"] = round(sum(prob_deltas) / len(prob_deltas), 4)
    if pairs:
        pf = pd.DataFrame(pairs)
        graded = pf[pf["price_clv"].notna()]
        summary["by_market_type"] = {
            str(mt): round(float(g["price_clv"].mean()), 4)
            for mt, g in graded.groupby("market_type")
        } if not graded.empty else {}
        summary["by_bookmaker"] = {
            str(bk): round(float(g["price_clv"].mean()), 4)
            for bk, g in graded.groupby("bookmaker")
        } if not graded.empty else {}

    if summary["markets_with_clv"]:
        summary["verdict"] = (
            f"World Cup CLV computed for {summary['markets_with_clv']} outcome(s) "
            f"({comparable} same-line comparable, {line_changed} line-changed)."
        )
    else:
        summary["verdict"] = (
            "Not enough snapshots for World Cup CLV yet: each market has only one "
            "snapshot. CLV needs an early AND a later/closing snapshot of the same "
            "outcome. Keep the World Cup watcher running across the 24-48h and "
            "pre-kickoff windows."
        )
    return summary


def render_clv_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# World Cup CLV Summary",
        f"_Research-only; isolated from NBA model gates. clv_ready={summary.get('clv_ready')}._",
        "",
        f"- Outcomes tracked: {summary.get('total_outcomes')}",
        f"- Outcomes with CLV (>=2 snapshots): **{summary.get('markets_with_clv')}**",
        f"- Same-line comparable: {summary.get('price_clv_comparable')}; line-changed: {summary.get('line_changed')}",
        f"- Avg price CLV: {summary.get('avg_price_clv')}; avg implied-prob CLV: {summary.get('avg_implied_prob_clv')}",
        f"- By market type: {summary.get('by_market_type')}",
        f"- By bookmaker: {summary.get('by_bookmaker')}",
        "",
        f"**Verdict:** {summary.get('verdict')}",
        "",
        "_World Cup CLV is a research measurement only — not edge, not a bet, and it "
        "never feeds the NBA model gates._",
    ]
    return "\n".join(lines)
