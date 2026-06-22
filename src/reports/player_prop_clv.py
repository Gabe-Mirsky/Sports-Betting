"""Player-prop closing-line-value (CLV) calculation (research-only).

For every prop market (league/player/prop_type/bookmaker/game) that has BOTH
an early snapshot and a closing-like snapshot, compares the early line/prices
against the closing line/prices:

- line movement (early line vs closing line, direction flagged)
- over/under price movement (direction flagged)
- CLV for the over and the under, computed from decimal prices as
  ``early_price / closing_price - 1`` (positive = the early price beat the
  close). Price CLV is only computed when the early and closing snapshots
  quote the SAME line — when the line itself moved, price differences are not
  apples-to-apples, so the market is flagged ``line_changed`` and excluded
  from price-CLV averages instead of being silently mixed in.

Settlement is NOT required: CLV compares prices, not outcomes.

If no market has both snapshot types yet, the reports are still written and
say so honestly (what is missing, when to collect next).

Research-only: no models, no recommendations, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked. CLV here is a
data-quality/measurement tool, not evidence of edge.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CLV_VERSION = "v1"

GROUP_KEYS = ["league", "player_name", "prop_type", "bookmaker", "canonical_game_key"]

# A snapshot is "early" for CLV when taken more than this many minutes before
# tip (mirrors the collector's closing window).
CLOSING_WINDOW_MINUTES = 60.0

OUTPUT_FILES = {
    "summary_json": "player_prop_clv_summary.json",
    "clv_csv": "player_prop_clv.csv",
    "by_bookmaker_csv": "player_prop_clv_by_bookmaker.csv",
    "by_prop_type_csv": "player_prop_clv_by_prop_type.csv",
    "clv_md": "player_prop_clv.md",
}

CLV_COLUMNS = GROUP_KEYS + [
    "game_date",
    "early_snapshot_time",
    "closing_snapshot_time",
    "early_minutes_to_tip",
    "closing_minutes_to_tip",
    "early_line",
    "closing_line",
    "line_move",
    "line_move_direction",
    "early_over_price",
    "closing_over_price",
    "over_price_move",
    "over_price_move_direction",
    "clv_over_pct",
    "early_under_price",
    "closing_under_price",
    "under_price_move",
    "under_price_move_direction",
    "clv_under_pct",
    "price_clv_comparable",
]


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _direction(move: float | None) -> str:
    if move is None or pd.isna(move):
        return "unknown"
    if move > 0:
        return "up"
    if move < 0:
        return "down"
    return "flat"


def _clv_pct(early_price: float | None, closing_price: float | None) -> float | None:
    """Decimal-odds CLV: positive when the early price beat the close."""

    if (
        early_price is None
        or closing_price is None
        or pd.isna(early_price)
        or pd.isna(closing_price)
        or closing_price <= 0
    ):
        return None
    return round(float(early_price) / float(closing_price) - 1.0, 4)


def _prepare(snaps: pd.DataFrame) -> pd.DataFrame:
    frame = snaps.copy()
    for key in GROUP_KEYS + ["game_date"]:
        if key not in frame.columns:
            frame[key] = "(missing)"
        frame[key] = frame[key].fillna("(missing)").astype(str)
    frame["line"] = pd.to_numeric(frame.get("line"), errors="coerce")
    frame["over_price"] = pd.to_numeric(frame.get("over_price"), errors="coerce")
    frame["under_price"] = pd.to_numeric(frame.get("under_price"), errors="coerce")
    frame["_minutes"] = pd.to_numeric(frame.get("minutes_to_game_start"), errors="coerce")
    frame["_ts"] = pd.to_datetime(frame.get("snapshot_time"), errors="coerce", utc=True)
    closing = frame.get("is_closing_snapshot")
    frame["_closing"] = _truthy(closing) if closing is not None else False
    return frame


def _pick_snapshot(rows: pd.DataFrame, *, latest: bool) -> pd.Series | None:
    """Pick one snapshot row: prefer rows with both prices, then by recency.

    ``latest=False`` returns the earliest pull (for the early side);
    ``latest=True`` the latest pull (for the closing side). When the same pull
    time carries several lines (an alt ladder), the line with the most
    balanced over/under prices wins — the main-line signature.
    """

    candidates = rows[rows["line"].notna()]
    if candidates.empty:
        return None
    priced = candidates[candidates["over_price"].notna() & candidates["under_price"].notna()]
    pool = priced if not priced.empty else candidates
    ts = pool["_ts"]
    if ts.notna().any():
        target_ts = ts.max() if latest else ts.min()
        batch = pool[ts == target_ts]
    else:
        # No parseable snapshot times: fall back to minutes-to-tip ordering.
        minutes = pool["_minutes"]
        if minutes.notna().any():
            target = minutes.min() if latest else minutes.max()
            batch = pool[minutes == target]
        else:
            batch = pool
    if len(batch) > 1 and batch["over_price"].notna().any() and batch["under_price"].notna().any():
        balance = (batch["over_price"] - batch["under_price"]).abs()
        return batch.loc[balance.idxmin()]
    return batch.iloc[0]


def compute_market_clv(group: pd.DataFrame) -> dict[str, Any] | None:
    """CLV record for one market, or None when early/closing sides are missing."""

    early_rows = group[(~group["_closing"]) & (group["_minutes"].isna() | (group["_minutes"] > CLOSING_WINDOW_MINUTES))]
    closing_rows = group[group["_closing"]]
    if early_rows.empty or closing_rows.empty:
        return None
    early = _pick_snapshot(early_rows, latest=False)
    closing = _pick_snapshot(closing_rows, latest=True)
    if early is None or closing is None:
        return None

    early_line = float(early["line"])
    closing_line = float(closing["line"])
    line_move = round(closing_line - early_line, 4)
    same_line = early_line == closing_line

    over_move = (
        round(float(closing["over_price"]) - float(early["over_price"]), 4)
        if pd.notna(closing["over_price"]) and pd.notna(early["over_price"])
        else None
    )
    under_move = (
        round(float(closing["under_price"]) - float(early["under_price"]), 4)
        if pd.notna(closing["under_price"]) and pd.notna(early["under_price"])
        else None
    )

    clv_over = _clv_pct(early["over_price"], closing["over_price"]) if same_line else None
    clv_under = _clv_pct(early["under_price"], closing["under_price"]) if same_line else None

    return {
        "game_date": group["game_date"].iloc[0],
        "early_snapshot_time": early["_ts"].isoformat() if pd.notna(early["_ts"]) else "",
        "closing_snapshot_time": closing["_ts"].isoformat() if pd.notna(closing["_ts"]) else "",
        "early_minutes_to_tip": round(float(early["_minutes"]), 2) if pd.notna(early["_minutes"]) else None,
        "closing_minutes_to_tip": round(float(closing["_minutes"]), 2) if pd.notna(closing["_minutes"]) else None,
        "early_line": early_line,
        "closing_line": closing_line,
        "line_move": line_move,
        "line_move_direction": _direction(line_move),
        "early_over_price": float(early["over_price"]) if pd.notna(early["over_price"]) else None,
        "closing_over_price": float(closing["over_price"]) if pd.notna(closing["over_price"]) else None,
        "over_price_move": over_move,
        "over_price_move_direction": _direction(over_move),
        "clv_over_pct": clv_over,
        "early_under_price": float(early["under_price"]) if pd.notna(early["under_price"]) else None,
        "closing_under_price": float(closing["under_price"]) if pd.notna(closing["under_price"]) else None,
        "under_price_move": under_move,
        "under_price_move_direction": _direction(under_move),
        "clv_under_pct": clv_under,
        "price_clv_comparable": same_line,
    }


def build_clv_frame(snaps: pd.DataFrame) -> pd.DataFrame:
    """One CLV row per market with both early and closing-like snapshots."""

    if snaps.empty:
        return pd.DataFrame(columns=CLV_COLUMNS)
    frame = _prepare(snaps)
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(GROUP_KEYS, dropna=False, sort=True):
        record = compute_market_clv(group)
        if record is None:
            continue
        records.append({**dict(zip(GROUP_KEYS, keys)), **record})
    return pd.DataFrame(records, columns=CLV_COLUMNS)


def _aggregate(clv: pd.DataFrame, by: str) -> pd.DataFrame:
    columns = [
        by, "markets", "price_clv_comparable_markets", "line_moved_markets",
        "lines_up", "lines_down", "lines_flat",
        "avg_clv_over_pct", "avg_clv_under_pct", "avg_abs_line_move",
    ]
    if clv.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for value, group in clv.groupby(clv[by].astype(str), sort=True):
        comparable = group[group["price_clv_comparable"].astype(bool)]
        over_vals = pd.to_numeric(comparable["clv_over_pct"], errors="coerce").dropna()
        under_vals = pd.to_numeric(comparable["clv_under_pct"], errors="coerce").dropna()
        moves = pd.to_numeric(group["line_move"], errors="coerce").dropna()
        records.append(
            {
                by: value,
                "markets": int(len(group)),
                "price_clv_comparable_markets": int(len(comparable)),
                "line_moved_markets": int((moves != 0).sum()),
                "lines_up": int((moves > 0).sum()),
                "lines_down": int((moves < 0).sum()),
                "lines_flat": int((moves == 0).sum()),
                "avg_clv_over_pct": round(float(over_vals.mean()), 4) if len(over_vals) else None,
                "avg_clv_under_pct": round(float(under_vals.mean()), 4) if len(under_vals) else None,
                "avg_abs_line_move": round(float(moves.abs().mean()), 4) if len(moves) else None,
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_clv_summary(
    snaps: pd.DataFrame,
    clv: pd.DataFrame,
    next_collection_hint: str | None = None,
) -> dict[str, Any]:
    """Assemble the CLV summary, honest about not-ready states."""

    warnings: list[str] = []
    nba_closing = 0
    nba_markets_total = 0
    if not snaps.empty and "league" in snaps.columns:
        frame = _prepare(snaps)
        nba = frame[frame["league"].str.upper().eq("NBA")]
        nba_closing = int(nba["_closing"].sum())
        nba_markets_total = int(nba.groupby(GROUP_KEYS).ngroups) if not nba.empty else 0

    clv_ready = not clv.empty
    nba_clv_rows = clv[clv["league"].astype(str).str.upper().eq("NBA")] if clv_ready else clv
    if not clv_ready:
        reason = (
            "no market has both an early snapshot and a closing-like snapshot yet"
            if nba_closing == 0
            else "closing-like snapshots exist but none pair with an early snapshot on the same market"
        )
        missing = (
            "NBA closing-like snapshots (within 60 minutes of tip)"
            if nba_closing == 0
            else "early/closing pairs on the same league/player/prop/bookmaker/game market"
        )
        verdict = f"CLV NOT READY: {reason}."
        warnings.append(verdict)
        warnings.append(f"Missing data: {missing}.")
        if next_collection_hint:
            warnings.append(f"Next collection window needed: {next_collection_hint}.")
        else:
            warnings.append(
                "Next collection window needed: run collection inside the 60-minute pre-tip "
                "window on the next NBA game day (see data/reports/nba_prop_closing_collection_plan.md)."
            )
    else:
        comparable = clv[clv["price_clv_comparable"].astype(bool)]
        verdict = (
            f"CLV computed for {len(clv)} market(s) ({len(nba_clv_rows)} NBA); "
            f"{len(comparable)} have same-line price CLV, "
            f"{len(clv) - len(comparable)} had line changes (price CLV not comparable)."
        )
        if len(clv) < 25:
            warnings.append(
                f"SMALL SAMPLE: only {len(clv)} CLV markets; averages are noise at this size."
            )

    comparable = clv[clv["price_clv_comparable"].astype(bool)] if clv_ready else clv
    over_vals = (
        pd.to_numeric(comparable["clv_over_pct"], errors="coerce").dropna()
        if clv_ready
        else pd.Series(dtype=float)
    )
    under_vals = (
        pd.to_numeric(comparable["clv_under_pct"], errors="coerce").dropna()
        if clv_ready
        else pd.Series(dtype=float)
    )

    return {
        "report": "player_prop_clv_summary",
        "clv_version": CLV_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clv_ready": clv_ready,
        "verdict": verdict,
        "markets_with_clv": int(len(clv)),
        "nba_markets_with_clv": int(len(nba_clv_rows)),
        "nba_closing_like_snapshots": nba_closing,
        "nba_markets_total": nba_markets_total,
        "price_clv_comparable_markets": int(len(comparable)) if clv_ready else 0,
        "line_changed_markets": int((~clv["price_clv_comparable"].astype(bool)).sum()) if clv_ready else 0,
        "avg_clv_over_pct": round(float(over_vals.mean()), 4) if len(over_vals) else None,
        "avg_clv_under_pct": round(float(under_vals.mean()), 4) if len(under_vals) else None,
        "markets_with_clv_by_league": (
            clv.groupby("league").size().astype(int).to_dict() if clv_ready else {}
        ),
        "warnings": warnings,
        "settlement_not_required": True,
        "research_only": True,
        "approved": False,
        "notes": [
            "CLV compares early vs closing prices; settlement is not required.",
            "Price CLV is only computed when early and closing quote the same line.",
            "Line changes are flagged separately, never mixed into price CLV.",
            "Research-only measurement: CLV here is not evidence of edge.",
        ],
    }


def _render_markdown(summary: dict[str, Any], clv: pd.DataFrame) -> str:
    lines = [
        "# Player Prop CLV Report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "_Research-only CLV measurement. No models, recommendations, approved bets, or parlays._",
        "",
        f"**Status: {'READY' if summary['clv_ready'] else 'NOT READY'}** — {summary['verdict']}",
        "",
        "## Totals",
        "",
        f"- Markets with CLV: {summary['markets_with_clv']} (NBA: {summary['nba_markets_with_clv']})",
        f"- Same-line price-CLV markets: {summary['price_clv_comparable_markets']}",
        f"- Line-changed markets (price CLV excluded): {summary['line_changed_markets']}",
        f"- NBA closing-like snapshots: {summary['nba_closing_like_snapshots']}",
        f"- Avg over CLV (same line): "
        f"{summary['avg_clv_over_pct'] if summary['avg_clv_over_pct'] is not None else 'n/a'}",
        f"- Avg under CLV (same line): "
        f"{summary['avg_clv_under_pct'] if summary['avg_clv_under_pct'] is not None else 'n/a'}",
        "",
    ]
    if not clv.empty:
        lines += [
            "## Markets",
            "",
            "| market | line early→close | move | over CLV | under CLV | comparable |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in clv.head(50).iterrows():
            market = f"{row['league']} {row['player_name']} {row['prop_type']} @ {row['bookmaker']}"
            over = f"{row['clv_over_pct']:+.2%}" if pd.notna(row["clv_over_pct"]) else "n/a"
            under = f"{row['clv_under_pct']:+.2%}" if pd.notna(row["clv_under_pct"]) else "n/a"
            lines.append(
                f"| {market} | {row['early_line']:g} → {row['closing_line']:g} | "
                f"{row['line_move_direction']} | {over} | {under} | "
                f"{'yes' if row['price_clv_comparable'] else 'LINE CHANGED'} |"
            )
        if len(clv) > 50:
            lines.append(f"| … {len(clv) - 50} more market(s) in the CSV … | | | | | |")
        lines.append("")
    if summary["warnings"]:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in summary["warnings"]]
        lines.append("")
    lines += [
        "---",
        "Research-only: CLV measures collection quality (did we capture line movement?),",
        "not betting edge. Approved bets and approved parlays remain blocked.",
        "",
    ]
    return "\n".join(lines)


def write_clv_reports(project_root: str | Path) -> dict[str, Any]:
    """Read snapshots, compute CLV, and write all five outputs."""

    root = Path(project_root)
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    snapshots_path = root / "data" / "processed" / "player_prop_snapshots_normalized.csv"
    snaps = pd.read_csv(snapshots_path, low_memory=False) if snapshots_path.exists() else pd.DataFrame()

    next_hint: str | None = None
    readiness_path = reports / "nba_prop_clv_readiness_summary.json"
    if readiness_path.exists():
        try:
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            next_hint = readiness.get("next_recommended_collection_time_utc")
        except (json.JSONDecodeError, OSError):
            next_hint = None

    clv = build_clv_frame(snaps)
    summary = build_clv_summary(snaps, clv, next_collection_hint=next_hint)
    by_bookmaker = _aggregate(clv, "bookmaker")
    by_prop_type = _aggregate(clv, "prop_type")

    outputs = {key: reports / filename for key, filename in OUTPUT_FILES.items()}
    clv.to_csv(outputs["clv_csv"], index=False)
    by_bookmaker.to_csv(outputs["by_bookmaker_csv"], index=False)
    by_prop_type.to_csv(outputs["by_prop_type_csv"], index=False)
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs["clv_md"].write_text(_render_markdown(summary, clv), encoding="utf-8")
    summary["outputs"] = {key: str(path.relative_to(root)) for key, path in outputs.items()}
    return summary
