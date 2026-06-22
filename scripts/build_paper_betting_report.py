"""Build a research-only paper betting / P&L report.

This report normalizes existing paper/backtest rows when they exist.  It does
not create picks.  When no recognized model or paper ledger exists, it writes a
valid blocked report so the dashboard can explain why the betting tab is empty.

Outputs:
    data/reports/paper_betting_report.json
    data/reports/paper_betting_report.csv
    data/reports/paper_betting_report.md
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

BET_COLUMNS = [
    "bet_date",
    "sport",
    "league",
    "player",
    "prop_type",
    "market",
    "line",
    "side",
    "odds",
    "model_probability",
    "market_probability",
    "edge",
    "expected_value",
    "result",
    "profit_loss",
    "clv",
    "closing_price",
    "confidence_tier",
    "source",
    "status",
    "label",
]

INPUT_CANDIDATES = [
    "player_prop_paper_bets.csv",
    "paper_betting_ledger.csv",
    "paper_betting_report_input.csv",
    "graded_single_recommendations.csv",
    "paper_trade_suggestions.csv",
]

PARLAY_CANDIDATES = [
    "graded_paper_parlays.csv",
    "paper_parlay_candidates.csv",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _first(row: pd.Series, columns: list[str], default: Any = "") -> Any:
    for column in columns:
        if column in row.index and pd.notna(row[column]) and str(row[column]) != "":
            return row[column]
    return default


def _boolish(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if float(numeric) == 1.0:
            return True
        if float(numeric) == 0.0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "won", "win"}:
        return True
    if text in {"false", "0", "no", "lost", "loss"}:
        return False
    return None


def _status_from_row(row: pd.Series) -> str:
    explicit = str(_first(row, ["status", "result_status"], "")).strip().lower()
    if explicit in {"pending", "won", "lost", "pushed", "push", "void", "blocked"}:
        return "pushed" if explicit == "push" else explicit
    won = _boolish(_first(row, ["won"], None))
    if won is True:
        return "won"
    if won is False:
        return "lost"
    result = str(_first(row, ["result"], "")).strip().lower()
    if result in {"won", "lost", "pushed", "void"}:
        return result
    return "backtest-only" if explicit == "graded" else "pending"


def _normalize_generic(frame: pd.DataFrame, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        status = _status_from_row(row)
        tier = _first(row, ["confidence_tier", "confidence_label", "confidence", "recommendation_tier"], "")
        label = _first(row, ["label", "recommendation", "research_recommendation"], "")
        if not label:
            label = "Backtest-only" if status == "backtest-only" else "Paper bet"
        rows.append(
            {
                "bet_date": _first(row, ["bet_date", "snapshot_date", "game_date", "date"], ""),
                "sport": _first(row, ["sport"], "basketball" if "single_recommendations" in source_name else ""),
                "league": _first(row, ["league"], "NBA" if "single_recommendations" in source_name else ""),
                "player": _first(row, ["player", "player_name"], ""),
                "prop_type": _first(row, ["prop_type"], "game_winner" if "single_recommendations" in source_name else ""),
                "market": _first(row, ["market", "market_ticker"], ""),
                "line": _first(row, ["line"], ""),
                "side": _first(row, ["side", "graded_side", "research_side"], ""),
                "odds": _first(row, ["odds", "price", "graded_price", "research_price", "price_cents"], ""),
                "model_probability": _first(
                    row,
                    ["model_probability", "model_prob", "graded_model_probability", "research_model_probability"],
                    "",
                ),
                "market_probability": _first(
                    row,
                    [
                        "market_probability",
                        "market_prob",
                        "graded_market_implied_probability",
                        "research_market_implied_probability",
                        "market_implied_probability",
                    ],
                    "",
                ),
                "edge": _first(row, ["edge", "graded_edge", "final_edge"], ""),
                "expected_value": _first(row, ["expected_value", "expected_ev", "estimated_ev", "final_edge"], ""),
                "result": _first(row, ["result", "result_status"], status),
                "profit_loss": _first(row, ["profit_loss", "profit"], ""),
                "clv": _first(row, ["clv", "clv_cents"], ""),
                "closing_price": _first(row, ["closing_price", "clv_reference_price_cents"], ""),
                "confidence_tier": tier,
                "source": source_name,
                "status": status,
                "label": label,
            }
        )
    return rows


def _normalize_paper_trade_suggestions(frame: pd.DataFrame, source_name: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    usable = frame.copy()
    if "trade" in usable.columns:
        usable = usable[usable["trade"].map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})]
    rows = []
    for _, row in usable.iterrows():
        actual_yes = _boolish(_first(row, ["actual_yes_win"], None))
        side = str(_first(row, ["side"], "")).upper()
        won = actual_yes if side == "YES" else (not actual_yes if actual_yes is not None and side == "NO" else None)
        price = pd.to_numeric(pd.Series([_first(row, ["price_cents"], "")]), errors="coerce").iloc[0]
        if pd.notna(price) and won is not None:
            stake = float(price) / 100.0
            pnl = (1.0 - stake) if won else -stake
        else:
            pnl = ""
        rows.append(
            {
                "bet_date": _first(row, ["game_date"], ""),
                "sport": "basketball",
                "league": "NBA",
                "player": "",
                "prop_type": "game_winner",
                "market": _first(row, ["market_ticker"], ""),
                "line": "",
                "side": side,
                "odds": _first(row, ["price_cents"], ""),
                "model_probability": _first(row, ["model_prob", "model_yes_prob"], ""),
                "market_probability": _first(row, ["market_prob"], ""),
                "edge": _first(row, ["edge"], ""),
                "expected_value": _first(row, ["expected_value"], ""),
                "result": "won" if won else "lost" if won is False else "pending",
                "profit_loss": pnl,
                "clv": "",
                "closing_price": "",
                "confidence_tier": "",
                "source": source_name,
                "status": "won" if won else "lost" if won is False else "pending",
                "label": "Backtest-only",
            }
        )
    return rows


def find_bet_rows(reports_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources_used: list[str] = []
    for filename in INPUT_CANDIDATES:
        path = reports_dir / filename
        frame = _read_csv(path)
        if frame.empty:
            continue
        source_name = filename
        if filename == "paper_trade_suggestions.csv":
            normalized = _normalize_paper_trade_suggestions(frame, source_name)
        else:
            normalized = _normalize_generic(frame, source_name)
        if normalized:
            rows.extend(normalized)
            sources_used.append(filename)
            # Prefer the first real ledger/report so the tab does not double-count
            # the same strategy from multiple downstream exports.
            break
    return rows, sources_used


def _normalize_parlays(reports_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in PARLAY_CANDIDATES:
        frame = _read_csv(reports_dir / filename)
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "source": filename,
                    "legs": _first(row, ["legs"], ""),
                    "projected_hit_probability": _first(row, ["projected_hit_probability", "combined_model_probability"], ""),
                    "break_even_probability": _first(row, ["break_even_probability"], ""),
                    "estimated_payout": _first(row, ["estimated_payout", "offered_payout"], ""),
                    "expected_value": _first(row, ["expected_value", "estimated_ev"], ""),
                    "result": _first(row, ["result", "result_status"], "pending"),
                    "profit_loss": _first(row, ["profit_loss", "profit"], ""),
                    "correlation_risk": _first(row, ["correlation_risk"], ""),
                    "biggest_failure_reason": _first(row, ["biggest_failure_reason", "biggest_risk"], ""),
                    "research_only": _first(row, ["research_only"], True),
                    "approved": _first(row, ["approved"], False),
                }
            )
        if rows:
            break
    return rows


def _group_profit(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if key not in frame.columns:
        return []
    frame["profit_loss_numeric"] = pd.to_numeric(frame.get("profit_loss"), errors="coerce").fillna(0.0)
    grouped = frame.groupby(frame[key].fillna("").astype(str), dropna=False)
    out = []
    for value, group in grouped:
        out.append(
            {
                key: value or "(blank)",
                "bets": int(len(group)),
                "profit_loss": round(float(group["profit_loss_numeric"].sum()), 4),
            }
        )
    return sorted(out, key=lambda row: (-row["bets"], str(row[key])))


def build_report(reports_dir: Path = REPORTS_DIR, now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bets, sources_used = find_bet_rows(reports_dir)
    parlays = _normalize_parlays(reports_dir)
    gates = _read_json(reports_dir / "player_prop_data_quality_gates.json")
    gate_status = str(gates.get("status") or "unknown")

    frame = pd.DataFrame(bets, columns=BET_COLUMNS)
    if frame.empty:
        summary = {
            "total_paper_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "total_profit_loss": 0.0,
            "roi": None,
            "average_clv": None,
        }
    else:
        status = frame["status"].fillna("").astype(str).str.lower()
        wins = int((status == "won").sum())
        losses = int((status == "lost").sum())
        pushes = int(status.isin(["pushed", "push"]).sum())
        graded = wins + losses
        pnl = pd.to_numeric(frame["profit_loss"], errors="coerce").fillna(0.0)
        odds = pd.to_numeric(frame["odds"], errors="coerce")
        stake = (odds / 100.0).where(odds > 1.0, odds).fillna(1.0).abs()
        clv = pd.to_numeric(frame["clv"], errors="coerce")
        summary = {
            "total_paper_bets": int(len(frame)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / graded, 4) if graded else None,
            "total_profit_loss": round(float(pnl.sum()), 4),
            "roi": round(float(pnl.sum() / stake.sum()), 4) if float(stake.sum()) else None,
            "average_clv": round(float(clv.dropna().mean()), 4) if clv.notna().any() else None,
        }

    status = "available" if bets else "blocked"
    blocked_reason = None
    if not bets:
        blocked_reason = (
            "No model bets yet. Betting page is waiting for data quality gates and model outputs."
            if gate_status != "modeling_experiment_ready"
            else "No paper betting ledger or model-output file was found."
        )

    return {
        "report": "paper_betting_report",
        "generated_at_utc": now.isoformat(),
        "status": status,
        "blocked_reason": blocked_reason,
        "data_gate_status": gate_status,
        "sources_used": sources_used,
        "summary": summary,
        "bets": bets,
        "parlays": parlays,
        "profit_by_sport": _group_profit(bets, "sport"),
        "profit_by_league": _group_profit(bets, "league"),
        "profit_by_prop_type": _group_profit(bets, "prop_type"),
        "profit_by_source": _group_profit(bets, "source"),
        "profit_by_confidence_tier": _group_profit(bets, "confidence_tier"),
        "research_only": True,
        "approved": False,
        "approved_bets_enabled": False,
        "approved_parlays_enabled": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Paper Betting Report",
        "",
        f"Generated: {report.get('generated_at_utc')}",
        "",
        "_Research-only. Approved bets and approved parlays remain blocked._",
        "",
        f"- Status: {report.get('status')}",
        f"- Data gate status: {report.get('data_gate_status')}",
    ]
    if report.get("blocked_reason"):
        lines.append(f"- Blocked reason: {report['blocked_reason']}")
    lines += [
        f"- Total paper bets: {summary.get('total_paper_bets', 0)}",
        f"- Wins/losses/pushes: {summary.get('wins', 0)}/{summary.get('losses', 0)}/{summary.get('pushes', 0)}",
        f"- Total profit/loss: {summary.get('total_profit_loss', 0.0)}",
        f"- ROI: {summary.get('roi')}",
        f"- Average CLV: {summary.get('average_clv')}",
        "",
        "## Recent Bets",
        "",
        "| date | sport | league | market | player | prop | side | odds | edge | result | P/L | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in (report.get("bets") or [])[:100]:
        lines.append(
            f"| {row.get('bet_date', '')} | {row.get('sport', '')} | {row.get('league', '')} | "
            f"{row.get('market', '')} | {row.get('player', '')} | {row.get('prop_type', '')} | "
            f"{row.get('side', '')} | {row.get('odds', '')} | {row.get('edge', '')} | "
            f"{row.get('result', '')} | {row.get('profit_loss', '')} | {row.get('status', '')} |"
        )
    if not report.get("bets"):
        lines.append("| | | | No model bets yet - blocked by data quality gates. | | | | | | | | blocked |")
    lines += ["", "_No approved recommendations are produced by this report._", ""]
    return "\n".join(lines)


def write_report(report: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "paper_betting_report.json"
    csv_path = reports_dir / "paper_betting_report.csv"
    md_path = reports_dir / "paper_betting_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BET_COLUMNS)
        writer.writeheader()
        for row in report.get("bets", []):
            writer.writerow({column: row.get(column, "") for column in BET_COLUMNS})
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def main() -> int:
    report = build_report()
    outputs = write_report(report)
    print(f"Paper betting report: {report['status']} ({report['summary']['total_paper_bets']} bet row(s))")
    if report.get("blocked_reason"):
        print(report["blocked_reason"])
    for name, path in outputs.items():
        print(f"Wrote {name}: {path.relative_to(PROJECT_ROOT)}")
    print("Research-only: approved bets/parlays remain blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
