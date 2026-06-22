"""Snapshot and grade research recommendation outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SINGLE_TIERS = {"research_lean", "paper_trade_candidate", "approved_bet"}
PARLAY_TIERS = {"research_parlay", "paper_parlay", "approved_parlay"}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _to_number(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


def _snapshot_id(run_date: str | None = None) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    snapshot_date = run_date or now.date().isoformat()
    safe_date = str(snapshot_date).replace(":", "").replace("/", "-").replace("\\", "-")
    return snapshot_date, f"{safe_date}_{now.strftime('%Y%m%dT%H%M%S%fZ')}"


def _with_snapshot_columns(frame: pd.DataFrame, snapshot_date: str, snapshot_id: str) -> pd.DataFrame:
    output = frame.copy()
    output["snapshot_date"] = snapshot_date
    output["snapshot_id"] = snapshot_id
    return output


def save_recommendation_snapshot(
    fair_price_signals_path: str | Path,
    research_parlay_path: str | Path,
    paper_parlay_path: str | Path,
    output_dir: str | Path,
    run_date: str | None = None,
) -> dict[str, Any]:
    """Save immutable timestamped copies of recommendation CSV files."""

    snapshot_date, snapshot_id = _snapshot_id(run_date)
    output_root = Path(output_dir)
    snapshot_dir = output_root / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        "fair_price_signals": Path(fair_price_signals_path),
        "research_parlay_candidates": Path(research_parlay_path),
        "paper_parlay_candidates": Path(paper_parlay_path),
    }
    created: dict[str, str] = {}
    missing: dict[str, str] = {}
    for name, source in inputs.items():
        if not source.exists():
            missing[name] = str(source)
            continue
        frame = _with_snapshot_columns(_read_csv(source), snapshot_date, snapshot_id)
        destination = snapshot_dir / source.name
        frame.to_csv(destination, index=False)
        created[name] = str(destination)
    manifest = {
        "snapshot_id": snapshot_id,
        "snapshot_date": snapshot_date,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_created": created,
        "missing_inputs": missing,
    }
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _single_recommendation_rows(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame()
    frame = snapshot.copy()
    if "recommendation_tier" in frame.columns:
        frame = frame[frame["recommendation_tier"].fillna("").astype(str).isin(SINGLE_TIERS)].copy()
    frame["graded_side"] = _first_existing(frame, ["research_side", "side", "ungated_side"]).astype(str).str.upper()
    frame["graded_price"] = pd.to_numeric(_first_existing(frame, ["research_price", "price"]), errors="coerce")
    frame["graded_model_probability"] = pd.to_numeric(
        _first_existing(frame, ["research_model_probability", "model_prob"]),
        errors="coerce",
    )
    frame["graded_market_implied_probability"] = pd.to_numeric(
        _first_existing(frame, ["research_market_implied_probability", "market_implied_probability"]),
        errors="coerce",
    )
    frame["graded_edge"] = pd.to_numeric(_first_existing(frame, ["edge", "final_edge"]), errors="coerce")
    return frame


def _first_existing(frame: pd.DataFrame, columns: list[str], default: Any = "") -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _pending_or_unmatched(row: pd.Series) -> str:
    date_value = row.get("game_date") or row.get("date")
    game_date = pd.to_datetime(pd.Series([date_value]), errors="coerce").iloc[0]
    if pd.notna(game_date) and game_date.date() >= datetime.now().date():
        return "pending"
    return "unmatched"


def grade_single_recommendations(
    snapshot_fair_price_csv: str | Path,
    settled_results_file: str | Path,
) -> pd.DataFrame:
    """Grade snapshot single-game recommendations against settled result rows."""

    snapshot = _single_recommendation_rows(_read_csv(snapshot_fair_price_csv))
    results = _read_csv(settled_results_file)
    if snapshot.empty:
        return snapshot
    if results.empty:
        output = snapshot.copy()
        output["won"] = np.nan
        output["profit_loss"] = np.nan
        output["clv"] = np.nan
        output["result_status"] = output.apply(_pending_or_unmatched, axis=1)
        return output

    result_cols = [
        column
        for column in [
            "game_id",
            "market_ticker",
            "actual_yes_win",
            "profit",
            "clv_cents",
            "clv_reference_price_cents",
            "clv_reference_no_price_cents",
            "price_cents",
        ]
        if column in results.columns
    ]
    result_keyed = results[result_cols].drop_duplicates(["game_id", "market_ticker"], keep="last")
    output = snapshot.merge(result_keyed, on=["game_id", "market_ticker"], how="left", suffixes=("", "_result"))
    has_result = output["actual_yes_win"].notna() if "actual_yes_win" in output.columns else pd.Series(False, index=output.index)
    actual_yes = output.get("actual_yes_win", pd.Series(False, index=output.index)).astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    side = output["graded_side"].astype(str).str.upper()
    won = ((side == "YES") & actual_yes) | ((side == "NO") & ~actual_yes)
    output["won"] = np.where(has_result, won, np.nan)
    price = pd.to_numeric(output["graded_price"], errors="coerce") / 100.0
    output["profit_loss"] = np.where(has_result & price.notna(), np.where(won, 1.0 - price, -price), np.nan)
    yes_close = pd.to_numeric(output.get("clv_reference_price_cents", np.nan), errors="coerce")
    no_close = pd.to_numeric(output.get("clv_reference_no_price_cents", np.nan), errors="coerce")
    close = np.where(side == "NO", no_close, yes_close)
    output["clv"] = np.where(has_result & price.notna() & np.isfinite(close), close - output["graded_price"], np.nan)
    output["result_status"] = np.where(has_result, "graded", output.apply(_pending_or_unmatched, axis=1))
    return output


def _parse_parlay_leg_segments(legs: Any) -> list[tuple[str, str]]:
    segments = []
    for segment in str(legs or "").split("|"):
        text = segment.strip()
        if not text:
            continue
        side = "YES" if " YES " in f" {text} " else "NO" if " NO " in f" {text} " else ""
        market = text
        for token in [" YES ", " NO "]:
            if token in f" {text} ":
                market = text.split(token.strip(), 1)[0].strip()
                break
        segments.append((market, side))
    return segments


def grade_parlay_recommendations(
    snapshot_parlay_csv: str | Path,
    graded_single_recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Grade parlay rows by checking whether every leg won."""

    parlays = _read_csv(snapshot_parlay_csv)
    if parlays.empty:
        return parlays
    singles = graded_single_recommendations.copy()
    rows: list[dict[str, Any]] = []
    for _, row in parlays.iterrows():
        segments = _parse_parlay_leg_segments(row.get("legs"))
        leg_results = []
        failed_reasons = []
        for market, side in segments:
            candidates = singles[
                singles.get("market", pd.Series("", index=singles.index)).astype(str).eq(market)
                & singles.get("graded_side", pd.Series("", index=singles.index)).astype(str).str.upper().eq(side)
            ]
            if candidates.empty:
                leg_results.append("unmatched")
                failed_reasons.append(f"unmatched:{market}:{side}")
                continue
            match = candidates.iloc[0]
            status = str(match.get("result_status") or "")
            if status != "graded":
                leg_results.append(status or "pending")
                failed_reasons.append(f"{status or 'pending'}:{market}:{side}")
                continue
            if bool(match.get("won")):
                leg_results.append("won")
            else:
                leg_results.append("lost")
                failed_reasons.append(f"lost:{market}:{side}")
        legs_won = sum(result == "won" for result in leg_results)
        legs_lost = sum(result == "lost" for result in leg_results)
        all_matched = bool(segments) and all(result in {"won", "lost"} for result in leg_results)
        parlay_won = all_matched and legs_lost == 0 and legs_won == len(segments)
        if all_matched:
            status = "graded"
        elif any(result == "pending" for result in leg_results):
            status = "pending"
        else:
            status = "unmatched"
        offered = _to_number(row.get("offered_payout"))
        estimated_profit_loss = np.nan
        if status == "graded" and np.isfinite(offered):
            estimated_profit_loss = offered - 1.0 if parlay_won else -1.0
        output_row = row.to_dict()
        output_row.update(
            {
                "legs_won": int(legs_won),
                "legs_lost": int(legs_lost),
                "parlay_won": bool(parlay_won) if status == "graded" else np.nan,
                "combined_result_status": status,
                "estimated_profit_loss": estimated_profit_loss,
                "failed_leg_reason": "; ".join(failed_reasons),
            }
        )
        rows.append(output_row)
    return pd.DataFrame(rows)


def _bucketize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["model_probability_bucket"] = pd.cut(
        pd.to_numeric(output.get("graded_model_probability", np.nan), errors="coerce"),
        bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
        labels=["0-40", "40-50", "50-60", "60-70", "70-100"],
        include_lowest=True,
    )
    output["edge_bucket"] = pd.cut(
        pd.to_numeric(output.get("graded_edge", np.nan), errors="coerce"),
        bins=[-1, 0, 0.03, 0.05, 0.10, 1],
        labels=["<=0", "0-3", "3-5", "5-10", "10+"],
    )
    output["price_bucket"] = pd.cut(
        pd.to_numeric(output.get("graded_price", np.nan), errors="coerce"),
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )
    output["clv_bucket"] = pd.cut(
        pd.to_numeric(output.get("clv", np.nan), errors="coerce"),
        bins=[-100, -10, -3, 0, 3, 10, 100],
        labels=["<-10", "-10--3", "-3-0", "0-3", "3-10", "10+"],
    )
    return output


def _group_summary(frame: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    if frame.empty or by not in frame.columns:
        return []
    grouped = frame.groupby(by, dropna=False, observed=False)
    rows = []
    for key, group in grouped:
        graded = group[group["result_status"].eq("graded")] if "result_status" in group.columns else group.iloc[0:0]
        rows.append(
            {
                str(by): "" if pd.isna(key) else str(key),
                "rows": int(len(group)),
                "graded_rows": int(len(graded)),
                "win_rate": float(pd.to_numeric(graded.get("won", np.nan), errors="coerce").mean()) if not graded.empty else None,
                "profit_loss": float(pd.to_numeric(graded.get("profit_loss", np.nan), errors="coerce").sum()) if not graded.empty else 0.0,
                "average_clv": float(pd.to_numeric(graded.get("clv", np.nan), errors="coerce").mean()) if not graded.empty else None,
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _parlay_group_summary(frame: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    if frame.empty or by not in frame.columns:
        return []
    rows = []
    for key, group in frame.groupby(by, dropna=False, observed=False):
        graded = group[group["combined_result_status"].eq("graded")] if "combined_result_status" in group.columns else group.iloc[0:0]
        rows.append(
            {
                str(by): "" if pd.isna(key) else str(key),
                "rows": int(len(group)),
                "graded_rows": int(len(graded)),
                "win_rate": float(pd.to_numeric(graded.get("parlay_won", np.nan), errors="coerce").mean()) if not graded.empty else None,
                "estimated_profit_loss": float(pd.to_numeric(graded.get("estimated_profit_loss", np.nan), errors="coerce").sum()) if not graded.empty else 0.0,
            }
        )
    return rows


def build_recommendation_performance_summary(
    graded_single_recommendations: pd.DataFrame,
    graded_research_parlays: pd.DataFrame,
    graded_paper_parlays: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write recommendation grading outputs and return a summary dictionary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    singles = _bucketize(graded_single_recommendations) if not graded_single_recommendations.empty else graded_single_recommendations
    research_parlays = graded_research_parlays.copy()
    paper_parlays = graded_paper_parlays.copy()
    parlays = pd.concat([research_parlays, paper_parlays], ignore_index=True)

    singles.to_csv(output / "graded_single_recommendations.csv", index=False)
    research_parlays.to_csv(output / "graded_research_parlays.csv", index=False)
    paper_parlays.to_csv(output / "graded_paper_parlays.csv", index=False)

    failure_parts = []
    for column in ["recommendation_tier", "graded_side", "model_probability_bucket", "edge_bucket", "price_bucket", "clv_bucket", "confidence_label"]:
        rows = _group_summary(singles, column)
        for row in rows:
            row["bucket_type"] = column
            row["bucket"] = row.pop(column)
            failure_parts.append(row)
    failure_buckets = pd.DataFrame(failure_parts)
    if not failure_buckets.empty:
        failure_buckets = failure_buckets.sort_values(["profit_loss", "win_rate"], ascending=[True, True], na_position="last")
    failure_buckets.to_csv(output / "recommendation_failure_buckets.csv", index=False)

    graded_singles = singles[singles["result_status"].eq("graded")] if "result_status" in singles.columns else singles.iloc[0:0]
    summary = {
        "rows": int(len(singles)),
        "graded_rows": int(len(graded_singles)),
        "total_research_leans": int(singles.get("recommendation_tier", pd.Series(dtype=str)).astype(str).eq("research_lean").sum()),
        "total_paper_candidates": int(singles.get("recommendation_tier", pd.Series(dtype=str)).astype(str).eq("paper_trade_candidate").sum()),
        "win_rate": float(pd.to_numeric(graded_singles.get("won", np.nan), errors="coerce").mean()) if not graded_singles.empty else None,
        "profit_loss": float(pd.to_numeric(graded_singles.get("profit_loss", np.nan), errors="coerce").sum()) if not graded_singles.empty else 0.0,
        "average_clv": float(pd.to_numeric(graded_singles.get("clv", np.nan), errors="coerce").mean()) if not graded_singles.empty else None,
        "by_recommendation_tier": _group_summary(singles, "recommendation_tier"),
        "by_side": _group_summary(singles, "graded_side"),
        "by_model_probability_bucket": _group_summary(singles, "model_probability_bucket"),
        "by_edge_bucket": _group_summary(singles, "edge_bucket"),
        "by_price_bucket": _group_summary(singles, "price_bucket"),
        "by_clv_bucket": _group_summary(singles, "clv_bucket"),
        "by_confidence_label": _group_summary(singles, "confidence_label"),
        "parlay_by_tier": _parlay_group_summary(parlays, "parlay_tier"),
        "parlay_by_correlation_risk": _parlay_group_summary(parlays, "correlation_risk"),
        "parlay_by_number_of_legs": _parlay_group_summary(parlays, "number_of_legs"),
        "research_only": True,
        "approved": False,
    }
    summary = _json_safe(summary)
    (output / "recommendation_performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
