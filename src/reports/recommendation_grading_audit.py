"""Audit recommendation grading outputs for matching, payout, CLV, and leakage risks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MATCH_KEYS = ["game_id", "market_ticker"]
LEAKAGE_TOKENS = [
    "actual",
    "settlement",
    "result",
    "score",
    "points",
    "profit",
    "payout",
    "bankroll",
    "clv",
    "closing",
    "clv_reference",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


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


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _key_counts(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or not set(MATCH_KEYS).issubset(frame.columns):
        return pd.Series(dtype=int)
    return frame.groupby(MATCH_KEYS, dropna=False).size()


def _first_existing(frame: pd.DataFrame, columns: list[str], default: Any = np.nan) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _price_decimal(price_cents: pd.Series) -> pd.Series:
    return pd.to_numeric(price_cents, errors="coerce") / 100.0


def add_profit_math_checks(rows: pd.DataFrame) -> pd.DataFrame:
    """Add expected unit profit and payout correctness checks."""

    output = rows.copy()
    side = output.get("graded_side", pd.Series("", index=output.index)).astype(str).str.upper()
    actual_yes = _bool_series(output.get("actual_yes_win", pd.Series("", index=output.index)))
    result_status = output.get("result_status", pd.Series("", index=output.index)).astype(str)
    price = _price_decimal(_first_existing(output, ["graded_price", "price", "price_cents"]))
    won = ((side == "YES") & actual_yes) | ((side == "NO") & ~actual_yes)
    expected_profit = np.where(result_status.eq("graded") & price.notna(), np.where(won, 1.0 - price, -price), np.nan)
    actual_profit = pd.to_numeric(_first_existing(output, ["profit_loss"]), errors="coerce")
    output["audit_expected_profit_loss"] = expected_profit
    output["audit_profit_diff"] = actual_profit - output["audit_expected_profit_loss"]
    output["audit_profit_math_ok"] = np.where(
        result_status.eq("graded") & actual_profit.notna() & pd.notna(output["audit_expected_profit_loss"]),
        output["audit_profit_diff"].abs() <= 1e-9,
        np.nan,
    )
    output["audit_price_interpretation"] = np.where(price.between(0, 1, inclusive="neither"), "cents_0_100", "invalid")
    output["audit_losing_loss_ok"] = np.where(
        result_status.eq("graded") & ~won & actual_profit.notna(),
        (actual_profit + price).abs() <= 1e-9,
        np.nan,
    )
    output["audit_winning_profit_ok"] = np.where(
        result_status.eq("graded") & won & actual_profit.notna(),
        (actual_profit - (1.0 - price)).abs() <= 1e-9,
        np.nan,
    )
    output["audit_impossible_profit_loss"] = actual_profit.notna() & ((actual_profit < -1.0) | (actual_profit > 1.0))
    return output


def add_clv_math_checks(rows: pd.DataFrame) -> pd.DataFrame:
    """Add expected CLV and side-direction checks."""

    output = rows.copy()
    side = output.get("graded_side", pd.Series("", index=output.index)).astype(str).str.upper()
    entry = pd.to_numeric(_first_existing(output, ["graded_price", "price", "price_cents"]), errors="coerce")
    yes_close = pd.to_numeric(_first_existing(output, ["clv_reference_price_cents", "closing_price_cents"]), errors="coerce")
    no_close = pd.to_numeric(_first_existing(output, ["clv_reference_no_price_cents", "closing_no_price_cents"]), errors="coerce")
    fallback_clv = pd.to_numeric(_first_existing(output, ["clv_cents"]), errors="coerce")
    expected_from_close = pd.Series(np.where(side.eq("NO"), no_close - entry, yes_close - entry), index=output.index)
    expected = expected_from_close.where(expected_from_close.notna(), fallback_clv)
    actual = pd.to_numeric(_first_existing(output, ["clv"]), errors="coerce")
    output["audit_expected_clv"] = expected
    output["audit_clv_diff"] = actual - expected
    output["audit_clv_math_ok"] = np.where(
        actual.notna() & expected.notna(),
        output["audit_clv_diff"].abs() <= 1e-9,
        np.nan,
    )
    output["audit_clv_source"] = np.where(expected_from_close.notna(), "closing_price", np.where(fallback_clv.notna(), "clv_cents", "missing"))
    output["audit_clv_direction_issue"] = False
    output.loc[side.eq("YES") & yes_close.notna() & actual.notna(), "audit_clv_direction_issue"] = (
        np.sign(actual[side.eq("YES") & yes_close.notna() & actual.notna()])
        != np.sign((yes_close - entry)[side.eq("YES") & yes_close.notna() & actual.notna()])
    )
    output.loc[side.eq("NO") & no_close.notna() & actual.notna(), "audit_clv_direction_issue"] = (
        np.sign(actual[side.eq("NO") & no_close.notna() & actual.notna()])
        != np.sign((no_close - entry)[side.eq("NO") & no_close.notna() & actual.notna()])
    )
    output["audit_missing_clv_for_side"] = output["audit_clv_source"].eq("missing")
    return output


def _bucketize(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["audit_price_bucket"] = pd.cut(
        pd.to_numeric(_first_existing(output, ["graded_price", "price"]), errors="coerce"),
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )
    output["audit_edge_bucket"] = pd.cut(
        pd.to_numeric(_first_existing(output, ["graded_edge", "edge", "final_edge"]), errors="coerce"),
        bins=[-1, 0, 0.03, 0.05, 0.10, 1],
        labels=["<=0", "0-3", "3-5", "5-10", "10+"],
    )
    output["audit_clv_bucket"] = pd.cut(
        pd.to_numeric(_first_existing(output, ["clv"]), errors="coerce"),
        bins=[-100, -10, -3, 0, 3, 10, 100],
        labels=["<-10", "-10--3", "-3-0", "0-3", "3-10", "10+"],
    )
    return output


def _group_perf(rows: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    if rows.empty or by not in rows.columns:
        return []
    output = []
    for key, group in rows.groupby(by, dropna=False, observed=False):
        graded = group[group.get("result_status", pd.Series("", index=group.index)).astype(str).eq("graded")]
        output.append(
            {
                "bucket_type": by,
                "bucket": "" if pd.isna(key) else str(key),
                "rows": int(len(group)),
                "graded_rows": int(len(graded)),
                "win_rate": float(pd.to_numeric(_first_existing(graded, ["won"]), errors="coerce").mean()) if not graded.empty else None,
                "profit_loss": float(pd.to_numeric(_first_existing(graded, ["profit_loss"]), errors="coerce").sum()) if not graded.empty else 0.0,
                "average_clv": float(pd.to_numeric(_first_existing(graded, ["clv"]), errors="coerce").mean()) if not graded.empty else None,
            }
        )
    return output


def _snapshot_audit(snapshot_dir: str | Path) -> dict[str, Any]:
    root = Path(snapshot_dir)
    if not root.exists():
        return {"snapshot_dir_exists": False, "snapshots": 0, "timestamped_snapshots": 0, "manifest_count": 0}
    folders = [path for path in root.iterdir() if path.is_dir()]
    timestamped = [path for path in folders if "_" in path.name and path.name[-1:].upper() == "Z"]
    manifests = [path / "snapshot_manifest.json" for path in folders if (path / "snapshot_manifest.json").exists()]
    overwritten_risk = len({path.name for path in folders}) != len(folders)
    return {
        "snapshot_dir_exists": True,
        "snapshots": int(len(folders)),
        "timestamped_snapshots": int(len(timestamped)),
        "manifest_count": int(len(manifests)),
        "snapshot_overwrite_risk_detected": bool(overwritten_risk),
        "snapshot_ids": [path.name for path in folders],
    }


def _leakage_columns(recommendations: pd.DataFrame) -> list[str]:
    flagged = []
    for column in recommendations.columns:
        lower = str(column).lower()
        if any(token in lower for token in LEAKAGE_TOKENS):
            flagged.append(str(column))
    return flagged


def _match_quality(recommendations: pd.DataFrame, results: pd.DataFrame, graded: pd.DataFrame) -> dict[str, Any]:
    rec_counts = _key_counts(recommendations)
    result_counts = _key_counts(results)
    graded_counts = graded["result_status"].value_counts(dropna=False).to_dict() if "result_status" in graded.columns else {}
    duplicate_match_keys = sorted(
        set(tuple(key) for key, value in rec_counts.items() if value > 1)
        | set(tuple(key) for key, value in result_counts.items() if value > 1)
    )
    rec_multiple_results = 0
    result_multiple_recs = 0
    if not rec_counts.empty and not result_counts.empty:
        common = set(rec_counts.index).intersection(set(result_counts.index))
        rec_multiple_results = sum(1 for key in common if result_counts.loc[key] > 1)
        result_multiple_recs = sum(1 for key in common if rec_counts.loc[key] > 1)
    return {
        "recommendations": int(len(recommendations)),
        "graded_rows_file_rows": int(len(graded)),
        "graded": int(graded_counts.get("graded", 0)),
        "pending": int(graded_counts.get("pending", 0)),
        "unmatched": int(graded_counts.get("unmatched", 0)),
        "duplicate_recommendation_rows": int(recommendations.duplicated().sum()) if not recommendations.empty else 0,
        "duplicate_result_rows": int(results.duplicated().sum()) if not results.empty else 0,
        "duplicate_recommendation_match_keys": int((rec_counts > 1).sum()) if not rec_counts.empty else 0,
        "duplicate_result_match_keys": int((result_counts > 1).sum()) if not result_counts.empty else 0,
        "duplicate_match_keys": [list(key) for key in duplicate_match_keys[:50]],
        "recommendations_matching_multiple_results": int(rec_multiple_results),
        "multiple_recommendations_matching_one_result": int(result_multiple_recs),
    }


def _parse_leg_segments(legs: Any) -> list[tuple[str, str]]:
    """Split a parlay ``legs`` string into (market, side) pairs."""

    segments: list[tuple[str, str]] = []
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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def audit_parlays(parlays: pd.DataFrame, graded_singles: pd.DataFrame) -> pd.DataFrame:
    """Re-derive parlay leg outcomes from graded singles and flag inconsistencies.

    Verifies that each parlay leg matches a graded single row, that stored
    ``legs_won``/``legs_lost`` agree with the recomputed leg outcomes, and that
    ``parlay_won`` is true only when every leg won.
    """

    if parlays.empty:
        return pd.DataFrame()
    singles = graded_singles.copy()
    market_col = singles.get("market", pd.Series("", index=singles.index)).astype(str)
    side_col = singles.get("graded_side", pd.Series("", index=singles.index)).astype(str).str.upper()
    rows: list[dict[str, Any]] = []
    for _, parlay in parlays.iterrows():
        segments = _parse_leg_segments(parlay.get("legs"))
        outcomes = []
        for market, side in segments:
            matches = singles[market_col.eq(market) & side_col.eq(side)]
            if matches.empty:
                outcomes.append("unmatched")
                continue
            match = matches.iloc[0]
            status = str(match.get("result_status") or "")
            if status != "graded":
                outcomes.append(status or "pending")
            else:
                outcomes.append("won" if bool(match.get("won")) else "lost")
        legs_won = sum(value == "won" for value in outcomes)
        legs_lost = sum(value == "lost" for value in outcomes)
        all_graded = bool(segments) and all(value in {"won", "lost"} for value in outcomes)
        recomputed_won = all_graded and legs_lost == 0 and legs_won == len(segments)
        stored_won_raw = parlay.get("parlay_won")
        stored_won = None if pd.isna(stored_won_raw) else bool(stored_won_raw)
        legs_match = _int_or_none(parlay.get("legs_won")) == legs_won and _int_or_none(parlay.get("legs_lost")) == legs_lost
        rows.append(
            {
                "parlay_tier": parlay.get("parlay_tier"),
                "number_of_legs": parlay.get("number_of_legs"),
                "correlation_risk": parlay.get("correlation_risk"),
                "legs": parlay.get("legs"),
                "stored_legs_won": parlay.get("legs_won"),
                "stored_legs_lost": parlay.get("legs_lost"),
                "recomputed_legs_won": int(legs_won),
                "recomputed_legs_lost": int(legs_lost),
                "stored_parlay_won": stored_won,
                "recomputed_parlay_won": bool(recomputed_won) if all_graded else None,
                "combined_result_status": parlay.get("combined_result_status"),
                "audit_legs_match": bool(legs_match),
                "audit_parlay_won_ok": bool(stored_won == recomputed_won) if all_graded else True,
                "audit_all_legs_graded": bool(all_graded),
            }
        )
    return pd.DataFrame(rows)


def _parlay_sanity(parlay_audit: pd.DataFrame) -> dict[str, Any]:
    if parlay_audit.empty:
        return {
            "parlays": 0,
            "leg_count_mismatches": 0,
            "parlay_won_mismatches": 0,
            "fully_graded_parlays": 0,
            "by_parlay_tier": [],
            "by_correlation_risk": [],
            "by_number_of_legs": [],
        }
    won = parlay_audit.assign(
        won=parlay_audit["recomputed_parlay_won"],
        profit_loss=np.nan,
        result_status=np.where(parlay_audit["audit_all_legs_graded"], "graded", "pending"),
    )
    return {
        "parlays": int(len(parlay_audit)),
        "leg_count_mismatches": int((~parlay_audit["audit_legs_match"]).sum()),
        "parlay_won_mismatches": int((~parlay_audit["audit_parlay_won_ok"]).sum()),
        "fully_graded_parlays": int(parlay_audit["audit_all_legs_graded"].sum()),
        "by_parlay_tier": _group_perf(won, "parlay_tier"),
        "by_correlation_risk": _group_perf(won, "correlation_risk"),
        "by_number_of_legs": _group_perf(won, "number_of_legs"),
    }


def _collect_warnings(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn uncertain or suspicious audit findings into explicit warnings."""

    match = summary["match_quality"]
    payout = summary["payout_profit_math"]
    clv = summary["clv_math"]
    leakage = summary["leakage_risk"]
    parlay = summary["parlay_sanity"]
    warnings: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str) -> None:
        warnings.append({"severity": severity, "code": code, "message": message})

    if match["unmatched"]:
        add("minor", "unmatched_rows", f"{match['unmatched']} graded rows are unmatched and excluded from profit/CLV.")
    for key in [
        "duplicate_recommendation_match_keys",
        "duplicate_result_match_keys",
        "recommendations_matching_multiple_results",
        "multiple_recommendations_matching_one_result",
    ]:
        if match[key]:
            add("major", key, f"{match[key]} {key.replace('_', ' ')}; results may be double counted.")

    if payout["profit_math_failures"] or payout["impossible_profit_loss_rows"]:
        add(
            "major",
            "profit_math",
            f"{payout['profit_math_failures']} profit-math failures and "
            f"{payout['impossible_profit_loss_rows']} impossible profit/loss rows.",
        )
    backtest_profit = payout.get("backtest_stake_weighted_profit")
    total_profit = payout.get("total_profit")
    if (
        backtest_profit is not None
        and total_profit is not None
        and np.isfinite(backtest_profit)
        and np.sign(total_profit) != np.sign(backtest_profit)
        and np.sign(backtest_profit) != 0
    ):
        add(
            "major",
            "profit_basis_mismatch",
            f"Equal-weight per-share profit ({total_profit:+.2f}) disagrees in sign with backtest "
            f"stake-weighted profit ({backtest_profit:+.2f}); the headline number is normalized per-contract, "
            "not bankroll P/L.",
        )

    dropped_clv = clv.get("graded_rows_without_clv_value", clv["missing_clv_rows"])
    no_dropped = clv.get("no_side_rows_without_clv_value", 0)
    if dropped_clv:
        add(
            "major",
            "missing_clv",
            f"{dropped_clv} graded rows ({no_dropped} of them NO-side) have no CLV value in the column the "
            "performance summary averages, so reported average CLV is effectively YES-only. Reconstruct NO CLV "
            "from the NO closing price or use the backtest clv_cents column.",
        )
    if clv["clv_direction_issues"]:
        add("major", "clv_direction", f"{clv['clv_direction_issues']} rows have a CLV sign inconsistent with side/price movement.")
    overall_clv = clv.get("average_clv_overall")
    if total_profit is not None and total_profit > 0 and overall_clv is not None and overall_clv <= 0:
        add(
            "major",
            "profit_without_clv",
            f"Positive realized profit ({total_profit:+.2f}) with non-positive average CLV ({overall_clv:+.3f}); "
            "likely small-sample/cheap-contract variance, not proven edge. Do not treat as proof.",
        )

    if leakage["flagged_recommendation_input_columns"]:
        add(
            "major",
            "leakage_columns",
            "Recommendation inputs contain result-like columns: "
            + ", ".join(leakage["flagged_recommendation_input_columns"]),
        )
    if leakage["snapshot_audit"].get("snapshot_overwrite_risk_detected"):
        add("major", "snapshot_overwrite", "Snapshot folder names are not unique; snapshots may have been overwritten.")

    if parlay["leg_count_mismatches"] or parlay["parlay_won_mismatches"]:
        add(
            "major",
            "parlay_grading",
            f"{parlay['leg_count_mismatches']} parlay leg-count mismatches and "
            f"{parlay['parlay_won_mismatches']} parlay-won mismatches versus recomputed leg outcomes.",
        )

    return warnings


def build_recommendation_grading_audit(
    recommendations_path: str | Path,
    results_path: str | Path,
    graded_single_path: str | Path,
    snapshot_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit recommendation grading artifacts and write audit reports."""

    recommendations = _read_csv(recommendations_path)
    results = _read_csv(results_path)
    graded = _read_csv(graded_single_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = _bucketize(add_clv_math_checks(add_profit_math_checks(graded)))
    rows.to_csv(output / "recommendation_grading_audit_rows.csv", index=False)

    profit_bucket_rows = []
    for column in ["recommendation_tier", "graded_side", "audit_edge_bucket", "audit_price_bucket", "confidence_label"]:
        profit_bucket_rows.extend(_group_perf(rows, column))
    profit_by_bucket = pd.DataFrame(profit_bucket_rows)
    profit_by_bucket.to_csv(output / "recommendation_profit_by_bucket.csv", index=False)

    clv_bucket_rows = []
    for column in ["audit_clv_bucket", "graded_side", "recommendation_tier"]:
        clv_bucket_rows.extend(_group_perf(rows, column))
    clv_by_bucket = pd.DataFrame(clv_bucket_rows)
    clv_by_bucket.to_csv(output / "recommendation_clv_by_bucket.csv", index=False)

    graded_rows = rows[rows.get("result_status", pd.Series("", index=rows.index)).astype(str).eq("graded")]
    total_profit = float(pd.to_numeric(graded_rows.get("profit_loss", np.nan), errors="coerce").sum()) if not graded_rows.empty else 0.0
    row_sum = float(pd.to_numeric(rows.get("profit_loss", np.nan), errors="coerce").sum()) if not rows.empty else 0.0
    backtest_profit = float(pd.to_numeric(_first_existing(graded_rows, ["profit"]), errors="coerce").sum()) if not graded_rows.empty else 0.0
    overall_clv = pd.to_numeric(_first_existing(graded_rows, ["clv"]), errors="coerce") if not graded_rows.empty else pd.Series(dtype=float)
    average_clv_overall = float(overall_clv.mean()) if not overall_clv.empty and overall_clv.notna().any() else None
    graded_side_series = _first_existing(graded_rows, ["graded_side", "side"], "").astype(str).str.upper()
    rows_without_clv = int(overall_clv.isna().sum()) if not overall_clv.empty else 0
    no_rows_without_clv = int(overall_clv[graded_side_series.eq("NO")].isna().sum()) if not overall_clv.empty else 0
    paper = rows[rows.get("recommendation_tier", pd.Series("", index=rows.index)).astype(str).eq("paper_trade_candidate")]
    lean = rows[rows.get("recommendation_tier", pd.Series("", index=rows.index)).astype(str).eq("research_lean")]
    paper_profit = float(pd.to_numeric(paper.get("profit_loss", np.nan), errors="coerce").sum()) if not paper.empty else 0.0
    lean_profit = float(pd.to_numeric(lean.get("profit_loss", np.nan), errors="coerce").sum()) if not lean.empty else 0.0
    summary = {
        "match_quality": _match_quality(recommendations, results, rows),
        "payout_profit_math": {
            "profit_math_failures": int(rows["audit_profit_math_ok"].eq(False).sum()) if "audit_profit_math_ok" in rows.columns else 0,
            "yes_profit_math_failures": int(rows[rows.get("graded_side", "").astype(str).eq("YES")]["audit_profit_math_ok"].eq(False).sum()) if "graded_side" in rows.columns else 0,
            "no_profit_math_failures": int(rows[rows.get("graded_side", "").astype(str).eq("NO")]["audit_profit_math_ok"].eq(False).sum()) if "graded_side" in rows.columns else 0,
            "invalid_price_rows": int(rows["audit_price_interpretation"].eq("invalid").sum()) if "audit_price_interpretation" in rows.columns else 0,
            "losing_loss_failures": int(rows["audit_losing_loss_ok"].eq(False).sum()) if "audit_losing_loss_ok" in rows.columns else 0,
            "winning_profit_failures": int(rows["audit_winning_profit_ok"].eq(False).sum()) if "audit_winning_profit_ok" in rows.columns else 0,
            "impossible_profit_loss_rows": int(rows["audit_impossible_profit_loss"].sum()) if "audit_impossible_profit_loss" in rows.columns else 0,
            "total_profit": total_profit,
            "row_level_profit_sum": row_sum,
            "total_profit_equals_row_sum": abs(total_profit - row_sum) <= 1e-9,
            "backtest_stake_weighted_profit": backtest_profit,
            "profit_basis_note": "row profit_loss is per-share normalized to [0,1] cost; not bankroll P/L",
        },
        "clv_math": {
            "clv_math_failures": int(rows["audit_clv_math_ok"].eq(False).sum()) if "audit_clv_math_ok" in rows.columns else 0,
            "missing_clv_rows": int(rows["audit_missing_clv_for_side"].sum()) if "audit_missing_clv_for_side" in rows.columns else 0,
            "graded_rows_without_clv_value": rows_without_clv,
            "no_side_rows_without_clv_value": no_rows_without_clv,
            "clv_direction_issues": int(rows["audit_clv_direction_issue"].sum()) if "audit_clv_direction_issue" in rows.columns else 0,
            "average_clv_overall": average_clv_overall,
            "average_clv_by_side": _group_perf(rows, "graded_side"),
            "profit_by_clv_bucket": _group_perf(rows, "audit_clv_bucket"),
        },
        "leakage_risk": {
            "recommendation_input_columns": list(recommendations.columns),
            "flagged_recommendation_input_columns": _leakage_columns(recommendations),
            "snapshot_audit": _snapshot_audit(snapshot_dir),
            "grading_after_snapshot_check": "snapshot_ids_present_in_graded_rows"
            if "snapshot_id" in rows.columns and rows["snapshot_id"].notna().any()
            else "missing_snapshot_ids_in_graded_rows",
        },
        "tier_sanity": {
            "by_recommendation_tier": _group_perf(rows, "recommendation_tier"),
            "research_lean_profit": lean_profit,
            "paper_trade_candidate_profit": paper_profit,
            "paper_candidates_outperform_research_leans_by_profit": paper_profit > lean_profit,
            "by_side": _group_perf(rows, "graded_side"),
            "by_edge_bucket": _group_perf(rows, "audit_edge_bucket"),
            "by_confidence_label": _group_perf(rows, "confidence_label"),
        },
        "trusted": False,
    }
    # Parlay grading sanity: re-derive leg outcomes from graded singles.
    parlay_frames = [
        _read_csv(output / "graded_research_parlays.csv"),
        _read_csv(output / "graded_paper_parlays.csv"),
    ]
    parlays = (
        pd.concat([frame for frame in parlay_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in parlay_frames)
        else pd.DataFrame()
    )
    parlay_audit = audit_parlays(parlays, rows)
    summary["parlay_sanity"] = _parlay_sanity(parlay_audit)

    summary["trusted"] = (
        summary["match_quality"]["duplicate_recommendation_match_keys"] == 0
        and summary["match_quality"]["duplicate_result_match_keys"] == 0
        and summary["payout_profit_math"]["profit_math_failures"] == 0
        and summary["payout_profit_math"]["impossible_profit_loss_rows"] == 0
        and summary["clv_math"]["clv_math_failures"] == 0
        and summary["clv_math"]["missing_clv_rows"] == 0
        and summary["parlay_sanity"]["leg_count_mismatches"] == 0
        and summary["parlay_sanity"]["parlay_won_mismatches"] == 0
        and not summary["leakage_risk"]["flagged_recommendation_input_columns"]
    )
    summary["warnings"] = _collect_warnings(summary)
    summary["major_warnings"] = [warning for warning in summary["warnings"] if warning["severity"] == "major"]
    summary["verdict"] = "grading_math_consistent" if summary["trusted"] else "grading_needs_review"
    summary["research_only"] = True
    summary["approved"] = False

    parlay_audit_columns = [
        "parlay_tier",
        "number_of_legs",
        "correlation_risk",
        "legs",
        "stored_legs_won",
        "stored_legs_lost",
        "recomputed_legs_won",
        "recomputed_legs_lost",
        "stored_parlay_won",
        "recomputed_parlay_won",
        "combined_result_status",
        "audit_legs_match",
        "audit_parlay_won_ok",
        "audit_all_legs_graded",
    ]
    (parlay_audit if not parlay_audit.empty else pd.DataFrame(columns=parlay_audit_columns)).to_csv(
        output / "recommendation_parlay_audit.csv", index=False
    )

    summary = _json_safe(summary)
    (output / "recommendation_grading_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
