"""Research-only data quality gates for NBA player props.

Decides how far along the NBA prop database is on the readiness ladder:

    not_ready -> collection_ready -> settlement_ready -> clv_ready
        -> modeling_experiment_ready

Each rung requires every check of the rungs below it plus its own checks.
These gates decide whether the DATA is ready for modeling *experiments* —
they do not approve betting, do not loosen proof gates, and do not enable
approved bets or parlays. ``modeling_experiment_ready`` is only granted when
the data actually supports it (settled outcomes + CLV pairs at meaningful
sample sizes), never aspirationally.

Outputs:

- ``data/reports/player_prop_data_quality_gates.json``
- ``data/reports/player_prop_data_quality_gates.md``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


GATES_VERSION = "v2"

STATUS_LADDER = (
    "not_ready",
    "collection_ready",
    "settlement_ready",
    "clv_ready",
    "modeling_experiment_ready",
)

# Thresholds. Deliberately conservative; loosening them weakens the gates and
# is NOT allowed as a shortcut to a greener status. v2 tightened the modeling
# rung (settled/closing main-line row minimums, 50 CLV pairs, 3 bookmakers,
# 3 prop types, <1% missing core fields, 90% main-line confidence).
THRESHOLDS = {
    "min_nba_snapshots": 500,
    "min_player_match_rate": 0.95,
    "min_game_match_rate": 0.95,
    "min_main_line_rate": 0.80,
    "max_missing_price_rate": 0.05,
    "max_missing_line_rate": 0.02,
    "max_suspicious_line_rate": 0.02,
    "max_suspicious_price_rate": 0.02,
    "max_duplicate_rate": 0.05,
    "min_bookmakers": 2,
    "min_settled_props": 1,
    "min_settled_for_modeling": 200,
    "min_clv_markets": 1,
    "min_clv_markets_for_modeling": 50,
    "min_closing_market_rate": 0.10,
    "min_closing_market_rate_for_modeling": 0.25,
    # v2 modeling-readiness thresholds.
    "min_settled_main_line_rows_for_modeling": 500,
    "min_closing_main_line_rows_for_modeling": 100,
    "max_missing_core_field_rate": 0.01,
    "min_main_line_rate_for_modeling": 0.90,
    "min_bookmakers_for_modeling": 3,
    "min_prop_types_for_modeling": 3,
}

# Core fields a modeling row cannot do without.
CORE_FIELDS = (
    "snapshot_time", "player_name", "prop_type", "line", "bookmaker", "canonical_game_key",
)

MARKET_KEYS = ["player_name", "prop_type", "bookmaker", "canonical_game_key"]

OUTPUT_FILES = {
    "gates_json": "player_prop_data_quality_gates.json",
    "gates_md": "player_prop_data_quality_gates.md",
}


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _rate(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _check(name: str, value: Any, threshold: Any, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check": name,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
        "detail": detail,
    }


def compute_quality_metrics(
    nba: pd.DataFrame,
    line_quality: pd.DataFrame,
    clv_summary: dict[str, Any],
) -> dict[str, Any]:
    """Raw NBA prop quality metrics from the enriched snapshots + audits."""

    n = int(len(nba))
    if n == 0:
        return {"nba_snapshots": 0}

    player_matched = int(_truthy(nba.get("player_matched", pd.Series(dtype="object"))).sum())
    # Game matching is only verifiable for games that already started: the
    # nba_api games table cannot contain a game that has not been played, so
    # unplayed games would unfairly zero the rate. None = nothing to verify.
    game_matched_flags = _truthy(nba.get("game_matched", pd.Series(dtype="object")))
    starts = pd.to_datetime(
        nba.get("game_start_time", pd.Series(pd.NA, index=nba.index)), errors="coerce", utc=True
    )
    started_mask = starts.notna() & (starts <= pd.Timestamp.now(tz="UTC"))
    n_started = int(started_mask.sum())
    game_match_rate = (
        _rate(int(game_matched_flags[started_mask].sum()), n_started) if n_started else None
    )
    line = pd.to_numeric(nba.get("line"), errors="coerce")
    over = pd.to_numeric(nba.get("over_price"), errors="coerce")
    under = pd.to_numeric(nba.get("under_price"), errors="coerce")
    missing_line = int(line.isna().sum())
    missing_both_prices = int((over.isna() & under.isna()).sum())
    non_half_step = ((line * 2) % 1).fillna(0) != 0
    suspicious_line = int((line.notna() & ((line <= 0) | (line > 150.0) | non_half_step)).sum())
    sus_over = over.notna() & ((over < 1.01) | (over > 100.0))
    sus_under = under.notna() & ((under < 1.01) | (under > 100.0))
    suspicious_price = int((sus_over | sus_under).sum())
    dedup_columns = [
        c
        for c in (
            "snapshot_time", "canonical_game_key", "player_name", "prop_type",
            "line", "over_price", "under_price", "bookmaker", "source", "market_id",
        )
        if c in nba.columns
    ]
    duplicates = int(nba.duplicated(subset=dedup_columns).sum()) if dedup_columns else 0
    bookmakers = int(nba.get("bookmaker", pd.Series(dtype="object")).dropna().nunique())
    settled = int(
        nba.get("settlement_status", pd.Series(dtype="object")).astype(str).eq("settled").sum()
    )
    closing = int(_truthy(nba.get("is_closing_snapshot", pd.Series(dtype="object"))).sum())

    prop_types = int(nba.get("prop_type", pd.Series(dtype="object")).dropna().nunique())

    # Missing-core-field rate: any core field absent, or both prices missing.
    core_missing = over.isna() & under.isna()
    for field in CORE_FIELDS:
        series = nba.get(field, pd.Series(pd.NA, index=nba.index))
        core_missing = core_missing | series.isna() | series.astype(str).str.strip().eq("")
    missing_core_rows = int(core_missing.sum())

    nba_markets = pd.DataFrame()
    if not line_quality.empty and "league" in line_quality.columns:
        nba_markets = line_quality[line_quality["league"].astype(str).eq("NBA")]
    markets_total = int(len(nba_markets))

    # Row-level main-line annotation: a row is on the main line when its quoted
    # line equals the market's likely_main_line from the line-quality audit.
    settled_mask = nba.get("settlement_status", pd.Series(dtype="object")).astype(str).eq("settled")
    closing_mask = _truthy(nba.get("is_closing_snapshot", pd.Series(dtype="object")))
    if markets_total and "likely_main_line" in nba_markets.columns and all(
        key in nba.columns for key in MARKET_KEYS
    ):
        main_map = nba_markets[MARKET_KEYS + ["likely_main_line"]].drop_duplicates(MARKET_KEYS)
        annotated = nba[MARKET_KEYS + ["line"]].merge(main_map, on=MARKET_KEYS, how="left")
        annotated.index = nba.index
        main_row_mask = (
            pd.to_numeric(annotated["line"], errors="coerce")
            == pd.to_numeric(annotated["likely_main_line"], errors="coerce")
        )
    else:
        main_row_mask = pd.Series(False, index=nba.index)
    settled_main_line_rows = int((settled_mask & main_row_mask).sum())
    closing_main_line_rows = int((closing_mask & main_row_mask).sum())

    settlement_by_prop_type: dict[str, dict[str, Any]] = {}
    if "prop_type" in nba.columns:
        for prop_type, group_idx in nba.groupby(nba["prop_type"].astype(str)).groups.items():
            rows = len(group_idx)
            settled_count = int(settled_mask.loc[group_idx].sum())
            settlement_by_prop_type[str(prop_type)] = {
                "rows": rows,
                "settled": settled_count,
                "settlement_coverage": _rate(settled_count, rows),
            }
    confident_main = (
        int(
            (
                nba_markets["likely_main_line"].notna()
                & (nba_markets["line_quality_label"].astype(str) != "uncertain")
            ).sum()
        )
        if markets_total
        else 0
    )
    markets_with_closing = (
        int(_truthy(nba_markets["has_closing_snapshot"]).sum()) if markets_total else 0
    )

    return {
        "nba_snapshots": n,
        "player_match_rate": _rate(player_matched, n),
        "game_match_rate": game_match_rate,
        "snapshots_for_started_games": n_started,
        "missing_line_rate": _rate(missing_line, n),
        "missing_price_rate": _rate(missing_both_prices, n),
        "suspicious_line_rate": _rate(suspicious_line, n),
        "suspicious_price_rate": _rate(suspicious_price, n),
        "duplicate_rate": _rate(duplicates, n),
        "bookmakers": bookmakers,
        "settled_props": settled,
        "settlement_coverage": _rate(settled, n),
        "closing_like_snapshots": closing,
        "prop_types": prop_types,
        "missing_core_field_rate": _rate(missing_core_rows, n),
        "settled_main_line_rows": settled_main_line_rows,
        "closing_main_line_rows": closing_main_line_rows,
        "settlement_by_prop_type": settlement_by_prop_type,
        "markets_total": markets_total,
        "main_line_rate": _rate(confident_main, markets_total),
        "closing_market_rate": _rate(markets_with_closing, markets_total),
        "clv_markets": int(clv_summary.get("nba_markets_with_clv", 0) or 0),
        "clv_coverage": _rate(
            int(clv_summary.get("nba_markets_with_clv", 0) or 0), markets_total
        ),
    }


def evaluate_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the readiness ladder from the metrics. Pure and testable."""

    t = THRESHOLDS
    if not metrics or metrics.get("nba_snapshots", 0) == 0:
        return {
            "status": "not_ready",
            "checks": {
                "collection": [
                    _check(
                        "nba_snapshots", 0, t["min_nba_snapshots"], False,
                        "No NBA snapshots collected yet.",
                    )
                ],
                "settlement": [],
                "clv": [],
                "modeling": [],
            },
            "blockers": ["No NBA prop snapshots collected yet."],
        }

    collection_checks = [
        _check(
            "nba_snapshots", metrics["nba_snapshots"], t["min_nba_snapshots"],
            metrics["nba_snapshots"] >= t["min_nba_snapshots"],
            "Enough NBA snapshot rows to audit quality meaningfully.",
        ),
        _check(
            "player_match_rate", metrics["player_match_rate"], t["min_player_match_rate"],
            metrics["player_match_rate"] >= t["min_player_match_rate"],
            "NBA prop players resolved to nba_api player ids.",
        ),
        _check(
            "game_match_rate",
            metrics["game_match_rate"],
            t["min_game_match_rate"],
            (
                metrics["game_match_rate"] is None
                or metrics["game_match_rate"] >= t["min_game_match_rate"]
            ),
            (
                "WARNING: no collected NBA game has started yet, so game-key matching "
                "cannot be verified against actuals. Passing provisionally; re-check "
                "after the first game settles."
                if metrics["game_match_rate"] is None
                else "Snapshot game keys found in the nba_api games table (started games only; "
                "unplayed games cannot match by definition)."
            ),
        ),
        _check(
            "main_line_rate", metrics["main_line_rate"], t["min_main_line_rate"],
            metrics["main_line_rate"] >= t["min_main_line_rate"],
            "NBA markets with a confidently detected main line (alt ladders resolved).",
        ),
        _check(
            "missing_line_rate", metrics["missing_line_rate"], t["max_missing_line_rate"],
            metrics["missing_line_rate"] <= t["max_missing_line_rate"],
            "Rows without a line value.",
        ),
        _check(
            "missing_price_rate", metrics["missing_price_rate"], t["max_missing_price_rate"],
            metrics["missing_price_rate"] <= t["max_missing_price_rate"],
            "Rows missing both over and under prices.",
        ),
        _check(
            "suspicious_line_rate", metrics["suspicious_line_rate"], t["max_suspicious_line_rate"],
            metrics["suspicious_line_rate"] <= t["max_suspicious_line_rate"],
            "Lines that are non-positive, absurdly large, or off the half-point grid.",
        ),
        _check(
            "suspicious_price_rate", metrics["suspicious_price_rate"], t["max_suspicious_price_rate"],
            metrics["suspicious_price_rate"] <= t["max_suspicious_price_rate"],
            "Decimal prices outside [1.01, 100].",
        ),
        _check(
            "duplicate_rate", metrics["duplicate_rate"], t["max_duplicate_rate"],
            metrics["duplicate_rate"] <= t["max_duplicate_rate"],
            "Exact duplicate snapshot rows.",
        ),
        _check(
            "bookmakers", metrics["bookmakers"], t["min_bookmakers"],
            metrics["bookmakers"] >= t["min_bookmakers"],
            "Distinct NBA bookmakers collected.",
        ),
    ]

    settlement_checks = [
        _check(
            "settled_props", metrics["settled_props"], t["min_settled_props"],
            metrics["settled_props"] >= t["min_settled_props"],
            "At least one NBA prop actually settled against nba_api actuals.",
        ),
    ]

    clv_checks = [
        _check(
            "closing_market_rate", metrics["closing_market_rate"], t["min_closing_market_rate"],
            metrics["closing_market_rate"] >= t["min_closing_market_rate"],
            "Share of NBA markets with a closing-like snapshot.",
        ),
        _check(
            "clv_markets", metrics["clv_markets"], t["min_clv_markets"],
            metrics["clv_markets"] >= t["min_clv_markets"],
            "NBA markets with a computed early-vs-closing CLV pair.",
        ),
    ]

    modeling_checks = [
        _check(
            "settled_props_for_modeling", metrics["settled_props"], t["min_settled_for_modeling"],
            metrics["settled_props"] >= t["min_settled_for_modeling"],
            "Settled prop outcomes needed before any modeling experiment is honest.",
        ),
        _check(
            "clv_markets_for_modeling", metrics["clv_markets"], t["min_clv_markets_for_modeling"],
            metrics["clv_markets"] >= t["min_clv_markets_for_modeling"],
            "CLV pairs needed to evaluate whether collected prices are usable.",
        ),
        _check(
            "closing_market_rate_for_modeling",
            metrics["closing_market_rate"], t["min_closing_market_rate_for_modeling"],
            metrics["closing_market_rate"] >= t["min_closing_market_rate_for_modeling"],
            "Closing coverage needed so models can be benchmarked against the close.",
        ),
        _check(
            "settled_main_line_rows", metrics.get("settled_main_line_rows", 0),
            t["min_settled_main_line_rows_for_modeling"],
            metrics.get("settled_main_line_rows", 0) >= t["min_settled_main_line_rows_for_modeling"],
            "Settled rows on the detected MAIN line (alt-ladder rows excluded).",
        ),
        _check(
            "closing_main_line_rows", metrics.get("closing_main_line_rows", 0),
            t["min_closing_main_line_rows_for_modeling"],
            metrics.get("closing_main_line_rows", 0) >= t["min_closing_main_line_rows_for_modeling"],
            "Closing-like rows on the detected MAIN line.",
        ),
        _check(
            "missing_core_field_rate", metrics.get("missing_core_field_rate", 1.0),
            t["max_missing_core_field_rate"],
            metrics.get("missing_core_field_rate", 1.0) <= t["max_missing_core_field_rate"],
            "Rows missing any core field (player, prop type, line, bookmaker, time, game key, prices).",
        ),
        _check(
            "main_line_rate_for_modeling", metrics["main_line_rate"],
            t["min_main_line_rate_for_modeling"],
            metrics["main_line_rate"] >= t["min_main_line_rate_for_modeling"],
            "Main-line detection confidence required before modeling on main lines.",
        ),
        _check(
            "bookmakers_for_modeling", metrics["bookmakers"], t["min_bookmakers_for_modeling"],
            metrics["bookmakers"] >= t["min_bookmakers_for_modeling"],
            "Bookmaker diversity required so models do not overfit one book's quirks.",
        ),
        _check(
            "prop_types_for_modeling", metrics.get("prop_types", 0),
            t["min_prop_types_for_modeling"],
            metrics.get("prop_types", 0) >= t["min_prop_types_for_modeling"],
            "Prop-type diversity required for a meaningful baseline.",
        ),
    ]

    status = "not_ready"
    if all(c["passed"] for c in collection_checks):
        status = "collection_ready"
        if all(c["passed"] for c in settlement_checks):
            status = "settlement_ready"
            if all(c["passed"] for c in clv_checks):
                status = "clv_ready"
                if all(c["passed"] for c in modeling_checks):
                    status = "modeling_experiment_ready"

    all_checks = {
        "collection": collection_checks,
        "settlement": settlement_checks,
        "clv": clv_checks,
        "modeling": modeling_checks,
    }
    blockers: list[str] = []
    for rung, checks in all_checks.items():
        for check in checks:
            if not check["passed"]:
                blockers.append(
                    f"[{rung}] {check['check']}: value {check['value']} vs threshold "
                    f"{check['threshold']} — {check['detail']}"
                )
    return {"status": status, "checks": all_checks, "blockers": blockers}


def build_gates_summary(
    nba: pd.DataFrame,
    line_quality: pd.DataFrame,
    clv_summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = compute_quality_metrics(nba, line_quality, clv_summary)
    evaluation = evaluate_gates(metrics)
    return {
        "report": "player_prop_data_quality_gates",
        "gates_version": GATES_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": evaluation["status"],
        "status_ladder": list(STATUS_LADDER),
        "metrics": metrics,
        "checks": evaluation["checks"],
        "blockers": evaluation["blockers"],
        "thresholds": dict(THRESHOLDS),
        "scope": (
            "These gates decide whether NBA prop DATA is ready for research-only modeling "
            "experiments. They do not approve betting and do not loosen any proof gate."
        ),
        "research_only": True,
        "approved": False,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Player Prop Data Quality Gates",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"## Status: `{summary['status']}`",
        "",
        f"Ladder: {' -> '.join(summary['status_ladder'])}",
        "",
        "_Research-only. These gates qualify DATA for modeling experiments; they do not",
        "approve betting, approve parlays, or loosen any proof gate._",
        "",
    ]
    for rung in ("collection", "settlement", "clv", "modeling"):
        checks = summary["checks"].get(rung, [])
        if not checks:
            continue
        lines += [f"## {rung.capitalize()} Checks", "", "| check | value | threshold | passed |", "| --- | --- | --- | --- |"]
        for check in checks:
            lines.append(
                f"| {check['check']} | {check['value']} | {check['threshold']} | "
                f"{'PASS' if check['passed'] else 'FAIL'} |"
            )
        lines.append("")
    settlement_by_prop = summary["metrics"].get("settlement_by_prop_type") or {}
    if settlement_by_prop:
        lines += [
            "## Settlement Coverage By Prop Type",
            "",
            "| prop type | rows | settled | coverage |",
            "| --- | --- | --- | --- |",
        ]
        for prop_type, stats in sorted(settlement_by_prop.items()):
            lines.append(
                f"| {prop_type} | {stats['rows']} | {stats['settled']} | "
                f"{stats['settlement_coverage']:.1%} |"
            )
        lines.append("")
    if summary["blockers"]:
        lines += ["## Blockers", ""]
        lines += [f"- {b}" for b in summary["blockers"]]
        lines.append("")
    else:
        lines += ["## Blockers", "", "- (none)", ""]
    lines += [
        "---",
        "Research-only: data readiness gates. Approved bets and approved parlays remain blocked.",
        "",
    ]
    return "\n".join(lines)


def write_data_quality_gate_reports(project_root: str | Path) -> dict[str, Any]:
    """Read inputs, evaluate the gates, write JSON + MD, return the summary."""

    root = Path(project_root)
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    enriched_path = root / "data" / "processed" / "player_prop_snapshots_enriched.csv"
    enriched = (
        pd.read_csv(enriched_path, low_memory=False) if enriched_path.exists() else pd.DataFrame()
    )
    nba = (
        enriched[enriched["league"].astype(str).str.upper().eq("NBA")]
        if not enriched.empty and "league" in enriched.columns
        else pd.DataFrame()
    )
    line_quality_path = reports / "player_prop_line_quality.csv"
    line_quality = (
        pd.read_csv(line_quality_path, low_memory=False)
        if line_quality_path.exists()
        else pd.DataFrame()
    )
    clv_summary: dict[str, Any] = {}
    clv_path = reports / "player_prop_clv_summary.json"
    if clv_path.exists():
        try:
            clv_summary = json.loads(clv_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            clv_summary = {}

    summary = build_gates_summary(nba, line_quality, clv_summary)

    outputs = {key: reports / filename for key, filename in OUTPUT_FILES.items()}
    outputs["gates_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs["gates_md"].write_text(_render_markdown(summary), encoding="utf-8")
    summary["outputs"] = {key: str(path.relative_to(root)) for key, path in outputs.items()}
    return summary
