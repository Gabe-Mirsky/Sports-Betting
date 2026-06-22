"""Read-only data-quality audit for the free historical game inventory.

Audits ``data/processed/historical_game_inventory.csv`` (and the
``data/staging/free_backfill/*.csv`` it was built from) and emits a JSON + MD
report plus a single verdict: ``clean`` / ``usable_with_warnings`` / ``needs_fix``.

This is purely diagnostic. It enables NO betting, predictions, parlays, or gate
changes, and it makes NO network calls. The inventory is mostly outcome / game-odds
data, not player-prop odds - the audit makes that explicit.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.canonical_games import normalize_league
from data.free_backfill import INVENTORY_PATH, STAGING_DIR, _read_staging_files, to_bool_series

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
AUDIT_JSON = REPORTS_DIR / "historical_game_inventory_quality_audit.json"
AUDIT_MD = REPORTS_DIR / "historical_game_inventory_quality_audit.md"

FLAG_COLUMNS = ("outcome_available", "game_odds_available", "prop_odds_available", "closing_available")
STATUS_ORDER = [
    "settled", "closing_recorded", "prop_odds_recorded", "game_odds_recorded",
    "outcome_only", "scheduled_only",
]
_REQUIRED_FIELDS = ("home_team", "away_team", "game_date", "league")
_PROP_SOURCES = {"sportsgameodds"}


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _examples(values: Any, limit: int = 5) -> list[str]:
    return [str(value) for value in list(values)[:limit]]


def _coerce_flags(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for flag in FLAG_COLUMNS:
        frame[flag] = to_bool_series(frame[flag]) if flag in frame.columns else False
    return frame


def _derive_status(frame: pd.DataFrame) -> pd.Series:
    """Inventory-only status (mirrors the dashboard ladder minus live settlement).

    ``settled`` needs live prop settlement, which the inventory does not carry, so
    it is never assigned here.
    """

    status = pd.Series("scheduled_only", index=frame.index, dtype="object")
    status[frame["outcome_available"]] = "outcome_only"
    status[frame["game_odds_available"]] = "game_odds_recorded"
    status[frame["prop_odds_available"]] = "prop_odds_recorded"
    status[frame["closing_available"]] = "closing_recorded"
    return status


def build_quality_audit(
    inventory_path: Path | str = INVENTORY_PATH,
    staging_dir: Path | str = STAGING_DIR,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or datetime.now(timezone.utc).date()
    inventory_path = Path(inventory_path)
    staging_dir = Path(staging_dir)

    inv = _read_csv(inventory_path)
    staging = _read_staging_files(staging_dir) if Path(staging_dir).exists() else pd.DataFrame()

    checks: dict[str, Any] = {}
    critical: list[str] = []
    warnings: list[str] = []

    if inv.empty:
        return {
            "report": "historical_game_inventory_quality_audit",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "inventory_path": str(inventory_path),
            "staging_dir": str(staging_dir),
            "total_games": 0,
            "checks": {},
            "critical_issues": ["inventory is empty or missing"],
            "warning_issues": [],
            "verdict": "needs_fix",
        }

    inv = _coerce_flags(inv)
    status = _derive_status(inv)

    # 1) Totals by league and by (individual) source.
    by_league = inv.groupby("league").size().sort_values(ascending=False)
    exploded = (
        inv.get("source", pd.Series(dtype=str)).fillna("").astype(str)
        .str.split(",").explode().str.strip()
    )
    by_source = exploded[exploded != ""].value_counts()
    checks["totals_by_league"] = {str(k): int(v) for k, v in by_league.items()}
    checks["totals_by_source"] = {str(k): int(v) for k, v in by_source.items()}

    # 2) Counts by status (inventory-derived; settled needs live settlement).
    status_counts = status.value_counts()
    checks["counts_by_status"] = {s: int(status_counts.get(s, 0)) for s in STATUS_ORDER}

    # 3) Duplicate canonical_game_key.
    key_counts = inv["canonical_game_key"].astype(str).value_counts()
    dup_keys = key_counts[key_counts > 1]
    dup_total = int(dup_keys.sum() - len(dup_keys)) if len(dup_keys) else 0
    checks["duplicate_canonical_keys"] = {
        "distinct_duplicated_keys": int(len(dup_keys)),
        "excess_rows": dup_total,
        "examples": _examples(dup_keys.index),
    }
    if dup_total > 0:
        critical.append(f"{dup_total} duplicate canonical_game_key rows in the inventory")

    # 4) MLB doubleheader collisions (key has no game number) - measured in staging.
    collisions_by_league: dict[str, int] = {}
    mlb_collisions = 0
    if not staging.empty and "canonical_game_key" in staging.columns and "league" in staging.columns:
        for league, grp in staging.groupby(staging["league"].astype(str)):
            collided = int(len(grp) - grp["canonical_game_key"].astype(str).nunique())
            if collided:
                collisions_by_league[str(league)] = collided
        mlb_collisions = collisions_by_league.get("MLB", 0)
    checks["mlb_doubleheader_collisions"] = {
        "count": int(mlb_collisions),
        "collisions_by_league": collisions_by_league,
        "note": "Same date + same teams collapse because canonical_game_key omits game number.",
    }
    if mlb_collisions > 0:
        warnings.append(f"{mlb_collisions} MLB doubleheader rows collapsed (no game number in key)")

    # 5) Missing / blank required fields.
    missing: dict[str, int] = {}
    for field in _REQUIRED_FIELDS:
        if field in inv.columns:
            blank = inv[field].isna() | (inv[field].astype(str).str.strip() == "")
            missing[field] = int(blank.sum())
        else:
            missing[field] = int(len(inv))
    checks["missing_fields"] = missing
    nonzero_missing = {field: count for field, count in missing.items() if count}
    if nonzero_missing:
        critical.append(f"missing/blank required fields: {nonzero_missing}")

    # 6) Invalid / future dates.
    parsed = pd.to_datetime(inv["game_date"], errors="coerce")
    invalid_dates = int(parsed.isna().sum())
    future_mask = parsed.dt.date > today
    future_dates = int(future_mask.fillna(False).sum())
    valid = parsed.dropna()
    checks["dates"] = {
        "invalid": invalid_dates,
        "future": future_dates,
        "min": valid.min().date().isoformat() if not valid.empty else None,
        "max": valid.max().date().isoformat() if not valid.empty else None,
        "future_examples": _examples(inv.loc[future_mask.fillna(False), "canonical_game_key"]),
    }
    if invalid_dates > 0:
        critical.append(f"{invalid_dates} rows with invalid game_date")
    if future_dates > 0:
        warnings.append(f"{future_dates} rows with a future game_date")

    # 7) closing_available true but game_odds_available false, excluding the
    # documented prop-odds exception where a prop source has closing data but no
    # game-level price.
    closing_no_game = inv["closing_available"] & ~inv["game_odds_available"]
    documented_prop_closing = closing_no_game & inv["prop_odds_available"]
    undocumented_closing_no_game = closing_no_game & ~inv["prop_odds_available"]
    checks["closing_without_game_odds"] = {
        "count": int(undocumented_closing_no_game.sum()),
        "total_including_documented_exceptions": int(closing_no_game.sum()),
        "documented_prop_closing_exceptions": int(documented_prop_closing.sum()),
        "examples": _examples(inv.loc[undocumented_closing_no_game, "canonical_game_key"]),
        "note": "Undocumented rows usually mean a source flagged a closing column file-wide; prop-closing rows are allowed when prop_odds_available is true.",
    }
    if int(undocumented_closing_no_game.sum()) > 0:
        warnings.append(f"{int(undocumented_closing_no_game.sum())} rows flag closing but have no game odds")

    # 8) prop_odds_available true but no prop source file/row exists.
    prop_sources_seen = set()
    if not staging.empty and "prop_odds_available" in staging.columns and "source" in staging.columns:
        prop_rows_staging = staging[to_bool_series(staging["prop_odds_available"])]
        prop_sources_seen = {str(s) for s in prop_rows_staging["source"].unique()}
    prop_rows = inv[inv["prop_odds_available"]]
    prop_violation = 0
    for sources in prop_rows.get("source", pd.Series(dtype=str)).fillna("").astype(str):
        row_sources = {s.strip() for s in sources.split(",") if s.strip()}
        if not (row_sources & (_PROP_SOURCES | prop_sources_seen)):
            prop_violation += 1
    checks["prop_without_source"] = {
        "prop_rows": int(len(prop_rows)),
        "prop_sources_in_staging": sorted(prop_sources_seen),
        "violations": int(prop_violation),
    }
    if prop_violation > 0:
        critical.append(f"{prop_violation} rows claim prop odds with no prop source file")

    # 9) outcome_available false but status == outcome_only (logical inconsistency).
    inconsistency = int(((status == "outcome_only") & (~inv["outcome_available"])).sum())
    checks["status_outcome_inconsistency"] = {"count": inconsistency}
    if inconsistency > 0:
        critical.append(f"{inconsistency} rows are outcome_only but outcome_available is false")

    # 10) League normalization (esp. WORLD_CUP vs WORLDCUP).
    key_league = inv["canonical_game_key"].astype(str).str.split("|").str[1].fillna("")
    norm_stored = inv["league"].astype(str).map(normalize_league)
    key_mismatch = int((key_league != norm_stored).sum())
    labels_needing_norm = sorted(
        {lg for lg in inv["league"].astype(str).unique() if lg and lg != normalize_league(lg)}
    )
    world_cup_labels = sorted({lg for lg in inv["league"].astype(str).unique() if "WORLD" in lg.upper()})
    checks["league_normalization"] = {
        "key_vs_label_mismatch": key_mismatch,
        "labels_needing_normalization": labels_needing_norm,
        "world_cup_labels": world_cup_labels,
        "note": "Canonical keys store leagues normalized (WORLD_CUP -> WORLDCUP); dashboard filters normalize the tab label to match.",
    }
    if key_mismatch > 0:
        critical.append(f"{key_mismatch} rows whose key league != normalized league label")
    if labels_needing_norm:
        warnings.append(f"{len(labels_needing_norm)} league labels need normalization to match keys (e.g. WORLD_CUP)")

    # 11) Source-by-source row counts and date ranges (from staging).
    source_ranges: dict[str, Any] = {}
    if not staging.empty and "source" in staging.columns:
        for source, grp in staging.groupby(staging["source"].astype(str)):
            dates = pd.to_datetime(grp.get("game_date"), errors="coerce").dropna()
            source_ranges[str(source)] = {
                "rows": int(len(grp)),
                "min_date": dates.min().date().isoformat() if not dates.empty else None,
                "max_date": dates.max().date().isoformat() if not dates.empty else None,
            }
    checks["source_date_ranges"] = source_ranges

    # Informational: prop coverage (the inventory is mostly outcome/game-odds).
    prop_total = int(inv["prop_odds_available"].sum())
    checks["prop_odds_coverage"] = {
        "prop_games": prop_total,
        "share": round(prop_total / len(inv), 6) if len(inv) else 0.0,
        # Informational only (coverage, not a defect): the inventory is mostly
        # outcome/game-odds; prop odds require the capped SGO backfill.
        "note": "Historical inventory is outcome/game-odds; prop odds require the capped SGO backfill.",
    }

    verdict = "needs_fix" if critical else ("usable_with_warnings" if warnings else "clean")

    return {
        "report": "historical_game_inventory_quality_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "inventory_path": str(inventory_path),
        "staging_dir": str(staging_dir),
        "total_games": int(len(inv)),
        "checks": checks,
        "critical_issues": critical,
        "warning_issues": warnings,
        "verdict": verdict,
    }


def _kv_table(title_a: str, title_b: str, mapping: dict[str, Any], limit: int = 40) -> str:
    rows = [f"| {title_a} | {title_b} |", "| --- | ---: |"]
    for key, value in list(mapping.items())[:limit]:
        rows.append(f"| {key} | {value} |")
    if not mapping:
        rows.append("| (none) | 0 |")
    return "\n".join(rows)


def build_quality_audit_markdown(audit: dict[str, Any]) -> str:
    checks = audit.get("checks", {})
    verdict = audit.get("verdict", "unknown")
    emoji = {"clean": "OK", "usable_with_warnings": "WARN", "needs_fix": "FAIL"}.get(verdict, "?")
    dates = checks.get("dates", {})
    norm = checks.get("league_normalization", {})
    lines = [
        "# Historical Game Inventory - Quality Audit",
        "",
        f"**Verdict: {verdict.upper()} [{emoji}]**  -  research-only; no betting/predictions/gates affected.",
        "",
        f"- Generated: {audit.get('generated_at_utc')}",
        f"- Inventory: `{audit.get('inventory_path')}`",
        f"- Total games: **{audit.get('total_games'):,}**",
        "",
        "## Critical issues",
        "",
        ("\n".join(f"- {issue}" for issue in audit.get("critical_issues", [])) or "- none"),
        "",
        "## Warnings",
        "",
        ("\n".join(f"- {issue}" for issue in audit.get("warning_issues", [])) or "- none"),
        "",
        "## Counts by status (inventory-derived)",
        "",
        _kv_table("status", "games", checks.get("counts_by_status", {})),
        "",
        "_settled needs live prop settlement, which the historical inventory does not carry._",
        "",
        "## Totals by league",
        "",
        _kv_table("league", "games", checks.get("totals_by_league", {})),
        "",
        "## Totals by source",
        "",
        _kv_table("source", "games", checks.get("totals_by_source", {})),
        "",
        "## Source date ranges",
        "",
        _kv_table("source", "rows / min / max", {
            k: f"{v['rows']} / {v['min_date']} / {v['max_date']}"
            for k, v in checks.get("source_date_ranges", {}).items()
        }),
        "",
        "## Integrity checks",
        "",
        f"- Duplicate canonical_game_key (excess rows): **{checks.get('duplicate_canonical_keys', {}).get('excess_rows', 0)}**",
        f"- MLB doubleheader collisions (staging): **{checks.get('mlb_doubleheader_collisions', {}).get('count', 0)}**",
        f"- Missing/blank required fields: **{checks.get('missing_fields', {})}**",
        f"- Invalid dates: **{dates.get('invalid', 0)}**, future dates: **{dates.get('future', 0)}** (range {dates.get('min')} .. {dates.get('max')})",
        f"- closing_available true but game_odds_available false: **{checks.get('closing_without_game_odds', {}).get('count', 0)}** "
        f"(documented prop exceptions: {checks.get('closing_without_game_odds', {}).get('documented_prop_closing_exceptions', 0)})",
        f"- prop_odds_available with no prop source: **{checks.get('prop_without_source', {}).get('violations', 0)}** (prop rows: {checks.get('prop_without_source', {}).get('prop_rows', 0)})",
        f"- outcome_only but outcome_available false: **{checks.get('status_outcome_inconsistency', {}).get('count', 0)}**",
        f"- League key/label mismatch: **{norm.get('key_vs_label_mismatch', 0)}**; labels needing normalization: {norm.get('labels_needing_normalization', [])}",
        "",
    ]
    return "\n".join(lines)


def write_quality_audit(
    reports_dir: Path | str = REPORTS_DIR,
    inventory_path: Path | str = INVENTORY_PATH,
    staging_dir: Path | str = STAGING_DIR,
) -> dict[str, Path]:
    """Build the audit and write the JSON + MD report; return their paths."""

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit = build_quality_audit(inventory_path=inventory_path, staging_dir=staging_dir)
    json_path = reports_dir / "historical_game_inventory_quality_audit.json"
    md_path = reports_dir / "historical_game_inventory_quality_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    md_path.write_text(build_quality_audit_markdown(audit), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
