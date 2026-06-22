"""Manual data-inspection report for collected player-prop snapshots.

Reads the normalized (and, when present, enriched) prop-snapshot CSVs plus
config/prop_collection.yaml and writes human-reviewable reports under
data/reports/ so the collected data can be verified by eye.

Inspection only: no models, no recommendations, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REVIEW_FIELDS = ("player_name", "prop_type", "line", "over_price", "under_price", "bookmaker")
NBA_ONLY_FIELDS = ("player_id", "canonical_game_key")
LATEST_PROPS_MAX_ROWS = 250

OUTPUT_FILES = {
    "summary_json": "player_prop_manual_review_summary.json",
    "summary_md": "player_prop_manual_review.md",
    "counts_by_league_prop": "player_prop_counts_by_league_prop.csv",
    "counts_by_bookmaker": "player_prop_counts_by_bookmaker.csv",
    "missing_fields": "player_prop_missing_fields.csv",
    "nba_review": "nba_player_prop_review.csv",
    "latest_props": "latest_collected_props.csv",
}


def _missing_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([True] * len(frame), index=frame.index)
    series = frame[column]
    return series.isna() | series.astype(str).str.strip().isin({"", "nan", "None"})


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].fillna("(missing)").astype(str).replace("", "(missing)").value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def configured_league_prop_types(config: dict[str, Any]) -> dict[str, set[str]]:
    """Map each enabled configured league -> set of canonical prop types."""

    configured: dict[str, set[str]] = {}
    for league, league_cfg in (config.get("leagues") or {}).items():
        if not isinstance(league_cfg, dict) or not league_cfg.get("enabled", False):
            continue
        prop_types: set[str] = set()
        for source_cfg in (league_cfg.get("sources") or {}).values():
            if isinstance(source_cfg, dict):
                for canonical in (source_cfg.get("markets") or {}).values():
                    prop_types.add(str(canonical))
        configured[str(league)] = prop_types
    return configured


def _missing_league_reason(league: str, league_cfg: dict[str, Any], config: dict[str, Any]) -> str:
    defaults = config.get("defaults") or {}
    max_leagues = int(defaults.get("max_leagues_per_run", 0) or 0)
    priority = league_cfg.get("priority")
    reasons: list[str] = []
    if max_leagues and isinstance(priority, int) and priority > max_leagues:
        reasons.append(
            f"priority {priority} exceeds max_leagues_per_run={max_leagues}, so the league is "
            "skipped under the per-run quota cap"
        )
    if str(league_cfg.get("sport", "")).lower() == "soccer":
        reasons.append("soccer leagues are collect_only with small event caps; many are between matchdays")
    reasons.append(
        "no upcoming events inside event_horizon_hours (league may be off-season), "
        "or the quota guard stopped the run before reaching this league"
    )
    return f"{league}: " + "; ".join(reasons)


def _missing_prop_reason(league: str, prop_type: str) -> str:
    return (
        f"{league}/{prop_type}: bookmakers did not offer this market for the events collected so far, "
        "or the market was unavailable at snapshot time (some prop markets only post close to tip-off)"
    )


def _nba_usability(
    nba: pd.DataFrame,
    enriched_nba: pd.DataFrame | None,
    summary: dict[str, Any],
) -> tuple[bool, list[str], str]:
    checks: list[str] = []
    usable = True

    if nba.empty:
        return False, ["FAIL: no NBA snapshots collected yet"], "Not usable: no NBA snapshots collected yet."

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal usable
        checks.append(f"{'PASS' if passed else 'FAIL'}: {name} - {detail}")
        if not passed:
            usable = False

    total = len(nba)
    for field in ("player_name", "prop_type", "line"):
        missing = int(_missing_mask(nba, field).sum())
        check(f"{field} present", missing == 0, f"{missing}/{total} missing")

    missing_prices = int((_missing_mask(nba, "over_price") & _missing_mask(nba, "under_price")).sum())
    check("over/under price present", missing_prices == 0, f"{missing_prices}/{total} rows missing both prices")

    if enriched_nba is None or enriched_nba.empty:
        check("enrichment run", False, "no enriched NBA rows; run scripts/enrich_player_prop_snapshots.py")
    else:
        match_rate = summary.get("nba_player_match_rate")
        check(
            "player_id match rate >= 95%",
            match_rate is not None and match_rate >= 0.95,
            f"match rate {match_rate:.1%}" if match_rate is not None else "match rate unavailable",
        )
        if "settlement_supported" in enriched_nba.columns:
            supported = int(_truthy_mask(enriched_nba["settlement_supported"]).sum())
            check(
                "settlement-supported prop types",
                supported > 0,
                f"{supported}/{len(enriched_nba)} enriched rows have a supported settlement stat",
            )

    if usable:
        verdict = (
            "NBA prop data looks usable for future grading: required fields are populated, players match "
            "logged player_ids, and settlement is supported. Grading remains research-only."
        )
    else:
        verdict = "NBA prop data is NOT yet fully usable for grading; see failed checks."
    return usable, checks, verdict


def build_manual_review_summary(
    snaps: pd.DataFrame,
    enriched: pd.DataFrame | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the manual-review summary dict from loaded inputs (pure, testable)."""

    total = int(len(snaps))
    latest_snapshot = None
    closing_count = 0
    if not snaps.empty and "snapshot_time" in snaps.columns:
        times = pd.to_datetime(snaps["snapshot_time"], errors="coerce", utc=True)
        if times.notna().any():
            latest_snapshot = times.max().isoformat()
    if not snaps.empty and "is_closing_snapshot" in snaps.columns:
        closing_count = int(_truthy_mask(snaps["is_closing_snapshot"]).sum())

    by_league = _value_counts(snaps, "league")
    by_prop = _value_counts(snaps, "prop_type")
    by_bookmaker = _value_counts(snaps, "bookmaker")

    league_prop_records: list[dict[str, Any]] = []
    if not snaps.empty and {"league", "prop_type"}.issubset(snaps.columns):
        grouped = (
            snaps.groupby(
                [snaps["league"].fillna("(missing)").astype(str), snaps["prop_type"].fillna("(missing)").astype(str)]
            )
            .size()
            .reset_index()
        )
        grouped.columns = ["league", "prop_type", "snapshots"]
        grouped = grouped.sort_values(["league", "snapshots"], ascending=[True, False])
        league_prop_records = grouped.to_dict("records")

    missing_fields = {field: int(_missing_mask(snaps, field).sum()) for field in REVIEW_FIELDS}

    nba = snaps[snaps.get("league", pd.Series(dtype="object")).astype(str) == "NBA"] if not snaps.empty else snaps
    enriched_nba: pd.DataFrame | None = None
    if enriched is not None and not enriched.empty and "league" in enriched.columns:
        enriched_nba = enriched[enriched["league"].astype(str) == "NBA"]

    # player_id / canonical_game_key are filled by enrichment, so report missing
    # counts from the enriched file when it exists.
    nba_field_source = "enriched" if enriched_nba is not None and not enriched_nba.empty else "normalized"
    nba_field_frame = enriched_nba if nba_field_source == "enriched" else nba
    nba_missing = {
        field: int(_missing_mask(nba_field_frame, field).sum()) if nba_field_frame is not None else 0
        for field in NBA_ONLY_FIELDS
    }

    nba_player_match_rate = None
    if enriched_nba is not None and not enriched_nba.empty:
        if "player_matched" in enriched_nba.columns:
            nba_player_match_rate = float(_truthy_mask(enriched_nba["player_matched"]).mean())
        else:
            nba_player_match_rate = float((~_missing_mask(enriched_nba, "player_id")).mean())

    nba_players: list[str] = []
    nba_player_prop_types: dict[str, list[str]] = {}
    if not nba.empty and "player_name" in nba.columns:
        named = nba[~_missing_mask(nba, "player_name")]
        nba_players = sorted(named["player_name"].astype(str).str.strip().unique().tolist())
        if "prop_type" in named.columns:
            grouped_props = named.groupby(named["player_name"].astype(str).str.strip())["prop_type"]
            nba_player_prop_types = {
                player: sorted({str(p) for p in props.dropna()}) for player, props in grouped_props
            }

    configured = configured_league_prop_types(config)
    leagues_cfg = config.get("leagues") or {}
    collected_leagues = set(by_league)
    leagues_not_collected = sorted(set(configured) - collected_leagues)

    missing_prop_types_by_league: dict[str, list[str]] = {}
    for league, prop_types in configured.items():
        if league not in collected_leagues:
            continue
        collected_props = {
            str(record["prop_type"]) for record in league_prop_records if record["league"] == league
        }
        missing = sorted(prop_types - collected_props)
        if missing:
            missing_prop_types_by_league[league] = missing

    missing_reasons = [
        _missing_league_reason(league, leagues_cfg.get(league) or {}, config) for league in leagues_not_collected
    ]
    missing_reasons += [
        _missing_prop_reason(league, prop_type)
        for league, props in sorted(missing_prop_types_by_league.items())
        for prop_type in props
    ]

    league_readiness: dict[str, str] = {}
    for league in sorted(configured):
        league_cfg = leagues_cfg.get(league) or {}
        collected_count = by_league.get(league, 0)
        if league_cfg.get("modeling_priority"):
            league_readiness[league] = (
                f"modeling priority (NBA pipeline target); {collected_count} snapshots collected; "
                "still research-only - no prop models or recommendations exist"
            )
        else:
            league_readiness[league] = (
                f"collection-only (historical database build; no model planned yet); "
                f"{collected_count} snapshots collected"
            )

    summary: dict[str, Any] = {
        "report": "player_prop_manual_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "approved": False,
        "total_snapshots": total,
        "latest_snapshot_time_utc": latest_snapshot,
        "closing_like_snapshots": closing_count,
        "snapshots_by_league": by_league,
        "snapshots_by_prop_type": by_prop,
        "snapshots_by_league_prop": league_prop_records,
        "snapshots_by_bookmaker": by_bookmaker,
        "missing_fields": missing_fields,
        "nba_missing_fields": {"source": nba_field_source, **nba_missing},
        "nba_snapshots": int(len(nba)) if nba is not None else 0,
        "nba_player_match_rate": nba_player_match_rate,
        "nba_players": nba_players,
        "nba_player_prop_types": nba_player_prop_types,
        "configured_leagues_not_collected": leagues_not_collected,
        "configured_prop_types_not_collected": missing_prop_types_by_league,
        "possible_missing_reasons": missing_reasons,
        "league_readiness": league_readiness,
        "enriched_data_available": bool(enriched_nba is not None and not enriched_nba.empty),
    }

    usable, checks, verdict = _nba_usability(nba, enriched_nba, summary)
    summary["nba_usable_for_grading"] = usable
    summary["nba_usability_checks"] = checks
    summary["nba_usability_verdict"] = verdict
    return summary


def _nba_review_frame(nba: pd.DataFrame, enriched_nba: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "player_name", "prop_type", "snapshots", "bookmakers", "min_line", "max_line",
        "first_snapshot_time", "last_snapshot_time", "player_id",
    ]
    if nba.empty or "player_name" not in nba.columns:
        return pd.DataFrame(columns=columns)

    frame = nba.copy()
    frame["player_name"] = frame["player_name"].astype(str).str.strip()
    frame["line"] = pd.to_numeric(frame.get("line"), errors="coerce")
    grouped = (
        frame.groupby(["player_name", "prop_type"], dropna=False)
        .agg(
            snapshots=("player_name", "size"),
            bookmakers=("bookmaker", lambda s: s.dropna().astype(str).nunique()),
            min_line=("line", "min"),
            max_line=("line", "max"),
            first_snapshot_time=("snapshot_time", "min"),
            last_snapshot_time=("snapshot_time", "max"),
        )
        .reset_index()
    )
    if enriched_nba is not None and not enriched_nba.empty and "player_id" in enriched_nba.columns:
        ids = enriched_nba.copy()
        ids["player_name"] = ids["player_name"].astype(str).str.strip()
        id_map = (
            ids[~_missing_mask(ids, "player_id")]
            .groupby("player_name")["player_id"]
            .first()
        )
        grouped["player_id"] = grouped["player_name"].map(id_map)
    else:
        grouped["player_id"] = None
    return grouped.sort_values(["player_name", "prop_type"]).reset_index(drop=True)[columns]


def _missing_fields_frame(summary: dict[str, Any], total: int, nba_total: int) -> pd.DataFrame:
    rows = [
        {
            "field": field,
            "scope": "all leagues",
            "missing_count": count,
            "total_rows": total,
            "missing_pct": round(count / total, 4) if total else 0.0,
        }
        for field, count in summary["missing_fields"].items()
    ]
    nba_missing = summary["nba_missing_fields"]
    for field in NBA_ONLY_FIELDS:
        count = int(nba_missing.get(field, 0))
        rows.append(
            {
                "field": field,
                "scope": f"NBA ({nba_missing.get('source', 'normalized')} file)",
                "missing_count": count,
                "total_rows": nba_total,
                "missing_pct": round(count / nba_total, 4) if nba_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _latest_props_frame(snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty or "snapshot_time" not in snaps.columns:
        return snaps.head(0)
    ordered = snaps.copy()
    ordered["_ts"] = pd.to_datetime(ordered["snapshot_time"], errors="coerce", utc=True)
    ordered = ordered.sort_values("_ts", ascending=False).drop(columns=["_ts"])
    return ordered.head(LATEST_PROPS_MAX_ROWS)


def _fmt_count_lines(counts: dict[str, int]) -> str:
    if not counts:
        return "- (none collected yet)\n"
    return "".join(f"- {key}: {value}\n" for key, value in counts.items())


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Player Prop Manual Review",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "Research-only data inspection. No models, recommendations, approved bets, or parlays.",
        "",
        "## Totals",
        "",
        f"- Total snapshots: {summary['total_snapshots']}",
        f"- Latest snapshot (UTC): {summary['latest_snapshot_time_utc'] or 'n/a'}",
        f"- Closing-like snapshots: {summary['closing_like_snapshots']}",
        f"- NBA snapshots: {summary['nba_snapshots']}",
        "",
        "## Snapshots By League",
        "",
        _fmt_count_lines(summary["snapshots_by_league"]).rstrip(),
        "",
        "## Snapshots By Prop Type",
        "",
        _fmt_count_lines(summary["snapshots_by_prop_type"]).rstrip(),
        "",
        "## Snapshots By League + Prop Type",
        "",
        "| league | prop_type | snapshots |",
        "| --- | --- | --- |",
    ]
    for record in summary["snapshots_by_league_prop"]:
        lines.append(f"| {record['league']} | {record['prop_type']} | {record['snapshots']} |")
    lines += [
        "",
        "## Snapshots By Bookmaker",
        "",
        _fmt_count_lines(summary["snapshots_by_bookmaker"]).rstrip(),
        "",
        "## Missing Fields",
        "",
    ]
    for field, count in summary["missing_fields"].items():
        lines.append(f"- missing {field}: {count}")
    nba_missing = summary["nba_missing_fields"]
    for field in NBA_ONLY_FIELDS:
        lines.append(f"- NBA missing {field} ({nba_missing.get('source')} file): {nba_missing.get(field, 0)}")
    match_rate = summary["nba_player_match_rate"]
    lines += [
        "",
        "## NBA Players Collected",
        "",
        f"- Players: {len(summary['nba_players'])}",
        f"- Player match rate (enriched): {f'{match_rate:.1%}' if match_rate is not None else 'n/a (no enriched data)'}",
        "",
    ]
    for player in summary["nba_players"]:
        props = ", ".join(summary["nba_player_prop_types"].get(player, [])) or "(none)"
        lines.append(f"- {player}: {props}")
    lines += [
        "",
        "## Configured But Not Collected",
        "",
        "Leagues configured but with no snapshots yet:",
        "",
    ]
    if summary["configured_leagues_not_collected"]:
        lines += [f"- {league}" for league in summary["configured_leagues_not_collected"]]
    else:
        lines.append("- (every enabled league has at least one snapshot)")
    lines += ["", "Configured prop types missing from collected leagues:", ""]
    if summary["configured_prop_types_not_collected"]:
        for league, props in sorted(summary["configured_prop_types_not_collected"].items()):
            lines.append(f"- {league}: {', '.join(props)}")
    else:
        lines.append("- (every configured prop type was collected at least once)")
    lines += ["", "Possible reasons:", ""]
    if summary["possible_missing_reasons"]:
        lines += [f"- {reason}" for reason in summary["possible_missing_reasons"]]
    else:
        lines.append("- (nothing missing)")
    lines += [
        "",
        "## NBA Usability For Future Grading",
        "",
        f"Verdict: {summary['nba_usability_verdict']}",
        "",
    ]
    lines += [f"- {check}" for check in summary["nba_usability_checks"]]
    lines += [
        "",
        "## League Readiness",
        "",
    ]
    for league, readiness in summary["league_readiness"].items():
        lines.append(f"- {league}: {readiness}")
    lines += [
        "",
        "---",
        "Research-only: this report inspects collected data. It does not enable any "
        "models, recommendations, approved bets, or approved parlays.",
        "",
    ]
    return "\n".join(lines)


def write_manual_review_reports(
    project_root: str | Path,
    normalized_path: str | Path | None = None,
    enriched_path: str | Path | None = None,
    config_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read inputs, write all manual-review outputs, and return the summary."""

    from data.prop_collection import load_prop_collection_config

    root = Path(project_root)
    normalized = Path(normalized_path) if normalized_path else root / "data" / "processed" / "player_prop_snapshots_normalized.csv"
    enriched_file = Path(enriched_path) if enriched_path else root / "data" / "processed" / "player_prop_snapshots_enriched.csv"
    config_file = Path(config_path) if config_path else root / "config" / "prop_collection.yaml"
    out_dir = Path(reports_dir) if reports_dir else root / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = pd.read_csv(normalized, low_memory=False) if normalized.exists() else pd.DataFrame()
    enriched = pd.read_csv(enriched_file, low_memory=False) if enriched_file.exists() else None
    config = load_prop_collection_config(config_file)

    summary = build_manual_review_summary(snaps, enriched, config)

    nba = snaps[snaps.get("league", pd.Series(dtype="object")).astype(str) == "NBA"] if not snaps.empty else snaps
    enriched_nba = None
    if enriched is not None and not enriched.empty and "league" in enriched.columns:
        enriched_nba = enriched[enriched["league"].astype(str) == "NBA"]

    league_prop = pd.DataFrame(summary["snapshots_by_league_prop"], columns=["league", "prop_type", "snapshots"])
    bookmaker = pd.DataFrame(
        [{"bookmaker": key, "snapshots": value} for key, value in summary["snapshots_by_bookmaker"].items()],
        columns=["bookmaker", "snapshots"],
    )
    missing = _missing_fields_frame(summary, summary["total_snapshots"], summary["nba_snapshots"])
    nba_review = _nba_review_frame(nba, enriched_nba)
    latest = _latest_props_frame(snaps)

    outputs = {key: out_dir / filename for key, filename in OUTPUT_FILES.items()}
    league_prop.to_csv(outputs["counts_by_league_prop"], index=False)
    bookmaker.to_csv(outputs["counts_by_bookmaker"], index=False)
    missing.to_csv(outputs["missing_fields"], index=False)
    nba_review.to_csv(outputs["nba_review"], index=False)
    latest.to_csv(outputs["latest_props"], index=False)

    summary["inputs"] = {
        "normalized_path": str(normalized),
        "normalized_exists": normalized.exists(),
        "enriched_path": str(enriched_file),
        "enriched_exists": enriched_file.exists(),
        "config_path": str(config_file),
    }
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs["summary_md"].write_text(_build_markdown(summary), encoding="utf-8")
    return summary
