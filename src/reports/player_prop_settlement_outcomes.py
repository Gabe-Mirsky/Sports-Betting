"""Player-prop settlement outcome report (research-only).

Summarizes how settled NBA prop snapshots actually graded: win/loss/push by
prop type and bookmaker, over vs under win rates, likely-main-line vs
alternate-line results, and closing-like vs early snapshot results. Pending
props are reported per game with an honest explanation of why they are
pending (usually: the game has not been played or its result is not in the
local nba_api cache yet).

Outputs:

- ``data/reports/player_prop_settlement_outcomes_summary.json``
- ``data/reports/player_prop_settlement_outcomes.csv``
- ``data/reports/player_prop_settlement_outcomes.md``

Reporting only: no models, no recommendations, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked. Win rates here are
descriptive history, not edge — never treat them as a betting signal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


OUTCOMES_VERSION = "v1"

# Below this many settled props every split is a small sample; the report
# carries an explicit warning instead of letting tiny win rates look real.
SMALL_SAMPLE_THRESHOLD = 200

OUTPUT_FILES = {
    "summary_json": "player_prop_settlement_outcomes_summary.json",
    "outcomes_csv": "player_prop_settlement_outcomes.csv",
    "outcomes_md": "player_prop_settlement_outcomes.md",
}

OUTCOME_COLUMNS = [
    "snapshot_time",
    "canonical_game_key",
    "game_date",
    "player_name",
    "team",
    "prop_type",
    "line",
    "actual_stat_value",
    "outcome",
    "over_won",
    "under_won",
    "push",
    "bookmaker",
    "is_closing_snapshot",
    "is_likely_main_line",
    "minutes_to_game_start",
]


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _outcome_label(row: pd.Series) -> str:
    if bool(row["_push"]):
        return "push"
    if bool(row["_over_won"]):
        return "over_won"
    if bool(row["_under_won"]):
        return "under_won"
    return "unknown"


def attach_main_line_flag(settled: pd.DataFrame, likely_main: pd.DataFrame) -> pd.Series:
    """True where the snapshot's line equals the market's likely main line.

    ``likely_main`` is the market-quality output (one row per
    league/player/prop/book/game with ``likely_main_line``). Markets without a
    resolved main line yield False (honest default: unproven, not assumed).
    """

    if settled.empty:
        return pd.Series(dtype=bool)
    if likely_main.empty:
        return pd.Series(False, index=settled.index)
    keys = ["league", "player_name", "prop_type", "bookmaker", "canonical_game_key"]
    main = likely_main[[c for c in keys + ["likely_main_line"] if c in likely_main.columns]].copy()
    if "likely_main_line" not in main.columns or any(k not in main.columns for k in keys):
        return pd.Series(False, index=settled.index)
    merged = settled.merge(main, on=keys, how="left", suffixes=("", "_mq"))
    merged.index = settled.index
    line = pd.to_numeric(merged.get("line"), errors="coerce")
    main_line = pd.to_numeric(merged.get("likely_main_line"), errors="coerce")
    return (line.notna() & main_line.notna() & (line == main_line)).astype(bool)


def _split_counts(frame: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    """Win/loss/push counts per value of ``by`` (e.g. prop_type, bookmaker)."""

    if frame.empty or by not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for value, group in frame.groupby(frame[by].fillna("(missing)").astype(str), sort=True):
        n = int(len(group))
        overs = int(group["_over_won"].sum())
        unders = int(group["_under_won"].sum())
        pushes = int(group["_push"].sum())
        records.append(
            {
                by: value,
                "settled": n,
                "over_won": overs,
                "under_won": unders,
                "push": pushes,
                "over_win_rate": round(overs / n, 4) if n else 0.0,
                "under_win_rate": round(unders / n, 4) if n else 0.0,
            }
        )
    return records


def _rates(frame: pd.DataFrame) -> dict[str, Any]:
    n = int(len(frame))
    overs = int(frame["_over_won"].sum()) if n else 0
    unders = int(frame["_under_won"].sum()) if n else 0
    pushes = int(frame["_push"].sum()) if n else 0
    return {
        "settled": n,
        "over_won": overs,
        "under_won": unders,
        "push": pushes,
        "over_win_rate": round(overs / n, 4) if n else 0.0,
        "under_win_rate": round(unders / n, 4) if n else 0.0,
    }


def build_settlement_outcomes(
    enriched: pd.DataFrame,
    likely_main: pd.DataFrame,
) -> dict[str, Any]:
    """Build the outcome summary + per-row outcomes frame from enriched snapshots."""

    warnings: list[str] = []
    if enriched.empty or "settlement_status" not in enriched.columns:
        warnings.append(
            "No enriched snapshots found. Run scripts/enrich_player_prop_snapshots.py first."
        )
        empty = pd.DataFrame(columns=OUTCOME_COLUMNS)
        return {"summary": _empty_summary(warnings), "outcomes": empty}

    status = enriched["settlement_status"].fillna("").astype(str)
    settled = enriched[status.eq("settled")].copy()
    pending = enriched[status.eq("pending_result")].copy()

    pending_games: list[dict[str, Any]] = []
    if not pending.empty:
        for game_key, rows in pending.groupby(
            pending["canonical_game_key"].fillna("(missing)").astype(str)
        ):
            starts = pd.to_datetime(rows.get("game_start_time"), errors="coerce", utc=True)
            start = starts.dropna().iloc[0].isoformat() if starts.notna().any() else ""
            started = (
                bool(starts.dropna().iloc[0] <= pd.Timestamp.now(tz="UTC"))
                if starts.notna().any()
                else None
            )
            if started is False:
                reason = "game has not been played yet"
            elif started is True:
                reason = (
                    "game started but its result is not in the local nba_api cache yet; "
                    "rerun scripts/refresh_nba_results_and_settle_props.py --download after the game ends"
                )
            else:
                reason = "game start time unknown; result not in the local nba_api cache"
            pending_games.append(
                {
                    "canonical_game_key": game_key,
                    "game_date": str(rows.get("game_date", pd.Series(dtype="object")).dropna().iloc[0])
                    if rows.get("game_date") is not None and rows["game_date"].notna().any()
                    else "",
                    "game_start_time": start,
                    "pending_props": int(len(rows)),
                    "reason": reason,
                }
            )

    if settled.empty:
        warnings.append(
            "No props have settled yet. Settlement requires the game's player game logs in the "
            "local nba_api cache; nothing was forced."
        )
        summary = _empty_summary(warnings)
        summary["pending_props"] = int(len(pending))
        summary["pending_games"] = pending_games
        return {"summary": summary, "outcomes": pd.DataFrame(columns=OUTCOME_COLUMNS)}

    settled["_over_won"] = _truthy(settled.get("over_won", pd.Series(dtype="object")))
    settled["_under_won"] = _truthy(settled.get("under_won", pd.Series(dtype="object")))
    settled["_push"] = _truthy(settled.get("push", pd.Series(dtype="object")))
    settled["_closing"] = _truthy(settled.get("is_closing_snapshot", pd.Series(dtype="object")))
    settled["_main"] = attach_main_line_flag(settled, likely_main)
    settled["outcome"] = settled.apply(_outcome_label, axis=1)

    n_settled = int(len(settled))
    if n_settled < SMALL_SAMPLE_THRESHOLD:
        warnings.append(
            f"SMALL SAMPLE: only {n_settled} settled prop snapshots (threshold "
            f"{SMALL_SAMPLE_THRESHOLD}). Every split below is descriptive noise at this size — "
            "do not treat any win rate as meaningful or predictive."
        )
    warnings.append(
        "Win rates ignore prices/vig and are NOT edge. Research-only; no betting signal."
    )

    closing_rows = settled[settled["_closing"]]
    early_rows = settled[~settled["_closing"]]
    main_rows = settled[settled["_main"]]
    alt_rows = settled[~settled["_main"]]

    summary = {
        "report": "player_prop_settlement_outcomes",
        "outcomes_version": OUTCOMES_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settled_props": n_settled,
        "pending_props": int(len(pending)),
        "overall": _rates(settled),
        "by_prop_type": _split_counts(settled, "prop_type"),
        "by_bookmaker": _split_counts(settled, "bookmaker"),
        "main_line_vs_alt": {
            "likely_main_line": _rates(main_rows),
            "alternate_or_unresolved": _rates(alt_rows),
        },
        "closing_vs_early": {
            "closing_like": _rates(closing_rows),
            "early": _rates(early_rows),
        },
        "pending_games": pending_games,
        "small_sample": n_settled < SMALL_SAMPLE_THRESHOLD,
        "warnings": warnings,
        "research_only": True,
        "approved": False,
    }

    outcomes = pd.DataFrame(
        {
            "snapshot_time": settled.get("snapshot_time"),
            "canonical_game_key": settled.get("canonical_game_key"),
            "game_date": settled.get("game_date"),
            "player_name": settled.get("player_name"),
            "team": settled.get("team"),
            "prop_type": settled.get("prop_type"),
            "line": settled.get("line"),
            "actual_stat_value": settled.get("actual_stat_value"),
            "outcome": settled["outcome"],
            "over_won": settled["_over_won"],
            "under_won": settled["_under_won"],
            "push": settled["_push"],
            "bookmaker": settled.get("bookmaker"),
            "is_closing_snapshot": settled["_closing"],
            "is_likely_main_line": settled["_main"],
            "minutes_to_game_start": settled.get("minutes_to_game_start"),
        }
    )[OUTCOME_COLUMNS]
    return {"summary": summary, "outcomes": outcomes}


def _empty_summary(warnings: list[str]) -> dict[str, Any]:
    return {
        "report": "player_prop_settlement_outcomes",
        "outcomes_version": OUTCOMES_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settled_props": 0,
        "pending_props": 0,
        "overall": {"settled": 0, "over_won": 0, "under_won": 0, "push": 0,
                    "over_win_rate": 0.0, "under_win_rate": 0.0},
        "by_prop_type": [],
        "by_bookmaker": [],
        "main_line_vs_alt": {},
        "closing_vs_early": {},
        "pending_games": [],
        "small_sample": True,
        "warnings": warnings,
        "research_only": True,
        "approved": False,
    }


def _rates_line(label: str, rates: dict[str, Any]) -> str:
    return (
        f"| {label} | {rates.get('settled', 0)} | {rates.get('over_won', 0)} | "
        f"{rates.get('under_won', 0)} | {rates.get('push', 0)} | "
        f"{rates.get('over_win_rate', 0.0):.1%} | {rates.get('under_win_rate', 0.0):.1%} |"
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Player Prop Settlement Outcomes",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "_Research-only settlement history. Win rates ignore prices/vig and are not edge._",
        "",
        "## Totals",
        "",
        f"- Settled props: {summary['settled_props']}",
        f"- Pending props: {summary['pending_props']}",
        "",
    ]
    if summary["settled_props"]:
        header = "| split | settled | over won | under won | push | over win rate | under win rate |"
        divider = "| --- | --- | --- | --- | --- | --- | --- |"
        lines += ["## Overall", "", header, divider, _rates_line("all settled", summary["overall"]), ""]
        if summary["by_prop_type"]:
            lines += ["## By Prop Type", "", header, divider]
            lines += [
                _rates_line(record["prop_type"], record) for record in summary["by_prop_type"]
            ]
            lines.append("")
        if summary["by_bookmaker"]:
            lines += ["## By Bookmaker", "", header, divider]
            lines += [
                _rates_line(record["bookmaker"], record) for record in summary["by_bookmaker"]
            ]
            lines.append("")
        main_alt = summary.get("main_line_vs_alt") or {}
        if main_alt:
            lines += ["## Likely Main Line vs Alternate/Unresolved", "", header, divider]
            lines.append(_rates_line("likely main line", main_alt.get("likely_main_line", {})))
            lines.append(_rates_line("alternate/unresolved", main_alt.get("alternate_or_unresolved", {})))
            lines.append("")
        closing = summary.get("closing_vs_early") or {}
        if closing:
            lines += ["## Closing-Like vs Early Snapshots", "", header, divider]
            lines.append(_rates_line("closing-like", closing.get("closing_like", {})))
            lines.append(_rates_line("early", closing.get("early", {})))
            lines.append("")
    else:
        lines += ["## No Settled Props Yet", ""]
    if summary["pending_games"]:
        lines += ["## Pending Games", ""]
        for game in summary["pending_games"]:
            lines.append(
                f"- `{game['canonical_game_key']}` ({game.get('game_date', '')}): "
                f"{game['pending_props']} props pending — {game['reason']}"
            )
        lines.append("")
    if summary["warnings"]:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in summary["warnings"]]
        lines.append("")
    lines += [
        "---",
        "Research-only: settlement history reporting. No models, recommendations, approved",
        "bets, or parlays. Nothing here is a betting signal.",
        "",
    ]
    return "\n".join(lines)


def write_settlement_outcome_reports(project_root: str | Path) -> dict[str, Any]:
    """Read inputs, write the three outcome reports, return the summary."""

    root = Path(project_root)
    reports = root / "data" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    enriched_path = root / "data" / "processed" / "player_prop_snapshots_enriched.csv"
    enriched = (
        pd.read_csv(enriched_path, low_memory=False) if enriched_path.exists() else pd.DataFrame()
    )
    likely_main_path = reports / "player_prop_likely_main_lines.csv"
    likely_main = (
        pd.read_csv(likely_main_path, low_memory=False)
        if likely_main_path.exists()
        else pd.DataFrame()
    )

    result = build_settlement_outcomes(enriched, likely_main)
    summary = result["summary"]

    outputs = {key: reports / filename for key, filename in OUTPUT_FILES.items()}
    result["outcomes"].to_csv(outputs["outcomes_csv"], index=False)
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs["outcomes_md"].write_text(_render_markdown(summary), encoding="utf-8")
    summary["outputs"] = {key: str(path.relative_to(root)) for key, path in outputs.items()}
    return summary
