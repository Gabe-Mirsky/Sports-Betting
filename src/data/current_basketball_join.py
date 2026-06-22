"""Join current-era zachht game-odds snapshots to nba_api games via canonical keys.

zachht has no explicit game date or home/away flag, so each zachht game gets an
estimated game date (closing snapshot, UTC -> US/Eastern) and an assumed
orientation (Pinnacle lists 'away vs home', so team2 = home). The join tries
the assumed orientation first, then the reversed orientation, at date offsets
0, -1, +1 days, and reports exactly which combination matched so the
assumptions are verified empirically.

Research-only diagnostics: no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .canonical_games import (
    GAME_KEY_VERSION,
    build_canonical_game_key,
    collapse_zachht_games,
)


DATE_OFFSETS = (0, -1, 1)


def _shift_date(date_iso: str, offset_days: int) -> str:
    return (pd.Timestamp(date_iso) + pd.Timedelta(days=offset_days)).date().isoformat()


def _safe_key(league: object, date: object, home: object, away: object) -> str:
    try:
        return build_canonical_game_key("basketball", league, date, home, away)
    except ValueError:
        return ""


def join_zachht_to_current_games(
    zachht_games: pd.DataFrame,
    nba_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match zachht per-game rows to nba_api current games.

    Returns ``(matched, unmatched_zachht, unmatched_nba)``. ``matched`` records
    the date offset and orientation that produced each match so the zachht
    date/home-away assumptions can be audited.
    """

    if nba_games.empty or zachht_games.empty:
        return pd.DataFrame(), zachht_games.copy(), nba_games.copy()

    nba = nba_games.copy()
    nba["canonical_game_key"] = nba["canonical_game_key"].astype(str)
    nba_lookup: dict[str, list[int]] = {}
    for idx, key in nba["canonical_game_key"].items():
        if key:
            nba_lookup.setdefault(key, []).append(idx)

    matched_rows: list[dict[str, Any]] = []
    unmatched_rows: list[int] = []
    claimed_nba: set[int] = set()

    for idx, row in zachht_games.iterrows():
        est_date = str(row["game_date"])
        home, away = str(row["home_team_abbr"]), str(row["away_team_abbr"])
        league = row["league"]
        found = None
        if est_date:
            for offset in DATE_OFFSETS:
                date = _shift_date(est_date, offset)
                for orientation, (h, a) in (("as_assumed", (home, away)), ("reversed", (away, home))):
                    key = _safe_key(league, date, h, a)
                    candidates = [i for i in nba_lookup.get(key, []) if i not in claimed_nba]
                    if candidates:
                        found = (candidates[0], key, offset, orientation)
                        break
                if found:
                    break
        if found is None:
            unmatched_rows.append(idx)
            continue
        nba_idx, key, offset, orientation = found
        claimed_nba.add(nba_idx)
        nba_row = nba.loc[nba_idx]
        matched_rows.append(
            {
                "canonical_game_key": key,
                "key_version": GAME_KEY_VERSION,
                "zachht_game_ref": row["source_game_id"],
                "zachht_estimated_game_date": est_date,
                "zachht_closing_snapshot_time": row.get("closing_snapshot_time"),
                "zachht_n_snapshots": row.get("n_snapshots"),
                "nba_game_id": nba_row["game_id"],
                "nba_game_date": str(pd.Timestamp(nba_row["game_date"]).date()),
                "home_team_abbr": nba_row["home_team_abbr"],
                "away_team_abbr": nba_row["away_team_abbr"],
                "home_score": nba_row.get("home_score"),
                "away_score": nba_row.get("away_score"),
                "date_offset_days": offset,
                "orientation": orientation,
            }
        )

    matched = pd.DataFrame(matched_rows)
    unmatched_zachht = zachht_games.loc[unmatched_rows].copy()
    unmatched_nba = nba.loc[[i for i in nba.index if i not in claimed_nba]].copy()
    return matched, unmatched_zachht, unmatched_nba


def build_join_report(
    matched: pd.DataFrame,
    unmatched_zachht: pd.DataFrame,
    unmatched_nba: pd.DataFrame,
    zachht_games: pd.DataFrame,
    nba_games: pd.DataFrame,
    non_nba_zachht_games: int = 0,
) -> dict[str, Any]:
    """Build the join summary dictionary (counts, mismatch diagnostics, grading feasibility)."""

    n_zachht = int(len(zachht_games))
    n_nba = int(len(nba_games))
    n_matched = int(len(matched))

    offsets = (
        {str(k): int(v) for k, v in matched["date_offset_days"].value_counts().sort_index().items()}
        if n_matched
        else {}
    )
    orientations = (
        {str(k): int(v) for k, v in matched["orientation"].value_counts().items()} if n_matched else {}
    )

    dup_zachht = int(zachht_games["canonical_game_key"].astype(str).replace("", pd.NA).dropna().duplicated().sum()) if n_zachht else 0
    dup_nba = int(nba_games["canonical_game_key"].astype(str).replace("", pd.NA).dropna().duplicated().sum()) if n_nba else 0

    with_scores = int(matched[["home_score", "away_score"]].notna().all(axis=1).sum()) if n_matched else 0
    with_multi_snapshots = int(pd.to_numeric(matched.get("zachht_n_snapshots"), errors="coerce").gt(1).sum()) if n_matched else 0

    settlement_possible = with_scores > 0
    clv_possible = with_multi_snapshots > 0

    return {
        "report": "current_basketball_join_summary",
        "key_version": GAME_KEY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "zachht_nba_games": n_zachht,
            "zachht_non_nba_games_excluded": int(non_nba_zachht_games),
            "nba_api_games": n_nba,
        },
        "joined_games": n_matched,
        "join_rate": {
            "pct_of_zachht_games": round(100.0 * n_matched / max(n_zachht, 1), 2),
            "pct_of_nba_api_games": round(100.0 * n_matched / max(n_nba, 1), 2),
        },
        "unmatched": {
            "zachht_games": int(len(unmatched_zachht)),
            "nba_api_games": int(len(unmatched_nba)),
            "notes": [
                "Unmatched zachht games are typically preseason (nba_api caches cover Regular Season +"
                " Playoffs only) or games after the local nba_api cache cutoff - refresh caches to close the gap.",
                "Unmatched nba_api games are mostly earlier seasons before the zachht snapshot window began.",
            ],
        },
        "date_offset_days_histogram": offsets,
        "orientation_of_matches": orientations,
        "zachht_assumptions_verified": {
            "team2_is_home_holds_pct": round(
                100.0 * orientations.get("as_assumed", 0) / max(n_matched, 1), 2
            ),
            "closing_snapshot_date_exact_pct": round(
                100.0 * offsets.get("0", 0) / max(n_matched, 1), 2
            ),
        },
        "duplicate_canonical_keys": {
            "zachht": dup_zachht,
            "nba_api": dup_nba,
        },
        "grading_feasibility": {
            "settlement_grading_possible": bool(settlement_possible),
            "joined_games_with_final_scores": with_scores,
            "clv_grading_possible": bool(clv_possible),
            "joined_games_with_multiple_snapshots": with_multi_snapshots,
            "notes": [
                "Settlement grading: joined games have nba_api final scores, so game-market outcomes can be settled.",
                "CLV grading: zachht games with >1 snapshot have an opening and closing line to compare entry prices against.",
                "Player-prop settlement would join the same canonical keys to nba_current_player_game_logs_normalized.csv;"
                " prop LINES are still missing, so only actuals-side grading is possible today.",
            ],
        },
        "research_only": True,
        "approved": False,
    }


def build_join_examples(
    matched: pd.DataFrame,
    unmatched_zachht: pd.DataFrame,
    unmatched_nba: pd.DataFrame,
    per_category: int = 25,
) -> pd.DataFrame:
    """Assemble an examples table: matches, mismatch cases, and unmatched rows."""

    blocks: list[pd.DataFrame] = []

    if not matched.empty:
        exact = matched[(matched["date_offset_days"] == 0) & (matched["orientation"] == "as_assumed")]
        blocks.append(_example_block(exact.head(per_category), "joined_exact"))
        date_shift = matched[matched["date_offset_days"] != 0]
        blocks.append(_example_block(date_shift.head(per_category), "joined_date_mismatch"))
        flipped = matched[matched["orientation"] == "reversed"]
        blocks.append(_example_block(flipped.head(per_category), "joined_home_away_flipped"))

    if not unmatched_zachht.empty:
        cols = ["source_game_id", "game_date", "home_team_abbr", "away_team_abbr", "canonical_game_key"]
        block = unmatched_zachht[[c for c in cols if c in unmatched_zachht.columns]].head(per_category).copy()
        block = block.rename(columns={"source_game_id": "zachht_game_ref", "game_date": "zachht_estimated_game_date"})
        block["category"] = "unmatched_zachht"
        blocks.append(block)

    if not unmatched_nba.empty:
        cols = ["game_id", "game_date", "home_team_abbr", "away_team_abbr", "canonical_game_key"]
        block = unmatched_nba[[c for c in cols if c in unmatched_nba.columns]].head(per_category).copy()
        block = block.rename(columns={"game_id": "nba_game_id", "game_date": "nba_game_date"})
        block["category"] = "unmatched_nba_api"
        blocks.append(block)

    blocks = [b for b in blocks if not b.empty]
    if not blocks:
        return pd.DataFrame(columns=["category"])
    examples = pd.concat(blocks, ignore_index=True)
    front = ["category", "canonical_game_key"]
    ordered = front + [c for c in examples.columns if c not in front]
    return examples[ordered]


def _example_block(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    block = frame.copy()
    block["category"] = category
    return block


def build_current_basketball_join_outputs(
    processed_dir: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    """Run the zachht <-> nba_api join and write the summary JSON + examples CSV."""

    processed = Path(processed_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    snapshots_path = processed / "basketball_odds_snapshots_normalized.csv"
    games_path = processed / "nba_current_games_normalized.csv"
    snapshots = pd.read_csv(snapshots_path, low_memory=False) if snapshots_path.exists() else pd.DataFrame()
    nba_games = pd.read_csv(games_path, low_memory=False) if games_path.exists() else pd.DataFrame()

    zachht_all = collapse_zachht_games(snapshots)
    if zachht_all.empty:
        zachht_nba = zachht_all
        non_nba = 0
    else:
        zachht_nba = zachht_all[zachht_all["league"].astype(str).eq("NBA")].reset_index(drop=True)
        non_nba = int(len(zachht_all) - len(zachht_nba))

    matched, unmatched_zachht, unmatched_nba = join_zachht_to_current_games(zachht_nba, nba_games)
    report = build_join_report(matched, unmatched_zachht, unmatched_nba, zachht_nba, nba_games, non_nba)
    examples = build_join_examples(matched, unmatched_zachht, unmatched_nba)

    (reports / "current_basketball_join_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    examples.to_csv(reports / "current_basketball_join_examples.csv", index=False)
    return report
