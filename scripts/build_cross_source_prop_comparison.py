"""Cross-source player-prop comparison (research-only).

Compares prop snapshots across sources where they overlap on the same
(player, prop_type, game, bookmaker, line): price differences, snapshot
freshness, line disagreements per (game, player, prop, book), missing books,
and per-source coverage. When no overlap exists yet, the report still gets
written with an explicit explanation.

Outputs:
    data/reports/cross_source_prop_comparison_summary.json
    data/reports/cross_source_prop_comparison.csv
    data/reports/cross_source_prop_comparison.md
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "player_prop_snapshots_normalized.csv"

# Source-native bookmaker keys -> shared comparison key.
BOOKMAKER_ALIASES = {
    "williamhill_us": "caesars",  # The Odds API key for Caesars
}


def normalize_bookmaker(value: object) -> str:
    token = str(value or "").strip().lower()
    return BOOKMAKER_ALIASES.get(token, token)


def normalize_player(value: object) -> str:
    token = str(value or "").strip().lower()
    for char in (".", "'", "-", ","):
        token = token.replace(char, "")
    return " ".join(token.split())


def load_latest_per_market() -> pd.DataFrame:
    """Latest snapshot per (source, game, player, prop, book, line)."""

    if not SNAPSHOTS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SNAPSHOTS_PATH, low_memory=False)
    if frame.empty:
        return frame
    frame["book_norm"] = frame["bookmaker"].map(normalize_bookmaker)
    frame["player_norm"] = frame["player_name"].map(normalize_player)
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_time"], errors="coerce", utc=True)
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    frame = frame.dropna(subset=["line"])
    frame = frame.sort_values("snapshot_ts")
    keys = ["source", "canonical_game_key", "player_norm", "prop_type", "book_norm", "line"]
    return frame.groupby(keys, as_index=False, dropna=False).last()


def build_comparison(latest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summary: dict = {
        "report": "cross_source_prop_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "approved": False,
    }
    if latest.empty:
        summary["overlap_found"] = False
        summary["reason"] = "no snapshots on disk"
        return pd.DataFrame(), summary

    sources = sorted(latest["source"].astype(str).unique())
    summary["sources_present"] = sources
    summary["coverage_by_source"] = {
        source: {
            "markets": int(len(group)),
            "games": int(group["canonical_game_key"].nunique()),
            "players": int(group["player_norm"].nunique()),
            "prop_types": sorted(group["prop_type"].astype(str).unique().tolist()),
            "bookmakers": sorted(group["book_norm"].astype(str).unique().tolist()),
        }
        for source, group in latest.groupby(latest["source"].astype(str))
    }

    if len(sources) < 2:
        summary["overlap_found"] = False
        summary["reason"] = f"only one source has snapshots ({sources})"
        return pd.DataFrame(), summary

    join_keys = ["canonical_game_key", "player_norm", "prop_type", "book_norm", "line"]
    pair_frames: list[pd.DataFrame] = []
    pair_stats: list[dict] = []
    line_disagreements: list[dict] = []

    for source_a, source_b in itertools.combinations(sources, 2):
        a = latest[latest["source"] == source_a]
        b = latest[latest["source"] == source_b]

        # Shared games first: overlap requires the same canonical game.
        shared_games = sorted(
            set(a["canonical_game_key"].dropna().astype(str))
            & set(b["canonical_game_key"].dropna().astype(str))
        )
        merged = a.merge(
            b, on=join_keys, suffixes=(f"_{source_a}", f"_{source_b}"), how="inner"
        )
        stats = {
            "pair": f"{source_a} vs {source_b}",
            "shared_games": len(shared_games),
            "exact_market_overlap_rows": int(len(merged)),
        }
        if not merged.empty:
            over_a = pd.to_numeric(merged[f"over_price_{source_a}"], errors="coerce")
            over_b = pd.to_numeric(merged[f"over_price_{source_b}"], errors="coerce")
            under_a = pd.to_numeric(merged[f"under_price_{source_a}"], errors="coerce")
            under_b = pd.to_numeric(merged[f"under_price_{source_b}"], errors="coerce")
            comparable_over = over_a.notna() & over_b.notna()
            comparable_under = under_a.notna() & under_b.notna()
            stats["over_price_pairs"] = int(comparable_over.sum())
            stats["mean_abs_over_price_diff"] = (
                float((over_a - over_b).abs()[comparable_over].mean())
                if comparable_over.any() else None
            )
            stats["under_price_pairs"] = int(comparable_under.sum())
            stats["mean_abs_under_price_diff"] = (
                float((under_a - under_b).abs()[comparable_under].mean())
                if comparable_under.any() else None
            )
            ts_a = pd.to_datetime(merged[f"snapshot_ts_{source_a}"], utc=True, errors="coerce")
            ts_b = pd.to_datetime(merged[f"snapshot_ts_{source_b}"], utc=True, errors="coerce")
            fresher = (ts_a > ts_b).sum()
            stats["fresher_counts"] = {
                source_a: int(fresher),
                source_b: int((ts_b > ts_a).sum()),
            }
            export = merged[
                join_keys
                + [f"over_price_{source_a}", f"over_price_{source_b}",
                   f"under_price_{source_a}", f"under_price_{source_b}",
                   f"snapshot_time_{source_a}", f"snapshot_time_{source_b}"]
            ].copy()
            export.insert(0, "pair", stats["pair"])
            export["over_price_diff"] = (over_a - over_b).round(4)
            export["under_price_diff"] = (under_a - under_b).round(4)
            pair_frames.append(export)

        # Line disagreements: same (game, player, prop, book), different line.
        no_line_keys = ["canonical_game_key", "player_norm", "prop_type", "book_norm"]
        merged_lines = a.merge(b, on=no_line_keys, suffixes=("_a", "_b"), how="inner")
        if not merged_lines.empty:
            diff = merged_lines[merged_lines["line_a"] != merged_lines["line_b"]]
            stats["line_disagreement_markets"] = int(
                diff.groupby(no_line_keys).ngroups
            )
            for _, row in diff.head(50).iterrows():
                line_disagreements.append(
                    {
                        "pair": stats["pair"],
                        "game": row["canonical_game_key"],
                        "player": row["player_norm"],
                        "prop_type": row["prop_type"],
                        "bookmaker": row["book_norm"],
                        f"line_{source_a}": float(row["line_a"]),
                        f"line_{source_b}": float(row["line_b"]),
                    }
                )
        else:
            stats["line_disagreement_markets"] = 0

        # Book coverage differences within shared games.
        if shared_games:
            books_a = set(a[a["canonical_game_key"].isin(shared_games)]["book_norm"])
            books_b = set(b[b["canonical_game_key"].isin(shared_games)]["book_norm"])
            stats["books_only_in_" + source_a] = sorted(books_a - books_b)
            stats["books_only_in_" + source_b] = sorted(books_b - books_a)
        pair_stats.append(stats)

    comparison = (
        pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    )
    summary["pairs"] = pair_stats
    summary["line_disagreements_sample"] = line_disagreements
    overlap = any(p.get("exact_market_overlap_rows", 0) > 0 for p in pair_stats)
    shared_any = any(p.get("shared_games", 0) > 0 for p in pair_stats)
    summary["overlap_found"] = bool(overlap)
    if not overlap:
        if shared_any:
            summary["reason"] = (
                "Sources share games but no exact (player, prop, book, line) market yet — "
                "check bookmaker alias coverage and whether both sources collected near the "
                "same time."
            )
        else:
            summary["reason"] = (
                "No shared canonical games between sources yet: the sources collected "
                "different game windows (e.g. SportsGameOdds started with the next Finals "
                "game while The Odds API horizon had not reached it). Overlap will appear "
                "once both sources collect the same upcoming game day."
            )
    return comparison, summary


def render_md(summary: dict, comparison: pd.DataFrame) -> str:
    lines = ["# Cross-Source Prop Comparison", ""]
    lines.append(f"_Generated {summary['generated_at_utc']}. Research-only._")
    lines.append("")
    lines.append(f"- Overlap found: **{summary.get('overlap_found')}**")
    if summary.get("reason"):
        lines.append(f"- Why: {summary['reason']}")
    lines.append("")
    coverage = summary.get("coverage_by_source") or {}
    if coverage:
        lines.append("## Coverage by source (latest snapshot per market)")
        lines.append("")
        lines.append("| source | markets | games | players | bookmakers |")
        lines.append("| --- | --- | --- | --- | --- |")
        for source, info in coverage.items():
            lines.append(
                f"| {source} | {info['markets']} | {info['games']} | {info['players']} "
                f"| {', '.join(info['bookmakers'])} |"
            )
        lines.append("")
    for pair in summary.get("pairs") or []:
        lines.append(f"## {pair['pair']}")
        lines.append("")
        for key, value in pair.items():
            if key == "pair":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    if not comparison.empty:
        lines.append(f"Exact-market comparison rows: {len(comparison)} "
                     "(see cross_source_prop_comparison.csv).")
        lines.append("")
    lines.append("_Research-only. Approved bets/parlays remain blocked._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    latest = load_latest_per_market()
    comparison, summary = build_comparison(latest)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cross_source_prop_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    comparison.to_csv(REPORTS_DIR / "cross_source_prop_comparison.csv", index=False)
    (REPORTS_DIR / "cross_source_prop_comparison.md").write_text(
        render_md(summary, comparison), encoding="utf-8"
    )
    print(f"Wrote cross_source_prop_comparison reports "
          f"(overlap_found={summary.get('overlap_found')}, rows={len(comparison)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
