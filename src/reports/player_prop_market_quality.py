"""Player-prop market quality audit (research-only data inspection).

Audits the collected prop snapshots per league/player/prop_type/bookmaker/game
market: line stability, possible alternate lines mixed into the feed, price
validity, duplicate snapshots, bookmaker coverage, and closing-snapshot
coverage for future CLV work. Also derives a research-only "likely main line"
per market without deleting any alternate-line rows.

Inspection only: no models, no recommendations, no proof-gate or betting
changes. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MISSING = "(missing)"
GROUP_KEYS = ["league", "player_name", "prop_type", "bookmaker", "canonical_game_key"]

# A market's line range is "unusually wide" when it exceeds both an absolute
# floor and a fraction of the most common line (so a 3-point drift on a 25.5
# points line is normal, but a 3-line spread on a 1.5 home-run line is not).
WIDE_RANGE_MIN_ABS = 2.0
WIDE_RANGE_FRACTION = 0.25
# Several distinct lines for one market usually means alternate lines mixed in.
ALT_UNIQUE_LINE_THRESHOLD = 3
# Beyond this many distinct lines the "most frequent" rule is unreliable.
UNCERTAIN_UNIQUE_LINES = 5
LOW_SNAPSHOT_COUNT = 2
# Decimal odds outside this band are flagged for review (not removed).
SUSPICIOUS_PRICE_MIN = 1.01
SUSPICIOUS_PRICE_MAX = 100.0
SUSPICIOUS_LINE_MAX = 150.0
# Heuristic CLV-readiness floor: at least this share of a league's markets
# need a closing-like snapshot before closing-line comparisons are meaningful.
CLV_MIN_CLOSING_MARKET_RATE = 0.25
CLV_MIN_CLOSING_MARKETS = 25

FLAG_COLUMNS = [
    "possible_alt_lines",
    "wide_line_range",
    "missing_prices",
    "duplicate_exact_snapshots",
    "suspicious_line_values",
    "suspicious_price_values",
    "missing_bookmaker",
    "missing_player",
    "missing_game_key",
    "low_snapshot_count",
]

OUTPUT_FILES = {
    "summary_json": "player_prop_market_quality_summary.json",
    "summary_md": "player_prop_market_quality.md",
    "line_quality": "player_prop_line_quality.csv",
    "likely_main_lines": "player_prop_likely_main_lines.csv",
    "possible_alt_lines": "player_prop_possible_alt_lines.csv",
    "bookmaker_coverage": "player_prop_bookmaker_coverage.csv",
    "closing_coverage": "player_prop_closing_snapshot_coverage.csv",
}

LINE_QUALITY_COLUMNS = GROUP_KEYS + [
    "game_date",
    "snapshots",
    "unique_lines",
    "min_line",
    "max_line",
    "line_range",
    "most_common_line",
    "most_common_line_count",
    "latest_line",
    "closing_line",
    "has_closing_snapshot",
    "closing_snapshots",
    "max_lines_same_snapshot_time",
    "has_over_price",
    "has_under_price",
    "both_prices_present",
    "rows_missing_both_prices",
    "duplicate_snapshot_rows",
    "suspicious_line_count",
    "suspicious_price_count",
    "likely_main_line",
    "likely_alt_lines",
    "main_line_reason",
    "line_quality_label",
] + FLAG_COLUMNS + ["flags"]


def _clean_key(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.mask(series.isna() | text.isin({"", "nan", "None"}), MISSING)


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _prepare(snaps: pd.DataFrame) -> pd.DataFrame:
    frame = snaps.copy()
    for key in GROUP_KEYS + ["game_date"]:
        frame[key] = _clean_key(frame[key]) if key in frame.columns else MISSING
    frame["line"] = pd.to_numeric(frame.get("line"), errors="coerce")
    frame["over_price"] = pd.to_numeric(frame.get("over_price"), errors="coerce")
    frame["under_price"] = pd.to_numeric(frame.get("under_price"), errors="coerce")
    frame["_ts"] = pd.to_datetime(frame.get("snapshot_time"), errors="coerce", utc=True)
    closing = frame.get("is_closing_snapshot")
    frame["_closing"] = _truthy_mask(closing) if closing is not None else False
    return frame


def _suspicious_line_mask(line: pd.Series) -> pd.Series:
    non_half_step = ((line * 2) % 1).fillna(0) != 0
    return line.notna() & ((line <= 0) | (line > SUSPICIOUS_LINE_MAX) | non_half_step)


def _suspicious_price_mask(price: pd.Series) -> pd.Series:
    return price.notna() & ((price < SUSPICIOUS_PRICE_MIN) | (price > SUSPICIOUS_PRICE_MAX))


def _most_common_line(group: pd.DataFrame) -> tuple[float, int]:
    """Most frequent line; ties broken by the line seen most recently."""

    counts = group["line"].value_counts()
    top_count = int(counts.iloc[0])
    tied = set(counts[counts == top_count].index)
    if len(tied) == 1:
        return float(counts.index[0]), top_count
    tied_rows = group[group["line"].isin(tied)].sort_values("_ts")
    return float(tied_rows["line"].iloc[-1]), top_count


def _balanced_line(rows: pd.DataFrame) -> float | None:
    """Line whose over/under prices are closest to even (the main-line signature).

    In an alternate-line ladder the main line is the one priced near even on
    both sides; alternates are increasingly one-sided. Returns None when no
    row has both prices.
    """

    priced = rows[rows["line"].notna() & rows["over_price"].notna() & rows["under_price"].notna()]
    if priced.empty:
        return None
    balance = (priced["over_price"] - priced["under_price"]).abs()
    return float(priced.loc[balance.idxmin(), "line"])


def _audit_market(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("_ts")
    lines = group["line"].dropna()
    snapshots = int(len(group))
    unique_lines = int(lines.nunique())
    min_line = float(lines.min()) if not lines.empty else None
    max_line = float(lines.max()) if not lines.empty else None
    line_range = (max_line - min_line) if unique_lines else 0.0

    if not lines.empty:
        most_common_line, most_common_count = _most_common_line(group[group["line"].notna()])
        latest_line = float(group.loc[group["line"].notna(), "line"].iloc[-1])
    else:
        most_common_line, most_common_count, latest_line = None, 0, None

    closing_rows = group[group["_closing"] & group["line"].notna()]
    closing_snapshots = int(group["_closing"].sum())
    closing_line = float(closing_rows["line"].iloc[-1]) if not closing_rows.empty else None
    has_closing = closing_snapshots > 0

    max_lines_same_time = 0
    if not lines.empty:
        per_time = group[group["line"].notna()].groupby("snapshot_time")["line"].nunique()
        max_lines_same_time = int(per_time.max()) if not per_time.empty else 0

    has_over = bool(group["over_price"].notna().any())
    has_under = bool(group["under_price"].notna().any())
    rows_missing_both = int((group["over_price"].isna() & group["under_price"].isna()).sum())
    duplicate_rows = int(
        group.duplicated(subset=["snapshot_time", "line", "over_price", "under_price"]).sum()
    )
    suspicious_line_count = int(_suspicious_line_mask(group["line"]).sum())
    suspicious_price_count = int(
        (_suspicious_price_mask(group["over_price"]) | _suspicious_price_mask(group["under_price"])).sum()
    )

    wide_threshold = max(WIDE_RANGE_MIN_ABS, WIDE_RANGE_FRACTION * abs(most_common_line or 0.0))
    wide_line_range = bool(unique_lines and line_range > wide_threshold)
    possible_alt_lines = bool(
        max_lines_same_time > 1 or unique_lines >= ALT_UNIQUE_LINE_THRESHOLD
    )

    # Main-line selection (research-only; alternate lines are kept, not deleted).
    # Closing snapshots win; alt-line ladders are resolved by price balance
    # (the main line is priced near even, alternates are one-sided); otherwise
    # fall back to the most frequent line.
    main_resolved = True
    if has_closing and closing_line is not None:
        latest_closing = closing_rows[closing_rows["snapshot_time"] == closing_rows["snapshot_time"].iloc[-1]]
        balanced_closing = _balanced_line(latest_closing) if len(latest_closing) > 1 else None
        likely_main_line = balanced_closing if balanced_closing is not None else closing_line
        main_line_reason = "closing_snapshot"
    elif unique_lines == 1:
        likely_main_line, main_line_reason = most_common_line, "single_line"
    elif unique_lines > 1:
        balanced = None
        if max_lines_same_time > 1:
            lined = group[group["line"].notna()]
            latest_batch = lined[lined["snapshot_time"] == lined["snapshot_time"].iloc[-1]]
            balanced = _balanced_line(latest_batch)
        if balanced is not None:
            likely_main_line, main_line_reason = balanced, "balanced_prices"
        else:
            likely_main_line, main_line_reason = most_common_line, "most_frequent"
            main_resolved = False
    else:
        likely_main_line, main_line_reason = None, "no_lines"
        main_resolved = False

    too_many_lines = unique_lines > UNCERTAIN_UNIQUE_LINES
    if (too_many_lines or wide_line_range) and not main_resolved:
        line_quality_label = "uncertain"
    elif possible_alt_lines:
        line_quality_label = "main_plus_alt_lines"
    elif unique_lines > 1:
        line_quality_label = "line_movement"
    else:
        line_quality_label = "clean"

    alt_lines = sorted(
        value for value in lines.unique() if likely_main_line is None or value != likely_main_line
    ) if unique_lines > 1 else []

    record: dict[str, Any] = {
        "game_date": group["game_date"].iloc[0],
        "snapshots": snapshots,
        "unique_lines": unique_lines,
        "min_line": min_line,
        "max_line": max_line,
        "line_range": round(line_range, 4),
        "most_common_line": most_common_line,
        "most_common_line_count": most_common_count,
        "latest_line": latest_line,
        "closing_line": closing_line,
        "has_closing_snapshot": has_closing,
        "closing_snapshots": closing_snapshots,
        "max_lines_same_snapshot_time": max_lines_same_time,
        "has_over_price": has_over,
        "has_under_price": has_under,
        "both_prices_present": has_over and has_under,
        "rows_missing_both_prices": rows_missing_both,
        "duplicate_snapshot_rows": duplicate_rows,
        "suspicious_line_count": suspicious_line_count,
        "suspicious_price_count": suspicious_price_count,
        "likely_main_line": likely_main_line,
        "likely_alt_lines": "|".join(f"{value:g}" for value in alt_lines),
        "main_line_reason": main_line_reason,
        "line_quality_label": line_quality_label,
        "possible_alt_lines": possible_alt_lines,
        "wide_line_range": wide_line_range,
        "missing_prices": rows_missing_both > 0 or not (has_over or has_under),
        "duplicate_exact_snapshots": duplicate_rows > 0,
        "suspicious_line_values": suspicious_line_count > 0,
        "suspicious_price_values": suspicious_price_count > 0,
        "missing_bookmaker": group["bookmaker"].iloc[0] == MISSING,
        "missing_player": group["player_name"].iloc[0] == MISSING,
        "missing_game_key": group["canonical_game_key"].iloc[0] == MISSING,
        "low_snapshot_count": snapshots < LOW_SNAPSHOT_COUNT,
    }
    record["flags"] = ";".join(flag for flag in FLAG_COLUMNS if record[flag])
    return record


def build_line_quality(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per-market line quality audit: one row per league/player/prop/book/game."""

    if snaps.empty:
        return pd.DataFrame(columns=LINE_QUALITY_COLUMNS)
    frame = _prepare(snaps)
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(GROUP_KEYS, dropna=False, sort=True):
        record = dict(zip(GROUP_KEYS, keys))
        record.update(_audit_market(group))
        records.append(record)
    return pd.DataFrame(records)[LINE_QUALITY_COLUMNS]


def build_possible_alt_lines(snaps: pd.DataFrame, line_quality: pd.DataFrame) -> pd.DataFrame:
    """One row per non-main line in markets flagged possible_alt_lines."""

    columns = GROUP_KEYS + [
        "game_date", "likely_alt_line", "snapshots_at_line", "likely_main_line", "line_quality_label",
    ]
    if snaps.empty or line_quality.empty:
        return pd.DataFrame(columns=columns)
    flagged = line_quality[line_quality["possible_alt_lines"]]
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    frame = _prepare(snaps)
    line_counts = (
        frame[frame["line"].notna()]
        .groupby(GROUP_KEYS + ["line"], dropna=False)
        .size()
        .reset_index(name="snapshots_at_line")
    )
    merged = line_counts.merge(
        flagged[GROUP_KEYS + ["game_date", "likely_main_line", "line_quality_label"]],
        on=GROUP_KEYS,
        how="inner",
    )
    alt_rows = merged[merged["line"] != merged["likely_main_line"]].rename(
        columns={"line": "likely_alt_line"}
    )
    return alt_rows.sort_values(GROUP_KEYS).reset_index(drop=True)[columns]


def build_bookmaker_coverage(snaps: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per league+bookmaker coverage table plus overlap / missing-book summary."""

    columns = [
        "league", "bookmaker", "snapshots", "markets", "players", "prop_types",
        "closing_snapshots", "league_market_share",
    ]
    if snaps.empty:
        return pd.DataFrame(columns=columns), {
            "bookmaker_overlap_by_league": {},
            "missing_books_by_league": {},
            "nba_best_bookmakers": [],
        }

    frame = _prepare(snaps)
    frame["_market"] = (
        frame["canonical_game_key"] + "|" + frame["player_name"] + "|" + frame["prop_type"]
    )

    coverage = (
        frame.groupby(["league", "bookmaker"], dropna=False)
        .agg(
            snapshots=("bookmaker", "size"),
            markets=("_market", "nunique"),
            players=("player_name", "nunique"),
            prop_types=("prop_type", "nunique"),
            closing_snapshots=("_closing", "sum"),
        )
        .reset_index()
    )
    league_markets = frame.groupby("league")["_market"].nunique()
    coverage["league_market_share"] = coverage.apply(
        lambda row: round(row["markets"] / league_markets[row["league"]], 4)
        if league_markets.get(row["league"], 0)
        else 0.0,
        axis=1,
    )
    coverage["closing_snapshots"] = coverage["closing_snapshots"].astype(int)
    coverage = coverage.sort_values(["league", "markets"], ascending=[True, False]).reset_index(drop=True)

    # How many books quote the same player/prop/game market, per league.
    books_per_market = frame.groupby(["league", "_market"])["bookmaker"].nunique()
    overlap_by_league: dict[str, dict[str, int]] = {}
    for league, counts in books_per_market.groupby(level="league"):
        distribution = counts.value_counts().sort_index()
        overlap_by_league[str(league)] = {
            f"markets_with_{int(books)}_books": int(markets) for books, markets in distribution.items()
        }

    all_books = set(coverage["bookmaker"]) - {MISSING}
    missing_books_by_league = {
        str(league): sorted(all_books - set(group["bookmaker"]))
        for league, group in coverage.groupby("league")
    }

    nba_best = [
        {
            "bookmaker": row["bookmaker"],
            "markets": int(row["markets"]),
            "players": int(row["players"]),
            "prop_types": int(row["prop_types"]),
            "closing_snapshots": int(row["closing_snapshots"]),
            "league_market_share": float(row["league_market_share"]),
        }
        for _, row in coverage[coverage["league"] == "NBA"].iterrows()
    ]

    summary = {
        "bookmaker_overlap_by_league": overlap_by_league,
        "missing_books_by_league": missing_books_by_league,
        "nba_best_bookmakers": nba_best,
    }
    return coverage[columns], summary


def build_closing_coverage(
    snaps: pd.DataFrame, line_quality: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Closing-like snapshot coverage by league/prop/bookmaker plus CLV readiness."""

    columns = [
        "league", "prop_type", "bookmaker", "snapshots", "closing_snapshots",
        "markets", "markets_with_closing", "closing_market_rate",
    ]
    empty_summary = {
        "total_closing_snapshots": 0,
        "closing_by_league": {},
        "closing_by_prop_type": {},
        "closing_by_bookmaker": {},
        "markets_without_closing": 0,
        "markets_without_closing_by_league": {},
        "closing_market_rate_by_league": {},
        "nba_clv_ready": False,
        "clv_readiness_verdict": "No snapshots collected yet; CLV analysis is not possible.",
    }
    if snaps.empty:
        return pd.DataFrame(columns=columns), empty_summary

    frame = _prepare(snaps)
    grouped = (
        frame.groupby(["league", "prop_type", "bookmaker"], dropna=False)
        .agg(snapshots=("_closing", "size"), closing_snapshots=("_closing", "sum"))
        .reset_index()
    )
    grouped["closing_snapshots"] = grouped["closing_snapshots"].astype(int)

    market_stats = (
        line_quality.groupby(["league", "prop_type", "bookmaker"], dropna=False)
        .agg(markets=("has_closing_snapshot", "size"), markets_with_closing=("has_closing_snapshot", "sum"))
        .reset_index()
    )
    market_stats["markets_with_closing"] = market_stats["markets_with_closing"].astype(int)
    coverage = grouped.merge(market_stats, on=["league", "prop_type", "bookmaker"], how="left")
    coverage[["markets", "markets_with_closing"]] = (
        coverage[["markets", "markets_with_closing"]].fillna(0).astype(int)
    )
    coverage["closing_market_rate"] = (
        coverage["markets_with_closing"] / coverage["markets"].replace(0, pd.NA)
    ).fillna(0.0).round(4)
    coverage = coverage.sort_values(["league", "prop_type", "bookmaker"]).reset_index(drop=True)

    closing_rows = frame[frame["_closing"]]
    by_league = closing_rows.groupby("league").size().astype(int).to_dict()
    by_prop = closing_rows.groupby("prop_type").size().astype(int).to_dict()
    by_book = closing_rows.groupby("bookmaker").size().astype(int).to_dict()

    without_closing = line_quality[~line_quality["has_closing_snapshot"]]
    without_by_league = without_closing.groupby("league").size().astype(int).to_dict()
    rate_by_league = (
        line_quality.groupby("league")["has_closing_snapshot"].mean().round(4).to_dict()
        if not line_quality.empty
        else {}
    )

    nba_markets_with_closing = int(
        line_quality[(line_quality["league"] == "NBA") & line_quality["has_closing_snapshot"]].shape[0]
    )
    nba_rate = float(rate_by_league.get("NBA", 0.0))
    nba_clv_ready = (
        nba_markets_with_closing >= CLV_MIN_CLOSING_MARKETS and nba_rate >= CLV_MIN_CLOSING_MARKET_RATE
    )
    if nba_clv_ready:
        verdict = (
            f"NBA closing coverage looks workable for future CLV research: "
            f"{nba_markets_with_closing} NBA markets ({nba_rate:.1%}) have a closing-like snapshot."
        )
    else:
        verdict = (
            f"Not enough closing-like snapshots yet for reliable CLV: {nba_markets_with_closing} NBA "
            f"markets ({nba_rate:.1%}) have one; need at least {CLV_MIN_CLOSING_MARKETS} markets and "
            f"{CLV_MIN_CLOSING_MARKET_RATE:.0%} coverage. Keep collecting near tip-off."
        )

    summary = {
        "total_closing_snapshots": int(frame["_closing"].sum()),
        "closing_by_league": {str(k): v for k, v in by_league.items()},
        "closing_by_prop_type": {str(k): v for k, v in by_prop.items()},
        "closing_by_bookmaker": {str(k): v for k, v in by_book.items()},
        "markets_without_closing": int(len(without_closing)),
        "markets_without_closing_by_league": {str(k): v for k, v in without_by_league.items()},
        "closing_market_rate_by_league": {str(k): float(v) for k, v in rate_by_league.items()},
        "nba_clv_ready": nba_clv_ready,
        "clv_readiness_verdict": verdict,
    }
    return coverage[columns], summary


def _nba_modeling_verdict(line_quality: pd.DataFrame) -> tuple[bool, str]:
    nba = line_quality[line_quality["league"] == "NBA"] if not line_quality.empty else line_quality
    if nba.empty:
        return False, "No NBA prop markets collected yet."
    total = len(nba)
    clean_labels = {"clean", "line_movement", "main_plus_alt_lines"}
    usable = int(nba["line_quality_label"].isin(clean_labels).sum())
    uncertain = total - usable
    blocking = int(
        (nba["missing_player"] | nba["missing_game_key"] | nba["suspicious_line_values"]).sum()
    )
    clean_enough = blocking == 0 and (usable / total) >= 0.95
    if clean_enough:
        verdict = (
            f"NBA prop data looks clean enough for future modeling: {usable}/{total} markets have a "
            f"confident main line ({uncertain} uncertain), with no missing players, missing game keys, "
            "or suspicious line values. Alternate-line rows are flagged, not deleted, so modeling can "
            "filter on likely_main_line. Modeling itself remains out of scope (research-only)."
        )
    else:
        verdict = (
            f"NBA prop data is NOT yet clean enough for modeling: {usable}/{total} markets have a "
            f"confident main line, {uncertain} are uncertain, and {blocking} have blocking issues "
            "(missing player, missing game key, or suspicious lines)."
        )
    return clean_enough, verdict


def build_market_quality_summary(
    snaps: pd.DataFrame,
    line_quality: pd.DataFrame,
    bookmaker_summary: dict[str, Any],
    closing_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the audit summary dict from the component frames (pure, testable)."""

    markets = int(len(line_quality))
    flag_counts = {
        flag: int(line_quality[flag].sum()) if markets else 0 for flag in FLAG_COLUMNS
    }
    label_counts = (
        line_quality["line_quality_label"].value_counts().astype(int).to_dict() if markets else {}
    )
    reason_counts = (
        line_quality["main_line_reason"].value_counts().astype(int).to_dict() if markets else {}
    )
    likely_main_lines = int(line_quality["likely_main_line"].notna().sum()) if markets else 0
    confident_main_lines = int(
        (line_quality["likely_main_line"].notna() & (line_quality["line_quality_label"] != "uncertain")).sum()
    ) if markets else 0

    markets_by_league = (
        line_quality.groupby("league").size().astype(int).to_dict() if markets else {}
    )
    nba_clean, nba_verdict = _nba_modeling_verdict(line_quality)

    return {
        "report": "player_prop_market_quality",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "approved": False,
        "total_snapshots": int(len(snaps)),
        "total_markets_audited": markets,
        "markets_by_league": {str(k): v for k, v in markets_by_league.items()},
        "likely_main_lines": likely_main_lines,
        "confident_main_lines": confident_main_lines,
        "possible_alt_line_markets": flag_counts["possible_alt_lines"],
        "wide_line_range_markets": flag_counts["wide_line_range"],
        "missing_price_markets": flag_counts["missing_prices"],
        "flag_counts": flag_counts,
        "line_quality_labels": {str(k): v for k, v in label_counts.items()},
        "main_line_reasons": {str(k): v for k, v in reason_counts.items()},
        "closing_coverage": closing_summary,
        "bookmaker_overlap_by_league": bookmaker_summary["bookmaker_overlap_by_league"],
        "missing_books_by_league": bookmaker_summary["missing_books_by_league"],
        "nba_best_bookmakers": bookmaker_summary["nba_best_bookmakers"],
        "nba_clean_enough_for_modeling": nba_clean,
        "nba_modeling_verdict": nba_verdict,
        "thresholds": {
            "wide_range_min_abs": WIDE_RANGE_MIN_ABS,
            "wide_range_fraction": WIDE_RANGE_FRACTION,
            "alt_unique_line_threshold": ALT_UNIQUE_LINE_THRESHOLD,
            "uncertain_unique_lines": UNCERTAIN_UNIQUE_LINES,
            "low_snapshot_count": LOW_SNAPSHOT_COUNT,
            "suspicious_price_min": SUSPICIOUS_PRICE_MIN,
            "suspicious_price_max": SUSPICIOUS_PRICE_MAX,
            "suspicious_line_max": SUSPICIOUS_LINE_MAX,
            "clv_min_closing_markets": CLV_MIN_CLOSING_MARKETS,
            "clv_min_closing_market_rate": CLV_MIN_CLOSING_MARKET_RATE,
        },
    }


def _fmt_count_lines(counts: dict[str, int]) -> str:
    if not counts:
        return "- (none)"
    return "\n".join(f"- {key}: {value}" for key, value in counts.items())


def _build_markdown(summary: dict[str, Any]) -> str:
    closing = summary["closing_coverage"]
    lines: list[str] = [
        "# Player Prop Market Quality Audit",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "Research-only data-quality audit. No models, recommendations, approved bets, or parlays.",
        "Alternate lines are flagged for review only - nothing is deleted.",
        "",
        "## Totals",
        "",
        f"- Total snapshots audited: {summary['total_snapshots']}",
        f"- Total markets audited (league/player/prop/book/game): {summary['total_markets_audited']}",
        f"- Likely main lines identified: {summary['likely_main_lines']}",
        f"- Confident main lines (label != uncertain): {summary['confident_main_lines']}",
        f"- Possible alternate-line markets: {summary['possible_alt_line_markets']}",
        f"- Wide line range warnings: {summary['wide_line_range_markets']}",
        f"- Missing price warnings: {summary['missing_price_markets']}",
        "",
        "## Markets By League",
        "",
        _fmt_count_lines(summary["markets_by_league"]),
        "",
        "## Flag Counts",
        "",
        _fmt_count_lines(summary["flag_counts"]),
        "",
        "## Line Quality Labels",
        "",
        _fmt_count_lines(summary["line_quality_labels"]),
        "",
        "## Main Line Reasons",
        "",
        _fmt_count_lines(summary["main_line_reasons"]),
        "",
        "## Closing-Like Snapshot Coverage",
        "",
        f"- Total closing-like snapshots: {closing['total_closing_snapshots']}",
        f"- Markets without a closing-like snapshot: {closing['markets_without_closing']}",
        "",
        "By league:",
        "",
        _fmt_count_lines(closing["closing_by_league"]),
        "",
        "By prop type:",
        "",
        _fmt_count_lines(closing["closing_by_prop_type"]),
        "",
        "By bookmaker:",
        "",
        _fmt_count_lines(closing["closing_by_bookmaker"]),
        "",
        "Closing market coverage rate by league:",
        "",
        _fmt_count_lines(
            {league: f"{rate:.1%}" for league, rate in closing["closing_market_rate_by_league"].items()}
        ),
        "",
        f"CLV readiness: {closing['clv_readiness_verdict']}",
        "",
        "## Bookmaker Coverage",
        "",
        "Best-covered NBA bookmakers (by distinct player/prop/game markets):",
        "",
    ]
    if summary["nba_best_bookmakers"]:
        lines += ["| bookmaker | markets | players | prop_types | closing_snapshots | league_market_share |"]
        lines += ["| --- | --- | --- | --- | --- | --- |"]
        for book in summary["nba_best_bookmakers"]:
            lines.append(
                f"| {book['bookmaker']} | {book['markets']} | {book['players']} | "
                f"{book['prop_types']} | {book['closing_snapshots']} | {book['league_market_share']:.1%} |"
            )
    else:
        lines.append("- (no NBA bookmaker coverage yet)")
    lines += ["", "Books missing by league (seen in other leagues but not this one):", ""]
    missing_books = summary["missing_books_by_league"]
    if missing_books:
        for league, books in sorted(missing_books.items()):
            lines.append(f"- {league}: {', '.join(books) if books else '(none missing)'}")
    else:
        lines.append("- (no data)")
    lines += ["", "Bookmaker overlap (books quoting the same player/prop/game market):", ""]
    for league, distribution in sorted(summary["bookmaker_overlap_by_league"].items()):
        parts = ", ".join(f"{key}={value}" for key, value in distribution.items())
        lines.append(f"- {league}: {parts}")
    lines += [
        "",
        "## NBA Modeling Readiness",
        "",
        f"Verdict: {summary['nba_modeling_verdict']}",
        "",
        "---",
        "Research-only: this audit inspects collected data quality. It does not build "
        "models, create recommendations, loosen proof gates, or enable approved bets or parlays.",
        "",
    ]
    return "\n".join(lines)


def write_market_quality_reports(
    project_root: str | Path,
    normalized_path: str | Path | None = None,
    enriched_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read snapshot inputs, write all market-quality outputs, return the summary."""

    root = Path(project_root)
    normalized = (
        Path(normalized_path)
        if normalized_path
        else root / "data" / "processed" / "player_prop_snapshots_normalized.csv"
    )
    enriched_file = (
        Path(enriched_path)
        if enriched_path
        else root / "data" / "processed" / "player_prop_snapshots_enriched.csv"
    )
    out_dir = Path(reports_dir) if reports_dir else root / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the enriched file (same rows plus match metadata) when it exists.
    if enriched_file.exists():
        snaps = pd.read_csv(enriched_file, low_memory=False)
        source_file = enriched_file
    elif normalized.exists():
        snaps = pd.read_csv(normalized, low_memory=False)
        source_file = normalized
    else:
        snaps = pd.DataFrame()
        source_file = normalized

    line_quality = build_line_quality(snaps)
    alt_lines = build_possible_alt_lines(snaps, line_quality)
    bookmaker_coverage, bookmaker_summary = build_bookmaker_coverage(snaps)
    closing_coverage, closing_summary = build_closing_coverage(snaps, line_quality)
    summary = build_market_quality_summary(snaps, line_quality, bookmaker_summary, closing_summary)

    main_line_columns = GROUP_KEYS + [
        "game_date", "snapshots", "unique_lines", "likely_main_line", "closing_line",
        "latest_line", "main_line_reason", "line_quality_label",
    ]
    likely_main = (
        line_quality[line_quality["likely_main_line"].notna()][main_line_columns]
        if not line_quality.empty
        else pd.DataFrame(columns=main_line_columns)
    )

    outputs = {key: out_dir / filename for key, filename in OUTPUT_FILES.items()}
    line_quality.to_csv(outputs["line_quality"], index=False)
    likely_main.to_csv(outputs["likely_main_lines"], index=False)
    alt_lines.to_csv(outputs["possible_alt_lines"], index=False)
    bookmaker_coverage.to_csv(outputs["bookmaker_coverage"], index=False)
    closing_coverage.to_csv(outputs["closing_coverage"], index=False)

    summary["inputs"] = {
        "source_file": str(source_file),
        "normalized_path": str(normalized),
        "normalized_exists": normalized.exists(),
        "enriched_path": str(enriched_file),
        "enriched_exists": enriched_file.exists(),
    }
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    outputs["summary_md"].write_text(_build_markdown(summary), encoding="utf-8")
    return summary
