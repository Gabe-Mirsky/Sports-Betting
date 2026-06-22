"""Join-readiness and player-prop-path reporting for the basketball datasets.

Reads the normalized ehallmar + zachht outputs and reports whether they can be
joined (by id, or by date+teams), what is missing, and what the player-prop data
path looks like. Reporting only - no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs) if path.exists() else pd.DataFrame()


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {"min": None, "max": None}
    dates = pd.to_datetime(frame[column], errors="coerce")
    return {"min": _iso(dates.min()), "max": _iso(dates.max())}


def _iso(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _zachht_game_keys(snapshots: pd.DataFrame) -> set[str]:
    """Collapse zachht snapshots to one game-level date|team|team key set (order-independent)."""

    if snapshots.empty:
        return set()
    frame = snapshots.dropna(subset=["snapshot_time"]).copy()
    frame["date"] = pd.to_datetime(frame["snapshot_time"], errors="coerce").dt.date.astype(str)
    keys = set()
    for date, t1, t2 in zip(frame["date"], frame["team1_abbr"].astype(str), frame["team2_abbr"].astype(str)):
        pair = "|".join(sorted([t1, t2]))
        keys.add(f"{date}|{pair}")
    return keys


def _ehallmar_game_keys(games: pd.DataFrame) -> set[str]:
    if games.empty:
        return set()
    frame = games.copy()
    frame["date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.date.astype(str)
    return {
        f"{d}|" + "|".join(sorted([str(h), str(a)]))
        for d, h, a in zip(frame["date"], frame["home_team_abbr"], frame["away_team_abbr"])
    }


def build_join_readiness(processed_dir: str | Path) -> dict[str, Any]:
    """Compute the basketball join-readiness report dictionary."""

    processed = Path(processed_dir)
    games = _read_csv(processed / "nba_games_normalized.csv")
    odds = _read_csv(processed / "nba_game_odds_normalized.csv", usecols=["game_id", "market_type", "book_name"])
    logs = _read_csv(
        processed / "nba_player_game_logs_normalized.csv",
        usecols=["player_id", "player_name", "game_id", "game_date", "team_abbr"],
    )
    snapshots = _read_csv(processed / "basketball_odds_snapshots_normalized.csv")

    nba_snapshots = snapshots[snapshots.get("league").eq("NBA")] if not snapshots.empty else snapshots

    game_ids = set(games["game_id"]) if not games.empty else set()
    log_game_ids = set(logs["game_id"].dropna()) if not logs.empty else set()
    odds_game_ids = set(odds["game_id"].dropna()) if not odds.empty else set()
    zachht_refs = set(pd.to_numeric(snapshots.get("game_ref"), errors="coerce").dropna().astype("int64")) if not snapshots.empty else set()

    ehallmar_keys = _ehallmar_game_keys(games)
    zachht_keys = _zachht_game_keys(nba_snapshots)

    logs_in_games = len(log_game_ids & game_ids)
    games_with_odds = len(game_ids & odds_game_ids)

    report = {
        "report": "basketball_data_join_readiness",
        "research_only": True,
        "approved": False,
        "sources": {
            "ehallmar": {
                "role": "player_actuals_source",
                "nba_games": int(len(game_ids)),
                "player_game_logs": int(len(logs)),
                "unique_log_players": int(logs["player_id"].nunique()) if not logs.empty else 0,
                "game_odds_rows": int(len(odds)),
                "games_date_range": _date_range(games, "game_date"),
                "player_logs_date_range": _date_range(logs, "game_date"),
            },
            "zachht": {
                "role": "game_odds_market_source",
                "snapshot_rows": int(len(snapshots)),
                "nba_snapshot_rows": int(len(nba_snapshots)),
                "unique_games": int(snapshots.groupby(["league", "game_ref"]).ngroups) if not snapshots.empty else 0,
                "snapshot_date_range": _date_range(snapshots, "snapshot_time"),
            },
        },
        "id_joins": {
            "within_ehallmar_player_logs_to_games": {
                "works": bool(logs_in_games > 0),
                "matched_game_ids": int(logs_in_games),
                "log_game_ids": int(len(log_game_ids)),
                "coverage_pct": round(100.0 * logs_in_games / max(len(log_game_ids), 1), 2),
            },
            "within_ehallmar_games_to_odds": {
                "works": bool(games_with_odds > 0),
                "games_with_odds": int(games_with_odds),
                "coverage_pct": round(100.0 * games_with_odds / max(len(game_ids), 1), 2),
            },
            "cross_source_ehallmar_gameid_vs_zachht_gameref": {
                "works": bool(len(game_ids & zachht_refs) > 0),
                "overlap": int(len(game_ids & zachht_refs)),
                "note": "Different id namespaces (NBA-stats game_id vs Pinnacle game_link); do not join by id.",
            },
        },
        "date_team_join": {
            "ehallmar_game_keys": int(len(ehallmar_keys)),
            "zachht_nba_game_keys": int(len(zachht_keys)),
            "overlapping_game_keys": int(len(ehallmar_keys & zachht_keys)),
            "note": (
                "Cross-source joins must use game_date+teams. Overlap is limited because ehallmar "
                "coverage ends years before the zachht snapshot window."
            ),
        },
        "missing": {
            "player_logs_missing_player_name": int(logs["player_name"].isna().sum()) if not logs.empty else 0,
            "player_logs_missing_team": int((logs["team_abbr"].isna() | logs["team_abbr"].astype(str).eq("")).sum()) if not logs.empty else 0,
            "games_missing_final_score": int(games[["home_score", "away_score"]].isna().any(axis=1).sum()) if not games.empty else 0,
            "games_missing_team_abbr": int((games["home_team_abbr"].astype(str).eq("") | games["away_team_abbr"].astype(str).eq("")).sum()) if not games.empty else 0,
            "games_without_any_odds": int(len(game_ids - odds_game_ids)),
            "zachht_snapshots_missing_team_abbr": int((nba_snapshots["team1_abbr"].astype(str).eq("") | nba_snapshots["team2_abbr"].astype(str).eq("")).sum()) if not nba_snapshots.empty else 0,
            "zachht_snapshots_missing_timestamp": int(snapshots["snapshot_time"].isna().sum()) if not snapshots.empty else 0,
        },
        "duplicates": {
            "ehallmar_duplicate_game_ids": int(games["game_id"].duplicated().sum()) if not games.empty else 0,
            "ehallmar_duplicate_player_game_rows": int(logs.duplicated(["player_id", "game_id"]).sum()) if not logs.empty else 0,
            "zachht_duplicate_snapshot_rows": int(snapshots.duplicated().sum()) if not snapshots.empty else 0,
        },
        "player_props_available": False,
    }
    return report


def render_join_readiness_md(report: dict[str, Any]) -> str:
    src = report["sources"]
    idj = report["id_joins"]
    dtj = report["date_team_join"]
    miss = report["missing"]
    dup = report["duplicates"]
    lines: list[str] = ["# Basketball Data Join Readiness", ""]
    lines.append("_Research-only. Checks whether the ehallmar (actuals) and zachht (odds) sources can be joined._")
    lines.append("")
    lines.append("## Row counts")
    lines.append("")
    lines.append("| metric | ehallmar | zachht |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| NBA games | {src['ehallmar']['nba_games']:,} | {src['zachht']['unique_games']:,} (all leagues) |")
    lines.append(f"| Player game logs | {src['ehallmar']['player_game_logs']:,} | n/a |")
    lines.append(f"| Game odds rows | {src['ehallmar']['game_odds_rows']:,} | {src['zachht']['snapshot_rows']:,} snapshots |")
    lines.append(f"| Date range | {src['ehallmar']['games_date_range']['min']} -> {src['ehallmar']['games_date_range']['max']} | {src['zachht']['snapshot_date_range']['min']} -> {src['zachht']['snapshot_date_range']['max']} |")
    lines.append("")
    lines.append("## Joins")
    lines.append("")
    a = idj["within_ehallmar_player_logs_to_games"]
    b = idj["within_ehallmar_games_to_odds"]
    c = idj["cross_source_ehallmar_gameid_vs_zachht_gameref"]
    lines.append(f"- **ehallmar player logs -> games (game_id):** {'works' if a['works'] else 'fails'} - {a['matched_game_ids']:,} matched ({a['coverage_pct']}% of log game_ids).")
    lines.append(f"- **ehallmar games -> game odds (game_id):** {'works' if b['works'] else 'fails'} - {b['games_with_odds']:,} games have odds ({b['coverage_pct']}%).")
    lines.append(f"- **Cross-source game_id vs game_ref:** {'works' if c['works'] else 'does NOT work'} - {c['note']}")
    lines.append(f"- **Cross-source by date+teams:** {dtj['overlapping_game_keys']:,} overlapping game keys "
                 f"({dtj['ehallmar_game_keys']:,} ehallmar vs {dtj['zachht_nba_game_keys']:,} zachht NBA). {dtj['note']}")
    lines.append("")
    lines.append("## Missing data")
    lines.append("")
    for key, value in miss.items():
        lines.append(f"- {key.replace('_', ' ')}: {value:,}")
    lines.append("")
    lines.append("## Duplicates")
    lines.append("")
    for key, value in dup.items():
        lines.append(f"- {key.replace('_', ' ')}: {value:,}")
    lines.append("")
    lines.append(f"**Player props available in these sources:** {report['player_props_available']}")
    lines.append("")
    return "\n".join(lines)


def build_join_readiness_reports(processed_dir: str | Path, reports_dir: str | Path) -> dict[str, Any]:
    """Write the join-readiness JSON + markdown and return the report dict."""

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    report = build_join_readiness(processed_dir)
    (reports / "basketball_data_join_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "basketball_data_join_readiness.md").write_text(render_join_readiness_md(report), encoding="utf-8")
    return report


def render_player_prop_path_md(readiness: dict[str, Any]) -> str:
    """Render the plain-English player-prop data-path report."""

    ehallmar = readiness["sources"]["ehallmar"]
    lines: list[str] = ["# Player Prop Data Path", ""]
    lines.append("_Research-only. Plain-English status of player-prop data for this project._")
    lines.append("")

    lines.append("## What we now have")
    lines.append("")
    lines.append(
        f"- **Player actuals to SETTLE props:** {ehallmar['player_game_logs']:,} NBA player game logs "
        "(points, rebounds, assists, threes, blocks, steals, turnovers, minutes) from ehallmar, "
        "joinable to games by `game_id`. These let us grade any prop *outcome* historically."
    )
    lines.append(
        f"- **NBA/WNBA game-level odds with timestamps:** {readiness['sources']['zachht']['snapshot_rows']:,} "
        "snapshots (moneyline/spread/total) from zachht, with opening/closing detection for closing-line value."
    )
    lines.append("- **Game results and team game odds:** from ehallmar (moneyline/spread/total per book).")
    lines.append("")

    lines.append("## What we still do NOT have")
    lines.append("")
    lines.append("- **Historical player-prop betting lines** (e.g. \"LeBron James points over/under 25.5 at -110\").")
    lines.append("- Any prop type/line/over-under price tied to a player and a game, at a timestamp.")
    lines.append("- Therefore we cannot yet compute prop edges or prop closing-line value from historical data.")
    lines.append("")

    lines.append("## Why historical prop lines are still missing")
    lines.append("")
    lines.append("- All 9 profiled Kaggle datasets are game-level markets; none archive player-prop lines.")
    lines.append("- Player-prop pricing is high-volume and book-specific, and is rarely published as free history.")
    lines.append("- The sources we have give us *actuals* (to settle props) and *game* odds (to model team markets), but not the prop market itself.")
    lines.append("")

    lines.append("## How to collect player-prop odds going forward")
    lines.append("")
    lines.append("1. **Kalshi player-prop tickers** (KXNBAPTS / KXNBAREB / KXNBAAST ...): the existing Kalshi client already")
    lines.append("   reads candlesticks; extend it to player-prop markets to capture priced Yes/No prop snapshots over time.")
    lines.append("2. **The Odds API player markets** (`player_points`, `player_rebounds`, ...): the planned odds_api adapter can")
    lines.append("   pull current multi-book Over/Under prop lines. Free tier is quota-limited, so snapshot selectively.")
    lines.append("3. **Snapshot near tip-off** to approximate a closing line (neither source publishes an explicit close).")
    lines.append("4. Store every pull as an immutable timestamped snapshot so closing-line value can be computed later.")
    lines.append("")

    lines.append("## Recommended schema for future player-prop snapshots")
    lines.append("")
    lines.append("One row per (player, game, stat, book, side, snapshot):")
    lines.append("")
    lines.append("| field | meaning |")
    lines.append("| --- | --- |")
    for field, meaning in _PROP_SCHEMA:
        lines.append(f"| {field} | {meaning} |")
    lines.append("")
    lines.append("Grade later by joining `player_id`+`game_id` to the ehallmar player actuals: "
                 "`actual_value` vs `line` settles over/under, and the latest pre-tip snapshot gives the closing line.")
    lines.append("")
    lines.append("_No prop betting is enabled. This is a data-collection plan only._")
    lines.append("")
    return "\n".join(lines)


_PROP_SCHEMA: tuple[tuple[str, str], ...] = (
    ("snapshot_id", "Immutable id for the pull (timestamped, never overwritten)."),
    ("snapshot_time", "UTC time the line was observed."),
    ("source", "kalshi | odds_api | ..."),
    ("book", "Sportsbook / exchange."),
    ("league", "NBA / WNBA."),
    ("game_id", "Join key to ehallmar games + player actuals."),
    ("game_date", "Game date for date+team fallback joins."),
    ("player_id", "Stable player id (join to actuals)."),
    ("player_name", "Player name (mapping fallback)."),
    ("team_abbr", "Player's team."),
    ("stat_type", "points | rebounds | assists | threes | pra | ..."),
    ("line", "The over/under line value."),
    ("side", "over | under | yes | no."),
    ("price", "Price for the side."),
    ("price_format", "american | decimal | cents_0_100."),
    ("implied_prob", "Implied probability from price."),
    ("is_closing", "Whether this is the last pre-tip snapshot."),
    ("market_type", "player_prop (vs game_market)."),
)


def build_player_prop_path_report(readiness: dict[str, Any], reports_dir: str | Path) -> str:
    """Write the player-prop-path markdown and return its text."""

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    text = render_player_prop_path_md(readiness)
    (reports / "player_prop_data_path.md").write_text(text, encoding="utf-8")
    return text
