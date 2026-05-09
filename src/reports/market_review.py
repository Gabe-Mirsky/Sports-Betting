"""Market coverage review, gap decisions, and match audit reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.kalshi_backfill import _parse_kxnbagame_ticker, teams_mentioned_in_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, **kwargs)


def _game_count(frame: pd.DataFrame) -> int:
    columns = ["game_date", "home_team_abbr", "away_team_abbr"]
    if not set(columns).issubset(frame.columns):
        return 0
    return int(frame[columns].drop_duplicates().shape[0])


def build_gap_decision_report(gap_report: pd.DataFrame) -> dict[str, Any]:
    """Summarize unmatched Kalshi markets and encode the play-in policy."""

    if gap_report.empty:
        reason_rows = {}
        reason_games = {}
    else:
        reason_rows = gap_report["gap_reason"].value_counts(dropna=False).to_dict()
        reason_games = {
            str(reason): _game_count(gap_report[gap_report["gap_reason"].eq(reason)])
            for reason in sorted(gap_report["gap_reason"].dropna().unique())
        }

    return {
        "decision": {
            "play_in_games": "include_when_model_game_rows_exist",
            "preseason_markets": "exclude_from_automated_backtests",
            "exhibition_non_nba_opponent": "exclude_from_automated_backtests",
            "unusual_scalar_settlement": "exclude_until_manually_reviewed",
            "future_or_active_market": "include_after_candles_and_outcomes_exist",
            "no_model_game_row": "exclude_until_prediction_row_exists_or_manual_review_adds_it",
        },
        "rationale": (
            "Play-in games are real NBA team-win markets and should be included in research coverage, "
            "but automated backtests need a leak-safe prediction row, start time, pregame candle, and final outcome. "
            "Preseason/exhibition games are a different distribution and should stay out of the core NBA model."
        ),
        "gap_rows_by_reason": {str(key): int(value) for key, value in reason_rows.items()},
        "gap_games_by_reason": {str(key): int(value) for key, value in reason_games.items()},
        "total_gap_rows": int(len(gap_report)),
        "total_gap_games": _game_count(gap_report),
    }


def _date_text(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(timestamp) else timestamp.date().isoformat()


def audit_matched_markets_sample(
    matches_df: pd.DataFrame,
    sample_size: int = 50,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit a deterministic random sample of auto-matched markets against tickers and titles."""

    if matches_df.empty:
        empty = pd.DataFrame()
        return empty, {"sample_size": 0, "audit_pass": 0, "audit_warning": 0, "audit_fail": 0}

    matches = matches_df[matches_df["match_status"].eq("auto_matched")].copy()
    if matches.empty:
        empty = pd.DataFrame()
        return empty, {"sample_size": 0, "audit_pass": 0, "audit_warning": 0, "audit_fail": 0}

    sample = matches.sample(n=min(sample_size, len(matches)), random_state=random_seed).copy()
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        parsed = _parse_kxnbagame_ticker(row.get("market_ticker", ""))
        parsed = parsed or {}
        row_date = _date_text(row.get("game_date"))
        title = str(row.get("market_title", ""))
        title_teams = set(teams_mentioned_in_text(title))
        home = str(row.get("home_team_abbr", ""))
        away = str(row.get("away_team_abbr", ""))
        yes = str(row.get("yes_team_abbr", ""))

        date_ok = parsed.get("game_date") == row_date
        home_ok = parsed.get("home_team_abbr") == home
        away_ok = parsed.get("away_team_abbr") == away
        yes_ok = parsed.get("yes_team_abbr") == yes
        yes_team_valid = yes in {home, away}
        title_winner_word = any(word in title.lower() for word in ["winner", "win", "wins", "beat", "defeat"])
        title_mentions_home = home in title_teams
        title_mentions_away = away in title_teams
        ticker_ok = bool(date_ok and home_ok and away_ok and yes_ok and yes_team_valid)
        title_ok = bool(title_winner_word and title_mentions_home and title_mentions_away)

        if ticker_ok and title_winner_word:
            audit_status = "pass" if title_ok else "pass_title_alias_warning"
        elif ticker_ok:
            audit_status = "warning"
        else:
            audit_status = "fail"

        rows.append(
            {
                "game_id": row.get("game_id", ""),
                "game_date": row_date,
                "home_team_abbr": home,
                "away_team_abbr": away,
                "yes_team_abbr": yes,
                "market_ticker": row.get("market_ticker", ""),
                "market_title": title,
                "match_score": row.get("match_score", ""),
                "ticker_date_ok": date_ok,
                "ticker_home_ok": home_ok,
                "ticker_away_ok": away_ok,
                "ticker_yes_ok": yes_ok,
                "yes_team_valid": yes_team_valid,
                "title_winner_word": title_winner_word,
                "title_mentions_home": title_mentions_home,
                "title_mentions_away": title_mentions_away,
                "audit_status": audit_status,
            }
        )

    audit = pd.DataFrame(rows).sort_values(["audit_status", "game_date", "market_ticker"]).reset_index(drop=True)
    summary = {
        "sample_size": int(len(audit)),
        "random_seed": int(random_seed),
        "audit_status_counts": audit["audit_status"].value_counts(dropna=False).to_dict(),
        "ticker_failures": int((~audit[["ticker_date_ok", "ticker_home_ok", "ticker_away_ok", "ticker_yes_ok"]].all(axis=1)).sum()),
        "invalid_yes_team_rows": int((~audit["yes_team_valid"]).sum()),
        "title_alias_warnings": int((audit["audit_status"] == "pass_title_alias_warning").sum()),
    }
    return audit, summary


def write_market_review_outputs(
    project_root: str | Path | None = None,
    sample_size: int = 50,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(project_root) if project_root else PROJECT_ROOT
    reports_dir = root / "data" / "reports"
    processed_dir = root / "data" / "processed"
    gap = _read_csv(reports_dir / "kalshi_unmatched_market_gap_report.csv")
    matches = _read_csv(processed_dir / "kalshi_game_market_matches.csv", dtype={"game_id": str})

    gap_decision = build_gap_decision_report(gap)
    audit, audit_summary = audit_matched_markets_sample(
        matches,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    payload = {
        "gap_decision": gap_decision,
        "audit_summary": audit_summary,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "kalshi_market_review_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    audit.to_csv(reports_dir / "kalshi_match_audit_sample_50.csv", index=False)
    return payload, audit
