"""End-to-end validation checks for research data artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.kalshi_backfill import _parse_kxnbagame_ticker


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _read_csv(path: Path, issues: list[dict[str, Any]], **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        issues.append(
            {
                "severity": "warning",
                "check": "file_exists",
                "count": 1,
                "detail": f"Missing optional CSV artifact: {path.name}",
            }
        )
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _read_json(path: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        issues.append(
            {
                "severity": "warning",
                "check": "file_exists",
                "count": 1,
                "detail": f"Missing optional JSON artifact: {path.name}",
            }
        )
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path, issues: list[dict[str, Any]]) -> pd.DataFrame:
    if not path.exists():
        issues.append(
            {
                "severity": "warning",
                "check": "file_exists",
                "count": 1,
                "detail": f"Missing optional parquet artifact: {path.name}",
            }
        )
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        issues.append(
            {
                "severity": "warning",
                "check": "parquet_readable",
                "count": 1,
                "detail": f"Could not read {path.name}: {exc}",
            }
        )
        return pd.DataFrame()


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    check: str,
    count: int,
    detail: str,
) -> None:
    if count:
        issues.append({"severity": severity, "check": check, "count": int(count), "detail": detail})


def _validate_auto_matches(matches: pd.DataFrame, issues: list[dict[str, Any]]) -> dict[str, Any]:
    if matches.empty:
        _add_issue(issues, "error", "auto_matches_present", 1, "No match rows were available.")
        return {"auto_matches": 0}

    auto = matches[matches["match_status"].eq("auto_matched")].copy()
    duplicates = int(auto["game_id"].duplicated().sum()) if "game_id" in auto.columns else 0
    _add_issue(
        issues,
        "error",
        "duplicate_auto_match_game_id",
        duplicates,
        "A game should not have more than one auto-matched Kalshi market row.",
    )

    if {"yes_team_abbr", "home_team_abbr", "away_team_abbr"}.issubset(auto.columns):
        invalid_yes = ~(
            auto["yes_team_abbr"].astype(str).eq(auto["home_team_abbr"].astype(str))
            | auto["yes_team_abbr"].astype(str).eq(auto["away_team_abbr"].astype(str))
        )
        _add_issue(
            issues,
            "error",
            "invalid_yes_team",
            int(invalid_yes.sum()),
            "YES team must be either the home team or away team.",
        )

    ticker_failures = 0
    ticker_date_timezone_warnings = 0
    parsed_rows = 0
    for _, row in auto.iterrows():
        parsed = _parse_kxnbagame_ticker(str(row.get("market_ticker", "")))
        if not parsed:
            continue
        parsed_rows += 1
        game_date = pd.to_datetime(row.get("game_date"), errors="coerce")
        row_date = "" if pd.isna(game_date) else game_date.date().isoformat()
        team_checks = [
            parsed.get("home_team_abbr") == str(row.get("home_team_abbr", "")),
            parsed.get("away_team_abbr") == str(row.get("away_team_abbr", "")),
            parsed.get("yes_team_abbr") == str(row.get("yes_team_abbr", "")),
        ]
        date_ok = parsed.get("game_date") == row_date
        if not all(team_checks):
            ticker_failures += 1
        elif not date_ok:
            parsed_date = pd.to_datetime(parsed.get("game_date"), errors="coerce")
            row_timestamp = pd.to_datetime(row_date, errors="coerce")
            date_delta = abs((row_timestamp - parsed_date).days) if not pd.isna(parsed_date) and not pd.isna(row_timestamp) else 999
            if date_delta <= 1:
                ticker_date_timezone_warnings += 1
            else:
                ticker_failures += 1
    _add_issue(
        issues,
        "error",
        "ticker_parse_mismatch",
        ticker_failures,
        "Parsed KXNBAGAME ticker teams and YES team should agree with the matched game.",
    )
    _add_issue(
        issues,
        "warning",
        "ticker_date_timezone_mismatch",
        ticker_date_timezone_warnings,
        "Ticker date differed from NBA game_date by one day, usually from local-vs-UTC date handling.",
    )
    return {
        "match_rows": int(len(matches)),
        "auto_matches": int(len(auto)),
        "needs_review": int(matches["match_status"].eq("needs_review").sum()),
        "no_match": int(matches["match_status"].eq("no_match").sum()),
        "auto_tickers_parsed": int(parsed_rows),
        "ticker_parse_mismatches": int(ticker_failures),
        "ticker_date_timezone_warnings": int(ticker_date_timezone_warnings),
        "duplicate_auto_match_game_ids": duplicates,
    }


def _validate_prices(prices: pd.DataFrame, issues: list[dict[str, Any]]) -> dict[str, Any]:
    if prices.empty:
        _add_issue(issues, "error", "pregame_prices_present", 1, "No pregame price rows were available.")
        return {"price_rows": 0}

    working = prices.copy()
    for column in ["minutes_before_tipoff", "yes_bid", "yes_ask", "volume", "period_interval"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    after_tipoff = int((working.get("minutes_before_tipoff", pd.Series(dtype=float)) < 0).sum())
    _add_issue(
        issues,
        "error",
        "post_tipoff_price_rows",
        after_tipoff,
        "Pregame snapshots must not come from after tipoff.",
    )

    duplicate_snapshots = 0
    if {"market_ticker", "snapshot_target"}.issubset(working.columns):
        duplicate_snapshots = int(working.duplicated(["market_ticker", "snapshot_target"]).sum())
        _add_issue(
            issues,
            "warning",
            "duplicate_market_snapshot_rows",
            duplicate_snapshots,
            "Each market should have at most one row per snapshot target.",
        )

    strict = working.copy()
    if "snapshot_target" in strict.columns:
        strict = strict[strict["snapshot_target"].eq("pregame_60m")]
    if "price_quality" in strict.columns:
        strict = strict[strict["price_quality"].eq("bid_ask_available")]
    if {"yes_bid", "yes_ask"}.issubset(strict.columns):
        strict = strict[strict["yes_bid"].notna() & strict["yes_ask"].notna()]
    if "period_interval" in strict.columns:
        strict = strict[strict["period_interval"].le(60)]
    if "volume" in strict.columns:
        strict = strict[strict["volume"].fillna(0).ge(10)]

    quality_counts = (
        working["price_quality"].value_counts(dropna=False).to_dict()
        if "price_quality" in working.columns
        else {}
    )
    return {
        "price_rows": int(len(working)),
        "strict_eligible_60m_rows": int(len(strict)),
        "strict_eligible_games": int(strict["game_id"].nunique()) if "game_id" in strict.columns else 0,
        "post_tipoff_price_rows": after_tipoff,
        "duplicate_market_snapshot_rows": duplicate_snapshots,
        "price_quality_counts": {str(key): int(value) for key, value in quality_counts.items()},
    }


def _validate_cross_artifacts(
    games: pd.DataFrame,
    matches: pd.DataFrame,
    prices: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not games.empty:
        summary["nba_game_rows"] = int(len(games))
        if "game_id" in games.columns:
            duplicate_games = int(games["game_id"].duplicated().sum())
            _add_issue(
                issues,
                "error",
                "duplicate_nba_game_rows",
                duplicate_games,
                "NBA game-level data should contain one row per game_id.",
            )
    if not games.empty and not matches.empty and {"game_id"}.issubset(games.columns) and {"game_id"}.issubset(matches.columns):
        game_ids = set(games["game_id"].astype(str))
        auto = matches[matches["match_status"].eq("auto_matched")].copy()
        missing = auto[~auto["game_id"].astype(str).isin(game_ids)].copy()
        missing_game_rows = int(len(missing))
        pending_game_rows = 0
        historical_missing_game_rows = missing_game_rows
        if not missing.empty and "game_date" in missing.columns and "game_date" in games.columns:
            max_completed_date = pd.to_datetime(games["game_date"], errors="coerce").max()
            missing_dates = pd.to_datetime(missing["game_date"], errors="coerce")
            pending_mask = missing_dates.gt(max_completed_date)
            pending_game_rows = int(pending_mask.sum())
            historical_missing_game_rows = int((~pending_mask).sum())
        _add_issue(
            issues,
            "warning",
            "historical_matched_game_missing_from_nba_dataset",
            historical_missing_game_rows,
            "Auto-matched historical markets should map back to the NBA game-level dataset.",
        )
        _add_issue(
            issues,
            "info",
            "pending_matched_game_not_final",
            pending_game_rows,
            "Auto-matched markets after the latest completed NBA game are pending/future rows.",
        )
        summary["auto_matches_with_nba_game_row"] = int(len(auto) - missing_game_rows)
        summary["auto_matches_pending_or_future_game_row"] = pending_game_rows
        summary["auto_matches_missing_historical_game_row"] = historical_missing_game_rows
    if not matches.empty and not prices.empty and {"market_ticker"}.issubset(matches.columns) and {"market_ticker"}.issubset(prices.columns):
        auto_tickers = set(matches.loc[matches["match_status"].eq("auto_matched"), "market_ticker"].astype(str))
        price_tickers = set(prices["market_ticker"].astype(str))
        missing_price = len(auto_tickers - price_tickers)
        summary["auto_matched_markets_missing_any_price_row"] = int(missing_price)
    return summary


def validate_research_data(project_root: str | Path | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Validate saved market, price, model, and audit artifacts."""

    root = Path(project_root) if project_root else PROJECT_ROOT
    processed_dir = root / "data" / "processed"
    reports_dir = root / "data" / "reports"
    interim_dir = root / "data" / "interim"
    issues: list[dict[str, Any]] = []

    matches = _read_csv(processed_dir / "kalshi_game_market_matches.csv", issues, dtype={"game_id": str})
    prices = _read_csv(processed_dir / "kalshi_pregame_prices.csv", issues, dtype={"game_id": str})
    games = _read_parquet(interim_dir / "nba_games.parquet", issues)
    market_review = _read_json(reports_dir / "kalshi_market_review_summary.json", issues)
    coverage = _read_json(reports_dir / "kalshi_coverage_summary.json", issues)

    match_summary = _validate_auto_matches(matches, issues)
    price_summary = _validate_prices(prices, issues)
    cross_summary = _validate_cross_artifacts(games, matches, prices, issues)

    audit_summary = market_review.get("audit_summary", {})
    _add_issue(
        issues,
        "error",
        "sample_audit_ticker_failures",
        int(audit_summary.get("ticker_failures", 0) or 0),
        "The deterministic 50-market audit should have no ticker failures.",
    )
    _add_issue(
        issues,
        "error",
        "sample_audit_invalid_yes_team",
        int(audit_summary.get("invalid_yes_team_rows", 0) or 0),
        "The deterministic 50-market audit should have no invalid YES-team rows.",
    )

    issue_frame = pd.DataFrame(issues, columns=["severity", "check", "count", "detail"])
    error_count = int((issue_frame["severity"] == "error").sum()) if not issue_frame.empty else 0
    warning_count = int((issue_frame["severity"] == "warning").sum()) if not issue_frame.empty else 0
    summary = {
        "validation_status": "pass" if error_count == 0 else "fail",
        "error_checks": error_count,
        "warning_checks": warning_count,
        "matches": match_summary,
        "prices": price_summary,
        "cross_artifacts": cross_summary,
        "coverage": coverage,
        "audit_summary": audit_summary,
    }
    return _json_safe(summary), issue_frame


def write_validation_outputs(project_root: str | Path | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(project_root) if project_root else PROJECT_ROOT
    reports_dir = root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary, issues = validate_research_data(root)
    (reports_dir / "data_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    issues.to_csv(reports_dir / "data_validation_issues.csv", index=False)
    return summary, issues
