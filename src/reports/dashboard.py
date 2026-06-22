"""Generate a self-contained local HTML dashboard from report artifacts."""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.seasons import (
    MIN_TRAINING_SPORTSBOOK_MATCH_RATE,
    TRAIN_END_SEASON,
    TRAIN_START_SEASON,
    build_free_odds_split_plan,
    nba_season_display_label,
    season_start_year_from_dates,
)
from data.canonical_games import normalize_league
from data.sportsbook_odds import load_sportsbook_odds, sportsbook_coverage_by_season


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _fmt_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.2f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _timeline_from_summary(summary: dict[str, Any], trades: pd.DataFrame | None = None) -> str:
    timeline = summary.get("trade_timeline") or summary.get("market_timeline")
    if timeline:
        return str(timeline)
    if trades is None or trades.empty or "date" not in trades.columns:
        return "n/a"
    dates = pd.to_datetime(trades["date"], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    traded = trades.copy()
    if "trade" in traded.columns:
        mask = traded["trade"].astype(str).str.lower().isin(["true", "1", "yes"])
        trade_dates = pd.to_datetime(traded.loc[mask, "date"], errors="coerce").dropna()
        if not trade_dates.empty:
            dates = trade_dates
    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()
    return start if start == end else f"{start} to {end}"


def _metric(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div>'
        "</div>"
    )


def _table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int = 12,
    table_id: str | None = None,
) -> str:
    if df.empty:
        return '<div class="empty">No data available.</div>'

    display = df.copy()
    if columns:
        available = [column for column in columns if column in display.columns]
        display = display[available]
    display = display.head(max_rows)

    table_attrs = f' id="{html.escape(table_id)}"' if table_id else ""
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in display.columns)
    rows = []
    for _, row in display.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (list, tuple, dict)):
                text = json.dumps(value, default=str)
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap"><table{table_attrs}>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


STATIC_DASHBOARD_PAGES = [
    "dashboard.html",
    "matchup_predictions.html",
    "team_availability.html",
    "research_picks.html",
    "paper_candidates.html",
    "trade_results.html",
    "parlay_research.html",
    "player_props.html",
    "project_cleanup.html",
    "recommendation_performance.html",
    "recommendation_grading_audit.html",
    "proof_status.html",
]


def _static_nav(current: str) -> str:
    labels = {
        "dashboard.html": "Dashboard",
        "matchup_predictions.html": "Matchup Predictions",
        "team_availability.html": "Team Availability",
        "research_picks.html": "Research Picks",
        "paper_candidates.html": "Paper Candidates",
        "trade_results.html": "Trade Results",
        "parlay_research.html": "Parlay Research",
        "player_props.html": "Player Props",
        "project_cleanup.html": "Cleanup Audit",
        "recommendation_performance.html": "Recommendation Performance",
        "recommendation_grading_audit.html": "Grading Audit",
        "proof_status.html": "Proof Status",
    }
    links = []
    for page in STATIC_DASHBOARD_PAGES:
        label = labels[page]
        cls = ' class="active-link"' if page == current else ""
        links.append(f'<a{cls} href="{html.escape(page)}">{html.escape(label)}</a>')
    return '<div class="fallback-links">' + "".join(links) + "</div>"


def _csv_download_links(report_path: Path) -> str:
    links = [
        ("fair_price_signals.csv", "Fair price signals CSV"),
        ("backtest_trades.csv", "Backtest trades CSV"),
    ]
    if (report_path / "matchup_predictions_today.csv").exists():
        links.append(("matchup_predictions_today.csv", "Matchup predictions CSV"))
    if (report_path / "team_availability_validation.md").exists():
        links.append(("team_availability_validation.md", "Availability validation report"))
    if (report_path / "team_availability_validation.json").exists():
        links.append(("team_availability_validation.json", "Availability validation JSON"))
    if (report_path / "parlay_recommendations.csv").exists():
        links.append(("parlay_recommendations.csv", "Parlay recommendations CSV"))
    if (report_path / "research_parlay_candidates.csv").exists():
        links.append(("research_parlay_candidates.csv", "Research parlay CSV"))
    if (report_path / "paper_parlay_candidates.csv").exists():
        links.append(("paper_parlay_candidates.csv", "Paper parlay CSV"))
    if (report_path / "graded_single_recommendations.csv").exists():
        links.append(("graded_single_recommendations.csv", "Graded singles CSV"))
    if (report_path / "recommendation_failure_buckets.csv").exists():
        links.append(("recommendation_failure_buckets.csv", "Failure buckets CSV"))
    if (report_path / "recommendation_grading_audit_rows.csv").exists():
        links.append(("recommendation_grading_audit_rows.csv", "Grading audit rows CSV"))
    if (report_path / "recommendation_profit_by_bucket.csv").exists():
        links.append(("recommendation_profit_by_bucket.csv", "Profit audit buckets CSV"))
    if (report_path / "recommendation_clv_by_bucket.csv").exists():
        links.append(("recommendation_clv_by_bucket.csv", "CLV audit buckets CSV"))
    if (report_path / "recommendation_parlay_audit.csv").exists():
        links.append(("recommendation_parlay_audit.csv", "Parlay grading audit CSV"))
    items = "".join(
        f'<a href="{html.escape(filename)}" download>{html.escape(label)}</a>'
        for filename, label in links
    )
    return f'<div class="download-links">{items}</div>'


def _static_page(title: str, report_path: Path, current: str, body: str) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    main_class = "wrap wide-wrap" if current == "matchup_predictions.html" else "wrap"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - NBA Kalshi Predictor</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #18202a; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.4; }}
    header {{ background: #fff; border-bottom: 1px solid #dfe3ea; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 18px 20px 40px; }}
    .wide-wrap {{ max-width: min(1800px, calc(100vw - 24px)); }}
    h1 {{ margin: 0 0 4px; font-size: 28px; }}
    h2 {{ margin-top: 26px; }}
    .subtle, .note {{ color: #667085; }}
    .fallback-links, .download-links {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }}
    .fallback-links a, .download-links a {{ border: 1px solid #d0d5dd; border-radius: 8px; padding: 8px 11px; color: #18202a; background: #fff; text-decoration: none; font-weight: 700; }}
    .fallback-links a.active-link {{ background: #18202a; border-color: #18202a; color: #fff; }}
    .download-links a {{ background: #eef4ff; color: #1b63ce; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric-box {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: 10px; }}
    .metric-label {{ color: #667085; font-size: 12px; font-weight: 700; margin-bottom: 4px; }}
    .big-number {{ font-size: 24px; font-weight: 900; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 920px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #dfe3ea; font-size: 14px; vertical-align: top; }}
    th {{ color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; background: #f8fafc; }}
    .empty {{ padding: 28px; border: 1px dashed #dfe3ea; border-radius: 8px; background: #fff; color: #667085; text-align: center; }}
    .status {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: 14px; }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{html.escape(title)}</h1>
      <div class="subtle">Generated: {html.escape(generated_at)}</div>
      {_static_nav(current)}
      {_csv_download_links(report_path)}
    </div>
  </header>
  <main class="{main_class}">
    {body}
  </main>
</body>
</html>"""


def _summary_grid(items: list[tuple[str, Any]]) -> str:
    return '<div class="summary-grid">' + "".join(
        '<div class="metric-box">'
        f'<div class="metric-label">{html.escape(str(label))}</div>'
        f'<div class="big-number">{html.escape(str(value))}</div>'
        '</div>'
        for label, value in items
    ) + "</div>"


def _series_first(frame: pd.DataFrame, columns: list[str], default: Any = "") -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _static_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 300) -> str:
    return _table(frame, columns=columns, max_rows=max_rows)


LOCAL_TIMEZONE = ZoneInfo("America/New_York")
LEAGUE_DASHBOARD_TABS = [
    ("NBA", "NBA"),
    ("MLB", "MLB"),
    ("WNBA", "WNBA"),
    ("NHL", "NHL"),
    ("WORLD_CUP", "World Cup"),
]
PROP_LEAGUES = {"NBA", "MLB", "WNBA", "NHL"}


def format_local_datetime(value: Any) -> str:
    """Format a timestamp in America/New_York as YYYY-MM-DD HH:MM AM/PM."""
    if value is None:
        return "n/a"
    if isinstance(value, str) and not value.strip():
        return "n/a"
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return "n/a"
    if pd.isna(ts):
        return "n/a"
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TIMEZONE)
    else:
        ts = ts.tz_convert(LOCAL_TIMEZONE)
    return ts.strftime("%Y-%m-%d %I:%M %p")


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _first_existing_nonempty_column(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            values = frame[column].dropna().astype(str).str.strip()
            if not values.empty and values.ne("").any():
                return column
    return None


def _unique_count_from_columns(frame: pd.DataFrame, columns: list[str]) -> int:
    if frame.empty:
        return 0
    column = _first_existing_nonempty_column(frame, columns)
    if column:
        values = frame[column].dropna().astype(str).str.strip()
        return int(values[values.ne("")].nunique())

    combo_parts: list[pd.Series] = []
    for column in ["home_team", "away_team", "game_date", "event_start_time", "game_start_time"]:
        if column in frame.columns:
            combo_parts.append(frame[column].fillna("").astype(str).str.strip())
    if combo_parts:
        combo = combo_parts[0]
        for part in combo_parts[1:]:
            combo = combo + "|" + part
        combo = combo[combo.str.replace("|", "", regex=False).str.strip().ne("")]
        return int(combo.nunique())
    return 0


def _filter_league(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "league" not in frame.columns:
        return frame.iloc[0:0].copy()
    values = frame["league"].fillna("").astype(str).str.upper()
    return frame.loc[values.eq(league.upper())].copy()


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _latest_timestamp_from_column(frame: pd.DataFrame, column: str) -> pd.Timestamp | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
    if values.empty:
        return None
    return values.max()


def _read_active_config_leagues(report_path: Path) -> list[str]:
    config_path = report_path.parent.parent / "config" / "prop_collection.yaml"
    if not config_path.exists():
        return []
    try:
        import yaml

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    leagues = parsed.get("leagues", {})
    if not isinstance(leagues, dict):
        return []
    active = []
    for league, config in leagues.items():
        if isinstance(config, dict) and config.get("enabled", False):
            active.append(str(league))
    return active


def load_league_dashboard_data(report_dir: str | Path) -> dict[str, Any]:
    report_path = Path(report_dir)
    data_root = report_path.parent if report_path.name == "reports" else report_path.parent
    processed = data_root / "processed"
    prop_snapshots_path = processed / "player_prop_snapshots_enriched.csv"
    if not prop_snapshots_path.exists():
        prop_snapshots_path = processed / "player_prop_snapshots_normalized.csv"
    prop_clv_path = report_path / "player_prop_clv.csv"
    world_cup_pairs_path = report_path / "world_cup_clv_pairs.csv"
    return {
        "report_path": report_path,
        "data_root": data_root,
        "prop_snapshots": _read_csv(prop_snapshots_path),
        "prop_clv": _read_csv(prop_clv_path),
        "prop_clv_exists": prop_clv_path.exists(),
        "world_cup_snapshots": _read_csv(processed / "world_cup_odds_snapshots_normalized.csv"),
        "world_cup_clv_pairs": _read_csv(world_cup_pairs_path),
        "world_cup_clv_pairs_exists": world_cup_pairs_path.exists(),
        "prop_collection_summary": _read_json(report_path / "player_prop_collection_run_summary.json"),
        "prop_collection_health": _read_json(report_path / "prop_collection_health_summary.json"),
        "player_prop_gates": _read_json(report_path / "player_prop_data_quality_gates.json"),
        "player_prop_clv_summary": _read_json(report_path / "player_prop_clv_summary.json"),
        "world_cup_collection_summary": _read_json(report_path / "world_cup_collection_summary.json"),
        "world_cup_source_status": _read_json(report_path / "world_cup_source_status.json"),
        "world_cup_clv_summary": _read_json(report_path / "world_cup_clv_summary.json"),
        "single_game_proof": _read_json(report_path / "single_game_proof_summary.json"),
        "paper_bets": _read_csv(report_path / "paper_betting_report.csv"),
        "graded_single_recommendations": _read_csv(report_path / "graded_single_recommendations.csv"),
        "active_config_leagues": _read_active_config_leagues(report_path),
    }


def count_games_for_league(league: str, data: dict[str, Any]) -> int:
    if league == "WORLD_CUP":
        frame = data.get("world_cup_snapshots", pd.DataFrame())
        return _unique_count_from_columns(
            frame,
            ["event_id", "game_id", "canonical_game_key", "matchup"],
        )
    frame = _filter_league(data.get("prop_snapshots", pd.DataFrame()), league)
    return _unique_count_from_columns(
        frame,
        ["canonical_game_key", "game_id", "event_id", "matchup"],
    )


def _complete_market_group_count(
    frame: pd.DataFrame,
    closing_column: str,
    group_columns: list[str],
) -> int:
    if frame.empty or closing_column not in frame.columns:
        return 0
    available = [column for column in group_columns if column in frame.columns]
    if not available:
        return 0
    grouped = frame.copy()
    grouped["_is_closing_like_for_dashboard"] = _to_bool_series(grouped[closing_column])
    grouped["_is_early_for_dashboard"] = ~grouped["_is_closing_like_for_dashboard"]
    counts = grouped.groupby(available, dropna=False).agg(
        has_early=("_is_early_for_dashboard", "any"),
        has_close=("_is_closing_like_for_dashboard", "any"),
    )
    return int((counts["has_early"] & counts["has_close"]).sum())


def count_complete_markets_for_league(league: str, data: dict[str, Any]) -> int:
    if league == "WORLD_CUP":
        pairs = data.get("world_cup_clv_pairs", pd.DataFrame())
        if not pairs.empty:
            return _unique_count_from_columns(
                pairs,
                ["market_key", "event_id", "game_id", "canonical_game_key"],
            ) or int(len(pairs))
        summary = data.get("world_cup_clv_summary", {})
        if isinstance(summary.get("markets_with_clv"), (int, float)):
            return int(summary["markets_with_clv"])
        if not data.get("world_cup_clv_pairs_exists", False):
            return _complete_market_group_count(
                data.get("world_cup_snapshots", pd.DataFrame()),
                "is_closing_like",
                ["league", "event_id", "market_type", "bookmaker", "outcome_name", "line"],
            )
        return 0

    clv = data.get("prop_clv", pd.DataFrame())
    if not clv.empty:
        league_clv = _filter_league(clv, league)
        if league_clv.empty:
            return 0
        group_columns = [
            column
            for column in ["league", "canonical_game_key", "player_name", "prop_type", "bookmaker", "early_line"]
            if column in league_clv.columns
        ]
        if group_columns:
            return int(league_clv.drop_duplicates(group_columns).shape[0])
        return int(len(league_clv))
    by_league = data.get("player_prop_clv_summary", {}).get("markets_with_clv_by_league", {})
    if isinstance(by_league, dict) and isinstance(by_league.get(league), (int, float)):
        return int(by_league[league])
    if not data.get("prop_clv_exists", False):
        return _complete_market_group_count(
            _filter_league(data.get("prop_snapshots", pd.DataFrame()), league),
            "is_closing_snapshot",
            ["league", "canonical_game_key", "player_name", "prop_type", "bookmaker", "line"],
        )
    return 0


def get_last_collected_time_for_league(league: str, data: dict[str, Any]) -> str:
    timestamp = _get_last_collected_timestamp_for_league(league, data)
    return format_local_datetime(timestamp)


def _get_last_collected_timestamp_for_league(league: str, data: dict[str, Any]) -> pd.Timestamp | None:
    if league == "WORLD_CUP":
        timestamp = _latest_timestamp_from_column(data.get("world_cup_snapshots", pd.DataFrame()), "snapshot_time")
        if timestamp is not None:
            return timestamp
        summary = data.get("world_cup_collection_summary", {})
        return pd.Timestamp(summary["generated_at_utc"]) if summary.get("generated_at_utc") else None

    frame = _filter_league(data.get("prop_snapshots", pd.DataFrame()), league)
    timestamp = _latest_timestamp_from_column(frame, "snapshot_time")
    if timestamp is not None:
        return timestamp

    health = data.get("prop_collection_health", {})
    latest: pd.Timestamp | None = None
    for run in health.get("snapshots_by_run", []) if isinstance(health, dict) else []:
        by_league = run.get("snapshots_by_league", {}) if isinstance(run, dict) else {}
        if not isinstance(by_league.get(league), (int, float)) or by_league.get(league, 0) <= 0:
            continue
        try:
            run_ts = pd.Timestamp(run.get("run_time_utc"))
        except Exception:
            continue
        if pd.isna(run_ts):
            continue
        if latest is None or run_ts > latest:
            latest = run_ts
    return latest


def _league_quota_blocked(league: str, data: dict[str, Any]) -> bool:
    if league == "WORLD_CUP":
        return bool(data.get("world_cup_source_status", {}).get("odds_api_below_floor"))
    summary = data.get("prop_collection_summary", {})
    for skipped in summary.get("leagues_skipped", []) if isinstance(summary, dict) else []:
        if str(skipped.get("league", "")).upper() == league.upper() and "quota" in str(skipped).lower():
            return True
    return bool(data.get("prop_collection_health", {}).get("likely_quota_issue")) and league != "NBA"


def _model_gate_reason(league: str, data: dict[str, Any]) -> str:
    if league == "WORLD_CUP":
        return "No World Cup model gate is approved."
    proof = data.get("single_game_proof", {})
    gates = data.get("player_prop_gates", {})
    if proof and not proof.get("single_game_edge_proven", False):
        return "Model gate blocked: single-game edge is not proven."
    status = str(gates.get("status", "") or "")
    if status not in {"modeling_experiment_ready", "clv_ready"}:
        return f"Model gate blocked: data quality status is {status or 'unknown'}."
    return "Research review only."


def get_research_bets_for_league(league: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    proof = data.get("single_game_proof", {})
    gates = data.get("player_prop_gates", {})
    if league == "WORLD_CUP":
        return []
    if proof and not proof.get("single_game_edge_proven", False):
        return []
    if gates and str(gates.get("status", "")) not in {"modeling_experiment_ready", "clv_ready"}:
        return []

    candidate_frames = [
        data.get("paper_bets", pd.DataFrame()),
        data.get("graded_single_recommendations", pd.DataFrame()),
    ]
    rows: list[dict[str, Any]] = []
    for frame in candidate_frames:
        if frame.empty:
            continue
        league_frame = _filter_league(frame, league)
        if league_frame.empty:
            continue
        if "blocked_reason" in league_frame.columns:
            blocked = league_frame["blocked_reason"].fillna("").astype(str).str.strip()
            league_frame = league_frame.loc[blocked.eq("")]
        if "result_status" in league_frame.columns:
            statuses = league_frame["result_status"].fillna("").astype(str).str.lower()
            league_frame = league_frame.loc[~statuses.isin({"graded", "won", "lost", "push"})]
        if "status" in league_frame.columns:
            statuses = league_frame["status"].fillna("").astype(str).str.lower()
            league_frame = league_frame.loc[~statuses.isin({"graded", "won", "lost", "push"})]
        for _, row in league_frame.head(8).iterrows():
            recommendation = str(row.get("recommendation") or row.get("label") or "")
            if "no bet" in recommendation.lower():
                continue
            rows.append(
                {
                    "Market": row.get("market") or row.get("market_ticker") or "",
                    "Side": row.get("side") or row.get("research_side") or row.get("graded_side") or "",
                    "Line": row.get("line") or "",
                    "Price": row.get("price") or row.get("odds") or row.get("research_price") or "",
                    "Model Probability": row.get("model_probability") or row.get("model_prob") or row.get("calibrated_prob") or "",
                    "Edge": row.get("edge") or row.get("final_edge") or "",
                    "Confidence": row.get("confidence") or row.get("confidence_tier") or row.get("confidence_label") or "",
                    "Reason": row.get("main_reason") or "Research-only candidate bets",
                    "Risk": row.get("main_risk") or _model_gate_reason(league, data),
                    "Status": "Research-only candidate bets; approved=false",
                }
            )
    return rows


def build_league_summary(league: str, data: dict[str, Any]) -> dict[str, Any]:
    display = dict(LEAGUE_DASHBOARD_TABS).get(league, league)
    games = count_games_for_league(league, data)
    complete_markets = count_complete_markets_for_league(league, data)
    research_bets = get_research_bets_for_league(league, data)
    last_ts = _get_last_collected_timestamp_for_league(league, data)
    last_collected = format_local_datetime(last_ts)
    quota_blocked = _league_quota_blocked(league, data)
    gate_reason = _model_gate_reason(league, data)

    if quota_blocked and complete_markets == 0:
        data_status = "Blocked by quota"
    elif games == 0:
        data_status = "Waiting for data"
    elif complete_markets == 0:
        data_status = "Waiting for data"
    elif not research_bets:
        data_status = "Gate blocked"
    else:
        data_status = "Ready for review"

    complete_note = (
        "CLV/open-close pairs found."
        if complete_markets
        else "Waiting for both start and end snapshots."
    )
    return {
        "league": league,
        "display": display,
        "games": games,
        "complete_markets": complete_markets,
        "research_bets": research_bets,
        "research_bet_count": len(research_bets),
        "last_collected": last_collected,
        "last_collected_raw": last_ts,
        "data_status": data_status,
        "complete_note": complete_note,
        "gate_reason": gate_reason,
    }


def _build_all_league_summaries(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [build_league_summary(league, data) for league, _ in LEAGUE_DASHBOARD_TABS]


def _status_badge(label: str) -> str:
    tones = {
        "Working": "green",
        "Waiting for data": "gray",
        "Blocked by quota": "yellow",
        "Gate blocked": "red",
        "Ready for review": "blue",
    }
    tone = tones.get(label, "gray")
    return f'<span class="status-badge {tone}">{html.escape(label)}</span>'


def _summary_table_for_league(summary: dict[str, Any]) -> str:
    frame = pd.DataFrame(
        [
            {
                "League": summary["display"],
                "Games": _format_count(summary["games"]),
                "Complete Markets": _format_count(summary["complete_markets"]),
                "Last Collected": summary["last_collected"],
                "Data Status": summary["data_status"],
            }
        ]
    )
    return _table(frame, ["League", "Games", "Complete Markets", "Last Collected", "Data Status"], max_rows=1)


def _research_bets_table_for_league(summary: dict[str, Any]) -> str:
    rows = summary["research_bets"]
    if not rows:
        reason = summary["gate_reason"] if summary["complete_markets"] else summary["complete_note"]
        return (
            '<div class="empty bet-empty">'
            "<strong>No qualifying research bets yet.</strong>"
            f'<div class="empty-note">{html.escape(reason)}</div>'
            '<div class="empty-note">Future candidates stay research-only and approved=false.</div>'
            "</div>"
        )
    frame = pd.DataFrame(rows)
    columns = [
        "Market",
        "Side",
        "Line",
        "Price",
        "Model Probability",
        "Edge",
        "Confidence",
        "Reason",
        "Risk",
        "Status",
    ]
    return '<div class="table-note">Research-only candidate bets</div>' + _table(frame, columns, max_rows=8)


def _inactive_leagues_section(data: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    shown = {summary["league"] for summary in summaries}
    active = [league for league in data.get("active_config_leagues", []) if league not in shown]
    if not active:
        return ""
    rows = []
    for league in active:
        if count_games_for_league(league, data) > 0:
            continue
        rows.append(
            {
                "League": league,
                "Status": "Inactive on this dashboard",
                "Reason": "No collected games in the current local artifacts.",
            }
        )
    if not rows:
        return ""
    return (
        '<details class="advanced-section"><summary>More / inactive leagues</summary>'
        + _table(pd.DataFrame(rows), ["League", "Status", "Reason"], max_rows=30)
        + "</details>"
    )


def _advanced_reports_section(report_path: Path) -> str:
    links = [
        ("single_game_edge_recommendations.md", "Single-game edge notes"),
        ("single_game_proof_summary.json", "Single-game proof summary"),
        ("player_prop_clv.md", "Player prop CLV report"),
        ("nba_prop_clv_readiness_summary.json", "NBA CLV readiness summary"),
        ("nba_prop_closing_coverage.csv", "NBA closing coverage CSV"),
        ("player_prop_market_quality.md", "Player prop market quality"),
        ("player_prop_line_quality.csv", "Player prop line quality CSV"),
        ("player_prop_possible_alt_lines.csv", "Possible alt lines CSV"),
        ("player_prop_settlement_outcomes.md", "Settlement outcomes report"),
        ("player_prop_settlement_refresh_summary.json", "Settlement refresh summary"),
        ("player_prop_data_quality_gates.md", "Data quality gates"),
        ("prop_collection_health.md", "Collection health report"),
        ("source_health_summary.md", "Source health summary"),
        ("market_truth_audit_summary.json", "Market truth audit summary"),
        ("world_cup_clv_summary.md", "World Cup CLV summary"),
        ("paper_betting_report.md", "Paper tracking report"),
    ]
    items = []
    for filename, label in links:
        if (report_path / filename).exists():
            items.append(f'<a href="{html.escape(filename)}">{html.escape(label)}</a>')
    if not items:
        return ""
    return (
        '<details class="advanced-section"><summary>Advanced reports</summary>'
        '<div class="advanced-links">'
        + "".join(items)
        + "</div></details>"
    )


def _build_league_tab(summary: dict[str, Any], recorded_frame: "pd.DataFrame | None" = None) -> str:
    cards = [
        ("Games Collected", _format_count(summary["games"]), "Unique games/events collected."),
        ("Complete Markets", _format_count(summary["complete_markets"]), summary["complete_note"]),
        (
            "Research Bets",
            _format_count(summary["research_bet_count"]),
            "No real betting is enabled.",
        ),
        ("Last Data Collection", summary["last_collected"], "America/New_York."),
    ]
    card_html = "".join(
        '<div class="league-card">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(str(value))}</div>'
        f'<div class="card-note">{html.escape(str(note))}</div>'
        "</div>"
        for label, value, note in cards
    )
    return (
        f'<section id="league-{html.escape(summary["league"].lower().replace("_", "-"))}" '
        f'class="league-panel{" active" if summary["league"] == "NBA" else ""}">'
        f'<div class="league-heading"><h2>{html.escape(summary["display"])}</h2>{_status_badge(summary["data_status"])}</div>'
        f'<div class="league-cards">{card_html}</div>'
        "<h3>Games / Markets Summary</h3>"
        + _summary_table_for_league(summary)
        + "<h3>Research Bets We Would Place</h3>"
        + _research_bets_table_for_league(summary)
        + _render_recorded_games_section(recorded_frame, league=summary["league"])
        + "</section>"
    )


def _build_sports_market_research_dashboard_html(report_path: Path) -> str:
    data = load_league_dashboard_data(report_path)
    summaries = _build_all_league_summaries(data)
    total_games = sum(summary["games"] for summary in summaries)
    total_complete = sum(summary["complete_markets"] for summary in summaries)
    collecting = sum(1 for summary in summaries if summary["games"] > 0)
    latest_values = [summary["last_collected_raw"] for summary in summaries if summary["last_collected_raw"] is not None]
    latest_collection = format_local_datetime(max(latest_values)) if latest_values else "n/a"
    generated_at = format_local_datetime(datetime.now(tz=LOCAL_TIMEZONE))

    nav = "".join(
        '<button class="league-tab{active}" data-tab="{tab_id}">{label}</button>'.format(
            active=" active" if index == 0 else "",
            tab_id=html.escape(summary["league"].lower().replace("_", "-")),
            label=html.escape(summary["display"]),
        )
        for index, summary in enumerate(summaries)
    )
    recorded_frame = _recorded_games_frame(report_path)
    panels = "".join(_build_league_tab(summary, recorded_frame) for summary in summaries)
    nav += '<button class="league-tab" data-tab="historical-backfill">Historical Backfill</button>'
    panels += _build_historical_backfill_panel(recorded_frame, report_path)
    status_strip = "".join(
        '<div class="strip-item">'
        f'<span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong>'
        "</div>"
        for label, value in [
            ("Total games", _format_count(total_games)),
            ("Total complete markets", _format_count(total_complete)),
            ("Leagues collecting", _format_count(collecting)),
            ("Latest collection time", latest_collection),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sports Market Research Dashboard</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --blue: #175cd3;
      --green: #087443;
      --yellow: #b54708;
      --red: #b42318;
      --gray: #475467;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.42; }}
    header {{ background: #fff; border-bottom: 1px solid var(--line); }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 18px 18px 40px; }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 22px 0 10px; font-size: 16px; letter-spacing: 0; }}
    .subtitle {{ margin-top: 4px; color: var(--muted); font-size: 15px; }}
    .generated {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .status-strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .strip-item {{ background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 11px 12px; }}
    .strip-item span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 750; }}
    .strip-item strong {{ display: block; margin-top: 3px; font-size: 18px; overflow-wrap: anywhere; }}
    .league-tabs-wrap {{ position: sticky; top: 0; z-index: 20; background: rgba(244, 246, 248, 0.96); border-bottom: 1px solid var(--line); }}
    .league-tabs {{ display: flex; gap: 8px; overflow-x: auto; padding: 10px 18px; max-width: 1180px; margin: 0 auto; }}
    .league-tab {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 9px 13px; font: inherit; font-weight: 800; cursor: pointer; white-space: nowrap; }}
    .league-tab.active {{ background: var(--ink); border-color: var(--ink); color: #fff; }}
    .league-panel {{ display: none; }}
    .league-panel.active {{ display: block; }}
    .league-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }}
    .league-cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .league-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 120px; }}
    .card-label {{ color: var(--muted); font-size: 12px; font-weight: 850; text-transform: uppercase; }}
    .card-value {{ margin-top: 7px; font-size: 25px; font-weight: 900; overflow-wrap: anywhere; }}
    .card-note {{ margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .status-badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 850; border: 1px solid transparent; }}
    .status-badge.green {{ color: var(--green); background: #ecfdf3; border-color: #abefc6; }}
    .status-badge.blue {{ color: var(--blue); background: #eef4ff; border-color: #b2ccff; }}
    .status-badge.yellow {{ color: var(--yellow); background: #fffaeb; border-color: #fedf89; }}
    .status-badge.red {{ color: var(--red); background: #fef3f2; border-color: #fecdca; }}
    .status-badge.gray {{ color: var(--gray); background: #f2f4f7; border-color: #d0d5dd; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ text-align: left; padding: 11px 12px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }}
    th {{ background: #f8fafc; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0; }}
    .empty {{ padding: 24px; border: 1px dashed var(--line); border-radius: 8px; background: #fff; color: var(--muted); }}
    .empty strong {{ display: block; color: var(--ink); margin-bottom: 5px; }}
    .empty-note {{ margin-top: 4px; }}
    .table-note {{ color: var(--muted); font-size: 12px; font-weight: 800; margin-bottom: 8px; }}
    .advanced-section {{ margin-top: 18px; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 11px 13px; }}
    .advanced-section summary {{ cursor: pointer; font-weight: 850; }}
    .advanced-links {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .advanced-links a {{ border: 1px solid #b2ccff; background: #eef4ff; color: var(--blue); border-radius: 8px; padding: 8px 10px; font-weight: 800; text-decoration: none; }}
    .top-links {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .top-links a {{ border: 1px solid var(--ink); background: var(--ink); color: #fff; border-radius: 8px; padding: 9px 13px; font-weight: 850; text-decoration: none; }}
    @media (max-width: 860px) {{
      h1 {{ font-size: 24px; }}
      .status-strip, .league-cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .wrap {{ padding-left: 12px; padding-right: 12px; }}
      .league-tabs {{ padding-left: 12px; padding-right: 12px; }}
      .status-strip, .league-cards {{ grid-template-columns: 1fr; }}
      table {{ min-width: 720px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Sports Market Research Dashboard</h1>
      <div class="subtitle">Research-only. No approved bets are live.</div>
      <div class="generated">Generated {html.escape(generated_at)} in America/New_York.</div>
      <div class="top-links">
        <a href="matchup_predictions.html">Matchup Predictions (model probabilities) &rarr;</a>
        <a href="team_availability.html">Team Availability &rarr;</a>
      </div>
      <div class="status-strip" aria-label="Dashboard status">{status_strip}</div>
    </div>
  </header>
  <div class="league-tabs-wrap">
    <nav class="league-tabs" aria-label="League tabs">{nav}</nav>
  </div>
  <main class="wrap">
    {panels}
    {_inactive_leagues_section(data, summaries)}
    {_advanced_reports_section(report_path)}
  </main>
  <script>
    document.querySelectorAll(".league-tab").forEach((button) => {{
      button.addEventListener("click", () => {{
        const tabId = button.dataset.tab;
        document.querySelectorAll(".league-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
        document.querySelectorAll(".league-panel").forEach((section) => section.classList.toggle("active", section.id === `league-${{tabId}}`));
      }});
    }});
  </script>
</body>
</html>"""


def _dashboard_simplification_markdown(report_path: Path) -> str:
    data = load_league_dashboard_data(report_path)
    summaries = _build_all_league_summaries(data)
    generated_at = format_local_datetime(datetime.now(tz=LOCAL_TIMEZONE))
    rows = [
        "| League | Games | Complete markets | Last collected | Research bets | Data status |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for summary in summaries:
        rows.append(
            "| {league} | {games} | {complete} | {last} | {bets} | {status} |".format(
                league=summary["display"],
                games=summary["games"],
                complete=summary["complete_markets"],
                last=summary["last_collected"],
                bets=summary["research_bet_count"],
                status=summary["data_status"],
            )
        )
    any_bets = any(summary["research_bet_count"] for summary in summaries)
    tabs = ", ".join(summary["display"] for summary in summaries)
    return "\n".join(
        [
            "# Dashboard Simplification Report",
            "",
            f"Generated: {generated_at} America/New_York",
            "",
            "## What changed",
            "",
            "- Replaced the crowded dashboard entry point with league tabs and four core cards per league.",
            "- The main page now reads local CSV/JSON artifacts only; it does not make network calls.",
            "- Historical and technical detail is moved to collapsed advanced links at the bottom.",
            "- Research bet display is gated; no historical paper candidates are promoted into live recommendations.",
            "",
            "## League tabs included",
            "",
            tabs,
            "",
            "## League counts",
            "",
            *rows,
            "",
            "## Research bets shown",
            "",
            "Yes." if any_bets else "No. The dashboard shows \"No qualifying research bets yet.\" because proof/model gates are not cleared.",
            "",
            "## Why real bets are still not enabled",
            "",
            "- The dashboard is research-only and writes no orders.",
            "- `single_game_proof_summary.json` does not prove repeatable single-game edge.",
            "- The data-quality gate is still enforced before any candidate can be displayed.",
            "- Parlays are not enabled from the dashboard.",
            "",
            "## Test results",
            "",
            "Not recorded yet.",
            "",
        ]
    )


def write_dashboard_simplification_report(report_dir: str | Path) -> Path:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    output = report_path / "dashboard_simplification_report.md"
    output.write_text(_dashboard_simplification_markdown(report_path), encoding="utf-8")
    return output


def _plot_panel(title: str, data_uri: str) -> str:
    if not data_uri:
        return (
            '<section class="panel">'
            f"<h3>{html.escape(title)}</h3>"
            '<div class="empty">Plot not generated yet.</div>'
            "</section>"
        )
    return (
        '<section class="panel">'
        f"<h3>{html.escape(title)}</h3>"
        f'<img class="plot" src="{data_uri}" alt="{html.escape(title)}">'
        "</section>"
    )


def _quality_list(quality: dict[str, Any]) -> str:
    warnings = quality.get("warnings", [])
    if not warnings:
        return '<div class="status good">No market data quality warnings.</div>'
    items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
    return f'<div class="status warn"><ul>{items}</ul></div>'


def _fold_table(walk_forward_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for fold in walk_forward_metrics.get("folds", []):
        model = fold.get("model", {})
        rows.append(
            {
                "season": fold.get("test_season"),
                "train": f"{fold.get('train_start_season')}-{fold.get('train_end_season')}",
                "games": fold.get("test_rows"),
                "accuracy": _fmt_pct(model.get("accuracy")),
                "log_loss": _fmt_number(model.get("log_loss"), 3),
                "auc": _fmt_number(model.get("roc_auc"), 3),
            }
        )
    return pd.DataFrame(rows)


def _format_threshold_table(threshold_sweep: pd.DataFrame) -> pd.DataFrame:
    if threshold_sweep.empty:
        return threshold_sweep
    display = threshold_sweep.copy()
    for column in ["total_return_pct", "win_rate", "max_drawdown", "roi_on_amount_risked"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    if "ending_bankroll" in display.columns:
        display["ending_bankroll"] = display["ending_bankroll"].map(_fmt_money)
    return display


def _format_probability_bins(probability_bins: pd.DataFrame) -> pd.DataFrame:
    if probability_bins.empty:
        return probability_bins
    display = probability_bins.copy()
    for column in [
        "avg_predicted_prob",
        "observed_win_rate",
        "calibration_error",
        "abs_calibration_error",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    return display


def _format_season_summary(season_summary: pd.DataFrame) -> pd.DataFrame:
    if season_summary.empty:
        return season_summary
    display = season_summary.copy()
    for column in ["accuracy", "avg_predicted_home_win_prob", "actual_home_win_rate"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["brier_score", "mean_absolute_calibration_error"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: _fmt_number(value, 3))
    return display


def _format_edge_bins(edge_bins: pd.DataFrame) -> pd.DataFrame:
    if edge_bins.empty:
        return edge_bins
    display = edge_bins.copy()
    for column in ["avg_edge", "win_rate", "traded_win_rate", "roi_on_amount_risked"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["total_profit", "avg_profit", "amount_risked"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_top_trades(top_trades: pd.DataFrame) -> pd.DataFrame:
    if top_trades.empty:
        return top_trades
    display = top_trades.copy()
    for column in ["model_yes_prob", "market_prob", "edge"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["cost", "payout", "profit", "bankroll_before", "bankroll_after", "abs_profit"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_portfolio_trades(portfolio_trades: pd.DataFrame) -> pd.DataFrame:
    if portfolio_trades.empty:
        return portfolio_trades
    display = portfolio_trades.copy()
    for column in ["model_yes_prob", "market_prob", "edge", "expected_roi", "selection_expected_roi"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["cost", "payout", "profit", "bankroll_before_slate", "bankroll_after"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_portfolio_slates(portfolio_slates: pd.DataFrame) -> pd.DataFrame:
    if portfolio_slates.empty:
        return portfolio_slates
    display = portfolio_slates.copy()
    for column in ["slate_cost_fraction"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["bankroll_before", "slate_cost", "slate_payout", "slate_profit", "bankroll_after"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_edge_calibration(edge_calibration: pd.DataFrame) -> pd.DataFrame:
    if edge_calibration.empty:
        return edge_calibration
    display = edge_calibration.copy()
    for column in [
        "avg_edge",
        "avg_model_prob",
        "avg_market_prob",
        "observed_yes_rate",
        "avg_model_expected_roi",
        "realized_roi_on_cost",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["avg_model_expected_profit_per_share", "avg_realized_profit_per_share", "calibration_gap"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_edge_audit(edge_audit: pd.DataFrame) -> pd.DataFrame:
    if edge_audit.empty:
        return edge_audit
    display = edge_audit.copy()
    for column in [
        "avg_edge",
        "avg_calibrated_expected_roi",
        "observed_yes_rate",
        "signal_win_rate",
        "signal_avg_calibrated_expected_roi",
        "negative_raw_edge_signal_win_rate",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in [
        "all_avg_realized_profit_per_share",
        "signal_avg_realized_profit_per_share",
        "negative_raw_edge_signal_profit_per_share",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_negative_edge_signals(negative_signals: pd.DataFrame) -> pd.DataFrame:
    if negative_signals.empty:
        return negative_signals
    display = negative_signals.copy()
    for column in [
        "edge",
        "model_yes_prob",
        "market_prob",
        "calibrated_yes_rate",
        "calibrated_expected_roi",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["realized_profit_per_share"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    return display


def _format_signal_stability(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return stability
    display = stability.copy()
    for column in ["win_rate", "avg_profit_per_share", "total_profit_per_share", "avg_edge", "avg_expected_roi"]:
        if column in display.columns:
            if "profit" in column:
                display[column] = display[column].map(_fmt_money)
            else:
                display[column] = display[column].map(_fmt_pct)
    return display


def _format_strategy_readiness(readiness: pd.DataFrame) -> pd.DataFrame:
    if readiness.empty:
        return readiness
    display = readiness.copy()
    for column in [
        "positive_month_share",
        "avg_signal_profit_per_share",
        "total_return_pct",
        "max_drawdown",
    ]:
        if column in display.columns:
            if column == "avg_signal_profit_per_share":
                display[column] = display[column].map(_fmt_money)
            else:
                display[column] = display[column].map(_fmt_pct)
    if "ending_bankroll" in display.columns:
        display["ending_bankroll"] = display["ending_bankroll"].map(_fmt_money)
    return display


def _format_signal_rule_sweep(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return rules
    display = rules.copy()
    for column in [
        "score",
        "min_edge",
        "min_expected_roi",
        "positive_month_share",
        "overall_win_rate",
        "overall_avg_profit_per_share",
        "worst_month_avg_profit_per_share",
        "best_month_avg_profit_per_share",
        "avg_edge",
        "avg_expected_roi",
    ]:
        if column in display.columns:
            if "profit" in column:
                display[column] = display[column].map(_fmt_money)
            elif column == "score":
                display[column] = display[column].map(lambda value: _fmt_number(value, 3))
            else:
                display[column] = display[column].map(_fmt_pct)
    for column in ["min_price_cents", "max_price_cents"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: _fmt_number(value, 0))
    return display


def _format_ensemble_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return weights
    display = weights.copy()
    for column in display.columns:
        if column.startswith("weight_") and column != "weight_source":
            display[column] = display[column].map(_fmt_pct)
    for column in ["train_selection_log_loss", "test_log_loss", "test_brier_score", "test_accuracy", "test_roc_auc"]:
        if column in display.columns:
            if column == "test_accuracy":
                display[column] = display[column].map(_fmt_pct)
            else:
                display[column] = display[column].map(lambda value: _fmt_number(value, 4))
    return display


def _format_ensemble_static_audit(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    display = audit.copy()
    for column in display.columns:
        if column.startswith("weight_") and column != "weight_source":
            display[column] = display[column].map(_fmt_pct)
    for column in ["accuracy"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["brier_score", "log_loss", "roc_auc"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: _fmt_number(value, 4))
    return display


def _format_parlay_report(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return report
    display = report.copy()
    for column in [
        "pair_win_rate",
        "avg_market_pair_prob_independent",
        "avg_estimated_pair_prob_independent",
        "avg_pair_edge_independent",
        "avg_synthetic_profit_per_dollar",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    if "leg_outcome_correlation" in display.columns:
        display["leg_outcome_correlation"] = display["leg_outcome_correlation"].map(lambda value: _fmt_number(value, 4))
    return display


def _format_forward_recommendations(forward: pd.DataFrame) -> pd.DataFrame:
    if forward.empty:
        return forward
    display = forward.copy()
    for column in [
        "model_home_win_prob",
        "model_away_win_prob",
        "model_pick_prob",
        "model_yes_prob",
        "market_prob",
        "edge",
        "forward_expected_roi",
    ]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    for column in ["paper_amount_risked", "hypothetical_amount_risked", "starting_bankroll"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_money)
    if "price_cents" in display.columns:
        display["price_cents"] = display["price_cents"].map(lambda value: _fmt_number(value, 1))
    return display


def _format_tuning_results(tuning_results: pd.DataFrame) -> pd.DataFrame:
    if tuning_results.empty:
        return tuning_results
    display = tuning_results.copy()
    for column in ["accuracy", "brier_score", "log_loss", "roc_auc"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: _fmt_number(value, 4))
    return display


def _format_coverage_monthly(coverage_monthly: pd.DataFrame) -> pd.DataFrame:
    if coverage_monthly.empty:
        return coverage_monthly
    display = coverage_monthly.copy()
    for column in ["market_coverage_pct", "price_coverage_pct"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    return display


def _format_market_blend_rows(market_blend_metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name in ["model", "market", "market_blend"]:
        metrics = market_blend_metrics.get(name, {})
        if metrics:
            rows.append(
                {
                    "probability_source": name,
                    "accuracy": _fmt_pct(metrics.get("accuracy")),
                    "brier_score": _fmt_number(metrics.get("brier_score"), 3),
                    "log_loss": _fmt_number(metrics.get("log_loss"), 3),
                    "roc_auc": _fmt_number(metrics.get("roc_auc"), 3),
                }
            )
    return pd.DataFrame(rows)


def _portfolio_comparison_rows(items: list[tuple[str, dict[str, Any], str]]) -> pd.DataFrame:
    rows = []
    for name, summary, timeline in items:
        if not summary:
            continue
        rows.append(
            {
                "strategy": name,
                "trades": summary.get("num_selected_trades", summary.get("num_trades", "n/a")),
                "timeline": timeline,
                "ending_bankroll": _fmt_money(summary.get("ending_bankroll")),
                "return": _fmt_pct(summary.get("total_return_pct")),
                "max_drawdown": _fmt_pct(summary.get("max_drawdown")),
                "roi_on_risk": _fmt_pct(summary.get("roi_on_amount_risked")),
            }
        )
    return pd.DataFrame(rows)


def _headline_backtest_rows(summary: dict[str, Any]) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "headline": summary.get("headline_label", "n/a"),
                "settlement": summary.get("settlement_mode", "n/a"),
                "trades": summary.get("num_selected_trades", "n/a"),
                "timeline": summary.get("trade_timeline", "n/a"),
                "ending_bankroll": _fmt_money(summary.get("ending_bankroll")),
                "return": _fmt_pct(summary.get("total_return_pct")),
                "max_drawdown": _fmt_pct(summary.get("max_drawdown")),
                "readiness": summary.get("readiness_status", "n/a"),
                "parlays_blocked": summary.get("parlays_blocked", True),
            }
        ]
    )


def _format_market_type_calibration(market_type_calibration: pd.DataFrame) -> pd.DataFrame:
    if market_type_calibration.empty:
        return market_type_calibration
    display = market_type_calibration.copy()
    for column in ["avg_predicted_prob", "observed_rate", "calibration_error", "abs_calibration_error"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    return display


def _format_market_line_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return coverage
    display = coverage.copy()
    for column in ["line_coverage_pct", "high_confidence_coverage_pct"]:
        if column in display.columns:
            display[column] = display[column].map(_fmt_pct)
    return display


def _gap_reason_table(gap_report: pd.DataFrame) -> pd.DataFrame:
    if gap_report.empty or "gap_reason" not in gap_report.columns:
        return pd.DataFrame()
    counts = gap_report["gap_reason"].value_counts(dropna=False).reset_index()
    counts.columns = ["gap_reason", "market_rows"]
    if "market_ticker" in gap_report.columns:
        game_counts = (
            gap_report.drop_duplicates(["game_date", "home_team_abbr", "away_team_abbr", "gap_reason"])
            .groupby("gap_reason", as_index=False)
            .agg(games=("market_ticker", "count"))
        )
        counts = counts.merge(game_counts, on="gap_reason", how="left")
    return counts


TEAM_COLORS = {
    "ATL": "#E03A3E",
    "BKN": "#111111",
    "BOS": "#007A33",
    "CHA": "#1D1160",
    "CHI": "#CE1141",
    "CLE": "#860038",
    "DAL": "#00538C",
    "DEN": "#0E2240",
    "DET": "#C8102E",
    "GSW": "#1D428A",
    "HOU": "#CE1141",
    "IND": "#002D62",
    "LAC": "#C8102E",
    "LAL": "#552583",
    "MEM": "#5D76A9",
    "MIA": "#98002E",
    "MIL": "#00471B",
    "MIN": "#0C2340",
    "NOP": "#0C2340",
    "NYK": "#006BB6",
    "OKC": "#007AC1",
    "ORL": "#0077C0",
    "PHI": "#006BB6",
    "PHX": "#1D1160",
    "POR": "#E03A3E",
    "SAC": "#5A2D81",
    "SAS": "#5A5A5A",
    "TOR": "#CE1141",
    "UTA": "#002B5C",
    "WAS": "#002B5C",
}


def _json_records(frame: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    output = frame.copy()
    if max_rows is not None:
        output = output.head(max_rows)
    output = output.replace({np.nan: None})
    return output.to_dict(orient="records")


def _safe_file_mtime(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %I:%M %p")


def _safe_min_max_dates(frame: pd.DataFrame, column: str) -> tuple[str, str]:
    if frame.empty or column not in frame.columns:
        return "n/a", "n/a"
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return "n/a", "n/a"
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _read_source_frame(path: Path, file_type: str) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "missing"
    try:
        if file_type == "parquet":
            return pd.read_parquet(path), ""
        return pd.read_csv(path, low_memory=False), ""
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _source_record(label: str, path: Path, frame: pd.DataFrame, error: str = "") -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "error": error,
    }


def _first_non_empty_source(candidates: list[tuple[str, Path, str]]) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for label, path, file_type in candidates:
        frame, error = _read_source_frame(path, file_type)
        record = _source_record(label, path, frame, error)
        attempts.append(record)
        if not frame.empty:
            return frame, record, attempts
    return pd.DataFrame(), attempts[-1] if attempts else {}, attempts


def _coverage_seasons(frame: pd.DataFrame) -> pd.Series:
    if "season" in frame.columns:
        seasons = pd.to_numeric(frame["season"], errors="coerce")
        if seasons.notna().any():
            for column in ["game_date", "date"]:
                if seasons.isna().any() and column in frame.columns:
                    seasons = seasons.fillna(season_start_year_from_dates(frame[column]))
            return seasons
    for column in ["game_date", "date"]:
        if column in frame.columns:
            return season_start_year_from_dates(frame[column])
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _canonical_game_ids(frame: pd.DataFrame) -> pd.Series:
    if "game_id" in frame.columns:
        ids = frame["game_id"].astype(str).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        if ids.notna().any():
            return ids

    date_column = "game_date" if "game_date" in frame.columns else "date" if "date" in frame.columns else None
    home_column = "home_team_abbr" if "home_team_abbr" in frame.columns else "home_team" if "home_team" in frame.columns else None
    away_column = "away_team_abbr" if "away_team_abbr" in frame.columns else "away_team" if "away_team" in frame.columns else None
    if date_column and home_column and away_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce").dt.date.astype(str)
        return dates + "|" + frame[home_column].astype(str) + "|" + frame[away_column].astype(str)

    return pd.Series(frame.index.astype(str), index=frame.index)


def _team_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["home_team_abbr", "away_team_abbr", "home_team", "away_team"]
        if column in frame.columns
    ]


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in ["game_date", "date"]:
        if column in frame.columns:
            return column
    return None


def _dataset_split_for_season(
    season: int,
    train_start_season: int = TRAIN_START_SEASON,
    train_end_season: int = TRAIN_END_SEASON,
) -> str:
    if train_start_season <= season <= train_end_season:
        return "train"
    if season == 2024:
        return "validation"
    if season == 2025:
        return "test"
    return "outside_split"


def _load_games_for_coverage(report_path: Path, data_root: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        ("interim_nba_games_csv", data_root / "interim" / "nba_games.csv", "csv"),
        ("interim_nba_games_parquet", data_root / "interim" / "nba_games.parquet", "parquet"),
        ("modeling_dataset_parquet", data_root / "processed" / "modeling_dataset.parquet", "parquet"),
        ("all_game_predictions_csv", report_path / "all_game_predictions.csv", "csv"),
        ("walk_forward_predictions_csv", report_path / "walk_forward_predictions.csv", "csv"),
        ("kalshi_game_market_matches_csv", data_root / "processed" / "kalshi_game_market_matches.csv", "csv"),
        ("matched_markets_csv", report_path / "matched_markets.csv", "csv"),
        ("market_truth_audit_csv", report_path / "market_truth_audit.csv", "csv"),
    ]
    return _first_non_empty_source(candidates)


def _load_matches_for_coverage(report_path: Path, data_root: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        ("matched_markets_csv", report_path / "matched_markets.csv", "csv"),
        ("market_truth_audit_csv", report_path / "market_truth_audit.csv", "csv"),
        ("kalshi_game_market_matches_csv", data_root / "processed" / "kalshi_game_market_matches.csv", "csv"),
    ]
    return _first_non_empty_source(candidates)


def _load_sportsbook_for_coverage(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        ("processed_sportsbook_odds_csv", data_root / "processed" / "sportsbook_odds.csv"),
        ("raw_sportsbook_odds_csv", data_root / "raw" / "sportsbook" / "nba_moneyline_odds.csv"),
        ("raw_sportsbook_odds_legacy_csv", data_root / "raw" / "sportsbook_odds.csv"),
    ]
    attempts: list[dict[str, Any]] = []
    for label, path in candidates:
        if not path.exists():
            record = _source_record(label, path, pd.DataFrame(), "missing")
            attempts.append(record)
            continue
        try:
            frame = load_sportsbook_odds(path)
            record = _source_record(label, path, frame, "")
        except Exception as exc:
            frame = pd.DataFrame()
            record = _source_record(label, path, frame, f"{type(exc).__name__}: {exc}")
        attempts.append(record)
        if not frame.empty:
            return frame, record, attempts
    return pd.DataFrame(), attempts[-1] if attempts else {}, attempts


def _free_odds_split_mode(data_root: Path) -> str:
    split_config = _read_json(data_root / "processed" / "sportsbook_split_config.json")
    mode = str(split_config.get("free_odds_split_mode") or "latest_available")
    return mode


def _season_counts(frame: pd.DataFrame) -> dict[int, int]:
    if frame.empty:
        return {}
    working = frame.copy()
    working["_coverage_season"] = _coverage_seasons(working)
    working = working.dropna(subset=["_coverage_season"]).copy()
    if working.empty:
        return {}
    working["_coverage_season"] = working["_coverage_season"].astype(int)
    working["_game_key"] = _canonical_game_ids(working)
    return {
        int(season): int(count)
        for season, count in working.groupby("_coverage_season")["_game_key"].nunique().items()
    }


def _print_dashboard_data_diagnostics(
    data_summary: dict[str, Any],
    data_coverage: dict[str, Any],
    game_frame: pd.DataFrame,
    match_frame: pd.DataFrame,
) -> None:
    print("Dashboard data diagnostics:")
    for record in [
        *data_coverage.get("game_attempts", []),
        *data_coverage.get("sportsbook_attempts", []),
        *data_coverage.get("match_attempts", []),
    ]:
        status = "loaded" if record.get("rows") else record.get("error") or "empty"
        print(f"- {record.get('label')}: {record.get('path')}")
        print(f"  rows={record.get('rows')} status={status}")
        print(f"  columns={record.get('columns')}")

    required_game_columns = ["game_id", "game_date", "home_team_abbr", "away_team_abbr"]
    required_match_columns = ["game_id", "market_ticker"]
    missing_game = [column for column in required_game_columns if column not in game_frame.columns]
    missing_match = [column for column in required_match_columns if column not in match_frame.columns]
    print(f"- Missing NBA game columns: {missing_game if missing_game else 'none'}")
    print(f"- Missing matched market columns: {missing_match if missing_match else 'none'}")

    date_col = _date_column(game_frame)
    start, end = _safe_min_max_dates(game_frame, date_col) if date_col else ("n/a", "n/a")
    print(f"- NBA game date range: {start} to {end}")
    print(f"- NBA season counts: {_season_counts(game_frame)}")
    sportsbook_counts = {
        int(row["season_start_year"]): int(row["sportsbook_games"])
        for row in data_coverage.get("seasons", [])
    }
    print(f"- Sportsbook odds counts by season: {sportsbook_counts}")
    market_counts = {
        int(row["season_start_year"]): int(row["matched_kalshi_markets"])
        for row in data_coverage.get("seasons", [])
    }
    print(f"- Matched Kalshi market counts by season: {market_counts}")

    warnings: list[str] = []
    if int(data_summary.get("nba_games", 0) or 0) == 0 and int(data_summary.get("games_with_nba_and_market", 0) or 0) > 0:
        warnings.append("NBA games is 0 while Games with NBA and market data is positive.")
    if data_summary.get("date_start") == "n/a" and (_date_column(game_frame) or _date_column(match_frame)):
        warnings.append("Data starts is n/a while dated rows exist.")
    if not data_coverage.get("seasons") and int(data_summary.get("matched_markets", 0) or 0) > 0:
        warnings.append("Season coverage is empty while matched markets exist.")
    training_rows = [
        row for row in data_coverage.get("seasons", [])
        if row.get("dataset_split") in {"train", "strict_train", "validation", "strict_validation"}
        and int(row.get("sportsbook_games", 0) or 0) == 0
    ]
    if training_rows:
        labels = ", ".join(str(row.get("season")) for row in training_rows)
        warnings.append(f"Sportsbook odds are missing for train/validation seasons: {labels}.")
    if data_coverage.get("sportsbook_training_coverage_too_low"):
        warnings.append("Historical sportsbook coverage is too low for reliable model training.")
    if data_coverage.get("partial_validation_warning"):
        warnings.append(str(data_coverage["partial_validation_warning"]))
    for warning in warnings:
        print(f"WARNING: {warning}")


def _build_data_coverage(report_path: Path, data_root: Path) -> dict[str, Any]:
    games, game_source, game_attempts = _load_games_for_coverage(report_path, data_root)
    matches, match_source, match_attempts = _load_matches_for_coverage(report_path, data_root)
    sportsbook_odds, sportsbook_source, sportsbook_attempts = _load_sportsbook_for_coverage(data_root)
    if games.empty:
        return {
            "total_nba_games": 0,
            "total_kalshi_markets": 0,
            "total_sportsbook_games": 0,
            "games_with_market_proxy": 0,
            "overall_match_rate": None,
            "train_start_season": TRAIN_START_SEASON,
            "training_sportsbook_match_rate": None,
            "sportsbook_training_coverage_too_low": True,
            "seasons": [],
            "empty_field": "seasons",
            "expected_source": str(game_source.get("path", data_root / "interim" / "nba_games.parquet")),
            "backend_rows": 0,
            "game_source": game_source,
            "sportsbook_source": sportsbook_source,
            "match_source": match_source,
            "game_attempts": game_attempts,
            "sportsbook_attempts": sportsbook_attempts,
            "match_attempts": match_attempts,
            "coverage_note": "Historical market proxy uses free Kaggle sportsbook odds data. No paid API data is being used. Kalshi is used only for current or live market comparison where available.",
        }

    game_frame = games.copy()
    game_frame["_coverage_season"] = _coverage_seasons(game_frame)
    game_frame = game_frame.dropna(subset=["_coverage_season"]).copy()
    game_frame["_coverage_season"] = game_frame["_coverage_season"].astype(int)
    game_frame["_game_key"] = _canonical_game_ids(game_frame)
    game_counts = game_frame.groupby("_coverage_season")["_game_key"].nunique()
    sportsbook_coverage = sportsbook_coverage_by_season(game_frame, sportsbook_odds)
    if sportsbook_coverage.empty:
        sportsbook_counts = pd.Series(dtype=int)
    else:
        sportsbook_counts = sportsbook_coverage.set_index("season")["sportsbook_games"]
    split_mode = _free_odds_split_mode(data_root)
    split_plan = build_free_odds_split_plan(sportsbook_coverage, mode=split_mode)
    season_splits = {int(key): value for key, value in split_plan["season_splits"].items()}

    if matches.empty:
        market_counts = pd.Series(dtype=int)
    else:
        market_frame = matches.copy()
        if "market_ticker" in market_frame.columns:
            market_frame = market_frame[market_frame["market_ticker"].notna()]
            market_frame = market_frame[market_frame["market_ticker"].astype(str).str.len() > 0]
        if "match_status" in market_frame.columns:
            market_frame = market_frame[~market_frame["match_status"].astype(str).str.lower().eq("no_match")]
        if "game_id" in market_frame.columns and "game_id" in game_frame.columns:
            season_map = game_frame[["game_id", "_coverage_season"]].drop_duplicates("game_id")
            season_map["game_id"] = season_map["game_id"].astype(str)
            market_frame["game_id"] = market_frame["game_id"].astype(str)
            market_frame = market_frame.merge(season_map, on="game_id", how="left")
        if "_coverage_season" not in market_frame.columns or market_frame["_coverage_season"].isna().all():
            market_frame["_coverage_season"] = _coverage_seasons(market_frame)
        market_frame = market_frame.dropna(subset=["_coverage_season"]).copy()
        market_frame["_coverage_season"] = market_frame["_coverage_season"].astype(int)
        if "market_ticker" in market_frame.columns:
            market_counts = market_frame.groupby("_coverage_season")["market_ticker"].nunique()
        else:
            market_counts = market_frame.groupby("_coverage_season").size()

    seasons: list[dict[str, Any]] = []
    for season in sorted(game_counts.index.astype(int).tolist()):
        nba_games = int(game_counts.get(season, 0))
        sportsbook_games = int(sportsbook_counts.get(season, 0))
        kalshi_markets = int(market_counts.get(season, 0))
        seasons.append(
            {
                "season": nba_season_display_label(season),
                "season_start_year": int(season),
                "nba_games": nba_games,
                "sportsbook_games": sportsbook_games,
                "sportsbook_match_rate": (sportsbook_games / nba_games) if nba_games else None,
                "matched_kalshi_markets": kalshi_markets,
                "kalshi_match_rate": (kalshi_markets / nba_games) if nba_games else None,
                "kalshi_markets": kalshi_markets,
                "match_rate": (sportsbook_games / nba_games) if nba_games else None,
                "dataset_split": season_splits.get(season, _dataset_split_for_season(season)),
            }
        )

    total_nba_games = int(sum(row["nba_games"] for row in seasons))
    total_sportsbook_games = int(sum(row["sportsbook_games"] for row in seasons))
    total_kalshi_markets = int(sum(row["matched_kalshi_markets"] for row in seasons))
    games_with_market_proxy = int(
        sum(
            row["sportsbook_games"]
            if row["dataset_split"] in {"train", "strict_train", "validation", "strict_validation"}
            else row["matched_kalshi_markets"]
            for row in seasons
        )
    )
    training_rows = [row for row in seasons if row["dataset_split"] in {"train", "strict_train"}]
    training_nba_games = int(sum(row["nba_games"] for row in training_rows))
    training_sportsbook_games = int(sum(row["sportsbook_games"] for row in training_rows))
    training_sportsbook_match_rate = (
        training_sportsbook_games / training_nba_games if training_nba_games else None
    )
    coverage_too_low = (
        training_sportsbook_match_rate is not None
        and training_sportsbook_match_rate < MIN_TRAINING_SPORTSBOOK_MATCH_RATE
    )
    return {
        "total_nba_games": total_nba_games,
        "total_kalshi_markets": total_kalshi_markets,
        "total_sportsbook_games": total_sportsbook_games,
        "games_with_market_proxy": games_with_market_proxy,
        "overall_match_rate": (games_with_market_proxy / total_nba_games) if total_nba_games else None,
        "free_odds_split_mode": split_mode,
        "train_start_season": int(min(split_plan["train_seasons"])) if split_plan["train_seasons"] else None,
        "train_end_season": int(max(split_plan["train_seasons"])) if split_plan["train_seasons"] else None,
        "train_seasons": [nba_season_display_label(int(season)) for season in split_plan["train_seasons"]],
        "validation_season": (
            nba_season_display_label(int(split_plan["validation_season"]))
            if split_plan["validation_season"] is not None
            else None
        ),
        "validation_sportsbook_match_rate": split_plan["validation_match_rate"],
        "excluded_due_to_missing_odds": [
            nba_season_display_label(int(season)) for season in split_plan["excluded_due_to_missing_odds"]
        ],
        "partial_validation_warning": split_plan["partial_validation_warning"],
        "training_sportsbook_match_rate": training_sportsbook_match_rate,
        "sportsbook_training_coverage_too_low": coverage_too_low,
        "seasons": seasons,
        "empty_field": "" if seasons else "seasons",
        "expected_source": str(game_source.get("path", data_root / "interim" / "nba_games.parquet")),
        "backend_rows": int(len(games)),
        "game_source": game_source,
        "sportsbook_source": sportsbook_source,
        "match_source": match_source,
        "game_attempts": game_attempts,
        "sportsbook_attempts": sportsbook_attempts,
        "match_attempts": match_attempts,
        "coverage_note": "Historical market proxy uses free Kaggle sportsbook odds data. No paid API data is being used. Kalshi is used only for current or live market comparison where available.",
    }


def _build_simple_data_summary(report_path: Path, data_root: Path) -> dict[str, Any]:
    games, game_source, _ = _load_games_for_coverage(report_path, data_root)
    modeling = _read_parquet_if_exists(data_root / "processed" / "modeling_dataset.parquet")
    matched_markets = _read_csv(report_path / "matched_markets.csv")
    market_truth = _read_csv(report_path / "market_truth_audit.csv")
    upcoming = _read_csv(report_path / "upcoming_predictions.csv")
    player_summary = _read_json(report_path / "player_data_summary.json")
    player_feature_coverage = _read_csv(report_path / "player_feature_coverage.csv")
    player_rows = 0
    player_dir = data_root / "raw" / "nba" / "player"
    if player_dir.exists():
        for path in player_dir.glob("*.parquet"):
            try:
                player_rows += int(len(pd.read_parquet(path, columns=[])))
            except Exception:
                continue

    team_frame = games if not games.empty else modeling
    team_columns = _team_columns(team_frame)
    teams: set[str] = set()
    for column in team_columns:
        teams.update(team_frame[column].dropna().astype(str).tolist())
    date_frame = games if not games.empty else modeling
    date_column = _date_column(date_frame)
    date_start, date_end = _safe_min_max_dates(date_frame, date_column) if date_column else ("n/a", "n/a")
    game_keys = _canonical_game_ids(games) if not games.empty else pd.Series(dtype=str)
    if player_summary.get("player_feature_row_coverage", 0):
        player_feature_games = player_summary.get("modeling_rows", "Not available yet")
    elif not player_feature_coverage.empty and "non_null_rows" in player_feature_coverage.columns:
        player_feature_games = int(pd.to_numeric(player_feature_coverage["non_null_rows"], errors="coerce").max())
    else:
        player_feature_games = "Not available yet"
    both_count = int(market_truth["game_id"].nunique()) if "game_id" in market_truth.columns else int(len(market_truth))
    return {
        "nba_games": int(game_keys.nunique(dropna=True)) if not games.empty else int(len(modeling)),
        "matched_markets": int(matched_markets["market_ticker"].nunique())
        if "market_ticker" in matched_markets.columns
        else int(len(matched_markets)),
        "teams": len(teams),
        "player_records": player_summary.get("raw_player_log_rows") or (player_rows if player_rows else "Not available yet"),
        "player_feature_games": player_feature_games,
        "player_feature_coverage": player_summary.get("player_feature_row_coverage", "Not available yet"),
        "date_start": date_start,
        "date_end": date_end,
        "games_with_nba_and_market": both_count,
        "upcoming_rows": int(len(upcoming)),
        "last_data_update": _safe_file_mtime(report_path / "matched_markets.csv")
        or _safe_file_mtime(report_path / "upcoming_predictions.csv"),
        "sources": {
            "nba_games": game_source,
            "kalshi_markets_matched": str(report_path / "matched_markets.csv"),
            "teams_covered": game_source,
            "player_records": str(report_path / "player_data_summary.json"),
            "games_with_player_features": str(report_path / "player_data_summary.json"),
            "games_with_nba_and_market": str(report_path / "market_truth_audit.csv"),
            "date_range": game_source,
            "last_data_update": str(report_path / "matched_markets.csv"),
        },
    }


def _build_simple_dashboard_html(report_path: Path) -> str:
    data_root = report_path.parent if report_path.name == "reports" else report_path.parent
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    upcoming = _read_csv(report_path / "upcoming_predictions.csv")
    market_suggestions = _read_csv(report_path / "upcoming_market_suggestions.csv")
    fair_prices = _read_csv(report_path / "fair_price_signals.csv")
    backtest_trades = _read_csv(report_path / "backtest_trades.csv")
    parlay_recommendations = _read_csv(report_path / "parlay_recommendations.csv")
    market_truth_summary = _read_json(report_path / "market_truth_audit_summary.json")
    proof_summary = _read_json(report_path / "single_game_proof_summary.json")
    fair_price_summary = _read_json(report_path / "fair_price_summary.json")
    backtest_summary = _read_json(report_path / "backtest_summary.json")
    parlay_summary = _read_json(report_path / "parlay_recommendations_summary.json")
    model_performance = _read_json(report_path.parent.parent / "outputs" / "model_performance_summary.json")
    player_features = _read_csv(report_path.parent.parent / "outputs" / "player_features_by_game.csv")
    kalshi_paper_summary = _read_csv(report_path.parent.parent / "outputs" / "kalshi_paper_trade_summary.csv")
    kalshi_mapping_audit = _read_csv(report_path.parent.parent / "outputs" / "kalshi_market_mapping_audit.csv")
    kalshi_paper_diagnostics = _read_csv(report_path.parent.parent / "outputs" / "kalshi_paper_trade_diagnostics.csv")
    kalshi_strategy_selected = _read_json(report_path.parent.parent / "outputs" / "kalshi_strategy_selected.json")
    kalshi_strategy_holdout = _read_csv(report_path.parent.parent / "outputs" / "kalshi_strategy_holdout_results.csv")
    data_summary = _build_simple_data_summary(report_path, data_root)
    data_coverage = _build_data_coverage(report_path, data_root)
    diagnostic_games, _, _ = _load_games_for_coverage(report_path, data_root)
    diagnostic_matches, _, _ = _load_matches_for_coverage(report_path, data_root)
    _print_dashboard_data_diagnostics(data_summary, data_coverage, diagnostic_games, diagnostic_matches)
    kalshi_comparison_label = str(model_performance.get("kalshi_comparison_label") or "Research only")

    if not upcoming.empty and "game_date" in upcoming.columns:
        upcoming = upcoming.sort_values("game_date")
    if not market_suggestions.empty and "game_date" in market_suggestions.columns:
        market_suggestions = market_suggestions.sort_values(["game_date", "edge"], ascending=[True, False])
    if not fair_prices.empty and "final_edge" in fair_prices.columns:
        fair_prices = fair_prices.sort_values("final_edge", ascending=False)
    if not player_features.empty and "game_id" in player_features.columns:
        quality_columns = [
            column
            for column in [
                "game_id",
                "player_data_available",
                "projected_rotation_available",
                "missing_key_player_uncertainty",
                "home_missing_key_players_count",
                "away_missing_key_players_count",
            ]
            if column in player_features.columns
        ]
        quality = player_features[quality_columns].copy()
        quality["game_id"] = quality["game_id"].astype(str)
        for frame in [market_suggestions, fair_prices]:
            if not frame.empty and "game_id" in frame.columns:
                frame["game_id"] = frame["game_id"].astype(str)
                frame.drop(
                    columns=[column for column in quality_columns if column != "game_id" and column in frame.columns],
                    inplace=True,
                    errors="ignore",
                )
                merged = frame.merge(quality, on="game_id", how="left")
                frame.drop(frame.index, inplace=True)
                for column in merged.columns:
                    frame[column] = merged[column]
    for frame in [market_suggestions, fair_prices]:
        if not frame.empty:
            if "player_data_available" not in frame.columns:
                frame["player_data_available"] = "not available"
            if "projected_rotation_available" not in frame.columns:
                frame["projected_rotation_available"] = "not available"
            if "missing_key_player_uncertainty" not in frame.columns:
                frame["missing_key_player_uncertainty"] = "not available"
            frame["sportsbook_benchmark_available"] = False
            frame["kalshi_market_available"] = True
            frame["prediction_label"] = kalshi_comparison_label

    dashboard_data = {
        "generatedAt": generated_at,
        "teamColors": TEAM_COLORS,
        "upcoming": _json_records(upcoming, 80),
        "markets": _json_records(market_suggestions, 200),
        "fairPrices": _json_records(fair_prices, 300),
        "backtestTrades": _json_records(backtest_trades, 2000),
        "parlays": _json_records(parlay_recommendations, 100),
        "dataSummary": data_summary,
        "dataCoverage": data_coverage,
        "modelPerformance": model_performance,
        "kalshiPaperSummary": _json_records(kalshi_paper_summary, 20),
        "kalshiMappingAudit": _json_records(kalshi_mapping_audit, 200),
        "kalshiPaperDiagnostics": _json_records(kalshi_paper_diagnostics, 200),
        "kalshiStrategySelected": kalshi_strategy_selected,
        "kalshiStrategyHoldout": _json_records(kalshi_strategy_holdout, 20),
        "marketTruthSummary": market_truth_summary,
        "proofSummary": proof_summary,
        "fairPriceSummary": fair_price_summary,
        "backtestSummary": backtest_summary,
        "parlaySummary": parlay_summary,
    }

    dashboard_json = json.dumps(dashboard_data, default=str)
    page = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="300">
  <title>NBA Kalshi Predictions</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #667085;
      --line: #dfe3ea;
      --good: #087443;
      --warn: #b54708;
      --bad: #b42318;
      --blue: #1b63ce;
      --shadow: 0 12px 30px rgba(18, 27, 38, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.4;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar {
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px 20px 12px;
    }
    .title-row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    .subtle { color: var(--muted); font-size: 14px; }
    nav {
      display: flex;
      gap: 8px;
      margin-top: 16px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      padding: 9px 13px;
      font-weight: 650;
      cursor: pointer;
      white-space: nowrap;
    }
    .fallback-links {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .fallback-links a {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 11px;
      font-weight: 700;
      text-decoration: none;
      white-space: nowrap;
    }
    .fallback-links a.active-link {
      background: #eef4ff;
      border-color: #b2ccff;
      color: var(--blue);
    }
    .download-links {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .download-links a {
      border: 1px solid #b2ccff;
      background: #eef4ff;
      color: var(--blue);
      border-radius: 8px;
      padding: 8px 11px;
      font-weight: 700;
      text-decoration: none;
      white-space: nowrap;
    }
    .tab.active {
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 20px 44px;
    }
    .controls {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .input-group {
      display: grid;
      gap: 6px;
    }
    label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    input, select {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      min-width: 150px;
    }
    .section { display: none; }
    .section.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .game-card { display: grid; gap: 14px; }
    .game-time {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
    }
    .matchup-title { color: var(--ink); font-weight: 850; font-size: 15px; margin-bottom: 2px; }
    .game-status { display: grid; justify-items: end; gap: 4px; text-align: right; }
    .badge {
      width: 42px;
      height: 42px;
      border-radius: 50%;
      color: #fff;
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 13px;
      box-shadow: inset 0 0 0 2px rgba(255,255,255,.25);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }
    .team-odds {
      display: grid;
      gap: 10px;
    }
    .team-odds-row {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #f8fafc;
    }
    .team-odds-head {
      display: grid;
      grid-template-columns: 42px 1fr;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
      font-weight: 850;
    }
    .team-title { color: var(--ink); font-size: 16px; }
    .odds-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .odds-cell {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }
    .metric-box {
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .metric-value {
      font-size: 20px;
      font-weight: 850;
    }
    .metric-value.good { color: var(--good); }
    .metric-value.bad { color: var(--bad); }
    .pick {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
      font-weight: 800;
      background: #eef4ff;
      color: var(--blue);
    }
    .pill.good { background: #ecfdf3; color: var(--good); }
    .pill.warn { background: #fffaeb; color: var(--warn); }
    .pill.bad { background: #fef3f2; color: var(--bad); }
    .empty {
      padding: 28px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      text-align: center;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .big-number {
      font-size: 26px;
      font-weight: 900;
      margin-top: 3px;
    }
    .simple-copy {
      max-width: 760px;
      color: #344054;
      font-size: 16px;
    }
    .chart {
      width: 100%;
      height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .table-wrap {
      overflow-x: auto;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }
    th, td {
      text-align: left;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      background: #f8fafc;
    }
    .note {
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
    }
    @media (max-width: 700px) {
      h1 { font-size: 23px; }
      main { padding: 16px 12px 34px; }
      .topbar { padding-left: 12px; padding-right: 12px; }
      .grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr; }
      .odds-grid { grid-template-columns: 1fr; }
      .game-time { display: grid; }
      .game-status { justify-items: start; text-align: left; }
      .pick { align-items: flex-start; flex-direction: column; }
      input, select { width: 100%; min-width: 0; }
      .controls { align-items: stretch; }
      .input-group { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="title-row">
        <div>
          <h1>NBA Kalshi Predictions</h1>
          <div class="subtle" id="last-updated">Last updated: __GENERATED_AT__</div>
        </div>
        <div class="subtle">This page refreshes every 5 minutes.</div>
      </div>
      <nav>
        <button class="tab active" data-tab="upcoming">Upcoming Games</button>
        <button class="tab" data-tab="research">Research Picks</button>
        <button class="tab" data-tab="parlays">Parlays</button>
        <button class="tab" data-tab="backtest">Backtest</button>
        <button class="tab" data-tab="info">Model Info</button>
      </nav>
      __STATIC_NAV__
      __CSV_LINKS__
    </div>
  </header>
  <main>
    <section id="upcoming" class="section active">
      <div class="controls">
        <div>
          <h2>Upcoming Games</h2>
          <p class="subtle">Clean game cards with our odds, market odds, edge, and a conservative bet size.</p>
        </div>
        <div class="input-group">
          <label for="bankroll">Portfolio amount</label>
          <input id="bankroll" type="number" min="1" step="1" value="100">
        </div>
      </div>
      <div id="upcoming-grid" class="grid"></div>
    </section>

    <section id="research" class="section">
      <div class="controls">
        <div>
          <h2>Research Picks</h2>
          <p class="subtle">Research-only fair-price output. Approved bets remain unavailable unless single-game proof gates pass.</p>
        </div>
      </div>
      <div id="research-summary" class="summary-grid"></div>
      <p class="note" id="research-proof-note"></p>
      <div id="research-table"></div>
    </section>

    <section id="parlays" class="section">
      <div class="controls">
        <div>
          <h2>Parlays</h2>
          <p class="subtle">Two-pick ideas from different games only. Same-game parlays stay excluded until correlation is modeled.</p>
        </div>
      </div>
      <div id="parlay-status" class="card"></div>
      <div id="parlay-table"></div>
    </section>

    <section id="backtest" class="section">
      <div class="controls">
        <div>
          <h2>Backtest</h2>
          <p class="subtle">Pick a bankroll and date range to review canonical Kalshi bid/ask backtest results. Sportsbook odds remain a separate benchmark, not the canonical single-game proof source.</p>
        </div>
      </div>
      <div class="controls card">
        <div class="input-group">
          <label for="bt-bankroll">Starting bankroll</label>
          <input id="bt-bankroll" type="number" min="1" step="1" value="100">
        </div>
        <div class="input-group">
          <label for="bt-start">Start date</label>
          <input id="bt-start" type="date">
        </div>
        <div class="input-group">
          <label for="bt-end">End date</label>
          <input id="bt-end" type="date">
        </div>
        <div class="input-group">
          <label for="bt-edge">Minimum edge</label>
          <select id="bt-edge">
            <option value="0.02">2 percent</option>
            <option value="0.03">3 percent</option>
            <option value="0.05" selected>5 percent</option>
            <option value="0.07">7 percent</option>
          </select>
        </div>
      </div>
      <div id="backtest-summary" class="summary-grid"></div>
      <canvas id="bankroll-chart" class="chart" width="1000" height="260"></canvas>
      <h3>Example Bets</h3>
      <div id="backtest-table"></div>
      <p class="note" id="backtest-source-note"></p>
    </section>

    <section id="info" class="section">
      <h2>Model Info</h2>
      <p class="simple-copy">Our model estimates fair win probability. Sportsbook closing odds are used as a historical benchmark. Kalshi prices are used for live market comparison.</p>
      <div id="info-grid" class="summary-grid"></div>
      <div class="card">
        <h3>Data Coverage</h3>
        <p class="note" id="coverage-note"></p>
        <div id="coverage-summary" class="summary-grid"></div>
        <div id="coverage-table"></div>
      </div>
      <div class="card">
        <h3>Artifact Source Status</h3>
        <p class="note" id="artifact-source-warning"></p>
        <div id="artifact-source-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h3>Fair Probability Model</h3>
        <p class="note" id="model-diagnostics-warning"></p>
        <div id="model-performance-summary" class="summary-grid"></div>
        <div id="model-performance-thresholds"></div>
      </div>
      <div class="card">
        <h3>Player Availability Features</h3>
        <div id="player-feature-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h3>Model Audit</h3>
        <p class="note" id="model-audit-warning"></p>
        <div id="model-audit-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h3>Walk-Forward Validation</h3>
        <p class="note" id="walk-forward-warning"></p>
        <div id="walk-forward-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h3>Sportsbook Benchmark</h3>
        <p class="note">Closing odds are pregame odds and valid as a benchmark, but they often include late market information near tipoff. They are not treated as the same thing as an early fair prediction.</p>
      </div>
      <div class="card">
        <h3>Kalshi Market Comparison</h3>
        <p class="note">Kalshi prices are used for current/live market comparison. If the fair model has not validated historically, opportunities are labeled research only.</p>
      </div>
      <div class="card">
        <h3>Kalshi Paper Trading</h3>
        <p class="note" id="kalshi-paper-warning"></p>
        <div id="kalshi-paper-summary" class="summary-grid"></div>
        <div id="kalshi-paper-thresholds"></div>
      </div>
      <div class="card">
        <h3>Paper Trading Audit</h3>
        <p class="note" id="paper-audit-warning"></p>
        <div id="paper-audit-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h3>Strategy Filter Testing</h3>
        <p class="note" id="strategy-filter-warning"></p>
        <div id="strategy-filter-summary" class="summary-grid"></div>
      </div>
      <div class="card">
        <h3>Current betting status</h3>
        <p id="proof-note" class="simple-copy"></p>
      </div>
    </section>
  </main>

  <script>
    const DATA = __DASHBOARD_DATA__;
    const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
    const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
    const pct = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "Not available";
    const fmtMoney = value => money.format(Number(value) || 0);
    const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    const dateOnly = value => {
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? "Date not available" : d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
    };
    const dateLabel = value => {
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? "Date not available" : d.toLocaleDateString([], { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" });
    };
    const timeEt = value => {
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? "Time not available" : `${d.toLocaleTimeString([], { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" })} ET`;
    };
    const teamColor = team => DATA.teamColors[team] || "#344054";
    const proofBlocked = DATA.proofSummary && DATA.proofSummary.single_game_edge_proven === false;

    function betRule(edge, bankroll, ignoreProof=false) {
      const e = Number(edge) || 0;
      let label = "No bet";
      let pctSize = 0;
      if (ignoreProof || !proofBlocked) {
        if (e >= 0.07) { label = "Strong bet"; pctSize = 0.05; }
        else if (e >= 0.04) { label = "Medium bet"; pctSize = 0.025; }
        else if (e >= 0.02) { label = "Small bet"; pctSize = 0.01; }
      }
      return { label, amount: bankroll * pctSize, pctSize };
    }

    function plainStrength(edge) {
      const e = Number(edge) || 0;
      if (proofBlocked) return "No bet";
      if (e >= 0.07) return "Strong bet";
      if (e >= 0.04) return "Medium bet";
      if (e >= 0.02) return "Small bet";
      return "No bet";
    }

    function marketForTeam(gameId, team) {
      const rows = DATA.markets.filter(row => String(row.game_id) === String(gameId) && row.yes_team_abbr === team);
      return rows.sort((a, b) => Number(b.edge || 0) - Number(a.edge || 0))[0] || null;
    }

    function bestMarketForGame(game) {
      const rows = DATA.markets.filter(row => String(row.game_id) === String(game.game_id));
      return rows.sort((a, b) => Number(b.edge || 0) - Number(a.edge || 0))[0] || null;
    }

    function teamOddsRow(team, ourOdds, market) {
      const marketOdds = market ? Number(market.market_prob) : NaN;
      const edge = Number.isFinite(marketOdds) ? Number(ourOdds) - marketOdds : NaN;
      const edgeClass = Number(edge) > 0 ? "good" : Number(edge) < 0 ? "bad" : "";
      return `<div class="team-odds-row">
        <div class="team-odds-head">
          <span class="badge" style="background:${teamColor(team)}">${team || "NBA"}</span>
          <span class="team-title">${team || "Team"}</span>
        </div>
        <div class="odds-grid">
          <div class="odds-cell"><div class="metric-label">Our odds</div><div class="metric-value">${pct(ourOdds)}</div></div>
          <div class="odds-cell"><div class="metric-label">Market odds</div><div class="metric-value">${pct(marketOdds)}</div></div>
          <div class="odds-cell"><div class="metric-label">Edge</div><div class="metric-value ${edgeClass}">${Number.isFinite(edge) && edge > 0 ? "+" : ""}${pct(edge)}</div></div>
        </div>
      </div>`;
    }

    function gameCard(game, bankroll, isBestEdge=false) {
      const away = game.away_team_abbr;
      const home = game.home_team_abbr;
      const awayOdds = Number(game.model_away_win_prob);
      const homeOdds = Number(game.model_home_win_prob);
      const awayMarket = marketForTeam(game.game_id, away);
      const homeMarket = marketForTeam(game.game_id, home);
      const qualitySource = awayMarket || homeMarket || {};
      const awayEdge = awayMarket ? awayOdds - Number(awayMarket.market_prob) : NaN;
      const homeEdge = homeMarket ? homeOdds - Number(homeMarket.market_prob) : NaN;
      const awayCandidate = { team: away, edge: awayEdge };
      const homeCandidate = { team: home, edge: homeEdge };
      const best = [awayCandidate, homeCandidate].sort((a, b) => Number(b.edge || -999) - Number(a.edge || -999))[0];
      const pickTeam = best && Number.isFinite(best.edge) ? best.team : "";
      const edge = best ? Number(best.edge) : NaN;
      const rec = betRule(edge, bankroll);
      const pillClass = rec.label === "No bet" ? "bad" : edge >= 0.07 ? "good" : "warn";
      return `<article class="card game-card">
        <div class="game-time">
          <div>
            <div class="matchup-title">${away || "Away"} vs ${home || "Home"}</div>
            <div>${dateLabel(game.game_date)}</div>
          </div>
          <div class="game-status">
            <span>${timeEt(game.game_date)}</span>
            <span>${isBestEdge ? '<span class="pill good">Best edge today</span>' : (game.upcoming_status || "")}</span>
          </div>
        </div>
        <div class="team-odds">
          ${teamOddsRow(away, awayOdds, awayMarket)}
          ${teamOddsRow(home, homeOdds, homeMarket)}
        </div>
        <div class="note">
          Player data: ${qualitySource.player_data_available ?? "not available"}.
          Projected rotation: ${qualitySource.projected_rotation_available ?? "not available"}.
          Missing key player uncertainty: ${qualitySource.missing_key_player_uncertainty || "not available"}.
          Sportsbook benchmark: ${qualitySource.sportsbook_benchmark_available ? "yes" : "no"}.
          Kalshi market: ${qualitySource.kalshi_market_available ? "yes" : "no"}.
          Label: ${qualitySource.prediction_label || "Research only"}.
        </div>
        <div class="pick">
          <div>
            <div class="metric-label">Suggested bet</div>
            <strong>${rec.label === "No bet" ? "No bet" : `Bet ${pickTeam}`}</strong>
            ${rec.label !== "No bet" ? `<div class="note">${rec.label}</div>` : ""}
            ${proofBlocked ? '<div class="note">Bets are blocked until the model proves a repeatable single-game edge.</div>' : ""}
          </div>
          <div>
            <div class="metric-label">Bet size</div>
            <span class="pill ${pillClass}">${fmtMoney(rec.amount)}</span>
          </div>
        </div>
      </article>`;
    }

    function renderUpcoming() {
      const bankroll = Number(document.getElementById("bankroll").value) || 100;
      const now = new Date();
      const upcoming = DATA.upcoming.filter(game => {
        const d = new Date(game.game_date);
        return Number.isNaN(d.getTime()) || d >= new Date(now.getTime() - 12 * 60 * 60 * 1000);
      }).slice(0, 24);
      const bestPositiveEdge = Math.max(...upcoming.map(game => {
        const away = game.away_team_abbr;
        const home = game.home_team_abbr;
        const awayMarket = marketForTeam(game.game_id, away);
        const homeMarket = marketForTeam(game.game_id, home);
        const awayEdge = awayMarket ? Number(game.model_away_win_prob) - Number(awayMarket.market_prob) : -999;
        const homeEdge = homeMarket ? Number(game.model_home_win_prob) - Number(homeMarket.market_prob) : -999;
        return Math.max(awayEdge, homeEdge);
      }));
      document.getElementById("upcoming-grid").innerHTML = upcoming.length
        ? upcoming.map(game => {
            const away = game.away_team_abbr;
            const home = game.home_team_abbr;
            const awayMarket = marketForTeam(game.game_id, away);
            const homeMarket = marketForTeam(game.game_id, home);
            const awayEdge = awayMarket ? Number(game.model_away_win_prob) - Number(awayMarket.market_prob) : -999;
            const homeEdge = homeMarket ? Number(game.model_home_win_prob) - Number(homeMarket.market_prob) : -999;
            const gameEdge = Math.max(awayEdge, homeEdge);
            const isBest = !proofBlocked && gameEdge >= 0.07 && gameEdge === bestPositiveEdge;
            return gameCard(game, bankroll, isBest);
          }).join("")
        : '<div class="empty">No upcoming games are available yet. Run the data refresh, then rebuild the dashboard.</div>';
    }

    function renderResearchPicks() {
      const fps = DATA.fairPriceSummary || {};
      const rows = DATA.fairPrices
        .filter(row => ["approved_bet", "paper_trade_candidate", "research_lean"].includes(String(row.recommendation_tier || "")))
        .sort((a, b) => Number(b.edge || b.final_edge || 0) - Number(a.edge || a.final_edge || 0))
        .slice(0, 50);
      const proofStatus = fps.proof_status || fps.proof_gate_status || (DATA.proofSummary || {}).status || "unknown";
      const blockedReason = fps.blocked_reason || (proofBlocked ? "single_game_edge_not_proven" : "");
      const summary = [
        ["Approved bets", intFmt.format(Number(fps.approved_bets_count) || 0)],
        ["Paper candidates", intFmt.format(Number(fps.paper_trade_candidates_count) || 0)],
        ["Research leans", intFmt.format(Number(fps.research_leans_count) || 0)],
        ["Proof status", proofStatus],
        ["Blocked reason", blockedReason || "None"],
      ];
      document.getElementById("research-summary").innerHTML = summary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${esc(value)}</div></div>`
      ).join("");
      document.getElementById("research-proof-note").textContent = proofBlocked
        ? "These are research-only model-ranked picks. Actionable bets and parlays remain blocked because single-game edge is not proven."
        : "Single-game proof gates are marked proven, so approved bet tiers may appear. Parlay rules still apply separately.";
      document.getElementById("research-table").innerHTML = rows.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Tier</th><th>Market</th><th>Side</th><th>Model prob</th><th>Market implied</th><th>Edge</th><th>Price</th><th>Confidence</th><th>Reason</th><th>Main risk</th></tr></thead>
        <tbody>${rows.map(row => {
          const tier = String(row.recommendation_tier || "none");
          const side = row.research_side || row.side || row.ungated_side || "";
          const price = Number(row.research_price || row.price);
          const marketProb = row.research_market_implied_probability ?? row.market_implied_probability;
          const edge = row.edge ?? row.final_edge;
          return `<tr>
            <td><span class="pill ${tier === "approved_bet" ? "good" : tier === "paper_trade_candidate" ? "warn" : ""}">${esc(tier)}</span></td>
            <td>${esc(row.market || row.market_ticker || "")}</td>
            <td>${esc(side)}</td>
            <td>${pct(row.research_model_probability ?? row.model_prob)}</td>
            <td>${pct(marketProb)}</td>
            <td>${Number.isFinite(Number(edge)) && Number(edge) > 0 ? "+" : ""}${pct(edge)}</td>
            <td>${Number.isFinite(price) ? `${price.toFixed(1)}c` : "Not available"}</td>
            <td>${esc(row.confidence_label || row.confidence || "none")}</td>
            <td>${esc(row.main_reason || row.ungated_main_reason || "")}</td>
            <td>${esc(row.main_risk || row.blocked_reason || "")}</td>
          </tr>`;
        }).join("")}</tbody>
      </table></div>` : '<div class="empty">No research fair-price rows are available. Rebuild fair-price signals after market matching.</div>';
    }

    function backtestRows() {
      const start = document.getElementById("bt-start").value;
      const end = document.getElementById("bt-end").value;
      const minEdge = Number(document.getElementById("bt-edge").value);
      return DATA.backtestTrades.filter(row => {
        const edge = Number(row.edge);
        const price = Number(row.price_cents);
        const d = String(row.date || "").slice(0, 10);
        if (!Number.isFinite(edge) || !Number.isFinite(price) || price <= 0) return false;
        if (edge < minEdge) return false;
        if (start && d < start) return false;
        if (end && d > end) return false;
        return true;
      });
    }

    function simulateBacktest() {
      const startBankroll = Number(document.getElementById("bt-bankroll").value) || 100;
      let bankroll = startBankroll;
      const rows = backtestRows().sort((a, b) => String(a.date).localeCompare(String(b.date)));
      const curve = [{ date: "Start", bankroll }];
      const bets = [];
      let wins = 0;
      let losses = 0;
      let biggestWin = 0;
      let biggestLoss = 0;
      for (const row of rows) {
        const edge = Number(row.edge);
        const rec = betRule(edge, bankroll, true);
        if (rec.amount <= 0) continue;
        const price = Number(row.price_cents) / 100;
        const side = String(row.candidate_side || row.side || "YES").toUpperCase();
        const actualYes = String(row.actual_yes_win).toLowerCase() === "true";
        const won = side === "NO" ? !actualYes : actualYes;
        const profit = won ? rec.amount * ((1 / price) - 1) : -rec.amount;
        bankroll += profit;
        if (won) wins += 1; else losses += 1;
        biggestWin = Math.max(biggestWin, profit);
        biggestLoss = Math.min(biggestLoss, profit);
        curve.push({ date: row.date, bankroll });
        bets.push({ ...row, side, won, betSize: rec.amount, profit, bankroll });
      }
      return { startBankroll, endingBankroll: bankroll, bets, wins, losses, biggestWin, biggestLoss, curve, skipped: Math.max(0, DATA.backtestTrades.length - rows.length) };
    }

    function renderBacktest() {
      const result = simulateBacktest();
      const placed = result.bets.length;
      const profit = result.endingBankroll - result.startBankroll;
      const returnPct = result.startBankroll ? profit / result.startBankroll : 0;
      const winRate = placed ? result.wins / placed : 0;
      const summary = [
        ["Started with", fmtMoney(result.startBankroll)],
        ["Ended with", fmtMoney(result.endingBankroll)],
        ["Profit", fmtMoney(profit)],
        ["Return", pct(returnPct)],
        ["Bets placed", String(placed)],
        ["Win rate", pct(winRate)],
        ["Biggest win", fmtMoney(result.biggestWin)],
        ["Biggest loss", fmtMoney(result.biggestLoss)],
      ];
      document.getElementById("backtest-summary").innerHTML = summary.map(([label, value]) =>
        `<div class="card"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const bt = DATA.backtestSummary || {};
      const stale = bt.stale_artifacts_detected ? ` Artifact warning: ${(bt.artifact_warnings || []).join(", ")}` : "";
      document.getElementById("backtest-source-note").textContent =
        `Canonical source: ${bt.market_source || "unknown"}. Price source: ${bt.price_source || "unknown"}. ` +
        `Snapshot target order: ${bt.snapshot_target || "not recorded"}. Bid/ask required: ${bt.bid_ask_required ? "yes" : "no"}. ` +
        `NO trades allowed: ${bt.no_trades_allowed ? "yes" : "no"}.${stale}`;
      drawChart(result.curve);
      const sample = result.bets.slice(-30).reverse();
      document.getElementById("backtest-table").innerHTML = sample.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Date</th><th>Game</th><th>Pick</th><th>Our odds</th><th>Market odds</th><th>Edge</th><th>Bet size</th><th>Result</th><th>Profit</th></tr></thead>
        <tbody>${sample.map(row => `<tr>
          <td>${dateOnly(row.date)}</td>
          <td>${row.away_team_abbr || ""} at ${row.home_team_abbr || ""}</td>
          <td>${row.side} ${row.yes_team_abbr || ""}</td>
          <td>${pct(row.model_prob)}</td>
          <td>${pct(row.market_prob)}</td>
          <td>${pct(row.edge)}</td>
          <td>${fmtMoney(row.betSize)}</td>
          <td>${row.won ? "Win" : "Loss"}</td>
          <td>${fmtMoney(row.profit)}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : '<div class="empty">No bets matched those settings.</div>';
    }

    function drawChart(curve) {
      const canvas = document.getElementById("bankroll-chart");
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#dfe3ea";
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i++) {
        const y = (height / 5) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }
      if (!curve.length) return;
      const values = curve.map(point => Number(point.bankroll));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const pad = Math.max(5, (max - min) * 0.12);
      const lo = min - pad;
      const hi = max + pad;
      ctx.strokeStyle = "#1b63ce";
      ctx.lineWidth = 3;
      ctx.beginPath();
      curve.forEach((point, index) => {
        const x = curve.length === 1 ? 0 : (index / (curve.length - 1)) * width;
        const y = height - ((Number(point.bankroll) - lo) / (hi - lo || 1)) * height;
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function renderParlays() {
      const summary = DATA.parlaySummary || {};
      const status = summary.status || "not available";
      const blocked = status.startsWith("blocked") || !summary.parlay_recommendations_allowed;
      document.getElementById("parlay-status").innerHTML = `
        <div class="metric-label">Parlay status</div>
        <div class="big-number">${blocked ? "No parlays yet" : "Parlays available"}</div>
        <p class="simple-copy">${blocked
          ? "Parlays stay off until the straight-bet system proves a repeatable edge. This protects the bankroll from multiplying model errors."
          : "These are research-only two-pick ideas from different games."}</p>
        <div class="note">Same-game parlays: excluded. Risk assumption: ${summary.assumption || "different-game legs are treated as approximately independent."}</div>
      `;
      const rows = DATA.parlays || [];
      document.getElementById("parlay-table").innerHTML = rows.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Rank</th><th>Pick 1</th><th>Pick 2</th><th>Our odds</th><th>Market odds</th><th>Edge</th><th>Stake</th><th>Risk</th></tr></thead>
        <tbody>${rows.map(row => `<tr>
          <td>${row.rank || ""}</td>
          <td>${row.leg_1_pick || ""}<div class="note">${row.leg_1_game || ""}</div></td>
          <td>${row.leg_2_pick || ""}<div class="note">${row.leg_2_game || ""}</div></td>
          <td>${pct(row.combined_model_probability)}</td>
          <td>${pct(row.combined_market_probability)}</td>
          <td>${pct(row.combined_edge)}</td>
          <td>${fmtMoney(row.suggested_stake)}</td>
          <td>${row.risk || ""}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : '<div class="empty">No parlay candidates are available. The single-game proof gate is still blocking parlays.</div>';
    }

    function renderInfo() {
      const s = DATA.dataSummary || {};
      const info = [
        ["NBA games", s.nba_games],
        ["Current Kalshi markets", s.matched_markets],
        ["Teams covered", s.teams],
        ["Player stat records", s.player_records],
        ["Games with player features", typeof s.player_feature_games === "number" ? intFmt.format(s.player_feature_games) : s.player_feature_games],
        ["Data starts", s.date_start],
        ["Data ends", s.date_end],
        ["Games with live Kalshi market", s.games_with_nba_and_market],
        ["Last data update", s.last_data_update || "Not available"],
      ];
      document.getElementById("info-grid").innerHTML = info.map(([label, value]) =>
        `<div class="card"><div class="metric-label">${label}</div><div class="big-number">${value ?? "Not available"}</div></div>`
      ).join("");
      const coverage = DATA.dataCoverage || {};
      const coverageSummary = [
        ["Total NBA games", intFmt.format(Number(coverage.total_nba_games) || 0)],
        ["Historical sportsbook games", intFmt.format(Number(coverage.total_sportsbook_games) || 0)],
        ["Current Kalshi markets", intFmt.format(Number(coverage.total_kalshi_markets) || 0)],
        ["Games with market proxy", intFmt.format(Number(coverage.games_with_market_proxy) || 0)],
        ["Training sportsbook match rate", pct(coverage.training_sportsbook_match_rate)],
        ["Training seasons", (coverage.train_seasons || []).join(", ") || "None"],
        ["Validation season", coverage.validation_season || "None"],
        ["Validation sportsbook match rate", pct(coverage.validation_sportsbook_match_rate)],
        ["Excluded seasons due to missing odds", (coverage.excluded_due_to_missing_odds || []).join(", ") || "None"],
      ];
      const coverageWarnings = [];
      if (coverage.sportsbook_training_coverage_too_low) {
        coverageWarnings.push("Historical sportsbook coverage is too low for reliable model training.");
      }
      if (coverage.partial_validation_warning) {
        coverageWarnings.push(coverage.partial_validation_warning);
      }
      coverageWarnings.push(coverage.coverage_note || "Historical market proxy uses free Kaggle sportsbook odds data. No paid API data is being used. Kalshi is used only for current or live market comparison where available.");
      document.getElementById("coverage-note").textContent = coverageWarnings.join(" ");
      document.getElementById("coverage-summary").innerHTML = coverageSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const bt = DATA.backtestSummary || {};
      const fps = DATA.fairPriceSummary || {};
      const ps = DATA.parlaySummary || {};
      const artifactWarning = [];
      if (bt.market_source && bt.market_source !== "kalshi") artifactWarning.push(`Canonical backtest source is ${bt.market_source}, expected Kalshi.`);
      if (bt.price_source && bt.price_source !== "kalshi_candlesticks_bid_ask") artifactWarning.push(`Canonical price source is ${bt.price_source}, expected Kalshi bid/ask candles.`);
      if (bt.stale_artifacts_detected) artifactWarning.push(`Backtest artifact warnings: ${(bt.artifact_warnings || []).join(", ")}`);
      if (DATA.proofSummary && DATA.proofSummary.single_game_edge_proven === false) artifactWarning.push("Single-game proof is not proven, so fair-price and parlay actions remain blocked.");
      document.getElementById("artifact-source-warning").textContent = artifactWarning.join(" ");
      const artifactSummary = [
        ["Backtest source", bt.market_source || "Not recorded"],
        ["Price source", bt.price_source || "Not recorded"],
        ["Snapshot target", bt.snapshot_target || "Not recorded"],
        ["Bid/ask required", bt.bid_ask_required ? "Yes" : "No"],
        ["NO trades allowed", bt.no_trades_allowed ? "Yes" : "No"],
        ["Proof status", fps.proof_status || (DATA.proofSummary || {}).status || "Not available"],
        ["Approved bets", intFmt.format(Number(fps.approved_bets_count) || 0)],
        ["Paper candidates", intFmt.format(Number(fps.paper_trade_candidates_count) || 0)],
        ["Research leans", intFmt.format(Number(fps.research_leans_count) || 0)],
        ["Blocked reason", fps.blocked_reason || "None"],
        ["Parlay status", ps.status || "Not available"],
      ];
      document.getElementById("artifact-source-summary").innerHTML = artifactSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const seasons = coverage.seasons || [];
      document.getElementById("coverage-table").innerHTML = seasons.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Season</th><th>NBA games</th><th>Games with sportsbook odds</th><th>Sportsbook match rate</th><th>Matched Kalshi markets</th><th>Kalshi match rate</th><th>Dataset split</th></tr></thead>
        <tbody>${seasons.map(row => `<tr>
          <td>${row.season || ""}</td>
          <td>${intFmt.format(Number(row.nba_games) || 0)}</td>
          <td>${intFmt.format(Number(row.sportsbook_games) || 0)}</td>
          <td>${pct(row.sportsbook_match_rate)}</td>
          <td>${intFmt.format(Number(row.matched_kalshi_markets) || 0)}</td>
          <td>${pct(row.kalshi_match_rate)}</td>
          <td>${row.dataset_split || "Not assigned"}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty">
        No season coverage data is available.
        Backend field: ${coverage.empty_field || "dataCoverage.seasons"}.
        Expected source: ${coverage.expected_source || "NBA game or modeling dataset"}.
        Backend rows received: ${Number(coverage.backend_rows) || 0}.
      </div>`;
      const perf = DATA.modelPerformance || {};
      const modelMetrics = perf.model_metrics || {};
      const bookMetrics = perf.sportsbook_baseline_metrics || {};
      const oldMetrics = perf.old_team_only_fair_metrics || perf.old_baseline_metrics || modelMetrics || {};
      const playerMetrics = perf.team_plus_player_fair_metrics || {};
      const anchoredMetrics = perf.market_anchored_metrics || {};
      const bestMetrics = perf.best_calibrated_model_metrics || {};
      const diagnosticsWarnings = [perf.model_framing_note, perf.no_validated_edge_warning, perf.strategy_warning].filter(Boolean);
      document.getElementById("model-diagnostics-warning").textContent = diagnosticsWarnings.join(" ");
      const perfSummary = [
        ["Selected split mode", perf.selected_split_mode || "Not available"],
        ["Training seasons", (perf.training_seasons || []).join(", ") || "Not available"],
        ["Validation season", perf.validation_season || "Not available"],
        ["Training games used", intFmt.format(Number(perf.training_games_used) || 0)],
        ["Validation games used", intFmt.format(Number(perf.validation_games_used) || 0)],
        ["Sportsbook benchmark log loss", Number.isFinite(Number(bookMetrics.log_loss)) ? Number(bookMetrics.log_loss).toFixed(4) : "Not available"],
        ["Team-only fair log loss", Number.isFinite(Number(oldMetrics.log_loss)) ? Number(oldMetrics.log_loss).toFixed(4) : "Not available"],
        ["Team plus player fair log loss", Number.isFinite(Number(playerMetrics.log_loss)) ? Number(playerMetrics.log_loss).toFixed(4) : "Not available"],
        ["Market-anchored log loss", Number.isFinite(Number(anchoredMetrics.log_loss)) ? Number(anchoredMetrics.log_loss).toFixed(4) : "Not available"],
        ["Best fair model", perf.best_fair_model || perf.best_calibrated_model || "Not available"],
        ["Best calibrated log loss", Number.isFinite(Number(bestMetrics.log_loss)) ? Number(bestMetrics.log_loss).toFixed(4) : "Not available"],
        ["Model Brier score", Number.isFinite(Number(bestMetrics.brier_score)) ? Number(bestMetrics.brier_score).toFixed(4) : (Number.isFinite(Number(modelMetrics.brier_score)) ? Number(modelMetrics.brier_score).toFixed(4) : "Not available")],
        ["Sportsbook baseline Brier score", Number.isFinite(Number(bookMetrics.brier_score)) ? Number(bookMetrics.brier_score).toFixed(4) : "Not available"],
        ["Best edge threshold by validation ROI", Number.isFinite(Number(perf.best_edge_threshold_by_validation_roi)) ? pct(perf.best_edge_threshold_by_validation_roi) : "Not available"],
        ["Validation ROI at best threshold", Number.isFinite(Number(perf.best_validation_roi)) ? pct(perf.best_validation_roi) : "Not available"],
        ["Validation bets at best threshold", intFmt.format(Number(perf.best_validation_bets) || 0)],
      ];
      document.getElementById("model-performance-summary").innerHTML = perfSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const betsByThreshold = perf.validation_bets_by_threshold || {};
      const roiByThreshold = perf.validation_roi_by_threshold || {};
      const thresholdRows = Object.entries(betsByThreshold);
      document.getElementById("model-performance-thresholds").innerHTML = thresholdRows.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Edge threshold</th><th>Validation bets</th><th>Validation ROI</th></tr></thead>
        <tbody>${thresholdRows.map(([threshold, bets]) => `<tr><td>${pct(threshold)}</td><td>${intFmt.format(Number(bets) || 0)}</td><td>${pct(roiByThreshold[threshold])}</td></tr>`).join("")}</tbody>
      </table></div>` : '<div class="empty">Run scripts/train_model.py to populate model performance.</div>';
      const playerSummary = [
        ["Player feature count", intFmt.format(Number(perf.player_feature_count) || 0)],
        ["Selected player features", intFmt.format(Number(perf.selected_player_feature_count) || 0)],
        ["Player data coverage", pct(perf.player_data_coverage)],
        ["Projected rotation coverage", pct(perf.projected_rotation_coverage)],
        ["Player feature leakage check", (perf.leakage_checks || {}).player_features_use_only_prior_games ? "Passed" : "Not available"],
      ];
      document.getElementById("player-feature-summary").innerHTML = playerSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      document.getElementById("model-audit-warning").textContent = perf.player_feature_warning || "";
      const auditSummary = [
        ["Best fair model", perf.best_fair_model || "Not available"],
        ["Player features improved validation", perf.player_features_improved_validation ? "Yes" : "No"],
        ["Selected player features improved validation", perf.selected_player_features_improved_validation ? "Yes" : "No"],
        ["Best ablation model", perf.best_ablation_model || "Not available"],
        ["Best ablation log loss", Number.isFinite(Number(perf.best_ablation_log_loss)) ? Number(perf.best_ablation_log_loss).toFixed(4) : "Not available"],
        ["Worst ablation model", perf.worst_ablation_model || "Not available"],
        ["Worst ablation log loss", Number.isFinite(Number(perf.worst_ablation_log_loss)) ? Number(perf.worst_ablation_log_loss).toFixed(4) : "Not available"],
        ["Feature count used", intFmt.format(Number(perf.player_feature_count) || 0)],
      ];
      document.getElementById("model-audit-summary").innerHTML = auditSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const wf = perf.walk_forward_validation || {};
      const wfWarnings = [];
      if (!wf.champion_beats_sportsbook_benchmark) {
        wfWarnings.push("Champion fair model does not beat the sportsbook benchmark across walk-forward folds.");
      }
      if (!wf.selected_player_features_improved_walk_forward) {
        wfWarnings.push("Selected player features did not improve walk-forward performance.");
      }
      if (!wf.champion_reasonable_calibration || !wf.champion_no_extreme_probability_behavior) {
        wfWarnings.push("Champion model needs calibration or probability-range review before live use.");
      }
      document.getElementById("walk-forward-warning").textContent = wfWarnings.join(" ");
      const wfSummary = [
        ["Best fair model", wf.best_fair_model || "Not available"],
        ["Average walk-forward log loss", Number.isFinite(Number(wf.average_walk_forward_log_loss)) ? Number(wf.average_walk_forward_log_loss).toFixed(4) : "Not available"],
        ["Average walk-forward Brier score", Number.isFinite(Number(wf.average_walk_forward_brier_score)) ? Number(wf.average_walk_forward_brier_score).toFixed(4) : "Not available"],
        ["Best calibration method", wf.best_calibration_method || "Not available"],
        ["Selected player features improved", wf.selected_player_features_improved_walk_forward ? "Yes" : "No"],
        ["Beats Elo", wf.champion_beats_elo ? "Yes" : "No"],
        ["Beats sportsbook benchmark", wf.champion_beats_sportsbook_benchmark ? "Yes" : "No"],
        ["Kalshi comparison label", perf.kalshi_comparison_label || "Research only"],
      ];
      document.getElementById("walk-forward-summary").innerHTML = wfSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const paper = perf.kalshi_paper_trading || {};
      const paperRows = DATA.kalshiPaperSummary || [];
      const paperDiagnostics = DATA.kalshiPaperDiagnostics || [];
      const paperWarning = [];
      if (Number(paper.closed_trades || 0) < 50) {
        paperWarning.push("Sample size is too small to validate live Kalshi edge.");
      }
      if (perf.paper_trading_warning) {
        paperWarning.push(perf.paper_trading_warning);
      }
      if ((perf.kalshi_comparison_label || "Research only") !== "Validated edge") {
        paperWarning.push("These are paper trades only, not real betting recommendations.");
      }
      document.getElementById("kalshi-paper-warning").textContent = paperWarning.join(" ");
      const paperSummary = [
        ["Champion model", wf.best_fair_model || perf.best_fair_model || "Not available"],
        ["Paper trades", intFmt.format(Number(paper.paper_trades) || 0)],
        ["Open trades", intFmt.format(Number(paper.open_trades) || 0)],
        ["Closed trades", intFmt.format(Number(paper.closed_trades) || 0)],
        ["Best paper threshold", Number.isFinite(Number(paper.best_paper_threshold)) ? pct(paper.best_paper_threshold) : "Not available"],
        ["Paper ROI", Number.isFinite(Number(paper.paper_roi)) ? pct(paper.paper_roi) : "Not available"],
        ["Dashboard P/L method", paper.payout_method || "profit_loss_per_dollar_staked"],
      ];
      document.getElementById("kalshi-paper-summary").innerHTML = paperSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      document.getElementById("kalshi-paper-thresholds").innerHTML = paperRows.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Threshold</th><th>Paper trades</th><th>Wins</th><th>Losses</th><th>Win rate</th><th>Average edge</th><th>P/L</th><th>ROI</th><th>Open</th><th>Closed</th></tr></thead>
        <tbody>${paperRows.map(row => `<tr>
          <td>${pct(row.threshold_used)}</td>
          <td>${intFmt.format(Number(row.paper_trades) || 0)}</td>
          <td>${intFmt.format(Number(row.wins) || 0)}</td>
          <td>${intFmt.format(Number(row.losses) || 0)}</td>
          <td>${pct(row.win_rate)}</td>
          <td>${pct(row.average_edge)}</td>
          <td>${Number.isFinite(Number(row.profit_loss)) ? Number(row.profit_loss).toFixed(2) : "Not available"}</td>
          <td>${pct(row.roi)}</td>
          <td>${intFmt.format(Number(row.open_trades) || 0)}</td>
          <td>${intFmt.format(Number(row.closed_trades) || 0)}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : '<div class="empty">Run scripts/train_model.py to populate Kalshi paper trading.</div>';
      const bestEdgeBucket = paper.best_edge_bucket || "Not available";
      const worstEdgeBucket = paper.worst_edge_bucket || "Not available";
      const bestPriceBucket = paper.best_price_bucket || "Not available";
      const worstPriceBucket = paper.worst_price_bucket || "Not available";
      const auditSummary = [
        ["Mapping confidence rate", pct(paper.mapping_confidence_rate)],
        ["Pregame price rate", pct(paper.pregame_price_rate)],
        ["Excluded postgame prices", intFmt.format(Number(paper.excluded_postgame_prices) || 0)],
        ["Best edge bucket", bestEdgeBucket],
        ["Worst edge bucket", worstEdgeBucket],
        ["Best price bucket", bestPriceBucket],
        ["Worst price bucket", worstPriceBucket],
        ["Yes/No mapping audit", paper.mapping_passed_audit ? "Passed" : "Review needed"],
        ["Timing audit", paper.timing_passed_audit ? "Passed" : "Review needed"],
      ];
      document.getElementById("paper-audit-warning").textContent = perf.paper_trading_warning || "";
      document.getElementById("paper-audit-summary").innerHTML = auditSummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const strategy = DATA.kalshiStrategySelected || {};
      const selectedRule = strategy.selected_rule || {};
      const discovery = strategy.discovery || {};
      const holdout = strategy.holdout || {};
      const strategyWarning = [];
      if (strategy.warning) strategyWarning.push(strategy.warning);
      if (Number(holdout.trades || 0) < 30) strategyWarning.push("Holdout sample size is too small.");
      if (!strategy.validated_edge) strategyWarning.push("No live trades. Research-only display. Hide 0-20% longshot signals from highlighted opportunities. Require at least 7% edge to appear as a potential signal.");
      document.getElementById("strategy-filter-warning").textContent = strategyWarning.join(" ");
      const filterText = [
        `edge >= ${Number.isFinite(Number(selectedRule.min_edge)) ? pct(selectedRule.min_edge) : "n/a"}`,
        `price ${Number.isFinite(Number(selectedRule.min_contract_price)) ? pct(selectedRule.min_contract_price) : "n/a"}-${Number.isFinite(Number(selectedRule.max_contract_price)) ? pct(selectedRule.max_contract_price) : "n/a"}`,
        selectedRule.timing_filter || "timing n/a",
        selectedRule.side_filter || "side n/a",
        selectedRule.favorite_filter || "favorite n/a",
        selectedRule.home_away_filter || "home/away n/a",
      ].join(", ");
      const strategySummary = [
        ["Best discovery strategy", filterText],
        ["Discovery ROI", Number.isFinite(Number(discovery.roi)) ? pct(discovery.roi) : "Not available"],
        ["Discovery trades", intFmt.format(Number(discovery.trades) || 0)],
        ["Holdout ROI", Number.isFinite(Number(holdout.roi)) ? pct(holdout.roi) : "Not available"],
        ["Holdout trades", intFmt.format(Number(holdout.trades) || 0)],
        ["Holdout passed", strategy.validated_edge ? "Yes" : "No"],
        ["Strategy status", strategy.status || "research_only"],
      ];
      document.getElementById("strategy-filter-summary").innerHTML = strategySummary.map(([label, value]) =>
        `<div class="metric-box"><div class="metric-label">${label}</div><div class="big-number">${value}</div></div>`
      ).join("");
      const status = DATA.proofSummary.status || "not available";
      document.getElementById("proof-note").textContent = proofBlocked
        ? "Approved bets and parlays are blocked because historical proof checks have not passed. Research leans and paper trade candidates are shown for review only."
        : `Bet suggestions are enabled. Current status: ${status}.`;
    }

    function activate(tabName) {
      document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.tab === tabName));
      document.querySelectorAll(".section").forEach(section => section.classList.toggle("active", section.id === tabName));
      if (tabName === "research") renderResearchPicks();
      if (tabName === "backtest") renderBacktest();
      if (tabName === "parlays") renderParlays();
    }

    function initBacktestDates() {
      const dates = DATA.backtestTrades.map(row => String(row.date || "").slice(0, 10)).filter(Boolean).sort();
      if (dates.length) {
        document.getElementById("bt-start").value = dates[0];
        document.getElementById("bt-end").value = dates[dates.length - 1];
      }
    }

    document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => activate(tab.dataset.tab)));
    document.getElementById("bankroll").addEventListener("input", renderUpcoming);
    ["bt-bankroll", "bt-start", "bt-end", "bt-edge"].forEach(id => document.getElementById(id).addEventListener("input", renderBacktest));
    initBacktestDates();
    renderUpcoming();
    renderResearchPicks();
    renderParlays();
    renderInfo();
    renderBacktest();
  </script>
</body>
</html>"""
    return (
        page.replace("__GENERATED_AT__", html.escape(generated_at))
        .replace("__DASHBOARD_DATA__", dashboard_json)
        .replace("__STATIC_NAV__", _static_nav("dashboard.html"))
        .replace("__CSV_LINKS__", _csv_download_links(report_path))
    )


def build_dashboard_html(report_dir: str | Path) -> str:
    """Return a self-contained HTML dashboard string."""

    report_path = Path(report_dir)
    return _build_sports_market_research_dashboard_html(report_path)
    model_metrics = _read_json(report_path / "model_metrics.json")
    walk_forward_metrics = _read_json(report_path / "walk_forward_metrics.json")
    backtest_summary = _read_json(report_path / "backtest_summary.json")
    tuned_backtest_summary = _read_json(report_path / "backtest_summary_tuned.json")
    ensemble_backtest_summary = _read_json(report_path / "backtest_summary_ensemble.json")
    blend_backtest_summary = _read_json(report_path / "backtest_summary_market_blend.json")
    portfolio_summary = _read_json(report_path / "portfolio_summary.json")
    calibrated_portfolio_summary = _read_json(report_path / "portfolio_summary_calibrated.json")
    market_blend_calibrated_portfolio_summary = _read_json(
        report_path / "portfolio_summary_market_blend_calibrated.json"
    )
    consensus_portfolio_summary = _read_json(report_path / "portfolio_summary_consensus_calibrated.json")
    robust_portfolio_summary = _read_json(report_path / "portfolio_summary_robust_consensus.json")
    headline_backtest_summary = _read_json(report_path / "headline_backtest_summary.json")
    edge_calibration_summary = _read_json(report_path / "edge_calibration_summary.json")
    edge_calibration_audit_summary = _read_json(report_path / "edge_calibration_audit_summary.json")
    price_aware_calibration_summary = _read_json(report_path / "edge_calibration_price_aware_summary.json")
    price_aware_sweep_summary = _read_json(report_path / "price_aware_calibration_sweep_summary.json")
    residual_summary = _read_json(report_path / "residual_summary.json")
    residual_price_aware_best_summary = _read_json(report_path / "residual_price_aware_best_summary.json")
    market_movement_summary = _read_json(report_path / "market_movement_summary.json")
    corrected_clv_summary = _read_json(report_path / "corrected_clv_summary.json")
    corrected_clv_price_aware_best_summary = _read_json(report_path / "corrected_clv_price_aware_best_summary.json")
    market_blend_edge_calibration_summary = _read_json(report_path / "edge_calibration_summary_market_blend.json")
    market_blend_edge_audit_summary = _read_json(report_path / "edge_calibration_audit_summary_market_blend.json")
    consensus_edge_summary = _read_json(report_path / "edge_consensus_summary.json")
    robust_edge_summary = _read_json(report_path / "edge_robust_consensus_summary.json")
    consensus_stability_summary = _read_json(report_path / "signal_stability_consensus_summary.json")
    robust_stability_summary = _read_json(report_path / "signal_stability_robust_consensus_summary.json")
    strategy_readiness_summary = _read_json(report_path / "strategy_readiness_summary.json")
    signal_rule_sweep_summary = _read_json(report_path / "signal_rule_sweep_summary.json")
    signal_rule_walk_forward_summary = _read_json(report_path / "signal_rule_walk_forward_summary.json")
    parlay_correlation_summary = _read_json(report_path / "parlay_correlation_summary.json")
    home_win_ensemble_summary = _read_json(report_path / "home_win_ensemble_summary.json")
    forward_summary = _read_json(report_path / "forward_recommendations_summary.json")
    market_quality = _read_json(report_path / "market_data_quality_report.json")
    coverage_summary = _read_json(report_path / "kalshi_coverage_summary.json")
    market_blend_metrics = _read_json(report_path / "market_blend_metrics.json")
    market_type_metrics = _read_json(report_path / "market_type_model_metrics.json")
    model_tuning_summary = _read_json(report_path / "model_tuning_summary.json")
    market_review = _read_json(report_path / "kalshi_market_review_summary.json")
    data_validation = _read_json(report_path / "data_validation_summary.json")
    security_audit_summary = _read_json(report_path / "security_audit_summary.json")
    player_feature_comparison = _read_json(report_path / "player_feature_comparison.json")
    market_taxonomy_summary = _read_json(report_path / "kalshi_market_taxonomy_summary.json")
    broad_market_taxonomy_summary = _read_json(report_path / "kalshi_broad_market_taxonomy_summary.json")
    historical_series_summary = _read_json(report_path / "kalshi_historical_series_backfill_summary.json")
    market_line_coverage_summary = _read_json(report_path / "market_line_coverage_summary.json")
    multivariate_legs_summary = _read_json(report_path / "kalshi_multivariate_nba_legs_summary.json")
    underlying_leg_summary = _read_json(report_path / "underlying_nba_leg_market_summary.json")
    line_candle_summary = _read_json(report_path / "kalshi_line_candle_summary.json")
    line_market_eval_summary = _read_json(report_path / "line_market_model_eval_summary.json")

    threshold_sweep = _read_csv(report_path / "threshold_sweep.csv")
    feature_diagnostics = _read_csv(report_path / "model_feature_diagnostics.csv")
    backtest_trades = _read_csv(report_path / "backtest_trades.csv")
    tuned_backtest_trades = _read_csv(report_path / "backtest_trades_tuned.csv")
    ensemble_backtest_trades = _read_csv(report_path / "backtest_trades_ensemble.csv")
    blend_backtest_trades = _read_csv(report_path / "backtest_trades_market_blend.csv")
    portfolio_trades = _read_csv(report_path / "portfolio_trades.csv")
    portfolio_slates = _read_csv(report_path / "portfolio_slates.csv")
    calibrated_portfolio_trades = _read_csv(report_path / "portfolio_trades_calibrated.csv")
    calibrated_portfolio_slates = _read_csv(report_path / "portfolio_slates_calibrated.csv")
    market_blend_calibrated_portfolio_trades = _read_csv(
        report_path / "portfolio_trades_market_blend_calibrated.csv"
    )
    market_blend_calibrated_portfolio_slates = _read_csv(
        report_path / "portfolio_slates_market_blend_calibrated.csv"
    )
    consensus_portfolio_trades = _read_csv(report_path / "portfolio_trades_consensus_calibrated.csv")
    consensus_portfolio_slates = _read_csv(report_path / "portfolio_slates_consensus_calibrated.csv")
    robust_portfolio_trades = _read_csv(report_path / "portfolio_trades_robust_consensus.csv")
    robust_portfolio_slates = _read_csv(report_path / "portfolio_slates_robust_consensus.csv")
    edge_calibration_bins = _read_csv(report_path / "edge_calibration_bins.csv")
    edge_calibration_audit = _read_csv(report_path / "edge_calibration_audit.csv")
    price_aware_calibration_bins = _read_csv(report_path / "edge_calibration_price_aware_bins.csv")
    price_aware_sweep = _read_csv(report_path / "price_aware_calibration_sweep.csv")
    residual_by_calibrated = _read_csv(report_path / "residual_by_side_calibrated_residual.csv")
    residual_price_zone = _read_csv(report_path / "residual_by_side_price_zone.csv")
    market_movement_by_side = _read_csv(report_path / "market_movement_by_side_move.csv")
    negative_edge_signals = _read_csv(report_path / "edge_calibration_negative_edge_signals.csv")
    market_blend_edge_calibration_audit = _read_csv(report_path / "edge_calibration_audit_market_blend.csv")
    consensus_stability = _read_csv(report_path / "signal_stability_consensus.csv")
    robust_stability = _read_csv(report_path / "signal_stability_robust_consensus.csv")
    strategy_readiness = _read_csv(report_path / "strategy_readiness.csv")
    strategy_readiness_monthly = _read_csv(report_path / "strategy_readiness_monthly.csv")
    signal_rule_sweep = _read_csv(report_path / "signal_rule_sweep.csv")
    signal_rule_sweep_best_monthly = _read_csv(report_path / "signal_rule_sweep_best_monthly.csv")
    signal_rule_walk_forward_folds = _read_csv(report_path / "signal_rule_walk_forward_folds.csv")
    signal_rule_walk_forward_monthly = _read_csv(report_path / "signal_rule_walk_forward_monthly.csv")
    parlay_correlation_report = _read_csv(report_path / "parlay_correlation_report.csv")
    parlay_pair_rows = _read_csv(report_path / "parlay_pair_rows.csv")
    home_win_ensemble_weights = _read_csv(report_path / "home_win_ensemble_weights.csv")
    home_win_ensemble_static_audit = _read_csv(report_path / "home_win_ensemble_static_audit.csv")
    forward_recommendations = _read_csv(report_path / "forward_recommendations.csv")
    model_tuning_results = _read_csv(report_path / "model_tuning_results.csv")
    suggestions = _read_csv(report_path / "paper_trade_suggestions.csv")
    probability_bins = _read_csv(report_path / "prediction_probability_bins.csv")
    season_summary = _read_csv(report_path / "prediction_season_summary.csv")
    edge_bins = _read_csv(report_path / "backtest_edge_bins.csv")
    top_trades = _read_csv(report_path / "top_backtest_trades.csv")
    market_type_calibration = _read_csv(report_path / "market_type_probability_calibration.csv")
    market_line_coverage = _read_csv(report_path / "market_line_coverage.csv")
    coverage_monthly = _read_csv(report_path / "kalshi_coverage_by_month.csv")
    gap_report = _read_csv(report_path / "kalshi_unmatched_market_gap_report.csv")
    validation_issues = _read_csv(report_path / "data_validation_issues.csv")
    security_audit_findings = _read_csv(report_path / "security_audit_findings.csv")
    data_root = report_path.parent if report_path.name == "reports" else report_path.parent
    market_taxonomy = _read_csv(data_root / "processed" / "kalshi_market_taxonomy.csv")
    broad_market_taxonomy = _read_csv(data_root / "processed" / "kalshi_broad_market_taxonomy.csv")
    multivariate_nba_legs = _read_csv(data_root / "processed" / "kalshi_multivariate_nba_legs.csv")
    underlying_leg_markets = _read_csv(data_root / "raw" / "kalshi" / "underlying_nba_leg_markets.csv")
    underlying_leg_requests = _read_csv(report_path / "underlying_nba_leg_market_requests.csv")
    line_pregame_prices = _read_csv(data_root / "processed" / "kalshi_line_pregame_prices.csv")
    line_market_eval = _read_csv(report_path / "line_market_model_eval.csv")

    walk_model = walk_forward_metrics.get("overall", {}).get("model", {})
    walk_elo = walk_forward_metrics.get("overall", {}).get("elo_baseline", {})
    best_model = model_metrics.get("best_model", "n/a")
    best_metrics = model_metrics.get("models", {}).get(best_model, {})
    blend_metrics = market_blend_metrics.get("market_blend", {})
    market_metrics = market_blend_metrics.get("market", {})
    spread_metrics = market_type_metrics.get("overall", {}).get("margin", {})
    total_metrics = market_type_metrics.get("overall", {}).get("total", {})
    tuned_metrics = model_tuning_summary.get("best_overall", {})
    model_trade_timeline = _timeline_from_summary(backtest_summary, backtest_trades)
    tuned_trade_timeline = _timeline_from_summary(tuned_backtest_summary, tuned_backtest_trades)
    ensemble_trade_timeline = _timeline_from_summary(ensemble_backtest_summary, ensemble_backtest_trades)
    blend_trade_timeline = _timeline_from_summary(blend_backtest_summary, blend_backtest_trades)
    portfolio_trade_timeline = _timeline_from_summary(portfolio_summary, portfolio_trades)
    calibrated_portfolio_trade_timeline = _timeline_from_summary(
        calibrated_portfolio_summary,
        calibrated_portfolio_trades,
    )
    market_blend_calibrated_portfolio_timeline = _timeline_from_summary(
        market_blend_calibrated_portfolio_summary,
        market_blend_calibrated_portfolio_trades,
    )
    consensus_portfolio_timeline = _timeline_from_summary(consensus_portfolio_summary, consensus_portfolio_trades)
    robust_portfolio_timeline = _timeline_from_summary(robust_portfolio_summary, robust_portfolio_trades)
    headline_trade_timeline = headline_backtest_summary.get("trade_timeline", "n/a")
    audit_summary = market_review.get("audit_summary", {})
    gap_decision = market_review.get("gap_decision", {})
    player_delta = player_feature_comparison.get("walk_forward", {}).get(
        "player_minus_team_only",
        player_feature_comparison.get("single_split", {}).get(
            "player_minus_team_only",
            player_feature_comparison.get("player_minus_team_only", {}),
        ),
    )
    broad_category_counts = broad_market_taxonomy_summary.get("category_counts", {})
    historical_series_table = pd.DataFrame(historical_series_summary.get("cached_by_series", []))
    broad_non_winner_rows = sum(
        int(value)
        for category, value in broad_category_counts.items()
        if str(category) != "game_winner"
    )

    metric_html = "".join(
        [
            _metric("Kalshi Games", str(coverage_summary.get("games_with_kalshi_markets", "n/a")), "Markets found"),
            _metric(
                "Pregame Prices",
                str(coverage_summary.get("games_with_usable_pregame_price", "n/a")),
                "Usable candle snapshots",
            ),
            _metric("Auto Matches", str(coverage_summary.get("auto_matched_games", "n/a")), "High confidence"),
            _metric(
                "Market Dates",
                f"{coverage_summary.get('market_date_min', 'n/a')} to {coverage_summary.get('market_date_max', 'n/a')}",
                "Kalshi data range",
            ),
            _metric("Walk-Forward AUC", _fmt_number(walk_model.get("roc_auc"), 3), "Out-of-sample"),
            _metric("Blend Accuracy", _fmt_pct(blend_metrics.get("accuracy")), "Model + market"),
            _metric("Market Log Loss", _fmt_number(market_metrics.get("log_loss"), 3), "Market baseline"),
            _metric(
                "Headline Bankroll",
                _fmt_money(headline_backtest_summary.get("ending_bankroll")),
                headline_backtest_summary.get("headline_label", "Slate-settled"),
            ),
        ]
    )

    model_metric_html = "".join(
        [
            _metric("Best Split Model", str(best_model), "Single test split"),
            _metric("Split Accuracy", _fmt_pct(best_metrics.get("accuracy")), "Configured test season"),
            _metric("Split Log Loss", _fmt_number(best_metrics.get("log_loss"), 3), "Lower is better"),
            _metric("Walk Model AUC", _fmt_number(walk_model.get("roc_auc"), 3), "Out-of-sample"),
            _metric("Walk Elo AUC", _fmt_number(walk_elo.get("roc_auc"), 3), "Baseline"),
            _metric("Predictions", str(walk_forward_metrics.get("num_predictions", "n/a")), "Walk-forward rows"),
            _metric("Market AUC", _fmt_number(market_metrics.get("roc_auc"), 3), "60-minute price"),
            _metric("Blend AUC", _fmt_number(blend_metrics.get("roc_auc"), 3), "Expanding blend"),
            _metric("Blend Rows", str(market_blend_metrics.get("rows", "n/a")), "Matched games"),
            _metric(
                "Player Log Loss Delta",
                _fmt_number(player_delta.get("log_loss"), 4),
                "Walk-forward player minus team-only",
            ),
            _metric(
                "Player AUC Delta",
                _fmt_number(player_delta.get("roc_auc"), 4),
                "Walk-forward player minus team-only",
            ),
            _metric("Spread MAE", _fmt_number(spread_metrics.get("mae"), 2), "Predicted margin"),
            _metric("Total MAE", _fmt_number(total_metrics.get("mae"), 2), "Predicted total points"),
            _metric("Spread/Total Rows", str(market_type_metrics.get("num_predictions", "n/a")), "Walk-forward"),
            _metric(
                "Tuned Log Loss",
                _fmt_number(tuned_metrics.get("log_loss"), 4),
                f"{model_tuning_summary.get('best_model_name', 'n/a')} / {model_tuning_summary.get('best_feature_set', 'n/a')}",
            ),
            _metric(
                "Ensemble Log Loss",
                _fmt_number(home_win_ensemble_summary.get("ensemble", {}).get("log_loss"), 4),
                home_win_ensemble_summary.get("adoption_status", "n/a"),
            ),
            _metric(
                "Ensemble Delta",
                _fmt_number(home_win_ensemble_summary.get("log_loss_delta_vs_best_component"), 4),
                f"vs {home_win_ensemble_summary.get('best_component', 'n/a')}",
            ),
        ]
    )

    quality_metric_html = "".join(
        [
            _metric("Matched Rows", str(market_quality.get("matched_rows", "n/a")), "Prediction matches"),
            _metric(
                "Missing Prices",
                str(market_quality.get("price_quality", {}).get("missing_price_count", "n/a")),
                "Needs manual fill",
            ),
            _metric(
                "Close-Price Fallback",
                str(market_quality.get("price_quality", {}).get("close_price_only_count", "n/a")),
                "Use with care",
            ),
            _metric(
                "Avg Spread",
                _fmt_number(market_quality.get("spread_quality", {}).get("average_spread_cents"), 2),
                "Cents",
            ),
            _metric("Kalshi Games", str(coverage_summary.get("games_with_kalshi_markets", "n/a")), "Found"),
            _metric("Auto Matches", str(coverage_summary.get("auto_matched_games", "n/a")), "Backtest eligible"),
            _metric(
                "Usable Prices",
                str(coverage_summary.get("games_with_usable_pregame_price", "n/a")),
                "Pregame candles",
            ),
            _metric("Audit Ticker Fails", str(audit_summary.get("ticker_failures", "n/a")), "50-row sample"),
            _metric("YES Team Fails", str(audit_summary.get("invalid_yes_team_rows", "n/a")), "50-row sample"),
            _metric(
                "Gap Games",
                str(gap_decision.get("total_gap_games", "n/a")),
                "Unmatched market review",
            ),
            _metric(
                "Validation",
                str(data_validation.get("validation_status", "n/a")),
                "Saved artifact checks",
            ),
            _metric(
                "Security Audit",
                str(security_audit_summary.get("status", "n/a")),
                f"{security_audit_summary.get('findings', 'n/a')} findings",
            ),
            _metric(
                "Strict 60m Rows",
                str(data_validation.get("prices", {}).get("strict_eligible_60m_rows", "n/a")),
                "Bid/ask, volume, <=60m candles",
            ),
            _metric(
                "Taxonomy Rows",
                str(market_taxonomy_summary.get("unique_markets", market_taxonomy_summary.get("market_rows", "n/a"))),
                "Classified market types",
            ),
            _metric(
                "Broad NBA Rows",
                str(
                    broad_market_taxonomy_summary.get(
                        "unique_markets",
                        broad_market_taxonomy_summary.get("market_rows", "n/a"),
                    )
                ),
                "All-series discovery",
            ),
            _metric("Broad Non-Winners", str(broad_non_winner_rows), "Spreads, totals, props, series"),
            _metric(
                "Historical Series Markets",
                str(historical_series_summary.get("cached_unique_markets", "n/a")),
                "Archived NBA series crawl",
            ),
            _metric(
                "Historical Winners",
                str(
                    next(
                        (
                            row.get("unique_markets", "n/a")
                            for row in historical_series_summary.get("cached_by_series", [])
                            if row.get("series_ticker") == "KXNBAGAME"
                        ),
                        "n/a",
                    )
                ),
                "KXNBAGAME archive",
            ),
            _metric(
                "Combo NBA Legs",
                str(multivariate_legs_summary.get("unique_legs", "n/a")),
                "Inventory only",
            ),
            _metric(
                "Combo Line Legs",
                str(multivariate_legs_summary.get("unique_spread_total_legs", "n/a")),
                "Not single-leg prices",
            ),
            _metric(
                "Direct Leg Markets",
                str(underlying_leg_summary.get("cached_unique_markets", "n/a")),
                "Fetched single-leg rows",
            ),
            _metric(
                "Direct Fetch Status",
                "stopped" if underlying_leg_summary.get("stopped_after_consecutive_failures") else "ready",
                f"{underlying_leg_summary.get('attempted_tickers', 'n/a')} requests",
            ),
            _metric(
                "Line 60m Prices",
                str(line_candle_summary.get("usable_60m_rows", "n/a")),
                f"{line_candle_summary.get('candidate_markets', 'n/a')} spread/total markets",
            ),
            _metric(
                "Line Eval",
                str(line_market_eval_summary.get("status", "n/a")),
                f"{line_market_eval_summary.get('rows', 'n/a')} rows",
            ),
            _metric(
                "Line Edge Signals",
                str(line_market_eval_summary.get("signals", "n/a")),
                line_market_eval_summary.get("timeline", "n/a"),
            ),
            _metric(
                "Spread Line Ready",
                str(market_line_coverage_summary.get("spread_ready", False)),
                "Real Kalshi lines",
            ),
            _metric(
                "Total Line Ready",
                str(market_line_coverage_summary.get("total_ready", False)),
                "Real Kalshi lines",
            ),
        ]
    )

    forward_metric_html = "".join(
        [
            _metric("Forward Games", str(forward_summary.get("games", "n/a")), forward_summary.get("timeline", "n/a")),
            _metric(
                "Games With Kalshi Odds",
                str(forward_summary.get("games_with_kalshi_odds", "n/a")),
                "Saved public odds",
            ),
            _metric("Edge Signals", str(forward_summary.get("edge_signals", "n/a")), "Raw edge rule"),
            _metric(
                "Recommended Paper Bets",
                str(forward_summary.get("paper_bets", "n/a")),
                forward_summary.get("readiness_gate", "n/a"),
            ),
            _metric(
                "Paper Risk",
                _fmt_money(forward_summary.get("paper_amount_risked")),
                "Recommended stake",
            ),
            _metric(
                "Hypothetical Edge Bets",
                str(forward_summary.get("hypothetical_paper_bets", "n/a")),
                _fmt_money(forward_summary.get("hypothetical_amount_risked")),
            ),
            _metric(
                "Validated Rule Passes",
                str(forward_summary.get("best_sweep_rule_passes", "n/a")),
                forward_summary.get("rule_validation_status", forward_summary.get("best_sweep_rule_status", "n/a")),
            ),
        ]
    )

    price_sources = pd.DataFrame(
        [
            {"source": key, "rows": value}
            for key, value in market_quality.get("price_source_counts", {}).items()
        ]
    )

    dashboard_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_dir": str(report_path),
    }

    template = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA Kalshi Predictor Dashboard</title>
  <style>
    :root {
      --bg: #090b0f;
      --surface: #0f131a;
      --panel: #121720;
      --panel-2: #171d27;
      --text: #eef2f7;
      --muted: #8f9aaa;
      --line: #26303d;
      --accent: #7dd3fc;
      --green: #34d399;
      --amber: #fbbf24;
      --red: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 15px;
      letter-spacing: 0;
    }
    header {
      background: rgba(9, 11, 15, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .title-row {
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    h1 {
      font-size: 24px;
      margin: 0 0 4px;
      font-weight: 700;
    }
    .subtle { color: var(--muted); font-size: 13px; }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    .tab-button {
      border: 1px solid var(--line);
      background: #0c1016;
      border-radius: 8px;
      padding: 8px 11px;
      cursor: pointer;
      color: var(--text);
      font-weight: 600;
    }
    .tab-button:hover {
      border-color: #3a4655;
      background: #131923;
    }
    .tab-button.active {
      background: #172333;
      border-color: var(--accent);
      color: #e0f2fe;
    }
    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 20px;
    }
    .tab { display: none; }
    .tab.active { display: block; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      min-height: 96px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .metric-value {
      font-size: 24px;
      font-weight: 700;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }
    .metric-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .small-note {
      color: var(--muted);
      font-size: 12px;
      margin: -4px 0 12px;
      line-height: 1.45;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
      overflow: hidden;
    }
    h2 {
      font-size: 19px;
      margin: 0 0 12px;
    }
    h3 {
      font-size: 16px;
      margin: 0 0 12px;
    }
    .plot {
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f131a;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      max-height: 520px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background: var(--panel);
    }
    th, td {
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    th {
      background: #171d27;
      font-size: 12px;
      color: #c8d2df;
      position: sticky;
      top: 0;
    }
    tr:hover td { background: #171d27; }
    .status {
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      margin-bottom: 14px;
    }
    .status.good {
      background: #082018;
      border-color: #1f6f4f;
      color: #bbf7d0;
    }
    .status.warn {
      background: #241806;
      border-color: #7c4f12;
      color: #fde68a;
    }
    .status ul { margin: 0; padding-left: 18px; }
    .empty {
      color: var(--muted);
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #0c1016;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    input[type="search"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 11px;
      min-width: 240px;
      font-size: 14px;
      background: #0c1016;
      color: var(--text);
    }
    pre {
      color: #cbd5e1;
      background: #0c1016;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
    }
    @media (max-width: 720px) {
      main { padding: 14px; }
      header { padding: 14px; }
      h1 { font-size: 21px; }
      .metric-value { font-size: 20px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="title-row">
      <div>
        <h1>NBA Kalshi Predictor Dashboard</h1>
        <div class="subtle">Generated $generated_at from $report_dir</div>
      </div>
      <div class="subtle">Research and paper trading only</div>
    </div>
    <nav class="tabs" aria-label="Dashboard tabs">
      <button class="tab-button active" data-tab="overview">Overview</button>
      <button class="tab-button" data-tab="forward">Forward</button>
      <button class="tab-button" data-tab="model">Model</button>
      <button class="tab-button" data-tab="backtest">Backtest</button>
      <button class="tab-button" data-tab="quality">Market Quality</button>
      <button class="tab-button" data-tab="tables">Tables</button>
    </nav>
  </header>
  <main>
    <section id="overview" class="tab active">
      <div class="metric-grid">$metric_html</div>
      $quality_list
      <div class="grid">
        $equity_plot
        $threshold_plot
      </div>
    </section>

    <section id="forward" class="tab">
      <div class="metric-grid">$forward_metric_html</div>
      <section class="panel">
        <div class="toolbar">
          <h3>Forward Recommendations</h3>
          <input type="search" data-filter-table="forward-table" placeholder="Filter games">
        </div>
        $forward_recommendations_table
      </section>
    </section>

    <section id="model" class="tab">
      <div class="metric-grid">$model_metric_html</div>
      <div class="grid">
        $walk_calibration_plot
        $walk_probability_plot
        $probability_bin_plot
        $season_summary_plot
      </div>
      <section class="panel">
        <h3>Walk-Forward Folds</h3>
        $fold_table
      </section>
      <section class="panel">
        <h3>Probability Buckets</h3>
        $probability_bin_table
      </section>
      <section class="panel">
        <h3>Season Diagnostics</h3>
        $season_summary_table
      </section>
      <section class="panel">
        <h3>Top Model Features</h3>
        $feature_table
      </section>
      <section class="panel">
        <h3>Walk-Forward Model Tuning</h3>
        $model_tuning_table
      </section>
      <section class="panel">
        <h3>Home-Win Ensemble</h3>
        <p class="small-note">$home_win_ensemble_note</p>
        $home_win_ensemble_weights_table
      </section>
      <section class="panel">
        <h3>Fixed-Weight Ensemble Audit</h3>
        $home_win_ensemble_static_audit_table
      </section>
      <section class="panel">
        <h3>Market-Aware Probability Comparison</h3>
        $market_blend_table
      </section>
      <section class="panel">
        <h3>Spread and Total Calibration</h3>
        $market_type_calibration_table
      </section>
      <section class="panel">
        <h3>Player Feature Comparison</h3>
        <pre>$player_feature_comparison_json</pre>
      </section>
    </section>

    <section id="backtest" class="tab">
      <div class="metric-grid">$backtest_metric_html</div>
      <div class="grid">
        $edge_plot
        $edge_bin_profit_plot
        $threshold_plot_2
      </div>
      <section class="panel">
        <h3>Headline Slate Backtest</h3>
        <p class="small-note">$headline_backtest_note</p>
        $headline_backtest_table
      </section>
      <section class="panel">
        <h3>Portfolio Comparison</h3>
        $portfolio_comparison_table
      </section>
      <section class="panel">
        <h3>Strategy Readiness</h3>
        $strategy_readiness_table
      </section>
      <section class="panel">
        <h3>Signal Rule Sweep</h3>
        <p class="small-note">$signal_rule_sweep_note</p>
        $signal_rule_sweep_table
      </section>
      <section class="panel">
        <h3>Best Rule Monthly Stability</h3>
        $signal_rule_sweep_best_monthly_table
      </section>
      <section class="panel">
        <h3>Walk-Forward Rule Validation</h3>
        <p class="small-note">$signal_rule_walk_forward_note</p>
        $signal_rule_walk_forward_folds_table
        $signal_rule_walk_forward_monthly_table
      </section>
      <section class="panel">
        <h3>Parlay Correlation Research</h3>
        <p class="small-note">$parlay_correlation_note</p>
        $parlay_correlation_table
      </section>
      <section class="panel">
        <h3>Edge Bucket Results</h3>
        $edge_bin_table
      </section>
      <section class="panel">
        <h3>Largest Paper P/L</h3>
        $top_trades_table
      </section>
      <section class="panel">
        <h3>Optimized Individual Slate</h3>
        $portfolio_trades_table
      </section>
      <section class="panel">
        <h3>Calibrated Individual Slate</h3>
        $calibrated_portfolio_trades_table
      </section>
      <section class="panel">
        <h3>Market-Blend Calibrated Slate</h3>
        $market_blend_calibrated_portfolio_trades_table
      </section>
      <section class="panel">
        <h3>Consensus Calibrated Slate</h3>
        $consensus_portfolio_trades_table
      </section>
      <section class="panel">
        <h3>Robust Consensus Slate</h3>
        $robust_portfolio_trades_table
      </section>
      <section class="panel">
        <h3>Daily Slate Risk</h3>
        $portfolio_slates_table
      </section>
      <section class="panel">
        <h3>Calibrated Daily Slate Risk</h3>
        $calibrated_portfolio_slates_table
      </section>
      <section class="panel">
        <h3>Market-Blend Calibrated Daily Slate Risk</h3>
        $market_blend_calibrated_portfolio_slates_table
      </section>
      <section class="panel">
        <h3>Consensus Daily Slate Risk</h3>
        $consensus_portfolio_slates_table
      </section>
      <section class="panel">
        <h3>Robust Consensus Daily Slate Risk</h3>
        $robust_portfolio_slates_table
      </section>
      <section class="panel">
        <h3>Edge Calibration</h3>
        $edge_calibration_table
      </section>
      <section class="panel">
        <h3>Edge Calibration Audit</h3>
        $edge_calibration_audit_table
      </section>
      <section class="panel">
        <h3>Price-Aware Calibration</h3>
        $price_aware_calibration_table
      </section>
      <section class="panel">
        <h3>Price-Aware Calibration Sweep</h3>
        $price_aware_sweep_table
      </section>
      <section class="panel">
        <h3>Residual Calibration Audit</h3>
        $residual_by_calibrated_table
      </section>
      <section class="panel">
        <h3>Residual By Price Zone</h3>
        $residual_price_zone_table
      </section>
      <section class="panel">
        <h3>Market Movement Attribution</h3>
        $market_movement_by_side_table
      </section>
      <section class="panel">
        <h3>Negative Raw-Edge Calibrated Signals</h3>
        $negative_edge_signals_table
      </section>
      <section class="panel">
        <h3>Signal Stability By Month</h3>
        $consensus_stability_table
      </section>
      <section class="panel">
        <h3>Robust Signal Stability By Month</h3>
        $robust_stability_table
      </section>
      <section class="panel">
        <h3>All Strategy Monthly Stability</h3>
        $strategy_readiness_monthly_table
      </section>
      <section class="panel">
        <h3>Market-Blend Edge Calibration Audit</h3>
        $market_blend_edge_calibration_audit_table
      </section>
      <section class="panel">
        <h3>Threshold Sweep</h3>
        $threshold_table
      </section>
    </section>

    <section id="quality" class="tab">
      <div class="metric-grid">$quality_metric_html</div>
      $quality_list_2
      <div class="grid">
        <section class="panel">
          <h3>Price Sources</h3>
          $price_source_table
        </section>
        <section class="panel">
          <h3>Market Quality JSON</h3>
          <pre>$quality_json</pre>
        </section>
      </div>
      <section class="panel">
        <h3>Kalshi Coverage by Month</h3>
        $coverage_monthly_table
      </section>
      <section class="panel">
        <h3>Unmatched Market Gap Reasons</h3>
        $gap_reason_table
      </section>
      <section class="panel">
        <h3>Market Review Decision</h3>
        <pre>$market_review_json</pre>
      </section>
      <section class="panel">
        <h3>Data Validation Summary</h3>
        <pre>$data_validation_json</pre>
      </section>
      <section class="panel">
        <h3>Data Validation Issues</h3>
        $validation_issues_table
      </section>
      <section class="panel">
        <h3>Security Audit</h3>
        <pre>$security_audit_json</pre>
        $security_audit_table
      </section>
      <section class="panel">
        <h3>NBA Market Taxonomy</h3>
        <pre>$market_taxonomy_json</pre>
        $market_taxonomy_table
      </section>
      <section class="panel">
        <h3>Broad NBA Market Discovery</h3>
        <pre>$broad_market_taxonomy_json</pre>
        $broad_market_taxonomy_table
      </section>
      <section class="panel">
        <h3>Historical NBA Series Crawl</h3>
        <pre>$historical_series_json</pre>
        $historical_series_table
      </section>
      <section class="panel">
        <h3>Market Line Extraction Audit</h3>
        <pre>$market_line_coverage_json</pre>
        $market_line_coverage_table
      </section>
      <section class="panel">
        <h3>Multivariate NBA Leg Inventory</h3>
        <pre>$multivariate_nba_legs_json</pre>
        $multivariate_nba_legs_table
      </section>
      <section class="panel">
        <h3>Direct Underlying Leg Fetch</h3>
        <pre>$underlying_leg_json</pre>
        $underlying_leg_markets_table
        $underlying_leg_requests_table
      </section>
      <section class="panel">
        <h3>Line Market Candles</h3>
        <pre>$line_candle_json</pre>
        $line_pregame_prices_table
      </section>
      <section class="panel">
        <h3>Exploratory Spread/Total Evaluation</h3>
        <pre>$line_market_eval_json</pre>
        $line_market_eval_table
      </section>
      <section class="panel">
        <h3>Unmatched Market Gap Details</h3>
        $gap_detail_table
      </section>
    </section>

    <section id="tables" class="tab">
      <section class="panel">
        <div class="toolbar">
          <h3>Paper Trade Suggestions</h3>
          <input type="search" data-filter-table="suggestions-table" placeholder="Filter suggestions">
        </div>
        $suggestions_table
      </section>
      <section class="panel">
        <div class="toolbar">
          <h3>Backtest Trades</h3>
          <input type="search" data-filter-table="trades-table" placeholder="Filter trades">
        </div>
        $trades_table
      </section>
      <section class="panel">
        <h3>Market-Blend Backtest Trades</h3>
        $blend_trades_table
      </section>
    </section>
  </main>
  <script>
    const dashboardData = $dashboard_data;
    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
      });
    });
    document.querySelectorAll("input[data-filter-table]").forEach((input) => {
      input.addEventListener("input", () => {
        const table = document.getElementById(input.dataset.filterTable);
        if (!table) return;
        const query = input.value.toLowerCase();
        table.querySelectorAll("tbody tr").forEach((row) => {
          row.style.display = row.innerText.toLowerCase().includes(query) ? "" : "none";
        });
      });
    });
  </script>
</body>
</html>"""
    )

    backtest_metric_html = "".join(
        [
            _metric("Starting Bankroll", _fmt_money(backtest_summary.get("starting_bankroll")), ""),
            _metric(
                "Headline Bankroll",
                _fmt_money(headline_backtest_summary.get("ending_bankroll")),
                headline_backtest_summary.get("headline_label", "Slate-settled"),
            ),
            _metric(
                "Headline Trades",
                str(headline_backtest_summary.get("num_selected_trades", "n/a")),
                headline_trade_timeline,
            ),
            _metric(
                "Headline Readiness",
                str(headline_backtest_summary.get("readiness_status", "n/a")),
                "Slate-settled default",
            ),
            _metric(
                "Parlays Blocked",
                str(headline_backtest_summary.get("parlays_blocked", True)),
                "Until readiness and pair economics pass",
            ),
            _metric("Model Bankroll", _fmt_money(backtest_summary.get("ending_bankroll")), "Pure NBA model"),
            _metric("Tuned Bankroll", _fmt_money(tuned_backtest_summary.get("ending_bankroll")), "Tuned probability model"),
            _metric("Ensemble Bankroll", _fmt_money(ensemble_backtest_summary.get("ending_bankroll")), "Research-only ensemble"),
            _metric("Blend Bankroll", _fmt_money(blend_backtest_summary.get("ending_bankroll")), "Model + market"),
            _metric("Model Return", _fmt_pct(backtest_summary.get("total_return_pct")), ""),
            _metric("Blend Return", _fmt_pct(blend_backtest_summary.get("total_return_pct")), ""),
            _metric("Blend Trades", str(blend_backtest_summary.get("num_trades", "n/a")), blend_trade_timeline),
            _metric("Model Trades", str(backtest_summary.get("num_trades", "n/a")), model_trade_timeline),
            _metric("Tuned Trades", str(tuned_backtest_summary.get("num_trades", "n/a")), tuned_trade_timeline),
            _metric("Ensemble Trades", str(ensemble_backtest_summary.get("num_trades", "n/a")), ensemble_trade_timeline),
            _metric(
                "Portfolio Trades",
                str(portfolio_summary.get("num_selected_trades", "n/a")),
                portfolio_trade_timeline,
            ),
            _metric(
                "Calibrated Portfolio",
                str(calibrated_portfolio_summary.get("num_selected_trades", "n/a")),
                calibrated_portfolio_trade_timeline,
            ),
            _metric(
                "Blend-Calibrated Portfolio",
                str(market_blend_calibrated_portfolio_summary.get("num_selected_trades", "n/a")),
                market_blend_calibrated_portfolio_timeline,
            ),
            _metric(
                "Consensus Portfolio",
                str(consensus_portfolio_summary.get("num_selected_trades", "n/a")),
                consensus_portfolio_timeline,
            ),
            _metric(
                "Robust Consensus",
                str(robust_portfolio_summary.get("num_selected_trades", "n/a")),
                robust_portfolio_timeline,
            ),
            _metric(
                "Calibrated Signals",
                str(edge_calibration_summary.get("calibrated_trades", "n/a")),
                edge_calibration_summary.get("trade_timeline", "n/a"),
            ),
            _metric(
                "Price-Aware Signals",
                str(price_aware_calibration_summary.get("calibrated_trades", "n/a")),
                price_aware_calibration_summary.get("trade_timeline", "n/a"),
            ),
            _metric(
                "Price-Aware Sweep",
                str(price_aware_sweep_summary.get("best_status", "n/a")),
                f"signals={price_aware_sweep_summary.get('best_signals', 'n/a')}",
            ),
            _metric(
                "Residual CLV+",
                _fmt_pct(residual_summary.get("positive_clv_rate")),
                f"avg={_fmt_number(residual_summary.get('avg_clv_cents'), 2)}c",
            ),
            _metric(
                "Best Price-Aware CLV+",
                _fmt_pct(residual_price_aware_best_summary.get("positive_clv_rate")),
                f"profit/share={_fmt_number(residual_price_aware_best_summary.get('avg_profit_per_share'), 3)}",
            ),
            _metric(
                "Market Movement CLV+",
                _fmt_pct(market_movement_summary.get("positive_clv_rate")),
                "Later pregame price",
            ),
            _metric(
                "Corrected CLV WF",
                str(corrected_clv_summary.get("sides", {}).get("NO", {}).get("walk_forward_status", "n/a")),
                f"NO CLV+={_fmt_pct(corrected_clv_summary.get('sides', {}).get('NO', {}).get('walk_forward_positive_clv_rate'))}",
            ),
            _metric(
                "Price-Aware CLV WF",
                str(corrected_clv_price_aware_best_summary.get("sides", {}).get("NO", {}).get("walk_forward_status", "n/a")),
                f"NO CLV+={_fmt_pct(corrected_clv_price_aware_best_summary.get('sides', {}).get('NO', {}).get('walk_forward_positive_clv_rate'))}",
            ),
            _metric(
                "Negative Edge Signals",
                str(edge_calibration_audit_summary.get("negative_raw_edge_calibrated_trades", "n/a")),
                edge_calibration_audit_summary.get("negative_raw_edge_trade_timeline", "n/a"),
            ),
            _metric(
                "Blend Calibrated Signals",
                str(market_blend_edge_calibration_summary.get("calibrated_trades", "n/a")),
                market_blend_edge_calibration_summary.get("trade_timeline", "n/a"),
            ),
            _metric(
                "Consensus Signals",
                str(consensus_edge_summary.get("consensus_trades", "n/a")),
                consensus_edge_summary.get("trade_timeline", "n/a"),
            ),
            _metric(
                "Robust Signals",
                str(robust_edge_summary.get("robust_signals", "n/a")),
                robust_edge_summary.get("trade_timeline", "n/a"),
            ),
            _metric(
                "Consensus Positive Months",
                f"{consensus_stability_summary.get('positive_months', 'n/a')}/{consensus_stability_summary.get('months', 'n/a')}",
                consensus_stability_summary.get("timeline", "n/a"),
            ),
            _metric(
                "Robust Positive Months",
                f"{robust_stability_summary.get('positive_months', 'n/a')}/{robust_stability_summary.get('months', 'n/a')}",
                robust_stability_summary.get("timeline", "n/a"),
            ),
            _metric(
                "Paper Candidates",
                str(strategy_readiness_summary.get("paper_trade_candidates", "n/a")),
                "Readiness gate",
            ),
            _metric(
                "Rule Sweep Watchlist",
                str(signal_rule_sweep_summary.get("watchlist_rules", "n/a")),
                signal_rule_sweep_summary.get("best_rule_timeline", "n/a"),
            ),
            _metric(
                "Best Rule Signals",
                str(signal_rule_sweep_summary.get("best_rule_signals", "n/a")),
                signal_rule_sweep_summary.get("best_rule", "n/a"),
            ),
            _metric(
                "WF Rule Signals",
                str(signal_rule_walk_forward_summary.get("signals", "n/a")),
                signal_rule_walk_forward_summary.get("timeline", "n/a"),
            ),
            _metric(
                "WF Rule Status",
                str(signal_rule_walk_forward_summary.get("status", "n/a")),
                "Nested month-by-month",
            ),
            _metric(
                "Parlay Pair Obs",
                str(parlay_correlation_summary.get("pair_rows", "n/a")),
                parlay_correlation_summary.get("timeline", "n/a"),
            ),
            _metric(
                "Parlay Status",
                str(parlay_correlation_summary.get("status", "n/a")),
                "Research only",
            ),
            _metric(
                "Parlay Ready",
                str(parlay_correlation_summary.get("parlay_ready", False)),
                f"strategy_ready={parlay_correlation_summary.get('strategy_parlay_ready_count', 0)}",
            ),
            _metric("Blend ROI on Risk", _fmt_pct(blend_backtest_summary.get("roi_on_amount_risked")), ""),
        ]
    )

    return template.safe_substitute(
        generated_at=html.escape(dashboard_data["generated_at"]),
        report_dir=html.escape(dashboard_data["report_dir"]),
        metric_html=metric_html,
        forward_metric_html=forward_metric_html,
        model_metric_html=model_metric_html,
        backtest_metric_html=backtest_metric_html,
        quality_metric_html=quality_metric_html,
        quality_list=_quality_list(market_quality),
        quality_list_2=_quality_list(market_quality),
        equity_plot=_plot_panel("Equity Curve", _image_data_uri(report_path / "equity_curve.png")),
        threshold_plot=_plot_panel("Threshold Sweep", _image_data_uri(report_path / "threshold_sweep.png")),
        threshold_plot_2=_plot_panel("Threshold Sweep", _image_data_uri(report_path / "threshold_sweep.png")),
        edge_plot=_plot_panel("Edge Distribution", _image_data_uri(report_path / "edge_distribution.png")),
        walk_calibration_plot=_plot_panel(
            "Walk-Forward Calibration",
            _image_data_uri(report_path / "walk_forward_calibration_curve.png"),
        ),
        walk_probability_plot=_plot_panel(
            "Walk-Forward Probability Distribution",
            _image_data_uri(report_path / "walk_forward_probability_distribution.png"),
        ),
        probability_bin_plot=_plot_panel(
            "Probability Bucket Calibration",
            _image_data_uri(report_path / "prediction_probability_bins.png"),
        ),
        season_summary_plot=_plot_panel(
            "Season Diagnostics",
            _image_data_uri(report_path / "prediction_season_summary.png"),
        ),
        edge_bin_profit_plot=_plot_panel(
            "Profit by Edge Bucket",
            _image_data_uri(report_path / "backtest_edge_bins.png"),
        ),
        forward_recommendations_table=_table(
            _format_forward_recommendations(forward_recommendations),
            columns=[
                "game_date",
                "upcoming_status",
                "home_team_abbr",
                "away_team_abbr",
                "model_home_win_prob",
                "model_away_win_prob",
                "model_pick_team",
                "recommended_team",
                "price_cents",
                "market_prob",
                "model_yes_prob",
                "edge",
                "forward_expected_roi",
                "has_kalshi_odds",
                "recommendation",
                "passes_best_sweep_rule",
                "best_sweep_rule_status",
                "rule_validation_status",
                "paper_shares",
                "paper_amount_risked",
                "hypothetical_shares",
                "hypothetical_amount_risked",
                "readiness_gate",
            ],
            max_rows=80,
            table_id="forward-table",
        ),
        fold_table=_table(_fold_table(walk_forward_metrics), max_rows=20),
        probability_bin_table=_table(
            _format_probability_bins(probability_bins),
            columns=[
                "probability_bin",
                "games",
                "avg_predicted_prob",
                "observed_win_rate",
                "calibration_error",
                "abs_calibration_error",
            ],
            max_rows=20,
        ),
        season_summary_table=_table(
            _format_season_summary(season_summary),
            columns=[
                "season",
                "games",
                "accuracy",
                "brier_score",
                "avg_predicted_home_win_prob",
                "actual_home_win_rate",
                "mean_absolute_calibration_error",
            ],
            max_rows=20,
        ),
        feature_table=_table(
            feature_diagnostics,
            columns=["feature", "metric", "value", "abs_value"],
            max_rows=15,
        ),
        model_tuning_table=_table(
            _format_tuning_results(model_tuning_results),
            columns=[
                "model_name",
                "feature_set",
                "feature_count",
                "num_predictions",
                "log_loss",
                "brier_score",
                "roc_auc",
                "accuracy",
            ],
            max_rows=20,
        ),
        home_win_ensemble_note=html.escape(str(home_win_ensemble_summary.get("note", ""))),
        home_win_ensemble_weights_table=_table(
            _format_ensemble_weights(home_win_ensemble_weights),
            columns=[
                "season",
                "train_rows",
                "test_rows",
                "weight_source",
                "weight_base_home_win_prob",
                "weight_tuned_home_win_prob",
                "weight_margin_home_win_prob",
                "test_log_loss",
                "test_brier_score",
                "test_roc_auc",
                "test_accuracy",
            ],
            max_rows=15,
        ),
        home_win_ensemble_static_audit_table=_table(
            _format_ensemble_static_audit(home_win_ensemble_static_audit),
            columns=[
                "weight_base_home_win_prob",
                "weight_tuned_home_win_prob",
                "weight_margin_home_win_prob",
                "log_loss",
                "brier_score",
                "roc_auc",
                "accuracy",
                "note",
            ],
            max_rows=10,
        ),
        market_blend_table=_table(
            _format_market_blend_rows(market_blend_metrics),
            columns=["probability_source", "accuracy", "brier_score", "log_loss", "roc_auc"],
            max_rows=10,
        ),
        market_type_calibration_table=_table(
            _format_market_type_calibration(market_type_calibration),
            columns=[
                "market_type",
                "probability_bin",
                "rows",
                "avg_predicted_prob",
                "observed_rate",
                "abs_calibration_error",
            ],
            max_rows=30,
        ),
        headline_backtest_note=html.escape(str(headline_backtest_summary.get("note", ""))),
        headline_backtest_table=_table(
            _headline_backtest_rows(headline_backtest_summary),
            columns=[
                "headline",
                "settlement",
                "trades",
                "timeline",
                "ending_bankroll",
                "return",
                "max_drawdown",
                "readiness",
                "parlays_blocked",
            ],
            max_rows=1,
        ),
        player_feature_comparison_json=html.escape(json.dumps(player_feature_comparison, indent=2)),
        edge_bin_table=_table(
            _format_edge_bins(edge_bins),
            columns=[
                "edge_bin",
                "markets",
                "trades",
                "avg_edge",
                "win_rate",
                "traded_win_rate",
                "total_profit",
                "amount_risked",
                "roi_on_amount_risked",
            ],
            max_rows=20,
        ),
        portfolio_comparison_table=_table(
            _portfolio_comparison_rows(
                [
                    (
                        "Headline slate-settled result",
                        headline_backtest_summary,
                        headline_trade_timeline,
                    ),
                    ("Raw model backtest", backtest_summary, model_trade_timeline),
                    ("Home-win ensemble backtest", ensemble_backtest_summary, ensemble_trade_timeline),
                    ("Market-blend backtest", blend_backtest_summary, blend_trade_timeline),
                    ("Raw calibrated slate", calibrated_portfolio_summary, calibrated_portfolio_trade_timeline),
                    (
                        "Market-blend calibrated slate",
                        market_blend_calibrated_portfolio_summary,
                        market_blend_calibrated_portfolio_timeline,
                    ),
                    ("Consensus calibrated slate", consensus_portfolio_summary, consensus_portfolio_timeline),
                    ("Robust consensus stress test", robust_portfolio_summary, robust_portfolio_timeline),
                ]
            ),
            columns=[
                "strategy",
                "trades",
                "timeline",
                "ending_bankroll",
                "return",
                "max_drawdown",
                "roi_on_risk",
            ],
            max_rows=12,
        ),
        strategy_readiness_table=_table(
            _format_strategy_readiness(strategy_readiness),
            columns=[
                "strategy",
                "status",
                "parlay_ready",
                "signals",
                "months",
                "positive_month_share",
                "avg_signal_profit_per_share",
                "ending_bankroll",
                "max_drawdown",
                "failed_checks",
                "recommendation",
            ],
            max_rows=20,
        ),
        signal_rule_sweep_note=html.escape(str(signal_rule_sweep_summary.get("note", ""))),
        signal_rule_sweep_table=_table(
            _format_signal_rule_sweep(signal_rule_sweep),
            columns=[
                "status",
                "score",
                "min_edge",
                "min_expected_roi",
                "min_edge_bin_history_rows",
                "min_price_cents",
                "max_price_cents",
                "signals",
                "timeline",
                "months",
                "positive_month_share",
                "overall_avg_profit_per_share",
                "worst_month",
                "worst_month_avg_profit_per_share",
                "avg_edge",
                "avg_expected_roi",
                "parlay_ready",
            ],
            max_rows=35,
        ),
        signal_rule_sweep_best_monthly_table=_table(
            _format_signal_stability(signal_rule_sweep_best_monthly),
            columns=[
                "rule_status",
                "rule",
                "month",
                "signals",
                "win_rate",
                "avg_profit_per_share",
                "total_profit_per_share",
                "avg_edge",
                "avg_expected_roi",
            ],
            max_rows=24,
        ),
        signal_rule_walk_forward_note=html.escape(str(signal_rule_walk_forward_summary.get("note", ""))),
        signal_rule_walk_forward_folds_table=_table(
            signal_rule_walk_forward_folds,
            columns=[
                "test_month",
                "status",
                "train_rows",
                "train_months",
                "test_rows",
                "signals",
                "selected_rule",
                "selected_rule_status",
                "train_best_rule_timeline",
            ],
            max_rows=24,
        ),
        signal_rule_walk_forward_monthly_table=_table(
            _format_signal_stability(signal_rule_walk_forward_monthly),
            columns=[
                "month",
                "signals",
                "win_rate",
                "avg_profit_per_share",
                "total_profit_per_share",
                "avg_edge",
                "avg_expected_roi",
            ],
            max_rows=24,
        ),
        parlay_correlation_note=html.escape(str(parlay_correlation_summary.get("note", ""))),
        parlay_correlation_table=_table(
            _format_parlay_report(parlay_correlation_report),
            columns=[
                "group",
                "group_value",
                "pairs",
                "slates",
                "pair_win_rate",
                "avg_market_pair_prob_independent",
                "avg_estimated_pair_prob_independent",
                "avg_pair_edge_independent",
                "avg_synthetic_profit_per_dollar",
                "leg_outcome_correlation",
            ],
            max_rows=30,
        ),
        top_trades_table=_table(
            _format_top_trades(top_trades),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "result_type",
                "edge",
                "shares",
                "profit",
                "bankroll_after",
            ],
            max_rows=20,
        ),
        portfolio_trades_table=_table(
            _format_portfolio_trades(portfolio_trades),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "expected_roi",
                "shares",
                "profit",
                "bankroll_after",
            ],
            max_rows=30,
        ),
        calibrated_portfolio_trades_table=_table(
            _format_portfolio_trades(calibrated_portfolio_trades),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "expected_roi",
                "selection_expected_roi",
                "shares",
                "profit",
                "bankroll_after",
            ],
            max_rows=30,
        ),
        market_blend_calibrated_portfolio_trades_table=_table(
            _format_portfolio_trades(market_blend_calibrated_portfolio_trades),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "expected_roi",
                "selection_expected_roi",
                "shares",
                "profit",
                "bankroll_after",
            ],
            max_rows=30,
        ),
        consensus_portfolio_trades_table=_table(
            _format_portfolio_trades(consensus_portfolio_trades),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "expected_roi",
                "selection_expected_roi",
                "shares",
                "profit",
                "bankroll_after",
            ],
            max_rows=30,
        ),
        robust_portfolio_trades_table=_table(
            _format_portfolio_trades(robust_portfolio_trades),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "expected_roi",
                "selection_expected_roi",
                "shares",
                "profit",
                "bankroll_after",
            ],
            max_rows=30,
        ),
        portfolio_slates_table=_table(
            _format_portfolio_slates(portfolio_slates),
            columns=[
                "date",
                "candidate_bets",
                "selected_trades",
                "slate_cost_fraction",
                "slate_profit",
                "bankroll_after",
                "rejected_by_game_cap",
                "rejected_by_team_cap",
                "rejected_by_budget_cap",
            ],
            max_rows=40,
        ),
        calibrated_portfolio_slates_table=_table(
            _format_portfolio_slates(calibrated_portfolio_slates),
            columns=[
                "date",
                "candidate_bets",
                "selected_trades",
                "slate_cost_fraction",
                "slate_profit",
                "bankroll_after",
                "rejected_by_game_cap",
                "rejected_by_team_cap",
                "rejected_by_budget_cap",
            ],
            max_rows=40,
        ),
        market_blend_calibrated_portfolio_slates_table=_table(
            _format_portfolio_slates(market_blend_calibrated_portfolio_slates),
            columns=[
                "date",
                "candidate_bets",
                "selected_trades",
                "slate_cost_fraction",
                "slate_profit",
                "bankroll_after",
                "rejected_by_game_cap",
                "rejected_by_team_cap",
                "rejected_by_budget_cap",
            ],
            max_rows=40,
        ),
        consensus_portfolio_slates_table=_table(
            _format_portfolio_slates(consensus_portfolio_slates),
            columns=[
                "date",
                "candidate_bets",
                "selected_trades",
                "slate_cost_fraction",
                "slate_profit",
                "bankroll_after",
                "rejected_by_game_cap",
                "rejected_by_team_cap",
                "rejected_by_budget_cap",
            ],
            max_rows=40,
        ),
        robust_portfolio_slates_table=_table(
            _format_portfolio_slates(robust_portfolio_slates),
            columns=[
                "date",
                "candidate_bets",
                "selected_trades",
                "slate_cost_fraction",
                "slate_profit",
                "bankroll_after",
                "rejected_by_game_cap",
                "rejected_by_team_cap",
                "rejected_by_budget_cap",
            ],
            max_rows=40,
        ),
        edge_calibration_table=_table(
            _format_edge_calibration(edge_calibration_bins),
            columns=[
                "edge_bin",
                "markets",
                "avg_edge",
                "avg_model_prob",
                "avg_market_prob",
                "observed_yes_rate",
                "avg_model_expected_profit_per_share",
                "avg_realized_profit_per_share",
                "realized_roi_on_cost",
                "calibration_gap",
            ],
            max_rows=30,
        ),
        edge_calibration_audit_table=_table(
            _format_edge_audit(edge_calibration_audit),
            columns=[
                "edge_bin",
                "markets",
                "calibrated_signals",
                "avg_edge",
                "observed_yes_rate",
                "signal_win_rate",
                "signal_avg_realized_profit_per_share",
                "signal_avg_calibrated_expected_roi",
                "negative_raw_edge_signals",
                "negative_raw_edge_signal_win_rate",
                "negative_raw_edge_signal_profit_per_share",
            ],
            max_rows=30,
        ),
        price_aware_calibration_table=_table(
            price_aware_calibration_bins,
            columns=[
                "calibrated_side",
                "price_bin",
                "edge_bin",
                "rows",
                "calibrated_trades",
                "avg_price_cents",
                "avg_calibrated_win_rate",
                "observed_win_rate",
                "avg_realized_profit_per_share",
                "avg_clv_cents",
                "calibration_error",
            ],
            max_rows=30,
        ),
        price_aware_sweep_table=_table(
            price_aware_sweep,
            columns=[
                "min_history_rows",
                "min_price_history_rows",
                "shrinkage_rows",
                "min_calibrated_profit_per_share",
                "signals",
                "months",
                "positive_month_share",
                "avg_profit_per_share",
                "avg_clv_cents",
                "positive_clv_rate",
                "status",
            ],
            max_rows=20,
        ),
        residual_by_calibrated_table=_table(
            residual_by_calibrated,
            columns=[
                "_side",
                "calibrated_residual_bucket",
                "rows",
                "avg_price_cents",
                "avg_calibrated_win_rate",
                "realized_win_rate",
                "avg_calibration_error",
                "avg_profit_per_share",
                "avg_clv_cents",
                "positive_clv_rate",
            ],
            max_rows=30,
        ),
        residual_price_zone_table=_table(
            residual_price_zone,
            columns=[
                "_side",
                "market_price_zone",
                "rows",
                "avg_price_cents",
                "avg_calibrated_win_rate",
                "realized_win_rate",
                "avg_calibration_error",
                "avg_profit_per_share",
                "avg_clv_cents",
                "positive_clv_rate",
            ],
            max_rows=12,
        ),
        market_movement_by_side_table=_table(
            market_movement_by_side,
            columns=[
                "_side",
                "market_move",
                "rows",
                "avg_clv_cents",
                "positive_clv_rate",
                "win_rate",
                "avg_profit_per_share",
                "avg_price_cents",
                "avg_calibrated_roi",
            ],
            max_rows=12,
        ),
        negative_edge_signals_table=_table(
            _format_negative_edge_signals(negative_edge_signals),
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge_bin",
                "edge",
                "model_yes_prob",
                "market_prob",
                "calibrated_yes_rate",
                "calibrated_expected_roi",
                "actual_yes_win",
                "realized_profit_per_share",
                "edge_bin_history_rows",
            ],
            max_rows=40,
        ),
        consensus_stability_table=_table(
            _format_signal_stability(consensus_stability),
            columns=[
                "month",
                "signals",
                "win_rate",
                "avg_profit_per_share",
                "total_profit_per_share",
                "avg_edge",
                "avg_expected_roi",
            ],
            max_rows=24,
        ),
        robust_stability_table=_table(
            _format_signal_stability(robust_stability),
            columns=[
                "month",
                "signals",
                "win_rate",
                "avg_profit_per_share",
                "total_profit_per_share",
                "avg_edge",
                "avg_expected_roi",
            ],
            max_rows=24,
        ),
        strategy_readiness_monthly_table=_table(
            _format_signal_stability(strategy_readiness_monthly),
            columns=[
                "strategy",
                "month",
                "signals",
                "win_rate",
                "avg_profit_per_share",
                "total_profit_per_share",
                "avg_edge",
                "avg_expected_roi",
            ],
            max_rows=60,
        ),
        market_blend_edge_calibration_audit_table=_table(
            _format_edge_audit(market_blend_edge_calibration_audit),
            columns=[
                "edge_bin",
                "markets",
                "calibrated_signals",
                "avg_edge",
                "observed_yes_rate",
                "signal_win_rate",
                "signal_avg_realized_profit_per_share",
                "signal_avg_calibrated_expected_roi",
                "negative_raw_edge_signals",
            ],
            max_rows=30,
        ),
        threshold_table=_table(
            _format_threshold_table(threshold_sweep),
            columns=[
                "edge_threshold",
                "num_trades",
                "trade_timeline",
                "ending_bankroll",
                "total_return_pct",
                "win_rate",
                "max_drawdown",
                "roi_on_amount_risked",
            ],
            max_rows=50,
        ),
        price_source_table=_table(price_sources, max_rows=20),
        quality_json=html.escape(json.dumps(market_quality, indent=2)),
        market_review_json=html.escape(json.dumps(market_review, indent=2)),
        data_validation_json=html.escape(json.dumps(data_validation, indent=2)),
        market_taxonomy_json=html.escape(json.dumps(market_taxonomy_summary, indent=2)),
        market_taxonomy_table=_table(
            market_taxonomy,
            columns=[
                "market_ticker",
                "market_category",
                "market_title",
                "stat_type",
                "line_value",
                "direction",
                "yes_team_abbr",
                "taxonomy_confidence",
            ],
            max_rows=40,
        ),
        broad_market_taxonomy_json=html.escape(json.dumps(broad_market_taxonomy_summary, indent=2)),
        broad_market_taxonomy_table=_table(
            broad_market_taxonomy,
            columns=[
                "market_ticker",
                "market_category",
                "market_title",
                "stat_type",
                "line_value",
                "direction",
                "player_name",
                "mentioned_team_abbrs",
                "taxonomy_confidence",
            ],
            max_rows=40,
        ),
        historical_series_json=html.escape(json.dumps(historical_series_summary, indent=2)),
        historical_series_table=_table(
            historical_series_table,
            columns=[
                "series_ticker",
                "unique_markets",
                "expected_expiration_time_min",
                "expected_expiration_time_max",
                "open_time_min",
                "open_time_max",
            ],
            max_rows=20,
        ),
        market_line_coverage_json=html.escape(json.dumps(market_line_coverage_summary, indent=2)),
        market_line_coverage_table=_table(
            _format_market_line_coverage(market_line_coverage),
            columns=[
                "market_type",
                "rows",
                "rows_with_line",
                "rows_with_direction",
                "high_confidence_rows",
                "line_coverage_pct",
                "high_confidence_coverage_pct",
                "line_extraction_ready",
                "ready_for_market_specific_backtest",
                "blocked_reason",
            ],
            max_rows=10,
        ),
        multivariate_nba_legs_json=html.escape(json.dumps(multivariate_legs_summary, indent=2)),
        multivariate_nba_legs_table=_table(
            multivariate_nba_legs,
            columns=[
                "leg_market_ticker",
                "leg_category",
                "leg_stat_type",
                "leg_line_value",
                "leg_side",
                "leg_game_date",
                "away_team_abbr",
                "home_team_abbr",
                "leg_usage_status",
                "parent_market_ticker",
            ],
            max_rows=60,
        ),
        underlying_leg_json=html.escape(json.dumps(underlying_leg_summary, indent=2)),
        underlying_leg_markets_table=_table(
            underlying_leg_markets,
            columns=[
                "market_ticker",
                "series_ticker",
                "event_ticker",
                "market_title",
                "title",
                "direct_fetch_route",
                "status",
            ],
            max_rows=40,
        ),
        underlying_leg_requests_table=_table(
            underlying_leg_requests,
            columns=["market_ticker", "route", "status", "error"],
            max_rows=40,
        ),
        line_candle_json=html.escape(json.dumps(line_candle_summary, indent=2)),
        line_pregame_prices_table=_table(
            line_pregame_prices,
            columns=[
                "game_date",
                "market_category",
                "market_ticker",
                "snapshot_target",
                "line_value",
                "direction",
                "yes_team_abbr",
                "yes_ask",
                "mid_price",
                "last_price",
                "period_interval",
                "price_quality",
            ],
            max_rows=60,
        ),
        line_market_eval_json=html.escape(json.dumps(line_market_eval_summary, indent=2)),
        line_market_eval_table=_table(
            line_market_eval,
            columns=[
                "game_date",
                "market_category",
                "market_ticker",
                "line_value",
                "direction",
                "yes_team_abbr",
                "model_yes_prob",
                "market_prob",
                "edge",
                "trade_signal",
                "actual_yes",
                "profit_per_contract",
            ],
            max_rows=60,
        ),
        validation_issues_table=_table(
            validation_issues,
            columns=["severity", "check", "count", "detail"],
            max_rows=50,
        ),
        security_audit_json=html.escape(json.dumps(security_audit_summary, indent=2)),
        security_audit_table=_table(
            security_audit_findings,
            columns=["file", "pattern", "action"],
            max_rows=50,
        ),
        coverage_monthly_table=_table(
            _format_coverage_monthly(coverage_monthly),
            columns=[
                "month",
                "games",
                "games_with_markets",
                "games_with_prices",
                "market_coverage_pct",
                "price_coverage_pct",
            ],
            max_rows=40,
        ),
        gap_reason_table=_table(_gap_reason_table(gap_report), max_rows=20),
        gap_detail_table=_table(
            gap_report,
            columns=[
                "game_date",
                "home_team_abbr",
                "away_team_abbr",
                "market_ticker",
                "market_title",
                "gap_reason",
            ],
            max_rows=100,
        ),
        suggestions_table=_table(
            suggestions,
            columns=[
                "game_date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "model_prob",
                "market_prob",
                "edge",
                "player_data_available",
                "projected_rotation_available",
                "missing_key_player_uncertainty",
                "prediction_label",
                "price_cents",
                "trade",
                "reason",
            ],
            max_rows=100,
            table_id="suggestions-table",
        ),
        trades_table=_table(
            backtest_trades,
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "trade",
                "shares",
                "profit",
                "bankroll_after",
                "reason",
            ],
            max_rows=100,
            table_id="trades-table",
        ),
        blend_trades_table=_table(
            blend_backtest_trades,
            columns=[
                "date",
                "home_team_abbr",
                "away_team_abbr",
                "yes_team_abbr",
                "edge",
                "trade",
                "shares",
                "profit",
                "bankroll_after",
                "reason",
            ],
            max_rows=100,
        ),
        dashboard_data=json.dumps(dashboard_data),
    )


def _build_research_picks_page(report_path: Path) -> str:
    fair = _read_csv(report_path / "fair_price_signals.csv")
    summary = _read_json(report_path / "fair_price_summary.json")
    if not fair.empty:
        tiers = {"research_lean", "paper_trade_candidate", "approved_bet"}
        if "recommendation_tier" in fair.columns:
            fair = fair[fair["recommendation_tier"].fillna("").astype(str).isin(tiers)].copy()
        fair = pd.DataFrame(
            {
                "market": _series_first(fair, ["market"]),
                "game date": _series_first(fair, ["game_date", "date"]),
                "team/player": _series_first(fair, ["yes_team", "team", "player_name"]),
                "side": _series_first(fair, ["research_side", "side", "ungated_side"]),
                "price": _series_first(fair, ["research_price", "price"]),
                "model_probability": _series_first(fair, ["research_model_probability", "model_prob"]),
                "market implied probability": _series_first(
                    fair,
                    ["research_market_implied_probability", "market_implied_probability"],
                ),
                "edge": pd.to_numeric(_series_first(fair, ["edge", "final_edge"]), errors="coerce"),
                "confidence label": _series_first(fair, ["confidence_label", "confidence"]),
                "recommendation tier": _series_first(fair, ["recommendation_tier"]),
                "reason": _series_first(fair, ["main_reason", "ungated_main_reason"]),
                "main risk": _series_first(fair, ["main_risk", "blocked_reason"]),
            }
        ).sort_values("edge", ascending=False)
    body = (
        _summary_grid(
            [
                ("Approved bets", summary.get("approved_bets_count", 0)),
                ("Paper candidates", summary.get("paper_trade_candidates_count", 0)),
                ("Research leans", summary.get("research_leans_count", 0)),
                ("Proof status", summary.get("proof_status") or summary.get("proof_gate_status", "unknown")),
                ("Blocked reason", summary.get("blocked_reason", "None") or "None"),
            ]
        )
        + '<p class="note">Research-only picks remain visible even when approved bets are blocked. Proof gates are unchanged.</p>'
        + _static_table(
            fair,
            [
                "market",
                "game date",
                "team/player",
                "side",
                "price",
                "model_probability",
                "market implied probability",
                "edge",
                "confidence label",
                "recommendation tier",
                "reason",
                "main risk",
            ],
            max_rows=300,
        )
    )
    return _static_page("Research Picks", report_path, "research_picks.html", body)


def _build_paper_candidates_page(report_path: Path) -> str:
    fair = _read_csv(report_path / "fair_price_signals.csv")
    summary = _read_json(report_path / "fair_price_summary.json")
    if not fair.empty and "recommendation_tier" in fair.columns:
        fair = fair[fair["recommendation_tier"].fillna("").astype(str).eq("paper_trade_candidate")].copy()
        fair["side"] = _series_first(fair, ["research_side", "side", "ungated_side"])
        fair["price"] = _series_first(fair, ["research_price", "price"])
        fair["model_probability"] = _series_first(fair, ["research_model_probability", "model_prob"])
        fair["market_implied_probability"] = _series_first(
            fair,
            ["research_market_implied_probability", "market_implied_probability"],
        )
        fair["edge"] = pd.to_numeric(_series_first(fair, ["edge", "final_edge"]), errors="coerce")
        fair = fair.sort_values("edge", ascending=False)
    body = (
        _summary_grid(
            [
                ("Paper candidates", summary.get("paper_trade_candidates_count", 0)),
                ("Approved bets", summary.get("approved_bets_count", 0)),
                ("Proof status", summary.get("proof_status") or summary.get("proof_gate_status", "unknown")),
                ("Blocked reason", summary.get("blocked_reason", "None") or "None"),
            ]
        )
        + '<p class="note">Paper candidates are stronger research picks. They are not approved bets unless single-game proof is proven.</p>'
        + _static_table(
            fair,
            [
                "market",
                "game_date",
                "side",
                "price",
                "model_probability",
                "market_implied_probability",
                "edge",
                "confidence_label",
                "main_reason",
                "main_risk",
            ],
            max_rows=300,
        )
    )
    return _static_page("Paper Candidates", report_path, "paper_candidates.html", body)


def _build_trade_results_page(report_path: Path) -> str:
    trades = _read_csv(report_path / "backtest_trades.csv")
    fair = _read_csv(report_path / "fair_price_signals.csv")
    summary = _read_json(report_path / "backtest_summary.json")
    if not trades.empty:
        trades = trades.copy()
        if "trade" in trades.columns:
            traded = trades["trade"].astype(str).str.lower().isin(["true", "1", "yes"])
            if traded.any():
                trades = trades[traded].copy()
        if "market" not in trades.columns:
            trades["market"] = (
                _series_first(trades, ["away_team_abbr"]).astype(str)
                + " at "
                + _series_first(trades, ["home_team_abbr"]).astype(str)
            )
        trades["side_display"] = _series_first(trades, ["candidate_side", "side"])
        trades["entry price"] = _series_first(trades, ["price_cents"])
        trades["closing price"] = _series_first(trades, ["clv_reference_price_cents", "closing_price_cents"])
        trades["CLV"] = _series_first(trades, ["clv_cents"])
        actual_yes = _series_first(trades, ["actual_yes_win"])
        side = trades["side_display"].astype(str).str.upper()
        actual_bool = actual_yes.astype(str).str.lower().isin(["true", "1", "yes"])
        won = ((side == "YES") & actual_bool) | ((side == "NO") & ~actual_bool)
        trades["settlement result"] = np.where(won, "win", "loss")
        trades["profit/loss"] = _series_first(trades, ["profit"])
        if not fair.empty and {"game_id", "market_ticker", "recommendation_tier"}.issubset(fair.columns):
            tiers = fair[["game_id", "market_ticker", "recommendation_tier"]].drop_duplicates()
            trades = trades.merge(tiers, on=["game_id", "market_ticker"], how="left")
        if "recommendation_tier" not in trades.columns:
            trades["recommendation_tier"] = ""
        trades = trades.rename(columns={"side_display": "side", "date": "game date"})
        if "game date" in trades.columns:
            trades = trades.sort_values("game date", ascending=False)
    body = (
        _summary_grid(
            [
                ("Trades", summary.get("trades", 0)),
                ("Ending bankroll", _fmt_money(summary.get("ending_bankroll")) if summary else "n/a"),
                ("Profit", _fmt_money(summary.get("profit")) if summary else "n/a"),
                ("Average CLV", _fmt_number(summary.get("average_clv_cents"), 3) if summary else "n/a"),
            ]
        )
        + _static_table(
            trades,
            [
                "market",
                "side",
                "entry price",
                "closing price",
                "CLV",
                "settlement result",
                "profit/loss",
                "recommendation_tier",
                "game date",
            ],
            max_rows=300,
        )
    )
    return _static_page("Trade Results", report_path, "trade_results.html", body)


def _build_parlay_research_page(report_path: Path) -> str:
    parlays = _read_csv(report_path / "parlay_recommendations.csv")
    summary = _read_json(report_path / "parlay_recommendations_summary.json")
    research_parlays = _read_csv(report_path / "research_parlay_candidates.csv")
    paper_parlays = _read_csv(report_path / "paper_parlay_candidates.csv")
    research_summary = _read_json(report_path / "parlay_research_summary.json")
    research_detail = research_summary.get("research_parlay", {}) if isinstance(research_summary, dict) else {}
    paper_detail = research_summary.get("paper_parlay", {}) if isinstance(research_summary, dict) else {}
    display_columns = [
        "rank",
        "parlay_tier",
        "research_only",
        "approved",
        "legs",
        "number_of_legs",
        "combined_model_probability",
        "estimated_fair_payout",
        "offered_payout",
        "break_even_probability",
        "estimated_ev",
        "average_leg_edge",
        "lowest_leg_edge",
        "correlation_risk",
        "same_game_count",
        "same_team_count",
        "same_player_count",
        "biggest_risk",
        "reason_selected",
    ]
    body = (
        _summary_grid(
            [
                ("Status", summary.get("status", "unknown")),
                ("Parlays", summary.get("parlays", 0)),
                ("Eligible legs", summary.get("eligible_single_game_legs", 0)),
                ("Single-game edge proven", summary.get("single_game_edge_proven", False)),
                ("Parlays allowed", summary.get("parlay_recommendations_allowed", False)),
                ("Research parlays", research_detail.get("parlays", 0)),
                ("Paper parlays", paper_detail.get("parlays", 0)),
            ]
        )
        + '<p class="note">Approved parlay recommendations remain blocked unless single-game edge proof passes. Research and paper parlays below are experimental only, approved=false, research_only=true, and include no stake sizing.</p>'
        + "<h2>Research Parlays</h2>"
        + _static_table(research_parlays, display_columns, max_rows=200)
        + "<h2>Paper Parlays</h2>"
        + _static_table(paper_parlays, display_columns, max_rows=200)
        + "<h2>Approved Parlay Output</h2>"
        + _static_table(parlays, list(parlays.columns) if not parlays.empty else [], max_rows=200)
    )
    return _static_page("Parlay Research", report_path, "parlay_research.html", body)


def _build_proof_status_page(report_path: Path) -> str:
    proof = _read_json(report_path / "single_game_proof_summary.json")
    fair = _read_json(report_path / "fair_price_summary.json")
    parlay = _read_json(report_path / "parlay_recommendations_summary.json")
    gates = proof.get("gates", [])
    if isinstance(gates, list):
        gate_frame = pd.DataFrame(gates)
    elif isinstance(gates, dict):
        gate_frame = pd.DataFrame([{"gate": key, "value": value} for key, value in gates.items()])
    else:
        gate_frame = pd.DataFrame()
    body = (
        _summary_grid(
            [
                ("Proof status", proof.get("status", "unknown")),
                ("Single-game edge proven", proof.get("single_game_edge_proven", False)),
                ("Approved bets", fair.get("approved_bets_count", 0)),
                ("Research-only recommendations", fair.get("research_only_recommendations_count", 0)),
                ("Fair-price blocked reason", fair.get("blocked_reason", "None") or "None"),
                ("Parlay status", parlay.get("status", "unknown")),
            ]
        )
        + '<div class="status">Proof gates remain strict. This navigation update does not approve real betting or parlays.</div>'
        + "<h2>Gate Details</h2>"
        + _static_table(gate_frame, list(gate_frame.columns) if not gate_frame.empty else [], max_rows=100)
    )
    return _static_page("Proof Status", report_path, "proof_status.html", body)


def _records_table(records: list[dict[str, Any]]) -> str:
    return _static_table(pd.DataFrame(records), list(pd.DataFrame(records).columns) if records else [], max_rows=200)


def _build_recommendation_performance_page(report_path: Path) -> str:
    summary = _read_json(report_path / "recommendation_performance_summary.json")
    failures = _read_csv(report_path / "recommendation_failure_buckets.csv")
    graded_singles = _read_csv(report_path / "graded_single_recommendations.csv")
    graded_research = _read_csv(report_path / "graded_research_parlays.csv")
    graded_paper = _read_csv(report_path / "graded_paper_parlays.csv")
    parlay_frames = [frame for frame in [graded_research, graded_paper] if not frame.empty]
    graded_parlays = pd.concat(parlay_frames, ignore_index=True) if parlay_frames else pd.DataFrame()
    body = (
        _summary_grid(
            [
                ("Research leans", summary.get("total_research_leans", 0)),
                ("Paper candidates", summary.get("total_paper_candidates", 0)),
                ("Graded singles", summary.get("graded_rows", 0)),
                ("Win rate", _fmt_pct(summary.get("win_rate")) if summary.get("win_rate") is not None else "n/a"),
                ("Profit/loss", _fmt_money(summary.get("profit_loss"))),
                ("Average CLV", _fmt_number(summary.get("average_clv"), 3) if summary.get("average_clv") is not None else "n/a"),
            ]
        )
        + '<p class="note">This page grades saved recommendation snapshots. Research-only and paper-only outputs remain unapproved.</p>'
        + "<h2>Performance By Tier</h2>"
        + _records_table(summary.get("by_recommendation_tier", []))
        + "<h2>Performance By Side</h2>"
        + _records_table(summary.get("by_side", []))
        + "<h2>Worst Failure Buckets</h2>"
        + _static_table(
            failures,
            ["bucket_type", "bucket", "rows", "graded_rows", "win_rate", "profit_loss", "average_clv"],
            max_rows=30,
        )
        + "<h2>Parlay Performance By Tier</h2>"
        + _records_table(summary.get("parlay_by_tier", []))
        + "<h2>Parlay Performance By Correlation Risk</h2>"
        + _records_table(summary.get("parlay_by_correlation_risk", []))
        + "<h2>Recent Graded Singles</h2>"
        + _static_table(
            graded_singles,
            [
                "snapshot_date",
                "market",
                "recommendation_tier",
                "graded_side",
                "graded_price",
                "graded_model_probability",
                "graded_edge",
                "won",
                "profit_loss",
                "clv",
                "result_status",
            ],
            max_rows=100,
        )
        + "<h2>Recent Graded Research/Paper Parlays</h2>"
        + _static_table(
            graded_parlays,
            [
                "snapshot_date",
                "parlay_tier",
                "research_only",
                "approved",
                "legs",
                "legs_won",
                "legs_lost",
                "parlay_won",
                "combined_result_status",
                "estimated_profit_loss",
                "correlation_risk",
                "failed_leg_reason",
            ],
            max_rows=100,
        )
    )
    return _static_page("Recommendation Performance", report_path, "recommendation_performance.html", body)


def _build_recommendation_grading_audit_page(report_path: Path) -> str:
    summary = _read_json(report_path / "recommendation_grading_audit_summary.json")
    rows = _read_csv(report_path / "recommendation_grading_audit_rows.csv")
    profit_buckets = _read_csv(report_path / "recommendation_profit_by_bucket.csv")
    clv_buckets = _read_csv(report_path / "recommendation_clv_by_bucket.csv")
    match_quality = summary.get("match_quality", {})
    payout = summary.get("payout_profit_math", {})
    clv = summary.get("clv_math", {})
    leakage = summary.get("leakage_risk", {})
    tier = summary.get("tier_sanity", {})
    parlay = summary.get("parlay_sanity", {})
    warnings = summary.get("warnings", [])
    flagged_columns = leakage.get("flagged_recommendation_input_columns", [])
    warning_items = "".join(
        f'<li><strong>{html.escape(str(warning.get("severity", "")).upper())}</strong> '
        f'[{html.escape(str(warning.get("code", "")))}] {html.escape(str(warning.get("message", "")))}</li>'
        for warning in warnings
    )
    warnings_block = (
        f"<h2>Warnings</h2><ul>{warning_items}</ul>"
        if warning_items
        else '<h2>Warnings</h2><div class="status">No warnings raised.</div>'
    )
    body = (
        _summary_grid(
            [
                ("Trusted", summary.get("trusted", False)),
                ("Verdict", summary.get("verdict", "")),
                ("Major warnings", len(summary.get("major_warnings", []))),
                ("Recommendations", match_quality.get("recommendations", 0)),
                ("Graded", match_quality.get("graded", 0)),
                ("Unmatched", match_quality.get("unmatched", 0)),
                ("Profit math failures", payout.get("profit_math_failures", 0)),
                ("Summary profit", _fmt_money(payout.get("total_profit"))),
                ("Row-sum profit", _fmt_money(payout.get("row_level_profit_sum"))),
                ("Backtest stake P/L", _fmt_money(payout.get("backtest_stake_weighted_profit"))),
                ("CLV math failures", clv.get("clv_math_failures", 0)),
                ("Missing CLV rows", clv.get("missing_clv_rows", 0)),
            ]
        )
        + '<p class="note">This page audits whether recommendation grading can be trusted. It does not change proof gates or approve bets/parlays.</p>'
        + warnings_block
        + "<h2>Match Quality</h2>"
        + _records_table([match_quality])
        + "<h2>Payout And CLV Checks</h2>"
        + _records_table([payout])
        + _records_table([clv])
        + "<h2>Leakage Risk</h2>"
        + _records_table(
            [
                {
                    "flagged_recommendation_input_columns": ", ".join(flagged_columns) if flagged_columns else "None",
                    "grading_after_snapshot_check": leakage.get("grading_after_snapshot_check", ""),
                    "snapshot_count": (leakage.get("snapshot_audit", {}) or {}).get("snapshots", 0),
                    "manifest_count": (leakage.get("snapshot_audit", {}) or {}).get("manifest_count", 0),
                }
            ]
        )
        + "<h2>Tier Sanity</h2>"
        + _records_table(tier.get("by_recommendation_tier", []))
        + "<h2>Performance By Side</h2>"
        + _records_table(tier.get("by_side", []))
        + "<h2>Parlay Grading Sanity</h2>"
        + _records_table(
            [
                {
                    "parlays": parlay.get("parlays", 0),
                    "fully_graded_parlays": parlay.get("fully_graded_parlays", 0),
                    "leg_count_mismatches": parlay.get("leg_count_mismatches", 0),
                    "parlay_won_mismatches": parlay.get("parlay_won_mismatches", 0),
                }
            ]
        )
        + _records_table(parlay.get("by_parlay_tier", []))
        + "<h2>Profit By Bucket</h2>"
        + _static_table(
            profit_buckets,
            ["bucket_type", "bucket", "rows", "graded_rows", "win_rate", "profit_loss", "average_clv"],
            max_rows=80,
        )
        + "<h2>CLV By Bucket</h2>"
        + _static_table(
            clv_buckets,
            ["bucket_type", "bucket", "rows", "graded_rows", "win_rate", "profit_loss", "average_clv"],
            max_rows=80,
        )
        + "<h2>Rows Needing Review</h2>"
        + _static_table(
            rows[
                rows.get("audit_profit_math_ok", pd.Series(True, index=rows.index)).astype(str).eq("False")
                | rows.get("audit_clv_math_ok", pd.Series(True, index=rows.index)).astype(str).eq("False")
                | rows.get("audit_missing_clv_for_side", pd.Series(False, index=rows.index)).astype(str).eq("True")
                | rows.get("audit_impossible_profit_loss", pd.Series(False, index=rows.index)).astype(str).eq("True")
            ]
            if not rows.empty
            else rows,
            [
                "snapshot_date",
                "market",
                "market_ticker",
                "recommendation_tier",
                "graded_side",
                "graded_price",
                "profit_loss",
                "audit_expected_profit_loss",
                "audit_profit_math_ok",
                "clv",
                "audit_expected_clv",
                "audit_clv_source",
                "audit_clv_math_ok",
                "audit_missing_clv_for_side",
            ],
            max_rows=100,
        )
    )
    return _static_page("Recommendation Grading Audit", report_path, "recommendation_grading_audit.html", body)


def _count_table(series: pd.Series, label: str) -> str:
    counts = series.fillna("(missing)").astype(str).replace("", "(missing)").value_counts()
    frame = counts.rename_axis(label).reset_index(name="snapshots")
    return _static_table(frame, [label, "snapshots"], max_rows=60)


def _build_collection_health_section(health: dict) -> str:
    if not health:
        return (
            "<h2>Collection Health</h2>"
            '<div class="empty">No health report yet. Run scripts/build_prop_collection_health.py.</div>'
        )
    healthy = bool(health.get("healthy"))
    status_text = "HEALTHY" if healthy else "UNHEALTHY"
    runs = health.get("runs", {}) if isinstance(health.get("runs"), dict) else {}
    latest = health.get("latest_run", {}) if isinstance(health.get("latest_run"), dict) else {}
    reasons = health.get("health_reasons") or []
    reasons_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(r))}</li>" for r in reasons) + "</ul>"
        if reasons
        else '<div class="empty">All collection health checks passed.</div>'
    )
    by_sport = [
        {"sport": sport, "snapshots": count}
        for sport, count in sorted(
            (health.get("snapshots_by_sport") or {}).items(), key=lambda kv: -int(kv[1])
        )
    ]
    missed_days = health.get("days_with_no_collection") or []
    missed_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(day))}</li>" for day in missed_days[-14:]) + "</ul>"
        if missed_days
        else '<div class="empty">No missed days since collection started.</div>'
    )
    issues = [f"ERROR: {e}" for e in (health.get("latest_errors") or [])]
    issues += [f"WARNING: {w}" for w in (health.get("latest_warnings") or [])]
    issues_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(i))}</li>" for i in issues) + "</ul>"
        if issues
        else '<div class="empty">No errors or warnings in the latest run.</div>'
    )
    stale = health.get("leagues_with_no_recent_snapshots") or []
    return (
        "<h2>Collection Health</h2>"
        + _summary_grid(
            [
                ("Health status", status_text),
                ("Latest run", f"{latest.get('run_id') or 'n/a'} ({latest.get('outcome') or 'n/a'})"),
                ("Latest run snapshots", latest.get("snapshots_collected", 0)),
                ("Last successful collection", health.get("last_successful_collection_utc") or "never"),
                ("Last failed collection", health.get("last_failed_collection_utc") or "never"),
                (
                    "Runs (ok/failed/skipped)",
                    f"{runs.get('successful', 0)}/{runs.get('failed', 0)}/{runs.get('skipped', 0)}"
                    f" of {runs.get('total', 0)}",
                ),
                ("Missed days", health.get("missed_days_count", 0)),
                ("API key detected", str(health.get("api_key_detected"))),
                ("Likely quota issue", str(health.get("likely_quota_issue"))),
                ("Latest run log", health.get("latest_run_log") or "n/a"),
            ]
        )
        + "<h3>Health Check Results</h3>"
        + reasons_html
        + "<h3>Snapshots By Sport (health report)</h3>"
        + (_records_table(by_sport) if by_sport else '<div class="empty">No snapshots collected yet.</div>')
        + "<h3>Missed Collection Days</h3>"
        + missed_html
        + "<h3>Latest Errors / Warnings</h3>"
        + issues_html
        + "<h3>Leagues With No Recent Snapshots</h3>"
        + (
            _records_table(stale)
            if stale
            else '<div class="empty">All enabled leagues have recent snapshots.</div>'
        )
        + '<p class="note">Missed collection windows cannot be backfilled: The Odds API does not '
        "provide historical odds on the current plan, so odds from missed days are permanently lost.</p>"
    )


MANUAL_REVIEW_CSV_LINKS = [
    ("player_prop_manual_review.md", "Manual review report (MD)"),
    ("player_prop_counts_by_league_prop.csv", "Counts by league + prop CSV"),
    ("player_prop_counts_by_bookmaker.csv", "Counts by bookmaker CSV"),
    ("player_prop_missing_fields.csv", "Missing fields CSV"),
    ("nba_player_prop_review.csv", "NBA player/prop review CSV"),
    ("latest_collected_props.csv", "Latest collected props CSV"),
]


def _build_manual_review_section(report_path: Path) -> str:
    review = _read_json(report_path / "player_prop_manual_review_summary.json")
    if not review:
        return (
            "<h2>Manual Review</h2>"
            '<div class="empty">No manual review yet. Run scripts/build_player_prop_manual_review.py.</div>'
        )
    match_rate = review.get("nba_player_match_rate")
    try:
        match_rate_text = f"{float(match_rate):.1%}"
    except (TypeError, ValueError):
        match_rate_text = "n/a"
    missing_fields = review.get("missing_fields", {}) if isinstance(review.get("missing_fields"), dict) else {}
    missing_rows = [{"field": field, "missing": count} for field, count in missing_fields.items()]
    nba_missing = review.get("nba_missing_fields", {}) if isinstance(review.get("nba_missing_fields"), dict) else {}
    missing_rows += [
        {"field": f"NBA {field} ({nba_missing.get('source', 'normalized')})", "missing": nba_missing.get(field, 0)}
        for field in ("player_id", "canonical_game_key")
    ]
    not_collected_leagues = review.get("configured_leagues_not_collected") or []
    missing_props = review.get("configured_prop_types_not_collected") or {}
    missing_props_html = (
        "<ul>"
        + "".join(
            f"<li>{html.escape(str(league))}: {html.escape(', '.join(props))}</li>"
            for league, props in sorted(missing_props.items())
        )
        + "</ul>"
        if missing_props
        else '<div class="empty">Every configured prop type was collected at least once.</div>'
    )
    reasons = review.get("possible_missing_reasons") or []
    reasons_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reasons) + "</ul>"
        if reasons
        else '<div class="empty">Nothing missing: all configured leagues and prop types were collected.</div>'
    )
    checks = review.get("nba_usability_checks") or []
    checks_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(check))}</li>" for check in checks) + "</ul>" if checks else ""
    )
    links = "".join(
        f'<a href="{html.escape(filename)}" download>{html.escape(label)}</a>'
        for filename, label in MANUAL_REVIEW_CSV_LINKS
        if (report_path / filename).exists()
    )
    return (
        "<h2>Manual Review</h2>"
        + _summary_grid(
            [
                ("Snapshots reviewed", review.get("total_snapshots", 0)),
                ("Latest snapshot", review.get("latest_snapshot_time_utc") or "n/a"),
                ("Closing-like snapshots", review.get("closing_like_snapshots", 0)),
                ("Leagues collected", len(review.get("snapshots_by_league") or {})),
                ("NBA players collected", len(review.get("nba_players") or [])),
                ("NBA player match rate", match_rate_text),
                ("Leagues not collected yet", len(not_collected_leagues)),
                (
                    "NBA usable for grading",
                    "YES" if review.get("nba_usable_for_grading") else "NOT YET",
                ),
            ]
        )
        + f'<p class="note">{html.escape(str(review.get("nba_usability_verdict") or ""))}</p>'
        + checks_html
        + f'<div class="download-links">{links}</div>'
        + "<h3>Missing Field Counts</h3>"
        + _records_table(missing_rows)
        + "<h3>Configured Leagues Not Collected</h3>"
        + (
            "<ul>" + "".join(f"<li>{html.escape(str(league))}</li>" for league in not_collected_leagues) + "</ul>"
            if not_collected_leagues
            else '<div class="empty">Every enabled league has at least one snapshot.</div>'
        )
        + "<h3>Configured Prop Types Not Collected</h3>"
        + missing_props_html
        + "<h3>Possible Reasons</h3>"
        + reasons_html
    )


MARKET_QUALITY_CSV_LINKS = [
    ("player_prop_market_quality.md", "Market quality report (MD)"),
    ("player_prop_line_quality.csv", "Line quality per market CSV"),
    ("player_prop_likely_main_lines.csv", "Likely main lines CSV"),
    ("player_prop_possible_alt_lines.csv", "Possible alternate lines CSV"),
    ("player_prop_bookmaker_coverage.csv", "Bookmaker coverage CSV"),
    ("player_prop_closing_snapshot_coverage.csv", "Closing snapshot coverage CSV"),
]


def _build_market_quality_section(report_path: Path) -> str:
    quality = _read_json(report_path / "player_prop_market_quality_summary.json")
    if not quality:
        return (
            "<h2>Market Quality</h2>"
            '<div class="empty">No market quality audit yet. Run scripts/build_player_prop_market_quality.py.</div>'
        )
    closing = quality.get("closing_coverage", {}) if isinstance(quality.get("closing_coverage"), dict) else {}
    flag_counts = quality.get("flag_counts", {}) if isinstance(quality.get("flag_counts"), dict) else {}
    flag_rows = [{"flag": flag, "markets": count} for flag, count in flag_counts.items()]
    nba_rate = (quality.get("closing_coverage") or {}).get("closing_market_rate_by_league", {}).get("NBA")
    try:
        nba_rate_text = f"{float(nba_rate):.1%}"
    except (TypeError, ValueError):
        nba_rate_text = "n/a"
    best_books = quality.get("nba_best_bookmakers") or []
    links = "".join(
        f'<a href="{html.escape(filename)}" download>{html.escape(label)}</a>'
        for filename, label in MARKET_QUALITY_CSV_LINKS
        if (report_path / filename).exists()
    )
    return (
        "<h2>Market Quality</h2>"
        + _summary_grid(
            [
                ("Markets audited", quality.get("total_markets_audited", 0)),
                ("Likely main lines", quality.get("likely_main_lines", 0)),
                ("Possible alt-line markets", quality.get("possible_alt_line_markets", 0)),
                ("Wide line range warnings", quality.get("wide_line_range_markets", 0)),
                ("Missing price warnings", quality.get("missing_price_markets", 0)),
                ("Closing-like snapshots", closing.get("total_closing_snapshots", 0)),
                ("Markets without closing", closing.get("markets_without_closing", 0)),
                ("NBA closing market coverage", nba_rate_text),
                (
                    "NBA clean for modeling later",
                    "YES" if quality.get("nba_clean_enough_for_modeling") else "NOT YET",
                ),
            ]
        )
        + f'<p class="note">{html.escape(str(quality.get("nba_modeling_verdict") or ""))}</p>'
        + f'<p class="note">{html.escape(str(closing.get("clv_readiness_verdict") or ""))}</p>'
        + '<p class="note">Research-only audit: alternate lines are flagged for review, never deleted. '
        "No models, recommendations, approved bets, or parlays.</p>"
        + f'<div class="download-links">{links}</div>'
        + "<h3>Quality Flags</h3>"
        + (_records_table(flag_rows) if flag_rows else '<div class="empty">No flags computed.</div>')
        + "<h3>Best-Covered NBA Bookmakers</h3>"
        + (
            _records_table(best_books)
            if best_books
            else '<div class="empty">No NBA bookmaker coverage yet.</div>'
        )
    )


CLV_READINESS_LINKS = [
    ("nba_prop_closing_collection_plan.md", "Collection plan (MD)"),
    ("nba_prop_closing_collection_plan.json", "Collection plan (JSON)"),
    ("nba_prop_closing_coverage.csv", "Closing coverage CSV"),
    ("nba_prop_clv_readiness_summary.json", "CLV readiness (JSON)"),
]


def _build_clv_readiness_section(report_path: Path) -> str:
    readiness = _read_json(report_path / "nba_prop_clv_readiness_summary.json")
    plan = _read_json(report_path / "nba_prop_closing_collection_plan.json")
    if not readiness:
        return (
            "<h2>NBA CLV Readiness</h2>"
            '<div class="empty">No CLV readiness report yet. Run scripts/build_nba_collection_plan.py.</div>'
        )

    market_stats = (
        readiness.get("market_closing_coverage", {})
        if isinstance(readiness.get("market_closing_coverage"), dict)
        else {}
    )

    def _pct_or_na(value: object) -> str:
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return "n/a"

    games_with = readiness.get("games_with_closing_snapshots") or []
    games_missing = readiness.get("games_missing_closing_snapshots") or []
    clv_now = bool(readiness.get("clv_possible_now"))
    clv_later = bool(readiness.get("clv_possible_later"))
    if clv_now:
        clv_text = "YES"
    elif clv_later:
        clv_text = "NOT YET (still achievable)"
    else:
        clv_text = "NO"

    game_rows = []
    missed_windows: list[str] = []
    for game in plan.get("games", []) if isinstance(plan, dict) else []:
        game_rows.append(
            {
                "game": game.get("game"),
                "tip (UTC)": game.get("game_start_time"),
                "minutes to tip": game.get("minutes_until_game"),
                "timing": game.get("timing_classification"),
                "windows hit": ", ".join(game.get("windows_hit") or []) or "(none)",
                "windows missed": ", ".join(game.get("windows_missed") or []) or "(none)",
                "collect now": "YES" if game.get("collection_needed_now") else "no",
                "CLV possible": "YES" if game.get("clv_possible") else "NO",
            }
        )
        for window in game.get("windows_missed") or []:
            missed_windows.append(f"{game.get('game')}: {window}")

    warnings = readiness.get("warnings") or []
    warnings_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) + "</ul>"
        if warnings
        else '<div class="empty">No CLV readiness warnings.</div>'
    )
    links = "".join(
        f'<a href="{html.escape(filename)}" download>{html.escape(label)}</a>'
        for filename, label in CLV_READINESS_LINKS
        if (report_path / filename).exists()
    )
    return (
        "<h2>NBA CLV Readiness</h2>"
        + _summary_grid(
            [
                ("NBA closing-like snapshots", readiness.get("nba_closing_like_snapshots", 0)),
                ("Closing market coverage", _pct_or_na(market_stats.get("nba_closing_market_rate"))),
                ("Main-line closing coverage", _pct_or_na(market_stats.get("nba_main_line_closing_rate"))),
                ("Games with closing snapshots", len(games_with)),
                ("Games missing closing snapshots", len(games_missing)),
                ("Missed windows", len(missed_windows)),
                ("Last NBA collection", readiness.get("last_nba_snapshot_time_utc") or "never"),
                (
                    "Next recommended collection (UTC)",
                    readiness.get("next_recommended_collection_time_utc") or "n/a",
                ),
                ("Collection needed now", "YES" if readiness.get("collection_needed_now") else "no"),
                ("NBA CLV possible", clv_text),
            ]
        )
        + f'<p class="note">{html.escape(str(readiness.get("verdict") or ""))}</p>'
        + f'<div class="download-links">{links}</div>'
        + "<h3>Game Collection Plan</h3>"
        + (
            _records_table(game_rows)
            if game_rows
            else '<div class="empty">No NBA games with known start times in the snapshot store yet.</div>'
        )
        + "<h3>Missed Windows</h3>"
        + (
            "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in missed_windows) + "</ul>"
            if missed_windows
            else '<div class="empty">No missed collection windows for games in the plan.</div>'
        )
        + "<h3>Warnings</h3>"
        + warnings_html
        + '<p class="note">Missed windows are unrecoverable: The Odds API does not provide '
        "historical odds on the current plan. CLV stays research-only.</p>"
    )


SETTLEMENT_OUTCOMES_LINKS = [
    ("player_prop_settlement_outcomes.md", "Settlement outcomes (MD)"),
    ("player_prop_settlement_outcomes.csv", "Settlement outcomes CSV"),
    ("player_prop_settlement_outcomes_summary.json", "Settlement outcomes (JSON)"),
]

PROP_CLV_LINKS = [
    ("player_prop_clv.md", "CLV report (MD)"),
    ("player_prop_clv.csv", "CLV per market CSV"),
    ("player_prop_clv_by_bookmaker.csv", "CLV by bookmaker CSV"),
    ("player_prop_clv_by_prop_type.csv", "CLV by prop type CSV"),
    ("player_prop_clv_summary.json", "CLV summary (JSON)"),
]

QUALITY_GATES_LINKS = [
    ("player_prop_data_quality_gates.md", "Data quality gates (MD)"),
    ("player_prop_data_quality_gates.json", "Data quality gates (JSON)"),
]

RESEARCH_STATUS_LINKS = [
    ("nba_player_stat_baseline_blockers.md", "Baseline blocker report (MD)"),
    ("nba_player_stat_baseline.md", "Baseline report (MD)"),
    ("nba_player_stat_baseline_predictions.csv", "Baseline predictions CSV"),
    ("nba_player_prop_research_signals.csv", "Research signals CSV"),
    ("nba_player_prop_research.md", "Research signals (MD)"),
    ("nba_player_prop_research_parlays.csv", "Research parlays CSV"),
    ("nba_player_prop_research_parlays.md", "Research parlays (MD)"),
]


def _download_links(report_path: Path, links: list[tuple[str, str]]) -> str:
    rendered = "".join(
        f'<a href="{html.escape(filename)}" download>{html.escape(label)}</a>'
        for filename, label in links
        if (report_path / filename).exists()
    )
    return f'<div class="download-links">{rendered}</div>' if rendered else ""


def _build_settlement_outcomes_section(report_path: Path) -> str:
    outcomes = _read_json(report_path / "player_prop_settlement_outcomes_summary.json")
    if not outcomes:
        return (
            "<h2>Settlement Outcomes</h2>"
            '<div class="empty">No settlement outcome report yet. '
            "Run scripts/build_player_prop_settlement_outcomes.py.</div>"
        )
    overall = outcomes.get("overall", {}) if isinstance(outcomes.get("overall"), dict) else {}
    pending_games = outcomes.get("pending_games") or []
    warnings = outcomes.get("warnings") or []
    by_prop = outcomes.get("by_prop_type") or []
    by_book = outcomes.get("by_bookmaker") or []

    def _pct_rate(value: object) -> str:
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return "n/a"

    return (
        "<h2>Settlement Outcomes</h2>"
        + _summary_grid(
            [
                ("Settled props", outcomes.get("settled_props", 0)),
                ("Pending props", outcomes.get("pending_props", 0)),
                ("Over won", overall.get("over_won", 0)),
                ("Under won", overall.get("under_won", 0)),
                ("Pushes", overall.get("push", 0)),
                ("Over win rate", _pct_rate(overall.get("over_win_rate"))),
                ("Under win rate", _pct_rate(overall.get("under_win_rate"))),
                ("Small sample", "YES" if outcomes.get("small_sample") else "no"),
            ]
        )
        + _download_links(report_path, SETTLEMENT_OUTCOMES_LINKS)
        + "<h3>By Prop Type</h3>"
        + (_records_table(by_prop) if by_prop else '<div class="empty">No settled props yet.</div>')
        + "<h3>By Bookmaker</h3>"
        + (_records_table(by_book) if by_book else '<div class="empty">No settled props yet.</div>')
        + "<h3>Pending Games</h3>"
        + (
            _records_table(pending_games)
            if pending_games
            else '<div class="empty">No games waiting on results.</div>'
        )
        + "<h3>Warnings</h3>"
        + (
            "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) + "</ul>"
            if warnings
            else '<div class="empty">No warnings.</div>'
        )
    )


def _build_prop_clv_section(report_path: Path) -> str:
    clv = _read_json(report_path / "player_prop_clv_summary.json")
    if not clv:
        return (
            "<h2>Prop CLV</h2>"
            '<div class="empty">No CLV report yet. Run scripts/build_player_prop_clv.py.</div>'
        )
    warnings = clv.get("warnings") or []
    return (
        "<h2>Prop CLV</h2>"
        + _summary_grid(
            [
                ("CLV ready", "YES" if clv.get("clv_ready") else "NOT YET"),
                ("Markets with CLV", clv.get("markets_with_clv", 0)),
                ("NBA markets with CLV", clv.get("nba_markets_with_clv", 0)),
                ("Same-line price CLV markets", clv.get("price_clv_comparable_markets", 0)),
                ("Line-changed markets", clv.get("line_changed_markets", 0)),
                ("NBA closing-like snapshots", clv.get("nba_closing_like_snapshots", 0)),
                (
                    "Avg over CLV",
                    clv.get("avg_clv_over_pct") if clv.get("avg_clv_over_pct") is not None else "n/a",
                ),
                (
                    "Avg under CLV",
                    clv.get("avg_clv_under_pct") if clv.get("avg_clv_under_pct") is not None else "n/a",
                ),
            ]
        )
        + f'<p class="note">{html.escape(str(clv.get("verdict") or ""))}</p>'
        + _download_links(report_path, PROP_CLV_LINKS)
        + (
            "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) + "</ul>"
            if warnings
            else ""
        )
        + '<p class="note">CLV compares early vs closing prices (settlement not required). '
        "Price CLV is only computed when the line did not move; line changes are flagged "
        "separately. Research-only measurement, not evidence of edge.</p>"
    )


def _build_quality_gates_section(report_path: Path) -> str:
    gates = _read_json(report_path / "player_prop_data_quality_gates.json")
    if not gates:
        return (
            "<h2>Data Quality Gates</h2>"
            '<div class="empty">No gates report yet. '
            "Run scripts/build_player_prop_data_quality_gates.py.</div>"
        )
    metrics = gates.get("metrics", {}) if isinstance(gates.get("metrics"), dict) else {}
    blockers = gates.get("blockers") or []
    check_rows = []
    for rung, checks in (gates.get("checks") or {}).items():
        for check in checks or []:
            check_rows.append(
                {
                    "rung": rung,
                    "check": check.get("check"),
                    "value": check.get("value"),
                    "threshold": check.get("threshold"),
                    "passed": "PASS" if check.get("passed") else "FAIL",
                }
            )
    return (
        "<h2>Data Quality Gates</h2>"
        + _summary_grid(
            [
                ("Status", str(gates.get("status", "unknown")).upper()),
                ("NBA snapshots", metrics.get("nba_snapshots", 0)),
                ("Settled props", metrics.get("settled_props", 0)),
                ("CLV markets", metrics.get("clv_markets", 0)),
                ("Blockers", len(blockers)),
            ]
        )
        + f'<p class="note">Ladder: {html.escape(" -> ".join(gates.get("status_ladder") or []))}. '
        "Gates qualify DATA for research-only modeling experiments; they never approve betting.</p>"
        + _download_links(report_path, QUALITY_GATES_LINKS)
        + "<h3>Checks</h3>"
        + (_records_table(check_rows) if check_rows else '<div class="empty">No checks computed.</div>')
        + "<h3>Blockers</h3>"
        + (
            "<ul>" + "".join(f"<li>{html.escape(str(b))}</li>" for b in blockers) + "</ul>"
            if blockers
            else '<div class="empty">No blockers.</div>'
        )
    )


def _build_research_status_section(report_path: Path) -> str:
    gates = _read_json(report_path / "player_prop_data_quality_gates.json")
    status = str(gates.get("status", "unknown")) if gates else "unknown"
    baseline_built = (report_path / "nba_player_stat_baseline_summary.json").exists()
    signals_built = (report_path / "nba_player_prop_research_summary.json").exists()
    parlays_built = (report_path / "nba_player_prop_research_parlay_summary.json").exists()
    blocker_report = (report_path / "nba_player_stat_baseline_blockers.md").exists()

    def _state(built: bool) -> str:
        return "BUILT (research-only)" if built else "BLOCKED"

    return (
        "<h2>Baseline &amp; Research Signals Status</h2>"
        + _summary_grid(
            [
                ("Gate status", status.upper()),
                ("Player stat baseline", _state(baseline_built)),
                ("Research prop signals", _state(signals_built)),
                ("Research prop parlays", _state(parlays_built)),
            ]
        )
        + (
            '<p class="note">Baseline modeling is blocked by the data quality gates: see the '
            "blocker report below for exactly what is missing and how to unblock it.</p>"
            if blocker_report and not baseline_built
            else ""
        )
        + _download_links(report_path, RESEARCH_STATUS_LINKS)
        + '<p class="note">Everything on this page is research-only. Approved bets and approved '
        "parlays remain blocked unless the separate proof gates explicitly pass.</p>"
    )


NBA_REVIEW_LINKS = [
    ("nba_main_lines_review.csv", "NBA main lines review CSV"),
    ("nba_alt_lines_review.csv", "NBA alt lines review CSV"),
    ("nba_bookmaker_comparison.csv", "NBA bookmaker comparison CSV"),
    ("nba_prop_board_latest.csv", "NBA prop board (latest) CSV"),
]

NEXT_ACTION_LINKS = [
    ("next_action_report.md", "Next action report (MD)"),
    ("next_action_report.json", "Next action report (JSON)"),
]

QUOTA_LINKS = [
    ("odds_api_quota_report.md", "Quota report (MD)"),
    ("odds_api_quota_report.json", "Quota report (JSON)"),
]

ALL_SPORTS_LINKS = [
    ("all_sports_prop_readiness.md", "All-sports readiness (MD)"),
    ("all_sports_prop_readiness.json", "All-sports readiness (JSON)"),
]


def _build_next_action_section(report_path: Path) -> str:
    report = _read_json(report_path / "next_action_report.json")
    if not report:
        return (
            "<h2>Next Action</h2>"
            '<div class="empty">No next-action report yet. '
            "Run scripts/build_next_action_report.py.</div>"
        )
    next_action = report.get("next_action") or {}
    action_rows = [
        {
            "priority": a.get("priority"),
            "action": a.get("action"),
            "reason": a.get("reason"),
            "command": a.get("command") or "",
        }
        for a in (report.get("actions") or [])
    ]
    command_html = (
        f"<pre>{html.escape(str(next_action.get('command')))}</pre>"
        if next_action.get("command")
        else ""
    )
    return (
        "<h2>Next Action</h2>"
        + '<div class="status">'
        + f"<strong>{html.escape(str(next_action.get('action') or 'n/a'))}</strong>"
        + f"<p>{html.escape(str(next_action.get('reason') or ''))}</p>"
        + command_html
        + "</div>"
        + _download_links(report_path, NEXT_ACTION_LINKS)
        + "<h3>All Actions (by priority)</h3>"
        + (_records_table(action_rows) if action_rows else '<div class="empty">No actions.</div>')
        + f'<p class="note">Generated {html.escape(str(report.get("generated_at_utc") or "n/a"))}. '
        "Research-only data-pipeline actions; approved bets/parlays remain blocked.</p>"
    )


def _build_quota_section(report_path: Path) -> str:
    report = _read_json(report_path / "odds_api_quota_report.json")
    if not report:
        return (
            "<h2>Odds API Quota</h2>"
            '<div class="empty">No quota report yet. '
            "Run scripts/build_odds_api_quota_report.py.</div>"
        )
    usage = report.get("usage") or {}
    risk = report.get("risk_assessment") or {}
    leagues = report.get("league_recommendations") or {}
    warnings = list(risk.get("warnings") or [])
    return (
        "<h2>Odds API Quota</h2>"
        + _summary_grid(
            [
                ("Quota remaining", usage.get("quota_remaining", "n/a")),
                ("Requests/run (est.)", usage.get("avg_requests_per_run", "n/a")),
                ("Runs today", usage.get("runs_today", 0)),
                ("Runs this month", usage.get("runs_this_month", 0)),
                ("Risk", str(risk.get("risk", "unknown")).upper()),
                ("Max runs/day (rec.)", risk.get("recommended_max_runs_per_day", "n/a")),
                (
                    "Soccer capped",
                    "YES" if leagues.get("soccer_should_remain_capped") else "active",
                ),
            ]
        )
        + f'<p class="note">{html.escape(str(risk.get("detail") or ""))}</p>'
        + _download_links(report_path, QUOTA_LINKS)
        + (
            "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) + "</ul>"
            if warnings
            else ""
        )
        + "<h3>Leagues Consuming Requests</h3>"
        + (
            _records_table(usage.get("leagues_consuming_requests") or [])
            if usage.get("leagues_consuming_requests")
            else '<div class="empty">No league usage recorded this month.</div>'
        )
    )


CROSS_SPORT_LINKS = [
    ("cross_sport_collection_coverage.md", "Cross-sport coverage (MD)"),
    ("cross_sport_collection_coverage.csv", "Cross-sport coverage CSV"),
    ("cross_sport_collection_coverage_summary.json", "Cross-sport coverage (JSON)"),
    ("odds_api_available_sports.md", "Odds API sports (MD)"),
]


def _build_cross_sport_coverage_section(report_path: Path) -> str:
    report = _read_json(report_path / "cross_sport_collection_coverage_summary.json")
    if not report:
        return (
            "<h2>Cross-Sport Coverage</h2>"
            '<div class="empty">No coverage audit yet. '
            "Run scripts/build_cross_sport_collection_coverage.py.</div>"
        )
    leagues = report.get("leagues") or []
    by_status = report.get("leagues_by_status") or {}
    warnings = report.get("warnings") or []
    collecting = by_status.get("collecting", [])
    quota_skipped = by_status.get("configured_skipped_quota", [])
    not_collecting = [r["league"] for r in leagues if r.get("status") != "collecting"]
    groups = report.get("sport_groups") or []
    covered_groups = [g["sport_group"] for g in groups if g.get("all_configured")]
    league_rows = [
        {
            "league": r.get("league"),
            "group": r.get("sport_group"),
            "status": r.get("status"),
            "snapshots": r.get("snapshots_total"),
            "last_24h": r.get("snapshots_last_24h"),
            "latest_snapshot": r.get("latest_snapshot_time") or "-",
            "prop_types": ", ".join(r.get("prop_types_collected") or []) or "-",
            "priority": r.get("priority"),
            "quota_blocked": "YES" if r.get("quota_blocked_last_run") else "no",
        }
        for r in leagues
    ]
    return (
        "<h2>Cross-Sport Coverage</h2>"
        + _summary_grid(
            [
                ("Sport groups covered", f"{len(covered_groups)}/{len(groups)}"),
                ("Collecting leagues", len(collecting)),
                ("Configured, not collecting", len(not_collecting)),
                ("Quota-skipped last run", len(quota_skipped)),
            ]
        )
        + f'<p class="note">Collecting now: {html.escape(", ".join(collecting) or "(none)")}. '
        f'Quota-skipped: {html.escape(", ".join(quota_skipped) or "(none)")}.</p>'
        + _download_links(report_path, CROSS_SPORT_LINKS)
        + "<h3>Leagues</h3>"
        + (_records_table(league_rows) if league_rows else '<div class="empty">No leagues.</div>')
        + "<h3>Warnings</h3>"
        + (
            "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) + "</ul>"
            if warnings
            else '<div class="empty">No coverage warnings.</div>'
        )
        + '<p class="note">Quota-skipped leagues stay configured and resume automatically '
        "when quota and the per-run league cap allow. Research-only.</p>"
    )


def _build_all_sports_readiness_section(report_path: Path) -> str:
    report = _read_json(report_path / "all_sports_prop_readiness.json")
    if not report:
        return (
            "<h2>All-Sports Prop Readiness</h2>"
            '<div class="empty">No readiness report yet. '
            "Run scripts/build_all_sports_prop_readiness.py.</div>"
        )
    rows = [
        {
            "league": r.get("league"),
            "snapshots": r.get("snapshots_collected"),
            "closing": r.get("closing_like_snapshots"),
            "prop_types": ", ".join(r.get("prop_types_collected") or []),
            "model_readiness": r.get("model_readiness_status"),
            "next_adapter": r.get("next_required_adapter"),
        }
        for r in (report.get("leagues") or [])
    ]
    return (
        "<h2>All-Sports Prop Readiness</h2>"
        + _download_links(report_path, ALL_SPORTS_LINKS)
        + (_records_table(rows) if rows else '<div class="empty">No leagues configured.</div>')
        + f'<p class="note">{html.escape(str(report.get("policy") or ""))}</p>'
    )


def _build_nba_review_exports_section(report_path: Path) -> str:
    links = _download_links(report_path, NBA_REVIEW_LINKS)
    if not links:
        return (
            "<h2>NBA Review Exports</h2>"
            '<div class="empty">No review CSVs yet. '
            "Run scripts/build_nba_prop_review_exports.py.</div>"
        )
    summary = _read_json(report_path / "nba_prop_review_exports_summary.json")
    rows = summary.get("rows") or {}
    return (
        "<h2>NBA Review Exports (Excel-friendly)</h2>"
        + _summary_grid(
            [
                ("Main-line rows", rows.get("nba_main_lines_review.csv", "n/a")),
                ("Alt-line rows", rows.get("nba_alt_lines_review.csv", "n/a")),
                ("Bookmaker comparison rows", rows.get("nba_bookmaker_comparison.csv", "n/a")),
                ("Prop board rows", rows.get("nba_prop_board_latest.csv", "n/a")),
            ]
        )
        + links
        + '<p class="note">One row per market with player, prop type, likely main line, '
        "latest over/under prices, bookmaker, snapshot time, closing-like flag, line quality "
        "label, alt-line flag, settlement result, and game key.</p>"
    )


def _build_odds_sources_section(report_path: Path) -> str:
    """Multi-source status: SportsGameOdds, API-Sports, usage, cross-source."""

    sgo_probe = _read_json(report_path / "sportsgameodds_probe_summary.json")
    sgo_collect = _read_json(report_path / "sportsgameodds_collection_summary.json")
    sgo_hist = _read_json(report_path / "sportsgameodds_historical_prop_probe_summary.json")
    apisports = _read_json(report_path / "apisports_probe_summary.json")
    usage = _read_json(report_path / "odds_source_usage_summary.json")
    cross = _read_json(report_path / "cross_source_prop_comparison_summary.json")
    comparison = _read_json(report_path / "odds_source_comparison.json")

    parts: list[str] = ["<h2>Odds Sources (multi-source status)</h2>"]

    # SportsGameOdds status.
    sgo_quota = (sgo_collect.get("quota") or {}) if isinstance(sgo_collect, dict) else {}
    sgo_usage = (sgo_probe.get("usage") or {}) if isinstance(sgo_probe, dict) else {}
    sgo_totals = (sgo_collect.get("totals") or {}) if isinstance(sgo_collect, dict) else {}
    if sgo_probe or sgo_collect:
        parts.append("<h3>SportsGameOdds</h3>")
        parts.append(
            _summary_grid(
                [
                    ("Key works", sgo_probe.get("key_works", "n/a")),
                    ("Props visible (probe)", sgo_probe.get("player_prop_markets_visible", "n/a")),
                    ("Tier", sgo_usage.get("tier", "n/a")),
                    ("Last collection", f"{sgo_collect.get('run_id', 'n/a')} ({sgo_collect.get('status', 'n/a')})"),
                    ("Events last run", sgo_totals.get("events", 0)),
                    ("Rows added (shared CSV)", sgo_totals.get("snapshots_added_shared", 0)),
                    ("Monthly entities left", sgo_quota.get("entities_remaining_after", "n/a")),
                    ("Entity guard floor", sgo_quota.get("guard_floor", "n/a")),
                ]
            )
        )
    else:
        parts.append("<h3>SportsGameOdds</h3>"
                     '<div class="empty">No probe yet. Run scripts/probe_sportsgameodds.py.</div>')

    # SportsGameOdds historical prop availability (research probe).
    if sgo_hist:
        hist_verdict = (sgo_hist.get("verdict") or {}) if isinstance(sgo_hist, dict) else {}
        parts.append("<h3>SportsGameOdds historical props (research probe)</h3>")
        parts.append(
            _summary_grid(
                [
                    ("Historical props accessible", hist_verdict.get("historical_player_props_accessible", "n/a")),
                    ("Closing prices for props", hist_verdict.get("closing_prices_available_for_props", "n/a")),
                    ("Settled outcomes", hist_verdict.get("settlement_results_available", "n/a")),
                    ("Oldest game observed", hist_verdict.get("oldest_event_date", "n/a")),
                    ("Close-field horizon", hist_verdict.get("oldest_close_field_date", "n/a")),
                    ("Open-price horizon", hist_verdict.get("oldest_open_price_date", "n/a")),
                    ("Probe entity cost", hist_verdict.get("entity_cost_estimate", "n/a")),
                ]
            )
        )
        if hist_verdict.get("recommended_next_step"):
            parts.append(f'<p class="note">{hist_verdict["recommended_next_step"]}</p>')

    # API-Sports status.
    if apisports:
        parts.append("<h3>API-Sports (probe-only)</h3>")
        basketball_status = ((apisports.get("status_by_api") or {}).get("basketball") or {})
        parts.append(
            _summary_grid(
                [
                    ("Key works", apisports.get("key_works", "n/a")),
                    ("Plan", basketball_status.get("plan", "n/a")),
                    ("Player props", str(apisports.get("player_props_available", "unknown"))),
                    ("Daily request limit", basketball_status.get("requests_limit_day", "n/a")),
                ]
            )
        )
        if apisports.get("plan_restriction"):
            parts.append(f'<p class="note">BLOCKED: {apisports["plan_restriction"]}</p>')
        if apisports.get("recommended_next_action"):
            parts.append(f'<p class="note">{apisports["recommended_next_action"]}</p>')
    else:
        parts.append("<h3>API-Sports (probe-only)</h3>"
                     '<div class="empty">No probe yet. Run scripts/probe_apisports.py.</div>')

    # Per-source usage table.
    sources = (usage.get("sources") or {}) if isinstance(usage, dict) else {}
    if sources:
        rows = []
        for name, info in sources.items():
            snaps = info.get("snapshots") or {}
            quota_info = info.get("quota") or {}
            quota_text = "; ".join(
                f"{k}={v}" for k, v in quota_info.items() if v is not None and k != "note"
            )
            rows.append(
                {
                    "source": name,
                    "status": info.get("status"),
                    "snapshots": snaps.get("snapshots", 0),
                    "closing": snaps.get("closing_snapshots", 0),
                    "latest_snapshot": snaps.get("latest_snapshot_utc") or "n/a",
                    "quota": quota_text or "n/a",
                    "errors": len(info.get("errors") or []),
                }
            )
        parts.append("<h3>Source usage &amp; quota</h3>")
        parts.append(_records_table(rows))

    # Primary/backup per league.
    per_league = (usage.get("primary_backup_by_league") or {}) if isinstance(usage, dict) else {}
    if per_league:
        rows = [
            {
                "league": league,
                "primary": info.get("primary"),
                "backup": info.get("backup") or "-",
                "odds_api_rows": info.get("odds_api_rows"),
                "sgo_rows": info.get("sportsgameodds_rows"),
                "note": info.get("note"),
            }
            for league, info in per_league.items()
        ]
        parts.append("<h3>Primary / backup source by league</h3>")
        parts.append(_records_table(rows))

    # Cross-source comparison.
    parts.append("<h3>Cross-source comparison</h3>")
    if cross:
        overlap = cross.get("overlap_found")
        parts.append(f"<p>Overlap found: <strong>{overlap}</strong></p>")
        if cross.get("reason"):
            parts.append(f'<p class="note">{cross["reason"]}</p>')
        pair_rows = [
            {
                "pair": pair.get("pair"),
                "shared_games": pair.get("shared_games"),
                "overlap_rows": pair.get("exact_market_overlap_rows"),
                "line_disagreements": pair.get("line_disagreement_markets"),
                "mean_abs_over_diff": pair.get("mean_abs_over_price_diff"),
            }
            for pair in cross.get("pairs") or []
        ]
        if pair_rows:
            parts.append(_records_table(pair_rows))
    else:
        parts.append('<div class="empty">No cross-source comparison yet. '
                     "Run scripts/build_cross_source_prop_comparison.py.</div>")

    # Warnings/blockers across the new sources.
    warnings: list[str] = []
    for blocker in (sgo_probe.get("blockers") or []):
        warnings.append(f"sportsgameodds probe: {blocker}")
    for blocker in (sgo_collect.get("blockers") or []):
        warnings.append(f"sportsgameodds collection: {blocker}")
    if apisports.get("plan_restriction"):
        warnings.append(f"apisports: {apisports['plan_restriction']}")
    if warnings:
        parts.append("<h3>Source warnings / blockers</h3>")
        parts.append("<ul>" + "".join(f"<li>{w}</li>" for w in warnings) + "</ul>")

    headline = (comparison.get("headline") or {}) if isinstance(comparison, dict) else {}
    if headline:
        parts.append(
            f'<p class="note">Source plan: priority-1 supplement = '
            f"{headline.get('priority_1_supplement', 'n/a')}; probe-only = "
            f"{', '.join(headline.get('probe_only') or []) or 'none'}.</p>"
        )

    parts.append(
        '<div class="download-links">'
        '<a href="odds_source_comparison.md" download>Source comparison MD</a>'
        '<a href="odds_source_comparison.json" download>Source comparison JSON</a>'
        '<a href="odds_source_adapter_plan.csv" download>Adapter plan CSV</a>'
        '<a href="odds_source_usage_summary.md" download>Usage summary MD</a>'
        '<a href="odds_source_usage_summary.json" download>Usage summary JSON</a>'
        '<a href="cross_source_prop_comparison.md" download>Cross-source MD</a>'
        '<a href="cross_source_prop_comparison.csv" download>Cross-source CSV</a>'
        '<a href="cross_source_prop_comparison_summary.json" download>Cross-source JSON</a>'
        '<a href="sportsgameodds_probe.md" download>SGO probe MD</a>'
        '<a href="sportsgameodds_collection.md" download>SGO collection MD</a>'
        '<a href="sportsgameodds_collection_summary.json" download>SGO collection JSON</a>'
        '<a href="sportsgameodds_historical_prop_probe.md" download>SGO historical probe MD</a>'
        '<a href="sportsgameodds_historical_prop_probe_summary.json" download>SGO historical probe JSON</a>'
        '<a href="apisports_probe.md" download>API-Sports probe MD</a>'
        '<a href="apisports_probe_summary.json" download>API-Sports probe JSON</a>'
        "</div>"
    )
    return "".join(parts)


def _dashboard_badge(text: Any, tone: str = "gray") -> str:
    tone = tone if tone in {"green", "yellow", "red", "gray", "blue"} else "gray"
    return f'<span class="status-badge {tone}">{html.escape(str(text))}</span>'


def _dashboard_tone(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"ready", "healthy", "success", "ok", "available", "passed", "yes", "true", "modeling_experiment_ready"}:
        return "green"
    if text in {"warning", "waiting", "pending", "settlement_ready", "collection_ready", "clv_ready", "not yet"}:
        return "yellow"
    if text in {"blocked", "error", "failed", "unhealthy", "not_ready", "no", "false"}:
        return "red"
    return "gray"


def _dashboard_list(items: list[Any], empty: str = "not available yet") -> str:
    if not items:
        return f'<div class="empty">{html.escape(empty)}</div>'
    return "<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in items) + "</ul>"


def _dashboard_counts(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[label, "rows"])
    counts = (
        frame[column]
        .fillna("(missing)")
        .astype(str)
        .value_counts()
        .rename_axis(label)
        .reset_index(name="rows")
    )
    return counts


def _dashboard_bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "t"})


def _dashboard_latest_warning(report_path: Path) -> str:
    health = _read_json(report_path / "prop_collection_health_summary.json")
    quota = _read_json(report_path / "odds_api_quota_report.json")
    plan = _read_json(report_path / "nba_prop_closing_collection_plan.json")
    for source in [
        health.get("latest_warnings") or [],
        health.get("health_reasons") or [],
        (quota.get("risk_assessment") or {}).get("warnings") or [],
        plan.get("warnings") or [],
    ]:
        if source:
            return str(source[0])
    return "No current warnings."


def _dashboard_collection_summary(report_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots_path = report_path.parent / "processed" / "player_prop_snapshots_normalized.csv"
    snaps = _read_csv(snapshots_path)
    summary = _read_json(report_path / "player_prop_collection_run_summary.json")
    return snaps, summary


def _build_home_tab(report_path: Path, generated_at: str) -> str:
    gates = _read_json(report_path / "player_prop_data_quality_gates.json")
    health = _read_json(report_path / "prop_collection_health_summary.json")
    next_action = _read_json(report_path / "next_action_report.json")
    quota = _read_json(report_path / "odds_api_quota_report.json")
    refresh = _read_json(report_path / "player_prop_settlement_refresh_summary.json")
    pipeline = _read_json(report_path / "full_prop_pipeline_summary.json")
    gate_status = str(gates.get("status") or "unknown")
    modeling_blocked = gate_status != "modeling_experiment_ready"
    quota_risk = ((quota.get("risk_assessment") or {}).get("risk") or "unknown")
    next_item = next_action.get("next_action") or {}
    settlement = refresh.get("settlement") or {}
    blockers = gates.get("blockers") or []
    cards = _summary_grid(
        [
            ("Current project status", pipeline.get("status") or health.get("latest_run", {}).get("outcome") or "unknown"),
            ("Data gate status", gate_status),
            ("Modeling blocked", "YES" if modeling_blocked else "NO"),
            ("API quota risk", str(quota_risk).upper()),
            ("Last successful collection", health.get("last_successful_collection_utc") or "not available yet"),
            ("Last successful settlement", refresh.get("generated_at_utc") or "not available yet"),
            ("Latest warning", _dashboard_latest_warning(report_path)),
            ("Dashboard rebuild time", generated_at),
        ]
    )
    command_html = (
        f'<pre class="command">{html.escape(str(next_item.get("command")))}</pre>'
        if next_item.get("command")
        else ""
    )
    return (
        '<div class="section-lead">'
        "<h2>Home / Next Action</h2>"
        "<p>Research-only dashboard for data collection, market quality, CLV, model readiness, and paper tracking. Approved bets and approved parlays are disabled.</p>"
        "</div>"
        + cards
        + '<div class="status-line">'
        + _dashboard_badge("research-only", "blue")
        + _dashboard_badge(gate_status, _dashboard_tone(gate_status))
        + _dashboard_badge(f"quota {quota_risk}", _dashboard_tone("warning" if quota_risk == "high" else quota_risk))
        + (" " + _dashboard_badge("modeling blocked", "red") if modeling_blocked else " " + _dashboard_badge("modeling gate passed", "green"))
        + "</div>"
        + "<h3>Exact Next Action</h3>"
        + '<div class="callout">'
        + f'<strong>{html.escape(str(next_item.get("action") or "not available yet"))}</strong>'
        + f'<p>{html.escape(str(next_item.get("reason") or "Run scripts/build_next_action_report.py to refresh this section."))}</p>'
        + command_html
        + "</div>"
        + "<h3>Why Modeling Is Blocked</h3>"
        + (
            _dashboard_list(blockers, "No data-gate blockers reported.")
            if modeling_blocked
            else '<div class="empty">Modeling gates passed. Betting approval is still separate and remains disabled.</div>'
        )
        + "<h3>Settlement Snapshot</h3>"
        + _records_table(
            [
                {
                    "latest_refresh": refresh.get("generated_at_utc") or "not available yet",
                    "newly_settled": settlement.get("newly_settled", 0),
                    "pending_props": settlement.get("pending_after_refresh", 0),
                    "settled_total": settlement.get("settled_total", 0),
                }
            ]
        )
        + _download_links(
            report_path,
            [
                ("next_action_report.md", "Next action MD"),
                ("player_prop_data_quality_gates.md", "Data gates MD"),
                ("full_prop_pipeline_summary.json", "Pipeline summary JSON"),
            ],
        )
    )


def _build_upcoming_games_tab(report_path: Path) -> str:
    report = _read_json(report_path / "upcoming_games.json")
    games = report.get("games") or []
    if not games:
        fallback = _read_json(report_path / "nba_prop_closing_collection_plan.json")
        for game in fallback.get("games", []) if isinstance(fallback, dict) else []:
            key = str(game.get("canonical_game_key") or "")
            parts = key.split("|")
            games.append(
                {
                    "game_id": key,
                    "game_datetime_utc": game.get("game_start_time"),
                    "sport": parts[0] if len(parts) > 0 else "basketball",
                    "league": parts[1] if len(parts) > 1 else "NBA",
                    "away_team": parts[4] if len(parts) > 4 else "",
                    "home_team": parts[3] if len(parts) > 3 else "",
                    "source": "nba_prop_closing_collection_plan",
                    "props_expected": True,
                    "props_collected": bool(game.get("snapshots_total", 0)),
                    "latest_snapshot_time_utc": "",
                    "minutes_until_game": game.get("minutes_until_game"),
                    "closing_window_status": game.get("timing_classification"),
                    "recommended_collection_action": game.get("next_collection_reason"),
                }
            )
    columns = [
        "game_datetime_utc",
        "sport",
        "league",
        "away_team",
        "home_team",
        "source",
        "props_expected",
        "props_collected",
        "latest_snapshot_time_utc",
        "minutes_until_game",
        "closing_window_status",
        "recommended_collection_action",
    ]
    return (
        '<div class="section-lead"><h2>Upcoming Games</h2>'
        "<p>Collection timing by sport and league. Closing windows are collection actions, not betting signals.</p></div>"
        + _summary_grid(
            [
                ("Upcoming rows", len(games)),
                ("Need collection now", len(report.get("collection_needed_now") or [])),
                ("Generated", report.get("generated_at_utc") or "not available yet"),
                ("Warnings", len(report.get("warnings") or [])),
            ]
        )
        + (_records_table(games) if games else '<div class="empty">not available yet - run scripts/build_upcoming_games_report.py.</div>')
        + _download_links(
            report_path,
            [
                ("upcoming_games.json", "Upcoming games JSON"),
                ("upcoming_games.csv", "Upcoming games CSV"),
                ("upcoming_games.md", "Upcoming games MD"),
                ("nba_prop_closing_collection_plan.md", "NBA collection plan MD"),
            ],
        )
        + "<h3>Required Fields</h3>"
        + _records_table([{"field": column} for column in columns])
    )


# Canonical game key layout shared across the project:
#   sport|league|game_date|home_team|away_team
_CANONICAL_KEY_FIELDS = ("sport", "league", "game_date", "home_team", "away_team")


def _parse_canonical_game_key(key: str) -> dict[str, str]:
    """Split a canonical_game_key into its named parts (missing parts -> "")."""
    parts = str(key).split("|")
    parts += [""] * (len(_CANONICAL_KEY_FIELDS) - len(parts))
    return dict(zip(_CANONICAL_KEY_FIELDS, parts))


_RECORDED_GAMES_COLUMNS = [
    "league", "game_date", "away_team", "home_team", "canonical_game_key",
    "outcome_available", "game_odds_available", "odds_snapshot_count", "first_snapshot_time",
    "latest_snapshot_time", "has_closing_snapshot", "sportsbook_count",
    "prop_market_count", "settled_prop_count", "pending_prop_count", "status",
]
_RECORDED_GAMES_STATUS_ORDER = [
    "settled", "closing_recorded", "prop_odds_recorded", "game_odds_recorded",
    "outcome_only", "scheduled_only",
]


def _recorded_games_frame(report_path: Path) -> "pd.DataFrame":
    """Build the read-only recorded-games inventory (one row per canonical_game_key).

    Built only from existing local processed/report files (no network, no live
    collection). Research-only: enables no betting, predictions, or parlays. The key
    layout is ``sport|league|game_date|home_team|away_team``.
    """
    processed = report_path.parent / "processed"
    snaps = _read_csv(processed / "player_prop_snapshots_normalized.csv")
    enriched = _read_csv(processed / "player_prop_snapshots_enriched.csv")
    outcomes = _read_csv(report_path / "player_prop_settlement_outcomes.csv")
    games = _read_csv(processed / "nba_current_games_normalized.csv")
    upcoming = _read_csv(report_path / "upcoming_games.csv")
    inventory = _read_csv(processed / "historical_game_inventory.csv")

    # --- Odds-snapshot aggregates per game (all leagues). ---
    odds: dict[str, dict[str, Any]] = {}
    if not snaps.empty and "canonical_game_key" in snaps.columns:
        frame = snaps.copy()
        frame["_ts"] = pd.to_datetime(frame.get("snapshot_time"), errors="coerce", utc=True)
        closing = (
            _dashboard_bool_series(frame["is_closing_snapshot"])
            if "is_closing_snapshot" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        if "minutes_to_game_start" in frame.columns:
            mins = pd.to_numeric(frame["minutes_to_game_start"], errors="coerce")
            closing = closing | mins.le(60).fillna(False)
        frame["_closing"] = closing
        for key, grp in frame.groupby("canonical_game_key"):
            times = grp["_ts"].dropna()
            odds[str(key)] = {
                "odds_snapshot_count": int(len(grp)),
                "first_snapshot_time": times.min().isoformat() if not times.empty else "",
                "latest_snapshot_time": times.max().isoformat() if not times.empty else "",
                "has_closing_snapshot": bool(grp["_closing"].any()),
                "sportsbook_count": int(grp["bookmaker"].nunique()) if "bookmaker" in grp.columns else 0,
                "prop_market_count": int(grp["prop_type"].nunique()) if "prop_type" in grp.columns else 0,
            }

    # --- Settlement state per game from the enriched snapshots. ---
    settled_counts: dict[str, int] = {}
    pending_counts: dict[str, int] = {}
    if not enriched.empty and "canonical_game_key" in enriched.columns:
        keys = enriched["canonical_game_key"].astype(str)
        status = enriched.get("settlement_status", pd.Series("", index=enriched.index)).astype(str).str.lower()
        is_settled = status.eq("settled")
        supported = (
            _dashboard_bool_series(enriched["settlement_supported"])
            if "settlement_supported" in enriched.columns
            else is_settled
        )
        settled_counts = {str(k): int(v) for k, v in is_settled.groupby(keys).sum().items()}
        pending_counts = {str(k): int(v) for k, v in (supported & ~is_settled).groupby(keys).sum().items()}
    elif not outcomes.empty and "canonical_game_key" in outcomes.columns:
        # Fallback when no enriched file: count graded rows in the outcomes report.
        settled_counts = {
            str(k): int(v)
            for k, v in outcomes.groupby(outcomes["canonical_game_key"].astype(str)).size().items()
        }

    # --- Outcome availability: a final score exists, or the game was graded. ---
    outcome_keys: set[str] = set()
    if not games.empty and "canonical_game_key" in games.columns:
        if "home_score" in games.columns:
            played = games[pd.to_numeric(games["home_score"], errors="coerce").notna()]
            outcome_keys.update(played["canonical_game_key"].astype(str))
        else:
            outcome_keys.update(games["canonical_game_key"].astype(str))
    if not outcomes.empty and "canonical_game_key" in outcomes.columns:
        outcome_keys.update(outcomes["canonical_game_key"].astype(str))

    # --- Schedule universe: a game appears on a schedule/upcoming file. ---
    schedule_keys: set[str] = set()
    if not upcoming.empty and "game_id" in upcoming.columns:
        schedule_keys.update(upcoming["game_id"].astype(str))
    if not games.empty and "canonical_game_key" in games.columns:
        schedule_keys.update(games["canonical_game_key"].astype(str))

    # --- Free historical backfill inventory (outcomes, game odds, capped props). ---
    inv_game_odds: set[str] = set()
    inv_prop: set[str] = set()
    inv_closing: set[str] = set()
    if not inventory.empty and "canonical_game_key" in inventory.columns:
        inv_keys = inventory["canonical_game_key"].astype(str)

        def _inv_flag(column: str) -> pd.Series:
            if column in inventory.columns:
                return _dashboard_bool_series(inventory[column])
            return pd.Series(False, index=inventory.index)

        outcome_keys.update(inv_keys[_inv_flag("outcome_available").values])
        inv_game_odds = set(inv_keys[_inv_flag("game_odds_available").values])
        inv_prop = set(inv_keys[_inv_flag("prop_odds_available").values])
        inv_closing = set(inv_keys[_inv_flag("closing_available").values])

    all_keys = set(odds) | set(settled_counts) | set(pending_counts) | outcome_keys | schedule_keys
    all_keys |= inv_game_odds | inv_prop | inv_closing
    all_keys -= {"", "nan", "None"}

    rows: list[dict[str, Any]] = []
    for key in all_keys:
        parsed = _parse_canonical_game_key(key)
        info = odds.get(key, {})
        odds_count = int(info.get("odds_snapshot_count", 0))
        has_closing = bool(info.get("has_closing_snapshot", False)) or key in inv_closing
        settled_n = int(settled_counts.get(key, 0))
        pending_n = int(pending_counts.get(key, 0))
        outcome_avail = key in outcome_keys or settled_n > 0
        prop_recorded = odds_count > 0 or key in inv_prop
        game_odds_avail = key in inv_game_odds

        # Status priority (first match wins).
        if prop_recorded and outcome_avail and settled_n > 0:
            status = "settled"
        elif has_closing:
            status = "closing_recorded"
        elif prop_recorded:
            status = "prop_odds_recorded"
        elif game_odds_avail:
            status = "game_odds_recorded"
        elif outcome_avail:
            status = "outcome_only"
        else:
            status = "scheduled_only"

        rows.append({
            "league": parsed["league"],
            "game_date": parsed["game_date"],
            "away_team": parsed["away_team"],
            "home_team": parsed["home_team"],
            "canonical_game_key": key,
            "outcome_available": outcome_avail,
            "game_odds_available": game_odds_avail,
            "odds_snapshot_count": odds_count,
            "first_snapshot_time": info.get("first_snapshot_time", ""),
            "latest_snapshot_time": info.get("latest_snapshot_time", ""),
            "has_closing_snapshot": has_closing,
            "sportsbook_count": int(info.get("sportsbook_count", 0)),
            "prop_market_count": int(info.get("prop_market_count", 0)),
            "settled_prop_count": settled_n,
            "pending_prop_count": pending_n,
            "status": status,
        })

    frame = pd.DataFrame(rows, columns=_RECORDED_GAMES_COLUMNS)
    if not frame.empty:
        # Surface games with recorded odds/settlement first; outcome/scheduled tail by date.
        frame = frame.sort_values(
            by=["odds_snapshot_count", "settled_prop_count", "game_date"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return frame


def _render_recorded_games_section(frame: "pd.DataFrame", league: str | None = None) -> str:
    """Render the Recorded Games section, optionally filtered to one league.

    ``league`` matches the canonical-key league code (e.g. "NBA"); ``None`` shows
    every league. Read-only and research-only: no betting, predictions, or parlays.
    """
    if frame is None:
        frame = pd.DataFrame(columns=_RECORDED_GAMES_COLUMNS)
    if league is not None and not frame.empty:
        # Canonical keys store leagues normalized (WORLD_CUP -> WORLDCUP), so
        # compare on the normalized form to match tab labels with underscores.
        frame = frame[frame["league"] == normalize_league(league)].reset_index(drop=True)

    max_rows = 300
    explanation = (
        "Recorded does not always mean odds were recorded. A game can have outcomes/stats "
        "without historical odds. Odds_recorded means at least one market snapshot exists. "
        "Closing_recorded means a snapshot was captured near game start. Settled means odds "
        "were matched to final stats and graded."
    )
    counts = frame["status"].value_counts().to_dict() if not frame.empty else {}
    count_summary = " &bull; ".join(
        [f"Games tracked: {len(frame)}"]
        + [f"{status}: {int(counts.get(status, 0))}" for status in _RECORDED_GAMES_STATUS_ORDER]
    )
    truncated_note = (
        f'<p class="table-note">Showing the first {max_rows} of {len(frame)} games '
        "(games with recorded odds/settlement first).</p>"
        if len(frame) > max_rows
        else ""
    )
    empty_label = (
        f"No recorded games for {html.escape(league)} yet." if league else "No recorded games yet."
    )
    table_html = (
        _static_table(frame, _RECORDED_GAMES_COLUMNS, max_rows=max_rows)
        if not frame.empty
        else f'<div class="empty"><strong>{empty_label}</strong>'
             "Run the collection and settlement pipeline to populate snapshots and outcomes.</div>"
    )
    return (
        '<section class="recorded-games" aria-label="Recorded games inventory">'
        "<h3>Recorded Games</h3>"
        f'<p class="subtitle">{html.escape(explanation)}</p>'
        f'<p class="table-note">{count_summary}</p>'
        + truncated_note
        + table_html
        + "</section>"
    )


def _build_recorded_games_tab(report_path: Path) -> str:
    """Render the full (all-league) Recorded Games section from local files."""
    return _render_recorded_games_section(_recorded_games_frame(report_path), league=None)


def _recorded_games_by_league_table(frame: "pd.DataFrame") -> str:
    """Per-league x status breakdown of every recorded game (incl. backfill)."""

    if frame is None or frame.empty:
        return '<div class="empty">No recorded games yet.</div>'
    crosstab = pd.crosstab(frame["league"], frame["status"])
    for status in _RECORDED_GAMES_STATUS_ORDER:
        if status not in crosstab.columns:
            crosstab[status] = 0
    crosstab = crosstab[_RECORDED_GAMES_STATUS_ORDER]
    crosstab.insert(0, "games", crosstab.sum(axis=1))
    crosstab = crosstab.sort_values("games", ascending=False).reset_index()
    return _static_table(crosstab, ["league", "games", *_RECORDED_GAMES_STATUS_ORDER], max_rows=60)


def _historical_backfill_quality_banner(report_path: Path) -> str:
    """Warn when the inventory quality audit verdict is not 'clean'."""

    audit = _read_json(report_path / "historical_game_inventory_quality_audit.json")
    verdict = str(audit.get("verdict") or "").strip()
    if not verdict or verdict == "clean":
        return ""
    tone = "red" if verdict == "needs_fix" else "yellow"
    issues = list(audit.get("critical_issues") or []) + list(audit.get("warning_issues") or [])
    top = "; ".join(str(issue) for issue in issues[:3])
    return (
        '<div class="table-note">'
        f'<span class="status-badge {tone}">data quality: {html.escape(verdict)}</span> '
        f'{html.escape(top)}'
        ' &mdash; see <a href="historical_game_inventory_quality_audit.md">quality audit</a>'
        ' before running more large backfills.'
        "</div>"
    )


def _build_historical_backfill_panel(recorded_frame: "pd.DataFrame", report_path: Path) -> str:
    """A single panel surfacing the full inventory across ALL leagues (incl. ones
    without a dedicated tab, e.g. soccer/NFL). Research-only inventory."""

    total = 0 if recorded_frame is None else int(len(recorded_frame))
    return (
        '<section id="league-historical-backfill" class="league-panel">'
        '<div class="league-heading"><h2>Historical Backfill - All Recorded Games</h2>'
        f'{_status_badge("Research-only")}</div>'
        f'<p class="subtitle">Every game in the inventory across all leagues ({total:,} total), '
        "including free historical backfill (soccer, World Cup, MLB, NFL) that has no dedicated "
        "league tab yet. No betting, predictions, or parlays.</p>"
        + _historical_backfill_quality_banner(report_path)
        + "<h3>By League and Status</h3>"
        + _recorded_games_by_league_table(recorded_frame)
        + "<h3>All Games (sample)</h3>"
        + _render_recorded_games_section(recorded_frame, league=None)
        + "</section>"
    )


def _build_sports_markets_tab(report_path: Path) -> str:
    snaps, summary = _dashboard_collection_summary(report_path)
    coverage = _read_json(report_path / "cross_sport_collection_coverage_summary.json")
    readiness = _read_json(report_path / "all_sports_prop_readiness.json")
    quality = _read_json(report_path / "player_prop_market_quality_summary.json")
    outcomes = _read_json(report_path / "player_prop_settlement_outcomes_summary.json")
    clv_rows = _read_csv(report_path / "player_prop_clv.csv")
    sgo_hist = _read_json(report_path / "sportsgameodds_historical_prop_probe_summary.json")

    total_snapshots = int(len(snaps))
    closing_count = int(_dashboard_bool_series(snaps["is_closing_snapshot"]).sum()) if not snaps.empty and "is_closing_snapshot" in snaps else 0
    current_sgo_rows = int((snaps.get("source", pd.Series(dtype=str)).astype(str) == "sportsgameodds").sum()) if not snaps.empty and "source" in snaps else 0
    current_odds_api_rows = int((snaps.get("source", pd.Series(dtype=str)).astype(str) == "odds_api").sum()) if not snaps.empty and "source" in snaps else 0
    historical_sgo_rows = 0
    for window in sgo_hist.get("windows", []) if isinstance(sgo_hist, dict) else []:
        event = window.get("event") or {}
        if not window.get("is_current_week"):
            historical_sgo_rows += int(event.get("n_player_prop_odds") or 0)

    zero_rows = [
        {
            "sport": row.get("sport_group"),
            "league": row.get("league"),
            "reason": row.get("likely_reason") or row.get("status"),
        }
        for row in (coverage.get("leagues") or [])
        if int(row.get("snapshots_total") or 0) == 0
    ]
    expected_groups = ["basketball", "baseball", "hockey", "football", "soccer"]
    group_rows = []
    for group in expected_groups:
        found = [
            row for row in (coverage.get("leagues") or [])
            if row.get("sport_group") == group
        ]
        group_rows.append(
            {
                "sport": group,
                "leagues_configured": ", ".join(str(row.get("league")) for row in found) or "not available yet",
                "snapshots": sum(int(row.get("snapshots_total") or 0) for row in found),
                "collecting": ", ".join(str(row.get("league")) for row in found if row.get("status") == "collecting") or "(none)",
            }
        )

    closing_by_league = pd.DataFrame(columns=["league", "closing_like_snapshots"])
    settled_by_league = pd.DataFrame(columns=["league", "settled_rows"])
    if not snaps.empty and "league" in snaps.columns:
        if "is_closing_snapshot" in snaps.columns:
            tmp = snaps.copy()
            tmp["_closing"] = _dashboard_bool_series(tmp["is_closing_snapshot"])
            closing_by_league = tmp.groupby(tmp["league"].astype(str))["_closing"].sum().astype(int).rename_axis("league").reset_index(name="closing_like_snapshots")
        if "has_result" in snaps.columns:
            tmp = snaps.copy()
            tmp["_settled"] = _dashboard_bool_series(tmp["has_result"])
            settled_by_league = tmp.groupby(tmp["league"].astype(str))["_settled"].sum().astype(int).rename_axis("league").reset_index(name="settled_rows")
    clv_by_league = (
        clv_rows.groupby(clv_rows["league"].astype(str)).size().rename_axis("league").reset_index(name="clv_ready_rows")
        if not clv_rows.empty and "league" in clv_rows.columns
        else pd.DataFrame(columns=["league", "clv_ready_rows"])
    )

    return (
        '<div class="section-lead"><h2>Sports &amp; Markets Overview</h2>'
        "<p>What has actually been collected across basketball, baseball, hockey, football, and soccer.</p></div>"
        + _summary_grid(
            [
                ("Total snapshots", total_snapshots),
                ("Closing-like snapshots", closing_count),
                ("Sports covered", snaps["sport"].nunique() if not snaps.empty and "sport" in snaps else 0),
                ("Leagues with snapshots", snaps["league"].nunique() if not snaps.empty and "league" in snaps else 0),
                ("Prop types collected", snaps["prop_type"].nunique() if not snaps.empty and "prop_type" in snaps else 0),
                ("Bookmakers collected", snaps["bookmaker"].nunique() if not snaps.empty and "bookmaker" in snaps else 0),
                ("Settled props by league", outcomes.get("settled_props", 0)),
                ("Historical rows from SportsGameOdds", historical_sgo_rows),
                ("Current rows from SportsGameOdds", current_sgo_rows),
                ("Current rows from The Odds API", current_odds_api_rows),
            ]
        )
        + "<h3>Sports Covered</h3>" + _records_table(group_rows)
        + "<h3>Leagues Configured / Actively Collecting / Zero Snapshots</h3>"
        + (_records_table(coverage.get("leagues") or []) if coverage.get("leagues") else '<div class="empty">not available yet - run scripts/build_cross_sport_collection_coverage.py.</div>')
        + "<h3>Leagues With Zero Snapshots</h3>" + (_records_table(zero_rows) if zero_rows else '<div class="empty">No zero-snapshot leagues in the latest coverage report.</div>')
        + "<h3>Snapshots By Sport</h3>" + _static_table(_dashboard_counts(snaps, "sport", "sport"), ["sport", "rows"], max_rows=50)
        + "<h3>Snapshots By League</h3>" + _static_table(_dashboard_counts(snaps, "league", "league"), ["league", "rows"], max_rows=80)
        + "<h3>Snapshots By Source</h3>" + _static_table(_dashboard_counts(snaps, "source", "source"), ["source", "rows"], max_rows=30)
        + "<h3>Prop Types Collected</h3>" + _static_table(_dashboard_counts(snaps, "prop_type", "prop_type"), ["prop_type", "rows"], max_rows=80)
        + "<h3>Bookmakers Collected</h3>" + _static_table(_dashboard_counts(snaps, "bookmaker", "bookmaker"), ["bookmaker", "rows"], max_rows=80)
        + "<h3>Latest Snapshot / Closing / Settlement / CLV By League</h3>"
        + _table(closing_by_league.merge(settled_by_league, on="league", how="outer").merge(clv_by_league, on="league", how="outer"), max_rows=80)
        + _build_all_sports_readiness_section(report_path)
        + _download_links(
            report_path,
            [
                ("cross_sport_collection_coverage.csv", "Cross-sport coverage CSV"),
                ("all_sports_prop_readiness.md", "All-sports readiness MD"),
                ("player_prop_market_quality.md", "Market quality MD"),
            ],
        )
        + f'<p class="note">Latest run status: {html.escape(str(summary.get("status") or "not available yet"))}. '
        f'Market quality verdict: {html.escape(str(quality.get("nba_modeling_verdict") or "not available yet"))}</p>'
    )


def _build_odds_sources_tab(report_path: Path) -> str:
    usage = _read_json(report_path / "odds_source_usage_summary.json")
    sgo_hist = _read_json(report_path / "sportsgameodds_historical_prop_probe_summary.json")
    sgo_verdict = sgo_hist.get("verdict") or {}
    sources = usage.get("sources") or {}
    rows = []
    for name in ["odds_api", "sportsgameodds", "apisports", "kalshi"]:
        info = sources.get(name) or {}
        rows.append(
            {
                "source": name,
                "status": info.get("status") or "not available yet",
                "supports_player_props": "yes" if name in {"odds_api", "sportsgameodds"} else "probe-only" if name == "apisports" else "not wired for props",
                "supports_closing_prices": "yes" if name == "sportsgameodds" and sgo_verdict.get("closing_prices_available_for_props") else "snapshot-derived" if name == "odds_api" else "unknown",
                "supports_settled_results": "yes" if name == "sportsgameodds" and sgo_verdict.get("settlement_results_available") else "external settlement required",
                "quota_or_plan": "; ".join(f"{k}={v}" for k, v in (info.get("quota") or {}).items() if v is not None) or "n/a",
                "reliability": info.get("reliability") or info.get("note") or "n/a",
            }
        )
    return (
        '<div class="section-lead"><h2>Odds Sources</h2>'
        "<p>Source status, quota pressure, reliability, and whether SportsGameOdds can reduce The Odds API usage.</p></div>"
        + _records_table(rows)
        + _build_quota_section(report_path)
        + _build_odds_sources_section(report_path)
    )


def _build_market_quality_tab(report_path: Path) -> str:
    quality = _read_json(report_path / "player_prop_market_quality_summary.json")
    flags = quality.get("flag_counts") or {}
    closing = quality.get("closing_coverage") or {}
    problem_rows = [
        {"market_quality_by": "sport", "detail": "see snapshots by sport and quality flags below"},
        {"market_quality_by": "league", "detail": json.dumps(quality.get("markets_by_league") or {}, default=str)},
        {"market_quality_by": "prop_type", "detail": "see player_prop_line_quality.csv"},
    ]
    cleanest = quality.get("nba_best_bookmakers") or []
    return (
        '<div class="section-lead"><h2>Market Quality</h2>'
        "<p>Main lines, alternate lines, bad prices, bookmaker coverage, liquidity proxies, and problematic markets.</p></div>"
        + _summary_grid(
            [
                ("Main line count", quality.get("likely_main_lines", 0)),
                ("Alternate line count", quality.get("possible_alt_line_markets", 0)),
                ("Suspicious price count", flags.get("suspicious_price_values", 0)),
                ("Missing price count", quality.get("missing_price_markets", 0)),
                ("Low snapshot warnings", flags.get("low_snapshot_count", 0)),
                ("Line range warnings", quality.get("wide_line_range_markets", 0)),
                ("Markets without closing", closing.get("markets_without_closing", 0)),
            ]
        )
        + "<h3>Market Quality By Sport / League / Prop Type</h3>" + _records_table(problem_rows)
        + "<h3>Top Problematic Markets</h3>"
        + _table(_read_csv(report_path / "player_prop_possible_alt_lines.csv"), max_rows=30)
        + "<h3>Cleanest Markets / Bookmakers</h3>"
        + (_records_table(cleanest) if cleanest else '<div class="empty">not available yet</div>')
        + _build_market_quality_section(report_path)
    )


def _build_historical_backfill_tab(report_path: Path) -> str:
    hist = _read_json(report_path / "sportsgameodds_historical_prop_probe_summary.json")
    verdict = hist.get("verdict") or hist
    windows = hist.get("windows") or []
    rows_collected = sum(int((w.get("event") or {}).get("n_player_prop_odds") or 0) for w in windows)
    windows_completed = sum(1 for w in windows if w.get("ok"))
    backfill_status = _read_json(report_path / "sportsgameodds_historical_backfill_summary.json")
    ran_backfill = bool(backfill_status)
    safe_command = r".\.venv\Scripts\python.exe scripts\probe_sportsgameodds_historical_props.py --max-windows 1 --limit 1 --pause 6.5"
    return (
        '<div class="section-lead"><h2>Historical Backfill</h2>'
        "<p>SportsGameOdds historical availability and the cautious path toward NBA prop backfill.</p></div>"
        + _summary_grid(
            [
                ("Probe result", "available" if hist else "not available yet"),
                ("Historical games accessible", verdict.get("historical_events_accessible", "n/a")),
                ("Historical player props accessible", verdict.get("historical_player_props_accessible", "n/a")),
                ("Closing prices available", verdict.get("closing_prices_available_for_props") or verdict.get("open_close_prices_available", "n/a")),
                ("Settlement results available", verdict.get("settlement_results_available", "n/a")),
                ("Oldest useful game date", verdict.get("oldest_successful_game_date") or verdict.get("oldest_props_date") or "n/a"),
                ("Windows completed", windows_completed),
                ("Rows collected/probed", rows_collected),
                ("Quota used", hist.get("entity_cost") or verdict.get("entity_cost_estimate") or "n/a"),
                ("Quota remaining", hist.get("entities_after") or "n/a"),
                ("Close price coverage", verdict.get("oldest_close_field_date") or "n/a"),
                ("Settlement coverage", "available" if verdict.get("settlement_results_available") else "n/a"),
                ("Backtesting-ready rows", rows_collected if verdict.get("settlement_results_available") else 0),
                ("CLV-ready rows", rows_collected if verdict.get("closing_prices_available_for_props") else 0),
            ]
        )
        + "<h3>Backfill Plan Windows</h3>"
        + (_records_table(windows) if windows else '<div class="empty">No historical probe windows yet.</div>')
        + (
            '<div class="callout"><strong>Backfill has not run yet.</strong>'
            "<p>Start with one safe probe window before adding any importer/backfill loop.</p>"
            f'<pre class="command">{html.escape(safe_command)}</pre></div>'
            if not ran_backfill
            else _records_table([backfill_status])
        )
        + "<h3>Warnings</h3>"
        + _dashboard_list(hist.get("rejected_params") or [], "No rejected probe parameters.")
        + _download_links(
            report_path,
            [
                ("sportsgameodds_historical_prop_probe.md", "Historical probe MD"),
                ("sportsgameodds_historical_prop_probe_summary.json", "Historical probe JSON"),
            ],
        )
    )


def _build_model_readiness_tab(report_path: Path) -> str:
    gates = _read_json(report_path / "player_prop_data_quality_gates.json")
    metrics = gates.get("metrics") or {}
    thresholds = gates.get("thresholds") or {}
    threshold_rows = [
        {"threshold": key, "needed": value, "current": metrics.get(key.replace("min_", "").replace("max_", ""), "see checks")}
        for key, value in thresholds.items()
    ]
    return (
        '<div class="section-lead"><h2>Model Readiness</h2>'
        "<p>Calibration-first readiness gates. This tab does not start fake modeling and does not approve betting.</p></div>"
        + _summary_grid(
            [
                ("Current gate status", gates.get("status", "unknown")),
                ("Settled main-line rows", metrics.get("settled_main_line_rows", 0)),
                ("Closing-like main-line rows", metrics.get("closing_main_line_rows", 0)),
                ("CLV pairs", metrics.get("clv_markets", 0)),
                ("Player match rate", _fmt_pct(metrics.get("player_match_rate")) if metrics.get("player_match_rate") is not None else "n/a"),
                ("Missing core field rate", _fmt_pct(metrics.get("missing_core_field_rate")) if metrics.get("missing_core_field_rate") is not None else "n/a"),
                ("Main-line confidence", _fmt_pct(metrics.get("main_line_rate")) if metrics.get("main_line_rate") is not None else "n/a"),
                ("Bookmaker count", metrics.get("bookmakers", 0)),
                ("Prop type count", metrics.get("prop_types", 0)),
            ]
        )
        + "<h3>Gate Thresholds</h3>" + _records_table(threshold_rows)
        + _build_quality_gates_section(report_path)
        + _build_clv_readiness_section(report_path)
        + _build_settlement_outcomes_section(report_path)
        + _build_prop_clv_section(report_path)
        + _build_research_status_section(report_path)
    )


def _build_betting_tab(report_path: Path) -> str:
    report = _read_json(report_path / "paper_betting_report.json")
    bets = _read_csv(report_path / "paper_betting_report.csv")
    summary = report.get("summary") or {}
    parlays = report.get("parlays") or []
    if not report:
        return (
            '<div class="section-lead"><h2>Betting / Paper Tracking</h2>'
            "<p>Research-only tracking, never approved betting.</p></div>"
            '<div class="empty">No model bets yet. Betting page is waiting for data quality gates and model outputs.</div>'
            '<p class="note">Run scripts/build_paper_betting_report.py to create the blocked/empty report.</p>'
        )
    if report.get("status") == "blocked":
        blocked = (
            '<div class="empty">No model bets yet. Betting page is waiting for data quality gates and model outputs.</div>'
            f'<p class="note">{html.escape(str(report.get("blocked_reason") or ""))}</p>'
        )
    else:
        blocked = ""
    return (
        '<div class="section-lead"><h2>Betting / Paper Tracking</h2>'
        "<p>Only real saved research/paper outputs are shown. Approved bets and approved parlays remain disabled.</p></div>"
        + '<div class="status-line">'
        + _dashboard_badge("Research signal", "blue")
        + _dashboard_badge("Paper bet", "yellow")
        + _dashboard_badge("Backtest-only", "gray")
        + _dashboard_badge("Blocked", "red")
        + "</div>"
        + _summary_grid(
            [
                ("Total paper bets", summary.get("total_paper_bets", 0)),
                ("Wins", summary.get("wins", 0)),
                ("Losses", summary.get("losses", 0)),
                ("Pushes", summary.get("pushes", 0)),
                ("Win rate", _fmt_pct(summary.get("win_rate")) if summary.get("win_rate") is not None else "n/a"),
                ("Total profit/loss", _fmt_money(summary.get("total_profit_loss"))),
                ("ROI", _fmt_pct(summary.get("roi")) if summary.get("roi") is not None else "n/a"),
                ("Average CLV", summary.get("average_clv") if summary.get("average_clv") is not None else "n/a"),
            ]
        )
        + blocked
        + "<h3>Paper / Research Bet Rows</h3>"
        + (_table(bets, max_rows=200) if not bets.empty else '<div class="empty">No model bets yet - blocked by data quality gates.</div>')
        + "<h3>Profit By Sport</h3>" + _records_table(report.get("profit_by_sport") or [])
        + "<h3>Profit By League</h3>" + _records_table(report.get("profit_by_league") or [])
        + "<h3>Profit By Prop Type</h3>" + _records_table(report.get("profit_by_prop_type") or [])
        + "<h3>Profit By Source</h3>" + _records_table(report.get("profit_by_source") or [])
        + "<h3>Profit By Confidence Tier</h3>" + _records_table(report.get("profit_by_confidence_tier") or [])
        + "<h3>Research-Only / Paper Parlays</h3>"
        + (_records_table(parlays) if parlays else '<div class="empty">No research-only/paper parlays in the normalized paper report.</div>')
        + _download_links(
            report_path,
            [
                ("paper_betting_report.json", "Paper betting JSON"),
                ("paper_betting_report.csv", "Paper betting CSV"),
                ("paper_betting_report.md", "Paper betting MD"),
            ],
        )
    )


def _build_logs_health_tab(report_path: Path) -> str:
    health = _read_json(report_path / "prop_collection_health_summary.json")
    pipeline = _read_json(report_path / "full_prop_pipeline_summary.json")
    quota = _read_json(report_path / "odds_api_quota_report.json")
    runs = health.get("snapshots_by_run") or []
    steps = pipeline.get("steps") or []
    failed = pipeline.get("failed_steps") or []
    missing_keys = []
    if health and not health.get("api_key_detected"):
        missing_keys.append("ODDS_API_KEY")
    return (
        '<div class="section-lead"><h2>Logs / System Health</h2>'
        "<p>Collection runs, pipeline status, scheduler checks, API errors, quota warnings, backup status, and test status when available.</p></div>"
        + _summary_grid(
            [
                ("Health status", "HEALTHY" if health.get("healthy") else "UNHEALTHY" if health else "not available yet"),
                ("Latest collection run", (health.get("latest_run") or {}).get("run_id") or "not available yet"),
                ("Last full pipeline status", pipeline.get("status") or "not available yet"),
                ("Failed tasks", len(failed)),
                ("API errors", len(health.get("latest_errors") or [])),
                ("Quota warnings", len((quota.get("risk_assessment") or {}).get("warnings") or [])),
                ("Missing API keys", ", ".join(missing_keys) or "none detected"),
                ("Backup status", "not available yet"),
                ("Latest backup", "not available yet"),
                ("Tests status", "not available yet"),
            ]
        )
        + _build_collection_health_section(health)
        + "<h3>Latest Pipeline Runs / Steps</h3>" + (_records_table(steps) if steps else '<div class="empty">No pipeline summary yet.</div>')
        + "<h3>Latest Collection Runs</h3>" + (_records_table(runs[-25:]) if runs else '<div class="empty">No collection run history yet.</div>')
        + "<h3>Scheduled Task Health</h3>"
        + '<div class="empty">Use scripts\\verify_scheduled_tasks.ps1 for current Windows Task Scheduler status.</div>'
        + "<h3>Failed Tasks</h3>" + (_records_table(failed) if failed else '<div class="empty">No failed pipeline tasks in latest summary.</div>')
    )


def _build_settlement_refresh_tab_section(report_path: Path) -> str:
    refresh = _read_json(report_path / "player_prop_settlement_refresh_summary.json")
    newly_settled = _read_csv(report_path / "player_prop_newly_settled.csv")
    if not refresh:
        return (
            "<h2>NBA Settlement Refresh</h2>"
            '<div class="empty">No settlement refresh run yet. Run scripts/refresh_nba_results_and_settle_props.py.</div>'
        )
    settlement = refresh.get("settlement", {}) if isinstance(refresh, dict) else {}
    enrichment = refresh.get("enrichment", {}) if isinstance(refresh, dict) else {}
    settled_by_type = [
        {"prop_type": prop_type, "settled": count}
        for prop_type, count in sorted((settlement.get("settled_by_prop_type") or {}).items())
    ]
    return (
        "<h2>NBA Settlement Refresh</h2>"
        + _summary_grid(
            [
                ("Latest refresh", refresh.get("generated_at_utc", "n/a")),
                ("Refresh mode", refresh.get("mode", "n/a")),
                ("Newly settled", settlement.get("newly_settled", 0)),
                ("Pending props", settlement.get("pending_after_refresh", 0)),
                ("Still pending (carried over)", settlement.get("still_pending", 0)),
                ("Settled total", settlement.get("settled_total", 0)),
                ("Player match rate", _fmt_pct(enrichment.get("player_match_rate")) if enrichment.get("player_match_rate") is not None else "n/a"),
                ("Game match rate", _fmt_pct(enrichment.get("game_match_rate")) if enrichment.get("game_match_rate") is not None else "n/a"),
            ]
        )
        + "<h3>Newly Settled Props</h3>"
        + (
            _static_table(
                newly_settled,
                [
                    "player_name",
                    "team",
                    "prop_type",
                    "line",
                    "actual_stat_value",
                    "over_won",
                    "under_won",
                    "push",
                    "canonical_game_key",
                    "bookmaker",
                ],
                max_rows=100,
            )
            if not newly_settled.empty
            else '<div class="empty">No props settled in the latest refresh.</div>'
        )
        + "<h3>Settled Props By Prop Type</h3>"
        + (_records_table(settled_by_type) if settled_by_type else '<div class="empty">No settled props yet.</div>')
        + "<h3>Unsettled Games</h3>"
        + (_records_table(settlement.get("unsettled_games") or []) if settlement.get("unsettled_games") else '<div class="empty">No games are waiting on results.</div>')
    )


def _build_world_cup_tab(report_path: Path) -> str:
    """World Cup collection status (research-only). Renders 'not available yet'
    placeholders if a World Cup report is missing; never crashes."""
    status = _read_json(report_path / "world_cup_watcher_status.json")
    coll = _read_json(report_path / "world_cup_collection_summary.json")
    probe = _read_json(report_path / "world_cup_collection_probe.json")
    clv = _read_json(report_path / "world_cup_clv_summary.json")
    results = _read_json(report_path / "world_cup_results_summary.json")
    snaps = _read_csv(report_path.parent / "processed" / "world_cup_odds_snapshots_normalized.csv")

    # Prefer the live collection summary; fall back to the probe for quota/markets.
    coll_src = coll if coll else probe

    total_rows = int(len(snaps))
    closing_like = 0
    markets: list[str] = []
    bookmakers: list[str] = []
    matches_today = 0
    if not snaps.empty:
        if "is_closing_like" in snaps.columns:
            closing_like = int(_dashboard_bool_series(snaps["is_closing_like"]).sum())
        if "market_type" in snaps.columns:
            markets = sorted(snaps["market_type"].dropna().astype(str).unique().tolist())
        if "bookmaker" in snaps.columns:
            bookmakers = sorted(snaps["bookmaker"].dropna().astype(str).unique().tolist())
        if "event_start_time" in snaps.columns:
            starts = pd.to_datetime(snaps["event_start_time"], errors="coerce", utc=True).dropna()
            if not starts.empty:
                today = pd.Timestamp.now(tz="UTC").normalize()
                matches_today = int(starts.dt.normalize().eq(today).groupby(snaps.get("event_id")).any().sum())

    before = (coll_src or {}).get("credits_remaining_before")
    after = (coll_src or {}).get("credits_remaining_after")
    quota_used = "n/a"
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        quota_used = f"{before - after:g} credit(s) last pull"
    credits_now = status.get("credits_remaining") if status else None

    latest_watcher = "no watcher run yet"
    latest_errors = "none"
    if status:
        latest_watcher = status.get("generated_at_utc", "n/a")
        errs = []
        if isinstance(status.get("collection"), dict):
            errs += status["collection"].get("errors", []) or []
        errs += status.get("errors", []) or []
        latest_errors = "; ".join(str(e) for e in errs) if errs else "none"

    events_found = (status or {}).get("events_found", (coll_src or {}).get("events_found", 0))

    grid = _summary_grid([
        ("Events found (last list)", events_found),
        ("Matches today", matches_today),
        ("Snapshots collected", total_rows),
        ("Closing-like snapshots", closing_like),
        ("Markets collected", len(markets)),
        ("Bookmakers collected", len(bookmakers)),
        ("Quota used", quota_used),
        ("Credits remaining", credits_now if credits_now is not None else "n/a"),
        ("Latest watcher run", latest_watcher),
        ("CLV outcomes (research)", (clv or {}).get("markets_with_clv", 0)),
        ("Results settled", (results or {}).get("settled_rows", 0)),
        ("Latest errors", latest_errors),
    ])

    considered = (status or {}).get("considered", [])
    considered_rows = [
        {
            "match": f"{c.get('home_team')} v {c.get('away_team')}",
            "minutes_to_kickoff": c.get("minutes_until_event"),
            "status": c.get("event_status"),
            "planner_action": c.get("recommended_action"),
        }
        for c in considered[:40]
    ]

    market_rows = [{"market_type": m, "rows": int((snaps["market_type"] == m).sum())} for m in markets] if markets else []
    book_rows = [{"bookmaker": b, "rows": int((snaps["bookmaker"] == b).sum())} for b in bookmakers] if bookmakers else []

    lead = (
        '<div class="banner">RESEARCH ONLY. World Cup odds are collected for research. '
        'No World Cup bets, parlays, predictions, or recommendations are enabled, and World Cup '
        'data never feeds the NBA model gates.</div>'
        '<p class="section-lead">Separate, additive World Cup (FIFA) game-market collection '
        '(1X2 / totals) via The Odds API, driven by the World Cup Game Watcher. A strict quota '
        'floor (100 credits) protects NBA closing capture; the watcher lists events for free and '
        'only pulls odds near kickoff.</p>'
    )

    body = lead + grid
    body += '<div class="download-links">'
    body += '<a href="../processed/world_cup_odds_snapshots_normalized.csv" download>World Cup snapshots CSV</a>'
    body += '<a href="world_cup_collection_probe.md" download>Collection probe report</a>'
    body += '<a href="world_cup_clv_summary.md" download>CLV summary</a>'
    body += '<a href="world_cup_automation_audit.md" download>Automation audit</a>'
    body += "</div>"

    body += "<h3>Matches in the watcher's view</h3>"
    body += (_records_table(considered_rows) if considered_rows
             else '<div class="empty">No watcher run yet, or no World Cup events in window. '
                  'Run <code>world_cup_game_watcher.py --dry-run</code>.</div>')
    body += "<h3>Markets collected</h3>"
    body += (_records_table(market_rows) if market_rows
             else '<div class="empty">No World Cup market rows collected yet.</div>')
    body += "<h3>Bookmakers collected</h3>"
    body += (_records_table(book_rows) if book_rows
             else '<div class="empty">No World Cup bookmaker rows yet.</div>')
    body += "<h3>CLV (research-only, isolated from NBA gates)</h3>"
    body += f'<div class="callout">{html.escape(str((clv or {}).get("verdict", "World Cup CLV not built yet.")))}</div>'
    return body


def _cc_card(title: str, value: str, tone: str, badge: str, note: str, tip: str = "") -> str:
    """One big Command Center status card."""
    tip_attr = f' title="{html.escape(tip)}"' if tip else ""
    tip_mark = ' <span class="tip-mark">?</span>' if tip else ""
    return (
        f'<div class="cc-card"{tip_attr}>'
        f'<div class="cc-title">{html.escape(title)}{tip_mark}</div>'
        f'<div class="cc-value">{html.escape(value)}</div>'
        f'<div><span class="status-badge {tone}">{html.escape(badge)}</span></div>'
        f'<div class="cc-note">{html.escape(note)}</div>'
        "</div>"
    )


def _friendly_gate(status: str) -> str:
    return {
        "not_ready": "Just starting to collect",
        "collection_ready": "Collecting data",
        "settlement_ready": "Collecting & grading results",
        "clv_ready": "Closing-line value ready",
        "modeling_experiment_ready": "Ready for research modeling",
    }.get(status, status or "unknown")


def _build_command_center_tab(report_path: Path, generated_at: str) -> str:
    health = _read_json(report_path / "prop_collection_health_summary.json")
    gates = _read_json(report_path / "player_prop_data_quality_gates.json")
    sh = _read_json(report_path / "source_health_summary.json")
    wc = _read_json(report_path / "world_cup_source_status.json")
    state = _read_json(report_path / "source_state.json")
    snaps, summary = _dashboard_collection_summary(report_path)

    # NBA automation
    if health.get("healthy"):
        nba_tone, nba_badge, nba_note = "green", "Working", "Collecting & settling on schedule."
    elif health.get("likely_quota_issue"):
        nba_tone, nba_badge, nba_note = "yellow", "Blocked by quota", "Odds API credits low; collection thinned."
    else:
        nba_tone, nba_badge, nba_note = "yellow", "Needs attention", "Recent run was not fully healthy."

    # World Cup automation
    if not wc:
        wc_tone, wc_badge, wc_note = "gray", "Not run yet", "Run the World Cup watcher to populate."
    elif wc.get("odds_api_below_floor"):
        wc_tone, wc_badge, wc_note = "yellow", "Blocked by quota", \
            f"{wc.get('events_found', 0)} matches found; free schedule works, paid odds paused."
    elif wc.get("events_found"):
        wc_tone, wc_badge, wc_note = "green", "Working", f"{wc.get('events_found')} matches tracked."
    else:
        wc_tone, wc_badge, wc_note = "blue", "Waiting for matches", "No World Cup matches in window."

    # Data health (gate)
    gate_status = gates.get("status", "unknown")
    data_tone = "green" if gate_status in ("clv_ready", "modeling_experiment_ready") else "blue"
    data_note = "Data passes collection & settlement checks." if gate_status == "settlement_ready" else \
        "See Model Readiness for details."

    # API quota
    odds_credits = (state.get("sources", {}).get("odds_api", {}) or {}).get("quota_remaining")
    sgo_ent = (state.get("sources", {}).get("sportsgameodds", {}) or {}).get("quota_remaining")
    if odds_credits is None:
        q_tone, q_badge, q_note = "gray", "Unknown", "Run the source probe to read quotas."
    elif odds_credits < 100:
        q_tone, q_badge, q_note = "red", "Blocked by quota", f"Odds API ~{odds_credits:g} credits (low). SGO ~{sgo_ent:g} entities."
    else:
        q_tone, q_badge, q_note = "green", "Healthy", f"Odds API ~{odds_credits:g} credits. SGO ~{sgo_ent:g} entities."

    # Latest fetch / error
    latest = "n/a"
    if not snaps.empty and "snapshot_time" in snaps.columns:
        t = pd.to_datetime(snaps["snapshot_time"], errors="coerce", utc=True).dropna()
        if not t.empty:
            latest = t.max().strftime("%Y-%m-%d %H:%M UTC")
    warns = health.get("latest_warnings") or []
    blocked_src = sh.get("sources_blocked") or []
    err_text = (warns[0] if warns else (f"{', '.join(blocked_src)} blocked" if blocked_src else "none"))
    err_tone = "yellow" if (warns or blocked_src) else "green"

    cards = "".join([
        _cc_card("NBA automation", "Player props", nba_tone, nba_badge, nba_note,
                 "Collects NBA player-prop odds near tip-off and grades results."),
        _cc_card("World Cup automation", "Game markets", wc_tone, wc_badge, wc_note,
                 "Collects World Cup match odds (1X2/totals) when quota allows; schedule is free."),
        _cc_card("Data health", _friendly_gate(gate_status), data_tone, "Research-only", data_note,
                 "How far the data has progressed toward research modeling readiness."),
        _cc_card("API quota", "Odds budget", q_tone, q_badge, q_note,
                 "Remaining monthly budget across paid data sources."),
        _cc_card("Latest fetch", latest, "blue", "Most recent", "When the newest odds snapshot was saved.",
                 "The newest snapshot stored across all sources."),
        _cc_card("Latest issue", "", err_tone, "Heads up" if err_tone == "yellow" else "All clear", err_text,
                 "The most recent warning or blocked source, if any."),
        _cc_card("Model gate", _friendly_gate(gate_status), "blue", "Research-only",
                 "No bets, parlays, or predictions are enabled.",
                 "Modeling stays locked until closing-line value data exists."),
    ])
    return (
        '<div class="banner">RESEARCH ONLY — this project collects and grades sports data for research. '
        'It does <strong>not</strong> place bets, build parlays, make predictions, or give picks.</div>'
        '<p class="section-lead">A quick health check of everything. Green = working, yellow = needs '
        'attention, red = blocked by quota. Tap a tab above for detail.</p>'
        f'<div class="cc-grid">{cards}</div>'
        f'<p class="subtle">Snapshot generated {html.escape(generated_at)}.</p>'
    )


def _build_today_games_tab(report_path: Path) -> str:
    plan = _read_json(report_path / "nba_prop_closing_collection_plan.json")
    wc = _read_json(report_path / "world_cup_watcher_status.json")
    nba_rows = []
    for g in (plan.get("games") or [])[:30]:
        nba_rows.append({
            "league": "NBA", "match": g.get("game"),
            "starts_in_min": round(g.get("minutes_until_game", 0) or 0),
            "planned_action": "collect closing snapshot" if g.get("collection_needed_now") else g.get("timing_classification"),
            "source": "odds_api / sportsgameodds",
            "snapshots": g.get("snapshots_total"),
            "closing_like": g.get("closing_like_snapshots"),
        })
    wc_rows = []
    for c in (wc.get("considered") or [])[:30]:
        wc_rows.append({
            "league": "WORLD_CUP", "match": f"{c.get('home_team')} v {c.get('away_team')}",
            "starts_in_min": round(c.get("minutes_until_event", 0) or 0),
            "planned_action": c.get("recommended_action"),
            "source": (wc.get("routing", {}).get("game_odds", {}) or {}).get("selected") or "free schedule only",
            "snapshots": "", "closing_like": "",
        })
    body = (
        '<p class="section-lead">NBA and World Cup matches the watchers can see right now, with what each '
        'watcher plans to do. Research-only — no bets or picks.</p>'
    )
    body += "<h3>🏀 NBA</h3>"
    body += (_records_table(nba_rows) if nba_rows
             else '<div class="empty">No NBA games in the current window.</div>')
    body += "<h3>🌍 World Cup</h3>"
    body += (_records_table(wc_rows) if wc_rows
             else '<div class="empty">No World Cup matches in the current window (run the World Cup watcher).</div>')
    return body


def _build_source_health_cards(report_path: Path) -> str:
    sh = _read_json(report_path / "source_health_summary.json")
    sources = sh.get("sources", {})
    labels = {
        "odds_api": "The Odds API", "sportsgameodds": "SportsGameOdds",
        "apisports": "API-Sports / API-Football", "kalshi": "Kalshi",
    }
    if not sources:
        return '<div class="empty">No source health yet. Run <code>scripts/probe_all_sources.py --real</code> then <code>scripts/build_source_health_report.py</code>.</div>'
    cards = []
    for key, label in labels.items():
        v = sources.get(key, {})
        status = v.get("status", "unknown")
        tone = {"available": "green", "blocked": "red", "no_key": "gray"}.get(status, "gray")
        if v.get("below_floor"):
            tone, status = "yellow", "low quota"
        quota = v.get("quota_remaining")
        quota_str = f"{quota:g}" if isinstance(quota, (int, float)) else "n/a"
        cards.append(
            '<div class="src-card">'
            f'<div class="src-head"><span class="src-name">{html.escape(label)}</span>'
            f'<span class="status-badge {tone}">{html.escape(status)}</span></div>'
            f'<div class="src-row"><span>Key detected</span><span>{"yes" if v.get("key_detected") else "no"}</span></div>'
            f'<div class="src-row"><span>Quota remaining</span><span>{quota_str}</span></div>'
            f'<div class="src-row"><span>Last success</span><span>{html.escape(str(v.get("last_success_utc") or "—"))}</span></div>'
            f'<div class="src-row"><span>Last error</span><span>{html.escape(str(v.get("blocked_reason") or v.get("last_failure_utc") or "—"))}</span></div>'
            f'<div class="src-use">{html.escape(str(v.get("used_for") or ""))}</div>'
            "</div>"
        )
    return f'<div class="src-grid">{"".join(cards)}</div>'


def _build_source_health_tab(report_path: Path) -> str:
    sh = _read_json(report_path / "source_health_summary.json")
    best = sh.get("best_source_by_data_type", {})
    body = (
        '<p class="section-lead">Each data provider and whether it is available, blocked, or low on quota. '
        'Keys are detected only — never shown. Research-only.</p>'
    )
    body += _build_source_health_cards(report_path)
    if best:
        body += "<h3>Who does each job right now</h3>"
        for scope, table in best.items():
            rows = [{"data_type": k, "selected_source": v} for k, v in table.items()]
            body += f"<h4>{html.escape(scope)}</h4>" + _records_table(rows)
    body += ('<div class="download-links">'
             '<a href="source_health_summary.md" download>Source health (markdown)</a>'
             '<a href="source_health_summary.json" download>Source health (json)</a></div>')
    return body


def _build_nba_props_tab(report_path: Path) -> str:
    gates = _read_json(report_path / "player_prop_data_quality_gates.json")
    clv = _read_json(report_path / "player_prop_clv_summary.json")
    settle = _read_json(report_path / "player_prop_settlement_outcomes_summary.json")
    m = gates.get("metrics", {})
    grid = _summary_grid([
        ("NBA snapshots", m.get("nba_snapshots", 0)),
        ("Closing-like snapshots", m.get("closing_like_snapshots", 0)),
        ("Settled props", settle.get("settled_props", 0)),
        ("Pending props", settle.get("pending_props", 0)),
        ("CLV pairs (NBA)", m.get("clv_markets", 0)),
        ("Bookmakers", m.get("bookmakers", 0)),
        ("Prop types", m.get("prop_types", 0)),
        ("Data stage", _friendly_gate(gates.get("status", "unknown"))),
    ])
    blockers = gates.get("blockers", [])
    why = "".join(f"<li>{html.escape(str(b))}</li>" for b in blockers) or "<li>No blockers recorded.</li>"
    return (
        '<p class="section-lead">NBA player-prop collection and grading. Research-only — no prop bets, '
        'parlays, or predictions exist or are enabled.</p>'
        + grid
        + '<div class="callout"><strong>Why modeling is still locked:</strong><ul>' + why + "</ul>"
        + '<p class="subtle">Modeling stays locked until NBA closing-line value (CLV) data exists. '
        'Capturing snapshots within ~60 minutes of tip-off is the missing piece.</p></div>'
        + '<div class="download-links">'
        '<a href="../processed/player_prop_snapshots_normalized.csv" download>NBA snapshots CSV</a>'
        '<a href="nba_source_routing_audit.md" download>Source routing audit</a></div>'
    )


def _build_advanced_reports_tab(report_path: Path, generated_at: str) -> str:
    """Keep every existing audit detail here, tucked into collapsible sections."""
    parts = [
        ("Home / Next Action", _build_home_tab(report_path, generated_at)),
        ("Upcoming Games", _build_upcoming_games_tab(report_path)),
        ("Sports & Markets Overview", _build_sports_markets_tab(report_path)),
        ("Odds Sources", _build_odds_sources_tab(report_path)),
        ("Market Quality", _build_market_quality_tab(report_path)),
        ("Historical Backfill", _build_historical_backfill_tab(report_path)),
        ("Model Readiness", _build_model_readiness_tab(report_path)),
        ("Betting / Paper Tracking", _build_betting_tab(report_path)),
        ("Logs / System Health", _build_logs_health_tab(report_path)),
    ]
    intro = ('<p class="section-lead">The full technical audit, for power users. Average users can stay on '
             'the Command Center. Research-only throughout.</p>')
    blocks = "".join(
        f'<details class="adv"><summary>{html.escape(title)}</summary><div class="adv-body">{body}</div></details>'
        for title, body in parts
    )
    return intro + blocks


def _build_research_prop_dashboard_html(report_path: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    snaps, summary = _dashboard_collection_summary(report_path)
    latest_snapshot = "n/a"
    if not snaps.empty and "snapshot_time" in snaps.columns:
        times = pd.to_datetime(snaps["snapshot_time"], errors="coerce", utc=True).dropna()
        if not times.empty:
            latest_snapshot = times.max().isoformat()
    closing_count = int(_dashboard_bool_series(snaps["is_closing_snapshot"]).sum()) if not snaps.empty and "is_closing_snapshot" in snaps else 0
    totals = summary.get("totals", {}) if isinstance(summary, dict) else {}
    last_run_status = summary.get("status", "no runs yet") if summary else "no runs yet"
    last_run_id = summary.get("run_id", "n/a") if summary else "n/a"

    tabs = [
        ("command-center", "Command Center", _build_command_center_tab(report_path, generated_at)),
        ("today-games", "Today's Games", _build_today_games_tab(report_path)),
        ("source-health", "Source Health", _build_source_health_tab(report_path)),
        ("world-cup", "World Cup", _build_world_cup_tab(report_path)),
        ("nba-props", "NBA Props", _build_nba_props_tab(report_path)),
        ("advanced", "Advanced Reports", _build_advanced_reports_tab(report_path, generated_at)),
    ]
    nav = "".join(
        f'<button class="tab{" active" if index == 0 else ""}" data-tab="{tab_id}">{html.escape(label)}</button>'
        for index, (tab_id, label, _) in enumerate(tabs)
    )
    sections = "".join(
        f'<section id="{tab_id}" class="tab-section{" active" if index == 0 else ""}">{content}</section>'
        for index, (tab_id, _, content) in enumerate(tabs)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Player Props Research Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #637083;
      --line: #d8dee8;
      --green: #087443;
      --yellow: #b54708;
      --red: #b42318;
      --blue: #175cd3;
      --gray: #667085;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.42; }}
    header {{ background: #fff; border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 10; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 16px 20px; }}
    h1 {{ margin: 0; font-size: 27px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 24px 0 10px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    .subtle, .note {{ color: var(--muted); font-size: 13px; }}
    .banner {{ background: #eef4ff; border: 1px solid #b2ccff; color: #102a56; border-radius: 8px; padding: 10px 12px; margin-top: 12px; font-weight: 700; }}
    nav {{ display: flex; gap: 8px; overflow-x: auto; padding: 12px 0 2px; }}
    .tab {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 8px; padding: 9px 12px; font: inherit; font-weight: 750; cursor: pointer; white-space: nowrap; }}
    .tab.active {{ background: var(--ink); border-color: var(--ink); color: #fff; }}
    main.wrap {{ padding-top: 22px; padding-bottom: 44px; }}
    .tab-section {{ display: none; }}
    .tab-section.active {{ display: block; }}
    .section-lead {{ margin-bottom: 16px; max-width: 980px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin: 14px 0 18px; }}
    .metric-box, .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 11px; }}
    .metric-label {{ color: var(--muted); font-size: 12px; font-weight: 800; margin-bottom: 4px; }}
    .big-number, .metric-value {{ font-size: 20px; font-weight: 900; overflow-wrap: anywhere; }}
    .metric-note {{ color: var(--muted); font-size: 12px; min-height: 14px; }}
    .status-line {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 16px; }}
    .status-badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 850; border: 1px solid transparent; }}
    .status-badge.green {{ background: #ecfdf3; color: var(--green); border-color: #abefc6; }}
    .status-badge.yellow {{ background: #fffaeb; color: var(--yellow); border-color: #fedf89; }}
    .status-badge.red {{ background: #fef3f2; color: var(--red); border-color: #fecdca; }}
    .status-badge.gray {{ background: #f2f4f7; color: var(--gray); border-color: #d0d5dd; }}
    .status-badge.blue {{ background: #eef4ff; color: var(--blue); border-color: #b2ccff; }}
    .callout, .status {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin: 12px 0; }}
    .command, pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #111827; color: #f9fafb; border-radius: 8px; padding: 10px; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 12px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 820px; }}
    th, td {{ text-align: left; padding: 10px 11px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }}
    th {{ background: #f8fafc; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0; cursor: pointer; }}
    .empty {{ padding: 22px; border: 1px dashed var(--line); border-radius: 8px; background: #fff; color: var(--muted); text-align: center; }}
    .download-links {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px; }}
    .download-links a {{ border: 1px solid #b2ccff; background: #eef4ff; color: var(--blue); border-radius: 8px; padding: 8px 10px; font-weight: 800; text-decoration: none; }}
    ul {{ margin-top: 8px; }}
    @media (max-width: 760px) {{
      .wrap {{ padding-left: 12px; padding-right: 12px; }}
      h1 {{ font-size: 22px; }}
      h2 {{ font-size: 19px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
      table {{ min-width: 720px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Player Props Research Dashboard</h1>
      <div class="subtle">Generated: {html.escape(generated_at)} | Main entry: player_props.html</div>
      <div class="banner">RESEARCH ONLY. No approved bets. No approved parlays. No fake model picks. Most games should remain no-bet until data quality, CLV, and model proof gates pass.</div>
      <nav aria-label="Dashboard tabs">{nav}</nav>
    </div>
  </header>
  <main class="wrap">
    <div class="summary-grid" aria-label="Collection summary">
      {_summary_grid([
        ("Total snapshots", int(len(snaps))),
        ("Closing-like snapshots", closing_count),
        ("Latest snapshot", latest_snapshot),
        ("Latest run", last_run_id),
        ("Latest run status", last_run_status),
        ("Added last run", totals.get("snapshots_added", 0)),
        ("Duplicates removed last run", totals.get("duplicates_removed", 0)),
        ("Raw files saved last run", totals.get("raw_files_saved", 0)),
      ])}
    </div>
    {sections}
    <h2>Legacy Player Prop Detail Sections</h2>
    {_build_settlement_refresh_tab_section(report_path)}
    <h3>Snapshots By Sport</h3>
    {_static_table(_dashboard_counts(snaps, "sport", "sport"), ["sport", "rows"], max_rows=20)}
    <h3>Snapshots By League</h3>
    {_static_table(_dashboard_counts(snaps, "league", "league"), ["league", "rows"], max_rows=40)}
    <h3>Snapshots By Source</h3>
    {_static_table(_dashboard_counts(snaps, "source", "source"), ["source", "rows"], max_rows=20)}
    <h3>Snapshots By Prop Type</h3>
    {_static_table(_dashboard_counts(snaps, "prop_type", "prop_type"), ["prop_type", "rows"], max_rows=60)}
    <h3>Missing player_id By Sport</h3>
    {_static_table(_missing_by_sport_for_dashboard(snaps, "player_id"), ["sport", "missing_player_id"], max_rows=20)}
    <h3>Missing game_key By Sport</h3>
    {_static_table(_missing_by_sport_for_dashboard(snaps, "canonical_game_key"), ["sport", "missing_canonical_game_key"], max_rows=20)}
    <h3>Latest Run By League</h3>
    {_records_table(summary.get("leagues", []) if isinstance(summary, dict) else [])}
    <div class="download-links"><a href="../processed/player_prop_snapshots_normalized.csv" download>Player prop snapshots CSV</a></div>
  </main>
  <script>
    document.querySelectorAll(".tab").forEach((button) => {{
      button.addEventListener("click", () => {{
        const tabId = button.dataset.tab;
        document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab === button));
        document.querySelectorAll(".tab-section").forEach((section) => section.classList.toggle("active", section.id === tabId));
      }});
    }});
    document.querySelectorAll("th").forEach((header) => {{
      header.addEventListener("click", () => {{
        const table = header.closest("table");
        const tbody = table && table.querySelector("tbody");
        if (!tbody) return;
        const index = Array.from(header.parentElement.children).indexOf(header);
        const asc = header.dataset.sortDir !== "asc";
        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort((a, b) => {{
          const av = a.children[index] ? a.children[index].textContent.trim() : "";
          const bv = b.children[index] ? b.children[index].textContent.trim() : "";
          const an = Number(av.replace(/[$,%]/g, ""));
          const bn = Number(bv.replace(/[$,%]/g, ""));
          const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
          return asc ? cmp : -cmp;
        }});
        header.dataset.sortDir = asc ? "asc" : "desc";
        rows.forEach((row) => tbody.appendChild(row));
      }});
    }});
  </script>
</body>
</html>"""


def _missing_by_sport_for_dashboard(snaps: pd.DataFrame, column: str) -> pd.DataFrame:
    if snaps.empty or column not in snaps.columns or "sport" not in snaps.columns:
        return pd.DataFrame(columns=["sport", f"missing_{column}"])
    missing = snaps[column].isna() | snaps[column].astype(str).str.strip().eq("")
    grouped = missing.groupby(snaps["sport"].fillna("(missing)").astype(str)).sum().astype(int)
    return grouped.rename_axis("sport").reset_index(name=f"missing_{column}")


def _build_player_props_page(report_path: Path) -> str:
    return _build_sports_market_research_dashboard_html(report_path)


def _build_project_cleanup_page(report_path: Path) -> str:
    summary = _read_json(report_path / "project_cleanup_audit_summary.json")
    candidates = _read_csv(report_path / "project_cleanup_candidates.csv")
    if not summary:
        body = (
            '<div class="empty">No cleanup audit yet. Run scripts/audit_project_cleanup.py '
            "(audit-only by default; --apply moves safe items to data/quarantine/).</div>"
        )
        return _static_page("Project Cleanup Audit", report_path, "project_cleanup.html", body)

    counts = summary.get("counts", {}) if isinstance(summary.get("counts"), dict) else {}
    status_counts = summary.get("status_counts", {}) if isinstance(summary.get("status_counts"), dict) else {}
    apply_info = summary.get("apply") or {}
    mode = "AUDIT ONLY (nothing touched)" if summary.get("audit_only", True) else "APPLY (moved to quarantine)"
    largest = pd.DataFrame(summary.get("top_largest_files") or [])
    duplicate_rows = [
        {"group": group.get("group"), "size_bytes": group.get("size_bytes"), "files": " | ".join(group.get("files") or [])}
        for group in (summary.get("duplicate_groups") or [])
    ]
    links = "".join(
        f'<a href="{html.escape(filename)}" download>{html.escape(label)}</a>'
        for filename, label in [
            ("project_cleanup_audit.md", "Cleanup audit report (MD)"),
            ("project_cleanup_candidates.csv", "Cleanup candidates CSV"),
        ]
        if (report_path / filename).exists()
    )
    body = (
        _summary_grid(
            [
                ("Mode", mode),
                ("Files scanned", summary.get("total_files_scanned", 0)),
                ("Folders scanned", summary.get("total_folders_scanned", 0)),
                ("Empty files", counts.get("empty_files", 0)),
                ("Empty folders", counts.get("empty_folders", 0)),
                ("Duplicate groups", counts.get("duplicate_groups", 0)),
                ("Large files (>50 MB)", counts.get("large_files", 0)),
                ("safe_to_delete", status_counts.get("safe_to_delete", 0)),
                ("needs_review", status_counts.get("needs_review", 0)),
                ("should_keep", status_counts.get("should_keep", 0)),
                (
                    "Estimated cleanup size",
                    f"{float(summary.get('estimated_cleanup_bytes', 0)) / (1024 * 1024):.1f} MB",
                ),
                ("Moved to quarantine", apply_info.get("moved_count", 0)),
            ]
        )
        + '<p class="note">Audit-only by default: nothing is deleted automatically. With --apply, '
        "safe_to_delete items are moved to data/quarantine/project_cleanup/ instead of being deleted. "
        "Protected paths (src/, scripts/, tests/, config/, data/raw/, data/processed/, data/reports/, "
        "README.md, TODO.md, .venv/, .git/) are never touched.</p>"
        + f'<div class="download-links">{links}</div>'
        + "<h2>Recommended Next Action</h2>"
        + f'<p>{html.escape(str(summary.get("recommended_next_action") or ""))}</p>'
        + "<h2>Top 20 Largest Files</h2>"
        + (_static_table(largest, ["path", "size_bytes", "modified_utc"], max_rows=20) if not largest.empty else '<div class="empty">No files found.</div>')
        + "<h2>Duplicate File Groups</h2>"
        + (_records_table(duplicate_rows) if duplicate_rows else '<div class="empty">No duplicate files found.</div>')
        + "<h2>Cleanup Candidates</h2>"
        + (
            _static_table(
                candidates,
                ["path", "kind", "category", "status", "size_bytes", "modified_utc", "reason"],
                max_rows=300,
            )
            if not candidates.empty
            else '<div class="empty">No cleanup candidates found.</div>'
        )
    )
    return _static_page("Project Cleanup Audit", report_path, "project_cleanup.html", body)


# --- Team Availability page ------------------------------------------------

def _availability_report(report_path: Path) -> dict[str, Any]:
    return _read_json(report_path / "team_availability_validation.json")


def _build_availability_coverage_summary(report_path: Path) -> str:
    report = _availability_report(report_path)
    if not report:
        return (
            "<div class='status'><strong>Availability coverage:</strong> not validated yet. "
            "Run the team availability validation workflow to populate this summary.</div>"
        )
    coverage = report.get("coverage", {})
    injury = report.get("injury_data", {})
    total = int(coverage.get("total_fixture_teams", 0) or 0)
    covered = int(coverage.get("fixture_teams_with_availability", 0) or 0)
    missing = coverage.get("missing_teams", []) or []
    stale = int(injury.get("stale_rows_older_than_48h", 0) or 0)
    warning = (report.get("warnings") or ["None"])[0]
    missing_text = "none" if not missing else (
        ", ".join(str(team) for team in missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
    )
    return (
        "<div class='status'>"
        "<strong>Availability coverage:</strong> "
        f"{covered} of {total} fixture teams. "
        f"<strong>Missing teams:</strong> {html.escape(missing_text)}. "
        f"<strong>Stale rows:</strong> {stale}. "
        f"<strong>Main warning:</strong> {html.escape(str(warning))}"
        "</div>"
    )


def _build_team_availability_page(report_path: Path) -> str:
    report = _availability_report(report_path)
    intro = (
        "<p class='note'><strong>Team availability is a data-quality input, not betting information.</strong> "
        "Missing, stale, or manual rows should be treated as uncertainty rather than confirmed lineup news.</p>"
    )
    if not report:
        body = (
            intro
            + "<div class='empty'>No team availability validation report found. "
            + "<code>python scripts/validate_team_availability.py</code> will populate this page.</div>"
        )
        return _static_page("Team Availability", report_path, "team_availability.html", body)

    coverage = report.get("coverage", {})
    injury = report.get("injury_data", {})
    warnings = report.get("warnings", []) or []
    issues = report.get("issues", []) or []
    team_rows = pd.DataFrame(report.get("team_rows", []) or [])

    summary = _summary_grid(
        [
            ("Status", report.get("overall_status", "n/a")),
            ("Fixture Teams", coverage.get("total_fixture_teams", 0)),
            ("Covered", coverage.get("fixture_teams_with_availability", 0)),
            ("Coverage", f"{coverage.get('coverage_percentage', 0)}%"),
            ("Missing Teams", coverage.get("fixture_teams_missing_availability", 0)),
            ("Stale Rows", injury.get("stale_rows_older_than_48h", 0)),
        ]
    )

    status_counts = injury.get("status_counts", {}) or {}
    counts_html = _summary_grid(
        [(f"{status.title()} Rows", status_counts.get(status, 0)) for status in
         ["out", "doubtful", "questionable", "probable", "available", "unknown"]]
    )

    warning_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings) or "<li>none</li>"
    issue_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in issues) or "<li>none</li>"
    diagnostics = (
        "<h2>Validation Messages</h2>"
        "<div class='status'><strong>Issues</strong><ul>"
        + issue_items
        + "</ul><strong>Warnings</strong><ul>"
        + warning_items
        + "</ul></div>"
    )

    columns = [
        "team",
        "has_availability_data",
        "players_listed",
        "key_players_out",
        "questionable_players",
        "stale_data_warning",
        "last_updated",
        "source",
        "notes",
    ]
    table = "<h2>Fixture Team Availability</h2>" + _table(team_rows, columns=columns, max_rows=200)
    body = intro + "<h2>Availability Coverage</h2>" + summary + "<h2>Status Counts</h2>" + counts_html + diagnostics + table
    return _static_page("Team Availability", report_path, "team_availability.html", body)


# --- Matchup Predictions page (no-odds model probabilities) ----------------
_CONFIDENCE_RANK = {"High": 3, "Medium": 2, "Low": 1, "Very low": 0}
_CONFIDENCE_CLASS = {"High": "high", "Medium": "med", "Low": "low", "Very low": "vlow"}
_QUALITY_CLASS = {"strong": "high", "usable": "med", "weak": "low", "very_weak": "vlow"}


def _mp_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _mp_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _mp_list_html(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value if str(part).strip()]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    items = [str(part).strip() for part in parsed if str(part).strip()]
                else:
                    items = [text]
            except json.JSONDecodeError:
                items = [part.strip() for part in text.split(";") if part.strip()]
        else:
            items = [part.strip() for part in text.split(";") if part.strip()]
    if not items:
        return ""
    full_text = "; ".join(items)
    return (
        f"<div class='mp-text-cell' title='{html.escape(full_text)}'>"
        "<ul class='mp-list'>"
        + "".join(f"<li>{html.escape(i)}</li>" for i in items)
        + "</ul></div>"
    )


def _mp_badge(label: str, css: str) -> str:
    return f"<span class='mp-badge mp-{html.escape(css)}'>{html.escape(label)}</span>"


def _mp_pick_probability(row: Any) -> float | None:
    """Return the model probability for the predicted outcome, as a 0-1 float."""

    def _as_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    prob_a = _as_float(row.get("prob_team_a_win"))
    prob_draw = _as_float(row.get("prob_draw"))
    prob_b = _as_float(row.get("prob_team_b_win"))
    outcome = str(row.get("predicted_outcome", "")).lower()
    team_a = str(row.get("team_a", "")).lower()
    team_b = str(row.get("team_b", "")).lower()

    if "draw" in outcome and prob_draw:
        return prob_draw
    if team_b and team_b in outcome:
        return prob_b
    if team_a and team_a in outcome:
        return prob_a
    # Fall back to the largest available probability (argmax of the prediction).
    candidates = [p for p in (prob_a, prob_draw, prob_b) if p is not None]
    return max(candidates) if candidates else None


def _mp_inline_text(value: Any) -> str:
    """Flatten a reasons/risks/warnings field into a single inline string."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value if str(part).strip()]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                items = (
                    [str(part).strip() for part in parsed if str(part).strip()]
                    if isinstance(parsed, list)
                    else [text]
                )
            except json.JSONDecodeError:
                items = [part.strip() for part in text.split(";") if part.strip()]
        else:
            items = [part.strip() for part in text.split(";") if part.strip()]
    return "; ".join(items)


def _build_parlay_creator() -> str:
    """Render the interactive parlay creator panel (driven by row checkboxes)."""

    return (
        "<div class='parlay-creator'>"
        "<div class='parlay-head'>"
        "<h2>Parlay creator</h2>"
        "<span class='subtle'>Tick the <strong>+</strong> box on any prediction below to add it as a leg.</span>"
        "</div>"
        "<div class='parlay-body'>"
        "<p id='parlay-empty' class='parlay-empty'>No legs yet. Select predictions from the table to build a parlay.</p>"
        "<ul id='parlay-legs' class='parlay-legs'></ul>"
        "<div class='parlay-summary'>"
        "<div class='parlay-metric'><div class='metric-label'>Legs</div>"
        "<div class='metric-value' id='parlay-count'>0</div></div>"
        "<div class='parlay-metric'><div class='metric-label'>Combined model probability</div>"
        "<div class='metric-value' id='parlay-prob'>n/a</div></div>"
        "<div class='parlay-metric'><div class='metric-label'>Fair decimal odds</div>"
        "<div class='metric-value' id='parlay-decimal'>n/a</div></div>"
        "<div class='parlay-metric'><div class='metric-label'>Fair American odds</div>"
        "<div class='metric-value' id='parlay-american'>n/a</div></div>"
        "</div>"
        "<div class='parlay-actions'><button type='button' id='parlay-clear'>Clear all</button></div>"
        "<p class='parlay-note'>Combined probability multiplies each leg's model probability and assumes the "
        "legs are independent (different games). Fair odds are the break-even price implied by that probability "
        "&mdash; they are not sportsbook prices or a betting recommendation.</p>"
        "</div>"
        "</div>"
    )


def _mp_breakdown_table(title: str, data: dict[str, Any]) -> str:
    if not data:
        return ""
    rows = []
    for key, stats in sorted(data.items(), key=lambda kv: -float(kv[1].get("accuracy", 0))):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(key))}</td>"
            f"<td>{_fmt_pct(stats.get('accuracy'))}</td>"
            f"<td>{html.escape(str(stats.get('n', '')))}</td>"
            "</tr>"
        )
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Group</th><th>Accuracy</th><th>Games</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _build_matchup_predictions_backtest_section(report_path: Path) -> str:
    metrics = _read_json(report_path / "matchup_model_backtest.json")
    buckets = _read_csv(report_path / "matchup_model_backtest_by_bucket.csv")
    if not metrics and buckets.empty:
        return (
            "<h2>Model Backtest</h2>"
            "<div class='empty'>No backtest has been run yet. Run "
            "<code>python scripts/backtest_matchup_model.py</code> to populate this section.</div>"
        )

    grid = _summary_grid(
        [
            ("Games scored", _format_count(int(metrics.get("n_games", 0) or 0))),
            ("Accuracy", _fmt_pct(metrics.get("accuracy"))),
            ("Log loss", _fmt_number(metrics.get("log_loss"), 3)),
            ("Brier score", _fmt_number(metrics.get("brier_score"), 3)),
            ("Mean prob of actual", _fmt_number(metrics.get("mean_prob_of_actual_outcome"), 3)),
            ("Favourite win rate", _fmt_pct(metrics.get("favorite_win_rate"))),
        ]
    )

    breakdowns = (
        _mp_breakdown_table("Accuracy by confidence level", metrics.get("accuracy_by_confidence", {}))
        + _mp_breakdown_table("Accuracy by sport", metrics.get("accuracy_by_sport", {}))
        + _mp_breakdown_table("Accuracy by league", metrics.get("accuracy_by_league", {}))
        + _mp_breakdown_table("Accuracy by competition type", metrics.get("accuracy_by_competition_type", {}))
    )

    calibration = ""
    if not buckets.empty:
        calibration = (
            "<h3>Calibration by probability bucket</h3>"
            "<p class='note'>A well-calibrated model wins close to its predicted rate "
            "(e.g. 60% predictions win about 60% of the time).</p>"
            + _table(
                buckets,
                ["prob_bucket", "n_games", "mean_predicted_prob", "actual_win_rate", "calibration_gap"],
                max_rows=20,
            )
        )

    draw_html = ""
    draw = metrics.get("draw_quality")
    if isinstance(draw, dict) and draw:
        draw_html = (
            "<h3>Draw prediction quality (soccer)</h3>"
            + _summary_grid(
                [
                    ("Actual draw rate", _fmt_pct(draw.get("actual_draw_rate"))),
                    ("Predicted draw rate", _fmt_pct(draw.get("predicted_draw_rate"))),
                    ("Mean predicted draw prob", _fmt_number(draw.get("mean_predicted_draw_prob"), 3)),
                    ("Draw precision", _fmt_pct(draw.get("draw_precision"))),
                    ("Draw recall", _fmt_pct(draw.get("draw_recall"))),
                ]
            )
        )

    return "<h2>Model Backtest</h2>" + grid + breakdowns + calibration + draw_html


def _build_matchup_predictions_page(report_path: Path) -> str:
    """Build the Matchup Predictions page from the no-odds report artifacts."""

    preds = _read_csv(report_path / "matchup_predictions_today.csv")

    extra_style = """
    <style>
      .mp-page { max-width: 100%; }
      .mp-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin: 14px 0; }
      .mp-controls { display: flex; flex-wrap: wrap; gap: 10px; }
      .mp-controls input, .mp-controls select { padding: 8px 10px; border: 1px solid #d0d5dd; border-radius: 8px; font-size: 14px; }
      .mp-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
      .mp-actions a { border: 1px solid #d0d5dd; border-radius: 8px; padding: 8px 10px; background: #fff; color: #18202a; text-decoration: none; font-size: 13px; font-weight: 800; cursor: pointer; }
      /* Simplified, fits-on-screen predictions table */
      .mp-table-wrap { background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; overflow: hidden; }
      #mp-table { width: 100%; min-width: 0; border-collapse: collapse; table-layout: fixed; }
      #mp-table th, #mp-table td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #eef0f3; font-size: 14px; vertical-align: top; }
      #mp-table th { color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; background: #f8fafc; }
      #mp-table tbody tr:hover { background: #f9fbff; }
      .mp-col-pick { width: 44px; text-align: center; }
      .mp-col-date { width: 124px; }
      .mp-col-sport { width: 84px; }
      .mp-col-league { width: 120px; }
      .mp-col-teams { width: 200px; }
      .mp-col-pred { width: 220px; }
      .mp-col-reason { width: auto; }
      .mp-col-pick, .mp-col-date, .mp-col-sport, .mp-col-league, .mp-col-teams, .mp-col-pred { white-space: normal; word-break: break-word; }
      .mp-teams { font-weight: 800; }
      .mp-teams .mp-vs { color: #98a2b3; font-weight: 600; margin: 0 4px; }
      .mp-lean { font-weight: 850; display: block; margin-bottom: 4px; }
      .mp-probs { color: #667085; font-size: 12px; font-weight: 700; }
      .mp-prob { font-weight: 800; color: #18202a; }
      .mp-conf-row { margin-top: 5px; }
      .mp-list { margin: 0; padding-left: 16px; font-size: 13px; }
      .mp-list li { margin-bottom: 3px; }
      .mp-risk-note { margin-top: 6px; color: #9a6400; font-size: 12px; }
      .mp-warn-note { margin-top: 6px; color: #b42318; font-size: 12px; }
      .mp-badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 800; }
      .mp-high { background: #e3f6e9; color: #137a3a; }
      .mp-med  { background: #e7effe; color: #1b63ce; }
      .mp-low  { background: #fdf1dd; color: #9a6400; }
      .mp-vlow { background: #f0f1f4; color: #5a6472; }
      .mp-pick-box { width: 18px; height: 18px; cursor: pointer; }
      /* Parlay creator */
      .parlay-creator { margin: 18px 0 6px; border: 1px solid #d6e0f5; border-radius: 10px; background: #f7faff; overflow: hidden; }
      .parlay-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px; padding: 12px 16px; background: #eef4ff; border-bottom: 1px solid #d6e0f5; }
      .parlay-head h2 { margin: 0; font-size: 18px; }
      .parlay-head .subtle { font-size: 13px; }
      .parlay-body { padding: 14px 16px; }
      .parlay-empty { color: #667085; font-style: italic; }
      .parlay-legs { list-style: none; margin: 0 0 12px; padding: 0; display: grid; gap: 8px; }
      .parlay-leg { display: flex; align-items: center; justify-content: space-between; gap: 10px; background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: 8px 12px; }
      .parlay-leg .leg-main { font-weight: 800; }
      .parlay-leg .leg-sub { color: #667085; font-size: 12px; font-weight: 600; }
      .parlay-leg .leg-prob { font-weight: 800; color: #1b63ce; margin-left: auto; padding: 0 12px; }
      .parlay-leg .leg-remove { border: none; background: #fdecec; color: #b42318; border-radius: 6px; padding: 4px 9px; font-weight: 800; cursor: pointer; }
      .parlay-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 4px; }
      .parlay-metric { background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: 10px 12px; }
      .parlay-metric .metric-label { color: #667085; font-size: 12px; font-weight: 700; margin-bottom: 4px; }
      .parlay-metric .metric-value { font-size: 22px; font-weight: 900; }
      .parlay-actions { margin-top: 12px; display: flex; gap: 8px; }
      .parlay-actions button { border: 1px solid #d0d5dd; background: #fff; border-radius: 8px; padding: 8px 12px; font-weight: 800; cursor: pointer; }
      .parlay-note { margin-top: 10px; color: #667085; font-size: 12px; }
    </style>
    """

    intro = (
        "<p class='note'><strong>These are model-implied probabilities, not sportsbook odds.</strong> "
        "They are not betting odds, sportsbook prices, or bet recommendations. They estimate what the model thinks "
        "is most likely to happen based on history, team strength, form, rest, schedule, and "
        "availability. Treat low-confidence and weak-data rows with extra caution.</p>"
    )

    if preds.empty:
        body = (
            extra_style
            + intro
            + _build_availability_coverage_summary(report_path)
            + "<h2>Today &amp; Upcoming Games</h2>"
            + "<div class='empty'>No matchup predictions found. Run "
            + "<code>python scripts/build_matchup_predictions.py</code> to populate this page.</div>"
            + _build_matchup_predictions_backtest_section(report_path)
        )
        return _static_page("Matchup Predictions", report_path, "matchup_predictions.html", body)

    display_preds = preds.copy()
    if "fixture_id" not in display_preds.columns and "id" in display_preds.columns:
        display_preds["fixture_id"] = display_preds["id"]
    for column in [
        "game_date", "sport", "league", "team_a", "team_b", "prob_team_a_win",
        "prob_draw", "prob_team_b_win", "predicted_outcome", "confidence_level",
        "confidence_score", "data_quality", "key_reasons", "main_risks",
        "data_quality_warnings", "team_a_availability_present",
        "team_b_availability_present", "team_a_availability_source",
        "team_b_availability_source", "team_a_availability_last_updated",
        "team_b_availability_last_updated", "model_version", "fixture_id",
    ]:
        if column not in display_preds.columns:
            display_preds[column] = ""

    sports = sorted(display_preds.get("sport", pd.Series(dtype=str)).dropna().astype(str).unique())
    leagues = sorted(display_preds.get("league", pd.Series(dtype=str)).dropna().astype(str).unique())

    sport_options = "".join(f"<option value='{html.escape(s)}'>{html.escape(s)}</option>" for s in sports)
    league_options = "".join(f"<option value='{html.escape(l)}'>{html.escape(l)}</option>" for l in leagues)

    # Simplified, fits-on-screen columns. Order matches the user-facing layout.
    table_columns = [
        ("pick", "mp-col-pick"),
        ("game_date", "mp-col-date"),
        ("sport", "mp-col-sport"),
        ("league", "mp-col-league"),
        ("teams", "mp-col-teams"),
        ("prediction", "mp-col-pred"),
        ("reasoning", "mp-col-reason"),
    ]
    column_labels = {
        "pick": "+",
        "game_date": "Date",
        "sport": "Sport",
        "league": "League",
        "teams": "Teams",
        "prediction": "Prediction",
        "reasoning": "Reasoning",
    }

    rows_html = []
    for idx, (_, row) in enumerate(display_preds.iterrows()):
        sport = str(row.get("sport", ""))
        league = str(row.get("league", ""))
        team_a = str(row.get("team_a", ""))
        team_b = str(row.get("team_b", ""))
        confidence = str(row.get("confidence_level", ""))
        prob_draw = row.get("prob_draw")
        has_draw = not (pd.isna(prob_draw) or float(prob_draw or 0) == 0.0)
        date_text = format_local_datetime(row.get("game_date"))
        search_key = f"{team_a} {team_b}".lower()
        conf_rank = _CONFIDENCE_RANK.get(confidence, 0)
        outcome = str(row.get("predicted_outcome", ""))
        pick_prob = _mp_pick_probability(row)
        fixture_id = str(row.get("fixture_id", "")) or f"row-{idx}"

        teams_cell = (
            f"<span class='mp-teams'>{html.escape(team_a)}"
            f"<span class='mp-vs'>vs</span>{html.escape(team_b)}</span>"
        )
        prob_bits = (
            f"<span class='mp-prob'>{_mp_pct(row.get('prob_team_a_win'))}</span> {html.escape(team_a)}"
        )
        if has_draw:
            prob_bits += f" &middot; <span class='mp-prob'>{_mp_pct(prob_draw)}</span> Draw"
        prob_bits += (
            f" &middot; <span class='mp-prob'>{_mp_pct(row.get('prob_team_b_win'))}</span> {html.escape(team_b)}"
        )
        prediction_cell = (
            f"<span class='mp-lean'>{html.escape(outcome)}</span>"
            f"<span class='mp-probs'>{prob_bits}</span>"
            f"<div class='mp-conf-row'>{_mp_badge(confidence, _CONFIDENCE_CLASS.get(confidence, 'vlow'))}</div>"
        )

        reasoning_cell = _mp_list_html(row.get("key_reasons"))
        risks = _mp_inline_text(row.get("main_risks"))
        warnings = _mp_inline_text(row.get("data_quality_warnings"))
        if risks:
            reasoning_cell += f"<div class='mp-risk-note'><strong>Risks:</strong> {html.escape(risks)}</div>"
        if warnings:
            reasoning_cell += f"<div class='mp-warn-note'><strong>Data warning:</strong> {html.escape(warnings)}</div>"

        pick_label = outcome if outcome else f"{team_a} vs {team_b}"
        pick_attrs = (
            f"data-pick-id='{html.escape(fixture_id)}' "
            f"data-pick-prob='{pick_prob if pick_prob is not None else ''}' "
            f"data-pick-label='{html.escape(pick_label)}' "
            f"data-pick-match='{html.escape(team_a + ' vs ' + team_b)}' "
            f"data-pick-meta='{html.escape((sport + ' / ' + league).strip(' /'))} &middot; {html.escape(date_text)}'"
        )
        pick_cell = (
            f"<input type='checkbox' class='mp-pick-box' {pick_attrs}>"
            if pick_prob is not None
            else "<span class='subtle' title='No probability available'>&mdash;</span>"
        )

        values = {
            "pick": pick_cell,
            "game_date": html.escape(date_text),
            "sport": html.escape(sport),
            "league": html.escape(league),
            "teams": teams_cell,
            "prediction": prediction_cell,
            "reasoning": reasoning_cell,
        }

        rows_html.append(
            f"<tr class='mp-row' data-sport='{html.escape(sport)}' data-league='{html.escape(league)}' "
            f"data-team='{html.escape(search_key)}' data-conf='{conf_rank}'>"
            + "".join(
                f"<td class='{css}' data-col='{html.escape(col)}'>{values[col]}</td>"
                for col, css in table_columns
            )
            + "</tr>"
        )

    controls = (
        "<div class='mp-toolbar'>"
        "<div class='mp-controls'>"
        "<input id='mp-team' type='text' placeholder='Search team...'>"
        f"<select id='mp-sport'><option value=''>All sports</option>{sport_options}</select>"
        f"<select id='mp-league'><option value=''>All leagues</option>{league_options}</select>"
        "<select id='mp-conf'>"
        "<option value='0'>Any confidence</option>"
        "<option value='1'>Low or better</option>"
        "<option value='2'>Medium or better</option>"
        "<option value='3'>High only</option>"
        "</select>"
        "</div>"
        "<div class='mp-actions'>"
        "<a href='matchup_predictions_today.csv' download>Download CSV</a>"
        "</div>"
        "</div>"
    )

    header = "".join(
        f"<th class='{html.escape(css)}' data-col='{html.escape(col)}'>{html.escape(column_labels[col])}</th>"
        for col, css in table_columns
    )
    table = (
        "<div class='mp-table-wrap'><table id='mp-table'><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )

    parlay_creator = _build_parlay_creator()

    script = """
    <script>
      (function () {
        var team = document.getElementById('mp-team');
        var sport = document.getElementById('mp-sport');
        var league = document.getElementById('mp-league');
        var conf = document.getElementById('mp-conf');
        function apply() {
          var t = (team.value || '').toLowerCase();
          var s = sport.value, l = league.value, c = parseInt(conf.value || '0', 10);
          document.querySelectorAll('#mp-table tbody tr.mp-row').forEach(function (r) {
            var ok = true;
            if (t && r.getAttribute('data-team').indexOf(t) === -1) ok = false;
            if (s && r.getAttribute('data-sport') !== s) ok = false;
            if (l && r.getAttribute('data-league') !== l) ok = false;
            if (c && parseInt(r.getAttribute('data-conf'), 10) < c) ok = false;
            r.style.display = ok ? '' : 'none';
          });
        }
        [team, sport, league, conf].forEach(function (el) {
          el.addEventListener('input', apply);
          el.addEventListener('change', apply);
        });

        // ---- Parlay creator ----
        var legs = {};
        var legsEl = document.getElementById('parlay-legs');
        var emptyEl = document.getElementById('parlay-empty');
        var countEl = document.getElementById('parlay-count');
        var probEl = document.getElementById('parlay-prob');
        var decEl = document.getElementById('parlay-decimal');
        var amEl = document.getElementById('parlay-american');
        function americanFromDecimal(dec) {
          if (!isFinite(dec) || dec <= 1) return 'n/a';
          if (dec >= 2) return '+' + Math.round((dec - 1) * 100);
          return '-' + Math.round(100 / (dec - 1));
        }
        function render() {
          var ids = Object.keys(legs);
          countEl.textContent = ids.length;
          if (!ids.length) {
            emptyEl.style.display = '';
            legsEl.innerHTML = '';
            probEl.textContent = 'n/a';
            decEl.textContent = 'n/a';
            amEl.textContent = 'n/a';
            return;
          }
          emptyEl.style.display = 'none';
          var combined = 1, html = '';
          ids.forEach(function (id) {
            var leg = legs[id];
            combined *= leg.prob;
            html += '<li class="parlay-leg"><div><div class="leg-main">' + leg.label +
              '</div><div class="leg-sub">' + leg.match + ' &middot; ' + leg.meta + '</div></div>' +
              '<div class="leg-prob">' + (leg.prob * 100).toFixed(1) + '%</div>' +
              '<button class="leg-remove" data-remove="' + id + '">Remove</button></li>';
          });
          legsEl.innerHTML = html;
          probEl.textContent = (combined * 100).toFixed(1) + '%';
          var dec = combined > 0 ? 1 / combined : Infinity;
          decEl.textContent = isFinite(dec) ? dec.toFixed(2) : 'n/a';
          amEl.textContent = americanFromDecimal(dec);
          legsEl.querySelectorAll('[data-remove]').forEach(function (btn) {
            btn.addEventListener('click', function () {
              var id = btn.getAttribute('data-remove');
              delete legs[id];
              var box = document.querySelector('.mp-pick-box[data-pick-id="' + (window.CSS && CSS.escape ? CSS.escape(id) : id) + '"]');
              if (box) box.checked = false;
              render();
            });
          });
        }
        document.querySelectorAll('.mp-pick-box').forEach(function (box) {
          box.addEventListener('change', function () {
            var id = box.getAttribute('data-pick-id');
            var prob = parseFloat(box.getAttribute('data-pick-prob'));
            if (box.checked && isFinite(prob)) {
              legs[id] = {
                prob: prob,
                label: box.getAttribute('data-pick-label'),
                match: box.getAttribute('data-pick-match'),
                meta: box.getAttribute('data-pick-meta')
              };
            } else {
              delete legs[id];
            }
            render();
          });
        });
        var clearBtn = document.getElementById('parlay-clear');
        if (clearBtn) clearBtn.addEventListener('click', function () {
          legs = {};
          document.querySelectorAll('.mp-pick-box:checked').forEach(function (b) { b.checked = false; });
          render();
        });
        render();
      })();
    </script>
    """

    body = (
        extra_style
        + "<div class='mp-page'>"
        + intro
        + _build_availability_coverage_summary(report_path)
        + parlay_creator
        + "<h2>Today &amp; Upcoming Games</h2>"
        + controls
        + table
        + _build_matchup_predictions_backtest_section(report_path)
        + script
        + "</div>"
    )
    return _static_page("Matchup Predictions", report_path, "matchup_predictions.html", body)


def write_static_dashboard_pages(report_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Write static fallback dashboard pages and return their paths."""

    report_path = Path(report_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pages = {
        "matchup_predictions.html": _build_matchup_predictions_page(report_path),
        "team_availability.html": _build_team_availability_page(report_path),
        "research_picks.html": _build_research_picks_page(report_path),
        "paper_candidates.html": _build_paper_candidates_page(report_path),
        "trade_results.html": _build_trade_results_page(report_path),
        "parlay_research.html": _build_parlay_research_page(report_path),
        "player_props.html": _build_player_props_page(report_path),
        "project_cleanup.html": _build_project_cleanup_page(report_path),
        "recommendation_performance.html": _build_recommendation_performance_page(report_path),
        "recommendation_grading_audit.html": _build_recommendation_grading_audit_page(report_path),
        "proof_status.html": _build_proof_status_page(report_path),
    }
    written: list[Path] = []
    for filename, content in pages.items():
        path = output / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def write_dashboard(report_dir: str | Path, output_path: str | Path) -> Path:
    """Write dashboard HTML and return the output path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_dashboard_html(report_dir), encoding="utf-8")
    write_static_dashboard_pages(report_dir, output.parent)
    write_dashboard_simplification_report(report_dir)
    return output
