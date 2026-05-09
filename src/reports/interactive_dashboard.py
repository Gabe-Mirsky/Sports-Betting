"""Optional Streamlit dashboard for interactive report exploration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.kalshi_client import (
    MARKET_TEMPLATE_COLUMNS,
    match_games_to_markets,
    prepare_kalshi_markets,
    validate_kalshi_markets,
)
from strategy.backtest import run_backtest, summarize_backtest
from strategy.signal import add_yes_signals


ID_DTYPES = {
    "game_id": "string",
    "market_ticker": "string",
    "event_ticker": "string",
}

DEFAULT_STARTING_BANKROLL = 100.0
DEFAULT_MINIMUM_ADVANTAGE = 0.05
DEFAULT_MAX_RISK_PER_PICK = 0.03

FRIENDLY_COLUMNS = {
    "game_id": "Game ID",
    "market_ticker": "Market",
    "event_ticker": "Event",
    "game_date": "Date",
    "date": "Date",
    "season": "Season",
    "season_type": "Game Type",
    "home_team_abbr": "Home",
    "away_team_abbr": "Away",
    "yes_team_abbr": "Picked Team",
    "model_home_win_prob": "Our Home Win Chance",
    "model_away_win_prob": "Our Away Win Chance",
    "model_yes_prob": "Our Picked Team Win Chance",
    "model_prob": "Our Win Chance",
    "market_prob": "Market-Implied Chance",
    "yes_mid_cents": "Market Price (cents)",
    "price_cents": "Price Used (cents)",
    "edge": "Our Advantage",
    "expected_value": "Expected Profit per Contract",
    "trade": "Paper Pick?",
    "side": "Side",
    "shares": "Contracts",
    "cost": "Amount Risked",
    "payout": "Payout",
    "profit": "Profit/Loss",
    "bankroll_before": "Bankroll Before",
    "bankroll_after": "Bankroll After",
    "actual_home_win": "Home Won?",
    "actual_yes_win": "Picked Team Won?",
    "settlement": "Final Result",
    "reason": "Why",
    "price_source": "Price Source",
    "model_pick_team": "Model Pick",
    "model_pick_prob": "Model Pick Chance",
    "model_pick_won": "Model Pick Won?",
    "paper_decision": "Paper Decision",
    "paper_pick_won": "Paper Pick Won?",
    "has_market_price": "Market Price Loaded?",
    "upcoming_status": "Status",
}

PERCENT_COLUMNS = {
    "model_home_win_prob",
    "model_away_win_prob",
    "model_yes_prob",
    "model_prob",
    "model_pick_prob",
    "market_prob",
    "edge",
    "total_return_pct",
    "win_rate",
    "max_drawdown",
    "roi_on_amount_risked",
}

MONEY_COLUMNS = {
    "cost",
    "payout",
    "profit",
    "bankroll_before",
    "bankroll_after",
    "ending_bankroll",
    "starting_bankroll",
    "amount_risked",
    "total_profit",
    "expected_value",
}

REASON_LABELS = {
    "edge_met": "Advantage is big enough",
    "edge_below_threshold": "Advantage is too small",
    "market_price_below_minimum": "Market price is too low",
    "market_price_above_maximum": "Market price is too high",
    "invalid_market_price": "Missing or invalid price",
    "insufficient_bankroll_for_one_share": "Not enough bankroll for one contract",
}


@dataclass(frozen=True)
class ReportBundle:
    """Loaded report artifacts for dashboard rendering."""

    report_dir: Path
    model_metrics: dict[str, Any]
    walk_forward_metrics: dict[str, Any]
    backtest_summary: dict[str, Any]
    market_quality: dict[str, Any]
    threshold_sweep: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    backtest_trades: pd.DataFrame
    suggestions: pd.DataFrame
    probability_bins: pd.DataFrame
    season_summary: pd.DataFrame
    edge_bins: pd.DataFrame
    top_trades: pd.DataFrame
    walk_forward_predictions: pd.DataFrame
    all_game_predictions: pd.DataFrame
    upcoming_predictions: pd.DataFrame
    upcoming_market_suggestions: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=ID_DTYPES)


def load_report_bundle(report_dir: str | Path) -> ReportBundle:
    """Load all dashboard report artifacts from a report directory."""

    report_path = Path(report_dir)
    return ReportBundle(
        report_dir=report_path,
        model_metrics=_read_json(report_path / "model_metrics.json"),
        walk_forward_metrics=_read_json(report_path / "walk_forward_metrics.json"),
        backtest_summary=_read_json(report_path / "backtest_summary.json"),
        market_quality=_read_json(report_path / "market_data_quality_report.json"),
        threshold_sweep=_read_csv(report_path / "threshold_sweep.csv"),
        feature_diagnostics=_read_csv(report_path / "model_feature_diagnostics.csv"),
        backtest_trades=_read_csv(report_path / "backtest_trades.csv"),
        suggestions=_read_csv(report_path / "paper_trade_suggestions.csv"),
        probability_bins=_read_csv(report_path / "prediction_probability_bins.csv"),
        season_summary=_read_csv(report_path / "prediction_season_summary.csv"),
        edge_bins=_read_csv(report_path / "backtest_edge_bins.csv"),
        top_trades=_read_csv(report_path / "top_backtest_trades.csv"),
        walk_forward_predictions=_read_csv(report_path / "walk_forward_predictions.csv"),
        all_game_predictions=_read_csv(report_path / "all_game_predictions.csv"),
        upcoming_predictions=_read_csv(report_path / "upcoming_predictions.csv"),
        upcoming_market_suggestions=_read_csv(report_path / "upcoming_market_suggestions.csv"),
    )


def format_number(value: Any, digits: int = 2) -> str:
    """Format numeric values for dashboard display."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def format_money(value: Any) -> str:
    """Format money values for dashboard display."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.2f}"


def format_pct(value: Any) -> str:
    """Format decimal percentages for dashboard display."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def format_pct_points(value: Any) -> str:
    """Format decimal values as percentage points."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.1f} pts"


def season_label(season_start_year: Any) -> str:
    """Return a friendly NBA season label like 2025-26."""

    if season_start_year is None or pd.isna(season_start_year):
        return "n/a"
    year = int(season_start_year)
    return f"{year}-{(year + 1) % 100:02d}"


def friendly_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a readable display dataframe with friendly labels and formatting."""

    if df.empty:
        return df

    display = df.copy()
    if columns:
        display = display[[column for column in columns if column in display.columns]]

    for column in display.columns:
        if column == "season":
            display[column] = display[column].map(season_label)
        elif column in PERCENT_COLUMNS:
            display[column] = display[column].map(format_pct)
        elif column in MONEY_COLUMNS:
            display[column] = display[column].map(format_money)
        elif column in {"trade", "model_pick_won", "paper_pick_won", "has_market_price"}:
            display[column] = _format_bool_series(display[column])
        elif column in {"actual_home_win", "actual_yes_win"}:
            display[column] = _format_bool_series(display[column])
        elif column == "reason":
            display[column] = display[column].map(lambda value: REASON_LABELS.get(str(value), str(value)))

    renamed_columns = []
    seen: dict[str, int] = {}
    for column in display.columns:
        label = FRIENDLY_COLUMNS.get(column, column)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({seen[label]})"
        renamed_columns.append(label)

    display.columns = renamed_columns
    return display


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _format_bool_series(series: pd.Series) -> pd.Series:
    formatted = _coerce_bool(series).map(lambda value: "Yes" if value else "No")
    return formatted.mask(series.isna(), "n/a")


def available_seasons(predictions: pd.DataFrame) -> list[int]:
    """Return sorted seasons from prediction output."""

    if predictions.empty or "season" not in predictions.columns:
        return []
    return sorted(int(season) for season in predictions["season"].dropna().unique())


def available_teams(*frames: pd.DataFrame) -> list[str]:
    """Return sorted team abbreviations found in common report columns."""

    team_columns = [
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "no_team_abbr",
    ]
    teams: set[str] = set()
    for frame in frames:
        if frame.empty:
            continue
        for column in team_columns:
            if column in frame.columns:
                teams.update(str(value) for value in frame[column].dropna().unique())
    return sorted(teams)


def latest_game_date(predictions: pd.DataFrame) -> str:
    """Return the latest game date available in prediction reports."""

    if predictions.empty or "game_date" not in predictions.columns:
        return "n/a"
    dates = pd.to_datetime(predictions["game_date"], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    return dates.max().date().isoformat()


def latest_season_game_count(predictions: pd.DataFrame) -> int:
    """Return prediction row count for the latest available NBA season."""

    if predictions.empty or "season" not in predictions.columns:
        return 0
    seasons = pd.to_numeric(predictions["season"], errors="coerce")
    if seasons.dropna().empty:
        return 0
    latest_season = int(seasons.dropna().max())
    return int((seasons == latest_season).sum())


def summary_timeline(summary: dict[str, Any]) -> str:
    timeline = summary.get("trade_timeline") or summary.get("market_timeline")
    return str(timeline) if timeline else "n/a"


def build_all_game_decisions(
    predictions: pd.DataFrame,
    suggestions: pd.DataFrame,
) -> pd.DataFrame:
    """Build one plain-English decision row for every predicted historical game."""

    if predictions.empty:
        return pd.DataFrame()

    rows = predictions.copy()
    home_prob = pd.to_numeric(rows["model_home_win_prob"], errors="coerce")
    rows["model_pick_team"] = rows["away_team_abbr"]
    rows.loc[home_prob >= 0.5, "model_pick_team"] = rows.loc[home_prob >= 0.5, "home_team_abbr"]
    rows["model_pick_prob"] = rows["model_away_win_prob"]
    rows.loc[home_prob >= 0.5, "model_pick_prob"] = rows.loc[home_prob >= 0.5, "model_home_win_prob"]
    if "actual_home_win" in rows.columns:
        actual_home_win = _coerce_bool(rows["actual_home_win"])
        picked_home = rows["model_pick_team"] == rows["home_team_abbr"]
        rows["model_pick_won"] = (picked_home & actual_home_win) | (~picked_home & ~actual_home_win)
    else:
        rows["model_pick_won"] = pd.NA

    market_columns = [
        "game_id",
        "yes_team_abbr",
        "market_prob",
        "edge",
        "price_cents",
        "trade",
        "actual_yes_win",
        "reason",
    ]
    if suggestions.empty:
        market = pd.DataFrame(columns=market_columns)
    else:
        market = suggestions.reindex(columns=market_columns).copy()
        market = market.drop_duplicates(subset=["game_id"], keep="first")

    rows = rows.merge(market, on="game_id", how="left", suffixes=("", "_market"))
    rows["has_market_price"] = rows["price_cents"].notna()
    rows["paper_pick_won"] = rows["actual_yes_win"]
    rows["paper_decision"] = "No market price loaded"

    has_market = rows["has_market_price"]
    if "trade" in rows.columns:
        trade_bool = _coerce_bool(rows["trade"])
        rows.loc[has_market & trade_bool, "paper_decision"] = "Paper bet"
        rows.loc[has_market & ~trade_bool, "paper_decision"] = rows.loc[
            has_market & ~trade_bool,
            "reason",
        ].map(lambda value: REASON_LABELS.get(str(value), str(value)))

    return rows.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build_upcoming_display(upcoming_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a friendly upcoming game prediction table."""

    if upcoming_predictions.empty:
        return pd.DataFrame()

    rows = upcoming_predictions.copy()
    home_prob = pd.to_numeric(rows["model_home_win_prob"], errors="coerce")
    rows["model_pick_team"] = rows["away_team_abbr"]
    rows.loc[home_prob >= 0.5, "model_pick_team"] = rows.loc[home_prob >= 0.5, "home_team_abbr"]
    rows["model_pick_prob"] = rows["model_away_win_prob"]
    rows.loc[home_prob >= 0.5, "model_pick_prob"] = rows.loc[home_prob >= 0.5, "model_home_win_prob"]
    if "upcoming_status" not in rows.columns:
        rows["upcoming_status"] = "Scheduled"
    return rows.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build_upcoming_market_display(
    upcoming_predictions: pd.DataFrame,
    market_suggestions: pd.DataFrame,
) -> pd.DataFrame:
    """Build one upcoming row per game with the market price for the model's pick."""

    upcoming = build_upcoming_display(upcoming_predictions)
    if upcoming.empty:
        return pd.DataFrame()

    if market_suggestions.empty:
        upcoming["has_market_price"] = False
        upcoming["paper_decision"] = "No public market price found"
        return upcoming

    suggestions = market_suggestions.copy()
    suggestions["game_id"] = suggestions["game_id"].astype(str)
    upcoming["game_id"] = upcoming["game_id"].astype(str)

    merged = upcoming.merge(
        suggestions[
            [
                column
                for column in [
                    "game_id",
                    "yes_team_abbr",
                    "model_yes_prob",
                    "market_prob",
                    "edge",
                    "price_cents",
                    "trade",
                    "reason",
                    "market_ticker",
                    "price_source",
                ]
                if column in suggestions.columns
            ]
        ],
        left_on=["game_id", "model_pick_team"],
        right_on=["game_id", "yes_team_abbr"],
        how="left",
    )
    merged["has_market_price"] = merged["price_cents"].notna()
    merged["paper_decision"] = "No public market price found"
    if "trade" in merged.columns:
        trade_bool = _coerce_bool(merged["trade"])
        merged.loc[merged["has_market_price"] & trade_bool, "paper_decision"] = "Paper bet"
        merged.loc[merged["has_market_price"] & ~trade_bool, "paper_decision"] = merged.loc[
            merged["has_market_price"] & ~trade_bool,
            "reason",
        ].map(lambda value: REASON_LABELS.get(str(value), str(value)))
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def filter_predictions(
    predictions: pd.DataFrame,
    seasons: list[int] | None = None,
    teams: list[str] | None = None,
    min_home_prob: float | None = None,
    max_home_prob: float | None = None,
) -> pd.DataFrame:
    """Filter walk-forward predictions by season, team, and probability range."""

    if predictions.empty:
        return predictions

    filtered = predictions.copy()
    if seasons and "season" in filtered.columns:
        filtered = filtered[filtered["season"].isin(seasons)]
    if teams:
        mask = pd.Series(False, index=filtered.index)
        if "home_team_abbr" in filtered.columns:
            mask = mask | filtered["home_team_abbr"].isin(teams)
        if "away_team_abbr" in filtered.columns:
            mask = mask | filtered["away_team_abbr"].isin(teams)
        filtered = filtered[mask]
    if min_home_prob is not None and "model_home_win_prob" in filtered.columns:
        filtered = filtered[filtered["model_home_win_prob"] >= min_home_prob]
    if max_home_prob is not None and "model_home_win_prob" in filtered.columns:
        filtered = filtered[filtered["model_home_win_prob"] <= max_home_prob]
    return filtered


def filter_backtest_trades(
    trades: pd.DataFrame,
    teams: list[str] | None = None,
    min_edge: float | None = None,
    only_trades: bool = True,
) -> pd.DataFrame:
    """Filter backtest rows by team, edge, and trade flag."""

    if trades.empty:
        return trades

    filtered = trades.copy()
    if teams:
        mask = pd.Series(False, index=filtered.index)
        for column in ["home_team_abbr", "away_team_abbr", "yes_team_abbr"]:
            if column in filtered.columns:
                mask = mask | filtered[column].isin(teams)
        filtered = filtered[mask]
    if min_edge is not None and "edge" in filtered.columns:
        filtered = filtered[filtered["edge"] >= min_edge]
    if only_trades and "trade" in filtered.columns:
        filtered = filtered[_coerce_bool(filtered["trade"])]
    return filtered


def run_manual_market_backtest(
    predictions: pd.DataFrame,
    markets: pd.DataFrame,
    starting_bankroll: float = 100.0,
    edge_threshold: float = 0.05,
    max_bet_fraction: float = 0.03,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
) -> dict[str, Any]:
    """Validate, match, signal, and backtest a manual market dataframe."""

    if predictions.empty or markets.empty:
        empty_trades = pd.DataFrame()
        return {
            "markets": prepare_kalshi_markets(markets) if not markets.empty else pd.DataFrame(),
            "validation": validate_kalshi_markets(markets, predictions) if not markets.empty else {},
            "matched": pd.DataFrame(),
            "signals": pd.DataFrame(),
            "resolved": pd.DataFrame(),
            "trades": empty_trades,
            "summary": summarize_backtest(empty_trades, starting_bankroll),
        }

    prepared = prepare_kalshi_markets(markets)
    validation = validate_kalshi_markets(prepared, predictions)
    if validation.get("missing_columns"):
        empty_trades = pd.DataFrame()
        return {
            "markets": prepared,
            "validation": validation,
            "matched": pd.DataFrame(),
            "signals": pd.DataFrame(),
            "resolved": pd.DataFrame(),
            "trades": empty_trades,
            "summary": summarize_backtest(empty_trades, starting_bankroll),
        }

    matched = match_games_to_markets(predictions, prepared)
    if matched.empty:
        empty_trades = pd.DataFrame()
        return {
            "markets": prepared,
            "validation": validation,
            "matched": matched,
            "signals": pd.DataFrame(),
            "resolved": pd.DataFrame(),
            "trades": empty_trades,
            "summary": summarize_backtest(empty_trades, starting_bankroll),
        }

    signals = add_yes_signals(
        matched,
        edge_threshold=edge_threshold,
        min_market_price=min_market_price,
        max_market_price=max_market_price,
    )
    resolved = matched[matched["actual_yes_win"].notna()].copy()
    if resolved.empty:
        empty_trades = pd.DataFrame()
        return {
            "markets": prepared,
            "validation": validation,
            "matched": matched,
            "signals": signals,
            "resolved": resolved,
            "trades": empty_trades,
            "summary": summarize_backtest(empty_trades, starting_bankroll),
        }

    trades = run_backtest(
        resolved,
        starting_bankroll=starting_bankroll,
        edge_threshold=edge_threshold,
        max_bet_fraction=max_bet_fraction,
        min_market_price=min_market_price,
        max_market_price=max_market_price,
    )
    return {
        "markets": prepared,
        "validation": validation,
        "matched": matched,
        "signals": signals,
        "resolved": resolved,
        "trades": trades,
        "summary": summarize_backtest(trades, starting_bankroll),
    }


def file_status(report_dir: str | Path) -> pd.DataFrame:
    """Return availability and size for dashboard source files."""

    report_path = Path(report_dir)
    files = [
        "model_metrics.json",
        "walk_forward_metrics.json",
        "backtest_summary.json",
        "market_data_quality_report.json",
        "live_market_quality_report.json",
        "threshold_sweep.csv",
        "model_feature_diagnostics.csv",
        "backtest_trades.csv",
        "paper_trade_suggestions.csv",
        "prediction_probability_bins.csv",
        "prediction_season_summary.csv",
        "backtest_edge_bins.csv",
        "top_backtest_trades.csv",
        "walk_forward_predictions.csv",
        "all_game_predictions.csv",
        "upcoming_predictions.csv",
        "upcoming_market_suggestions.csv",
    ]
    rows = []
    for filename in files:
        path = report_path / filename
        rows.append(
            {
                "file": filename,
                "exists": path.exists(),
                "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_app(default_report_dir: str | Path | None = None) -> None:
    """Run the Streamlit dashboard."""

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Streamlit is not installed. Install it with "
            "`python -m pip install -r requirements-dashboard.txt`."
        ) from exc

    project_root = Path(__file__).resolve().parents[2]
    default_dir = Path(default_report_dir) if default_report_dir else project_root / "data" / "reports"

    st.set_page_config(
        page_title="NBA Kalshi Predictor",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #090b0f; color: #eef2f7; }
        .block-container { padding-top: 1.6rem; }
        [data-testid="stMetric"] {
            background: #121720;
            border: 1px solid #26303d;
            border-radius: 8px;
            padding: 0.8rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #26303d;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Advanced", expanded=False):
        report_dir_text = st.text_input("Report Folder", value=str(default_dir))
        st.caption(
            "Pick rules are automatic: the app uses the saved model, a $100 fake bankroll, "
            "and conservative paper-trading defaults."
        )
    report_dir = Path(report_dir_text)
    bundle = load_report_bundle(report_dir)

    historical_predictions = (
        bundle.all_game_predictions
        if not bundle.all_game_predictions.empty
        else bundle.walk_forward_predictions
    )
    all_game_decisions = build_all_game_decisions(historical_predictions, bundle.suggestions)
    upcoming_games = build_upcoming_display(bundle.upcoming_predictions)
    upcoming_market_games = build_upcoming_market_display(
        bundle.upcoming_predictions,
        bundle.upcoming_market_suggestions,
    )
    paper_trades = filter_backtest_trades(bundle.backtest_trades, only_trades=True)
    walk_model = bundle.walk_forward_metrics.get("overall", {}).get("model", {})
    model_pick_accuracy = walk_model.get("accuracy")

    st.title("NBA Paper Trading Dashboard")
    st.caption(
        "Research only. The model makes the picks; the app shows every historical game, "
        "paper-trading decisions, and upcoming predictions."
    )

    backtest = bundle.backtest_summary
    overview_cols = st.columns(6)
    overview_cols[0].metric("Latest Game Data", latest_game_date(historical_predictions))
    overview_cols[1].metric("Historical Games", f"{len(all_game_decisions):,}")
    overview_cols[2].metric("Model Pick Accuracy", format_pct(model_pick_accuracy))
    overview_cols[3].metric(f"Paper Bets ({summary_timeline(backtest)})", str(backtest.get("num_trades", "n/a")))
    overview_cols[4].metric("Ending Bankroll", format_money(backtest.get("ending_bankroll")))
    overview_cols[5].metric("Upcoming Games", f"{len(upcoming_games):,}")

    overview_tab, all_games_tab, upcoming_tab, paper_tab, manual_tab, quality_tab = st.tabs(
        ["Quick View", "All Games", "Upcoming", "Paper Results", "Try Market CSV", "Data Checks"]
    )

    with overview_tab:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Fake Bankroll Over Time")
            if not paper_trades.empty and {"date", "bankroll_after"}.issubset(paper_trades.columns):
                equity = paper_trades.copy()
                equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
                st.line_chart(equity.set_index("date")["bankroll_after"])
            else:
                st.info("No resolved paper bets yet. Load market prices to create paper-trading results.")
        with col_b:
            st.subheader("Model Check by Season")
            if not bundle.season_summary.empty:
                season_summary = bundle.season_summary.set_index("season")
                columns = [column for column in ["accuracy", "brier_score"] if column in season_summary.columns]
                st.line_chart(season_summary[columns])
            else:
                st.info("No season check data yet.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Upcoming Model Picks")
            upcoming_preview = upcoming_market_games if not upcoming_market_games.empty else upcoming_games
            if not upcoming_preview.empty:
                st.dataframe(
                    friendly_table(
                        upcoming_preview.head(12),
                        columns=[
                            "game_date",
                            "season_type",
                            "home_team_abbr",
                            "away_team_abbr",
                            "model_pick_team",
                            "model_pick_prob",
                            "price_cents",
                            "paper_decision",
                            "upcoming_status",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No upcoming predictions saved yet. Run the upcoming-games command after this update.")
        with col_b:
            st.subheader("Estimated Chance vs. Actual Results")
            if not bundle.probability_bins.empty:
                prob_bins = bundle.probability_bins.set_index("probability_bin")
                columns = [
                    column
                    for column in ["avg_predicted_prob", "observed_win_rate"]
                    if column in prob_bins.columns
                ]
                st.line_chart(prob_bins[columns])
            else:
                st.info("No probability bucket data yet.")

    with all_games_tab:
        st.subheader("Every Historical Game in the Dataset")
        st.caption(
            "Paper bets need market prices. Games without a loaded market price still show the final model pick "
            "and whether that pick won. The accuracy number above comes from walk-forward testing."
        )
        st.dataframe(
            friendly_table(
                all_game_decisions.sort_values(["game_date", "game_id"], ascending=[False, False]),
                columns=[
                    "game_date",
                    "season",
                    "season_type",
                    "home_team_abbr",
                    "away_team_abbr",
                    "model_pick_team",
                    "model_pick_prob",
                    "model_pick_won",
                    "has_market_price",
                    "paper_decision",
                    "trade",
                    "paper_pick_won",
                    "actual_home_win",
                    "model_home_win_prob",
                    "model_away_win_prob",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

    with upcoming_tab:
        st.subheader("Upcoming Games and Predictions")
        upcoming_display = upcoming_market_games if not upcoming_market_games.empty else upcoming_games
        if upcoming_display.empty:
            st.info("No upcoming predictions are saved yet. I am adding the command that creates this table.")
        else:
            st.dataframe(
                friendly_table(
                    upcoming_display,
                    columns=[
                        "game_date",
                        "season",
                        "season_type",
                        "home_team_abbr",
                        "away_team_abbr",
                        "model_pick_team",
                        "model_pick_prob",
                        "price_cents",
                        "market_prob",
                        "edge",
                        "paper_decision",
                        "model_home_win_prob",
                        "model_away_win_prob",
                        "upcoming_status",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )
        if not bundle.upcoming_market_suggestions.empty:
            st.subheader("All Public Kalshi Contracts Matched")
            st.dataframe(
                friendly_table(
                    bundle.upcoming_market_suggestions.sort_values(["game_date", "market_ticker"]),
                    columns=[
                        "game_date",
                        "home_team_abbr",
                        "away_team_abbr",
                        "yes_team_abbr",
                        "model_yes_prob",
                        "market_prob",
                        "edge",
                        "price_cents",
                        "trade",
                        "reason",
                        "market_ticker",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )

    with paper_tab:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Paper Profit by Model Advantage")
            if not bundle.edge_bins.empty and {"edge_bin", "total_profit"}.issubset(bundle.edge_bins.columns):
                st.bar_chart(bundle.edge_bins.set_index("edge_bin")["total_profit"])
            else:
                st.info("No paper-profit breakdown yet.")
        with col_b:
            st.subheader("Biggest Paper Wins and Losses")
            if not bundle.top_trades.empty:
                st.dataframe(
                    friendly_table(
                        bundle.top_trades,
                        columns=[
                            "date",
                            "home_team_abbr",
                            "away_team_abbr",
                            "yes_team_abbr",
                            "edge",
                            "shares",
                            "profit",
                            "bankroll_after",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No paper pick data yet.")
        st.subheader("Resolved Paper Bets")
        st.dataframe(
            friendly_table(
                paper_trades,
                columns=[
                    "date",
                    "home_team_abbr",
                    "away_team_abbr",
                    "yes_team_abbr",
                    "model_yes_prob",
                    "market_prob",
                    "edge",
                    "price_cents",
                    "shares",
                    "profit",
                    "bankroll_after",
                    "reason",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

    with manual_tab:
        st.subheader("Try Your Own Market CSV")
        st.caption(
            "Use pre-game prices only. The app automatically applies the saved model and conservative "
            "paper-trading rules."
        )

        default_market_path = project_root / "data" / "kalshi" / "markets_mock.csv"
        market_path_text = st.text_input("Local Market CSV", value=str(default_market_path))
        uploaded_market_file = st.file_uploader("Or upload a market CSV", type=["csv"])

        markets_source = "none"
        manual_markets = pd.DataFrame()
        if uploaded_market_file is not None:
            manual_markets = pd.read_csv(uploaded_market_file, dtype=ID_DTYPES)
            markets_source = uploaded_market_file.name
        elif market_path_text:
            local_market_path = Path(market_path_text)
            if local_market_path.exists():
                manual_markets = pd.read_csv(local_market_path, dtype=ID_DTYPES)
                markets_source = str(local_market_path)
            else:
                st.warning("Market CSV path does not exist.")

        template = pd.DataFrame(columns=MARKET_TEMPLATE_COLUMNS)
        st.download_button(
            "Download Blank Market Template",
            data=template.to_csv(index=False),
            file_name="markets_template.csv",
            mime="text/csv",
        )

        if manual_markets.empty:
            st.info("Load a market CSV to preview signals and run a resolved-game backtest.")
        else:
            result = run_manual_market_backtest(
                bundle.walk_forward_predictions,
                manual_markets,
                starting_bankroll=DEFAULT_STARTING_BANKROLL,
                edge_threshold=DEFAULT_MINIMUM_ADVANTAGE,
                max_bet_fraction=DEFAULT_MAX_RISK_PER_PICK,
            )
            summary = result["summary"]
            validation = result["validation"]

            st.caption(f"Loaded from: {markets_source}")
            metric_cols = st.columns(5)
            metric_cols[0].metric("Rows Loaded", str(len(result["markets"])))
            metric_cols[1].metric("Matched Games", str(len(result["matched"])))
            metric_cols[2].metric("With Final Results", str(len(result["resolved"])))
            metric_cols[3].metric(f"Paper Picks ({summary_timeline(summary)})", str(summary.get("num_trades", 0)))
            metric_cols[4].metric("Ending Bankroll", format_money(summary.get("ending_bankroll")))

            issues = validation.get("issues", [])
            if issues:
                for issue in issues:
                    st.warning(str(issue))
            else:
                st.success("Market file passed validation checks.")

            st.subheader("Suggested Paper Picks")
            st.dataframe(
                friendly_table(
                    result["signals"],
                    columns=[
                        "game_date",
                        "home_team_abbr",
                        "away_team_abbr",
                        "yes_team_abbr",
                        "model_yes_prob",
                        "market_prob",
                        "edge",
                        "price_cents",
                        "trade",
                        "reason",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )

            if result["trades"].empty:
                st.info("No matched rows have final results yet. Add settlement values for historical rows to see paper profit/loss.")
            else:
                st.subheader("Paper Profit/Loss")
                st.dataframe(
                    friendly_table(
                        result["trades"],
                        columns=[
                            "date",
                            "home_team_abbr",
                            "away_team_abbr",
                            "yes_team_abbr",
                            "model_yes_prob",
                            "market_prob",
                            "edge",
                            "shares",
                            "cost",
                            "profit",
                            "bankroll_after",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    "Download Manual Backtest Trades",
                    data=result["trades"].to_csv(index=False),
                    file_name="manual_backtest_trades.csv",
                    mime="text/csv",
                )

    with quality_tab:
        warnings = bundle.market_quality.get("warnings", [])
        if warnings:
            for warning in warnings:
                st.warning(str(warning))
        else:
            st.success("No market data quality warnings.")
        st.json(bundle.market_quality)
        st.dataframe(file_status(report_dir), width="stretch", hide_index=True)
        st.subheader("Model Feature Diagnostics")
        st.dataframe(bundle.feature_diagnostics, width="stretch", hide_index=True)
