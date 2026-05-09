"""Generate a self-contained local HTML dashboard from report artifacts."""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
            text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap"><table{table_attrs}>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


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


def build_dashboard_html(report_dir: str | Path) -> str:
    """Return a self-contained HTML dashboard string."""

    report_path = Path(report_dir)
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


def write_dashboard(report_dir: str | Path, output_path: str | Path) -> Path:
    """Write dashboard HTML and return the output path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_dashboard_html(report_dir), encoding="utf-8")
    return output
