"""Plotting helpers for reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve


PLOT_BG = "#0f131a"
AX_BG = "#121720"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
ACCENT = "#7dd3fc"
GREEN = "#34d399"
RED = "#fb7185"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#26303d")
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(PLOT_BG)
        legend.get_frame().set_edgecolor("#26303d")
        for text in legend.get_texts():
            text.set_color(TEXT)


def _save_dark_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.patch.set_facecolor(PLOT_BG)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")


def save_equity_curve(trades: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a simple bankroll equity curve plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if trades.empty:
        ax.plot([], [])
    else:
        ax.plot(pd.to_datetime(trades["date"]), trades["bankroll_after"], linewidth=2, color=ACCENT)
    ax.set_title("Paper Trading Bankroll")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll ($)")
    ax.grid(True, alpha=0.25, color=GRID)
    _style_axes(ax)
    fig.autofmt_xdate()
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_calibration_plot(
    predictions: pd.DataFrame,
    output_path: str | Path,
    probability_column: str = "model_home_win_prob",
    target_column: str = "actual_home_win",
    n_bins: int = 10,
) -> Path:
    """Save a calibration plot for predicted win probabilities."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed, predicted = calibration_curve(
        predictions[target_column],
        predictions[probability_column],
        n_bins=n_bins,
        strategy="uniform",
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1.5, label="Perfect")
    ax.plot(predicted, observed, marker="o", linewidth=2, label="Model", color=ACCENT)
    ax.set_title("Home Win Probability Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Win Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, color=GRID)
    ax.legend()
    _style_axes(ax)
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_probability_distribution(
    predictions: pd.DataFrame,
    output_path: str | Path,
    probability_column: str = "model_home_win_prob",
) -> Path:
    """Save a histogram of model probabilities."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(predictions[probability_column], bins=25, color=ACCENT, alpha=0.85)
    ax.set_title("Predicted Home Win Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Games")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="y", alpha=0.25, color=GRID)
    _style_axes(ax)
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_edge_distribution(trades: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a histogram of model edge versus market probability."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    if not trades.empty:
        ax.hist(trades["edge"], bins=20, color=GREEN, alpha=0.85)
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.25)
    ax.set_title("Model Edge Distribution")
    ax.set_xlabel("Model Probability - Market Probability")
    ax.set_ylabel("Markets")
    ax.grid(True, axis="y", alpha=0.25, color=GRID)
    _style_axes(ax)
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_threshold_sweep_plot(sweep: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a multi-panel plot for threshold sweep metrics."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()

    plot_specs = [
        ("num_trades", "Trades", "Number of Trades"),
        ("total_return_pct", "Total Return", "Total Return"),
        ("win_rate", "Win Rate", "Win Rate"),
        ("max_drawdown", "Max Drawdown", "Max Drawdown"),
    ]

    for ax, (column, ylabel, title) in zip(axes, plot_specs):
        if column in sweep.columns:
            values = sweep[column]
            if column in {"total_return_pct", "win_rate", "max_drawdown"}:
                values = values * 100.0
                ylabel = f"{ylabel} (%)"
            ax.plot(sweep["edge_threshold"], values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Edge Threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, color=GRID)
        _style_axes(ax)

    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_probability_bin_plot(bins: pd.DataFrame, output_path: str | Path) -> Path:
    """Save predicted probability bucket calibration plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not bins.empty:
        labels = bins["probability_bin"].astype(str)
        ax.plot(labels, bins["avg_predicted_prob"], marker="o", label="Predicted", color=ACCENT)
        ax.plot(labels, bins["observed_win_rate"], marker="o", label="Observed", color=GREEN)
        ax.tick_params(axis="x", rotation=35)
    ax.set_title("Probability Bucket Calibration")
    ax.set_xlabel("Predicted Probability Bucket")
    ax.set_ylabel("Home Win Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, color=GRID)
    ax.legend()
    _style_axes(ax)
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_season_summary_plot(season_summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Save season-by-season model summary plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not season_summary.empty:
        ax.plot(season_summary["season"], season_summary["accuracy"], marker="o", label="Accuracy", color=ACCENT)
        ax.plot(season_summary["season"], season_summary["brier_score"], marker="o", label="Brier", color=GREEN)
    ax.set_title("Walk-Forward Season Summary")
    ax.set_xlabel("Season")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.25, color=GRID)
    ax.legend()
    _style_axes(ax)
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path


def save_edge_bin_plot(edge_bins: pd.DataFrame, output_path: str | Path) -> Path:
    """Save edge bucket profit plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not edge_bins.empty:
        labels = edge_bins["edge_bin"].astype(str)
        colors = [GREEN if value >= 0 else RED for value in edge_bins["total_profit"]]
        ax.bar(labels, edge_bins["total_profit"], color=colors, alpha=0.85)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title("Backtest Profit by Edge Bucket")
    ax.set_xlabel("Edge Bucket")
    ax.set_ylabel("Profit ($)")
    ax.grid(True, axis="y", alpha=0.25, color=GRID)
    _style_axes(ax)
    _save_dark_figure(fig, output_path)
    plt.close(fig)
    return output_path
