"""Terminal summary helpers."""

from __future__ import annotations

from typing import Any


def format_model_results(metrics: dict[str, Any]) -> str:
    """Format model metrics for terminal output."""

    best_model = metrics.get("best_model", "unknown")
    best = metrics.get("models", {}).get(best_model, {})
    return (
        "Model Results:\n"
        f"- Best model: {best_model}\n"
        f"- Accuracy: {best.get('accuracy', 0):.3f}\n"
        f"- Log loss: {best.get('log_loss', 0):.3f}\n"
        f"- Brier score: {best.get('brier_score', 0):.3f}\n"
        f"- ROC AUC: {best.get('roc_auc', 0):.3f}"
    )


def format_backtest_results(summary: dict[str, Any]) -> str:
    """Format backtest metrics for terminal output."""

    trade_timeline = summary.get("trade_timeline") or summary.get("market_timeline") or "n/a"
    return (
        "Backtest Results:\n"
        f"- Starting bankroll: ${summary.get('starting_bankroll', 0):.2f}\n"
        f"- Ending bankroll: ${summary.get('ending_bankroll', 0):.2f}\n"
        f"- Total return: {summary.get('total_return_pct', 0) * 100:.2f}%\n"
        f"- Number of trades ({trade_timeline}): {summary.get('num_trades', 0)}\n"
        f"- Win rate: {summary.get('win_rate', 0) * 100:.1f}%\n"
        f"- Max drawdown: {summary.get('max_drawdown', 0) * 100:.2f}%\n"
        f"- Best trade: ${summary.get('largest_win', 0):.2f}\n"
        f"- Worst trade: ${summary.get('largest_loss', 0):.2f}"
    )
