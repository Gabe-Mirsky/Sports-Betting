# NBA Kalshi Predictor Project State Report

Date: 2026-06-09  
Workspace: `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor`  
Current goal: prove real, repeatable edge on single-game Kalshi NBA markets before allowing fair-price recommendations or parlay work.

## Executive Summary

The project is a local NBA/Kalshi market research system. It has a mature repeatable pipeline for NBA data, leak-safe features, walk-forward modeling, Kalshi market matching, bid/ask candle pricing, market truth audits, signal generation, fake-bankroll backtests, calibration diagnostics, shrinkage research, NO-side settlement audits, proof gates, and a static dashboard.

The current state is still research-only:

- Single-game edge is not proven.
- Proof gates remain strict.
- Fair-price recommendations are proof-gated to `No bet`.
- Parlays remain blocked.
- Reliability/source safety rework is complete.
- The canonical backtest is still Kalshi bid/ask candle based.
- Side-specific shrinkage reduced losses but did not prove edge.
- NO-side settlement audit shows NO CLV is useful but NO probabilities are overconfident.

The system should not be used as a live betting tool. The next useful work is calibration research, especially NO probability recalibration by prior-period settlement buckets.

## Current Codebase Shape

Current Python footprint:

- `src`: 106 files, about 30,436 lines.
- `scripts`: 99 files, about 12,482 lines.
- `tests`: 78 files, about 8,644 lines.
- `src/strategy`: 44 files, about 12,149 lines.
- `src/data`: 29 files, about 7,783 lines.
- `src/reports`: 11 files, about 6,560 lines.
- `src/models`: 10 files, about 2,073 lines.
- `src/features`: 7 files, about 1,196 lines.

Generated reports:

- `data/reports`: 533 entries after the latest cached pipeline run.

Important note: most files under `data/reports`, `data/raw`, `data/interim`, `data/processed`, `data/models`, and `outputs` are generated artifacts, not source.

## Current Canonical Pipeline

The repeatable cached command is:

```powershell
.\run_cached_pipeline.bat
```

Latest successful run:

- Completed 59 single-game research steps.
- Rebuilt `data/reports/dashboard.html`.
- Regenerated canonical Kalshi bid/ask backtest reports.
- Regenerated side-specific shrinkage reports.
- Regenerated NO settlement calibration reports.

The pipeline now includes these late-stage proof/research steps:

1. Build single-game proof gates.
2. Build fair-price single-game signals.
3. Build conservative parlay recommendations.
4. Sweep side-specific probability shrinkage.
5. Audit NO settlement calibration.
6. Analyze single-game edge root causes.
7. Build dashboard.

## Canonical Backtest Status

Source file: `data/reports/backtest_summary.json`

Current canonical metadata:

- `market_source`: `kalshi`
- `price_source`: `kalshi_candlesticks_bid_ask`
- `canonical_kalshi_backtest`: `true`
- `stale_artifacts_detected`: `false`
- `bid_ask_required`: `true`
- `no_trades_allowed`: `true`

Backtest result:

- Markets seen: 1,224.
- Trades: 634.
- YES trades: 325.
- NO trades: 309.
- Starting bankroll: $100.00.
- Ending bankroll: $28.86.
- ROI on amount risked: -2.90%.
- Average CLV: -0.0298 cents.
- Positive CLV rate: 24.3% in `backtest_summary.json`, 25.5% in `clv_summary.json` after dropping rows without CLV.
- YES profit: -$81.43.
- YES average CLV: -0.1625 cents.
- NO profit: +$10.29.
- NO average CLV: +0.1197 cents.
- Max drawdown: -90.9%.

Interpretation:

The canonical strategy loses badly overall. YES is the largest loss driver. NO is less bad and slightly profitable in the canonical backtest, but NO settlement calibration is weak.

## Market Truth Audit

Source file: `data/reports/market_truth_audit_summary.json`

Current market data quality:

- Matched game markets: 1,232.
- Auto matched: 1,232.
- Needs review: 0.
- Usable 60m prices: 1,232.
- Usable 30m prices: 1,232.
- Usable 5m prices: 1,232.
- Usable best <=120m prices: 1,231.
- Ticker/team mapping mismatches: 0.
- Wide spreads: 0.
- Low-liquidity rows: 185.

Interpretation:

Market matching and historical pregame price coverage are no longer the main blocker. The main blocker is strategy quality under tradable bid/ask prices.

## Proof Gates

Source file: `data/reports/single_game_proof_summary.json`

Current proof status:

- `status`: `not_proven`
- `single_game_edge_proven`: `false`
- Hard failures: 5.
- Warning failures: 0.

Failed hard gates:

- `strategy_backtest_profit`
- `average_clv`
- `positive_clv_rate`
- `calibrated_strategy_readiness`
- `repeatability_months`

Readiness:

- `data/reports/strategy_readiness_summary.json`
- Paper-trade candidates: 0.
- Parlay-ready strategies: 0.

Interpretation:

The system must continue to block recommendations. No agent should bypass these gates.

## Fair-Price and Parlay Status

Fair-price source: `data/reports/fair_price_summary.json`

- Rows: 1,224.
- Actionable bets: 0.
- Ungated research bets: 540.
- `single_game_edge_proven`: `false`.

Parlay source: `data/reports/parlay_recommendations_summary.json`

- `status`: `blocked_single_game_edge_not_proven`.
- `parlay_recommendations_allowed`: `false`.
- Eligible single-game legs: 0.
- Parlays: 0.

Interpretation:

The fair-price engine and parlay builder are correctly blocked by proof state. This is intentional.

## Side-Specific Shrinkage Research

Source files:

- `data/reports/side_specific_shrinkage_summary.json`
- `data/reports/side_specific_shrinkage_sweep.csv`
- `data/reports/side_specific_shrinkage_walk_forward.csv`
- `data/reports/side_specific_shrinkage_recommendations.md`

Current status:

- `status`: `watchlist_found`.
- Policies tested: 2,100.
- Candidate policies: 0.
- Watchlist policies: 276.
- Research-only: true.

Best policy:

- Policy: `yes_0.10|no_0.75|edge_0.03|none|prior_20`.
- Trades: 363.
- Profit: -$9.17.
- Average CLV: +0.163 cents.
- Positive CLV rate: 25.3%.
- YES profit: +$7.08.
- NO profit: -$16.25.
- Repeatability: 2/7 profitable months; 5/7 positive-CLV months.
- Overfit risk: high.
- Final status: `watchlist`.

Interpretation:

Shrinkage helped mostly by suppressing YES exposure. It improved loss size and average CLV but did not create a proven strategy. Profit remained negative, positive CLV frequency stayed low, and repeatability was not strong enough.

## NO Settlement Calibration Audit

Source files:

- `data/reports/no_settlement_calibration_summary.json`
- `data/reports/no_settlement_calibration_by_bucket.csv`
- `data/reports/no_clv_vs_profit.csv`
- `data/reports/no_settlement_failure_segments.csv`
- `data/reports/no_suppression_rule_sweep.csv`
- `data/reports/no_suppression_walk_forward.csv`
- `data/reports/no_settlement_recommendations.md`

Current status:

- `status`: `watchlist`.
- NO rows: 309.
- Average predicted NO probability: 41.5%.
- Actual NO win rate: 32.0%.
- Average break-even probability: 31.0%.
- Calibration error: -9.4%.
- NO overconfident: true.
- NO profit: +$10.29.
- Average NO CLV: +0.1197 cents.
- Positive NO CLV rate: 25.6%.
- Positive-CLV NO rows: 79.
- Positive-CLV NO profit: +$7.84.
- Positive-CLV NO win rate: 32.9%.
- Expensive misses: 42.
- Expensive miss profit: -$160.99.

Best suppression rule:

- Rule: `price_bucket+liquidity_bucket|rows>=10|profit>=-0.020|clv>=0.00|pos_clv>=0.20`.
- Trades: 65.
- Profit: +$13.23.
- ROI on amount risked: +5.8%.
- Average CLV: +0.081 cents.
- Positive CLV rate: 30.8%.
- Max drawdown: -22.9%.
- Repeatability: 2/5 profitable months; 4/5 positive-CLV months.
- Overfit risk: high.
- Final status: `watchlist`.

Interpretation:

NO CLV has some signal, but it does not fully translate into reliable settlement profit. The model is overconfident on NO: actual NO win rate trails predicted NO probability by about 9.4 percentage points. A few expensive misses can erase many small CLV gains. Suppression rules can improve results but are not validated candidates yet.

## Single-Game Edge Diagnostics

Source files:

- `data/reports/single_game_edge_diagnostics_summary.json`
- `data/reports/single_game_edge_diagnostics.csv`
- `data/reports/single_game_edge_failure_segments.csv`
- `data/reports/single_game_edge_walk_forward_slices.csv`
- `data/reports/single_game_edge_recommendations.md`

Current status:

- `status`: `not_proven`.
- Trades: 634.
- Profit: -$71.14.
- ROI on amount risked: -2.90%.
- Average CLV: -0.0298 cents.
- Positive CLV rate: 25.5%.
- Calibration error: -10.0%.
- Walk-forward validated slices: 0.
- Actionable fair-price bets: 0.
- Parlay status: `blocked_single_game_edge_not_proven`.

Side breakdown:

- YES: 325 trades, -$81.43 profit, -0.1625c average CLV, 23.4% positive CLV.
- NO: 309 trades, +$10.29 profit, +0.1197c average CLV, 27.8% positive CLV.

Interpretation:

YES remains the main canonical loss driver. NO is the only side showing positive average CLV and profit in the raw canonical backtest, but NO settlement calibration is not strong enough to prove edge.

## Current Best Explanation of Failure

The project is no longer primarily failing because of missing Kalshi price data or bad market matching. It is failing because:

1. YES model-market disagreements are too aggressive and lose CLV.
2. NO has modest positive CLV but poor settlement calibration.
3. Positive CLV frequency is far below 50%.
4. Strategy profit is unstable by month and sensitive to expensive misses.
5. Suppression/shrinkage rules improve some metrics but remain watchlist-only.
6. No rule currently clears enough trades, profit, CLV, drawdown, and repeatability gates together.

## Current Report Inventory to Give Another Agent

The next agent should read these first:

1. `README.md`
2. `TODO.md`
3. `CLEANUP_REPORT.md`
4. `IMPLEMENTATION_REPORT_2026-06-08.md`
5. `PROJECT_STATE_REPORT_2026-06-08.md`
6. `PROJECT_STATE_REPORT_2026-06-09.md`
7. `data/reports/backtest_summary.json`
8. `data/reports/single_game_proof_summary.json`
9. `data/reports/fair_price_summary.json`
10. `data/reports/parlay_recommendations_summary.json`
11. `data/reports/side_specific_shrinkage_summary.json`
12. `data/reports/no_settlement_calibration_summary.json`
13. `data/reports/single_game_edge_diagnostics_summary.json`

## Validation State

Latest full unit test command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Latest result:

- 234 tests.
- OK.

Latest cached pipeline:

```powershell
.\run_cached_pipeline.bat
```

Latest result:

- 59 steps completed.
- Dashboard rebuilt.
- Canonical Kalshi bid/ask metadata remained valid.
- Proof/fair-price/parlay blocks remained intact.

## Next Best Task

Build a NO probability recalibration experiment that shrinks or caps NO probabilities by prior-period settlement calibration buckets, then compare it against the current NO suppression watchlist using strict walk-forward logic. Do not change proof gates, enable fair-price bets, or enable parlays.

