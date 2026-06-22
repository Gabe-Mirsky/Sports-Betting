# Implementation Report

Date: 2026-06-09  
Project: `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor`

## Summary

This report updates the implementation state after the side-specific shrinkage research and NO settlement calibration audit were added to the project.

The project remains research-only. No proof gates were loosened, no fair-price bets were enabled, no parlays were enabled, and no live trading path was added.

Current conclusion:

- Side-specific shrinkage reduced losses mainly by suppressing YES exposure.
- NO-side CLV has some signal but NO settlement calibration is overconfident.
- The best NO suppression rule is watchlist-only, not a deployable strategy.
- Single-game edge remains not proven.

## New Research Modules

### `src/strategy/probability_shrinkage.py`

Added reusable probability shrinkage primitives:

- Shrink model probability toward market probability.
- Apply side-specific shrink factors.
- Recalculate adjusted edge after shrinkage and uncertainty penalty.
- Clamp adjusted probabilities to `[0, 1]`.

Purpose:

Test whether model-market disagreement was too aggressive, especially on YES trades.

### `src/strategy/uncertainty_penalty.py`

Added prior-period uncertainty penalty helpers:

- `none`
- `side-only`
- `side+price_bucket`
- `side+price_bucket+edge_bucket`
- `side+price_bucket+liquidity_bucket`

Small-sample behavior:

- Buckets without enough prior samples receive a conservative default penalty.
- Small buckets cannot create aggressive signals.

Purpose:

Prevent future leakage and avoid promoting tiny historical slices.

### `src/strategy/shrinkage_policy_sweep.py`

Added the research-only side-specific shrinkage sweep:

- YES shrink factors: `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, `0.75`, `1.00`.
- NO shrink factors: `0.25`, `0.40`, `0.50`, `0.75`, `1.00`.
- Minimum edge thresholds: `0.03`, `0.05`, `0.07`, `0.10`.
- Uncertainty modes listed above.
- Minimum prior sample sizes: `20`, `50`, `100`.

The sweep uses prior-period-only penalty calculations and writes research-only results.

### `src/strategy/no_settlement_calibration.py`

Added the NO settlement calibration audit:

- Normalizes canonical NO trade rows.
- Computes break-even probability from NO buy price.
- Compares predicted NO probability to actual NO settlement win rate.
- Buckets NO performance by probability, price, CLV, edge, liquidity, month, and team.
- Measures whether positive NO CLV translates into realized profit.
- Identifies expensive misses.
- Runs prior-period-only NO suppression rule sweeps.

Purpose:

Answer whether NO-side CLV is real settlement edge or just noisy market movement with poor calibration.

## New Scripts

### `scripts/sweep_side_specific_shrinkage.py`

Reads canonical `matched_markets.csv` and validates `backtest_summary.json` before running.

Requires:

- `market_source = kalshi`
- `price_source = kalshi_candlesticks_bid_ask`
- `canonical_kalshi_backtest = true`
- `stale_artifacts_detected = false`

Outputs:

- `data/reports/side_specific_shrinkage_sweep.csv`
- `data/reports/side_specific_shrinkage_summary.json`
- `data/reports/side_specific_shrinkage_walk_forward.csv`
- `data/reports/side_specific_shrinkage_recommendations.md`

### `scripts/audit_no_settlement_calibration.py`

Reads canonical `backtest_trades.csv` and validates `backtest_summary.json`.

Outputs:

- `data/reports/no_settlement_calibration_summary.json`
- `data/reports/no_settlement_calibration_by_bucket.csv`
- `data/reports/no_clv_vs_profit.csv`
- `data/reports/no_settlement_failure_segments.csv`
- `data/reports/no_suppression_rule_sweep.csv`
- `data/reports/no_suppression_walk_forward.csv`
- `data/reports/no_settlement_recommendations.md`

## Pipeline Integration

Updated `scripts/run_single_game_research_pipeline.py`.

The cached pipeline now runs 59 steps and includes:

- `Sweep side-specific probability shrinkage`
- `Audit NO settlement calibration`

These steps run after proof/fair-price/parlay artifacts exist, so their summaries can verify that gates remain blocked.

## Config Updates

Updated:

- `config.yaml`
- `src/config.py`

Added disabled/research-only shrinkage config:

```yaml
strategy:
  use_side_specific_shrinkage: false

side_specific_shrinkage:
  enabled_for_research: true
  yes_shrink_factor: 0.5
  no_shrink_factor: 0.75
  uncertainty_penalty_mode: "side-only"
  min_prior_samples: 50
```

Important:

- `use_side_specific_shrinkage` is false.
- The fair-price engine does not use the shrinkage policy for recommendations.
- The shrinkage section is documentation/config scaffolding for research only.

## Tests Added

### `tests/test_side_specific_shrinkage.py`

Covers:

- Shrinkage formula.
- YES and NO shrink factors separately.
- Probability clamping to `[0, 1]`.
- Edge recalculation after shrinkage.
- Uncertainty penalty reducing edge.
- Small-sample conservative fallback.
- Prior-period-only penalty calculation.
- No future leakage in sweep mechanics.
- Proof and parlay blocks remaining false when single-game edge is not proven.

### `tests/test_no_settlement_calibration.py`

Covers:

- NO settlement logic.
- Break-even probability from NO buy price.
- Calibration bucket math.
- CLV bucket math.
- Segment summaries.
- Prior-period-only suppression.
- No future leakage in grouped suppression.
- Fair-price and parlay blocked behavior.

## Current Research Results

### Side-Specific Shrinkage

Source: `data/reports/side_specific_shrinkage_summary.json`

- Status: `watchlist_found`.
- Policies tested: 2,100.
- Candidate policies: 0.
- Watchlist policies: 276.

Best policy:

- `yes_0.10|no_0.75|edge_0.03|none|prior_20`
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

Shrinkage showed that the project was too aggressive on YES. Suppressing YES materially reduced losses, but the resulting mostly-NO strategy still did not prove repeatable edge.

### NO Settlement Calibration

Source: `data/reports/no_settlement_calibration_summary.json`

- Status: `watchlist`.
- NO rows: 309.
- Average predicted NO probability: 41.5%.
- Actual NO win rate: 32.0%.
- Calibration error: -9.4%.
- Average NO CLV: +0.120 cents.
- Positive NO CLV rate: 25.6%.
- Positive-CLV NO profit: +$7.84.
- Expensive miss profit: -$160.99.

Best suppression rule:

- `price_bucket+liquidity_bucket|rows>=10|profit>=-0.020|clv>=0.00|pos_clv>=0.20`
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

NO CLV has some signal but does not fully solve settlement risk. NO probabilities are overconfident and expensive misses can erase many small gains.

## Current Canonical Proof State

Source: `data/reports/single_game_proof_summary.json`

- `status`: `not_proven`
- `single_game_edge_proven`: `false`
- Hard failures: 5.
- Failed gates:
  - `strategy_backtest_profit`
  - `average_clv`
  - `positive_clv_rate`
  - `calibrated_strategy_readiness`
  - `repeatability_months`

Fair-price:

- Rows: 1,224.
- Bets: 0.
- Ungated research bets: 540.

Parlays:

- Status: `blocked_single_game_edge_not_proven`.
- `parlay_recommendations_allowed`: `false`.

## Validation

Latest full unit suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Result:

- 234 tests.
- OK.

Latest cached pipeline:

```powershell
.\run_cached_pipeline.bat
```

Result:

- 59 steps completed.
- Dashboard rebuilt.
- Canonical Kalshi metadata remained valid.
- Proof/fair-price/parlay blocks remained intact.

## Current Next Task

Build a NO probability recalibration experiment that shrinks or caps NO probabilities by prior-period settlement calibration buckets, then compare it against the current NO suppression watchlist using strict walk-forward logic. Keep proof gates strict and keep recommendations/parlays blocked.

