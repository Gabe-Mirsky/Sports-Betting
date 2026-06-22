# Implementation Report

Date: 2026-06-08  
Project: `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor`

## Summary

This rework focused on reliability, source safety, Kalshi market truth, and keeping proof gates strict. It did not loosen betting gates, did not create live-looking recommendations, and did not optimize parlays.

The canonical single-game pipeline now explicitly runs the backtest in Kalshi mode and refuses to let sportsbook or mock/manual data silently write unsuffixed canonical backtest files.

## Files Changed

- `scripts/run_single_game_research_pipeline.py`
  - Added explicit `--market-source kalshi` to the canonical bid/ask backtest step.

- `scripts/run_backtest.py`
  - Added source resolution and stricter mode handling.
  - `--market-source kalshi` now requires matched Kalshi markets and pregame candle prices.
  - Unsuffixed canonical outputs are refused unless the resolved source is Kalshi.
  - Sportsbook outputs remain suffixed with `_sportsbook`.
  - Mock/manual outputs now default to `_mock`.
  - Backtest summaries now include:
    - `market_source`
    - `requested_market_source`
    - `canonical_kalshi_backtest`
    - `price_source`
    - `snapshot_target`
    - `bid_ask_required`
    - `no_trades_allowed`
    - `stale_artifacts_detected`
    - `artifact_warnings`
    - `artifact_inputs`
    - `generated_at_utc`
  - Backtest trade rows now include source columns:
    - `market_source`
    - `price_source`
    - `canonical_kalshi_backtest`
    - `snapshot_target_order`

- `src/strategy/backtest.py`
  - Added explicit `mode: kalshi_candlestick` diagnostics for candlestick backtest preparation.

- `scripts/build_fair_prices.py`
  - Added canonical backtest source validation before building fair-price signals.
  - Requires Kalshi source, bid/ask Kalshi candle pricing, canonical marker, and no stale artifact flag.
  - Fair-price summary now records validated backtest source metadata.

- `scripts/build_parlay_recommendations.py`
  - Added fair-price source validation before building parlay candidates.
  - Requires fair-price rows to come from validated Kalshi bid/ask backtest metadata.

- `src/data/kalshi_candles.py`
  - Added batch candle download path using `get_batch_market_candlesticks` where possible.
  - Preserved per-ticker fallback.
  - Preserved existing cache compatibility.
  - Preserved 1-minute, 60-minute, daily fallback behavior.
  - Preserved bid/ask price quality labeling.

- `src/reports/dashboard.py`
  - Updated backtest copy to describe canonical Kalshi bid/ask backtest results.
  - Added dashboard source/freshness/proof status block.
  - Dashboard now surfaces backtest source, price source, snapshot target, bid/ask requirement, NO trade setting, proof status, fair-price bet count, and parlay status.

- `tests/test_pipeline_source.py`
  - Added regression tests for explicit Kalshi market-source selection.
  - Added fair-price source-validation tests.
  - Added parlay source-validation test.

- `tests/test_kalshi_candles.py`
  - Added batch candle orchestration test verifying batch success avoids per-ticker calls.

## Validation

Full unit suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Result:

- 214 tests run.
- OK.

Cached end-to-end pipeline:

```powershell
.\run_cached_pipeline.bat
```

Result:

- Completed all 56 single-game research steps.
- Rebuilt `data/reports/dashboard.html`.
- Canonical backtest regenerated as Kalshi bid/ask mode.

## Current Canonical Backtest Metadata

From `data/reports/backtest_summary.json`:

- `market_source`: `kalshi`
- `requested_market_source`: `kalshi`
- `canonical_kalshi_backtest`: `true`
- `price_source`: `kalshi_candlesticks_bid_ask`
- `snapshot_target`: `pregame_60m,pregame_30m,pregame_5m`
- `bid_ask_required`: `true`
- `no_trades_allowed`: `true`
- `stale_artifacts_detected`: `false`
- Trades: 634
- Ending bankroll: $28.86 from $100.00
- Average CLV: -0.03 cents
- Positive CLV rate: 24.3%

`data/reports/backtest_trades.csv` now has `market_source=kalshi` on all 1,224 rows.

## Current Proof and Recommendation Status

Single-game proof:

- `status`: `not_proven`
- `single_game_edge_proven`: `false`

Fair-price:

- Rows: 1,224
- Actionable bets: 0
- Ungated research bets: 540
- Final recommendation state: all `No bet`

Parlays:

- `status`: `blocked_single_game_edge_not_proven`
- Eligible single-game legs: 0
- Parlays: 0
- `parlay_recommendations_allowed`: `false`

These blocked states are correct and intentional.

## Current Failure Drivers

The current system still does not prove repeatable single-game edge.

Top observed failure drivers:

- Overall canonical backtest loses money: ending bankroll $28.86 from $100.00.
- Average CLV remains slightly negative at -0.03 cents.
- Positive CLV rate remains too low at about 25%.
- Best price-aware calibrated signals are still only `watchlist`, not proven.
- `edge_failure_summary.json` status remains `not_proven`.
- Worst segment source is `by_side_edge`, especially YES signals in the 8-12% edge bucket:
  - 28 rows
  - average CLV -0.61 cents
  - positive CLV rate 7.1%
  - average profit/share -0.030
- YES calibrated signals have negative average CLV:
  - YES average CLV -0.16 cents
  - YES positive CLV rate 24.6%
- NO calibrated signals have small positive CLV but poor settlement calibration:
  - NO average CLV +0.26 cents
  - NO positive CLV rate 29.4%
  - NO average profit/share -0.006
  - NO forecast win rate 31.4%
  - NO actual win rate 23.6%
  - NO calibration error -7.8 percentage points
- Low-liquidity is still present in market truth audit:
  - 185 low-liquidity rows.
- Availability input remains incomplete:
  - `availability_gap_summary.json` status `needs_availability_input`
  - 240 missing statuses
  - 121 high-impact missing statuses
  - 12 games with missing statuses

## Remaining Blockers Before Parlays

Do not optimize parlays yet. Remaining blockers:

- Positive CLV is not broad or repeatable.
- Backtest is negative after bid/ask pricing.
- Proof gates still fail.
- NO-side calibration is overconfident versus realized outcomes.
- YES edge buckets contain major CLV failures.
- Availability/injury input is missing and should be improved only with leak-safe pregame data.
- Batch candle orchestration was added, but historical batch efficiency should be monitored during real refreshes.

The next useful work is still single-game research: fix calibration/CLV failure slices and availability inputs, then validate by Kalshi bid/ask backtest and month-by-month repeatability.
