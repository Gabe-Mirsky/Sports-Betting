# NBA Kalshi Single-Game Edge Research

> **Web UI:** Matchup predictions and the parlay creator now live in a Django site
> (`predictions/` app). It replaces the old static HTML dashboard. See
> [WEBAPP.md](WEBAPP.md) — quick start: `python manage.py migrate` then
> `python manage.py import_predictions` then `python manage.py runserver`.

This is a local research project for NBA prediction-market analysis. It does not place real trades. The current goal is not to find optimal parlays. The goal is to prove that the model has real, repeatable edge on single-game Kalshi-style markets.

Parlays are intentionally deferred until straight single-game bets pass the evidence gates:

- at least 300 historical matched markets
- positive closing-line value
- positive backtest after spread and realistic tradable prices
- good calibration in high-edge buckets
- no single season driving all profit
- no one team or price range driving all profit

## Main Command

Run the cached single-game research path from the project folder in VS Code:

```powershell
.\run_cached_pipeline.bat
```

That wrapper defaults to cached market data and cached candles, finds a runnable Python, and works around a broken local `.venv` executable by reusing the installed `.venv` packages with the bundled Python.

To refresh public Kalshi markets and candles too:

```powershell
.\run_cached_pipeline.bat -RefreshMarkets -RefreshCandles
```

To refresh NBA data too:

```powershell
.\run_cached_pipeline.bat -Download
```

The direct Python command is still available when your active Python environment is healthy:

```powershell
python scripts/run_single_game_research_pipeline.py --skip-market-pull --skip-candles --kalshi-start-date 2023-10-01 --kalshi-end-date 2026-05-11
```

This command runs the boring repeatable path:

1. refresh NBA data when `--download` is set
2. build leak-safe features
3. train home-win models
4. run walk-forward evaluation
5. tune calibrated home-win models
6. train margin and total research models
7. build the home-win ensemble audit
8. backfill raw public Kalshi Sports/NBA series, events, and markets
9. pull and cache Kalshi NBA markets
10. match markets to games
11. download candles and extract pregame prices
12. build the market truth audit
13. build the Kalshi coverage audit
14. run a bid/ask-based fake-bankroll backtest
15. calibrate edge bins
16. calibrate side/price/edge bins
17. sweep price-aware calibration settings
18. build the best price-aware calibration artifact
19. audit calibrated residuals
20. audit best price-aware residuals
21. audit market movement attribution
22. audit best price-aware market movement
23. diagnose edge failure drivers
24. sweep side-suppression research policies
25. audit NO-only market regimes
26. audit NO probability calibration against CLV
27. sweep NO calibration guardrails
28. sweep corrected CLV rules
29. sweep corrected best price-aware CLV rules
30. sweep residual guardrails
31. analyze closing-line value
32. build side-specific CLV-filtered strategy
33. sweep CLV price/month stability rules
34. walk-forward validate CLV price/month rules
35. analyze CLV decay drivers
36. build defensive CLV-filtered strategy
37. sweep defensive rule thresholds
38. walk-forward validate defensive rules
39. test defensive sample expansion
40. audit defensive failure month
41. optimize a conservative calibrated single-bet slate
42. optimize a CLV-filtered single-bet slate
43. optimize a defensive CLV-filtered single-bet slate
44. score single-game strategy readiness
45. build single-game proof gates
46. build fair-price single-game signals
47. build the dashboard

Outputs are written under `data/reports/`, especially:

- `market_truth_audit.csv`
- `market_truth_audit_summary.json`
- `data/raw/kalshi/public_api/sports_series.csv`
- `data/raw/kalshi/public_api/nba_series.csv`
- `data/raw/kalshi/public_api/nba_events.csv`
- `data/raw/kalshi/public_api/nba_markets.csv`
- `data/processed/kalshi_public_possible_nba_markets.csv`
- `kalshi_coverage_summary.json`
- `backtest_trades.csv`
- `backtest_summary.json`
- `edge_calibration_bins.csv`
- `edge_calibrated_price_aware_trades.csv`
- `edge_calibration_price_aware_bins.csv`
- `edge_calibration_price_aware_summary.json`
- `price_aware_calibration_sweep.csv`
- `price_aware_calibration_sweep_summary.json`
- `edge_calibration_price_aware_best_trades.csv`
- `edge_calibration_price_aware_best_summary.json`
- `residual_summary.json`
- `residual_price_aware_best_summary.json`
- `residual_by_side_calibrated_residual.csv`
- `market_movement_summary.json`
- `market_movement_price_aware_best_summary.json`
- `market_movement_by_side_move.csv`
- `edge_failure_summary.json`
- `edge_failure_worst_segments.csv`
- `edge_failure_by_side_price.csv`
- `side_suppression_summary.json`
- `side_suppression_descriptive.csv`
- `side_suppression_walk_forward_folds.csv`
- `no_regime_summary.json`
- `no_regime_by_entry_price_bucket.csv`
- `no_regime_by_liquidity_bucket.csv`
- `no_regime_by_edge_bucket.csv`
- `no_calibration_summary.json`
- `no_calibration_by_forecast_win_bucket.csv`
- `no_calibration_by_entry_price_bucket.csv`
- `no_calibration_by_month_price.csv`
- `no_calibration_guardrail_summary.json`
- `no_calibration_guardrail_descriptive.csv`
- `no_calibration_guardrail_walk_forward_folds.csv`
- `corrected_clv_summary.json`
- `corrected_clv_price_aware_best_summary.json`
- `residual_guardrail_summary.json`
- `clv_summary.json`
- `clv_by_edge_bucket.csv`
- `clv_by_price_bucket.csv`
- `clv_by_team.csv`
- `clv_by_side.csv`
- `clv_by_season.csv`
- `clv_by_liquidity.csv`
- `clv_filtered_trades.csv`
- `clv_filtered_side_audit.csv`
- `clv_filtered_summary.json`
- `clv_price_month_sweep.csv`
- `clv_price_month_sweep_monthly.csv`
- `clv_price_month_sweep_summary.json`
- `clv_price_month_walk_forward_trades.csv`
- `clv_price_month_walk_forward_folds.csv`
- `clv_price_month_walk_forward_monthly.csv`
- `clv_price_month_walk_forward_summary.json`
- `clv_decay_summary.json`
- `clv_decay_monthly.csv`
- `clv_decay_decay_drivers.csv`
- `clv_decay_negative_clv_rows.csv`
- `defensive_filtered_trades.csv`
- `defensive_filter_audit.csv`
- `defensive_filter_summary.json`
- `defensive_rule_sweep.csv`
- `defensive_walk_forward_trades.csv`
- `defensive_walk_forward_summary.json`
- `defensive_sample_expansion.csv`
- `defensive_sample_expansion_summary.json`
- `defensive_failure_summary.json`
- `defensive_failure_monthly.csv`
- `defensive_failure_failure_month_rows.csv`
- `fair_price_signals.csv`
- `fair_price_summary.json`
- `strategy_readiness.csv`
- `single_game_proof_gates.csv`
- `single_game_proof_summary.json`
- `dashboard.html`

## Manual Command Path

Use this when you want to run or debug one stage at a time:

```powershell
python scripts/download_nba_data.py --start-season 2018 --end-season 2025
python scripts/download_nba_player_data.py --start-season 2018 --end-season 2025
python scripts/build_features.py
python scripts/train.py
python scripts/walk_forward.py
python scripts/tune_model.py
python scripts/train_market_type_models.py
python scripts/build_home_win_ensemble.py
python scripts/kalshi_backfill_public_sports.py --series-ticker KXNBAGAME --event-status all --market-status all --sleep-seconds 0.5 --timeout 20
python scripts/kalshi_backfill_markets.py --start-date 2023-10-01 --end-date 2026-05-08
python scripts/kalshi_match_games.py
python scripts/kalshi_download_candles.py --fetch-game-times
python scripts/market_truth_audit.py
python scripts/kalshi_coverage_report.py
python scripts/run_backtest.py --predictions-path data/reports/walk_forward_predictions.csv --bankroll 100 --edge-threshold 0.05
python scripts/calibrate_edges.py
python scripts/calibrate_edges_price_aware.py
python scripts/sweep_price_aware_calibration.py
python scripts/build_best_price_aware_calibration.py
python scripts/audit_residuals.py
python scripts/audit_residuals.py --input-path data/reports/edge_calibration_price_aware_best_trades.csv --prefix residual_price_aware_best
python scripts/audit_market_movement.py
python scripts/audit_market_movement.py --input-path data/reports/edge_calibration_price_aware_best_trades.csv --prefix market_movement_price_aware_best
python scripts/diagnose_edge_failures.py
python scripts/sweep_side_suppression.py
python scripts/audit_no_regimes.py
python scripts/audit_no_calibration.py
python scripts/sweep_no_calibration_guardrails.py
python scripts/sweep_corrected_clv_rules.py
python scripts/sweep_corrected_clv_rules.py --input-path data/reports/edge_calibration_price_aware_best_trades.csv --prefix corrected_clv_price_aware_best
python scripts/sweep_residual_guardrails.py
python scripts/analyze_clv.py
python scripts/build_clv_filtered_strategy.py
python scripts/sweep_clv_price_month_rules.py
python scripts/sweep_clv_price_month_rules.py --walk-forward
python scripts/analyze_clv_decay.py
python scripts/build_defensive_strategy.py
python scripts/build_defensive_strategy.py --sweep
python scripts/build_defensive_strategy.py --walk-forward
python scripts/build_defensive_strategy.py --sample-expansion
python scripts/audit_defensive_failure_month.py
python scripts/optimize_portfolio.py --use-calibrated-edges
python scripts/optimize_portfolio.py --trades-path data/reports/clv_filtered_trades.csv --trade-column clv_filtered_trade --expected-roi-column calibrated_expected_roi --min-edge -1.0 --output-summary-path data/reports/portfolio_summary_clv_filtered.json
python scripts/optimize_portfolio.py --trades-path data/reports/defensive_filtered_trades.csv --trade-column defensive_trade --expected-roi-column calibrated_expected_roi --min-edge -1.0 --output-summary-path data/reports/portfolio_summary_defensive.json
python scripts/strategy_readiness.py
python scripts/single_game_proof.py
python scripts/build_fair_prices.py
python scripts/build_dashboard.py
```

## Market Truth Audit

The market truth audit is the main pre-backtest quality gate. It creates one row per matched game-market pair with:

- `game_id`
- `date`
- `home_team`
- `away_team`
- `market_ticker`
- `series_ticker`
- `tipoff_time`
- `market_close_time`
- `pregame_price_60m`
- `pregame_price_30m`
- `pregame_price_5m`
- `yes_bid`
- `yes_ask`
- `mid_price`
- `spread`
- `volume`
- `open_interest`
- `match_status`

It also flags ticker/team mapping mismatches, wide spreads, and low-liquidity rows. Bad market matching or bad historical prices can fake edge, so this report matters more than adding model complexity.

## Pricing Rule

Backtests should use realistic tradable bid/ask prices. Do not treat last price alone as a tradable entry. For YES bets, the entry is normally the YES ask. For NO bets, the program should use the tradable NO-side ask or a defensible ask-equivalent.

The current backtest path filters candle prices to bid/ask-available rows by default through `config.yaml`:

```yaml
strategy:
  allow_no_trades: true

backtest:
  allowed_price_qualities: "bid_ask_available"
  require_bid_ask: true
  max_bid_ask_spread_cents: 10.0
```

When `allow_no_trades` is true, the signal engine evaluates both sides:

- buy YES at `yes_ask`
- buy NO at `100 - yes_bid`

It takes only the side with enough edge after the configured threshold and price filters.

## Evaluation Priorities

The model is useful only if its calibrated probability is meaningfully better than Kalshi's tradable price after spread, fees, uncertainty, and liquidity screens.

Track these before adding more model types:

- log loss
- Brier score
- calibration curve
- edge-bin realized win rate
- profit after spread and realistic pricing
- closing-line value
- season-by-season stability
- team, price, and liquidity concentration

CLV reports are generated from `backtest_trades.csv`:

```powershell
python scripts/analyze_clv.py
```

This writes average and median CLV overall, by edge bucket, by price bucket, by side, by team, by season, and by liquidity bucket.

The CLV-filtered strategy is generated with side-specific expanding history:

```powershell
python scripts/build_clv_filtered_strategy.py
```

YES and NO are gated separately. The default NO gate is stricter because the current raw NO path is the largest loss source.

The price/month stability sweep checks whether the CLV-filtered YES-only strategy is concentrated in one price range or a few months:

```powershell
python scripts/sweep_clv_price_month_rules.py
```

This is descriptive research only. A narrow price rule still needs walk-forward proof before it can be trusted.

The nested walk-forward version chooses a price rule from prior months and applies it to the next month:

```powershell
python scripts/sweep_clv_price_month_rules.py --walk-forward
```

Use the walk-forward result as the trust gate, not the in-sample sweep.

Price-aware calibration is generated separately from the baseline edge-bin calibration:

```powershell
python scripts/calibrate_edges_price_aware.py
python scripts/sweep_price_aware_calibration.py
```

This tests whether side + price + edge history reduces false cheap-contract edges. The current best in-sample price-aware setting improves profit slightly but still fails CLV:

- best sweep status: `watchlist`
- signals: `300`
- average profit/share: `+0.014`
- average CLV: `+0.01c`
- positive CLV rate: `25.7%`

Residual and market-movement audits explain whether calibrated edges are real price-discovery signals:

```powershell
python scripts/audit_residuals.py
python scripts/audit_market_movement.py
python scripts/diagnose_edge_failures.py
python scripts/sweep_corrected_clv_rules.py
python scripts/sweep_residual_guardrails.py
```

Current diagnosis: baseline edge-bin calibration overstates cheap-contract win probability, and price-aware calibration does not yet produce repeatable positive CLV. Treat any profitable slice as research-only until positive CLV and walk-forward proof improve.

`diagnose_edge_failures.py` writes `edge_failure_worst_segments.csv`, which ranks side, price, edge, ROI, liquidity, and month slices by CLV/profit failure. Use it to pick model hypotheses; do not convert those slices directly into betting rules without walk-forward CLV proof.

`sweep_side_suppression.py` tests whether YES suppression or NO-only selection helps. Current cached result: `no_only` is the descriptive best policy, but the nested walk-forward result remains `not_ready` with weak positive CLV rate, so it is research-only.

`audit_no_regimes.py` audits calibrated NO-only signals by NO entry price, implied YES market price, spread, liquidity, edge, ROI, and month. Current cached result: spreads are not the issue; NO has slightly positive average CLV but low positive-CLV frequency across most regimes.

CLV decay diagnostics explain whether later months degrade because of price bucket, edge bucket, liquidity, or team mix:

```powershell
python scripts/analyze_clv_decay.py
```

The defensive strategy blocks non-team decay-prone slices before portfolio selection:

```powershell
python scripts/build_defensive_strategy.py
```

Default defensive rules block very cheap `0-10c` YES contracts, extreme calibrated ROI `3+`, and very high-volume rows `>=1000`. Team exclusions are intentionally avoided unless a separate walk-forward test supports them.

The defensive sweep now tests both lower and upper price bounds plus lower and upper calibrated ROI bounds. This was added after the March audit showed weakness in low-ROI and mid-price slices:

```powershell
python scripts/build_defensive_strategy.py --sweep
```

Current best in-sample rule family is `price=15-40c, roi=0.5-3.0, volume<=10000`. Treat that as a hypothesis until it survives walk-forward checks.

Defensive rules must also be walk-forward validated:

```powershell
python scripts/build_defensive_strategy.py --walk-forward
```

This chooses thresholds from prior months and applies them to future months.

The current expanded walk-forward validation improves March but still stays `not_ready` because the stricter rule set leaves fewer than 100 validated trades:

- signals: `73`
- positive CLV rate: `61.6%`
- positive month share: `100.0%`
- average CLV: `21.86c`
- average profit/share: `+0.142`

Sample expansion tests nearby broader thresholds:

```powershell
python scripts/build_defensive_strategy.py --sample-expansion
```

Current sample expansion result remains `not_ready`. A looser `price=15-40c, roi=0.25-3.0, volume<=10000` rule reaches `107` signals, but March positive CLV falls below 50%, so it is rejected. The best rule that preserves every evaluated month above 50% positive CLV is still undersized:

- rule: `price=15-40c, roi=0.5-3.0, volume<=10000`
- signals: `72`
- positive CLV rate: `62.5%`
- weakest monthly positive CLV: `51.5%`
- average CLV: `22.17c`
- average profit/share: `+0.146`

If a walk-forward month fails, audit it before adding rules:

```powershell
python scripts/audit_defensive_failure_month.py --failure-month 2026-03
```

Schedule context is included only when those columns are present or supplied through `--schedule-context-path`.

## Fair-Price Engine

Fair-price signals are generated from `matched_markets.csv`:

```powershell
python scripts/build_fair_prices.py
```

For each game-market row, the fair-price engine evaluates both sides:

- `model_prob`
- `calibrated_prob`
- `market_yes_ask`
- `market_no_ask`
- `fair_yes_price`
- `fair_no_price`
- `gross_edge`
- `fee_adjusted_edge`
- `spread_penalty`
- `uncertainty_penalty`
- `final_edge`
- `recommendation`

The output is explicit about "Bet YES", "Bet NO", and "No bet". It also keeps parlays blocked until single-game edge is proven.

By default, `build_fair_prices.py` reads `single_game_proof_summary.json` and blocks action-looking recommendations unless `single_game_edge_proven` is true. When proof is not proven, rows keep `ungated_side`, `ungated_recommendation`, and `ungated_main_reason` for research, but the actionable `side` is blank, `recommendation` is `No bet`, and `max_size` is `0`.

Current cached run:

- proof gate status: `not_proven`
- actionable fair-price bets: `0`
- ungated research bets: `539`

## Single-Game Proof Gates

The proof report is generated from the saved audit, backtest, CLV, calibration, and readiness artifacts:

```powershell
python scripts/single_game_proof.py
```

It blocks parlay research unless the straight-bet system passes hard gates for market coverage, usable pregame prices, market matching quality, raw bid/ask backtest profit, positive CLV, calibrated readiness, enough months, and concentration risk.

## Daily Output Goal

The final user-facing interface should answer:

- what single bets should be placed today
- what games should be avoided
- which bets are ineligible for parlays
- why each recommendation exists

Most games should be `no bet`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the static dashboard:

```powershell
python scripts/build_dashboard.py
```

Then open:

```text
data/reports/dashboard.html
```

## Caveats

- Old Kalshi markets can be unavailable or incomplete.
- Not every NBA game has a Kalshi market.
- Automated backtests should use only high-confidence `auto_matched` rows.
- Daily candles are low quality for pregame pricing.
- This project is paper trading and research only.
