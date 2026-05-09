# NBA Kalshi Market Predictor and Paper Trading Simulator

This is a free, local research project for predicting NBA team-based Kalshi-style markets and testing a paper-trading strategy with a fake bankroll. It does not place real trades.

The project is designed so it does not need to run 24/7. Kalshi candlestick history is the main market-price source: if you miss a live pregame snapshot, the catch-up scripts can later download the market's historical candles and recover the closest available pregame price. Live snapshots are useful, but optional.

The current working version downloads and caches NBA data, builds leak-safe pre-game features, trains baseline home-win models, backfills likely Kalshi NBA markets, matches games to markets, downloads candlesticks, extracts pregame prices, and runs a fake $100 paper-trading backtest.

## Recommended Workflow

```powershell
python scripts/download_nba_data.py --start-season 2018 --end-season 2025
python scripts/download_nba_player_data.py --start-season 2018 --end-season 2025
python scripts/download_nba_player_data.py --start-season 2018 --end-season 2025 --season-type Playoffs
python scripts/build_features.py
python scripts/train.py
python scripts/walk_forward.py
python scripts/tune_model.py
python scripts/train_market_type_models.py
python scripts/build_home_win_ensemble.py
python scripts/compare_player_features.py
python scripts/kalshi_backfill_markets.py --start-date 2023-10-01 --end-date 2026-05-07
python scripts/discover_kalshi_nba_markets.py --start-date 2026-04-01 --end-date 2026-05-07
python scripts/build_kalshi_market_taxonomy.py
python scripts/kalshi_match_games.py
python scripts/kalshi_download_candles.py --fetch-game-times
python scripts/run_backtest.py --bankroll 100 --edge-threshold 0.05
python scripts/market_blend.py
python scripts/calibrate_edges.py
python scripts/optimize_portfolio.py
python scripts/optimize_portfolio.py --use-calibrated-edges
python scripts/calibrate_edges.py --trades-path data/reports/backtest_trades_market_blend.csv --output-calibrated-path data/reports/edge_calibrated_trades_market_blend.csv --output-bins-path data/reports/edge_calibration_bins_market_blend.csv --output-summary-path data/reports/edge_calibration_summary_market_blend.json --output-audit-path data/reports/edge_calibration_audit_market_blend.csv --output-negative-edge-path data/reports/edge_calibration_negative_edge_signals_market_blend.csv --output-audit-summary-path data/reports/edge_calibration_audit_summary_market_blend.json
python scripts/optimize_portfolio.py --use-calibrated-edges --trades-path data/reports/edge_calibrated_trades_market_blend.csv --output-trades-path data/reports/portfolio_trades_market_blend_calibrated.csv --output-slates-path data/reports/portfolio_slates_market_blend_calibrated.csv --output-summary-path data/reports/portfolio_summary_market_blend_calibrated.json
python scripts/build_consensus_edges.py
python scripts/screen_robust_edges.py
python scripts/analyze_signal_stability.py
python scripts/optimize_portfolio.py --use-calibrated-edges --trades-path data/reports/edge_consensus_calibrated_trades.csv --trade-column consensus_trade --expected-roi-column consensus_expected_roi --output-trades-path data/reports/portfolio_trades_consensus_calibrated.csv --output-slates-path data/reports/portfolio_slates_consensus_calibrated.csv --output-summary-path data/reports/portfolio_summary_consensus_calibrated.json
python scripts/optimize_portfolio.py --use-calibrated-edges --trades-path data/reports/edge_robust_consensus_trades.csv --trade-column robust_calibrated_trade --expected-roi-column robust_expected_roi --output-trades-path data/reports/portfolio_trades_robust_consensus.csv --output-slates-path data/reports/portfolio_slates_robust_consensus.csv --output-summary-path data/reports/portfolio_summary_robust_consensus.json
python scripts/strategy_readiness.py
python scripts/sweep_signal_rules.py
python scripts/analyze_parlay_correlations.py
python scripts/build_forward_recommendations.py
python scripts/kalshi_coverage_report.py
python scripts/review_kalshi_market_quality.py
python scripts/validate_research_data.py
python scripts/build_dashboard.py
```

To open the local website, double-click:

```text
open_website.bat
```

or run:

```powershell
python scripts/open_dashboard.py
```

Then, whenever you want to catch up after a day, week, or longer gap:

```powershell
python scripts/kalshi_catchup.py
```

Optional same-day paper suggestions:

```powershell
python scripts/predict_upcoming.py --days 1
python scripts/paper_trade_today.py
```

Important caveats:

- Old Kalshi markets may be unavailable or incomplete.
- Not every NBA game has a Kalshi market.
- Market matching is imperfect; automated backtests use only high-confidence `auto_matched` rows.
- Exact pregame price quality depends on available candles. Daily candles are marked low quality.
- This is paper trading only, not real trading.

## What It Will Do

1. Download historical NBA team game logs with `nba_api`.
2. Build pre-game team and player-rotation features without data leakage.
3. Train and tune models that estimate `P(home team wins)`.
4. Train market-type models for predicted margin and total points.
5. Backfill recent and archived Kalshi NBA markets.
6. Classify NBA markets by type: winner, spread, total, team total, player prop, series, or ambiguous.
7. Match NBA games to high-confidence Kalshi team-win markets.
8. Download Kalshi candlesticks and extract 60-minute, 30-minute, and 5-minute pregame prices.
9. Compare model probabilities to Kalshi-style binary market prices.
10. Generate paper-trade signals only when the model edge is large enough.
11. Simulate a fake $100 bankroll and save trade logs, metrics, and plots.
12. Optimize a same-day slate of individual paper bets before considering parlays.

## Install

From this folder:

```powershell
cd "C:\Users\arilo\Downloads\Prediction Market Project\nba_kalshi_predictor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation with `running scripts is disabled on this system`,
you can skip activation and call the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_nba_data.py --start-season 2018 --end-season 2025
.\.venv\Scripts\python.exe scripts\build_features.py
```

Or, for only the current PowerShell window, allow activation temporarily:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Python 3.10 or newer is recommended.

## Run Without Kalshi Credentials

Kalshi integration is optional. The project defaults to local/mock data mode.

The first command to run is:

```powershell
python scripts/download_nba_data.py --start-season 2018 --end-season 2025
```

This downloads regular-season team logs and caches them under:

```text
data/raw/
```

To download both regular-season and playoff team logs:

```powershell
python scripts/download_all_nba_data.py --start-season 2018 --end-season 2025
```

To add player-level box-score history for rotation-strength features:

```powershell
python scripts/download_nba_player_data.py --start-season 2018 --end-season 2025
python scripts/download_nba_player_data.py --start-season 2018 --end-season 2025 --season-type Playoffs
```

Player features are built from prior games only. They summarize recent rotation strength, top-player minutes, top-eight production, continuity, and active-count proxies. They do not use same-game player stats or confirmed lineups from after tipoff.

Source choice:

- `nba_api` / NBA Stats is the primary automated source for player box-score history because it is already in the project and maps official NBA.com stats endpoints.
- `pbpstats` is a good optional next source for lineup and possession-level work, but it should stay separate until the main winner/spread/total pipeline is stable.
- Basketball Reference is useful for manual research and CSV-style checks, but the project should not depend on aggressive automated scraping from it.

To add free injury or availability data, create a local CSV at `data/raw/nba/injuries/availability.csv` with:

```text
report_date,game_date,team_abbr,player_name,status,impact_weight
```

Allowed statuses include `out`, `doubtful`, `questionable`, `probable`, and `available`. `impact_weight` is optional but useful: use expected minutes, recent average minutes, or another free rotation-weight estimate so an injured starter matters more than a bench player. If you do not provide `impact_weight`, the project uses `expected_minutes`, `avg_minutes`, `rotation_minutes`, or `minutes` if one of those columns exists; otherwise it falls back to `1`.

The feature builder uses only reports dated on or before the game date, then adds home/away counts, weighted injury impact, projected minutes lost, and differences. Missing files are fine; the model simply runs without these features.

To create an availability entry template with the likely rotation players and recent-minutes impact weights:

```powershell
python scripts/build_availability_template.py --start-date 2026-01-01 --end-date 2026-01-31
```

This writes `data/raw/nba/injuries/availability_template.csv`. Fill the `status` column from a free/allowed injury source, then save the rows you want as `data/raw/nba/injuries/availability.csv`.

To rebuild the game-level dataset from cached raw files:

```powershell
python scripts/build_features.py
```

That script creates:

```text
data/interim/nba_games.parquet
data/processed/modeling_dataset.parquet
```

Train the first models:

```powershell
python scripts/train.py
```

This also saves calibration data, model probability plots, and feature diagnostics under:

```text
data/reports/
```

Run the fake $100 mock-market backtest:

```powershell
python scripts/run_backtest.py --bankroll 100 --edge-threshold 0.05
```

This also saves matched-market diagnostics and edge distribution plots.

Plain-English translation:

- `edge-threshold 0.05` means only make a paper pick when our estimate is at least 5 percentage points better than the market price. If our model says 62% and the market is 55 cents, the advantage is 7 points.
- `max-bet-fraction 0.03` means risk at most 3% of the fake bankroll on one paper pick. With a fake $100 bankroll, that is about $3.
- `model_yes_prob` means our estimated chance that the selected YES team wins.
- `market_prob` means the market-implied chance from the YES price. A 55-cent price is treated like 55%.
- `trade=True` means the simulator made a paper pick. It never places a real trade.

For a stronger out-of-sample research run, use walk-forward predictions:

```powershell
python scripts/walk_forward.py
python scripts/compare_player_features.py
```

That trains on past seasons only, predicts the next season, then repeats. For example, 2024 predictions are trained on 2018 through 2023, and 2025 predictions are trained on 2018 through 2024.
The comparison script writes `data/reports/player_feature_comparison.json`, showing whether player-aware features improved the same split versus the team-only feature set.

NBA season years use the first year of the season. The current 2026 NBA season is `2025`, meaning `2025-26`.

To refresh the current 2025-26 season and rebuild reports:

```powershell
python scripts/refresh_current_season.py
```

That command forces a fresh download for the current regular season and playoffs, rebuilds features, retrains reports, updates walk-forward predictions, reruns the mock/manual-market reports, and refreshes the dashboard files.

If you only want regular-season data:

```powershell
python scripts/refresh_current_season.py --regular-season-only
```

You can backtest against the walk-forward file with:

```powershell
python scripts/run_backtest.py --bankroll 100 --edge-threshold 0.05 --predictions-path data/reports/walk_forward_predictions.csv
```

Run unit tests:

```powershell
python -m unittest discover -s tests
```

Run the saved-artifact validation report:

```powershell
python scripts/validate_research_data.py
```

That writes `data/reports/data_validation_summary.json` and `data/reports/data_validation_issues.csv`, and the dashboard shows the current validation status.

## Add Kalshi Credentials Later

Copy `.env.example` to `.env` and fill in values only if you want to experiment with demo or live API access later:

```text
KALSHI_API_KEY_ID=
KALSHI_PRIVATE_KEY_PATH=
KALSHI_ENV=prod
```

The project should still run without this file.

If you have a local text file containing a Kalshi API key ID and RSA private key, you can set up the local ignored credential files without printing the secret:

```powershell
python scripts/setup_kalshi_credentials.py --source "$env:USERPROFILE\Downloads\Kalshi API.txt"
```

This writes `.secrets/kalshi_private_key.pem` and updates `.env`. Both are ignored by git. The project only uses authenticated GET requests for market data; it does not place orders.

If an API key or private key has ever been pasted into chat, a non-ignored file, or a screenshot, rotate it in Kalshi before treating the local setup as clean. This project ignores `.env`, `.secrets/`, PEM/key files, and `Kalshi API*.txt`.

Run the local secret hygiene check anytime:

```powershell
python scripts/security_audit.py
```

It saves `data/reports/security_audit_findings.csv` and `data/reports/security_audit_summary.json` without printing secret values.

After installing requirements, verify authenticated read access with:

```powershell
python scripts/check_kalshi_auth.py
```

## Model Design

The baseline model will predict the probability that the home team wins. The first model target will be:

```text
target_home_win
```

Current model features:

- pre-game Elo difference
- pre-game Elo home win probability
- rest-day difference
- back-to-back flags
- whether the game is regular season or playoffs
- rolling win percentage differences
- rolling point differential differences
- season-to-date win percentage and margin differences
- rolling shooting percentage differences
- rolling three-point percentage differences
- rolling free-throw percentage differences
- rolling rebound, assist, and turnover differences
- optional local injury/availability counts and differences

Same-game result columns like final score, plus/minus, and win/loss are not allowed as model inputs.

The first training script compares:

- Elo baseline
- logistic regression
- random forest
- histogram gradient boosting

The saved production artifact is whichever sklearn model has the best test log loss.

For spread and total-point markets, the project trains separate market-type engines:

```powershell
python scripts/train_market_type_models.py
```

This creates predicted home margin, predicted total points, residual standard deviations, and probability helpers for `P(home covers spread)` and `P(total goes over line)`. It also writes an early spread/total calibration report using common historical line grids until real non-winner Kalshi lines are available. Outputs are saved to `data/reports/market_type_predictions.csv`, `data/reports/market_type_model_metrics.json`, `data/reports/market_type_probability_calibration.csv`, and `data/models/margin_total_model.joblib`.

Before backtesting spread, total, team-total, or player-prop markets, audit whether the real Kalshi market text contains usable line values:

```powershell
python scripts/audit_market_type_lines.py
```

This writes `data/reports/market_line_coverage.csv` and `data/reports/market_line_coverage_summary.json`. Spread and total betting remain separate projects until this audit passes for real Kalshi lines.

To tune the home-win model without random train/test leakage:

```powershell
python scripts/tune_model.py
```

That compares feature families and logistic-regression regularization values with walk-forward season splits. It saves `data/reports/model_tuning_results.csv`, `data/reports/model_tuning_summary.json`, `data/reports/tuned_walk_forward_predictions.csv`, and `data/models/home_win_model_tuned.joblib`. A tuned probability model can have better log loss while still producing worse paper-bet thresholds, so tuned model metrics and betting results are shown separately.

To audit whether an ensemble improves the home-win model:

```powershell
python scripts/build_home_win_ensemble.py
```

This merges the standard walk-forward model, tuned walk-forward model, and margin-derived win probability. Expanding-season weights are selected from prior seasons only. It writes `data/reports/home_win_ensemble_predictions.csv`, `data/reports/home_win_ensemble_weights.csv`, `data/reports/home_win_ensemble_static_audit.csv`, and `data/reports/home_win_ensemble_summary.json`. If the ensemble does not clearly beat the best component, it stays research-only.

## Paper-Trading Simulator

The simulator will start with a fake bankroll, defaulting to `$100`.

For a Kalshi-style YES contract:

- a 52-cent price is treated as roughly 52% implied probability
- cost per share is `price_cents / 100`
- winning pays `$1` per share
- losing pays `$0`

The first strategy will only paper-trade YES contracts when:

```text
model_probability - market_probability >= edge_threshold
```

No real orders are placed.

## Portfolio Before Parlays

Before building parlays, run a constrained slate optimizer:

```powershell
python scripts/optimize_portfolio.py
```

It reads `data/reports/backtest_trades.csv`, ranks eligible individual paper bets by expected ROI, limits same-day bankroll exposure, caps same-team exposure, and avoids taking multiple markets from the same game by default. Outputs are saved to `data/reports/portfolio_trades.csv`, `data/reports/portfolio_slates.csv`, and `data/reports/portfolio_summary.json`. Trade counts are always reported with the date range they cover.

The default headline paper-trading result is the best available slate-settled portfolio summary, not the looser row-by-row backtest:

```powershell
python scripts/build_headline_backtest.py
```

This writes `data/reports/headline_backtest_summary.json` and keeps parlays blocked unless individual strategy readiness and out-of-sample pair economics both pass.

For the stricter pre-parlay version, first calibrate whether historical model-vs-market edges actually paid off:

```powershell
python scripts/calibrate_edges.py
python scripts/optimize_portfolio.py --use-calibrated-edges
```

Edge calibration uses an expanding window by prior slate date only, so a market never learns from results on its own date. The calibrated optimizer writes `data/reports/portfolio_trades_calibrated.csv`, `data/reports/portfolio_slates_calibrated.csv`, and `data/reports/portfolio_summary_calibrated.json`.

In calibrated mode, the portfolio step uses the `calibrated_trade` flag as the main filter instead of re-applying the raw 5% edge threshold. You can still force a raw-edge floor with `--min-edge` if you want to compare stricter variants.

The market-blend path repeats the same calibration on `data/reports/backtest_trades_market_blend.csv`, then writes `data/reports/portfolio_trades_market_blend_calibrated.csv`, `data/reports/portfolio_slates_market_blend_calibrated.csv`, and `data/reports/portfolio_summary_market_blend_calibrated.json`. This is the preferred pre-parlay comparison because it checks model edges against the market's own probability signal.

The consensus path is stricter: it only keeps markets where raw edge calibration and market-blend edge calibration both agree. It writes `data/reports/edge_consensus_calibrated_trades.csv`, `data/reports/edge_consensus_summary.json`, and `data/reports/portfolio_summary_consensus_calibrated.json`.

The robust consensus path is stricter again. It requires the lower confidence bound of the calibrated win rate to clear the contract cost before the signal can enter the portfolio. It writes `data/reports/edge_robust_consensus_trades.csv`, `data/reports/edge_robust_consensus_summary.json`, and `data/reports/portfolio_summary_robust_consensus.json`.

Signal stability is checked month by month with:

```powershell
python scripts/analyze_signal_stability.py
```

This writes `data/reports/signal_stability_consensus.csv` and `data/reports/signal_stability_consensus_summary.json`. A signal that makes money overall but only wins in one narrow month should not be trusted for parlays.

Finally, score the strategy families with:

```powershell
python scripts/strategy_readiness.py
```

This writes `data/reports/strategy_readiness.csv`, `data/reports/strategy_readiness_monthly.csv`, and `data/reports/strategy_readiness_summary.json`. The readiness gate is intentionally conservative: anything with unstable month-by-month profit, a losing portfolio result, or excessive drawdown stays out of parlay research.

To search for stricter paper-watch filters inside the consensus signal history:

```powershell
python scripts/sweep_signal_rules.py
```

This writes `data/reports/signal_rule_sweep.csv`, `data/reports/signal_rule_sweep_best_monthly.csv`, and `data/reports/signal_rule_sweep_summary.json`. Treat it as in-sample hypothesis generation only: a good-looking rule can move onto the watchlist, but it still does not become parlay-ready without forward evidence and correlation modeling.

Then validate rule selection month by month:

```powershell
python scripts/validate_signal_rules_walk_forward.py
```

Each test month chooses its rule from prior months only. This writes `data/reports/signal_rule_walk_forward_trades.csv`, `data/reports/signal_rule_walk_forward_folds.csv`, `data/reports/signal_rule_walk_forward_monthly.csv`, and `data/reports/signal_rule_walk_forward_summary.json`.

To measure same-slate dependency before building any parlay simulator:

```powershell
python scripts/analyze_parlay_correlations.py
```

This writes `data/reports/parlay_pair_rows.csv`, `data/reports/parlay_correlation_report.csv`, and `data/reports/parlay_correlation_summary.json`. It estimates historical two-leg signal behavior, pair win rate, independence-priced pair edge, and leg-outcome correlation. Parlays remain blocked unless individual strategy readiness and correlation diagnostics both pass.

For the website's forward-looking tab, build upcoming recommendations from saved model predictions and saved Kalshi odds:

```powershell
python scripts/predict_upcoming.py --days 14
python scripts/download_kalshi_markets.py
python scripts/build_forward_recommendations.py
```

This writes `data/reports/forward_recommendations.csv` and `data/reports/forward_recommendations_summary.json`. The table shows model odds, Kalshi-implied odds, the edge signal, whether any sweep rule is allowed by nested walk-forward validation, readiness-gated paper stake, and a hypothetical paper stake for comparison. No real trades are placed.

The included mock market file is:

```text
data/kalshi/markets_mock.csv
```

It is tiny on purpose and only exists to prove the pipeline works. Replace it with your own manually collected pre-game market snapshots for serious backtests.

A blank CSV template is available here:

```text
data/kalshi/markets_template.csv
```

You can also generate a fill-in market CSV directly from model predictions:

```powershell
python scripts/export_market_template.py --date 2024-10-22 --yes-side both --output-path data/kalshi/markets_to_fill.csv
```

Then fill in `yes_mid_cents`, or fill `yes_bid_cents` and `yes_ask_cents` so the script can calculate the midpoint.

Team fields are forgiving. The matcher accepts official abbreviations and common aliases, for example:

- `NY`, `Knicks`, `New York Knicks` -> `NYK`
- `GS`, `Warriors`, `Golden State Warriors` -> `GSW`
- `SA`, `Spurs`, `San Antonio Spurs` -> `SAS`
- `PHO`, `Suns`, `Phoenix Suns` -> `PHX`

Validate a market CSV before using it:

```powershell
python scripts/validate_market_file.py --markets-path data/kalshi/markets_to_fill.csv
```

Validation also writes a market data quality report. It warns about issues like:

- tiny sample sizes
- missing prices
- prices filled from close price instead of pre-game midpoint
- missing bid/ask spread
- unresolved settlements
- unmatched prediction rows

Print paper-trade suggestions from a manual CSV:

```powershell
python scripts/paper_trade_today.py --markets-path data/kalshi/markets_to_fill.csv --edge-threshold 0.05
```

That command only prints and saves suggestions. It never places real trades.

Compare several edge thresholds:

```powershell
python scripts/sweep_thresholds.py --thresholds 0.00,0.02,0.05,0.08,0.10,0.12,0.15
```

With a manual CSV:

```powershell
python scripts/sweep_thresholds.py --markets-path data/kalshi/markets_to_fill.csv --predictions-path data/reports/walk_forward_predictions.csv
```

Generate extra result diagnostics:

```powershell
python scripts/analyze_results.py
```

That creates probability-bucket calibration tables, season-by-season model summaries, edge-bucket backtest summaries, and a largest paper P/L table.

Audit where Kalshi prices beat the model:

```powershell
python scripts/audit_kalshi_vs_model.py
```

That creates `data/reports/kalshi_model_gap_audit.csv`, `data/reports/kalshi_model_gap_segments.csv`, `data/reports/kalshi_beat_model_examples.csv`, and `data/reports/kalshi_model_gap_summary.json`. Positive values in `kalshi_edge_over_model` mean Kalshi's pregame probability was closer to the actual result than our model.

Build the local dashboard:

```powershell
python scripts/build_dashboard.py
```

Open the generated file:

```text
data/reports/dashboard.html
```

Run the optional interactive dashboard:

```powershell
python -m pip install -r requirements-dashboard.txt
python scripts/run_dashboard.py
```

This starts a local Streamlit app that reads the files in `data/reports/`. It does not place trades or call Kalshi. The main dashboard uses automatic paper-trading defaults so you do not need to choose an advantage threshold, bet fraction, or bankroll setting.

To add the next scheduled games to the dashboard:

```powershell
python scripts/predict_upcoming.py --days 14
```

That creates `data/reports/upcoming_predictions.csv`.

To add public Kalshi NBA game market prices:

```powershell
python scripts/download_kalshi_markets.py --status open
```

That creates `data/kalshi/markets_live.csv` and `data/reports/upcoming_market_suggestions.csv`. These are public market-data snapshots only; the project still does not place real trades.

To search recent/live Kalshi markets across all series for NBA winners, spreads, totals, team totals, and player props:

```powershell
python scripts/discover_kalshi_nba_markets.py --start-date 2026-04-01 --end-date 2026-05-07
```

That creates `data/raw/kalshi/broad_nba_markets.csv`, `data/processed/kalshi_broad_market_taxonomy.csv`, and `data/reports/kalshi_broad_market_taxonomy_summary.json`. This is the first pass at finding non-winner NBA markets; it is intentionally separate from the high-confidence game-winner backtest path until each market type has its own model and calibration.

To crawl older archived Kalshi NBA markets, use the historical series endpoint directly:

```powershell
python scripts/kalshi_backfill_historical_series.py --max-pages 100
```

This writes `data/raw/kalshi/historical_series_markets.csv`, updates `data/processed/kalshi_possible_nba_markets.csv`, and writes `data/reports/kalshi_historical_series_backfill_summary.json`. This is the preferred old-data path because Kalshi historical markets are paginated by `series_ticker`, not searched by date window. The default series list includes full-game winners, spreads, totals, team totals, half-game markets, player stat props, and playoff/series markets; only `KXNBAGAME` rows are allowed into the full-game winner backtest.

To use Kalshi's series/events endpoints as a discovery layer for new NBA series:

```powershell
python scripts/kalshi_discover_series_events.py --event-max-pages 50 --market-max-pages 100
```

This writes `data/raw/kalshi/series_list.csv`, `data/raw/kalshi/events_discovery.csv`, candidate reports in `data/reports/`, and then crawls discovered `KXNBA...` historical market series. The event endpoint is especially useful because old events can still be listed even when nested historical markets are hidden from the event response.

Some broad rows are multivariate combination markets. To inventory the NBA legs inside those combos without treating combo prices as single-leg prices:

```powershell
python scripts/extract_multivariate_nba_legs.py
```

This writes `data/processed/kalshi_multivariate_nba_legs.csv` and `data/reports/kalshi_multivariate_nba_legs_summary.json`. These rows are useful evidence of spread, total, and player-prop inventory, but they remain blocked for single-leg backtesting until direct underlying market prices are fetched.

To try fetching direct market rows for the spread, total, and player-prop legs found inside those combo markets:

```powershell
python scripts/fetch_underlying_nba_leg_markets.py --max-tickers 25 --max-consecutive-failures 3
```

This writes `data/raw/kalshi/underlying_nba_leg_markets.csv`, `data/reports/underlying_nba_leg_market_requests.csv`, and `data/reports/underlying_nba_leg_market_summary.json`. The command tries spread and total legs first, then player props. It stops early if repeated requests fail, which keeps a blocked or rate-limited run from hanging for a long time. Player props remain deferred even if their lines parse cleanly; spread and total models come first.

To download pregame candles for direct spread and total markets:

```powershell
python scripts/kalshi_download_line_candles.py
```

This writes `data/raw/kalshi/line_candles/`, `data/processed/kalshi_line_pregame_prices.csv`, and `data/reports/kalshi_line_candle_summary.json`. These prices are kept separate from the game-winner backtest until spread and total probability engines have their own validation.

To evaluate direct spread and total markets against the margin/total prediction engines:

```powershell
python scripts/evaluate_line_markets.py --edge-threshold 0.05
```

This writes `data/reports/line_market_model_eval.csv` and `data/reports/line_market_model_eval_summary.json`. The result is exploratory only. It does not affect the headline slate backtest and it keeps parlays blocked unless individual spread/total economics become positive out-of-sample.

To classify all cached NBA Kalshi markets by bet type:

```powershell
python scripts/build_kalshi_market_taxonomy.py
```

That creates `data/processed/kalshi_market_taxonomy.csv` and `data/reports/kalshi_market_taxonomy_summary.json`. The taxonomy is the broad all-bets inventory; the full-game winner backtest still uses only high-confidence `KXNBAGAME` rows until spread, total, team-total, and prop engines have separate validation.

Inside the Streamlit app, the `Manual Markets` tab lets you:

- load a local or uploaded Kalshi-style CSV
- download a blank market template
- preview model-vs-market YES signals
- run a paper backtest for rows that include final settlement values

The Streamlit labels are written for normal use:

- `Our Picked Team Win Chance` is the old model YES probability.
- `Market-Implied Chance` is the market price converted into a probability.

Or refresh the main project outputs in one command:

```powershell
python scripts/run_full_pipeline.py
```

That uses cached NBA data, rebuilds features, trains models, runs walk-forward predictions, runs the mock/manual-market backtest, sweeps thresholds, analyzes results, and rebuilds the dashboard. Add `--download` only when you want it to fetch NBA data again.

## Metrics

Model metrics shown:

- accuracy
- log loss
- Brier score
- ROC AUC
- calibration curve

Backtest metrics shown:

- ending bankroll
- total return
- number of trades
- win rate
- max drawdown
- average edge
- average profit per trade
- ROI on amount risked

Saved report files currently include:

- `data/reports/model_metrics.json`
- `data/reports/model_predictions.csv`
- `data/reports/model_feature_diagnostics.csv`
- `data/reports/model_tuning_results.csv`
- `data/reports/model_tuning_summary.json`
- `data/reports/tuned_walk_forward_predictions.csv`
- `data/reports/home_win_ensemble_predictions.csv`
- `data/reports/home_win_ensemble_summary.json`
- `data/reports/calibration_curve.csv`
- `data/reports/calibration_curve.png`
- `data/reports/probability_distribution.png`
- `data/reports/backtest_trades.csv`
- `data/reports/backtest_summary.json`
- `data/reports/edge_calibrated_trades.csv`
- `data/reports/edge_calibration_bins.csv`
- `data/reports/edge_calibration_summary.json`
- `data/reports/edge_calibration_audit.csv`
- `data/reports/edge_calibration_negative_edge_signals.csv`
- `data/reports/edge_calibrated_trades_market_blend.csv`
- `data/reports/edge_calibration_audit_market_blend.csv`
- `data/reports/edge_consensus_calibrated_trades.csv`
- `data/reports/edge_consensus_summary.json`
- `data/reports/edge_robust_consensus_trades.csv`
- `data/reports/edge_robust_consensus_summary.json`
- `data/reports/signal_stability_consensus.csv`
- `data/reports/signal_stability_robust_consensus.csv`
- `data/reports/strategy_readiness.csv`
- `data/reports/strategy_readiness_summary.json`
- `data/reports/signal_rule_sweep.csv`
- `data/reports/signal_rule_sweep_summary.json`
- `data/reports/parlay_correlation_report.csv`
- `data/reports/parlay_correlation_summary.json`
- `data/reports/forward_recommendations.csv`
- `data/reports/forward_recommendations_summary.json`
- `data/reports/backtest_trades_tuned.csv`
- `data/reports/backtest_summary_tuned.json`
- `data/reports/matched_markets.csv`
- `data/reports/market_matching_report.json`
- `data/reports/market_validation_report.json`
- `data/reports/market_data_quality_report.json`
- `data/reports/paper_trade_suggestions.csv`
- `data/reports/threshold_sweep.csv`
- `data/reports/threshold_sweep.png`
- `data/reports/portfolio_trades.csv`
- `data/reports/portfolio_slates.csv`
- `data/reports/portfolio_summary.json`
- `data/reports/portfolio_trades_calibrated.csv`
- `data/reports/portfolio_slates_calibrated.csv`
- `data/reports/portfolio_summary_calibrated.json`
- `data/reports/portfolio_trades_market_blend_calibrated.csv`
- `data/reports/portfolio_slates_market_blend_calibrated.csv`
- `data/reports/portfolio_summary_market_blend_calibrated.json`
- `data/reports/portfolio_trades_consensus_calibrated.csv`
- `data/reports/portfolio_slates_consensus_calibrated.csv`
- `data/reports/portfolio_summary_consensus_calibrated.json`
- `data/reports/portfolio_trades_robust_consensus.csv`
- `data/reports/portfolio_slates_robust_consensus.csv`
- `data/reports/portfolio_summary_robust_consensus.json`
- `data/reports/headline_backtest_summary.json`
- `data/reports/signal_rule_walk_forward_trades.csv`
- `data/reports/signal_rule_walk_forward_folds.csv`
- `data/reports/signal_rule_walk_forward_monthly.csv`
- `data/reports/signal_rule_walk_forward_summary.json`
- `data/reports/market_line_coverage.csv`
- `data/reports/market_line_coverage_summary.json`
- `data/reports/kalshi_multivariate_nba_legs_summary.json`
- `data/reports/security_audit_findings.csv`
- `data/reports/security_audit_summary.json`
- `data/reports/prediction_probability_bins.csv`
- `data/reports/prediction_probability_bins.png`
- `data/reports/prediction_season_summary.csv`
- `data/reports/prediction_season_summary.png`
- `data/reports/backtest_edge_bins.csv`
- `data/reports/backtest_edge_bins.png`
- `data/reports/top_backtest_trades.csv`
- `data/reports/diagnostics_summary.json`
- `data/reports/dashboard.html`
- `data/reports/equity_curve.png`
- `data/reports/edge_distribution.png`
- `data/reports/walk_forward_predictions.csv`
- `data/reports/all_game_predictions.csv`
- `data/reports/walk_forward_metrics.json`
- `data/reports/market_type_predictions.csv`
- `data/reports/market_type_model_metrics.json`
- `data/reports/market_type_probability_calibration.csv`
- `data/reports/market_type_calibration_summary.json`
- `data/reports/player_feature_comparison.json`
- `data/reports/upcoming_predictions.csv`
- `data/reports/upcoming_market_suggestions.csv`
- `data/kalshi/markets_live.csv`
- `data/reports/walk_forward_calibration_curve.csv`
- `data/reports/walk_forward_calibration_curve.png`
- `data/reports/walk_forward_probability_distribution.png`
- `data/reports/data_validation_summary.json`
- `data/reports/data_validation_issues.csv`

## Known Limitations

- `nba_api` can timeout or reject rapid requests. The downloader uses retries and local cache, but slow requests can still happen.
- Current code can cache regular-season and playoff data, but play-in coverage still depends on model game rows, start times, and candles being present.
- Kalshi market matching is intentionally local/mock-first because market naming can vary.
- Backtests use pregame candles only; if a candle source is missing or coarse, the row is filtered or marked lower quality.
- The interactive dashboard requires the optional `requirements-dashboard.txt` install.
- Player features are recent-rotation proxies from prior box scores. Optional availability features depend on the local CSV you provide; they are not a paid feed or guaranteed confirmed starting lineups.

## Future Improvements

- Add real pre-game Kalshi market snapshots once a clean free source or manual workflow is available.
- Add richer dashboard controls for comparing models and market files side by side.
- Add closing-line-value tracking.
- Add fractional Kelly with strict caps.
- Improve injury/player availability inputs only when the data source is free, clean, and allowed.
- Compare player-aware models against team-only models with walk-forward splits before trusting any uplift.
- Expand beyond team win markets only after the team model is stable.

## Files To Inspect First

- `src/data/nba_client.py` for NBA download and cache behavior.
- `src/data/loaders.py` for loading raw logs and building one row per game.
- `src/data/player_client.py` for player game-log download and cache behavior.
- `src/features/team_features.py` for final modeling rows.
- `src/features/player_features.py` for leak-safe player rotation features.
- `src/features/rolling_stats.py` for leak-safe recent-form stats.
- `src/reports/data_validation.py` for saved-artifact quality checks.
- `src/models/train_model.py` for baseline model training.
- `src/models/tuning.py` for time-aware model tuning.
- `src/models/market_blend.py` for expanding-window model plus market probability blending.
- `src/strategy/backtest.py` for fake bankroll accounting.
- `src/strategy/edge_calibration.py` for expanding-window edge validation before portfolio selection.
- `src/strategy/consensus.py` for requiring raw and market-blend calibrated signals to agree.
- `src/strategy/robustness.py` for lower-confidence-bound signal screening.
- `src/strategy/stability.py` for month-by-month signal stability diagnostics.
- `src/strategy/readiness.py` for strategy readiness gates before parlay research.
- `src/strategy/forward.py` for upcoming game recommendations and paper sizing.
- `src/strategy/portfolio.py` for pre-parlay individual slate selection.
- `config.yaml` for default settings.
- `scripts/download_nba_data.py` for the first CLI command.
- `scripts/download_all_nba_data.py` for regular-season plus playoff downloads.
- `scripts/build_features.py` for turning cached logs into feature data.
- `scripts/run_backtest.py` for the mock Kalshi paper-trading run.
- `scripts/calibrate_edges.py` for validating edge buckets before selecting slates.
- `scripts/build_consensus_edges.py` for the stricter pre-parlay consensus filter.
- `scripts/screen_robust_edges.py` for lower-confidence-bound filtering.
- `scripts/analyze_signal_stability.py` for signal stability diagnostics.
- `scripts/strategy_readiness.py` for conservative paper-trade readiness scoring.
- `scripts/build_forward_recommendations.py` for the website's current/upcoming games tab.
- `scripts/market_blend.py` for market-aware probability blending and its paper backtest.
- `scripts/optimize_portfolio.py` for constrained individual-bet slate selection before parlays.
- `scripts/walk_forward.py` for multi-season out-of-sample predictions.
- `scripts/tune_model.py` for walk-forward model tuning.
- `scripts/train_market_type_models.py` for spread and total-points engines.
- `scripts/predict_upcoming.py` for upcoming NBA schedule predictions.
- `scripts/download_kalshi_markets.py` for public Kalshi NBA game market snapshots.
- `scripts/discover_kalshi_nba_markets.py` for broader NBA market discovery across recent Kalshi series.
- `scripts/sweep_thresholds.py` for comparing edge-threshold settings.
- `scripts/analyze_results.py` for extra model and backtest diagnostics.
- `scripts/dashboard_app.py` for the optional Streamlit dashboard.
- `scripts/run_dashboard.py` for launching the optional Streamlit dashboard.
- `scripts/refresh_current_season.py` for refreshing the current NBA season and rebuilding reports.
- `scripts/run_full_pipeline.py` for refreshing all main outputs and the dashboard.

## Dashboard Plan

The project includes a self-contained local HTML dashboard:

```powershell
python scripts/build_dashboard.py
```

It reads from `data/reports/`, embeds the generated plots, and provides tabs for overview, model metrics, probability calibration, season diagnostics, backtest results, edge-bucket results, market quality, and searchable data tables.

The project also includes an optional Streamlit dashboard:

```powershell
python -m pip install -r requirements-dashboard.txt
python scripts/run_dashboard.py
```

It shows every historical game in the saved prediction dataset, the model's pick, whether the pick won, whether a market price was loaded, and whether the paper simulator chose to bet.

Run this when you want the `Upcoming` tab populated:

```powershell
python scripts/predict_upcoming.py --days 14
python scripts/download_kalshi_markets.py --status open
```

It also includes a `Manual Markets` workspace for testing a market CSV without overwriting saved reports. Historical backtests need `settlement` values; future or same-day files without settlements will show signal previews but cannot produce resolved P/L yet.

Both dashboards use the same report files:

- model metrics and calibration plots
- walk-forward predictions
- upcoming predictions
- matched public Kalshi game prices
- market validation and quality reports
- paper-trade suggestions
- backtest trades and summaries
- threshold sweeps
