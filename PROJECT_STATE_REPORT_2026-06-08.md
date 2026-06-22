# NBA Kalshi Predictor Project State Report

Date: 2026-06-08  
Workspace: `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor`  
Primary current goal: prove real, repeatable edge on single-game Kalshi NBA markets before doing parlay optimization.

## 1. Executive Summary

This repository is a local NBA/Kalshi market research system. It is not currently a production trading system and it does not place real trades. The project has evolved from a broad prediction/parlay/dashboard project into a single-game edge research workflow focused on:

- leak-safe NBA feature generation
- walk-forward home-win modeling
- Kalshi NBA market discovery and matching
- pregame bid/ask candle extraction
- market truth auditing
- YES and NO bid/ask-aware signal generation
- fake bankroll backtesting
- calibration, CLV, residual, NO-side, and defensive filter audits
- proof gates that intentionally block live-looking recommendations until single-game edge is actually proven
- a static HTML dashboard

The current research conclusion is clear: the system has useful infrastructure, but single-game edge is not proven. Current fair-price signals are proof-gated to `No bet`. Parlay recommendations are also blocked by design.

Key current evidence:

- Matched Kalshi game markets: 1,232.
- Usable 60m, 30m, and 5m pregame prices: 1,232 each.
- Backtest markets seen: 1,224.
- Backtest trades: 634, split 325 YES and 309 NO.
- Backtest ending bankroll: $28.86 from $100.00.
- Average CLV: -0.03 cents.
- Positive CLV rate: about 25%.
- Single-game proof status: `not_proven`.
- Fair-price output rows: 1,224.
- Actionable fair-price bets after proof gate: 0.
- Ungated research fair-price bets: 540.
- Parlay recommendation status: `blocked_single_game_edge_not_proven`.

The repo is broad and fairly mature, but the next agent should treat it as a research system that needs calibration/market-quality improvement, not as a betting system ready to use.

## 2. Repository Size and Shape

Current codebase counts:

- `src`: 101 Python files, about 28,402 lines.
- `scripts`: 96 Python files, about 11,963 lines.
- `tests`: 74 Python files, about 7,954 lines.
- `src/data`: 29 files, about 7,698 lines.
- `src/features`: 7 files, about 1,196 lines.
- `src/models`: 10 files, about 2,073 lines.
- `src/reports`: 11 files, about 6,527 lines.
- `src/strategy`: 39 files, about 10,243 lines.

Large generated artifact areas:

- `data/raw/kalshi/candles`: 2,148 files, about 342.74 MB.
- `data/raw/kalshi/public_api/json`: 41,285 files, about 881.84 MB.
- `data/reports`: 605 files, about 63.17 MB.
- `outputs`: 26 files, about 12.64 MB.

The project contains a lot of generated state. Do not assume every CSV/JSON is a source file. Most files under `data/reports`, `outputs`, `data/raw`, `data/interim`, `data/processed`, and `data/models` are generated artifacts.

## 3. Git and Working Tree State

Git reports a dubious ownership warning unless commands are run with:

```powershell
git -c safe.directory='C:/Users/arilo/Downloads/Python Projects/nba_kalshi_predictor' status --short
```

The working tree is dirty. There are many modified files, deleted old dashboard/notebook files, and untracked new scripts, tests, data artifacts, reports, and runner files. This appears intentional and consistent with recent cleanup/research work. Do not revert these changes without explicit user approval.

Notable repository-state facts:

- Old Streamlit dashboard files were deleted from active locations and archived under `_archive_unused_files`.
- Several notebooks were removed from active locations and archived.
- `CLEANUP_REPORT.md` and `TODO.md` describe the recent cleanup and research direction.
- `.env`, `.secrets`, `.venv`, `.streamlit`, `.pip_tmp`, and `tmpmvt2s4u0` exist locally.
- `.pip_tmp` and `tmpmvt2s4u0` are described as questionable locked temp folders in cleanup notes.

## 4. Active Top-Level Files

Important project-root files:

- `README.md`: current user-facing description and command path.
- `TODO.md`: detailed completed/current/next research log.
- `CLEANUP_REPORT.md`: cleanup audit and active workflow description.
- `config.yaml`: central configuration for data ranges, model, strategy, Kalshi, matching, and backtest filters.
- `requirements.txt`: Python dependencies.
- `run_cached_pipeline.bat`: easiest Windows entry point.
- `run_single_game_pipeline.ps1`: PowerShell wrapper used by the batch file.
- `open_dashboard.bat` / `open_website.bat`: open the static dashboard.

## 5. Runtime and Dependencies

The dependency file currently lists:

- `pandas`
- `numpy`
- `scikit-learn`
- `requests`
- `cryptography`
- `python-dotenv`
- `pydantic`
- `tqdm`
- `matplotlib`
- `joblib`
- `nba_api`
- `pyarrow`
- `pyyaml`
- `kagglehub`
- `openpyxl`
- `xlrd`
- `xgboost`

Python observations:

- System/default `python --version`: Python 3.12.10.
- Project `.venv\Scripts\python.exe --version`: Python 3.12.10.
- Bundled Codex runtime Python exists at `C:\Users\arilo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, version 3.12.13.
- The project `.venv` is the correct environment for tests because it has `nba_api`.
- Running tests with default `python` failed on missing `nba_api`.
- Running tests with `.venv\Scripts\python.exe` passed.

Recommended validation command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## 6. Configuration State

`config.yaml` is the main config.

Important active settings:

- NBA data seasons: 2018 through 2025.
- Model target: `target_home_win`.
- Model type: `logistic_regression`.
- Training seasons: 2018 through 2023.
- Validation season: 2024.
- Test season: 2025.
- Starting bankroll: 100.
- Edge threshold: 0.05.
- Max bet fraction: 0.03.
- Min market price: 0.05.
- Max market price: 0.95.
- YES ask for buys: true.
- NO trades allowed: true.
- Kalshi production base URL: `https://external-api.kalshi.com/trade-api/v2`.
- Kalshi demo base URL: `https://demo-api.kalshi.co/trade-api/v2`.
- Kalshi auth required: false.
- Default candle interval: 1 minute.
- Fallback candle interval: 60 minutes.
- Match thresholds: auto 0.85, review 0.60.
- Backtest snapshot: 60 minutes before tipoff.
- Backtest min volume: 10.
- Allowed price quality: `bid_ask_available`.
- Backtest requires bid/ask: true.
- Max candle interval: 60 minutes.
- Max bid/ask spread: 10 cents.

Important caveat: `scripts/run_backtest.py` has a `--market-source auto` default. Since `data/raw/sportsbook/nba_moneyline_odds.csv` now exists, a fresh standalone run of `scripts/run_backtest.py` will choose sportsbook mode unless `--market-source kalshi` is supplied. The existing canonical `data/reports/backtest_summary.json` still contains Kalshi candlestick diagnostics, but this default is a workflow hazard.

## 7. Current Command Path

The intended simple command is:

```powershell
.\run_cached_pipeline.bat
```

The batch file calls:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File run_single_game_pipeline.ps1
```

The PowerShell wrapper:

- resolves Python in this order: `.venv`, system Python, bundled Codex runtime
- sets `PYTHONPATH` to `.venv\Lib\site-packages` only when it falls back to bundled Python
- defaults to cached mode unless `-RefreshMarkets` or `-RefreshCandles` is provided
- passes `--skip-market-pull` by default
- passes `--skip-candles` by default
- optionally passes `--download`, `--force-download`, and `--skip-dashboard`

The underlying Python command is:

```powershell
python scripts\run_single_game_research_pipeline.py --skip-market-pull --skip-candles --kalshi-start-date 2023-10-01 --kalshi-end-date <today>
```

With the current default cached flags and dashboard enabled, the pipeline builds 56 steps:

1. Build leak-safe features
2. Audit player data coverage
3. Audit availability input gaps
4. Train home-win models
5. Run walk-forward evaluation
6. Compare player-aware market edge
7. Sweep player/team edge agreement
8. Tune calibrated home-win model
9. Train margin and total models
10. Build home-win ensemble audit
11. Match Kalshi markets to NBA games
12. Build market truth audit
13. Build Kalshi coverage audit
14. Run realistic bid/ask backtest
15. Sweep market-anchored probability blends
16. Compare pregame snapshot entry policies
17. Audit best two-hour snapshot CLV
18. Calibrate edge bins
19. Calibrate side/price/edge bins
20. Sweep price-aware calibration settings
21. Build best price-aware calibration
22. Sweep calibrated player/team edge agreement
23. Audit calibrated residuals
24. Audit best price-aware residuals
25. Audit market movement attribution
26. Audit best price-aware market movement
27. Diagnose edge failure drivers
28. Sweep prior-month CLV slice filters
29. Sweep side-suppression research policies
30. Audit NO-only market regimes
31. Audit NO probability calibration vs CLV
32. Sweep NO calibration guardrails
33. Sweep NO probability shrinkage
34. Sweep NO player-agreement guardrails
35. Sweep corrected CLV rules
36. Sweep corrected best price-aware CLV rules
37. Sweep residual guardrails
38. Sweep best price-aware residual guardrails
39. Analyze closing-line value
40. Build side-specific CLV-filtered strategy
41. Sweep CLV price/month stability rules
42. Walk-forward validate CLV price/month rules
43. Analyze CLV decay drivers
44. Build defensive CLV-filtered strategy
45. Sweep defensive rule thresholds
46. Walk-forward validate defensive rules
47. Test defensive sample expansion
48. Audit defensive failure month
49. Optimize conservative calibrated single-bet slate
50. Optimize CLV-filtered single-bet slate
51. Optimize defensive CLV-filtered single-bet slate
52. Score single-game strategy readiness
53. Build single-game proof gates
54. Build fair-price single-game signals
55. Build conservative parlay recommendations
56. Build dashboard

If `-RefreshMarkets` is passed, two additional early Kalshi API steps run:

- Backfill raw public Kalshi Sports/NBA markets.
- Pull and cache Kalshi NBA markets.

If `-RefreshCandles` is passed, the candle download step runs:

- Download candles and extract pregame prices.

## 8. Backend / Core Architecture

There is no active web backend or API server in the cleaned workflow. The "backend" is a local Python batch research pipeline.

Main package layout:

- `src/config.py`: pydantic config models, YAML loading, environment overrides.
- `src/logging_setup.py`: logging setup.
- `src/pipeline.py`: older command-list builder for a broader full pipeline.
- `src/data`: data loaders, API clients, Kalshi matching/candles, sportsbook odds, team aliases, validation, game times.
- `src/features`: Elo, rolling team stats, schedule features, player features, target building.
- `src/models`: training, prediction, evaluation, calibration, ensemble, tuning, market-type models, market blending.
- `src/strategy`: signal generation, backtest, CLV, calibration/filters, fair-price engine, proof gates, portfolio, parlay research, paper trading.
- `src/reports`: dashboard renderer, coverage reports, market truth audit, diagnostics, plots, data validation.

Most user-facing work happens through scripts under `scripts/`, which import `src` modules after inserting `src` onto `sys.path`.

## 9. Data Sources and Data Flow

### NBA Data

NBA data is stored under:

- `data/raw/nba`
- `data/interim/nba_games.parquet`
- `data/processed/modeling_dataset.parquet`

Current modeling dataset:

- Path: `data/processed/modeling_dataset.parquet`
- Rows: 9,519.
- Columns: 170.
- Date range from dashboard diagnostics: 2018-10-16 through 2026-04-12.
- Season counts:
  - 2018: 1,230
  - 2019: 1,059
  - 2020: 1,080
  - 2021: 1,230
  - 2022: 1,230
  - 2023: 1,230
  - 2024: 1,230
  - 2025: 1,230

NBA clients and related modules:

- `src/data/nba_client.py`
- `src/data/player_client.py`
- `src/data/scoreboard.py`
- `src/data/game_times.py`
- `src/data/seasons.py`
- `src/data/cache.py`
- `src/data/loaders.py`

Game start times:

- `src/data/game_times.py` prefers NBA official scoreboard data and falls back to ESPN.
- `data/interim/nba_game_start_times.csv` exists.
- Candle extraction uses exact game start time when available and otherwise falls back to date/time heuristics.

### Player Data and Availability

Player features are active:

- `src/features/player_features.py`
- `scripts/download_nba_player_data.py`
- `scripts/build_player_features.py`
- `scripts/audit_player_data.py`

Current output artifacts include:

- `outputs/player_features_by_game.csv`
- `outputs/player_features_summary.json`
- `outputs/player_feature_diagnostics.csv`
- `outputs/player_feature_importance.csv`
- `data/reports/player_data_summary.json`
- `data/reports/player_feature_coverage.csv`

Availability/injury input is not solved. The current TODO says the availability gap status is `needs_availability_input`, with missing manual/free statuses. The configured input path is:

- `data/raw/nba/injuries/availability.csv`

Availability-related code:

- `src/data/injury_availability.py`
- `src/data/availability_template.py`
- `src/reports/availability_gaps.py`
- `scripts/build_availability_template.py`
- `scripts/audit_availability_gaps.py`

### Sportsbook Odds

Sportsbook odds are used as a historical market proxy and benchmark, not as the final Kalshi target.

Important files:

- Raw schema: `data/raw/sportsbook/nba_moneyline_odds.schema.csv`
- Raw odds: `data/raw/sportsbook/nba_moneyline_odds.csv`
- Processed odds: `data/processed/sportsbook_odds.csv`
- Split config: `data/processed/sportsbook_split_config.json`
- Coverage report: `data/reports/sportsbook_coverage_by_season.csv`
- Match report: `outputs/sportsbook_match_report.csv`

Current processed sportsbook diagnostics from dashboard test output:

- Processed rows: 18,649.
- Sportsbook odds counts by season:
  - 2018: 1,230
  - 2019: 1,059
  - 2020: 1,080
  - 2021: 1,230
  - 2022: 664
  - 2023: 0
  - 2024: 0
  - 2025: 0

Sportsbook code:

- `src/data/sportsbook_odds.py`
- `scripts/download_kaggle_nba_odds.py`
- `scripts/import_kaggle_nba_odds.py`
- `scripts/prepare_sportsbook_odds.py`
- `scripts/validate_sportsbook_odds.py`

### Kalshi Data

Kalshi-related source modules:

- `src/data/kalshi_client.py`
- `src/data/kalshi_backfill.py`
- `src/data/kalshi_public_backfill.py`
- `src/data/kalshi_series_backfill.py`
- `src/data/kalshi_event_discovery.py`
- `src/data/kalshi_discovery.py`
- `src/data/kalshi_matcher.py`
- `src/data/kalshi_candles.py`
- `src/data/kalshi_taxonomy.py`
- `src/data/line_market_candles.py`
- `src/data/market_quality.py`
- `src/data/market_line_audit.py`

Important Kalshi artifacts:

- `data/raw/kalshi/broad_nba_markets.csv`
- `data/raw/kalshi/public_api/*.csv`
- `data/raw/kalshi/public_api/json/*.json`
- `data/raw/kalshi/candles/*`
- `data/processed/kalshi_possible_nba_markets.csv`
- `data/processed/kalshi_public_possible_nba_markets.csv`
- `data/processed/kalshi_game_market_matches.csv`
- `data/processed/kalshi_matches_needs_review.csv`
- `data/processed/kalshi_pregame_prices.csv`
- `data/reports/market_truth_audit.csv`
- `data/reports/market_truth_audit_summary.json`
- `data/reports/kalshi_coverage_summary.json`

The Kalshi client supports:

- public unauthenticated market reads
- optional authenticated reads with `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`
- market paging
- historical market fetches
- recent and historical candlestick fetches
- batch candlestick parsing and a `get_batch_market_candlesticks` method

Important gap: `src/data/kalshi_client.py` has a batch candlestick method, but the active candle extraction pipeline in `src/data/kalshi_candles.py` still loops per market ticker and calls `get_market_candlesticks` or `get_historical_market_candlesticks`. Batch candle orchestration is not wired into the active workflow yet.

## 10. Feature Engineering

Feature modules:

- `src/features/elo.py`
- `src/features/rolling_stats.py`
- `src/features/schedule_features.py`
- `src/features/team_features.py`
- `src/features/player_features.py`
- `src/features/target_builder.py`

Feature categories include:

- pregame Elo and Elo win probability
- rest differential
- back-to-back flags
- recent rolling win percentage
- recent rolling point differential
- season-to-date win percentage
- season-to-date margin
- shooting, rebounding, assists, turnovers form
- neutral site and playoff flags
- player rotation features
- availability/injury-derived features when input exists

The modeling code explicitly checks for leakage:

- chronological season splits
- no validation/test seasons in training
- rolling features use prior games
- player features use prior games
- final score/result is not used as a feature

## 11. Modeling State

Main model modules:

- `src/models/train_model.py`
- `src/models/walk_forward.py`
- `src/models/evaluate.py`
- `src/models/predict.py`
- `src/models/calibration.py`
- `src/models/tuning.py`
- `src/models/ensemble.py`
- `src/models/market_type_models.py`
- `src/models/market_blend.py`

Main model scripts:

- `scripts/train.py`
- `scripts/walk_forward.py`
- `scripts/tune_model.py`
- `scripts/train_market_type_models.py`
- `scripts/build_home_win_ensemble.py`
- `scripts/train_model.py`

There are two model tracks:

1. The current modular pipeline around `src/models/*` and scripts like `train.py`, `walk_forward.py`, and `tune_model.py`.
2. A large legacy/research script, `scripts/train_model.py`, that creates many `outputs/*` artifacts and compares fair model variants, sportsbook baselines, player features, Kalshi paper trades, and strategy grids.

Current walk-forward artifact:

- `data/reports/walk_forward_predictions.csv`
- `data/reports/walk_forward_metrics.json`

Current walk-forward metrics:

- Model type: logistic regression.
- Target: `target_home_win`.
- First test season: 2019.
- Last test season: 2025.
- Number of predictions: 8,289.
- Overall model accuracy: 0.6541.
- Overall model Brier score: 0.2165.
- Overall model log loss: 0.6229.
- Overall model ROC AUC: 0.7038.
- Elo baseline accuracy: 0.6460.
- Elo baseline Brier score: 0.2196.
- Elo baseline log loss: 0.6290.
- Elo baseline ROC AUC: 0.6943.

From `outputs/model_performance_summary.json`:

- Best fair model: `team_plus_selected_player_random_forest`.
- Best calibrated model: `walk_forward_champion_uncalibrated`.
- Champion validation log loss: 0.6539.
- Champion validation Brier: 0.2308.
- Champion validation AUC: 0.6540.
- Champion beats Elo: true.
- Champion beats team-only: true.
- Champion beats sportsbook benchmark: false.
- Fair model historically validated: false.
- Paper trading warning: "Paper trading has not validated this strategy."

Interpretation: the model has some predictive signal, but it does not yet beat market-quality benchmarks strongly enough to claim tradable edge.

## 12. Market Matching and Market Truth

Market matching module:

- `src/data/kalshi_matcher.py`

Market matching script:

- `scripts/kalshi_match_games.py`

Market matching output:

- `data/processed/kalshi_game_market_matches.csv`

Current match rows:

- Total rows: 9,539.
- Auto matched: 1,232.
- No match: 8,307.
- Needs review: 0.

Matching logic:

- normalizes market title/subtitle text
- uses team aliases from Kalshi backfill/team aliases modules
- checks expected Kalshi event ticker orientation
- scores home/away mention, win language, exact matchup ticker, market date, negative prop/spread/total/futures terms, and YES-team identifiability
- infers YES team from explicit fields, ticker suffix, or first team mentioned in win-language text

Market truth audit module:

- `src/reports/coverage.py`

Market truth script:

- `scripts/market_truth_audit.py`

Market truth outputs:

- `data/reports/market_truth_audit.csv`
- `data/reports/market_truth_audit_summary.json`

Current market truth summary:

- Matched game markets: 1,232.
- Auto matched: 1,232.
- Needs review: 0.
- Usable price counts:
  - `pregame_60m`: 1,232.
  - `pregame_30m`: 1,232.
  - `pregame_5m`: 1,232.
  - `pregame_best_le_120m`: 1,231.
- Ticker mapping mismatch count: 0.
- Wide spread count: 0.
- Low liquidity count: 185.
- Max spread threshold: 10 cents.
- Min volume threshold: 10.

Market truth audit columns include:

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
- `pregame_price_best_le_120m`
- `yes_bid`
- `yes_ask`
- `mid_price`
- `spread`
- `volume`
- `open_interest`
- `match_status`
- `price_quality`
- `selected_snapshot`
- `ticker_mapping_mismatch`
- `wide_spread`
- `low_liquidity`

Interpretation: market coverage is now good enough to do serious audits, but liquidity remains a real filter issue.

## 13. Candle and Pregame Price State

Candle module:

- `src/data/kalshi_candles.py`

Candle script:

- `scripts/kalshi_download_candles.py`

Pregame price output:

- `data/processed/kalshi_pregame_prices.csv`

Current pregame price rows:

- Rows: 4,928.
- Columns: 16.
- Snapshot targets include:
  - `pregame_60m`
  - `pregame_30m`
  - `pregame_5m`
  - `pregame_best_le_120m`

Current coverage summary:

- Pregame price rows: 4,928.
- Games with usable pregame price: 1,232.
- Games with start times: 1,234.
- Market date min: 2025-04-15.
- Market date max: 2026-05-10.
- Price quality counts:
  - `bid_ask_available`: 4,924.
  - `daily_candle_low_quality`: 3.
  - `missing`: 1.
- Period interval counts:
  - 1 minute: 4,924.
  - 1 day: 3.
  - missing: 1.

Important candle behavior:

- Uses bid/ask if present.
- Computes mid price from bid/ask.
- Falls back to last price only when bid/ask and mid are missing, but the backtest default filters to `bid_ask_available`.
- Marks daily candle prices as low quality.
- Uses 1-minute, then 60-minute, then daily intervals.
- Caches each market ticker separately under `data/raw/kalshi/candles`.

## 14. Backtest State

Backtest module:

- `src/strategy/backtest.py`

Backtest script:

- `scripts/run_backtest.py`

Current canonical outputs:

- `data/reports/backtest_trades.csv`
- `data/reports/backtest_summary.json`
- `data/reports/matched_markets.csv`

Current backtest summary:

- Starting bankroll: $100.00.
- Ending bankroll: $28.86.
- Total return: -71.14%.
- Markets seen: 1,224.
- Trades: 634.
- YES trades: 325.
- NO trades: 309.
- Market timeline: 2025-10-21 to 2026-04-12.
- Trade timeline: 2025-10-21 to 2026-04-12.
- Win rate: 35.49%.
- Average edge: 10.89%.
- Average profit per trade: -$0.1122.
- Max drawdown: -90.86%.
- Largest win: $36.54.
- Largest loss: -$7.79.
- ROI on amount risked: -2.90%.
- Average CLV: -0.03 cents.
- Median CLV: 0.00 cents.
- Positive CLV rate: 24.29%.
- YES win rate: 38.77%.
- YES profit: -$81.43.
- YES average CLV: -0.1625 cents.
- YES positive CLV rate: 23.44%.
- NO win rate: 32.04%.
- NO profit: +$10.29.
- NO average CLV: +0.1197 cents.
- NO positive CLV rate: 27.82%.

Backtest filters:

- Input games available: 8,289.
- Games with matched Kalshi market: 1,232.
- Games with usable pregame price: 1,224.
- Skipped due to no market: 7,057.
- Skipped due to no price: 8.
- Price rows seen: 4,928.
- Price rows non-missing: 4,927.
- After quality filter: 4,924.
- After interval filter: 4,924.
- After bid/ask filter: 4,924.
- After spread filter: 4,924.
- After volume filter: 4,517.
- Allowed price quality: `bid_ask_available`.
- Require bid/ask: true.
- Min volume: 10.
- Max candle interval: 60.
- Max spread: 10 cents.
- Preferred snapshot targets: 60m, then 30m, then 5m.

Important implementation detail:

- YES buy price uses YES ask.
- NO buy price uses `100 - yes_bid`, i.e. an ask-equivalent for buying NO.
- CLV for NO uses the NO-side reference price, not the YES-side reference price. A previous NO-candidate CLV bug was fixed and tested.

## 15. Signal Generation

Signal module:

- `src/strategy/signal.py`

The project still has `generate_yes_signal`, but the active backtest uses `add_two_sided_signals`.

Two-sided signal behavior:

- YES candidate:
  - model probability: `model_yes_prob`
  - market probability: YES ask cents / 100
  - edge: model YES probability minus YES ask probability
- NO candidate:
  - model probability: `1 - model_yes_prob`
  - market probability: `(100 - yes_bid) / 100`
  - edge: model NO probability minus NO ask-equivalent probability
- Best side is chosen by edge.
- Trade only occurs when:
  - price is within min/max market price bounds
  - edge is at or above threshold
  - `allow_no` permits NO if NO is best

This meets the project requirement to evaluate both YES and NO rather than only searching for YES bets.

## 16. Calibration, CLV, and Research Audits

Important strategy modules:

- `src/strategy/edge_calibration.py`
- `src/strategy/clv.py`
- `src/strategy/clv_filter.py`
- `src/strategy/clv_decay.py`
- `src/strategy/clv_concentration.py`
- `src/strategy/market_movement_audit.py`
- `src/strategy/residual_audit.py`
- `src/strategy/edge_failure.py`
- `src/strategy/no_side_audit.py`
- `src/strategy/no_calibration_audit.py`
- `src/strategy/no_calibration_guardrail.py`
- `src/strategy/no_shrinkage.py`
- `src/strategy/side_suppression.py`
- `src/strategy/defensive_filter.py`
- `src/strategy/defensive_failure_audit.py`
- `src/strategy/prior_clv_slice_filter.py`
- `src/strategy/market_anchor.py`

Current calibration artifacts:

- `data/reports/edge_calibrated_trades.csv`
- `data/reports/edge_calibration_summary.json`
- `data/reports/edge_calibrated_price_aware_trades.csv`
- `data/reports/edge_calibration_price_aware_summary.json`
- `data/reports/edge_calibration_price_aware_best_trades.csv`
- `data/reports/edge_calibration_price_aware_best_summary.json`
- `data/reports/price_aware_calibration_sweep.csv`
- `data/reports/price_aware_calibration_sweep_summary.json`

Current raw edge calibration summary:

- Rows: 1,224.
- Calibrated trades: 384.
- Trade timeline: 2025-12-01 to 2026-04-12.
- Uses expanding-window prior market dates only.

Current best price-aware calibration summary:

- Rows: 1,224.
- Calibrated trades: 344.
- Trade timeline: 2025-11-12 to 2026-04-12.
- Uses side + price bucket + edge bucket with prior dates only.
- Selected sweep rule status: `watchlist`.
- Selected sweep rule signals: 344.
- Selected sweep rule average profit per share: about +0.0017.
- Selected sweep rule average CLV: +0.0515 cents.
- Selected sweep positive CLV rate: 25.87%.

Current CLV summary:

- Trades with CLV: 604.
- Average CLV: -0.0298 cents.
- Median CLV: 0.0 cents.
- Positive CLV rate: 25.50%.

Interpretation: CLV is the main failure. Some narrow calibrated slices show watchlist behavior, but positive CLV rate is too low and repeatability is not proven.

## 17. Fair-Price Engine

Fair-price module:

- `src/strategy/fair_price.py`

Fair-price script:

- `scripts/build_fair_prices.py`

Outputs:

- `data/reports/fair_price_signals.csv`
- `data/reports/fair_price_summary.json`

The fair-price engine computes:

- model probability
- calibrated probability
- market YES ask
- market NO ask
- fair YES price
- fair NO price
- gross edge
- fee-adjusted edge
- spread penalty
- uncertainty penalty
- final edge
- recommendation
- confidence
- max size
- main reason
- main risk
- parlay eligibility

Current fair-price summary:

- Rows: 1,224.
- Actionable bets after proof gate: 0.
- YES bets after proof gate: 0.
- NO bets after proof gate: 0.
- No-bet rows: 1,224.
- Average final edge: 0.0368.
- Max final edge: 0.5452.
- Ungated research bets: 540.
- Ungated YES bets: 280.
- Ungated NO bets: 260.
- Proof gate status: `not_proven`.
- Single-game edge proven: false.

Important behavior: even when the ungated fair-price engine sees possible bets, `apply_single_game_proof_gate` blocks them all unless the proof report says single-game edge is proven.

## 18. Single-Game Proof Gates

Proof gate module:

- `src/strategy/single_game_proof.py`

Proof gate script:

- `scripts/single_game_proof.py`

Outputs:

- `data/reports/single_game_proof_gates.csv`
- `data/reports/single_game_proof_summary.json`

Current proof summary:

- Status: `not_proven`.
- Strategy under test: `defensive_clv_filtered`.
- Single-game edge proven: false.
- Parlay research allowed: false.
- Hard failures: 5.
- Warning failures: 0.
- Failed gates:
  - `strategy_backtest_profit`
  - `average_clv`
  - `positive_clv_rate`
  - `calibrated_strategy_readiness`
  - `repeatability_months`
- Recommendation: continue single-game calibration, CLV, and stability work. Do not optimize parlays.

There are 13 proof gates total; 8 currently pass and 5 fail.

The proof gates are exactly aligned with the current project rule: no parlays until single-game betting passes coverage, pricing, CLV, calibration, repeatability, and concentration checks.

## 19. Portfolio and Parlay State

Portfolio module:

- `src/strategy/portfolio.py`

Parlay modules:

- `src/strategy/parlay_research.py`
- `src/strategy/parlay_recommendations.py`

Parlay scripts:

- `scripts/analyze_parlay_correlations.py`
- `scripts/build_parlay_recommendations.py`

Current parlay recommendation output:

- `data/reports/parlay_recommendations.csv`
- `data/reports/parlay_recommendations_summary.json`

Current parlay summary:

- Status: `blocked_single_game_edge_not_proven`.
- Bankroll: 100.
- Input rows: 1,224.
- Eligible single-game legs: 0.
- Parlays: 0.
- Single-game edge proven: false.
- Same-game parlays allowed: false.
- Parlay recommendations allowed: false.
- Assumption: same-game parlays are excluded until correlation is modeled.

The code contains parlay research pieces, but the project instructions explicitly say not to make "find optimal parlays" the next goal. The current code correctly blocks parlay recommendations until single-game gates pass.

## 20. Frontend / Dashboard State

There is no live frontend server in the active workflow. The active frontend is a generated static HTML dashboard:

- Source renderer: `src/reports/dashboard.py`
- Build script: `scripts/build_dashboard.py`
- Output: `data/reports/dashboard.html`
- Open command: `open_dashboard.bat`

Current dashboard artifact:

- `data/reports/dashboard.html`
- Size: about 1.6 MB.
- Last modified: 2026-05-17 20:27:58 local time.

The old Streamlit dashboard is no longer active. Cleanup notes say these were archived:

- `requirements-dashboard.txt`
- `scripts/run_dashboard.py`
- `scripts/dashboard_app.py`
- `src/reports/interactive_dashboard.py`
- `tests/test_interactive_dashboard.py`

The dashboard renderer is large and self-contained. It embeds:

- data coverage diagnostics
- model performance summaries
- sportsbook benchmark context
- Kalshi market coverage
- backtest and bankroll controls
- fair-price recommendations
- proof-gated no-bet state
- parlay blocked state
- report tables and charts where available

Recent cleanup note: the dashboard was simplified into a sports-prediction layout with tabs such as Upcoming Games, Backtest, Model Info, and later a Parlays tab. Tests validate expected dashboard content and that removed old "Best Spots" behavior does not return.

## 21. Tests and Validation

Current test suite:

- 74 test files.
- 209 tests discovered with the project `.venv`.

Validation commands run for this report:

```powershell
python -m unittest discover tests
```

Result with default Python:

- Ran 203 tests.
- Failed with 2 import errors.
- Both errors were due to missing `nba_api` in the default Python environment:
  - `test_game_times`
  - `test_scoreboard`

Then:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Result with project venv:

- Ran 209 tests in 10.766 seconds.
- OK.

This means code/tests are currently green in the intended project environment, but the default Python environment is missing at least one required dependency.

Important test coverage areas:

- walk-forward split safety
- training split safety
- feature no-leakage
- team aliases
- Kalshi matching/backfill/candles/coverage
- market truth and market quality
- sportsbook odds import/matching
- bid/ask backtest
- YES/NO signals
- NO CLV bug coverage
- calibration
- CLV reports
- CLV filters/decay/concentration
- residual audits
- market movement audits
- defensive filters
- proof gates
- fair-price engine
- parlay research/recommendations
- dashboard rendering
- security audit

## 22. Current Output Directories

`data/reports` is the active report directory for the cleaned single-game research workflow. It contains hundreds of CSV/JSON/PNG/HTML artifacts. The most important files for handoff are:

- `dashboard.html`
- `walk_forward_predictions.csv`
- `walk_forward_metrics.json`
- `market_truth_audit.csv`
- `market_truth_audit_summary.json`
- `kalshi_coverage_summary.json`
- `matched_markets.csv`
- `backtest_trades.csv`
- `backtest_summary.json`
- `edge_calibration_summary.json`
- `edge_calibration_price_aware_best_summary.json`
- `edge_calibration_price_aware_best_trades.csv`
- `clv_summary.json`
- `clv_by_edge_bucket.csv`
- `clv_by_price_bucket.csv`
- `clv_by_team.csv`
- `clv_by_side.csv`
- `clv_by_season.csv`
- `clv_by_liquidity.csv`
- `strategy_readiness.csv`
- `strategy_readiness_summary.json`
- `single_game_proof_gates.csv`
- `single_game_proof_summary.json`
- `fair_price_signals.csv`
- `fair_price_summary.json`
- `parlay_recommendations.csv`
- `parlay_recommendations_summary.json`

`outputs` is used by the older/parallel fair-model and paper-trading path. Important files include:

- `model_performance_summary.json`
- `fair_model_performance_summary.json`
- `model_validation_predictions.csv`
- `fair_model_validation_predictions.csv`
- `fair_model_walk_forward_results.csv`
- `fair_model_walk_forward_summary.csv`
- `kalshi_paper_trades.csv`
- `kalshi_paper_trade_summary.csv`
- `kalshi_strategy_grid.csv`
- `kalshi_strategy_selected.json`
- `kalshi_strategy_holdout_results.csv`
- `player_features_by_game.csv`
- `player_features_summary.json`

Do not mix up `data/reports` and `outputs`. The current README/pipeline path emphasizes `data/reports`; `outputs` contains the older but still informative fair-model/paper-trading research path.

## 23. Known Hazards and Open Issues

### 23.1 Backtest Market Source Default

`scripts/run_backtest.py` defaults to `--market-source auto`. If `data/raw/sportsbook/nba_moneyline_odds.csv` exists, it will run sportsbook mode and write suffixed outputs such as `backtest_summary_sportsbook.json`.

The single-game pipeline step named "Run realistic bid/ask backtest" does not currently pass `--market-source kalshi`. That creates a risk that a fresh pipeline run will not refresh canonical Kalshi `backtest_summary.json` / `backtest_trades.csv`, or that downstream scripts will keep consuming stale canonical Kalshi outputs.

Recommended fix: make the single-game Kalshi pipeline pass `--market-source kalshi` explicitly in `scripts/run_single_game_research_pipeline.py`.

### 23.2 Batch Candle Downloads Not Fully Wired

`KalshiAPIClient.get_batch_market_candlesticks` exists, and batch payload parsing is tested, but `download_candles_for_matches` still loops per market ticker. This does not violate correctness, but it leaves rate-limit/performance efficiency on the table.

Recommended fix: add a batch download path for eligible ticker groups while preserving per-ticker fallback and cache compatibility.

### 23.3 Single-Game Edge Not Proven

This is the central project state, not a bug.

The proof gates fail on:

- profit
- average CLV
- positive CLV rate
- calibrated strategy readiness
- repeatability months

Any next agent should focus on market truth, calibration, CLV, and feature improvements. Do not relax proof gates to make recommendations appear.

### 23.4 Availability Data Missing

Availability/injury data is a known missing input. The project has templates and feature slots, but the TODO says there are still missing free/manual statuses.

Recommended next work: improve availability input from reliable/free/manual sources and validate that it improves walk-forward CLV, not just classification metrics.

### 23.5 Artifacts May Be Stale Relative to Latest Code

Many `data/reports` files were last updated on 2026-05-13, while cleanup/dashboard artifacts and `outputs` files were updated on 2026-05-17. Because the tree is dirty and the command path has changed, always check artifact timestamps before trusting a metric.

### 23.6 Environment Drift

The default Python on PATH is missing `nba_api`. The project `.venv` is valid. Use `.venv` explicitly for validation or use the provided runner.

### 23.7 Legacy and Current Pipelines Coexist

The modular pipeline and the monolithic `scripts/train_model.py` both produce useful artifacts. Do not assume `outputs/*` is obsolete, but do not treat it as the canonical current pipeline either. The current single-game proof path runs through `scripts/run_single_game_research_pipeline.py` and `data/reports`.

## 24. Recommended Next Work

The correct next goal remains: prove real, repeatable edge on single-game bets.

Recommended order:

1. Fix the `run_backtest.py` market-source hazard by explicitly passing `--market-source kalshi` in the single-game pipeline.
2. Re-run the cached pipeline and confirm canonical `data/reports/backtest_summary.json` updates as a Kalshi candlestick backtest.
3. Add batch candle orchestration to `src/data/kalshi_candles.py` using the existing client method.
4. Improve availability/injury input coverage and verify whether it improves CLV, not only model log loss.
5. Investigate the worst edge-failure slices in `edge_failure_worst_segments.csv`.
6. Continue NO-side calibration research, especially overconfidence and low positive-CLV frequency.
7. Treat `market_anchor_sweep.csv` as a rejection report unless a walk-forward rule shows broad CLV improvement.
8. Keep fair-price recommendations proof-gated to `No bet`.
9. Keep parlay recommendations blocked until proof gates pass.

Avoid these next:

- Do not optimize parlays.
- Do not loosen proof gates just to produce bets.
- Do not add new model types unless they are evaluated by CLV, calibration, and market-priced backtests.
- Do not use last price alone for backtests.

## 25. Quick Start for the Next Agent

Use this sequence to orient:

```powershell
cd "C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor"
.\.venv\Scripts\python.exe -m unittest discover tests
```

Then read:

```text
README.md
TODO.md
CLEANUP_REPORT.md
PROJECT_STATE_REPORT_2026-06-08.md
config.yaml
scripts/run_single_game_research_pipeline.py
scripts/run_backtest.py
src/strategy/backtest.py
src/strategy/fair_price.py
src/strategy/single_game_proof.py
src/reports/coverage.py
src/data/kalshi_candles.py
src/data/kalshi_client.py
src/reports/dashboard.py
```

To run the cached workflow:

```powershell
.\run_cached_pipeline.bat
```

To refresh market pulls:

```powershell
.\run_cached_pipeline.bat -RefreshMarkets
```

To refresh candles too:

```powershell
.\run_cached_pipeline.bat -RefreshMarkets -RefreshCandles
```

To build only the dashboard:

```powershell
.\.venv\Scripts\python.exe scripts\build_dashboard.py
```

Open:

```text
data/reports/dashboard.html
```

The next agent should verify the `run_backtest.py` market-source issue before trusting a fresh full pipeline run.
