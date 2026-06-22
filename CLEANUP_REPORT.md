# Project Cleanup Report

## Active workflow

The current project is a static NBA/Kalshi prediction dashboard backed by generated CSV and JSON reports. There is no active web backend/API server in the cleaned workflow.

Main command path:

1. `run_cached_pipeline.bat`
2. `run_single_game_pipeline.ps1`
3. `scripts/run_single_game_research_pipeline.py`
4. Data/model/report scripts under `scripts/`
5. Core logic under `src/`
6. Static dashboard build through `scripts/build_dashboard.py`
7. Output dashboard at `data/reports/dashboard.html`

## Files kept

### Frontend/dashboard

- `scripts/build_dashboard.py` - builds the static HTML dashboard.
- `src/reports/dashboard.py` - active dashboard renderer for Upcoming Games, Backtest, and Model Info.
- `data/reports/dashboard.html` - generated dashboard output.

### Pipeline runners

- `run_cached_pipeline.bat` - easiest Windows entry point.
- `run_single_game_pipeline.ps1` - PowerShell runner with local Python fallback.
- `scripts/run_single_game_research_pipeline.py` - active end-to-end single-game research pipeline.
- `scripts/run_full_pipeline.py` - older full pipeline runner kept because it may still be useful for manual research.

### NBA and Kalshi data collection

- `src/data/` - active loaders, NBA clients, Kalshi clients, matching, candles, public backfill, market quality, validation, and team aliases.
- `scripts/download_*.py`, `scripts/kalshi_*.py`, `scripts/discover_kalshi_nba_markets.py`, `scripts/market_truth_audit.py` - active/manual data and audit commands.

### Modeling, predictions, strategy, and backtests

- `src/features/`, `src/models/`, `src/strategy/` - active feature, model, CLV, fair-price, proof-gate, and backtest logic.
- `scripts/train.py`, `scripts/tune_model.py`, `scripts/walk_forward.py`, `scripts/run_backtest.py`, `scripts/predict_upcoming.py`, `scripts/build_fair_prices.py`, and related audit/sweep scripts - current research and dashboard inputs.

### Data and model artifacts

- `data/raw/`, `data/interim/`, `data/processed/`, `data/models/`, `data/kalshi/`, and active `data/reports/` outputs were kept.
- Raw historical NBA/Kalshi files, model files, processed features, report CSVs/JSONs, and dashboard input files were not deleted.

### Config, docs, and tests

- `.env`, `.env.example`, `.secrets/`, `.streamlit/`, `.venv/`, `config.yaml`, `requirements.txt`, `README.md`, `TODO.md`, and tests under `tests/` were kept.
- `tests/test_dashboard.py` was updated to validate the simplified dashboard tabs and confirm the removed Best Spots tab does not return.

## Files archived

Archived files were moved to `_archive_unused_files/` instead of being deleted.

- `_archive_unused_files/old_streamlit_dashboard/`
  - `requirements-dashboard.txt`
  - `run_dashboard.py`
  - `dashboard_app.py`
  - `interactive_dashboard.py`
  - `test_interactive_dashboard.py`
  - Reason: old Streamlit dashboard path is no longer referenced by README or the active static dashboard workflow.

- `_archive_unused_files/old_exploration_notebooks/notebooks/`
  - `01_explore_nba_data.ipynb`
  - `02_feature_debugging.ipynb`
  - `03_model_evaluation.ipynb`
  - Reason: placeholder/old exploration notebooks, not part of the current repeatable pipeline.

- `_archive_unused_files/generated_test_outputs/`
  - `_test_event_discovery/`
  - `_test_kalshi_candles/`
  - `_test_kalshi_series_backfill/`
  - `kalshi_public_default_test_matches.csv`
  - `kalshi_public_default_test_needs_review.csv`
  - `kalshi_public_test_matches.csv`
  - `kalshi_public_test_needs_review.csv`
  - Reason: generated test output artifacts, not raw source data.

- `_archive_unused_files/probe_outputs/`
  - `clv_filtered_side_audit_best_probe.csv`
  - `clv_filtered_side_audit_loose_probe.csv`
  - `clv_filtered_side_audit_relaxed_probe.csv`
  - `clv_filtered_summary_best_probe.json`
  - `clv_filtered_summary_loose_probe.json`
  - `clv_filtered_summary_relaxed_probe.json`
  - `clv_filtered_trades_best_probe.csv`
  - `clv_filtered_trades_loose_probe.csv`
  - `clv_filtered_trades_relaxed_probe.csv`
  - Reason: one-off CLV probe outputs superseded by the canonical report files.

- `_archive_unused_files/nested_duplicate_project/Prediction Market Project/`
  - Reason: nested duplicate project folder found inside the active project. The remaining active tree is `nba_kalshi_predictor`.

## Files deleted

Only obvious junk was deleted.

- `.codex_runtime_pkgs` - empty temporary runtime folder.
- `tmp7bz6kdxk` - temporary folder.
- `tmpjyoz8jh5` - temporary folder.
- `tmpt6fkv2sw` - temporary folder.
- `scripts/__pycache__/`
- `src/__pycache__/`
- `src/data/__pycache__/`
- `src/features/__pycache__/`
- `src/models/__pycache__/`
- `src/reports/__pycache__/`
- `src/strategy/__pycache__/`
- `tests/__pycache__/`

Note: Python tests recreate `__pycache__` folders during validation. They can be safely deleted again any time.

## Files kept but questionable

- `.pip_tmp/` - looks temporary, but Windows denied deletion.
- `tmpmvt2s4u0/` - looks temporary, but Windows denied deletion.
- `.streamlit/` - kept because it is config-like and harmless, even though the active dashboard is now static HTML.
- `.venv/` - kept even though the local executable has had issues; dependencies are still useful and should not be deleted during cleanup.
- `data/kalshi/markets_mock.csv` and `data/kalshi/markets_template.csv` - kept because they may still be useful manual fixtures/templates.
- Older reports and research scripts under `data/reports/` and `scripts/` - kept unless clearly generated test/probe output, because they feed audits or document historical model work.

## Validation

Commands run:

```powershell
$env:PYTHONPATH='C:\Users\arilo\Downloads\Prediction Market Project\nba_kalshi_predictor\.venv\Lib\site-packages'
& 'C:\Users\arilo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\build_dashboard.py
```

```powershell
node -e "const fs=require('fs'); const html=fs.readFileSync('data/reports/dashboard.html','utf8'); const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]); for (const s of scripts) new Function(s); if(html.includes('data-tab=' + String.fromCharCode(34) + 'best' + String.fromCharCode(34)) || html.includes('Best Spots')) throw new Error('Best Spots still present'); console.log('dashboard scripts parse ok:', scripts.length);"
```

```powershell
$env:PYTHONPATH='C:\Users\arilo\Downloads\Prediction Market Project\nba_kalshi_predictor\.venv\Lib\site-packages'
& 'C:\Users\arilo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover tests
```

Result: dashboard build passed, dashboard JavaScript parsed, Best Spots checks passed, and 176 unit tests passed.

## How to run the cleaned project

From:

```powershell
cd "C:\Users\arilo\Downloads\Prediction Market Project\nba_kalshi_predictor"
```

Run the cached end-to-end pipeline:

```powershell
.\run_cached_pipeline.bat
```

Build only the dashboard:

```powershell
python scripts\build_dashboard.py
```

Open:

```powershell
.\data\reports\dashboard.html
```

If the activated `.venv` Python executable fails, use `run_cached_pipeline.bat`; it already works around that issue.

## What changed

- Removed the old Streamlit dashboard path from active docs and archived its files.
- Kept the current static dashboard as the single clear user-facing dashboard.
- Removed the Best Spots test expectation and replaced it with checks for Upcoming Games, Backtest, and Model Info.
- Archived old notebooks, generated test outputs, CLV probe outputs, and a nested duplicate project folder.
- Deleted safe temporary/cache junk where Windows allowed it.
- Left raw data, models, credentials/config, and active report artifacts in place.
