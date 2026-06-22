@echo off
rem NBA pregame prop collection chain: collect near tip-off, enrich, refresh
rem settlement (cache-only), rebuild quality/health/CLV-readiness reports and
rem the dashboard. Schedule this in the evenings around NBA tip times (see
rem scripts\nba_near_tip_collection_schedule.md).
rem Research-only: no recommendations, approved bets/parlays stay blocked.
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\daily_collect_props.py
.\.venv\Scripts\python.exe scripts\enrich_player_prop_snapshots.py
.\.venv\Scripts\python.exe scripts\refresh_nba_results_and_settle_props.py
.\.venv\Scripts\python.exe scripts\build_player_prop_market_quality.py
.\.venv\Scripts\python.exe scripts\build_player_prop_manual_review.py
.\.venv\Scripts\python.exe scripts\build_prop_collection_health.py
.\.venv\Scripts\python.exe scripts\build_nba_collection_plan.py
.\.venv\Scripts\python.exe scripts\build_player_prop_settlement_outcomes.py
.\.venv\Scripts\python.exe scripts\build_player_prop_clv.py
.\.venv\Scripts\python.exe scripts\build_player_prop_data_quality_gates.py
.\.venv\Scripts\python.exe scripts\build_dashboard.py
