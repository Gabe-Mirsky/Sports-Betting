@echo off
rem Refresh NBA results (cache-only), settle pending player props, rebuild dashboard.
rem Pass --download as an argument to re-fetch nba_api caches first.
rem Research-only: no recommendations, approved bets/parlays stay blocked.
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\refresh_nba_results_and_settle_props.py %*
.\.venv\Scripts\python.exe scripts\build_dashboard.py
