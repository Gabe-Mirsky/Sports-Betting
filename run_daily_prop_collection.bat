@echo off
rem Daily multi-sport player-prop snapshot collection + dashboard rebuild.
rem Research-only: no recommendations, approved bets/parlays stay blocked.
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\daily_collect_props.py
.\.venv\Scripts\python.exe scripts\build_dashboard.py
