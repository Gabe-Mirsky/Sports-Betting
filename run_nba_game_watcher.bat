@echo off
rem Awake-time NBA game watcher: every ~10 min, decide whether to collect a
rem pregame closing snapshot or settle a recently-ended game. Cheap when nothing
rem is due (no API calls on a SKIP tick).
rem Research-only: no recommendations, approved bets/parlays stay blocked.
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\nba_game_watcher.py %*
