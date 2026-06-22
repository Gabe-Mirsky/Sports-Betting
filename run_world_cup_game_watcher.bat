@echo off
rem Awake-time World Cup (FIFA) odds watcher: every ~10 min, decide whether to
rem collect a closing-like game-market snapshot, fetch results, or skip. Cheap on
rem a skip tick (lists events via the free /events endpoint, 0 credits). A strict
rem quota guard protects the NBA Odds API budget.
rem Research-only: no recommendations, predictions, approved bets/parlays.
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\world_cup_game_watcher.py %*
