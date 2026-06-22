@echo off
rem Startup-friendly prop collection: collect snapshots, settle NBA results,
rem rebuild the health report and dashboard, and write everything to one log.
rem Safe to schedule at Windows login/startup (Task Scheduler: "At log on").
rem Research-only: no recommendations; approved bets/parlays stay blocked.

cd /d "%~dp0"

if not exist "data\logs\startup_runs" mkdir "data\logs\startup_runs"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"
set "LOG=data\logs\startup_runs\startup_%STAMP%.log"

echo ================================================================ > "%LOG%"
echo Startup prop collection run %STAMP% (local time: %date% %time%) >> "%LOG%"
echo ================================================================ >> "%LOG%"

echo. >> "%LOG%"
echo [1/4] daily_collect_props.py >> "%LOG%"
.\.venv\Scripts\python.exe scripts\daily_collect_props.py >> "%LOG%" 2>&1
echo [1/4] exit code: %errorlevel% >> "%LOG%"

echo. >> "%LOG%"
echo [2/4] refresh_nba_results_and_settle_props.py >> "%LOG%"
.\.venv\Scripts\python.exe scripts\refresh_nba_results_and_settle_props.py >> "%LOG%" 2>&1
echo [2/4] exit code: %errorlevel% >> "%LOG%"

echo. >> "%LOG%"
echo [3/4] build_prop_collection_health.py >> "%LOG%"
.\.venv\Scripts\python.exe scripts\build_prop_collection_health.py >> "%LOG%" 2>&1
echo [3/4] exit code: %errorlevel% >> "%LOG%"

echo. >> "%LOG%"
echo [4/4] build_dashboard.py >> "%LOG%"
.\.venv\Scripts\python.exe scripts\build_dashboard.py >> "%LOG%" 2>&1
echo [4/4] exit code: %errorlevel% >> "%LOG%"

echo. >> "%LOG%"
echo Done. Health report: data\reports\prop_collection_health.md >> "%LOG%"
echo Dashboard: data\reports\dashboard.html >> "%LOG%"
