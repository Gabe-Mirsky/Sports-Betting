@echo off
rem Master daily prop pipeline: collect, enrich, settle (cache-only), rebuild
rem every report, and refresh the dashboard. One entry point for scheduling.
rem Research-only: no recommendations, approved bets/parlays stay blocked.
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\run_full_prop_pipeline.py
