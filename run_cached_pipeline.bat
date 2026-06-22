@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_single_game_pipeline.ps1" %*
exit /b %ERRORLEVEL%
