@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\open_dashboard.py"
  if %ERRORLEVEL% EQU 0 exit /b 0
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py "scripts\open_dashboard.py"
  if %ERRORLEVEL% EQU 0 exit /b 0
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  python "scripts\open_dashboard.py"
  if %ERRORLEVEL% EQU 0 exit /b 0
)

if exist "data\reports\dashboard.html" (
  start "" "data\reports\dashboard.html"
  exit /b 0
)

echo Could not find Python or an existing dashboard file.
echo Try running: python scripts\build_dashboard.py
pause
exit /b 1
