<#
.SYNOPSIS
    Create/update a Windows Task Scheduler task that runs the NBA game watcher
    every 10 minutes while the computer is awake.

.DESCRIPTION
    The watcher (scripts/nba_game_watcher.py via run_nba_game_watcher.bat) checks
    upcoming/recent NBA games on each tick and decides whether to collect a
    pregame closing snapshot, settle a recently-ended game, or skip. The task
    uses run_nba_game_watcher_hidden.vbs so Task Scheduler does not flash a
    visible Command Prompt or PowerShell window. It is cheap on a skip tick (no
    API calls) and de-duplicates its own actions, so running it every 10 minutes
    is safe.

    Trigger : repeats every 10 minutes, all day, every day.
    Settings: StartWhenAvailable (runs ASAP after a missed start, e.g. the PC was
              asleep), allowed on battery, IgnoreNew (no overlapping instances).
              WakeToRun is intentionally OFF, so it only runs while the computer
              is awake.

    Idempotent: re-running updates the task in place. Current-user task; no admin
    required. Use -Remove to delete it.

    Research-only: scheduling this watcher enables NO models, recommendations,
    approved bets, or approved parlays. It only automates the existing
    research-only collection + settlement steps.

.PARAMETER ProjectRoot
    Project root containing run_nba_game_watcher_hidden.vbs and
    run_nba_game_watcher.bat. Defaults to the parent of this script's folder.

.PARAMETER IntervalMinutes
    Watcher cadence in minutes (default 10).

.PARAMETER Remove
    Unregister the watcher task and exit.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_nba_game_watcher_task.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_nba_game_watcher_task.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [int]$IntervalMinutes = 10,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName = "NBA Game Watcher"

if (-not $ProjectRoot) {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $ProjectRoot = Split-Path -Parent $scriptDir
}
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[removed] $TaskName" -ForegroundColor Yellow
    } else {
        Write-Host "Task '$TaskName' not found; nothing to remove." -ForegroundColor DarkGray
    }
    return
}

$LauncherPath = Join-Path $ProjectRoot "run_nba_game_watcher_hidden.vbs"
$BatPath = Join-Path $ProjectRoot "run_nba_game_watcher.bat"
if (-not (Test-Path $LauncherPath)) {
    Write-Error "Cannot find $LauncherPath - is -ProjectRoot correct?"
    exit 1
}
if (-not (Test-Path $BatPath)) {
    Write-Error "Cannot find $BatPath - the hidden launcher calls this existing batch file."
    exit 1
}

Write-Host "NBA game watcher scheduler setup" -ForegroundColor Cyan
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Launcher     : $LauncherPath"
Write-Host "  Batch file   : $BatPath (manual visible run path)"
Write-Host "  Cadence      : every $IntervalMinutes minute(s), while awake"
Write-Host "  Research-only: no approved bets/parlays are enabled by scheduling." -ForegroundColor DarkGray
Write-Host ""

# wscript.exe runs the tiny VBS launcher without a console window. The launcher
# calls the existing .bat file silently, preserving watcher code and file logs.
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$LauncherPath`"" -WorkingDirectory $ProjectRoot

# Repeat every N minutes, all day, every day. A daily trigger carries the
# repetition borrowed from a one-time trigger (compatible with PowerShell 5.1).
$start = (Get-Date).Date
$trigger = New-ScheduledTaskTrigger -Daily -At $start
$repeat = New-ScheduledTaskTrigger -Once -At $start `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours 24)
$trigger.Repetition = $repeat.Repetition

# StartWhenAvailable => run as soon as possible after a missed start (req #5).
# WakeToRun is deliberately NOT set => only runs while the computer is awake (req #4).
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

$desc = "Every $IntervalMinutes min while awake: hidden launcher decides pregame-collect / settle / skip for NBA games " +
        "(scripts/nba_game_watcher.py). Research-only: no bets, parlays, predictions, or gate changes."

try {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Description $desc -Force | Out-Null
    $verb = if ($existing) { "updated" } else { "created" }
    Write-Host "  [$verb] $TaskName (hidden launcher, every $IntervalMinutes min, StartWhenAvailable on, runs only while awake)" -ForegroundColor Green
} catch {
    Write-Host "  [FAILED] $TaskName : $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "           If this is an access error, retry from an elevated PowerShell." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Verify  : Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "Run hidden: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Run visible manually: .\run_nba_game_watcher.bat"
Write-Host "Dry run  : .\.venv\Scripts\python.exe scripts\nba_game_watcher.py --dry-run"
Write-Host "Logs     : data\logs\nba_watcher\watcher.log  and  run_log.jsonl"
Write-Host "Remove   : powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_nba_game_watcher_task.ps1 -Remove"
