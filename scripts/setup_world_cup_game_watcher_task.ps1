<#
.SYNOPSIS
    Create/update a Windows Task Scheduler task that runs the World Cup odds
    watcher every 10 minutes while the computer is awake.

.DESCRIPTION
    The watcher (scripts/world_cup_game_watcher.py via run_world_cup_game_watcher.bat)
    lists FIFA World Cup events with the free /events endpoint and decides per
    match whether to collect a closing-like snapshot, fetch results, or skip.
    The task uses run_world_cup_game_watcher_hidden.vbs so Task Scheduler does
    not flash a visible Command Prompt or PowerShell window. A skip tick spends
    0 credits; a strict quota guard protects the NBA budget.

    Trigger : repeats every 10 minutes, all day, every day.
    Settings: StartWhenAvailable (runs ASAP after a missed start), allowed on
              battery, IgnoreNew (no overlapping instances). WakeToRun is OFF, so
              it only runs while the computer is awake.

    Idempotent (Register-ScheduledTask -Force). Current-user task; no admin
    required. Use -Remove to delete it. This is completely separate from the
    "NBA Game Watcher" task, which is left untouched.

    Research-only: enables NO models, recommendations, predictions, approved
    bets, or approved parlays.

.PARAMETER ProjectRoot
    Project root containing run_world_cup_game_watcher_hidden.vbs and
    run_world_cup_game_watcher.bat.

.PARAMETER IntervalMinutes
    Cadence in minutes (default 10).

.PARAMETER Early
    Pass --early to the watcher so it also collects 24-48h opening-line snapshots
    (uses extra credits). Off by default to conserve quota.

.PARAMETER Remove
    Unregister the task and exit.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_world_cup_game_watcher_task.ps1
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [int]$IntervalMinutes = 10,
    [switch]$Early,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName = "World Cup Game Watcher"

if (-not $ProjectRoot) {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $ProjectRoot = Split-Path -Parent $scriptDir
}
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[removed] $TaskName" -ForegroundColor Yellow
    } else {
        Write-Host "Task '$TaskName' not found; nothing to remove." -ForegroundColor DarkGray
    }
    return
}

$LauncherPath = Join-Path $ProjectRoot "run_world_cup_game_watcher_hidden.vbs"
$BatPath = Join-Path $ProjectRoot "run_world_cup_game_watcher.bat"
if (-not (Test-Path $LauncherPath)) {
    Write-Error "Cannot find $LauncherPath - is -ProjectRoot correct?"
    exit 1
}
if (-not (Test-Path $BatPath)) {
    Write-Error "Cannot find $BatPath - the hidden launcher calls this existing batch file."
    exit 1
}

# wscript.exe runs the tiny VBS launcher without a console window. The launcher
# calls the existing .bat file silently, preserving watcher code and file logs.
# Optionally append --early as a watcher argument.
$launcherArg = if ($Early) { "`"$LauncherPath`" --early" } else { "`"$LauncherPath`"" }
$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument $launcherArg -WorkingDirectory $ProjectRoot

$start = (Get-Date).Date
$trigger = New-ScheduledTaskTrigger -Daily -At $start
$repeat = New-ScheduledTaskTrigger -Once -At $start `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours 24)
$trigger.Repetition = $repeat.Repetition

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

$desc = "Every $IntervalMinutes min while awake: hidden launcher for World Cup closing-snapshot / results / skip " +
        "(scripts/world_cup_game_watcher.py). Research-only: no bets, parlays, predictions, or gate changes."

Write-Host "World Cup game watcher scheduler setup" -ForegroundColor Cyan
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Launcher     : $LauncherPath"
Write-Host "  Batch file   : $BatPath (manual visible run path)"
Write-Host "  Cadence      : every $IntervalMinutes minute(s), while awake"
Write-Host "  Early snaps  : $([bool]$Early)"
Write-Host ""

try {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Description $desc -Force | Out-Null
    $verb = if ($existing) { "updated" } else { "created" }
    Write-Host "  [$verb] $TaskName (hidden launcher, every $IntervalMinutes min, StartWhenAvailable on, awake-only)" -ForegroundColor Green
} catch {
    Write-Host "  [FAILED] $TaskName : $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Verify    : Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "Run hidden: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Run visible manually: .\run_world_cup_game_watcher.bat"
Write-Host "Dry run   : .\.venv\Scripts\python.exe scripts\world_cup_game_watcher.py --dry-run"
Write-Host "Logs      : data\logs\world_cup_watcher\watcher.log  and  run_log.jsonl"
Write-Host "Remove    : powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_world_cup_game_watcher_task.ps1 -Remove"
