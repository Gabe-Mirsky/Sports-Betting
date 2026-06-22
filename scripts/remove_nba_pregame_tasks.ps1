<#
.SYNOPSIS
    Remove the NBA pregame prop collection scheduled tasks created by
    scripts\setup_nba_pregame_tasks.ps1.

.DESCRIPTION
    Only removes tasks whose names start with "NBA Pregame Prop Collection".
    Leaves the every-4-hours and at-login collection tasks untouched.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\remove_nba_pregame_tasks.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$TaskPrefix = "NBA Pregame Prop Collection"

$tasks = Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object { $_.TaskName -like "$TaskPrefix*" }
if (-not $tasks) {
    Write-Host "No tasks starting with '$TaskPrefix' found; nothing to remove." -ForegroundColor Yellow
    exit 0
}

$removed = 0
$failed = 0
foreach ($task in $tasks) {
    try {
        Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
        $removed++
        Write-Host "  [removed] $($task.TaskName)" -ForegroundColor Green
    } catch {
        $failed++
        Write-Host "  [FAILED ] $($task.TaskName) : $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host ("Done: {0} removed, {1} failed." -f $removed, $failed) -ForegroundColor Cyan
Write-Host "The every-4-hours and at-login collection tasks were NOT touched."
if ($failed -gt 0) { exit 1 }
