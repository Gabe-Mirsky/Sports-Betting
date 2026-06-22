<#
.SYNOPSIS
    Verify the prop-collection scheduled tasks: existence, last result,
    next run, command, working directory, and common misconfigurations.

.DESCRIPTION
    Checks:
      - "Player Prop Collection Every 4 Hours" exists
      - "Player Prop Collection At Login" exists
      - the seven "NBA Pregame Prop Collection *ET" tasks exist
      - last run result (warns on nonzero)
      - next run time
      - task command + working directory
      - action paths that no longer exist (the unquoted-path bug)
      - battery / missed-start settings that block unattended collection

    Exit code 0 = all present and healthy, 1 = warnings found.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_scheduled_tasks.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$PregameSlots = @("1800", "1830", "1900", "1930", "2030", "2130", "2200")
$ExpectedTasks = @("Player Prop Collection Every 4 Hours", "Player Prop Collection At Login", "NBA Game Watcher")
$ExpectedTasks += $PregameSlots | ForEach-Object { "NBA Pregame Prop Collection $($_)ET" }

$warnings = New-Object System.Collections.Generic.List[string]

function Format-TaskResult([uint32]$code) {
    switch ($code) {
        0          { return "0 (success)" }
        1          { return "1 (incorrect call or generic failure)" }
        267009     { return "267009 (currently running)" }
        267011     { return "267011 (has not yet run)" }
        2147942402 { return "0x80070002 (file not found - check the action path/quoting)" }
        2147942405 { return "0x80070005 (access denied)" }
        default    { return ("0x{0:X8}" -f $code) }
    }
}

Write-Host "Prop collection scheduled task verification" -ForegroundColor Cyan
Write-Host ("Checked at: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"))
Write-Host ""

foreach ($name in $ExpectedTasks) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "[MISSING] $name" -ForegroundColor Red
        if ($name -like "NBA Pregame*") {
            $warnings.Add("$name is missing - run scripts\setup_nba_pregame_tasks.ps1")
        } else {
            $warnings.Add("$name is missing - see scripts\windows_task_scheduler_setup.md")
        }
        continue
    }

    $info = Get-ScheduledTaskInfo -TaskName $name
    $action = $task.Actions[0]
    $command = ("{0} {1}" -f $action.Execute, $action.Arguments).Trim()
    $workDir = if ($action.WorkingDirectory) { $action.WorkingDirectory } else { "(not set)" }
    $lastResult = Format-TaskResult $info.LastTaskResult
    $nextRun = if ($info.NextRunTime) { $info.NextRunTime.ToString("yyyy-MM-dd HH:mm") } else { "(none scheduled)" }
    $lastRun = if ($info.LastRunTime -and $info.LastRunTime.Year -gt 2000) { $info.LastRunTime.ToString("yyyy-MM-dd HH:mm") } else { "(never)" }

    $healthy = $true

    # Action sanity: the Execute must be cmd/powershell or an existing file.
    $exec = $action.Execute.Trim('"')
    $isShell = $exec -match "(cmd|powershell|pwsh)(\.exe)?$"
    if (-not $isShell -and -not (Test-Path -LiteralPath $exec -ErrorAction SilentlyContinue)) {
        $healthy = $false
        $warnings.Add("$name action executable '$exec' does not exist (unquoted path with spaces?). Fix: scripts\setup_nba_pregame_tasks.ps1 repairs the every-4-hours task.")
    }
    if ($info.LastTaskResult -ne 0 -and $info.LastTaskResult -ne 267011 -and $info.LastTaskResult -ne 267009) {
        $healthy = $false
        $warnings.Add("$name last run failed with $lastResult")
    }
    if ($task.State -eq "Disabled") {
        $healthy = $false
        $warnings.Add("$name is disabled")
    }
    if ($task.Settings.DisallowStartIfOnBatteries) {
        $warnings.Add("$name will not start on battery (DisallowStartIfOnBatteries=true) - evening runs on a laptop may be skipped")
    }
    if (-not $task.Settings.StartWhenAvailable -and $name -notlike "*At Login*") {
        $warnings.Add("$name will not catch up after a missed start (StartWhenAvailable=false)")
    }

    $status = if ($healthy) { "[OK]     " } else { "[WARN]   " }
    $color = if ($healthy) { "Green" } else { "Yellow" }
    Write-Host "$status$name" -ForegroundColor $color
    Write-Host "          state    : $($task.State)"
    Write-Host "          command  : $command"
    Write-Host "          work dir : $workDir"
    Write-Host "          last run : $lastRun -> $lastResult"
    Write-Host "          next run : $nextRun"
}

Write-Host ""
if ($warnings.Count -gt 0) {
    Write-Host "Warnings ($($warnings.Count)):" -ForegroundColor Yellow
    foreach ($w in $warnings) { Write-Host "  - $w" -ForegroundColor Yellow }
    exit 1
} else {
    Write-Host "All expected tasks exist and look healthy." -ForegroundColor Green
    exit 0
}
