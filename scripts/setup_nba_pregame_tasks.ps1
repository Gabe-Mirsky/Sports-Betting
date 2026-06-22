<#
.SYNOPSIS
    Create/update Windows Task Scheduler tasks that run NBA pregame prop
    collection at fixed evening times (Eastern), so closing-like snapshots get
    captured near tip-off.

.DESCRIPTION
    Creates one scheduled task per Eastern-time slot:
        6:00 PM, 6:30 PM, 7:00 PM, 7:30 PM, 8:30 PM, 9:30 PM, 10:00 PM ET

    Each task runs run_nba_pregame_prop_collection.bat via cmd.exe /c with the
    full path quoted (spaces in the project path are handled) and the
    WorkingDirectory set to the project root.

    Idempotent: re-running updates existing tasks in place (Register-ScheduledTask
    -Force). Does not require admin for current-user tasks.

    Also repairs the known-broken "Player Prop Collection Every 4 Hours" task if
    its action was registered with an unquoted path (Execute ends up as
    "C:\Users\...\Python" and fails with 0x80070002).

    Research-only: scheduling collection does NOT enable models, recommendations,
    approved bets, or approved parlays.

.PARAMETER ProjectRoot
    Project root containing run_nba_pregame_prop_collection.bat.
    Defaults to the parent of this script's folder.

.PARAMETER SkipEvery4hRepair
    Skip the repair of the broken every-4-hours collection task.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_nba_pregame_tasks.ps1
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [switch]$SkipEvery4hRepair
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot is not available inside param() defaults on Windows PowerShell 5.1.
if (-not $ProjectRoot) {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $ProjectRoot = Split-Path -Parent $scriptDir
}

$TaskPrefix = "NBA Pregame Prop Collection"
# Eastern-time slots (24h). Converted to local machine time below.
$EasternSlots = @("18:00", "18:30", "19:00", "19:30", "20:30", "21:30", "22:00")

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$BatPath = Join-Path $ProjectRoot "run_nba_pregame_prop_collection.bat"
if (-not (Test-Path $BatPath)) {
    Write-Error "Cannot find $BatPath - is -ProjectRoot correct?"
    exit 1
}

Write-Host "NBA pregame prop collection scheduler setup" -ForegroundColor Cyan
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Batch file   : $BatPath"
Write-Host "  Research-only: no approved bets/parlays are enabled by scheduling." -ForegroundColor DarkGray
Write-Host ""

# --- Convert Eastern slots to local time (handles DST via TimeZoneInfo) -----
$eastern = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$local = [System.TimeZoneInfo]::Local
if ($local.Id -ne $eastern.Id) {
    Write-Host "  Local timezone is '$($local.Id)'; converting ET slots to local time." -ForegroundColor Yellow
}

function Convert-EasternToLocal([string]$hhmm) {
    $parts = $hhmm.Split(":")
    $today = (Get-Date).Date
    $easternNaive = $today.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
    $easternNaive = [DateTime]::SpecifyKind($easternNaive, [DateTimeKind]::Unspecified)
    return [System.TimeZoneInfo]::ConvertTime($easternNaive, $eastern, $local)
}

# --- Shared task pieces ------------------------------------------------------
# cmd.exe /c with the .bat path quoted: this is the pattern the (working)
# login task uses and it survives the space in "Python Projects".
$actionArgs = "/c `"`"$BatPath`"`""

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$created = 0
$updated = 0
$failed = 0

foreach ($slot in $EasternSlots) {
    $localTime = Convert-EasternToLocal $slot
    $slotLabel = $slot.Replace(":", "")
    $taskName = "$TaskPrefix $($slotLabel)ET"

    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $actionArgs -WorkingDirectory $ProjectRoot
    $trigger = New-ScheduledTaskTrigger -Daily -At $localTime

    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    try {
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
            -Settings $settings -Description (
                "Collects NBA player-prop closing-like snapshots near tip-off ($slot ET daily). " +
                "Runs run_nba_pregame_prop_collection.bat. Research-only: no bets, no recommendations."
            ) -Force | Out-Null
        if ($existing) {
            $updated++
            Write-Host ("  [updated] {0,-40} daily at {1} local ({2} ET)" -f $taskName, $localTime.ToString("HH:mm"), $slot) -ForegroundColor Green
        } else {
            $created++
            Write-Host ("  [created] {0,-40} daily at {1} local ({2} ET)" -f $taskName, $localTime.ToString("HH:mm"), $slot) -ForegroundColor Green
        }
    } catch {
        $failed++
        Write-Host "  [FAILED ] $taskName : $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "            If this is an access error, retry from an elevated PowerShell." -ForegroundColor Yellow
    }
}

# --- Repair the broken every-4-hours task (unquoted-path bug) ---------------
if (-not $SkipEvery4hRepair) {
    $every4hName = "Player Prop Collection Every 4 Hours"
    $every4h = Get-ScheduledTask -TaskName $every4hName -ErrorAction SilentlyContinue
    if ($every4h) {
        $exec = $every4h.Actions[0].Execute
        $actionBroken = ($exec -notmatch "cmd(\.exe)?$") -and (-not (Test-Path -LiteralPath $exec -ErrorAction SilentlyContinue))
        $settingsWeak = $every4h.Settings.DisallowStartIfOnBatteries -or (-not $every4h.Settings.StartWhenAvailable)
        if ($actionBroken -or $settingsWeak) {
            Write-Host ""
            if ($actionBroken) {
                Write-Host "Repairing '$every4hName' (action was '$exec' - path split on space, runs fail with 0x80070002)..." -ForegroundColor Yellow
            } else {
                Write-Host "Updating '$every4hName' settings (allow battery, catch up missed starts)..." -ForegroundColor Yellow
            }
            $dailyBat = Join-Path $ProjectRoot "run_daily_prop_collection.bat"
            $fixedAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"`"$dailyBat`"`"" -WorkingDirectory $ProjectRoot
            try {
                Set-ScheduledTask -TaskName $every4hName -Action $fixedAction -Settings $settings | Out-Null
                Write-Host "  [repaired] $every4hName now runs cmd.exe /c `"$dailyBat`" (battery OK, StartWhenAvailable on)" -ForegroundColor Green
            } catch {
                Write-Host "  [FAILED ] Could not repair $every4hName : $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            Write-Host ""
            Write-Host "'$every4hName' action and settings look OK; no repair needed." -ForegroundColor DarkGray
        }
    }
}

Write-Host ""
Write-Host ("Done: {0} created, {1} updated, {2} failed." -f $created, $updated, $failed) -ForegroundColor Cyan
Write-Host "Verify with: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_scheduled_tasks.ps1"
Write-Host "Remove with: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\remove_nba_pregame_tasks.ps1"
if ($failed -gt 0) { exit 1 }
