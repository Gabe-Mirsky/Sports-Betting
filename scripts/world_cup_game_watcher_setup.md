# World Cup Game Watcher - setup & how to verify it worked

_Research-only automation. The watcher enables no betting, parlays, model
predictions, approved recommendations, or model gate changes._

## Install the scheduled task

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_world_cup_game_watcher_task.ps1
```

This registers a current-user task **"World Cup Game Watcher"** that:
- repeats every 10 minutes while the computer is awake;
- launches through `run_world_cup_game_watcher_hidden.vbs`, which calls the
  existing `run_world_cup_game_watcher.bat` without flashing a visible Command
  Prompt or PowerShell window;
- keeps the existing watcher logs in `data\logs\world_cup_watcher\watcher.log`
  and `data\logs\world_cup_watcher\run_log.jsonl`;
- leaves all collection logic, model gates, betting, and parlay behavior
  unchanged.

Remove it later with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_world_cup_game_watcher_task.ps1 -Remove
```

## Manual visible run

The scheduled task uses the hidden launcher. If you want to see the console
window while testing manually, run the existing batch file directly:

```powershell
.\run_world_cup_game_watcher.bat --dry-run
```

## Verify

```powershell
Get-ScheduledTask -TaskName "World Cup Game Watcher" | Get-ScheduledTaskInfo
Get-Content data\logs\world_cup_watcher\watcher.log -Tail 20 -Wait
```
