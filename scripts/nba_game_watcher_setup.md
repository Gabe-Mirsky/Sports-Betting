# NBA Game Watcher — setup & how to verify it worked

_Research-only automation. The watcher enables **no** betting, parlays, model
predictions, or approved recommendations, and changes **no** model gate. It only
automates the two existing research-only steps: pregame snapshot collection and
result settlement._

## What it does

`scripts/nba_game_watcher.py` (run via `run_nba_game_watcher.bat`) is meant to run
every ~10 minutes while your PC is awake. On each tick it reads the known NBA
games (from `data/processed/player_prop_snapshots_normalized.csv`, via the
existing `nba_collection_planner`) and decides **per game**:

| Decision | When | What it runs |
| --- | --- | --- |
| **PREGAME** | a game tips within the next **60 min** (10-min post-tip grace) | `run_nba_pregame_prop_collection.bat` (once per game) |
| **SETTLE** | a game started **180–420 min** ago (so it has ended) | `.\.venv\Scripts\python.exe scripts\refresh_nba_results_and_settle_props.py --download` (once per game) |
| **SKIP** | nothing is due | nothing (no API calls — cheap) |

**Duplicate prevention (req #3):** every fired action is appended to
`data/logs/nba_watcher/run_log.jsonl` with `game_id`, `action`, `status`, and
`timestamp_utc`. A game's action is done after **one success**; failures (e.g. a
transient Odds API 429 or a box score that posted late) are retried up to
`--max-attempts` (default 3). A lock file plus Task Scheduler's `IgnoreNew`
prevents overlapping instances.

## Install the scheduled task (every 10 min, while awake)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_nba_game_watcher_task.ps1
```

This registers a current-user task **"NBA Game Watcher"** that:
- repeats every 10 minutes, all day, every day;
- launches through `run_nba_game_watcher_hidden.vbs`, which calls the existing
  `run_nba_game_watcher.bat` without flashing a visible Command Prompt or
  PowerShell window;
- has **StartWhenAvailable** on → runs as soon as possible after a missed start
  (e.g. the PC was asleep at the scheduled minute) — requirement #5;
- does **not** set WakeToRun → only runs while the computer is awake —
  requirement #4;
- allows running on battery and ignores overlapping instances.

Remove it later with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_nba_game_watcher_task.ps1 -Remove
```

## Manual visible run

The scheduled task uses the hidden launcher. If you want to see the console
window while testing manually, run the existing batch file directly:

```powershell
.\run_nba_game_watcher.bat --dry-run
```

## How to verify it worked

### 1. Dry run (no side effects)
```powershell
.\.venv\Scripts\python.exe scripts\nba_game_watcher.py --dry-run
```
You should see one line per tick, e.g.
`PREGAME 1 game(s) within 60.0m: NYK @ SAS (2026-06-13) (T-39m)` or
`SKIP no NBA game within 60m and none recently ended; next tip in 312m`.
A dry run **never** executes the bat/refresh and **never** writes the run-log.

### 2. Confirm the scheduled task exists and is healthy
```powershell
Get-ScheduledTask -TaskName "NBA Game Watcher" | Get-ScheduledTaskInfo
```
Check `LastRunTime`, `LastTaskResult` (0 = success), and `NextRunTime`.

### 3. Watch the logs
- **Human log** — `data\logs\nba_watcher\watcher.log`: one timestamped line per
  tick (`SKIP` / `PREGAME` / `SETTLE` / `ERROR`). Tail it:
  ```powershell
  Get-Content data\logs\nba_watcher\watcher.log -Tail 20 -Wait
  ```
- **Run-log (dedup)** — `data\logs\nba_watcher\run_log.jsonl`: one JSON line per
  fired action with `game_id`, `action`, `status`, `return_code`, `timestamp_utc`.
- **Latest status** — `data\reports\nba_watcher_status.json`: the most recent
  tick's decision (games considered, what fired, minutes to next tip).

### 4. Confirm de-duplication
After a **PREGAME** fires successfully for a game, the next ticks for that same
game should log `SKIP` (or simply not re-fire it) — there will be exactly one
`"action":"pregame","status":"success"` line for that `game_id` in
`run_log.jsonl`.

### 5. Confirm a real result
- After a pregame fire: new NBA rows appear in
  `data/processed/player_prop_snapshots_normalized.csv`, and
  `data/reports/nba_prop_closing_collection_plan.json` shows the `60m_before` /
  `30m_before` window as `hit`.
- After a settle fire: `data/reports/player_prop_settlement_outcomes_summary.json`
  `settled_props` increases and `pending_props` drops for that game.

### 6. Run the unit tests for the dedup logic
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_nba_game_watcher -v
```

## Tuning (optional)
All thresholds are CLI flags (defaults shown): `--pregame-window-min 60`,
`--pregame-grace-min 10`, `--settle-after-min 180`, `--settle-lookback-min 420`,
`--max-attempts 3`. Edit `run_nba_game_watcher.bat` to pass overrides, e.g.
`... scripts\nba_game_watcher.py --pregame-window-min 45`.

## Relationship to the existing fixed-time tasks
The 7 fixed-time **NBA Pregame Prop Collection** tasks (18:00–22:00 ET) still run.
This watcher is the **awake-time safety net**: it catches game times the fixed
slots miss and handles settlement automatically. Both are research-only.
