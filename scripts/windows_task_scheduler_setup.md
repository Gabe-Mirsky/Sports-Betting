# Scheduling the Prop Collector on Windows (Task Scheduler)

This guide schedules `run_daily_prop_collection.bat`, which runs:

1. `.\.venv\Scripts\python.exe scripts\daily_collect_props.py` — collect multi-sport player-prop snapshots
2. `.\.venv\Scripts\python.exe scripts\build_dashboard.py` — rebuild the local dashboard

Everything is research-only. Scheduling collection does **not** enable models,
recommendations, approved bets, or parlays.

## Prerequisites

- The repo lives at `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor`
  (adjust paths below if you move it).
- `ODDS_API_KEY` must be set as a **user environment variable** so the scheduled
  task can see it (Task Scheduler does not load your PowerShell profile):
  - Settings → System → About → Advanced system settings → Environment Variables →
    *User variables* → New → name `ODDS_API_KEY`, value = your key. Or in PowerShell:

    ```powershell
    [Environment]::SetEnvironmentVariable("ODDS_API_KEY", "your-key-here", "User")
    ```

  - Without the key, the run still succeeds — every Odds API league is recorded
    as `skipped_no_api_key` in the run summary.
- Free tier is ~500 requests/month and player-prop markets cost extra credits.
  The per-run request cap lives in `config/prop_collection.yaml`
  (`defaults.max_events_per_league_per_run`). With 6 leagues × 6 events you can
  burn quota fast; lower the cap or disable leagues if you schedule frequent runs.

## Option A: GUI (Task Scheduler app)

1. Start → search **Task Scheduler** → *Create Task…* (not "Basic Task" if you want multiple triggers).
2. **General tab**: Name `NBA Prop Collection`. Select *Run whether user is logged on or not*
   if you want it to run unattended (requires your Windows password).
3. **Triggers tab**: add one or more triggers (examples below).
4. **Actions tab** → New:
   - Program/script: `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_daily_prop_collection.bat`
   - Start in: `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor`
     (the .bat also `cd`s to its own folder, but setting *Start in* avoids surprises).
5. **Settings tab**: check *Run task as soon as possible after a scheduled start is missed*
   so a sleeping laptop catches up.

## Option B: PowerShell one-liners (`schtasks`)

Run these in a normal PowerShell window. `schtasks` wants the full path quoted.

### Run every morning (9:00 AM daily)

```powershell
schtasks /Create /TN "PropCollection-Morning" /SC DAILY /ST 09:00 `
  /TR "\"C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_daily_prop_collection.bat\""
```

### Run every 4 hours

```powershell
schtasks /Create /TN "PropCollection-Every4h" /SC HOURLY /MO 4 /ST 08:00 `
  /TR "\"C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_daily_prop_collection.bat\""
```

(Every-4-hours uses more Odds API quota — consider lowering
`max_events_per_league_per_run` or disabling collect-only leagues.)

### Run ~1 hour before games (closing snapshots)

Task Scheduler cannot read game schedules, so approximate with fixed evening
times around typical US tip-offs (ET). NBA games mostly start 7:00–10:30 PM ET,
so schedule runs at 6:00 PM, 6:30 PM, and 9:30 PM local (adjust for your zone):

```powershell
schtasks /Create /TN "PropCollection-PreGame-1800" /SC DAILY /ST 18:00 `
  /TR "\"C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_daily_prop_collection.bat\""
schtasks /Create /TN "PropCollection-PreGame-1830" /SC DAILY /ST 18:30 `
  /TR "\"C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_daily_prop_collection.bat\""
schtasks /Create /TN "PropCollection-PreGame-2130" /SC DAILY /ST 21:30 `
  /TR "\"C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_daily_prop_collection.bat\""
```

Any snapshot taken within `closing_snapshot.window_minutes` (default 60) of a
game's start time is flagged `is_closing_snapshot=true` automatically — the
schedule just needs to land a run inside that window. Earlier snapshots are
kept too; line movement matters.

### Run the dashboard rebuild after collection

`run_daily_prop_collection.bat` already rebuilds the dashboard right after
collecting, so no separate task is needed. If you ever want a standalone
dashboard refresh (e.g. hourly):

```powershell
schtasks /Create /TN "PropDashboard-Rebuild" /SC HOURLY /MO 1 /ST 08:30 `
  /TR "powershell -NoProfile -Command \"Set-Location 'C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor'; .\.venv\Scripts\python.exe scripts\build_dashboard.py\""
```

## Managing tasks

```powershell
schtasks /Query /TN "PropCollection-Morning" /V /FO LIST   # inspect
schtasks /Run /TN "PropCollection-Morning"                 # run now (test)
schtasks /Delete /TN "PropCollection-Morning" /F           # remove
```

## Verifying a run

- Run summary: `data\reports\player_prop_collection_run_summary.json`
- Per-run logs: `data\logs\prop_collection_runs\run_<UTC timestamp>.log`
- Raw payloads: `data\raw\prop_odds\<LEAGUE>\odds_api\`
- Normalized history: `data\processed\player_prop_snapshots_normalized.csv`
- Dashboard page: open `data\reports\player_props.html`
