# NBA Near-Tip-Off Collection Schedule (Windows Task Scheduler)

_Research-only data collection. No models, recommendations, approved bets, or
parlays are enabled by anything on this page._

## What already runs (keep these)

| Task | Trigger | Command |
| --- | --- | --- |
| Every-4-hours collection | Daily, repeat every 4 hours | `run_daily_prop_collection.bat` |
| At-login collection | At log on | `run_prop_collection_startup.bat` |

Keep both. They build the early-snapshot history that CLV comparisons need
(CLV = early line vs closing line; without early snapshots there is nothing to
compare the closing line against).

## What to add: evening pregame tasks

NBA games tip mostly between 7:00 PM and 10:30 PM Eastern. The every-4-hours
cadence usually lands only 0–1 runs inside the final 2 hours before tip, which
is why NBA closing-like snapshots (within 60 minutes of tip) are still rare.

Add evening runs of **`run_nba_pregame_prop_collection.bat`** (in the project
root). Each run collects props, enriches NBA snapshots, refreshes settlement
from the local caches, rebuilds the market-quality / manual-review / health /
CLV-readiness reports, and rebuilds the dashboard.

### Suggested fixed Task Scheduler times (Eastern)

| Time (ET) | Why |
| --- | --- |
| 6:00 PM | ~1–2h before early tips: catches the "2h_before" window |
| 6:30 PM | 60m window for 7:30 tips; 2h window for 8:30 tips |
| 7:00 PM | 60m/30m windows for 7:30–8:00 tips |
| 7:30 PM | 30m/10m windows for early tips; 60m for 8:30 tips |
| 8:30 PM | closing-like coverage for 8:30–9:30 tips |
| 9:30 PM | closing-like coverage for late (10:00–10:30) tips |
| 10:00 PM | 30m/10m windows for the latest tips |

Task Scheduler setup (same pattern as `scripts/windows_task_scheduler_setup.md`):

1. Task Scheduler → Create Task… → name it e.g. `NBA pregame props 6:00 PM`.
2. Trigger: Daily at the chosen time.
3. Action: Start a program → `run_nba_pregame_prop_collection.bat`,
   "Start in" = the project folder.
4. Settings: enable "Run task as soon as possible after a scheduled start is
   missed".

## Why fixed times are imperfect

NBA start times vary by day (7:00/7:30/8:00/8:30/10:00 PM ET and playoff
one-game days like 8:40 PM). A fixed 7:30 PM run is 70 minutes before an 8:40
tip (good: hits the 2h/60m boundary) but lands *after* a 7:00 tip has already
started (useless for that game). Fixed times therefore guarantee decent — not
perfect — window coverage on multi-game days, and can miss the 30m/10m windows
entirely for off-schedule tips.

**True dynamic scheduling** would need a future scheduler that reads
`data/reports/nba_prop_closing_collection_plan.json` (built by
`scripts/build_nba_collection_plan.py`), takes
`next_recommended_collection_time_utc` per game, and registers one-shot tasks
at `tip − 120/60/30/10` minutes. That scheduler does not exist yet; until it
does, the fixed evening times above are the practical approximation.

## Missed snapshots are unrecoverable

The Odds API does **not** provide historical odds on the current plan. If no
run happens inside a window, that window's odds are permanently lost — there
is no backfill. The collection plan report lists missed windows per game
honestly instead of pretending they can be recovered. This is also why the
evening tasks are worth the quota: a missed closing window costs the game's
CLV measurement forever.

## Quota notes

- The collector always collects modeling-priority leagues (NBA) first.
- When quota is limited, collect-only leagues (MLB/WNBA/NHL/NCAAB/soccer) are
  skipped before NBA (`quota.low_priority_min_remaining` in
  `config/prop_collection.yaml`).
- When an NBA game is within 60 minutes, the run is flagged high priority in
  `data/reports/player_prop_collection_run_summary.json`
  (`nba_closing_priority`).
- Evening runs add ~7 collection passes; with `max_events_per_league_per_run`
  small (default 6) this is the main quota cost of the day. If the monthly
  quota gets tight, drop the 6:00 PM and 10:00 PM tasks first — the 60m/30m
  windows matter most for CLV.
