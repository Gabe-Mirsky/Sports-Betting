# Running Prop Collection Without Keeping Your PC On

_Research-only. These notes cover **data collection scheduling only** — no
models, recommendations, or betting are enabled by any of these options._

The core problem: The Odds API does **not** provide historical odds on the
current plan. If nothing calls `scripts/daily_collect_props.py` on a given day,
those odds snapshots are **permanently lost**. So collection reliability is
about maximizing the fraction of days something runs the collector.

Three options, from least to most reliable:

| | Windows Task Scheduler | GitHub Actions (scheduled) | Cheap VPS / cloud server |
|---|---|---|---|
| Runs when your PC is off | No | Yes | Yes |
| Cost | Free | Free (public/private repo minutes) | ~$4–6/month |
| Setup effort | Low | Medium | Medium-high |
| Data lives | On your PC | In the repo / artifacts | On the VPS |
| Best for | Now, while testing | 24/7 on a budget | 24/7 with full control |

---

## Option 1: Windows Task Scheduler (local)

### Pros
- Zero new infrastructure; data stays exactly where it is today.
- `run_prop_collection_startup.bat` already does collect → settle → health → dashboard with one log file per run.
- Catch-up warning + health report make missed days visible.

### Cons
- **Runs only while the PC is on and logged in.** A day with the PC off is a permanently missed day of odds.
- Sleep/hibernate also blocks runs unless you allow wake timers.

### Setup
1. Open Task Scheduler → Create Task.
2. Triggers (use both):
   - **At log on** → runs whenever you start the PC (catches irregular schedules).
   - **Daily** at the times in `config/prop_collection.yaml` `collection_windows` (09:00, 13:00, 17:30, 18:30) → captures line movement. Check "Run task as soon as possible after a scheduled start is missed."
3. Action: Start a program → `C:\Users\arilo\Downloads\Python Projects\nba_kalshi_predictor\run_prop_collection_startup.bat` (Start in: the repo folder).
4. Settings: enable "Run task as soon as possible after a scheduled start is missed"; optionally "Wake the computer to run this task."

### Where to store ODDS_API_KEY
- User environment variable: Settings → System → About → Advanced system settings → Environment Variables → New (user) → `ODDS_API_KEY`. Task Scheduler tasks inherit user env vars.

### How to keep data
- Already local: `data/processed/`, `data/raw/prop_odds/`, `data/reports/`. Back up `data/processed/player_prop_snapshots_normalized.csv` (it's the canonical history).

### How to check logs
- Per startup run: `data/logs/startup_runs/startup_*.log`
- Per collection run: `data/logs/prop_collection_runs/run_*.log`
- Health: `data/reports/prop_collection_health.md` and the Player Props dashboard page.

### What happens if the machine is off
- Nothing runs. The next run logs a missed-collection warning and the health report counts the missed day. **The odds from that gap are unrecoverable.**

---

## Option 2: GitHub Actions scheduled workflow

### Pros
- Runs even when every machine you own is off; free minutes are plenty (a run is a few minutes).
- Secrets management built in; logs kept per run in the Actions UI.
- Data can be committed back to the repo → automatic history + offsite backup.

### Cons
- `schedule:` cron is best-effort: runs can start late (minutes to ~an hour) or occasionally be skipped during GitHub load spikes — still far better than a PC that's off.
- Repo becomes the datastore: snapshot CSV grows in git history (fine for years at this volume, but it's a commitment).
- Requires pushing this repo to GitHub (make it **private** — the data and config don't belong in public).
- Scheduled workflows on GitHub are disabled after 60 days of repo inactivity (the data commits themselves count as activity, so in practice this self-sustains).

### Setup sketch
Create `.github/workflows/collect_props.yml`:

```yaml
name: Collect player props
on:
  schedule:
    # UTC. 13:00 / 17:00 / 21:30 / 22:30 UTC ≈ 09:00 / 13:00 / 17:30 / 18:30 ET (summer)
    - cron: "0 13 * * *"
    - cron: "0 17 * * *"
    - cron: "30 21 * * *"
    - cron: "30 22 * * *"
  workflow_dispatch: {}     # manual trigger button

permissions:
  contents: write

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python scripts/daily_collect_props.py
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      - run: python scripts/build_prop_collection_health.py
      - name: Commit collected data
        run: |
          git config user.name "prop-collector-bot"
          git config user.email "actions@users.noreply.github.com"
          git add data/processed data/raw/prop_odds data/reports data/logs
          git diff --cached --quiet || git commit -m "Collect prop snapshots ($(date -u +%Y-%m-%dT%H:%M)Z)"
          git push
```

Note: settlement (`refresh_nba_results_and_settle_props.py`) depends on local
nba_api caches — keep running that part locally (or extend the workflow later).
Pull before local work so the locally built dashboard sees the cloud-collected rows.

### Where to store ODDS_API_KEY
- Repo → Settings → Secrets and variables → Actions → New repository secret → `ODDS_API_KEY`. Never commit the key.

### How to keep data
- Commit data back to the repo (as above) — simplest and gives offsite backup; or upload as workflow artifacts (90-day expiry — not suitable as the only copy).

### How to check logs
- GitHub → Actions tab → each run's console output, plus the committed `data/logs/` and `data/reports/prop_collection_health.md`.

### What happens if the machine is off
- Irrelevant — GitHub's runners do the collecting. Your PC only needs to `git pull` before you analyze.

---

## Option 3: Cheap VPS / cloud server

Examples: Hetzner CX22 (~€4/mo), DigitalOcean Basic Droplet ($6/mo), Oracle
Cloud Always Free tier ($0, with availability caveats), AWS Lightsail ($5/mo).

### Pros
- True 24/7: exact cron timing (better closing-line capture than GitHub's best-effort cron), no usage limits, full control.
- Can also run the settlement refresh on a schedule, since the box is always on.
- Disk persists; no git-as-database compromise.

### Cons
- Costs money; you administer it (updates, disk space, security).
- Data lives off your PC — you need a sync/backup step (git, rsync, or scheduled copy).
- Linux box: the `.bat` files don't apply; use cron + the Python scripts directly (they're OS-neutral).

### Setup sketch (Ubuntu)
```bash
sudo apt update && sudo apt install -y python3-venv git
git clone <your-private-repo> ~/nba_kalshi_predictor
cd ~/nba_kalshi_predictor
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
crontab -e
```

Crontab (UTC; mirrors the four collection windows):
```cron
ODDS_API_KEY=your_key_here
0 13 * * * cd ~/nba_kalshi_predictor && .venv/bin/python scripts/daily_collect_props.py >> data/logs/cron.log 2>&1
0 17 * * * cd ~/nba_kalshi_predictor && .venv/bin/python scripts/daily_collect_props.py >> data/logs/cron.log 2>&1
30 21 * * * cd ~/nba_kalshi_predictor && .venv/bin/python scripts/daily_collect_props.py >> data/logs/cron.log 2>&1
30 22 * * * cd ~/nba_kalshi_predictor && .venv/bin/python scripts/daily_collect_props.py >> data/logs/cron.log 2>&1
45 22 * * * cd ~/nba_kalshi_predictor && .venv/bin/python scripts/build_prop_collection_health.py >> data/logs/cron.log 2>&1
```

### Where to store ODDS_API_KEY
- In the crontab (as above, readable only by your user), or in `~/.profile` / a systemd service `Environment=` line. Never in the repo.

### How to keep data
- Nightly `git add data && git commit && git push` cron line to a private repo (doubles as backup), or rsync/scp the `data/` folder down to your PC when you want to analyze.

### How to check logs
- `data/logs/cron.log`, the per-run logs in `data/logs/prop_collection_runs/`, and `data/reports/prop_collection_health.md` on the box (`ssh` in, or pull via git).

### What happens if the machine is off
- VPSs are effectively never off (provider uptime ~99.9%). If the provider has an outage, the next run logs the gap warning honestly, same as everywhere else.

---

## Recommendation

1. **Today:** keep Task Scheduler + `run_prop_collection_startup.bat` (login trigger + the four daily windows with "run missed task ASAP"). Costs nothing and works now.
2. **For real 24/7:** GitHub Actions is the best free step up — collection keeps running with your PC off, the key lives in a secret, and committed data is automatically backed up. Accept the ±minutes cron jitter.
3. **If/when closing-line timing matters more** (tighter snapshots near tip): a $4–6/mo VPS with cron gives exact timing and can take over settlement too.

Whatever runs the collector, the health report (`scripts/build_prop_collection_health.py`)
stays the single source of truth for "did collection actually happen?" — check
the Player Props dashboard page or `data/reports/prop_collection_health.md`.
