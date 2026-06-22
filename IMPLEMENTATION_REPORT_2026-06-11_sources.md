# Implementation Report — 2026-06-11 (Multi-source player props)

Research-only. No models, recommendations, bets, or parlays were enabled.
Approved bets and approved parlays remain blocked behind the existing proof gates.

## What was added

### SportsGameOdds (now live, second NBA prop source)
- `src/data/sportsgameodds_client.py` — defensive v2 client: `X-API-Key` header,
  env key `SPORTSGAMEODDS_API_KEY`, no-key graceful skip, 429/5xx retry+backoff,
  `nextCursor` pagination (page cap), structured error objects, collision-safe
  raw saving. Key never appears in URLs or logs.
- `scripts/probe_sportsgameodds.py` — small probe (usage, sports, leagues,
  markets, stats, tiny NBA events). `--cheap` flag = usage-only refresh for
  scheduled runs (the full probe costs ~470 monthly entities).
- `src/data/sportsgameodds_prop_adapter.py` — normalizes events into the
  player-prop snapshot schema: full-game `ou` player odds only (game/team and
  period odds excluded), over/under merged per book, per-book alternate lines
  preserved, one-sided rows kept and flagged, American→decimal conversion,
  team/opponent/home_away filled from the event players map.
- `scripts/collect_sportsgameodds_props.py` + `config/sportsgameodds.yaml` —
  NBA-first collection with a monthly-entity usage guard (floor 300), raw saves
  under `data/raw/sportsgameodds/player_props/`, schema validation that refuses
  to append invalid rows, source CSV + shared normalized CSV append/dedupe,
  summary JSON/MD + append-only run history.

### API-Sports (probe-only, blocked)
- `src/data/apisports_client.py` — `x-apisports-key` client across the
  api-sports.io host family, API-errors-inside-200 detection, no-key skip.
- `scripts/probe_apisports.py` — status/bets/games/odds probe with
  `--max-age-hours` freshness guard. **Verdict: free plan serves only
  2022–2024 seasons; current-season odds/props are inaccessible. No collector
  was built.** Player-prop bet types exist in the catalog, so a paid plan
  could be re-probed later.

### Reports, dashboard, pipeline
- `scripts/build_odds_source_comparison.py` → `odds_source_comparison.{json,md}`
  + `odds_source_adapter_plan.csv` (6 sources).
- `scripts/build_odds_source_usage_summary.py` → per-source status/quota/errors
  + primary/backup per league.
- `scripts/build_cross_source_prop_comparison.py` → overlap comparison with
  bookmaker alias normalization (`williamhill_us`→`caesars`); writes an explicit
  no-overlap explanation today.
- `build_odds_api_quota_report.py` gained a "SportsGameOdds offload" section.
- Dashboard (`player_props.html`) gained an "Odds Sources" section: SGO status,
  API-Sports status, usage/quota, primary/backup table, cross-source results,
  warnings, and download links for all new reports.
- `run_full_prop_pipeline.py` gained 6 optional steps (cheap SGO probe, SGO
  collection, API-Sports probe, 3 report builders); probe failures cannot break
  the pipeline. Validated end-to-end: 22 steps green.

### Tests
- 47 new tests across `test_sportsgameodds_client.py`,
  `test_sportsgameodds_adapter.py`, `test_apisports_client.py`,
  `test_odds_source_reports.py` (offline, no network, no keys).
- Full suite: **532 tests, all passing**.

## Measured findings
- SGO tier "amateur": 10 requests/min, **2,500 entities/month**. Measured:
  1 `/events` pull ≈ 1 entity even with ~1,000 odds objects; metadata endpoints
  cost ~470 entities (probe). Collection therefore spends entities only on
  `/events`. 2,024 entities remain this month.
- First SGO collection: 1 event (SAS@NYK Finals, 2026-06-13) → **710 schema-valid
  rows from 6 books**, 100% player-ID enrichment match.
- NBA coverage vs The Odds API: SGO adds `caesars` + `espnbet` books (odds_api
  adds `betrivers`) and widens NBA prop types from 4 to 10 (steals, blocks,
  pra, points_rebounds, points_assists, rebounds_assists added). SGO also
  carries opening lines (`openBookOdds`) — a future CLV input.
- The Odds API quota: ~260/500 credits left, risk "high" → SGO offload of NBA
  prop pulls recommended once cross-source agreement is verified.
- Cross-source overlap: none yet (different game windows). Expected to appear
  when the Odds API 36h horizon reaches the June 13 Finals game.

## Quota safety in place
- SGO: usage check before every collection, skip below 300 monthly entities,
  per-league event caps, max 2 leagues/run, NBA-only enabled.
- API-Sports: probe-only, daily freshness guard.
- Odds API: unchanged guards; offload note added to the quota report.
