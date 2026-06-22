# Matchup Predictions (no-odds model)

A general matchup-prediction engine that estimates **model-implied
probabilities** for upcoming games using only historical results and context —
**no odds, no closing lines, no CLV required.** It lives alongside (and does not
break) the existing odds/player-prop code.

> These are model probabilities, not betting odds or bet recommendations.

## What it predicts

For every fixture:

- `prob_team_a_win`, `prob_draw`, `prob_team_b_win` (draw is always 0 for
  no-draw sports such as basketball/baseball/hockey)
- predicted outcome + confidence level (High / Medium / Low / Very low)
- key reasons, main risks, data-quality warnings
- a data-quality tag (strong / usable / weak / very_weak)

## Pipeline

```
results CSV ─▶ match_results_loader ─┐
fixtures CSV ─▶ fixtures_loader ─────┤
injuries CSV ─▶ injuries_loader ─────┘
        │
        ▼
features.team_strength (Elo)  +  features.matchup_features (form, rest,
        │                         schedule congestion, injuries — leakage-safe)
        ▼
models.matchup_model  (logistic regression; multinomial for draw sports)
        ▼
models.prediction_explainer  +  quality.matchup_data_quality
        ▼
reports.matchup_prediction_report  ─▶  data/reports/matchup_predictions_today.{csv,json}
evaluation.backtest_matchup_model  ─▶  data/reports/matchup_model_backtest{,_by_bucket}.{json,csv}
        ▼
reports.dashboard  ─▶  "Matchup Predictions" page (today/upcoming + backtest)
```

No future leakage: every feature for a historical game uses only games that
finished before it; pre-game Elo is stored before the game updates it; the
backtest is strictly walk-forward (expanding window, never shuffled).

## Input formats

The loaders accept many source column names (case-insensitive) and normalize
them. Minimum useful columns:

- **results**: `date`, `home_team`/`away_team` (or `team_a`/`team_b`),
  `home_score`/`away_score`, `sport`, `league`, optional `competition_type`,
  `neutral_site`.
- **fixtures**: `date`, `team_a`/`team_b` (or home/away), `sport`, `league`,
  optional `competition_type`, `neutral`, `venue`, `status`.
- **injuries** (optional): `team`, `player`, `status` (out/doubtful/
  questionable/probable/available), optional `role`/`importance`, `position`,
  `last_updated`.

Team-name differences (e.g. "Korea Republic" vs "South Korea") are normalized
via `data/team_name_map.py`; extend it by editing
`data/manual/team_aliases_template.csv` and passing `--aliases-path`.

## Commands

```bash
# Backtest (no odds needed)
python scripts/backtest_matchup_model.py \
  --results-path data/processed/match_results.csv --sport soccer \
  --output-dir data/reports

# Build today's / upcoming predictions
python scripts/build_matchup_predictions.py \
  --results-path data/processed/match_results.csv \
  --fixtures-path data/processed/fixtures_today.csv \
  --injuries-path data/processed/injuries.csv \
  --output-dir data/reports

# Build the dashboard (Matchup Predictions tab links from the main page)
python scripts/build_dashboard.py --output-path data/reports/dashboard.html
```

Sample data to try it immediately lives in `data/samples/` (and is copied to
the `data/processed/` paths above). The samples include a neutral-site
`Japan vs Tunisia` friendly.

## Backtest metrics

Prediction-quality only — never ROI, never win-rate-only:
accuracy, log loss, Brier score, mean probability of the actual outcome,
calibration by probability bucket, and accuracy by confidence level / sport /
league / competition type, plus draw quality for soccer.

## Tests

```bash
python -m unittest tests.test_matchup_loaders tests.test_matchup_elo \
  tests.test_matchup_features_leakage tests.test_matchup_model \
  tests.test_matchup_data_quality tests.test_matchup_report
```
