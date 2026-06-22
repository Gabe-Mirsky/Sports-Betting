# Prediction website (Django)

The Django site under `webapp/` (config) and `predictions/` (app) is the primary UI
for browsing model matchup predictions and building parlays. It replaces the old
self-contained static HTML dashboard (`scripts/build_dashboard.py` /
`src/reports/dashboard.py`), which is kept in git history.

## What it does

- **Predictions** — a simplified, fits-on-screen table: sport, league, teams,
  prediction (outcome + win probabilities + confidence), reasoning, and date.
- **Leaderboard** — predictions ranked by confidence or by pick probability.
- **Parlay creator** — tick predictions to add them as legs; the page computes the
  combined model probability (legs assumed independent) and the implied fair
  decimal/American odds, and saves parlays to the database.
- **Recommended** — the site automatically suggests a few parlays (safest double,
  safest treble, different-leagues double, highest combined chance) and explains each
  in plain language.
- **Parlay creator** — tick predictions to add them as legs; the page computes the
  combined model probability and implied fair odds, and saves parlays.
- **How It Works** — a 5th-grade-readable page naming every variable the model uses,
  explaining how probability is computed, and showing the real learned coefficients
  ("points toward Team A winning").
- **Add** — enter a raw matchup; the AI reasoning service drafts the reasons/risks,
  you preview it, and it is submitted for review (appears once approved in admin).
- **Admin** — `/admin/` lets the owner approve, edit, or hide predictions.

These are model-implied probabilities, **not** sportsbook odds or betting advice.

## Populate the "How It Works" coefficients

```powershell
python manage.py export_model_explanation
```

This trains the matchup model on `data/processed/match_results.csv`, extracts the
logistic-regression weights, and writes `data/reports/model_explanation.json`, which
the How It Works page reads. Re-run it whenever the model or data changes. If the
file is missing, the page still shows every variable in plain language.

## Setup

```powershell
pip install -r requirements.txt          # includes Django
python manage.py migrate                  # creates db.sqlite3
python manage.py createsuperuser          # for the admin panel
```

## Load data

Import the latest predictions produced by the analysis pipeline
(`data/reports/matchup_predictions_today.csv`):

```powershell
python manage.py import_predictions               # new rows = approved
python manage.py import_predictions --status pending   # require manual approval
```

Or seed a few example rows without the pipeline:

```powershell
python manage.py seed_demo
```

Re-running `import_predictions` updates existing rows in place (matched on
`fixture_id`), so it is safe to run after every pipeline refresh.

## Run

```powershell
python manage.py runserver
```

Then open http://127.0.0.1:8000/ . The admin panel is at /admin/ .

## AI reasoning service

`predictions/services/ai_reasoning.py` reads `AI_PROVIDER`, `AI_API_KEY`, and
`AI_MODEL` from the environment. With no `AI_API_KEY` set, it returns useful
deterministic reasoning so the site works fully offline. With a key, it makes a
best-effort call to the configured provider (`anthropic` or `openai`) and falls
back automatically on any error.

## Checks

```powershell
python manage.py check
python manage.py test predictions
```

## Deploy to a real public URL

The project is ready to deploy to a Python web host (Render, Railway, or Fly.io).
Render is the simplest because `render.yaml` is included. **You** need to create the
GitHub repo and hosting account — those steps require your own logins.

### 1. Put the code on GitHub

```powershell
git add .
git commit -m "Django prediction website"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`db.sqlite3` and `staticfiles/` are gitignored; the host builds its own.

### 2. Deploy on Render (recommended)

1. Go to render.com → **New → Blueprint** and pick your GitHub repo. Render reads
   `render.yaml` and creates a web service **and** a free Postgres database.
2. In the web service's **Environment** tab, set your admin login:
   `DJANGO_SUPERUSER_USERNAME`, `DJANGO_SUPERUSER_PASSWORD`, and (optional)
   `DJANGO_SUPERUSER_EMAIL`.
3. Deploy. The build (`build.sh`) automatically runs `collectstatic`, `migrate`,
   loads the bundled predictions, and creates your admin — **no shell needed**
   (the free tier has no Shell tab). `DJANGO_SECRET_KEY` is auto-generated,
   `DATABASE_URL` is wired to Postgres, and `DJANGO_DEBUG=0` is set.

Render gives you a public URL like `https://matchup-predictor.onrender.com`, with
the predictions, recommendations, parlay creator, How It Works coefficients, and
`/admin/` all working immediately.

> **Data note:** small seed copies of the predictions and the model coefficients are
> committed in `predictions/seed_data/` so the live site has data without needing the
> gitignored `data/` files. To refresh the live site later, regenerate those files
> locally (`build_matchup_predictions.py` + `export_model_explanation`), copy them
> into `predictions/seed_data/`, and push.

### 3. Use your own domain

In Render: **Settings → Custom Domains → Add** your domain, then add the CNAME
record it shows at your domain registrar. Add the domain to `CSRF_TRUSTED_ORIGINS`
via the `DJANGO_CSRF_TRUSTED_ORIGINS` env var (comma-separated, `https://…`).

### Environment variables (any host)

| Variable | Purpose |
| --- | --- |
| `DJANGO_SECRET_KEY` | Long random secret (required in production) |
| `DJANGO_DEBUG` | `0` in production |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated hostnames (auto-set on Render) |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | `https://yourdomain.com` for forms/admin |
| `DATABASE_URL` | Postgres URL (falls back to SQLite if unset) |
| `AI_PROVIDER` / `AI_API_KEY` / `AI_MODEL` | Optional live AI reasoning |

Railway/Fly.io work too: set the same env vars, use `build.sh` as the build command
and `gunicorn webapp.wsgi` (from the `Procfile`) as the start command.
