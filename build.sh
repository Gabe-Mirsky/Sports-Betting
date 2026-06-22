#!/usr/bin/env bash
# Build step for Render (and similar hosts). Exit on first error.
set -o errexit

pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate

# Load data and create the admin automatically (idempotent, safe to re-run).
# Uses the committed seed copy in predictions/seed_data/ when the live report
# files aren't in the repo (which is the case on the host).
python manage.py import_predictions
python manage.py create_admin
