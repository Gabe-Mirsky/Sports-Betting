#!/usr/bin/env bash
# Build step for Render (and similar hosts). Exit on first error.
set -o errexit

pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate
