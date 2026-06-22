"""Create an admin (superuser) non-interactively from environment variables.

Runs safely on every deploy: it does nothing if the env vars are missing or the
user already exists. Set these in your host's environment:

    DJANGO_SUPERUSER_USERNAME
    DJANGO_SUPERUSER_PASSWORD
    DJANGO_SUPERUSER_EMAIL   (optional)
"""

from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create a superuser from DJANGO_SUPERUSER_* env vars if it doesn't exist."

    def handle(self, *args, **options):
        username = os.environ.get("DJANGO_SUPERUSER_USERNAME")
        password = os.environ.get("DJANGO_SUPERUSER_PASSWORD")
        email = os.environ.get("DJANGO_SUPERUSER_EMAIL", "")

        if not username or not password:
            self.stdout.write("No DJANGO_SUPERUSER_USERNAME/PASSWORD set; skipping admin creation.")
            return

        User = get_user_model()
        if User.objects.filter(username=username).exists():
            self.stdout.write(f"Admin '{username}' already exists; nothing to do.")
            return

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(self.style.SUCCESS(f"Created admin user '{username}'."))
