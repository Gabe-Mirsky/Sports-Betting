"""Logging helpers for CLI scripts."""

from __future__ import annotations

import logging


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure clean console logging for scripts."""

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
