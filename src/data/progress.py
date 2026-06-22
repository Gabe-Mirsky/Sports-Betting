"""Optional progress-bar compatibility helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


try:  # pragma: no cover - exercised when optional tqdm is installed.
    from tqdm import tqdm as tqdm
except ImportError:  # pragma: no cover - trivial fallback.

    def tqdm(iterable: Iterable[T], *args: object, **kwargs: object) -> Iterator[T]:
        """Return the iterable unchanged when tqdm is unavailable."""

        yield from iterable
