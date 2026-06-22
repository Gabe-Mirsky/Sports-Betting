"""SportsGameOdds API client (research-only collection plumbing).

Thin, defensive HTTP client for https://api.sportsgameodds.com/v2. Reads the
API key from the SPORTSGAMEODDS_API_KEY environment variable, authenticates
with the ``X-API-Key`` header (the key never appears in URLs or logs), retries
429/5xx with exponential backoff, follows ``nextCursor`` pagination, and
returns structured error objects instead of raising on network/API failures.

No models, no recommendations, no betting: this module only fetches and saves
raw odds data. Approved bets and approved parlays remain blocked.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_BASE_URL = "https://api.sportsgameodds.com/v2"
API_KEY_ENV = "SPORTSGAMEODDS_API_KEY"
USER_AGENT = "nba-kalshi-predictor/sportsgameodds-client (research-only)"

# Response headers worth surfacing for quota/usage reporting (lower-cased).
_INTERESTING_HEADERS = (
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-requests-remaining",
    "x-requests-used",
    "retry-after",
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def read_api_key(env: dict[str, str] | None = None) -> str:
    """Read the SportsGameOdds API key from the environment ('' when absent)."""

    source = os.environ if env is None else env
    return str(source.get(API_KEY_ENV, "") or "").strip()


def extract_items(payload: Any) -> tuple[list[Any], str | None]:
    """Pull the item list + nextCursor out of a response payload.

    Handles the documented ``{"success": ..., "data": [...], "nextCursor": ...}``
    envelope, a bare list, and a dict ``data`` (single-object endpoints like
    /account/usage). Unknown shapes return ``([payload], None)`` so nothing is
    silently dropped.
    """

    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        cursor = payload.get("nextCursor") or payload.get("next_cursor")
        cursor = str(cursor) if cursor else None
        data = payload.get("data", payload)
        if isinstance(data, list):
            return data, cursor
        if isinstance(data, dict):
            return [data], cursor
        return [payload], cursor
    return ([payload] if payload is not None else []), None


class SportsGameOddsClient:
    """Defensive GET-only client for the SportsGameOdds v2 API.

    ``http_get`` is injectable for offline tests/mock mode: it must accept
    ``(url, headers, timeout)`` and return ``(status_code, headers_dict, body_text)``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        env: dict[str, str] | None = None,
        http_get: Callable[[str, dict[str, str], float], tuple[int, dict[str, str], str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = (api_key or read_api_key(env)).strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self._http_get = http_get or self._default_http_get
        self._sleep = sleep
        self.last_headers: dict[str, str] = {}
        self.requests_made = 0

    # -- key handling -------------------------------------------------------

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def no_key_result(self, endpoint: str) -> dict[str, Any]:
        """Structured graceful-skip result used when no API key is configured."""

        return {
            "ok": False,
            "skipped": True,
            "status": None,
            "endpoint": endpoint,
            "error": f"no_api_key: set {API_KEY_ENV} to enable SportsGameOdds requests",
            "data": None,
        }

    # -- HTTP plumbing ------------------------------------------------------

    @staticmethod
    def _default_http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], str]:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                resp_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                return int(response.status), resp_headers, body
        except urllib.error.HTTPError as exc:  # non-2xx still has a body/headers
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            resp_headers = {str(k).lower(): str(v) for k, v in (exc.headers or {}).items()}
            return int(exc.code), resp_headers, body

    def _build_url(self, path: str, params: dict[str, Any] | None) -> str:
        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        query = f"?{urllib.parse.urlencode(clean)}" if clean else ""
        return f"{self.base_url}/{path.strip('/')}{query}"

    def request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET one endpoint; never raises on HTTP/network errors.

        Returns ``{"ok", "status", "endpoint", "data", "error", "headers",
        "attempts"}`` — ``data`` is the parsed JSON payload on success.
        """

        endpoint = path.strip("/")
        if not self.has_key:
            return self.no_key_result(endpoint)

        url = self._build_url(path, params)
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

        status: int | None = None
        error: str | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                status, resp_headers, body = self._http_get(url, headers, self.timeout)
            except Exception as exc:  # URLError, timeout, socket errors
                status, resp_headers, body = None, {}, ""
                error = f"network_error: {exc}"
            else:
                error = None
            self.requests_made += 1
            interesting = {k: v for k, v in resp_headers.items() if k in _INTERESTING_HEADERS}
            if interesting:
                self.last_headers = interesting

            if status is not None and 200 <= status < 300:
                try:
                    payload = json.loads(body) if body else None
                except json.JSONDecodeError as exc:
                    return {
                        "ok": False, "status": status, "endpoint": endpoint,
                        "data": None, "error": f"invalid_json: {exc}",
                        "headers": interesting, "attempts": attempts,
                    }
                return {
                    "ok": True, "status": status, "endpoint": endpoint,
                    "data": payload, "error": None,
                    "headers": interesting, "attempts": attempts,
                }

            if status is not None:
                error = f"http_{status}: {body[:300]}"
            retryable = status is None or status in _RETRYABLE_STATUS
            if retryable and attempt < self.max_retries:
                delay = self.backoff_seconds * (2**attempt)
                retry_after = interesting.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                self._sleep(delay)
                continue
            break

        return {
            "ok": False, "status": status, "endpoint": endpoint,
            "data": None, "error": error or "request_failed",
            "headers": dict(self.last_headers), "attempts": attempts,
        }

    def get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        max_pages: int = 5,
    ) -> dict[str, Any]:
        """Follow nextCursor pagination, capped at ``max_pages`` (quota guard)."""

        items: list[Any] = []
        raw_pages: list[Any] = []
        errors: list[str] = []
        cursor: str | None = None
        pages = 0
        while pages < max(1, int(max_pages)):
            page_params = dict(params or {})
            if cursor:
                page_params["cursor"] = cursor
            result = self.request(path, page_params)
            if not result["ok"]:
                errors.append(str(result.get("error")))
                if result.get("skipped"):
                    return {"ok": False, "skipped": True, "items": [], "raw_pages": [],
                            "pages": 0, "next_cursor": None, "errors": errors}
                break
            pages += 1
            raw_pages.append(result["data"])
            page_items, cursor = extract_items(result["data"])
            items.extend(page_items)
            if not cursor:
                break
        return {
            "ok": pages > 0 and not errors,
            "items": items,
            "raw_pages": raw_pages,
            "pages": pages,
            "next_cursor": cursor,
            "errors": errors,
        }

    # -- endpoint conveniences ----------------------------------------------

    def account_usage(self) -> dict[str, Any]:
        return self.request("account/usage")

    def sports(self) -> dict[str, Any]:
        return self.request("sports")

    def leagues(self, **params: Any) -> dict[str, Any]:
        return self.request("leagues", params or None)

    def stats(self, **params: Any) -> dict[str, Any]:
        return self.request("stats", params or None)

    def markets(self, **params: Any) -> dict[str, Any]:
        return self.request("markets", params or None)

    def players(self, **params: Any) -> dict[str, Any]:
        return self.request("players", params or None)

    def events(self, **params: Any) -> dict[str, Any]:
        return self.request("events", params or None)


def save_raw_payload(payload: Any, directory: str | Path, name: str) -> Path:
    """Save one raw JSON payload with a UTC-timestamped, collision-safe name.

    Never overwrites: existing files get a numeric suffix instead. Raw data is
    append-only by design.
    """

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}__{name}"
    path = target_dir / f"{base}.json"
    counter = 1
    while path.exists():
        path = target_dir / f"{base}_{counter}.json"
        counter += 1
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
