"""API-Sports client (probe/backup source; research-only).

Minimal defensive client for the api-sports.io family. Each sport lives on its
own host (v1.basketball, v2.nba, v3.football, ...); all share the
``x-apisports-key`` auth header and the standard envelope::

    {"get": ..., "parameters": ..., "errors": ..., "results": N, "response": [...]}

Reads APISPORTS_API_KEY from the environment, never logs or hardcodes the key,
retries 429/5xx with backoff, and returns structured error objects instead of
raising. Probe-only until player props are proven: no collector is built on
top of this module.

Research-only: no models, no recommendations, no betting changes.
"""

from __future__ import annotations

import json
import time
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable


API_KEY_ENV = "APISPORTS_API_KEY"
AUTH_HEADER = "x-apisports-key"
USER_AGENT = "nba-kalshi-predictor/apisports-client (research-only)"

# Sport family -> API host. /status on each host reports the subscription.
API_HOSTS: dict[str, str] = {
    "basketball": "https://v1.basketball.api-sports.io",
    "nba": "https://v2.nba.api-sports.io",
    "football": "https://v3.football.api-sports.io",  # soccer
    "american_football": "https://v1.american-football.api-sports.io",
    "baseball": "https://v1.baseball.api-sports.io",
    "hockey": "https://v1.hockey.api-sports.io",
}

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def read_api_key(env: dict[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    return str(source.get(API_KEY_ENV, "") or "").strip()


class ApiSportsClient:
    """GET-only client; ``http_get(url, headers, timeout)`` is injectable."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
        env: dict[str, str] | None = None,
        http_get: Callable[[str, dict[str, str], float], tuple[int, dict[str, str], str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = (api_key or read_api_key(env)).strip()
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self._http_get = http_get or self._default_http_get
        self._sleep = sleep
        self.requests_made = 0

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def no_key_result(self, endpoint: str) -> dict[str, Any]:
        return {
            "ok": False,
            "skipped": True,
            "status": None,
            "endpoint": endpoint,
            "error": f"no_api_key: set {API_KEY_ENV} to enable API-Sports requests",
            "data": None,
        }

    @staticmethod
    def _default_http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], str]:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                resp_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                return int(response.status), resp_headers, body
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            resp_headers = {str(k).lower(): str(v) for k, v in (exc.headers or {}).items()}
            return int(exc.code), resp_headers, body

    def request(self, api: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET ``{host}/{path}`` for one sport API; never raises on failure."""

        endpoint = f"{api}/{path.strip('/')}"
        if not self.has_key:
            return self.no_key_result(endpoint)
        host = API_HOSTS.get(api)
        if not host:
            return {
                "ok": False, "skipped": False, "status": None, "endpoint": endpoint,
                "error": f"unknown_api: {api} (known: {sorted(API_HOSTS)})", "data": None,
            }
        clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        query = f"?{urllib.parse.urlencode(clean)}" if clean else ""
        url = f"{host}/{path.strip('/')}{query}"
        headers = {
            AUTH_HEADER: self.api_key,
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

        status: int | None = None
        error: str | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                status, _resp_headers, body = self._http_get(url, headers, self.timeout)
            except Exception as exc:
                status, body = None, ""
                error = f"network_error: {exc}"
            else:
                error = None
            self.requests_made += 1

            if status is not None and 200 <= status < 300:
                try:
                    payload = json.loads(body) if body else None
                except json.JSONDecodeError as exc:
                    return {"ok": False, "status": status, "endpoint": endpoint,
                            "data": None, "error": f"invalid_json: {exc}", "attempts": attempts}
                # API-Sports reports auth/limit problems inside a 200 response.
                api_errors = payload.get("errors") if isinstance(payload, dict) else None
                if api_errors and (not isinstance(api_errors, (list, dict)) or len(api_errors) > 0):
                    return {"ok": False, "status": status, "endpoint": endpoint,
                            "data": payload, "error": f"api_errors: {api_errors}", "attempts": attempts}
                return {"ok": True, "status": status, "endpoint": endpoint,
                        "data": payload, "error": None, "attempts": attempts}

            if status is not None:
                error = f"http_{status}: {body[:300]}"
            retryable = status is None or status in _RETRYABLE_STATUS
            if retryable and attempt < self.max_retries:
                self._sleep(self.backoff_seconds * (2**attempt))
                continue
            break

        return {"ok": False, "status": status, "endpoint": endpoint,
                "data": None, "error": error or "request_failed", "attempts": attempts}

    def status(self, api: str) -> dict[str, Any]:
        """The /status endpoint: subscription, plan, and daily usage."""

        return self.request(api, "status")


def extract_response(payload: Any) -> list[Any]:
    """Pull the ``response`` list out of the API-Sports envelope."""

    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, list):
            return response
        if response is not None:
            return [response]
    if isinstance(payload, list):
        return payload
    return []
