"""Optional Kalshi data helpers.

The first working version is local/mock-first. Real Kalshi API access stays
optional and is intentionally not required for model training or backtesting.
"""

from __future__ import annotations

import csv
import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
try:  # Keep local/model-only workflows importable before requirements are installed.
    import requests
except ImportError:  # pragma: no cover - exercised only in under-provisioned runtimes.
    requests = None  # type: ignore[assignment]

try:  # Authenticated reads need this, public market data does not.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:  # pragma: no cover - optional in public-read-only mode.
    hashes = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    padding = None  # type: ignore[assignment]

from .team_aliases import normalize_team_abbr
from .validation import require_columns


KALSHI_TRADE_API_BASE = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_DEMO_API_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PUBLIC_API_BASE = KALSHI_TRADE_API_BASE
NBA_GAME_SERIES_TICKER = "KXNBAGAME"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAILED_REQUEST_COLUMNS = [
    "timestamp_utc",
    "method",
    "url",
    "params",
    "status_code",
    "response_text",
    "error",
]

logger = logging.getLogger(__name__)
_LOCAL_ENV_LOADED = False

MARKET_TEMPLATE_COLUMNS = [
    "market_ticker",
    "event_ticker",
    "game_date",
    "home_team_abbr",
    "away_team_abbr",
    "market_question",
    "yes_team_abbr",
    "no_team_abbr",
    "yes_bid_cents",
    "yes_ask_cents",
    "yes_mid_cents",
    "close_price_cents",
    "settlement",
    "volume",
    "open_time",
    "close_time",
]

PUBLIC_MARKET_COLUMNS = [
    *MARKET_TEMPLATE_COLUMNS,
    "series_ticker",
    "status",
    "last_price_cents",
    "public_api_updated_time",
    "public_api_expected_expiration_time",
    "price_source",
]

MARKET_MATCH_COLUMNS = [
    "market_ticker",
    "game_date",
    "home_team_abbr",
    "away_team_abbr",
    "yes_team_abbr",
    "yes_mid_cents",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _load_local_env_once() -> None:
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    _LOCAL_ENV_LOADED = True
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class KalshiAPIClient:
    """Small read-oriented wrapper around Kalshi market and candle endpoints."""

    def __init__(
        self,
        base_url: str = KALSHI_TRADE_API_BASE,
        timeout: int = 30,
        require_auth: bool = False,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        failed_requests_path: str | Path | None = None,
        session: Any | None = None,
    ) -> None:
        _load_local_env_once()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.require_auth = require_auth
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID") or os.getenv("KALSHI_API_KEY")
        self.private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self.failed_requests_path = (
            Path(failed_requests_path)
            if failed_requests_path
            else PROJECT_ROOT / "data" / "reports" / "kalshi_failed_requests.csv"
        )
        self.session = session or (requests.Session() if requests is not None else None)
        self._private_key: Any | None = None
        self._auth_warning_logged = False

    @classmethod
    def from_env(cls, use_demo: bool | None = None, **kwargs: Any) -> "KalshiAPIClient":
        """Create a client using environment settings and public read defaults."""

        _load_local_env_once()
        env = os.getenv("KALSHI_ENV", "prod").strip().lower()
        use_demo_value = use_demo if use_demo is not None else env == "demo"
        base_url = os.getenv("KALSHI_BASE_URL") or (KALSHI_DEMO_API_BASE if use_demo_value else KALSHI_TRADE_API_BASE)
        return cls(base_url=base_url, **kwargs)

    def _load_private_key(self) -> Any:
        if self._private_key is not None:
            return self._private_key
        if serialization is None:
            raise RuntimeError("The cryptography package is required for authenticated Kalshi requests.")
        if not self.private_key_path:
            raise RuntimeError("KALSHI_PRIVATE_KEY_PATH is not set.")
        key_path = Path(self.private_key_path)
        with key_path.open("rb") as file:
            self._private_key = serialization.load_pem_private_key(file.read(), password=None)
        return self._private_key

    def _create_signature(self, timestamp_ms: str, method: str, path: str) -> str:
        if hashes is None or padding is None:
            raise RuntimeError("The cryptography package is required for authenticated Kalshi requests.")
        private_key = self._load_private_key()
        path_without_query = path.split("?", 1)[0]
        message = f"{timestamp_ms}{method.upper()}{path_without_query}".encode("utf-8")
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str = "GET", path: str = "/") -> tuple[dict[str, str], str]:
        headers = {"Accept": "application/json"}
        if not (self.api_key_id and self.private_key_path):
            if self.require_auth:
                return headers, (
                    "Kalshi credentials are missing. Set KALSHI_API_KEY_ID and "
                    "KALSHI_PRIVATE_KEY_PATH, or run with require_auth=false/mock CSV mode."
                )
            return headers, ""

        try:
            timestamp_ms = str(int(time.time() * 1000))
            sign_path = urllib.parse.urlparse(f"{self.base_url}{path}").path
            headers.update(
                {
                    "KALSHI-ACCESS-KEY": self.api_key_id,
                    "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
                    "KALSHI-ACCESS-SIGNATURE": self._create_signature(
                        timestamp_ms,
                        method,
                        sign_path,
                    ),
                }
            )
        except Exception as exc:
            return headers, f"Kalshi authentication signing failed: {type(exc).__name__}: {exc}"
        return headers, ""

    def _record_failed_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        status_code: int | None = None,
        response_text: str = "",
        error: str = "",
    ) -> None:
        self.failed_requests_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp_utc": _utc_now_iso(),
            "method": method,
            "url": url,
            "params": _json_text(params or {}),
            "status_code": status_code if status_code is not None else "",
            "response_text": response_text[:2000],
            "error": error,
        }
        file_exists = self.failed_requests_path.exists()
        with self.failed_requests_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FAILED_REQUEST_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers, auth_error = self._headers("GET", path)
        if auth_error:
            if self.require_auth:
                logger.error(auth_error)
                self._record_failed_request("GET", url, params, error=auth_error)
                return {"error": auth_error}
            if not self._auth_warning_logged:
                logger.warning("%s Falling back to unauthenticated public reads.", auth_error)
                self._auth_warning_logged = True

        if requests is None or self.session is None:
            return self._get_with_urllib(path, url, params, headers)

        response = None
        for attempt in range(1, 4):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                error = str(exc)
                if attempt < 3:
                    time.sleep(attempt * 2.0)
                    continue
                logger.error("Kalshi request failed: %s params=%s error=%s", url, params, error)
                self._record_failed_request("GET", url, params, error=error)
                return {"error": error}
            if response.status_code == 429 and attempt < 3:
                time.sleep(attempt * 2.0)
                continue
            break
        if response is None:
            return {"error": "Kalshi request failed without a response"}

        if response.status_code >= 400:
            response_text = response.text
            error = f"Kalshi API returned HTTP {response.status_code}"
            logger.error("%s for %s params=%s response=%s", error, url, params, response_text[:500])
            self._record_failed_request(
                "GET",
                url,
                params,
                status_code=response.status_code,
                response_text=response_text,
                error=error,
            )
            return {"error": error, "status_code": response.status_code, "response_text": response_text}

        try:
            payload = response.json()
        except ValueError as exc:
            error = f"Kalshi API returned non-JSON response: {exc}"
            logger.error("%s for %s", error, url)
            self._record_failed_request(
                "GET",
                url,
                params,
                status_code=response.status_code,
                response_text=response.text,
                error=error,
            )
            return {"error": error, "status_code": response.status_code, "response_text": response.text}

        if not isinstance(payload, dict):
            return {"payload": payload}
        return payload

    def _get_with_urllib(
        self,
        path: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        query = urllib.parse.urlencode(params or {})
        request_url = f"{url}?{query}" if query else url
        request = urllib.request.Request(request_url, headers=headers or self._headers("GET", path)[0])
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_text = response.read().decode("utf-8")
                    status_code = response.status
                break
            except urllib.error.HTTPError as exc:
                response_text = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < 3:
                    time.sleep(attempt * 2.0)
                    continue
                error = f"Kalshi API returned HTTP {exc.code}"
                logger.error("%s for %s params=%s response=%s", error, url, params, response_text[:500])
                self._record_failed_request(
                    "GET",
                    url,
                    params,
                    status_code=exc.code,
                    response_text=response_text,
                    error=error,
                )
                return {"error": error, "status_code": exc.code, "response_text": response_text}
            except Exception as exc:
                if attempt < 3:
                    time.sleep(attempt * 2.0)
                    continue
                error = str(exc)
                logger.error("Kalshi request failed: %s params=%s error=%s", url, params, error)
                self._record_failed_request("GET", url, params, error=error)
                return {"error": error}

        if status_code >= 400:
            error = f"Kalshi API returned HTTP {status_code}"
            self._record_failed_request(
                "GET",
                url,
                params,
                status_code=status_code,
                response_text=response_text,
                error=error,
            )
            return {"error": error, "status_code": status_code, "response_text": response_text}
        try:
            payload = json.loads(response_text)
        except ValueError as exc:
            error = f"Kalshi API returned non-JSON response: {exc}"
            self._record_failed_request(
                "GET",
                url,
                params,
                status_code=status_code,
                response_text=response_text,
                error=error,
            )
            return {"error": error, "status_code": status_code, "response_text": response_text}
        return payload if isinstance(payload, dict) else {"payload": payload}

    def _paged_markets(self, path: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        request_params = dict(params or {})
        max_pages = int(request_params.pop("max_pages", 100))
        request_params.setdefault("limit", 1000)
        cursor = str(request_params.pop("cursor", "") or "")
        records: list[dict[str, Any]] = []
        last_error = ""

        for _ in range(max_pages):
            page_params = dict(request_params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._get(path, page_params)
            if payload.get("error"):
                last_error = str(payload["error"])
                break
            page_records = payload.get("markets") or payload.get("historical_markets") or []
            if isinstance(page_records, list):
                records.extend([record for record in page_records if isinstance(record, dict)])
            cursor = str(payload.get("cursor") or "")
            if not cursor:
                break

        frame = pd.DataFrame(records)
        if last_error:
            frame.attrs["error"] = last_error
        return _normalize_market_response(frame)

    def _paged_events(self, params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        request_params = dict(params or {})
        max_pages = int(request_params.pop("max_pages", 100))
        request_params.setdefault("limit", 200)
        cursor = str(request_params.pop("cursor", "") or "")
        events: list[dict[str, Any]] = []
        milestones: list[dict[str, Any]] = []
        last_error = ""

        for _ in range(max_pages):
            page_params = dict(request_params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._get("/events", page_params)
            if payload.get("error"):
                last_error = str(payload["error"])
                break
            page_events = payload.get("events") or []
            if isinstance(page_events, list):
                events.extend([event for event in page_events if isinstance(event, dict)])
            page_milestones = payload.get("milestones") or []
            if isinstance(page_milestones, list):
                milestones.extend([milestone for milestone in page_milestones if isinstance(milestone, dict)])
            cursor = str(payload.get("cursor") or "")
            if not cursor:
                break

        event_frame = pd.DataFrame(events)
        milestone_frame = pd.DataFrame(milestones)
        if last_error:
            event_frame.attrs["error"] = last_error
            milestone_frame.attrs["error"] = last_error
        return event_frame, milestone_frame

    def get_historical_cutoff(self) -> dict[str, Any]:
        return self._get("/historical/cutoff")

    def get_series_list(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        payload = self._get("/series", dict(params or {}))
        if payload.get("error"):
            frame = pd.DataFrame()
            frame.attrs["error"] = str(payload["error"])
            return frame
        records = payload.get("series") or []
        if not isinstance(records, list):
            return pd.DataFrame()
        return pd.DataFrame([record for record in records if isinstance(record, dict)])

    def get_events(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        events, _ = self._paged_events(params)
        return events

    def get_events_with_milestones(self, params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        request_params = dict(params or {})
        request_params["with_milestones"] = True
        return self._paged_events(request_params)

    def get_markets(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return self._paged_markets("/markets", params)

    def get_market(self, market_ticker: str) -> pd.DataFrame:
        payload = self._get(f"/markets/{market_ticker}")
        if payload.get("error"):
            frame = pd.DataFrame()
            frame.attrs["error"] = str(payload["error"])
            return frame
        record = payload.get("market") or payload
        if not isinstance(record, dict):
            return pd.DataFrame()
        return _normalize_market_response(pd.DataFrame([record]))

    def get_historical_markets(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return self._paged_markets("/historical/markets", params)

    def get_historical_market(self, market_ticker: str) -> pd.DataFrame:
        payload = self._get(f"/historical/markets/{market_ticker}")
        if payload.get("error"):
            frame = pd.DataFrame()
            frame.attrs["error"] = str(payload["error"])
            return frame
        record = payload.get("market") or payload.get("historical_market") or payload
        if not isinstance(record, dict):
            return pd.DataFrame()
        return _normalize_market_response(pd.DataFrame([record]))

    def get_market_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int,
    ) -> pd.DataFrame:
        payload = self._get(
            f"/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            {
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "period_interval": int(period_interval),
            },
        )
        return _candlestick_payload_to_frame(payload)

    def get_historical_market_candlesticks(
        self,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int,
    ) -> pd.DataFrame:
        payload = self._get(
            f"/historical/markets/{market_ticker}/candlesticks",
            {
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "period_interval": int(period_interval),
            },
        )
        return _candlestick_payload_to_frame(payload)


def _normalize_market_response(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return markets
    output = markets.copy()
    rename_map = {
        "ticker": "market_ticker",
        "market_ticker": "market_ticker",
        "series_ticker": "series_ticker",
        "event_ticker": "event_ticker",
        "title": "market_title",
        "subtitle": "market_subtitle",
    }
    output = output.rename(columns={key: value for key, value in rename_map.items() if key in output.columns})
    for column in [
        "open_time",
        "close_time",
        "expiration_time",
        "expected_expiration_time",
        "latest_expiration_time",
    ]:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce", utc=True)
    return output


def _flatten_candle_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = nested_value
        else:
            flattened[key] = value
    return flattened


def _candlestick_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    if payload.get("error"):
        frame = pd.DataFrame()
        frame.attrs["error"] = str(payload["error"])
        return frame

    records = (
        payload.get("candlesticks")
        or payload.get("candles")
        or payload.get("market_candlesticks")
        or []
    )
    if not isinstance(records, list) or not records:
        return pd.DataFrame()

    frame = pd.DataFrame([_flatten_candle_record(record) for record in records if isinstance(record, dict)])
    if frame.empty:
        return frame

    rename_map = {
        "ts": "snapshot_ts",
        "time": "snapshot_ts",
        "timestamp": "snapshot_ts",
        "end_period_ts": "end_ts",
        "yes_bid_close_dollars": "yes_bid",
        "yes_bid_close": "yes_bid",
        "yes_ask_close_dollars": "yes_ask",
        "yes_ask_close": "yes_ask",
        "price_close_dollars": "last_price",
        "price_close": "last_price",
        "last_price_close_dollars": "last_price",
        "last_price_close": "last_price",
        "yes_price_close_dollars": "yes_price",
        "yes_price_close": "yes_price",
        "volume_fp": "volume",
        "volume_close": "volume",
        "open_interest_fp": "open_interest",
        "open_interest_close": "open_interest",
    }
    frame = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})

    for column in ["snapshot_ts", "end_ts"]:
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                frame[column] = numeric

    for column in [
        "yes_bid",
        "yes_ask",
        "yes_price",
        "last_price",
        "volume",
        "open_interest",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _default_client() -> KalshiAPIClient:
    return KalshiAPIClient.from_env()


def get_historical_cutoff() -> dict[str, Any]:
    return _default_client().get_historical_cutoff()


def get_markets(params: dict[str, Any]) -> pd.DataFrame:
    return _default_client().get_markets(params)


def get_market(market_ticker: str) -> pd.DataFrame:
    return _default_client().get_market(market_ticker)


def get_historical_markets(params: dict[str, Any]) -> pd.DataFrame:
    return _default_client().get_historical_markets(params)


def get_historical_market(market_ticker: str) -> pd.DataFrame:
    return _default_client().get_historical_market(market_ticker)


def get_market_candlesticks(
    series_ticker: str,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> pd.DataFrame:
    return _default_client().get_market_candlesticks(
        series_ticker=series_ticker,
        market_ticker=market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )


def get_historical_market_candlesticks(
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> pd.DataFrame:
    return _default_client().get_historical_market_candlesticks(
        market_ticker=market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )


def kalshi_credentials_available() -> bool:
    """Return True when the expected optional Kalshi credentials are present."""

    _load_local_env_once()
    key_present = bool(os.getenv("KALSHI_API_KEY_ID") or os.getenv("KALSHI_API_KEY"))
    return bool(key_present and os.getenv("KALSHI_PRIVATE_KEY_PATH"))


def _to_cents(value: object) -> float | None:
    """Convert Kalshi dollar/string prices to cents."""

    if value is None or pd.isna(value):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def _best_public_price_source(row: dict[str, Any]) -> tuple[float | None, str]:
    bid = _to_cents(row.get("yes_bid_dollars") or row.get("yes_bid"))
    ask = _to_cents(row.get("yes_ask_dollars") or row.get("yes_ask"))
    last = _to_cents(row.get("last_price_dollars") or row.get("last_price"))

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, "public_bid_ask_mid"
    if last is not None:
        return last, "public_last_price"
    return None, "missing"


def _parse_kalshi_nba_game_ticker(ticker: str) -> dict[str, Any] | None:
    """Parse KXNBAGAME tickers like KXNBAGAME-26MAY11DETCLE-DET."""

    match = re.match(
        r"^KXNBAGAME-(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<day>\d{2})"
        r"(?P<away>[A-Z]{2,3})(?P<home>[A-Z]{2,3})-(?P<yes>[A-Z]{2,3})$",
        str(ticker),
    )
    if not match:
        return None

    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    year = 2000 + int(match.group("year"))
    month = month_map.get(match.group("month"))
    if month is None:
        return None
    game_date = pd.Timestamp(year=year, month=month, day=int(match.group("day")))
    return {
        "game_date": game_date.date().isoformat(),
        "away_team_abbr": _normalize_team(match.group("away")),
        "home_team_abbr": _normalize_team(match.group("home")),
        "yes_team_abbr": _normalize_team(match.group("yes")),
    }


def get_public_kalshi_markets(
    series_ticker: str = NBA_GAME_SERIES_TICKER,
    status: str | None = "open",
    limit: int = 1000,
    max_pages: int = 20,
    timeout: int = 30,
    api_base: str = KALSHI_PUBLIC_API_BASE,
) -> list[dict[str, Any]]:
    """Fetch public Kalshi markets for one series without credentials."""

    session = requests.Session()
    cursor = ""
    markets: list[dict[str, Any]] = []
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "limit": limit,
        }
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        response = session.get(f"{api_base}/markets", params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        markets.extend(payload.get("markets", []))
        cursor = str(payload.get("cursor") or "")
        if not cursor:
            break
    return markets


def public_kalshi_markets_to_game_rows(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert public KXNBAGAME market responses into local market CSV rows."""

    rows: list[dict[str, Any]] = []
    for market in markets:
        ticker = str(market.get("ticker", ""))
        parsed = _parse_kalshi_nba_game_ticker(ticker)
        if parsed is None:
            continue

        yes_team = parsed["yes_team_abbr"]
        teams = {parsed["home_team_abbr"], parsed["away_team_abbr"]}
        if yes_team not in teams:
            continue

        yes_bid = _to_cents(market.get("yes_bid_dollars") or market.get("yes_bid"))
        yes_ask = _to_cents(market.get("yes_ask_dollars") or market.get("yes_ask"))
        last_price = _to_cents(market.get("last_price_dollars") or market.get("last_price"))
        yes_mid, price_source = _best_public_price_source(market)
        no_team = next(team for team in teams if team != yes_team)
        settlement = str(market.get("result", "")).strip()
        close_price = last_price if str(market.get("status", "")).lower() in {"finalized", "settled"} else None

        rows.append(
            {
                "market_ticker": ticker,
                "event_ticker": market.get("event_ticker", ""),
                "game_date": parsed["game_date"],
                "home_team_abbr": parsed["home_team_abbr"],
                "away_team_abbr": parsed["away_team_abbr"],
                "market_question": market.get("title", ""),
                "yes_team_abbr": yes_team,
                "no_team_abbr": no_team,
                "yes_bid_cents": yes_bid,
                "yes_ask_cents": yes_ask,
                "yes_mid_cents": yes_mid,
                "close_price_cents": close_price,
                "settlement": settlement,
                "volume": pd.to_numeric(market.get("volume_fp"), errors="coerce"),
                "open_time": market.get("open_time", ""),
                "close_time": market.get("close_time", ""),
                "series_ticker": market.get("series_ticker") or NBA_GAME_SERIES_TICKER,
                "status": market.get("status", ""),
                "last_price_cents": last_price,
                "public_api_updated_time": market.get("updated_time", ""),
                "public_api_expected_expiration_time": market.get("expected_expiration_time", ""),
                "price_source": price_source,
            }
        )

    if not rows:
        return pd.DataFrame(columns=PUBLIC_MARKET_COLUMNS)
    return prepare_kalshi_markets(pd.DataFrame(rows)).reindex(columns=PUBLIC_MARKET_COLUMNS)


def fetch_public_nba_game_markets(
    statuses: list[str] | None = None,
    max_pages: int = 20,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch public Kalshi NBA game winner markets and normalize them."""

    statuses = statuses or ["open"]
    raw_markets: list[dict[str, Any]] = []
    for status in statuses:
        raw_markets.extend(
            get_public_kalshi_markets(
                series_ticker=NBA_GAME_SERIES_TICKER,
                status=status,
                max_pages=max_pages,
                timeout=timeout,
            )
        )

    rows = public_kalshi_markets_to_game_rows(raw_markets)
    if rows.empty:
        return rows
    rows = rows.drop_duplicates(subset=["market_ticker"], keep="first")
    return rows.sort_values(["game_date", "event_ticker", "market_ticker"]).reset_index(drop=True)


def load_mock_kalshi_markets(path: str | Path) -> pd.DataFrame:
    """Load local mock Kalshi markets from CSV."""

    market_path = Path(path)
    if not market_path.exists():
        raise FileNotFoundError(f"Mock Kalshi market file not found: {market_path}")

    markets = pd.read_csv(market_path)
    return prepare_kalshi_markets(markets)


def prepare_kalshi_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize local Kalshi-style market rows and fill usable mid prices."""

    markets = markets_df.copy()
    if "game_date" in markets.columns:
        markets["game_date"] = pd.to_datetime(markets["game_date"], errors="coerce").dt.normalize()

    if "yes_mid_cents" not in markets.columns:
        markets["yes_mid_cents"] = pd.NA

    raw_mid = pd.to_numeric(markets["yes_mid_cents"], errors="coerce")
    bid = (
        pd.to_numeric(markets["yes_bid_cents"], errors="coerce")
        if "yes_bid_cents" in markets.columns
        else pd.Series(float("nan"), index=markets.index, dtype="float64")
    )
    ask = (
        pd.to_numeric(markets["yes_ask_cents"], errors="coerce")
        if "yes_ask_cents" in markets.columns
        else pd.Series(float("nan"), index=markets.index, dtype="float64")
    )
    close = (
        pd.to_numeric(markets["close_price_cents"], errors="coerce")
        if "close_price_cents" in markets.columns
        else pd.Series(float("nan"), index=markets.index, dtype="float64")
    )

    filled_mid = raw_mid.copy()
    price_source = pd.Series("missing", index=markets.index, dtype="object")
    price_source.loc[raw_mid.notna()] = "yes_mid"

    spread_mid = (bid + ask) / 2.0
    spread_fill_mask = filled_mid.isna() & spread_mid.notna()
    filled_mid.loc[spread_fill_mask] = spread_mid.loc[spread_fill_mask]
    price_source.loc[spread_fill_mask] = "bid_ask_mid"

    close_fill_mask = filled_mid.isna() & close.notna()
    filled_mid.loc[close_fill_mask] = close.loc[close_fill_mask]
    price_source.loc[close_fill_mask] = "close_price"

    markets["yes_mid_cents"] = filled_mid
    if "price_source" not in markets.columns:
        markets["price_source"] = price_source

    for column in ["yes_bid_cents", "yes_ask_cents", "yes_mid_cents", "close_price_cents", "volume"]:
        if column in markets.columns:
            markets[column] = pd.to_numeric(markets[column], errors="coerce")
    return markets


def _normalize_team(value: object) -> str:
    return normalize_team_abbr(value)


def _actual_yes_win(row: pd.Series) -> bool:
    settlement = str(row.get("settlement", "")).strip().lower()
    if settlement in {"yes", "y", "1", "true", "win", "won"}:
        return True
    if settlement in {"no", "n", "0", "false", "loss", "lost"}:
        return False

    yes_team = _normalize_team(row["yes_team_abbr"])
    if yes_team == _normalize_team(row["home_team_abbr"]):
        return bool(row["actual_home_win"])
    if yes_team == _normalize_team(row["away_team_abbr"]):
        return not bool(row["actual_home_win"])
    raise ValueError(f"yes_team_abbr does not match game teams: {yes_team}")


def validate_kalshi_markets(
    markets_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a validation report for a local Kalshi-style market CSV."""

    report: dict[str, Any] = {
        "rows": int(len(markets_df)),
        "missing_columns": [],
        "invalid_price_rows": [],
        "invalid_team_rows": [],
        "invalid_date_rows": [],
        "duplicate_market_tickers": [],
        "matched_rows": None,
        "unmatched_rows": None,
        "issues": [],
    }

    missing = [column for column in MARKET_MATCH_COLUMNS if column not in markets_df.columns]
    report["missing_columns"] = missing
    if missing:
        report["issues"].append(f"Missing required columns: {missing}")
        return report

    markets = markets_df.copy()
    parsed_dates = pd.to_datetime(markets["game_date"], errors="coerce")
    invalid_dates = markets.index[parsed_dates.isna()].tolist()
    report["invalid_date_rows"] = [int(index) for index in invalid_dates]

    prices = pd.to_numeric(markets["yes_mid_cents"], errors="coerce")
    invalid_prices = markets.index[prices.isna() | (prices <= 0) | (prices >= 100)].tolist()
    report["invalid_price_rows"] = [int(index) for index in invalid_prices]

    for index, row in markets.iterrows():
        teams = {
            _normalize_team(row["home_team_abbr"]),
            _normalize_team(row["away_team_abbr"]),
        }
        if _normalize_team(row["yes_team_abbr"]) not in teams:
            report["invalid_team_rows"].append(int(index))

    duplicates = markets.loc[markets["market_ticker"].duplicated(), "market_ticker"].astype(str).tolist()
    report["duplicate_market_tickers"] = duplicates

    if invalid_dates:
        report["issues"].append(f"{len(invalid_dates)} rows have invalid game_date values.")
    if invalid_prices:
        report["issues"].append(f"{len(invalid_prices)} rows have invalid yes_mid_cents values.")
    if report["invalid_team_rows"]:
        report["issues"].append(
            f"{len(report['invalid_team_rows'])} rows have yes_team_abbr outside the game teams."
        )
    if duplicates:
        report["issues"].append(f"{len(duplicates)} duplicate market_ticker values found.")

    if predictions_df is not None and not markets.empty:
        matched = match_games_to_markets(predictions_df, markets)
        report["matched_rows"] = int(len(matched))
        report["unmatched_rows"] = int(len(markets) - len(matched))
        if report["unmatched_rows"]:
            report["issues"].append(f"{report['unmatched_rows']} rows did not match predictions.")

    return report


def match_games_to_markets(games_df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Match model predictions to local Kalshi-style markets by date and teams."""

    require_columns(
        games_df,
        [
            "game_id",
            "game_date",
            "home_team_abbr",
            "away_team_abbr",
            "model_home_win_prob",
            "model_away_win_prob",
        ],
        dataframe_name="games_df",
    )
    require_columns(
        markets_df,
        MARKET_MATCH_COLUMNS,
        dataframe_name="markets_df",
    )

    games = games_df.copy()
    markets = markets_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    markets["game_date"] = pd.to_datetime(markets["game_date"], errors="coerce").dt.normalize()

    for column in ["home_team_abbr", "away_team_abbr"]:
        games[column] = games[column].map(_normalize_team)
        markets[column] = markets[column].map(_normalize_team)
    markets["yes_team_abbr"] = markets["yes_team_abbr"].map(_normalize_team)

    direct = markets.merge(
        games,
        on=["game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
        suffixes=("", "_game"),
    )

    unmatched = direct["game_id"].isna()
    if unmatched.any():
        reversed_markets = markets.loc[unmatched].rename(
            columns={
                "home_team_abbr": "away_team_abbr",
                "away_team_abbr": "home_team_abbr",
            }
        )
        reversed_matches = reversed_markets.merge(
            games,
            on=["game_date", "home_team_abbr", "away_team_abbr"],
            how="left",
            suffixes=("", "_game"),
        )
        direct.loc[unmatched, games.columns] = reversed_matches[games.columns].to_numpy()

    matched = direct.dropna(subset=["game_id"]).copy()
    if matched.empty:
        return matched

    matched["model_yes_prob"] = matched.apply(
        lambda row: row["model_home_win_prob"]
        if row["yes_team_abbr"] == row["home_team_abbr"]
        else row["model_away_win_prob"],
        axis=1,
    )
    if "actual_home_win" in matched.columns and matched["actual_home_win"].notna().any():
        matched["actual_yes_win"] = matched.apply(
            lambda row: _actual_yes_win(row) if pd.notna(row.get("actual_home_win")) else pd.NA,
            axis=1,
        )
    elif "settlement" in matched.columns:
        settlement = matched["settlement"].astype(str).str.strip().str.lower()
        matched["actual_yes_win"] = settlement.isin(["yes", "y", "1", "true", "win", "won"]).astype("object")
        matched.loc[settlement.isin(["", "nan", "none"]), "actual_yes_win"] = pd.NA
    else:
        matched["actual_yes_win"] = pd.NA

    output_columns = [
        "game_id",
        "market_ticker",
        "event_ticker",
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "model_yes_prob",
        "yes_mid_cents",
        "price_source",
        "settlement",
        "actual_yes_win",
    ]
    available_columns = [column for column in output_columns if column in matched.columns]
    return matched[available_columns].reset_index(drop=True)


def build_market_entry_template(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    season: int | None = None,
    yes_side: str = "home",
) -> pd.DataFrame:
    """Build a manual market-entry CSV template from model predictions."""

    require_columns(
        predictions_df,
        [
            "game_id",
            "game_date",
            "home_team_abbr",
            "away_team_abbr",
            "model_home_win_prob",
            "model_away_win_prob",
        ],
        dataframe_name="predictions_df",
    )

    predictions = predictions_df.copy()
    predictions["game_date"] = pd.to_datetime(predictions["game_date"], errors="coerce").dt.normalize()
    if season is not None:
        predictions = predictions[predictions["season"].astype(int) == int(season)]
    if start_date is not None:
        predictions = predictions[predictions["game_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        predictions = predictions[predictions["game_date"] <= pd.Timestamp(end_date)]

    if yes_side not in {"home", "away", "both"}:
        raise ValueError("yes_side must be one of: home, away, both")

    rows: list[dict[str, Any]] = []
    for _, game in predictions.sort_values(["game_date", "game_id"]).iterrows():
        sides = ["home", "away"] if yes_side == "both" else [yes_side]
        for side in sides:
            yes_team = game["home_team_abbr"] if side == "home" else game["away_team_abbr"]
            no_team = game["away_team_abbr"] if side == "home" else game["home_team_abbr"]
            date_text = pd.Timestamp(game["game_date"]).strftime("%Y%m%d")
            rows.append(
                {
                    "market_ticker": f"MANUAL-{date_text}-{game['away_team_abbr']}{game['home_team_abbr']}-{yes_team}",
                    "event_ticker": f"MANUAL-{date_text}-{game['away_team_abbr']}{game['home_team_abbr']}",
                    "game_date": pd.Timestamp(game["game_date"]).date().isoformat(),
                    "home_team_abbr": game["home_team_abbr"],
                    "away_team_abbr": game["away_team_abbr"],
                    "market_question": f"Will {yes_team} beat {no_team}?",
                    "yes_team_abbr": yes_team,
                    "no_team_abbr": no_team,
                    "yes_bid_cents": "",
                    "yes_ask_cents": "",
                    "yes_mid_cents": "",
                    "close_price_cents": "",
                    "settlement": "",
                    "volume": "",
                    "open_time": "",
                    "close_time": "",
                    "model_home_win_prob": game["model_home_win_prob"],
                    "model_away_win_prob": game["model_away_win_prob"],
                }
            )

    template = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return template
