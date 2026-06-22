"""Offline tests for the API-Sports client (no network, no real key)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.apisports_client import (  # noqa: E402
    API_KEY_ENV,
    AUTH_HEADER,
    ApiSportsClient,
    extract_response,
)


def fake_http(responses: list[tuple[int, dict, str]]):
    calls: list[dict] = []

    def http_get(url: str, headers: dict, timeout: float):
        calls.append({"url": url, "headers": dict(headers)})
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index]

    http_get.calls = calls  # type: ignore[attr-defined]
    return http_get


class NoKeyTests(unittest.TestCase):
    def test_no_key_graceful_skip(self) -> None:
        http = fake_http([(200, {}, "{}")])
        client = ApiSportsClient(env={}, http_get=http)
        result = client.status("basketball")
        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertIn(API_KEY_ENV, result["error"])
        self.assertEqual(http.calls, [])


class AuthAndParsingTests(unittest.TestCase):
    def test_x_apisports_key_header(self) -> None:
        body = json.dumps({"get": "status", "errors": [], "results": 1,
                           "response": {"subscription": {"plan": "Free", "active": True}}})
        http = fake_http([(200, {}, body)])
        client = ApiSportsClient(api_key="abc123", http_get=http)
        result = client.status("basketball")
        self.assertTrue(result["ok"])
        call = http.calls[0]
        self.assertEqual(call["headers"][AUTH_HEADER], "abc123")
        self.assertNotIn("abc123", call["url"])
        self.assertIn("v1.basketball.api-sports.io", call["url"])

    def test_probe_response_parsing(self) -> None:
        body = json.dumps({
            "get": "bets", "errors": [], "results": 2,
            "response": [{"id": 1, "name": "Player Points"}, {"id": 2, "name": "Moneyline"}],
        })
        client = ApiSportsClient(api_key="k", http_get=fake_http([(200, {}, body)]))
        result = client.request("basketball", "bets")
        names = [b["name"] for b in extract_response(result["data"])]
        self.assertEqual(names, ["Player Points", "Moneyline"])

    def test_api_errors_in_200_response_flagged(self) -> None:
        body = json.dumps({
            "get": "games", "errors": {"plan": "Free plans do not have access to this season"},
            "results": 0, "response": [],
        })
        client = ApiSportsClient(api_key="k", http_get=fake_http([(200, {}, body)]))
        result = client.request("basketball", "games", {"league": 12})
        self.assertFalse(result["ok"])
        self.assertIn("plan", result["error"])
        # The payload is still preserved for the probe report.
        self.assertIsNotNone(result["data"])

    def test_unknown_api_rejected_without_request(self) -> None:
        http = fake_http([(200, {}, "{}")])
        client = ApiSportsClient(api_key="k", http_get=http)
        result = client.request("cricket", "status")
        self.assertFalse(result["ok"])
        self.assertIn("unknown_api", result["error"])
        self.assertEqual(http.calls, [])

    def test_extract_response_shapes(self) -> None:
        self.assertEqual(extract_response({"response": [1, 2]}), [1, 2])
        self.assertEqual(extract_response({"response": {"a": 1}}), [{"a": 1}])
        self.assertEqual(extract_response([3]), [3])
        self.assertEqual(extract_response(None), [])


class RetryTests(unittest.TestCase):
    def test_retries_5xx(self) -> None:
        http = fake_http([
            (503, {}, "unavailable"),
            (200, {}, json.dumps({"errors": [], "response": []})),
        ])
        client = ApiSportsClient(api_key="k", http_get=http, backoff_seconds=0,
                                 sleep=lambda s: None)
        result = client.status("basketball")
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
