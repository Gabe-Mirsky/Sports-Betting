"""Offline tests for the SportsGameOdds client (no network, no real key)."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.sportsgameodds_client import (  # noqa: E402
    API_KEY_ENV,
    SportsGameOddsClient,
    extract_items,
    read_api_key,
    save_raw_payload,
)


def fake_http(responses: list[tuple[int, dict, str]]):
    """Build an injectable http_get returning queued (status, headers, body)."""

    calls: list[dict] = []

    def http_get(url: str, headers: dict, timeout: float):
        calls.append({"url": url, "headers": dict(headers)})
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index]

    http_get.calls = calls  # type: ignore[attr-defined]
    return http_get


class NoKeyTests(unittest.TestCase):
    def test_read_api_key_empty_env(self) -> None:
        self.assertEqual(read_api_key(env={}), "")

    def test_no_key_graceful_skip_without_requests(self) -> None:
        http = fake_http([(200, {}, "{}")])
        client = SportsGameOddsClient(env={}, http_get=http)
        result = client.request("events", {"leagueID": "NBA"})
        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertIn(API_KEY_ENV, result["error"])
        self.assertEqual(http.calls, [])  # no network traffic at all

    def test_paginated_no_key_skips(self) -> None:
        client = SportsGameOddsClient(env={}, http_get=fake_http([(200, {}, "{}")]))
        result = client.get_paginated("events")
        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["items"], [])


class AuthAndParsingTests(unittest.TestCase):
    def test_auth_header_sent_and_key_never_in_url(self) -> None:
        http = fake_http([(200, {}, json.dumps({"success": True, "data": []}))])
        client = SportsGameOddsClient(api_key="secret-key-123", http_get=http)
        result = client.events(leagueID="NBA", limit=2)
        self.assertTrue(result["ok"])
        call = http.calls[0]
        self.assertEqual(call["headers"]["X-API-Key"], "secret-key-123")
        self.assertNotIn("secret-key-123", call["url"])
        self.assertIn("leagueID=NBA", call["url"])

    def test_usage_envelope_parsing(self) -> None:
        usage = {"success": True, "data": {"tier": "amateur", "rateLimits": {"per-month": {"max-entities": 2500}}}}
        client = SportsGameOddsClient(api_key="k", http_get=fake_http([(200, {}, json.dumps(usage))]))
        result = client.account_usage()
        items, cursor = extract_items(result["data"])
        self.assertTrue(result["ok"])
        self.assertIsNone(cursor)
        self.assertEqual(items[0]["tier"], "amateur")

    def test_extract_items_shapes(self) -> None:
        self.assertEqual(extract_items([1, 2]), ([1, 2], None))
        items, cursor = extract_items({"success": True, "data": [{"a": 1}], "nextCursor": "abc"})
        self.assertEqual(items, [{"a": 1}])
        self.assertEqual(cursor, "abc")
        items, cursor = extract_items({"data": {"only": "one"}})
        self.assertEqual(items, [{"only": "one"}])
        self.assertIsNone(cursor)

    def test_sports_and_leagues_parsing(self) -> None:
        payload = {"success": True, "data": [{"sportID": "BASKETBALL"}, {"sportID": "HOCKEY"}]}
        client = SportsGameOddsClient(api_key="k", http_get=fake_http([(200, {}, json.dumps(payload))]))
        items, _ = extract_items(client.sports()["data"])
        self.assertEqual([s["sportID"] for s in items], ["BASKETBALL", "HOCKEY"])


class RetryAndErrorTests(unittest.TestCase):
    def test_retries_429_then_succeeds(self) -> None:
        http = fake_http([
            (429, {"retry-after": "0"}, "slow down"),
            (200, {}, json.dumps({"success": True, "data": [{"eventID": "e1"}]})),
        ])
        sleeps: list[float] = []
        client = SportsGameOddsClient(
            api_key="k", http_get=http, max_retries=2, backoff_seconds=0.01,
            sleep=sleeps.append,
        )
        result = client.events()
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(sleeps), 1)

    def test_non_retryable_error_returns_error_object(self) -> None:
        http = fake_http([(401, {}, "bad key")])
        client = SportsGameOddsClient(api_key="k", http_get=http, sleep=lambda s: None)
        result = client.events()
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], 401)
        self.assertIn("http_401", result["error"])
        self.assertEqual(len(http.calls), 1)  # 401 is not retried

    def test_network_error_never_raises(self) -> None:
        def boom(url: str, headers: dict, timeout: float):
            raise OSError("connection refused")

        client = SportsGameOddsClient(api_key="k", http_get=boom, max_retries=1,
                                      backoff_seconds=0, sleep=lambda s: None)
        result = client.events()
        self.assertFalse(result["ok"])
        self.assertIn("network_error", result["error"])


class PaginationTests(unittest.TestCase):
    def test_follows_next_cursor_with_cap(self) -> None:
        pages = [
            (200, {}, json.dumps({"success": True, "data": [{"i": 1}], "nextCursor": "c2"})),
            (200, {}, json.dumps({"success": True, "data": [{"i": 2}], "nextCursor": None})),
        ]
        http = fake_http(pages)
        client = SportsGameOddsClient(api_key="k", http_get=http)
        result = client.get_paginated("events", {"leagueID": "NBA"}, max_pages=5)
        self.assertTrue(result["ok"])
        self.assertEqual([item["i"] for item in result["items"]], [1, 2])
        self.assertEqual(result["pages"], 2)
        self.assertIn("cursor=c2", http.calls[1]["url"])


class RawSavingTests(unittest.TestCase):
    def test_save_raw_never_overwrites(self) -> None:
        target = PROJECT_ROOT / "data" / "reports" / "_test_sgo_raw"
        if target.exists():
            shutil.rmtree(target)
        try:
            first = save_raw_payload({"a": 1}, target, "events")
            second = save_raw_payload({"a": 2}, target, "events")
            self.assertTrue(first.exists() and second.exists())
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text(encoding="utf-8")), {"a": 1})
            self.assertEqual(json.loads(second.read_text(encoding="utf-8")), {"a": 2})
        finally:
            shutil.rmtree(target, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
