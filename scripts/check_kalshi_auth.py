"""Verify local Kalshi authenticated GET setup without printing secrets."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.kalshi_client import KalshiAPIClient, kalshi_credentials_available  # noqa: E402


def main() -> None:
    if not kalshi_credentials_available():
        raise SystemExit("Kalshi credentials are not configured. Run scripts/setup_kalshi_credentials.py first.")

    client = KalshiAPIClient.from_env(timeout=20, require_auth=True)
    markets = client.get_markets({"series_ticker": "KXNBAGAME", "limit": 1, "max_pages": 1})
    if markets.attrs.get("error") or markets.empty:
        raise SystemExit("Authenticated Kalshi GET failed. Check cryptography install and credential validity.")

    print("Authenticated Kalshi GET succeeded.")
    print(f"Base URL: {client.base_url}")
    print("No key material was printed.")


if __name__ == "__main__":
    main()
