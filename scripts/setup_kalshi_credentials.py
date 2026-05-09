"""Create local Kalshi credential files from a user-owned text export.

The script never prints the API key or private key. It writes:
- .secrets/kalshi_private_key.pem
- .env with KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads" / "Kalshi API.txt"
KEY_ID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN RSA PRIVATE KEY-----\s+.*?\s+-----END RSA PRIVATE KEY-----",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up local read-only Kalshi API credentials.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Text file containing the key ID and RSA private key.")
    parser.add_argument("--api-key-id", default=None, help="Kalshi API key ID if the source file only contains the private key.")
    parser.add_argument("--env-path", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--key-path", default=str(PROJECT_ROOT / ".secrets" / "kalshi_private_key.pem"))
    return parser.parse_args()


def _load_existing_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _write_env(path: Path, values: dict[str, str]) -> None:
    ordered_keys = [
        "KALSHI_API_KEY_ID",
        "KALSHI_PRIVATE_KEY_PATH",
        "KALSHI_ENV",
        "KALSHI_BASE_URL",
    ]
    lines = []
    for key in ordered_keys:
        if key in values:
            lines.append(f"{key}={values[key]}")
    for key in sorted(set(values) - set(ordered_keys)):
        lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"Credential source file not found: {source_path}")

    text = source_path.read_text(encoding="utf-8")
    key_match = KEY_ID_PATTERN.search(text)
    private_key_match = PRIVATE_KEY_PATTERN.search(text)
    api_key_id = args.api_key_id or (key_match.group(0) if key_match else None)
    if api_key_id is None:
        raise SystemExit("No Kalshi API key ID found in the source file.")
    if private_key_match is None:
        raise SystemExit("No RSA private key block found in the source file.")

    key_path = Path(args.key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(private_key_match.group(0).strip() + "\n", encoding="utf-8")
    try:
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    env_path = Path(args.env_path)
    env_values = _load_existing_env(env_path)
    env_values.update(
        {
            "KALSHI_API_KEY_ID": api_key_id,
            "KALSHI_PRIVATE_KEY_PATH": str(key_path.resolve()),
            "KALSHI_ENV": env_values.get("KALSHI_ENV", "prod"),
            "KALSHI_BASE_URL": env_values.get(
                "KALSHI_BASE_URL",
                "https://external-api.kalshi.com/trade-api/v2",
            ),
        }
    )
    _write_env(env_path, env_values)

    print(f"Wrote private key file: {key_path.resolve()}")
    print(f"Updated environment file: {env_path.resolve()}")
    print("Kalshi credentials are configured locally. No key material was printed.")


if __name__ == "__main__":
    main()
