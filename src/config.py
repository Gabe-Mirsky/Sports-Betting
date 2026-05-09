"""Configuration loading for the NBA Kalshi predictor project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal local runtimes
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised in minimal local runtimes
    def load_dotenv(dotenv_path: object = None, *args: object, **kwargs: object) -> bool:
        path = Path(dotenv_path) if dotenv_path else PROJECT_ROOT / ".env"
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return True


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ConfigModel(BaseModel):
    class Config:
        extra = "ignore"


class ProjectConfig(_ConfigModel):
    random_seed: int = 42


class DataConfig(_ConfigModel):
    start_season: int = 2018
    end_season: int = 2025
    cache_dir: str = "data/raw/nba"
    player_cache_dir: str = "data/raw/nba/player"
    availability_report_path: str = "data/raw/nba/injuries/availability.csv"
    processed_dir: str = "data/processed"


class NBAConfig(_ConfigModel):
    start_season: int = 2018
    end_season: int = 2025
    season_type: str = "Regular Season"
    raw_dir: str = "data/raw/nba"


class ModelConfig(_ConfigModel):
    target: str = "target_home_win"
    train_start_season: int = 2018
    train_end_season: int = 2023
    test_season: int = 2024
    model_type: str = "logistic_regression"


class StrategyConfig(_ConfigModel):
    starting_bankroll: float = 100.0
    edge_threshold: float = 0.05
    max_bet_fraction: float = 0.03
    min_market_price: float = 0.05
    max_market_price: float = 0.95
    use_yes_ask_for_buys: bool = True
    allow_no_trades: bool = False
    use_fractional_kelly: bool = False
    kelly_fraction: float = 0.25


class KalshiConfig(_ConfigModel):
    enabled: bool = True
    env: str = "demo"
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    demo_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    use_demo: bool = False
    require_auth: bool = False
    default_candle_interval_minutes: int = 1
    fallback_candle_interval_minutes: int = 60
    market_search_days_before_game: int = 1
    market_search_days_after_game: int = 1
    api_key_id: str | None = None
    api_key: str | None = None
    private_key_path: str | None = None


class MatchingConfig(_ConfigModel):
    auto_match_threshold: float = 0.85
    review_match_threshold: float = 0.60


class BacktestConfig(_ConfigModel):
    pregame_snapshot_minutes_before_tipoff: int = 60
    min_volume: int = 10
    allowed_price_qualities: str = "bid_ask_available"
    require_bid_ask: bool = True
    max_candle_interval_minutes: int = 60


class AppConfig(_ConfigModel):
    project: ProjectConfig = ProjectConfig()
    nba: NBAConfig = NBAConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    strategy: StrategyConfig = StrategyConfig()
    kalshi: KalshiConfig = KalshiConfig()
    matching: MatchingConfig = MatchingConfig()
    backtest: BacktestConfig = BacktestConfig()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if yaml is None:
        data = _read_simple_yaml_mapping(text, path)
    else:
        data = yaml.safe_load(text) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return data


def _parse_simple_scalar(value: str) -> Any:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        return cleaned[1:-1]
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _read_simple_yaml_mapping(text: str, path: Path) -> dict[str, Any]:
    """Read the simple two-level config.yaml shape when PyYAML is unavailable."""

    data: dict[str, Any] = {}
    current_section: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        line = stripped.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported config line {line_number} in {path}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0 and not value:
            data[key] = {}
            current_section = key
        elif indent == 0:
            data[key] = _parse_simple_scalar(value)
            current_section = None
        elif current_section is not None:
            section = data.setdefault(current_section, {})
            if not isinstance(section, dict):
                raise ValueError(f"Config section must be a mapping: {current_section}")
            section[key] = _parse_simple_scalar(value)
        else:
            raise ValueError(f"Nested config value appears before a section on line {line_number} in {path}")
    return data


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    kalshi_config = dict(config.get("kalshi", {}))

    if os.getenv("KALSHI_ENV"):
        kalshi_config["env"] = os.environ["KALSHI_ENV"]
        kalshi_config["use_demo"] = os.environ["KALSHI_ENV"].strip().lower() == "demo"

    if os.getenv("KALSHI_API_KEY_ID"):
        kalshi_config["api_key_id"] = os.environ["KALSHI_API_KEY_ID"]

    if os.getenv("KALSHI_API_KEY"):
        kalshi_config["api_key"] = os.environ["KALSHI_API_KEY"]

    if os.getenv("KALSHI_PRIVATE_KEY_PATH"):
        kalshi_config["private_key_path"] = os.environ["KALSHI_PRIVATE_KEY_PATH"]

    if os.getenv("KALSHI_ENABLED"):
        kalshi_config["enabled"] = os.environ["KALSHI_ENABLED"].strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    config["kalshi"] = kalshi_config
    return config


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load YAML config and optional environment overrides."""

    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    path = path.resolve()
    load_dotenv(path.parent / ".env")

    raw_config = _read_yaml(path)
    raw_config = _apply_env_overrides(raw_config)
    return AppConfig(**raw_config)
