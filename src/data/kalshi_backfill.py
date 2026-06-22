"""Backfill and filter Kalshi markets that may be NBA team-win markets."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_client import KalshiAPIClient
from .team_aliases import normalize_team_abbr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_KALSHI_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NBA_TEAM_ALIASES: dict[str, list[str]] = {
    "ATL": ["Atlanta Hawks", "Hawks", "Atlanta"],
    "BOS": ["Boston Celtics", "Celtics", "Boston"],
    "BKN": ["Brooklyn Nets", "Nets", "Brooklyn"],
    "CHA": ["Charlotte Hornets", "Hornets", "Charlotte"],
    "CHI": ["Chicago Bulls", "Bulls", "Chicago"],
    "CLE": ["Cleveland Cavaliers", "Cavaliers", "Cavs", "Cleveland"],
    "DAL": ["Dallas Mavericks", "Mavericks", "Mavs", "Dallas"],
    "DEN": ["Denver Nuggets", "Nuggets", "Denver"],
    "DET": ["Detroit Pistons", "Pistons", "Detroit"],
    "GSW": ["Golden State Warriors", "Warriors", "Golden State"],
    "HOU": ["Houston Rockets", "Rockets", "Houston"],
    "IND": ["Indiana Pacers", "Pacers", "Indiana"],
    "LAC": ["Los Angeles Clippers", "Clippers", "LA Clippers"],
    "LAL": ["Los Angeles Lakers", "Lakers", "LA Lakers"],
    "MEM": ["Memphis Grizzlies", "Grizzlies", "Memphis"],
    "MIA": ["Miami Heat", "Heat", "Miami"],
    "MIL": ["Milwaukee Bucks", "Bucks", "Milwaukee"],
    "MIN": ["Minnesota Timberwolves", "Timberwolves", "Wolves", "Minnesota"],
    "NOP": ["New Orleans Pelicans", "Pelicans", "New Orleans"],
    "NYK": ["New York Knicks", "Knicks", "New York"],
    "OKC": ["Oklahoma City Thunder", "Thunder", "OKC", "Oklahoma City"],
    "ORL": ["Orlando Magic", "Magic", "Orlando"],
    "PHI": ["Philadelphia 76ers", "76ers", "Sixers", "Philadelphia"],
    "PHX": ["Phoenix Suns", "Suns", "Phoenix"],
    "POR": ["Portland Trail Blazers", "Trail Blazers", "Blazers", "Portland"],
    "SAC": ["Sacramento Kings", "Kings", "Sacramento"],
    "SAS": ["San Antonio Spurs", "Spurs", "San Antonio"],
    "TOR": ["Toronto Raptors", "Raptors", "Toronto"],
    "UTA": ["Utah Jazz", "Jazz", "Utah"],
    "WAS": ["Washington Wizards", "Wizards", "Washington"],
}

REJECT_MARKET_TERMS = [
    "player",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "triple-double",
    "double-double",
    "spread",
    "total",
    "over",
    "under",
    "quarter",
    "half",
    "series",
    "championship",
    "finals",
    "conference",
    "season wins",
    "mvp",
]

BASKETBALL_TERMS = [
    "nba",
    "basketball",
    "win",
    "wins",
    "beat",
    "defeat",
]

KALSHI_MONTHS = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}
KALSHI_MONTH_LOOKUP = {value: key for key, value in KALSHI_MONTHS.items()}


def _date_to_ts(value: str | pd.Timestamp, end_of_day: bool = False) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if end_of_day:
        timestamp = timestamp.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return int(timestamp.timestamp())


def _parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def _market_text(markets: pd.DataFrame) -> pd.Series:
    if "market_title" in markets.columns:
        title = markets["market_title"]
    elif "title" in markets.columns:
        title = markets["title"]
    else:
        title = pd.Series("", index=markets.index)

    if "market_subtitle" in markets.columns:
        subtitle = markets["market_subtitle"]
    elif "subtitle" in markets.columns:
        subtitle = markets["subtitle"]
    else:
        subtitle = pd.Series("", index=markets.index)

    title = title.fillna("").astype(str)
    subtitle = subtitle.fillna("").astype(str)
    return (title + " " + subtitle).str.lower()


def _parse_kxnbagame_ticker(ticker: object) -> dict[str, Any] | None:
    match = re.match(
        r"^KXNBAGAME-(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<day>\d{2})"
        r"(?P<away>[A-Z]{2,3})(?P<home>[A-Z]{2,3})-(?P<yes>[A-Z]{2,3})$",
        str(ticker).upper(),
    )
    if not match:
        return None
    month = KALSHI_MONTH_LOOKUP.get(match.group("month"))
    if month is None:
        return None
    game_date = pd.Timestamp(
        year=2000 + int(match.group("year")),
        month=month,
        day=int(match.group("day")),
    )
    away = normalize_team_abbr(match.group("away"))
    home = normalize_team_abbr(match.group("home"))
    yes = normalize_team_abbr(match.group("yes"))
    no_team = home if yes == away else away if yes == home else ""
    return {
        "game_date": game_date.date().isoformat(),
        "away_team_abbr": away,
        "home_team_abbr": home,
        "yes_team_abbr": yes,
        "no_team_abbr": no_team,
        "series_ticker": "KXNBAGAME",
        "event_ticker": str(ticker).rsplit("-", 1)[0],
    }


def _fill_missing_kxnbagame_fields(markets: pd.DataFrame) -> pd.DataFrame:
    output = markets.copy()
    ticker_column = "market_ticker" if "market_ticker" in output.columns else "ticker" if "ticker" in output.columns else None
    if ticker_column is None:
        return output
    parsed = output[ticker_column].map(_parse_kxnbagame_ticker)
    parsed_df = pd.DataFrame([item or {} for item in parsed], index=output.index)
    for column in [
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
        "yes_team_abbr",
        "no_team_abbr",
        "series_ticker",
        "event_ticker",
    ]:
        if column not in parsed_df.columns:
            continue
        if column not in output.columns:
            output[column] = parsed_df[column]
        else:
            current = output[column]
            missing = current.isna() | current.astype(str).str.strip().eq("")
            output[column] = current.where(~missing, parsed_df[column])
    return output


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text


def teams_mentioned_in_text(text: str) -> list[str]:
    """Return NBA team abbreviations that appear in a market title/subtitle."""

    lowered = text.lower()
    found = []
    for abbr, aliases in NBA_TEAM_ALIASES.items():
        if any(_contains_phrase(lowered, alias) for alias in aliases):
            found.append(abbr)
    return found


def _save_dataframe_append(df: pd.DataFrame, path: Path, dedupe_column: str = "market_ticker") -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    existing = _read_dataframe_cache(path)
    if not existing.empty:
        output = pd.concat([existing, output], ignore_index=True)
    if dedupe_column in output.columns:
        output = output.drop_duplicates(subset=[dedupe_column], keep="last")
    _write_dataframe_cache(output, path)
    return output


def _read_dataframe_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError, RuntimeError):
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _is_missing_cache_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_cache_scalar(value: Any) -> Any:
    if _is_missing_cache_value(value):
        return pd.NA
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _normalize_cache_dataframe_for_write(df: pd.DataFrame) -> pd.DataFrame:
    """Make raw Kalshi API cache frames safe for parquet serialization.

    Kalshi market cache appends can mix older string/bytes values with newly
    parsed Timestamp values in the same object column. PyArrow cannot infer a
    stable parquet type from that mix, so object columns are serialized as
    nullable strings after normalizing common API payload types.
    """

    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_object_dtype(output[column]):
            output[column] = output[column].map(_normalize_cache_scalar).astype("string")
    return output


def _write_dataframe_cache(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = _normalize_cache_dataframe_for_write(df)
    if path.suffix.lower() == ".csv":
        output.to_csv(path, index=False)
        return path
    try:
        output.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError, RuntimeError):
        csv_path = path.with_suffix(".csv")
        output.to_csv(csv_path, index=False)
        return csv_path


def kalshi_event_ticker_for_game(game_date: str | pd.Timestamp, away_team_abbr: str, home_team_abbr: str) -> str:
    """Build the observed KXNBAGAME event ticker for a matchup."""

    date_value = pd.Timestamp(game_date)
    away = normalize_team_abbr(away_team_abbr)
    home = normalize_team_abbr(home_team_abbr)
    return f"KXNBAGAME-{date_value:%y}{KALSHI_MONTHS[date_value.month]}{date_value.day:02d}{away}{home}"


def generate_expected_nba_game_markets(
    nba_games_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate expected Kalshi NBA game market tickers from local NBA games."""

    if nba_games_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "home_team_abbr",
                "away_team_abbr",
                "market_ticker",
                "series_ticker",
                "event_ticker",
                "yes_team_abbr",
                "no_team_abbr",
            ]
        )

    games = nba_games_df.copy()
    games["game_date"] = _parse_dates(games["game_date"]).dt.normalize()
    games = games.dropna(subset=["game_date", "home_team_abbr", "away_team_abbr"])
    if start_date is not None:
        games = games[games["game_date"] >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        games = games[games["game_date"] <= pd.Timestamp(end_date).normalize()]

    rows: list[dict[str, Any]] = []
    for _, game in games.iterrows():
        home = normalize_team_abbr(game["home_team_abbr"])
        away = normalize_team_abbr(game["away_team_abbr"])
        event_ticker = kalshi_event_ticker_for_game(game["game_date"], away, home)
        for yes_team, no_team in [(away, home), (home, away)]:
            rows.append(
                {
                    "game_id": str(game.get("game_id", "")),
                    "game_date": pd.Timestamp(game["game_date"]).date().isoformat(),
                    "home_team_abbr": home,
                    "away_team_abbr": away,
                    "market_ticker": f"{event_ticker}-{yes_team}",
                    "series_ticker": "KXNBAGAME",
                    "event_ticker": event_ticker,
                    "yes_team_abbr": yes_team,
                    "no_team_abbr": no_team,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["market_ticker"]).reset_index(drop=True)


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _fetch_ticker_batches(
    client: KalshiAPIClient,
    tickers: list[str],
    historical: bool,
    batch_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch in _chunked(tickers, batch_size):
        params = {
            "tickers": ",".join(batch),
            "limit": min(max(len(batch), 1), 1000),
        }
        frame = client.get_historical_markets(params) if historical else client.get_markets(params)
        if not frame.empty:
            frame["kalshi_data_tier"] = "historical" if historical else "recent"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    if "market_ticker" in output.columns:
        output = output.drop_duplicates(subset=["market_ticker"], keep="last")
    return output.reset_index(drop=True)


def backfill_expected_game_markets(
    nba_games_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    client: KalshiAPIClient | None = None,
    batch_size: int = 50,
    output_path: str | Path | None = None,
    missing_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Targeted backfill by generating expected market tickers for every NBA game."""

    expected = generate_expected_nba_game_markets(nba_games_df, start_date=start_date, end_date=end_date)
    if expected.empty:
        return pd.DataFrame()

    kalshi = client or KalshiAPIClient.from_env()
    tickers = expected["market_ticker"].dropna().astype(str).unique().tolist()
    recent = _fetch_ticker_batches(kalshi, tickers, historical=False, batch_size=batch_size)
    historical = _fetch_ticker_batches(kalshi, tickers, historical=True, batch_size=batch_size)
    found = pd.concat([recent, historical], ignore_index=True)
    if not found.empty and "market_ticker" in found.columns:
        found = found.drop_duplicates(subset=["market_ticker"], keep="last")
        if "series_ticker" not in found.columns:
            found["series_ticker"] = "KXNBAGAME"
        if "event_ticker" not in found.columns:
            found["event_ticker"] = found["market_ticker"].astype(str).str.rsplit("-", n=1).str[0]
        found = found.merge(expected, on=["market_ticker"], how="left", suffixes=("", "_expected"))
        for column in [
            "series_ticker",
            "event_ticker",
            "game_id",
            "game_date",
            "home_team_abbr",
            "away_team_abbr",
            "yes_team_abbr",
            "no_team_abbr",
        ]:
            expected_column = f"{column}_expected"
            if expected_column in found.columns:
                if column in found.columns:
                    found[column] = found[column].where(found[column].notna(), found[expected_column])
                else:
                    found[column] = found[expected_column]

    found_path = Path(output_path) if output_path else RAW_KALSHI_DIR / "targeted_nba_game_markets.parquet"
    _write_dataframe_cache(found, found_path)

    missing = expected.copy()
    if not found.empty and "market_ticker" in found.columns:
        missing = missing[~missing["market_ticker"].isin(found["market_ticker"])].copy()
    missing_path = (
        Path(missing_output_path)
        if missing_output_path
        else PROJECT_ROOT / "data" / "reports" / "kalshi_expected_markets_missing.csv"
    )
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(missing_path, index=False)
    return found.reset_index(drop=True)


def _market_date_params(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> dict[str, Any]:
    return {
        "min_close_ts": _date_to_ts(start_date),
        "max_close_ts": _date_to_ts(end_date, end_of_day=True),
    }


def backfill_recent_markets(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    client: KalshiAPIClient | None = None,
    extra_params: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download current/recent markets and cache the raw response locally."""

    kalshi = client or KalshiAPIClient.from_env()
    params = _market_date_params(start_date, end_date)
    params.update(extra_params or {})
    markets = kalshi.get_markets(params)
    path = Path(output_path) if output_path else RAW_KALSHI_DIR / "live_markets.parquet"
    if not markets.empty:
        markets = _save_dataframe_append(markets, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            markets = _read_dataframe_cache(path)
        else:
            _write_dataframe_cache(markets, path)
    return markets


def backfill_historical_markets(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    client: KalshiAPIClient | None = None,
    extra_params: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download archived historical markets and cache the raw response locally."""

    kalshi = client or KalshiAPIClient.from_env()
    params = dict(extra_params or {})
    markets = kalshi.get_historical_markets(params)
    if not markets.empty:
        date_column = None
        for candidate in ["close_time", "expected_expiration_time", "expiration_time", "latest_expiration_time"]:
            if candidate in markets.columns:
                date_column = candidate
                break
        if date_column:
            market_dates = pd.to_datetime(markets[date_column], errors="coerce", utc=True).dt.tz_convert(None)
            start = pd.Timestamp(start_date).normalize()
            end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
            markets = markets[(market_dates.isna()) | ((market_dates >= start) & (market_dates < end))].copy()
    path = Path(output_path) if output_path else RAW_KALSHI_DIR / "historical_markets.parquet"
    if not markets.empty:
        markets = _save_dataframe_append(markets, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            markets = _read_dataframe_cache(path)
        else:
            _write_dataframe_cache(markets, path)
    return markets


def filter_possible_nba_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Keep likely single-game NBA team-win markets and reject props/totals/futures."""

    if markets_df.empty:
        return markets_df.copy()

    markets = markets_df.copy()
    ticker_source = (
        markets["market_ticker"]
        if "market_ticker" in markets.columns
        else markets["ticker"]
        if "ticker" in markets.columns
        else pd.Series("", index=markets.index)
    )
    ticker = ticker_source.fillna("").astype(str).str.upper()
    series = (
        markets["series_ticker"].fillna("").astype(str).str.upper()
        if "series_ticker" in markets.columns
        else ticker.str.extract(r"^([A-Z0-9]+)-", expand=False).fillna("")
    )
    is_kxnbagame = ticker.str.startswith("KXNBAGAME-") | series.eq("KXNBAGAME")
    text = _market_text(markets)
    mentioned_teams = text.map(teams_mentioned_in_text)
    has_team = mentioned_teams.map(bool)
    has_basketball_word = text.map(lambda value: any(term in value for term in BASKETBALL_TERMS))
    has_reject_term = text.map(lambda value: any(term in value for term in REJECT_MARKET_TERMS))

    output = markets.loc[is_kxnbagame & (has_team | has_basketball_word) & ~has_reject_term].copy()
    output = _fill_missing_kxnbagame_fields(output)
    output["mentioned_team_abbrs"] = mentioned_teams.loc[output.index].map(lambda teams: ",".join(teams))
    output["possible_nba_reason"] = "kxnbagame_team_or_basketball_text_without_prop_terms"
    return output.reset_index(drop=True)


def backfill_all_markets(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    client: KalshiAPIClient | None = None,
    extra_params: dict[str, Any] | None = None,
    nba_games_df: pd.DataFrame | None = None,
    use_targeted_ticker_backfill: bool = True,
    filtered_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download recent plus historical markets, filter likely NBA markets, and save them."""

    kalshi = client or KalshiAPIClient.from_env()
    recent = backfill_recent_markets(start_date, end_date, client=kalshi, extra_params=extra_params)
    historical = backfill_historical_markets(start_date, end_date, client=kalshi, extra_params=extra_params)
    frames = [recent, historical]
    if use_targeted_ticker_backfill and nba_games_df is not None and not nba_games_df.empty:
        targeted = backfill_expected_game_markets(
            nba_games_df,
            start_date=start_date,
            end_date=end_date,
            client=kalshi,
        )
        frames.append(targeted)
    combined = pd.concat(frames, ignore_index=True)
    if "market_ticker" in combined.columns:
        combined = combined.drop_duplicates(subset=["market_ticker"], keep="last")

    possible = filter_possible_nba_markets(combined)
    output_path = Path(filtered_output_path) if filtered_output_path else PROCESSED_DIR / "kalshi_possible_nba_markets.parquet"
    _write_dataframe_cache(possible, output_path)
    return possible
