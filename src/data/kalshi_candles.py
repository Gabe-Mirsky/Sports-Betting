"""Download Kalshi candlesticks and extract pregame price snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .cache import read_dataframe, write_dataframe
from .kalshi_client import KalshiAPIClient, NBA_GAME_SERIES_TICKER


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CANDLE_DIR = PROJECT_ROOT / "data" / "raw" / "kalshi" / "candles"
PREGAME_PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "kalshi_pregame_prices.csv"
EASTERN = ZoneInfo("America/New_York")
SNAPSHOT_TARGETS = {
    "pregame_60m": 60,
    "pregame_30m": 30,
    "pregame_5m": 5,
}
BEST_PREGAME_TARGET = "pregame_best_le_120m"
BEST_PREGAME_WINDOW_MINUTES = 120


def _to_utc_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _game_start_timestamp(game: pd.Series) -> tuple[pd.Timestamp, str]:
    for column in ["game_start_time", "game_start_datetime", "game_datetime", "start_time"]:
        if column in game and pd.notna(game[column]):
            timestamp = pd.to_datetime(game[column], errors="coerce")
            if pd.notna(timestamp):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(EASTERN)
                else:
                    timestamp = timestamp.tz_convert(EASTERN)
                return timestamp.tz_convert("UTC"), "exact"

    game_date = pd.to_datetime(game["game_date"], errors="coerce")
    if pd.isna(game_date):
        raise ValueError(f"Cannot estimate game start time for game_id={game.get('game_id')}")
    if any([game_date.hour, game_date.minute, game_date.second]):
        if game_date.tzinfo is None:
            game_date = game_date.tz_localize(EASTERN)
        else:
            game_date = game_date.tz_convert(EASTERN)
        return game_date.tz_convert("UTC"), "game_date_time"

    estimated = pd.Timestamp(
        year=game_date.year,
        month=game_date.month,
        day=game_date.day,
        hour=19,
        minute=0,
        tz=EASTERN,
    )
    return estimated.tz_convert("UTC"), "estimated_7pm_eastern"


def _market_game_start_timestamp(match: pd.Series) -> tuple[pd.Timestamp | pd.NaT, str]:
    for column in ["expected_expiration_time", "expiration_time", "latest_expiration_time"]:
        if column in match and pd.notna(match[column]) and str(match[column]).strip():
            timestamp = pd.to_datetime(match[column], errors="coerce", utc=True)
            if pd.notna(timestamp):
                return timestamp, f"market_{column}"
    return pd.NaT, ""


def _extract_cutoff_ts(cutoff_payload: dict[str, Any]) -> int | None:
    for key in ["cutoff_ts", "historical_cutoff_ts", "market_settled_ts", "cutoff", "timestamp"]:
        value = cutoff_payload.get(key)
        if value is None:
            continue
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            return int(numeric)
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(timestamp):
            return int(timestamp.timestamp())
    return None


def _price_to_cents(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    number = float(numeric)
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def _with_normalized_candle_columns(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles
    output = candles.copy()
    if "snapshot_ts" not in output.columns and "end_ts" in output.columns:
        output["snapshot_ts"] = output["end_ts"]
    if "snapshot_ts" not in output.columns:
        output["snapshot_ts"] = pd.NA
    output["snapshot_ts"] = pd.to_numeric(output["snapshot_ts"], errors="coerce")
    for column in ["yes_bid", "yes_ask", "yes_price", "last_price", "volume", "open_interest"]:
        if column not in output.columns:
            output[column] = pd.NA
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["yes_bid"] = output["yes_bid"].map(_price_to_cents)
    output["yes_ask"] = output["yes_ask"].map(_price_to_cents)
    output["yes_price"] = output["yes_price"].map(_price_to_cents)
    output["last_price"] = output["last_price"].map(_price_to_cents)
    output["mid_price"] = (output["yes_bid"] + output["yes_ask"]) / 2.0
    return output


def _endpoint_order(game_start_ts: int, cutoff_ts: int | None) -> list[str]:
    if cutoff_ts is not None and game_start_ts <= cutoff_ts:
        return ["historical", "recent"]
    return ["recent", "historical"]


def _download_market_candles(
    client: KalshiAPIClient,
    market_ticker: str,
    series_ticker: str,
    start_ts: int,
    end_ts: int,
    intervals: list[int],
    cutoff_ts: int | None,
) -> pd.DataFrame:
    endpoint_order = _endpoint_order(end_ts, cutoff_ts)
    frames: list[pd.DataFrame] = []
    for interval in intervals:
        for endpoint in endpoint_order:
            if endpoint == "recent":
                candles = client.get_market_candlesticks(series_ticker, market_ticker, start_ts, end_ts, interval)
            else:
                candles = client.get_historical_market_candlesticks(market_ticker, start_ts, end_ts, interval)
            if candles.empty:
                continue
            candles = _with_normalized_candle_columns(candles)
            candles["market_ticker"] = market_ticker
            candles["series_ticker"] = series_ticker
            candles["period_interval"] = interval
            candles["endpoint_used"] = endpoint
            frames.append(candles)
            return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _download_batch_market_candles(
    client: KalshiAPIClient,
    market_tickers: list[str],
    start_ts: int,
    end_ts: int,
    interval: int,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download recent candles in ticker batches, returning an empty frame on unsupported clients."""

    if not hasattr(client, "get_batch_market_candlesticks"):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for index in range(0, len(market_tickers), batch_size):
        batch = market_tickers[index : index + batch_size]
        try:
            candles = client.get_batch_market_candlesticks(
                batch,
                start_ts,
                end_ts,
                interval,
            )
        except (AttributeError, TypeError, NotImplementedError):
            return pd.DataFrame()
        if candles.empty:
            continue
        candles = _with_normalized_candle_columns(candles)
        candles["period_interval"] = interval
        candles["endpoint_used"] = "batch_recent"
        frames.append(candles)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_candle_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return read_dataframe(path)
        except RuntimeError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            return pd.DataFrame()
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _write_candle_cache(candles: pd.DataFrame, path: Path) -> Path:
    try:
        return write_dataframe(candles, path)
    except RuntimeError:
        csv_path = path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        candles.to_csv(csv_path, index=False)
        return csv_path


def _extract_snapshot(
    candles: pd.DataFrame,
    game_id: str,
    market_ticker: str,
    series_ticker: str,
    game_start_ts: int,
    snapshot_target: str,
    minutes_before_tipoff: int,
    time_quality: str,
    max_minutes_before_tipoff: int | None = None,
) -> dict[str, Any]:
    base_row = {
        "game_id": game_id,
        "market_ticker": market_ticker,
        "series_ticker": series_ticker,
        "snapshot_target": snapshot_target,
        "snapshot_ts": "",
        "minutes_before_tipoff": minutes_before_tipoff,
        "yes_bid": "",
        "yes_ask": "",
        "yes_price": "",
        "last_price": "",
        "mid_price": "",
        "volume": "",
        "open_interest": "",
        "period_interval": "",
        "price_quality": "missing",
        "time_quality": time_quality,
    }
    if candles.empty or "snapshot_ts" not in candles.columns:
        return base_row

    usable = candles[pd.to_numeric(candles["snapshot_ts"], errors="coerce") <= game_start_ts].copy()
    usable = usable[pd.to_numeric(usable["snapshot_ts"], errors="coerce").notna()]
    if max_minutes_before_tipoff is not None:
        earliest_ts = game_start_ts - int(max_minutes_before_tipoff) * 60
        usable = usable[usable["snapshot_ts"].astype(float).ge(earliest_ts)].copy()
    if usable.empty:
        return base_row

    target_ts = game_start_ts - minutes_before_tipoff * 60
    usable["snapshot_distance"] = (usable["snapshot_ts"].astype(float) - target_ts).abs()
    candle = usable.sort_values(["snapshot_distance", "snapshot_ts"]).iloc[0]

    yes_bid = _price_to_cents(candle.get("yes_bid"))
    yes_ask = _price_to_cents(candle.get("yes_ask"))
    last_price = _price_to_cents(candle.get("last_price"))
    mid_price = _price_to_cents(candle.get("mid_price"))
    if mid_price is None and yes_bid is not None and yes_ask is not None:
        mid_price = (yes_bid + yes_ask) / 2.0

    period_interval = int(candle.get("period_interval")) if pd.notna(candle.get("period_interval")) else None
    price_quality = "missing"
    selected_price = None
    if yes_ask is not None:
        selected_price = yes_ask
        price_quality = "bid_ask_available"
    elif mid_price is not None:
        selected_price = mid_price
        price_quality = "mid_estimated"
    elif last_price is not None:
        selected_price = last_price
        price_quality = "last_price_only"

    if period_interval == 1440 and selected_price is not None:
        price_quality = "daily_candle_low_quality"

    snapshot_ts = int(candle["snapshot_ts"])
    actual_minutes_before = (game_start_ts - snapshot_ts) / 60.0
    return {
        **base_row,
        "snapshot_ts": snapshot_ts,
        "minutes_before_tipoff": round(actual_minutes_before, 2),
        "yes_bid": yes_bid if yes_bid is not None else "",
        "yes_ask": yes_ask if yes_ask is not None else "",
        "yes_price": selected_price if selected_price is not None else "",
        "last_price": last_price if last_price is not None else "",
        "mid_price": mid_price if mid_price is not None else "",
        "volume": candle.get("volume", ""),
        "open_interest": candle.get("open_interest", ""),
        "period_interval": period_interval if period_interval is not None else "",
        "price_quality": price_quality,
    }


def download_candles_for_matches(
    matches_df: pd.DataFrame,
    nba_games_df: pd.DataFrame,
    client: KalshiAPIClient | None = None,
    force: bool = False,
    candle_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    use_batch: bool = True,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download candles for matched markets and save extracted pregame prices."""

    kalshi = client or KalshiAPIClient.from_env()
    matches = matches_df.copy()
    games = nba_games_df.copy()
    if "match_status" in matches.columns:
        matches = matches[matches["match_status"] == "auto_matched"].copy()

    candle_root = Path(candle_dir) if candle_dir else RAW_CANDLE_DIR
    candle_root.mkdir(parents=True, exist_ok=True)
    prices_path = Path(output_path) if output_path else PREGAME_PRICES_PATH
    prices_path.parent.mkdir(parents=True, exist_ok=True)

    if matches.empty:
        empty = pd.DataFrame()
        empty.to_csv(prices_path, index=False)
        return empty

    games["game_id"] = games["game_id"].astype(str)
    game_lookup = games.set_index("game_id", drop=False)
    cutoff_ts: int | None = None

    rows: list[dict[str, Any]] = []
    intervals = [1, 60, 1440]
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    cache: dict[tuple[str, int, int, str], pd.DataFrame] = {}
    missing_downloads: list[dict[str, Any]] = []

    for _, match in matches.iterrows():
        game_id = str(match["game_id"])
        if game_id not in game_lookup.index:
            continue
        game = game_lookup.loc[game_id]
        if isinstance(game, pd.DataFrame):
            game = game.iloc[0]
        game_start, time_quality = _game_start_timestamp(game)
        game_start_ts = int(game_start.timestamp())
        start_ts = game_start_ts - 24 * 60 * 60
        market_ticker = str(match["market_ticker"])
        series_ticker = str(match.get("series_ticker") or NBA_GAME_SERIES_TICKER)
        candle_path = candle_root / f"{market_ticker}.parquet"

        if game_start_ts > now_ts:
            cache[(game_id, game_start_ts, 0, market_ticker)] = pd.DataFrame()
            continue

        if not force:
            candles = _read_candle_cache(candle_path)
            candles = _with_normalized_candle_columns(candles)
        else:
            candles = pd.DataFrame()

        if candles.empty:
            missing_downloads.append(
                {
                    "game_id": game_id,
                    "market_ticker": market_ticker,
                    "series_ticker": series_ticker,
                    "start_ts": start_ts,
                    "end_ts": game_start_ts,
                    "candle_path": candle_path,
                }
            )
        cache[(game_id, game_start_ts, start_ts, market_ticker)] = candles

    if missing_downloads and cutoff_ts is None:
        cutoff_ts = _extract_cutoff_ts(kalshi.get_historical_cutoff())

    if use_batch and missing_downloads:
        for interval in intervals:
            remaining = [
                spec for spec in missing_downloads
                if cache[(spec["game_id"], spec["end_ts"], spec["start_ts"], spec["market_ticker"])].empty
            ]
            groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for spec in remaining:
                if _endpoint_order(int(spec["end_ts"]), cutoff_ts)[0] != "recent":
                    continue
                groups.setdefault((int(spec["start_ts"]), int(spec["end_ts"])), []).append(spec)
            for (start_ts, end_ts), group in groups.items():
                tickers = [str(spec["market_ticker"]) for spec in group]
                batch_candles = _download_batch_market_candles(
                    kalshi,
                    tickers,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    interval=interval,
                    batch_size=batch_size,
                )
                if batch_candles.empty or "market_ticker" not in batch_candles.columns:
                    continue
                for spec in group:
                    market_ticker = str(spec["market_ticker"])
                    candles = batch_candles[batch_candles["market_ticker"].astype(str).eq(market_ticker)].copy()
                    if candles.empty:
                        continue
                    series_ticker = str(spec["series_ticker"])
                    candles["series_ticker"] = series_ticker
                    cache[(spec["game_id"], spec["end_ts"], spec["start_ts"], market_ticker)] = candles
                    _write_candle_cache(candles, Path(spec["candle_path"]))

    for spec in missing_downloads:
        key = (spec["game_id"], spec["end_ts"], spec["start_ts"], spec["market_ticker"])
        if not cache[key].empty:
            continue
        candles = _download_market_candles(
            kalshi,
            market_ticker=str(spec["market_ticker"]),
            series_ticker=str(spec["series_ticker"]),
            start_ts=int(spec["start_ts"]),
            end_ts=int(spec["end_ts"]),
            intervals=intervals,
            cutoff_ts=cutoff_ts,
        )
        if not candles.empty:
            _write_candle_cache(candles, Path(spec["candle_path"]))
        cache[key] = candles

    for _, match in matches.iterrows():
        game_id = str(match["game_id"])
        if game_id not in game_lookup.index:
            continue
        game = game_lookup.loc[game_id]
        if isinstance(game, pd.DataFrame):
            game = game.iloc[0]
        game_start, time_quality = _game_start_timestamp(game)
        game_start_ts = int(game_start.timestamp())
        start_ts = game_start_ts - 24 * 60 * 60
        market_ticker = str(match["market_ticker"])
        series_ticker = str(match.get("series_ticker") or NBA_GAME_SERIES_TICKER)
        future_time_quality = f"{time_quality};future_game_not_downloaded"
        if game_start_ts > now_ts:
            candles = pd.DataFrame()
            row_time_quality = future_time_quality
        else:
            candles = cache.get((game_id, game_start_ts, start_ts, market_ticker), pd.DataFrame())
            row_time_quality = time_quality

        for snapshot_target, minutes in SNAPSHOT_TARGETS.items():
            rows.append(
                _extract_snapshot(
                    candles=candles,
                    game_id=game_id,
                    market_ticker=market_ticker,
                    series_ticker=series_ticker,
                    game_start_ts=game_start_ts,
                    snapshot_target=snapshot_target,
                    minutes_before_tipoff=minutes,
                    time_quality=row_time_quality,
                )
            )
        rows.append(
            _extract_snapshot(
                candles=candles,
                game_id=game_id,
                market_ticker=market_ticker,
                series_ticker=series_ticker,
                game_start_ts=game_start_ts,
                snapshot_target=BEST_PREGAME_TARGET,
                minutes_before_tipoff=0,
                time_quality=row_time_quality,
                max_minutes_before_tipoff=BEST_PREGAME_WINDOW_MINUTES,
            )
        )

    prices = pd.DataFrame(rows)
    prices.to_csv(prices_path, index=False)
    return prices
