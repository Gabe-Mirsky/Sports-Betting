"""Catalog of candidate data sources for NBA player-prop modeling.

This is the single source of truth behind ``data/reports/data_source_audit.*``.
It records, for each free / free-tier source, what it costs, what it covers,
whether it carries player props / historical odds / closing prices, its rate
limits, and which of the fields the model needs it can actually supply.

Facts here reflect each vendor's documented free tier as of the project's last
review. Treat ``rate_limits`` and free-tier quotas as guidance to verify before
heavy use, not guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldRequirement:
    """One field the NBA player-prop model needs, and how important it is."""

    name: str
    importance: str  # "required" | "important" | "optional"
    group: str
    description: str


# Master list of fields the NBA player-prop pipeline needs. Sources are scored
# against this list to compute coverage and gaps.
NBA_PLAYER_PROP_FIELDS: tuple[FieldRequirement, ...] = (
    FieldRequirement("player_id", "required", "identity", "Stable player identifier for joins across sources."),
    FieldRequirement("player_name", "required", "identity", "Human-readable player name."),
    FieldRequirement("team_abbr", "required", "identity", "Player's team abbreviation."),
    FieldRequirement("opponent_abbr", "important", "identity", "Opponent team abbreviation."),
    FieldRequirement("position", "optional", "identity", "Player position for role context."),
    FieldRequirement("game_id", "required", "context", "Stable game identifier."),
    FieldRequirement("game_date", "required", "context", "Date of the game."),
    FieldRequirement("season", "required", "context", "Season start year."),
    FieldRequirement("is_home", "important", "context", "Whether the player's team is home."),
    FieldRequirement("rest_days", "optional", "context", "Days of rest / back-to-back flag."),
    FieldRequirement("projected_minutes", "optional", "context", "Pre-game minutes projection."),
    FieldRequirement("injury_status", "important", "availability", "Active / questionable / out status."),
    FieldRequirement("actual_minutes", "important", "actuals", "Minutes played (settles/backtests minutes-driven props)."),
    FieldRequirement("actual_points", "required", "actuals", "Points scored (settles points props)."),
    FieldRequirement("actual_rebounds", "important", "actuals", "Rebounds (settles rebound props)."),
    FieldRequirement("actual_assists", "important", "actuals", "Assists (settles assist props)."),
    FieldRequirement("actual_threes", "important", "actuals", "Made threes (settles 3PT props)."),
    FieldRequirement("actual_blocks", "optional", "actuals", "Blocks (settles block props)."),
    FieldRequirement("actual_steals", "optional", "actuals", "Steals (settles steal props)."),
    FieldRequirement("actual_turnovers", "optional", "actuals", "Turnovers (settles turnover props)."),
    FieldRequirement("prop_stat_type", "required", "market", "Which stat the prop is on (points, rebounds...)."),
    FieldRequirement("prop_line", "required", "market", "The over/under line value."),
    FieldRequirement("over_price", "required", "market", "Price for the Over side."),
    FieldRequirement("under_price", "required", "market", "Price for the Under side."),
    FieldRequirement("book_name", "important", "market", "Sportsbook / exchange the price came from."),
    FieldRequirement("odds_snapshot_time", "important", "market", "Timestamp of the odds snapshot."),
    FieldRequirement("implied_prob", "important", "market", "Implied probability from price."),
    FieldRequirement("no_vig_prob", "optional", "market", "Vig-removed fair probability."),
    FieldRequirement("is_closing", "important", "market", "Whether the price is the closing line."),
    FieldRequirement("settlement_value", "important", "settlement", "Final settled stat value / market result."),
)

_REQUIRED_FIELD_NAMES = frozenset(f.name for f in NBA_PLAYER_PROP_FIELDS if f.importance == "required")
_IMPORTANT_FIELD_NAMES = frozenset(f.name for f in NBA_PLAYER_PROP_FIELDS if f.importance == "important")
_ALL_FIELD_NAMES = frozenset(f.name for f in NBA_PLAYER_PROP_FIELDS)


@dataclass(frozen=True)
class DataSource:
    """A candidate data source and everything the audit needs to report on it."""

    key: str
    name: str
    cost: str  # "free" | "free_tier" | "paid"
    data_types: tuple[str, ...]
    sports_covered: tuple[str, ...]
    supports_player_props: bool
    supports_historical_odds: bool
    supports_closing_prices: bool
    rate_limits: str
    limitations: str
    fields_available: tuple[str, ...]
    adapter_capabilities: frozenset[str]
    priority: str  # "P0" (highest) .. "P3"
    priority_reason: str
    integration_status: str  # "implemented" | "wraps_existing" | "planned"
    role: str  # short statement of what we use it for
    notes: str = ""
    _unknown_fields: tuple[str, ...] = field(default=(), repr=False)

    def missing_required_fields(self) -> tuple[str, ...]:
        return tuple(sorted(_REQUIRED_FIELD_NAMES - set(self.fields_available)))

    def missing_important_fields(self) -> tuple[str, ...]:
        return tuple(sorted(_IMPORTANT_FIELD_NAMES - set(self.fields_available)))

    def missing_fields(self) -> tuple[str, ...]:
        return tuple(sorted(_ALL_FIELD_NAMES - set(self.fields_available)))

    def unknown_field_names(self) -> tuple[str, ...]:
        """Declared available fields that are not in the master requirement list."""

        return tuple(sorted(set(self.fields_available) - _ALL_FIELD_NAMES))


# ---------------------------------------------------------------------------
# The catalog. Free / free-tier sources only, ordered by modeling priority.
# ---------------------------------------------------------------------------
SOURCE_CATALOG: tuple[DataSource, ...] = (
    DataSource(
        key="nba_api",
        name="nba_api (NBA.com Stats)",
        cost="free",
        data_types=("player_game_logs", "team_stats", "box_scores", "schedule", "advanced_stats"),
        sports_covered=("NBA", "WNBA", "G-League"),
        supports_player_props=False,
        supports_historical_odds=False,
        supports_closing_prices=False,
        rate_limits=(
            "No official quota, but NBA.com throttles aggressive use; needs a browser-like "
            "User-Agent and ~0.6-2s sleeps. Endpoints time out intermittently."
        ),
        limitations="Provides stat actuals, not betting markets. Unofficial API; endpoints can change.",
        fields_available=(
            "player_id",
            "player_name",
            "team_abbr",
            "opponent_abbr",
            "position",
            "game_id",
            "game_date",
            "season",
            "is_home",
            "rest_days",
            "actual_minutes",
            "actual_points",
            "actual_rebounds",
            "actual_assists",
            "actual_threes",
            "actual_blocks",
            "actual_steals",
            "actual_turnovers",
            "settlement_value",
        ),
        adapter_capabilities=frozenset(
            {"fetch_events", "fetch_players", "fetch_player_game_logs", "fetch_results", "normalize_to_project_schema"}
        ),
        priority="P0",
        priority_reason="Free, complete box-score actuals that settle every prop and power features.",
        integration_status="wraps_existing",
        role="Player/team actuals, schedules, and features (settles props).",
        notes="Already used by src/data/nba_client.py and src/data/player_client.py.",
    ),
    DataSource(
        key="kalshi",
        name="Kalshi API (markets + candlesticks)",
        cost="free",
        data_types=("market_prices", "candlesticks", "order_book_top", "settlements"),
        sports_covered=("NBA", "NFL", "many_event_markets"),
        supports_player_props=True,
        supports_historical_odds=True,
        supports_closing_prices=True,
        rate_limits=(
            "Public read access with tiered per-second limits; candlestick history bounded by a "
            "historical cutoff (get_historical_cutoff). RSA key only needed for trading."
        ),
        limitations=(
            "Player-prop market coverage is thinner/newer than sportsbooks. Old markets can be "
            "missing or illiquid. Yes/No contracts must be mapped to Over/Under per market."
        ),
        fields_available=(
            "game_id",
            "game_date",
            "season",
            "prop_stat_type",
            "prop_line",
            "over_price",
            "under_price",
            "book_name",
            "odds_snapshot_time",
            "implied_prob",
            "is_closing",
            "settlement_value",
        ),
        adapter_capabilities=frozenset(
            {"fetch_events", "fetch_market_odds", "fetch_closing_prices", "fetch_results", "normalize_to_project_schema"}
        ),
        priority="P0",
        priority_reason="The prediction-market target. Provides historical prices and true closing lines.",
        integration_status="wraps_existing",
        role="Prediction-market prices, candlestick history, and closing lines (CLV).",
        notes="Already used by src/data/kalshi_client.py and src/data/kalshi_candles.py.",
    ),
    DataSource(
        key="odds_api",
        name="The Odds API (the-odds-api.com)",
        cost="free_tier",
        data_types=("moneyline", "spreads", "totals", "player_props", "current_snapshots"),
        sports_covered=("NBA", "NFL", "MLB", "NHL", "soccer", "many"),
        supports_player_props=True,
        supports_historical_odds=False,
        supports_closing_prices=False,
        rate_limits=(
            "Free tier ~500 requests/month. Each region+market multiplies credit cost; player-prop "
            "(featured/additional) markets cost more credits. Remaining quota in response headers."
        ),
        limitations=(
            "Free tier is effectively current/upcoming snapshots only; historical odds is a paid "
            "add-on. No native closing-line field - you must poll near tip-off to approximate it. "
            "Players are by name only, so they need mapping to player_id."
        ),
        fields_available=(
            "player_name",
            "team_abbr",
            "opponent_abbr",
            "game_id",
            "game_date",
            "is_home",
            "prop_stat_type",
            "prop_line",
            "over_price",
            "under_price",
            "book_name",
            "odds_snapshot_time",
            "implied_prob",
            "no_vig_prob",
        ),
        adapter_capabilities=frozenset(
            {"fetch_events", "fetch_market_odds", "normalize_to_project_schema"}
        ),
        priority="P1",
        priority_reason="Best free source of multi-book sportsbook player-prop lines to compare vs Kalshi.",
        integration_status="planned",
        role="Current sportsbook player-prop and game lines across many books.",
        notes="No adapter exists yet; requires ODDS_API_KEY. Snapshot near tip to build pseudo-closing lines.",
    ),
    DataSource(
        key="kaggle_csv",
        name="Kaggle / manual CSV imports",
        cost="free",
        data_types=("historical_box_scores", "historical_game_odds", "datasets_vary"),
        sports_covered=("NBA", "varies_by_dataset"),
        supports_player_props=False,
        supports_historical_odds=True,
        supports_closing_prices=True,
        rate_limits="Manual download under Kaggle ToS. Static snapshots; no live freshness.",
        limitations=(
            "Coverage is dataset-specific. Most NBA datasets are box scores or team moneyline/"
            "spread/total odds; historical *player-prop* odds are rare. Schemas are inconsistent."
        ),
        fields_available=(
            "player_name",
            "team_abbr",
            "game_id",
            "game_date",
            "season",
            "actual_minutes",
            "actual_points",
            "actual_rebounds",
            "actual_assists",
            "actual_threes",
            "settlement_value",
            "is_closing",
        ),
        adapter_capabilities=frozenset(
            {"fetch_player_game_logs", "fetch_market_odds", "fetch_closing_prices", "fetch_results", "normalize_to_project_schema"}
        ),
        priority="P1",
        priority_reason="Cheap historical backfill of actuals and team odds; fills gaps nba_api/Kalshi miss.",
        integration_status="wraps_existing",
        role="Historical actuals and team-market odds backfill from static files.",
        notes="Existing path: data/raw/sportsbook/kaggle + scripts/import_kaggle_nba_odds.py. Player-prop odds usually absent.",
    ),
    DataSource(
        key="basketball_reference",
        name="Basketball-Reference (manual/fallback)",
        cost="free",
        data_types=("box_scores", "player_game_logs", "schedule", "advanced_stats"),
        sports_covered=("NBA", "other_sports_sister_sites"),
        supports_player_props=False,
        supports_historical_odds=False,
        supports_closing_prices=False,
        rate_limits=(
            "Strict scraping limits (~20 requests/min before 429/blocks). ToS discourages bulk "
            "automated scraping; prefer manual export or sparse, throttled use."
        ),
        limitations="No betting odds at all. Best as a manual fallback when nba_api is down or incomplete.",
        fields_available=(
            "player_id",
            "player_name",
            "team_abbr",
            "opponent_abbr",
            "position",
            "game_id",
            "game_date",
            "season",
            "is_home",
            "rest_days",
            "actual_minutes",
            "actual_points",
            "actual_rebounds",
            "actual_assists",
            "actual_threes",
            "actual_blocks",
            "actual_steals",
            "actual_turnovers",
            "settlement_value",
        ),
        adapter_capabilities=frozenset(
            {"fetch_players", "fetch_player_game_logs", "fetch_results", "normalize_to_project_schema"}
        ),
        priority="P2",
        priority_reason="Redundant actuals source; valuable only as a fallback for nba_api gaps.",
        integration_status="planned",
        role="Fallback box-score actuals when the primary stats source fails.",
        notes="No adapter yet. Respect ToS/rate limits; manual import preferred.",
    ),
)


def get_source(key: str) -> DataSource:
    """Return the catalog entry for ``key`` or raise ``KeyError``."""

    for source in SOURCE_CATALOG:
        if source.key == key:
            return source
    raise KeyError(f"Unknown data source '{key}'. Known: {[s.key for s in SOURCE_CATALOG]}")


def source_keys() -> tuple[str, ...]:
    return tuple(source.key for source in SOURCE_CATALOG)
