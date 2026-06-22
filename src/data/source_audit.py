"""Build the NBA player-prop data-source audit from the source catalog.

Produces three machine-readable tables plus a human-readable markdown report:

    build_source_audit_table()    -> one row per source (the capability matrix)
    build_field_coverage_table()  -> one row per needed field x source coverage
    build_audit_summary()         -> headline dict for JSON / dashboards
    render_markdown_report()      -> the written audit the user reads
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .source_catalog import (
    NBA_PLAYER_PROP_FIELDS,
    SOURCE_CATALOG,
    DataSource,
)


def _join(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else ""


def build_source_audit_table() -> pd.DataFrame:
    """Return the per-source capability matrix with every audited attribute."""

    rows: list[dict[str, Any]] = []
    for source in SOURCE_CATALOG:
        rows.append(
            {
                "priority": source.priority,
                "source_key": source.key,
                "source_name": source.name,
                "cost": source.cost,
                "data_types": _join(source.data_types),
                "sports_covered": _join(source.sports_covered),
                "supports_player_props": source.supports_player_props,
                "supports_historical_odds": source.supports_historical_odds,
                "supports_closing_prices": source.supports_closing_prices,
                "rate_limits": source.rate_limits,
                "limitations": source.limitations,
                "fields_needed": len(NBA_PLAYER_PROP_FIELDS),
                "fields_available": len(source.fields_available),
                "missing_required_fields": _join(source.missing_required_fields()),
                "missing_important_fields": _join(source.missing_important_fields()),
                "adapter_methods": _join(tuple(sorted(source.adapter_capabilities))),
                "integration_status": source.integration_status,
                "role": source.role,
                "priority_reason": source.priority_reason,
                "notes": source.notes,
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["priority", "source_key"]).reset_index(drop=True)


def build_field_coverage_table() -> pd.DataFrame:
    """Return one row per needed field with per-source availability and counts."""

    rows: list[dict[str, Any]] = []
    for requirement in NBA_PLAYER_PROP_FIELDS:
        row: dict[str, Any] = {
            "field": requirement.name,
            "importance": requirement.importance,
            "group": requirement.group,
            "description": requirement.description,
        }
        covering = 0
        for source in SOURCE_CATALOG:
            available = requirement.name in source.fields_available
            row[source.key] = available
            covering += int(available)
        row["sources_covering"] = covering
        rows.append(row)
    return pd.DataFrame(rows)


def build_audit_summary() -> dict[str, Any]:
    """Return headline counts and the recommended source roles."""

    table = build_source_audit_table()
    coverage = build_field_coverage_table()
    required_fields = coverage[coverage["importance"].eq("required")]
    uncovered_required = required_fields[required_fields["sources_covering"].eq(0)]["field"].tolist()

    return {
        "report": "nba_player_prop_data_source_audit",
        "scope": "free and free-tier sources only",
        "sources_reviewed": int(len(table)),
        "fields_needed": int(len(NBA_PLAYER_PROP_FIELDS)),
        "player_prop_sources": table[table["supports_player_props"]]["source_key"].tolist(),
        "historical_odds_sources": table[table["supports_historical_odds"]]["source_key"].tolist(),
        "closing_price_sources": table[table["supports_closing_prices"]]["source_key"].tolist(),
        "priority_order": table.sort_values("priority")[["source_key", "priority", "role"]].to_dict("records"),
        "required_fields_with_no_source": uncovered_required,
        "single_source_player_prop_odds": _single_prop_odds_gap(),
        "recommended_stack": {
            "actuals_and_features": ["nba_api", "basketball_reference", "kaggle_csv"],
            "prediction_market_prices": ["kalshi"],
            "sportsbook_prop_lines_and_clv": ["odds_api"],
            "historical_team_odds_backfill": ["kaggle_csv"],
        },
        "research_only": True,
        "approved": False,
    }


def _single_prop_odds_gap() -> dict[str, Any]:
    """Flag that live sportsbook prop odds depend on a single quota-limited source."""

    prop_odds_sources = [
        source.key
        for source in SOURCE_CATALOG
        if source.supports_player_props and "over_price" in source.fields_available
    ]
    return {
        "sources_with_player_prop_prices": prop_odds_sources,
        "warning": (
            "Live multi-book sportsbook player-prop prices come only from the quota-limited Odds API "
            "free tier; Kalshi supplies prop prices as Yes/No contracts, not Over/Under book lines."
        ),
    }


def _bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = [str(row.get(column, "")).replace("\n", " ").replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown_report() -> str:
    """Render the full markdown audit the user reads."""

    summary = build_audit_summary()
    coverage = build_field_coverage_table()
    lines: list[str] = []
    lines.append("# NBA Player-Prop Data Source Audit")
    lines.append("")
    lines.append("_Research-only. Free and free-tier sources only. Verify vendor quotas before heavy use._")
    lines.append("")
    lines.append(
        f"Reviewed **{summary['sources_reviewed']}** sources against **{summary['fields_needed']}** "
        "fields the player-prop model needs."
    )
    lines.append("")

    # Per-source detail.
    lines.append("## Sources")
    for source in sorted(SOURCE_CATALOG, key=lambda item: (item.priority, item.key)):
        lines.append("")
        lines.append(f"### {source.priority} - {source.name} (`{source.key}`)")
        lines.append("")
        lines.append(f"- **Cost:** {source.cost}")
        lines.append(f"- **Role:** {source.role}")
        lines.append(f"- **Data types:** {_join(source.data_types)}")
        lines.append(f"- **Sports covered:** {_join(source.sports_covered)}")
        lines.append(f"- **Supports player props:** {_bool(source.supports_player_props)}")
        lines.append(f"- **Supports historical odds:** {_bool(source.supports_historical_odds)}")
        lines.append(f"- **Supports closing prices:** {_bool(source.supports_closing_prices)}")
        lines.append(f"- **Rate limits / limitations:** {source.rate_limits} {source.limitations}")
        lines.append(f"- **Fields we can get:** {_join(source.fields_available)}")
        missing_required = source.missing_required_fields()
        lines.append(
            f"- **Missing required fields:** {_join(missing_required) if missing_required else 'none'}"
        )
        lines.append(f"- **Missing important fields:** {_join(source.missing_important_fields()) or 'none'}")
        lines.append(f"- **Adapter methods supported:** {_join(tuple(sorted(source.adapter_capabilities)))}")
        lines.append(f"- **Integration status:** {source.integration_status}")
        lines.append(f"- **Recommended priority:** {source.priority} - {source.priority_reason}")
        if source.notes:
            lines.append(f"- **Notes:** {source.notes}")

    # Capability matrix.
    lines.append("")
    lines.append("## Capability Matrix")
    lines.append("")
    matrix = build_source_audit_table()
    matrix_display = matrix.copy()
    for column in ["supports_player_props", "supports_historical_odds", "supports_closing_prices"]:
        matrix_display[column] = matrix_display[column].map(_bool)
    lines.append(
        _markdown_table(
            matrix_display,
            [
                "priority",
                "source_key",
                "cost",
                "supports_player_props",
                "supports_historical_odds",
                "supports_closing_prices",
                "fields_available",
                "integration_status",
            ],
        )
    )

    # Field coverage.
    lines.append("")
    lines.append("## Field Coverage (what the model needs vs. what we can get)")
    lines.append("")
    coverage_display = coverage.copy()
    for source in SOURCE_CATALOG:
        coverage_display[source.key] = coverage_display[source.key].map(_bool)
    lines.append(
        _markdown_table(
            coverage_display,
            ["field", "importance", "group", *[source.key for source in SOURCE_CATALOG], "sources_covering"],
        )
    )

    # Recommendations.
    lines.append("")
    lines.append("## Recommended Stack")
    lines.append("")
    for role, keys in summary["recommended_stack"].items():
        lines.append(f"- **{role.replace('_', ' ')}:** {', '.join(keys)}")
    lines.append("")
    lines.append("### Key gaps")
    gap = summary["single_source_player_prop_odds"]
    lines.append(f"- {gap['warning']}")
    if summary["required_fields_with_no_source"]:
        lines.append(
            f"- Required fields with no free source: {', '.join(summary['required_fields_with_no_source'])}"
        )
    else:
        lines.append("- Every required field is covered by at least one free source (across sources, not within one).")
    lines.append("")
    lines.append("_This audit does not change proof gates or approve any betting. Player props remain research-only._")
    lines.append("")
    return "\n".join(lines)
