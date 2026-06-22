"""Team-name normalization for the no-odds matchup pipeline.

Fixtures, historical results, and injury feeds frequently spell the same team
differently ("USA" vs "United States", "Korea Republic" vs "South Korea").
This module provides a light, sport-agnostic normalization layer plus a CSV
alias table the user can extend by hand.

Design goals:
* Never crash on messy input – always return *some* string.
* Be conservative: only collapse names we are confident are the same team.
* Make it trivial to extend via ``data/manual/team_aliases.csv``.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path


# A small built-in alias table for common national-team spelling differences.
# Keys are match keys (see ``team_match_key``); values are canonical display
# names. The user-supplied CSV alias table takes precedence over this.
_BUILTIN_CANONICAL: dict[str, str] = {
    "usa": "United States",
    "unitedstates": "United States",
    "unitedstatesofamerica": "United States",
    "us": "United States",
    "southkorea": "South Korea",
    "koreareplublic": "South Korea",
    "korearepublic": "South Korea",
    "korea": "South Korea",
    "northkorea": "North Korea",
    "koreadpr": "North Korea",
    "dprkorea": "North Korea",
    "ivorycoast": "Ivory Coast",
    "cotedivoire": "Ivory Coast",
    "côtedivoire": "Ivory Coast",
    "czechrepublic": "Czechia",
    "czechia": "Czechia",
    "iran": "Iran",
    "iranislamicrepublic": "Iran",
    "uae": "United Arab Emirates",
    "unitedarabemirates": "United Arab Emirates",
    "drcongo": "DR Congo",
    "democraticrepublicofthecongo": "DR Congo",
    "congodr": "DR Congo",
    "capeverde": "Cape Verde",
    "caboverde": "Cape Verde",
    "bosnia": "Bosnia and Herzegovina",
    "bosniaandherzegovina": "Bosnia and Herzegovina",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "turkey": "Turkey",
}

# Common decorative suffixes that do not change team identity. Stripped only
# when they leave a non-empty remainder.
_STRIP_SUFFIXES: tuple[str, ...] = (
    "national team",
    "national football team",
    "men's national team",
    "mens national team",
    "nt",
)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def team_match_key(name: object) -> str:
    """Return a lowercase alphanumeric key used for fuzzy-but-safe matching.

    Accents are stripped and all non-alphanumeric characters removed so that
    "Côte d'Ivoire", "Cote d Ivoire" and "cotedivoire" collapse to one key.
    """

    text = _strip_accents(str(name)).lower().strip()
    for suffix in _STRIP_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            text = text[: -len(suffix)].strip()
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_team_name(name: object) -> str:
    """Return a cleaned, canonical display name for a team.

    Falls back to a tidied version of the original string (collapsed
    whitespace, trimmed) when the team is unknown, so the value stays readable.
    """

    raw = str(name).strip()
    if not raw:
        return ""

    key = team_match_key(raw)
    if key in _BUILTIN_CANONICAL:
        return _BUILTIN_CANONICAL[key]

    # Default: collapse internal whitespace but preserve the original casing,
    # which is usually already human-friendly (e.g. "Japan", "Real Madrid").
    return re.sub(r"\s+", " ", raw)


def load_team_aliases(path: str | Path) -> dict[str, str]:
    """Load a CSV alias table into a ``match_key -> canonical_team`` dict.

    The CSV is expected to have at least ``canonical_team`` and ``alias``
    columns (extra columns such as ``sport``/``league``/``country`` are kept in
    the file for the user's reference but ignored here). Both the alias and the
    canonical name are registered as keys so either spelling resolves.

    Returns an empty dict when the file does not exist, so callers can treat
    aliases as fully optional.
    """

    alias_path = Path(path)
    mapping: dict[str, str] = {}
    if not alias_path.exists():
        return mapping

    with alias_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            canonical = str(row.get("canonical_team", "")).strip()
            alias = str(row.get("alias", "")).strip()
            if not canonical:
                continue
            mapping[team_match_key(canonical)] = canonical
            if alias:
                mapping[team_match_key(alias)] = canonical
    return mapping


def resolve_team_name(name: object, aliases: dict[str, str] | None = None) -> str:
    """Resolve a single team name using the alias table, then built-in rules."""

    if aliases:
        key = team_match_key(name)
        if key in aliases:
            return aliases[key]
    return normalize_team_name(name)


def apply_team_aliases(
    df,
    columns: list[str],
    aliases: dict[str, str] | None = None,
):
    """Return a copy of ``df`` with the given team columns normalized.

    Each value is resolved through the user alias table first, then the
    built-in normalization rules. Missing columns are skipped silently so the
    same call works across results, fixtures, and injury frames.
    """

    output = df.copy()
    for column in columns:
        if column not in output.columns:
            continue
        output[column] = output[column].map(lambda value: resolve_team_name(value, aliases))
    return output
