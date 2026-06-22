"""Prior-period uncertainty penalties for research-only edge sweeps."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CONSERVATIVE_PENALTY = 0.03


def penalty_columns_for_mode(mode: str) -> list[str]:
    """Map a penalty mode to the grouping columns used for prior-only penalties."""

    if mode == "none":
        return []
    if mode == "side-only":
        return ["side"]
    if mode == "side+price_bucket":
        return ["side", "price_bucket"]
    if mode == "side+price_bucket+edge_bucket":
        return ["side", "price_bucket", "edge_bucket"]
    if mode == "side+price_bucket+liquidity_bucket":
        return ["side", "price_bucket", "liquidity_bucket"]
    raise ValueError(f"Unknown uncertainty penalty mode: {mode}")


def calculate_prior_penalty(
    prior_rows: pd.DataFrame,
    min_prior_samples: int,
    conservative_default: float = DEFAULT_CONSERVATIVE_PENALTY,
) -> float:
    """Calculate a conservative edge penalty from prior-only calibration and CLV performance."""

    if len(prior_rows) < int(min_prior_samples):
        return float(conservative_default)
    clv = pd.to_numeric(prior_rows.get("clv_cents", pd.Series(dtype=float)), errors="coerce").dropna()
    adjusted_prob = pd.to_numeric(prior_rows.get("adjusted_probability", pd.Series(dtype=float)), errors="coerce")
    won = pd.to_numeric(prior_rows.get("contract_won", pd.Series(dtype=float)), errors="coerce")
    if clv.empty:
        return float(conservative_default)

    average_clv_penalty = max(0.0, -float(clv.mean()) / 100.0)
    positive_clv_penalty = max(0.0, 0.50 - float((clv > 0).mean())) * 0.05
    calibration_error = float((won - adjusted_prob).dropna().mean()) if (won.notna() & adjusted_prob.notna()).any() else 0.0
    calibration_penalty = max(0.0, -calibration_error) * 0.10
    penalty = average_clv_penalty + positive_clv_penalty + calibration_penalty
    return float(min(max(penalty, 0.0), 0.12))


def build_prior_penalty_table(
    prior_rows: pd.DataFrame,
    group_columns: Iterable[str],
    min_prior_samples: int,
    conservative_default: float = DEFAULT_CONSERVATIVE_PENALTY,
) -> pd.DataFrame:
    """Build group-level penalties from rows that precede the evaluation period."""

    columns = list(group_columns)
    if not columns:
        return pd.DataFrame()
    output_rows: list[dict[str, object]] = []
    for key, group in prior_rows.groupby(columns, dropna=False, observed=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(columns, keys)}
        row["prior_rows"] = int(len(group))
        row["uncertainty_penalty"] = calculate_prior_penalty(
            group,
            min_prior_samples=min_prior_samples,
            conservative_default=conservative_default,
        )
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def apply_prior_penalties(
    current_rows: pd.DataFrame,
    prior_rows: pd.DataFrame,
    mode: str,
    min_prior_samples: int,
    conservative_default: float = DEFAULT_CONSERVATIVE_PENALTY,
) -> pd.DataFrame:
    """Attach prior-only uncertainty penalties to current-period candidate rows."""

    output = current_rows.copy()
    if mode == "none":
        output["uncertainty_penalty"] = 0.0
        output["prior_bucket_rows"] = len(prior_rows)
        output["penalty_source"] = "none"
        return output

    columns = penalty_columns_for_mode(mode)
    table = build_prior_penalty_table(
        prior_rows,
        columns,
        min_prior_samples=min_prior_samples,
        conservative_default=conservative_default,
    )
    if table.empty:
        output["uncertainty_penalty"] = float(conservative_default)
        output["prior_bucket_rows"] = 0
        output["penalty_source"] = "conservative_default_no_prior_bucket"
        return output

    output = output.merge(table, on=columns, how="left")
    output["prior_bucket_rows"] = pd.to_numeric(output["prior_rows"], errors="coerce").fillna(0).astype(int)
    output["uncertainty_penalty"] = pd.to_numeric(output["uncertainty_penalty"], errors="coerce").fillna(
        float(conservative_default)
    )
    output["penalty_source"] = np.where(
        output["prior_bucket_rows"] >= int(min_prior_samples),
        "prior_bucket",
        "conservative_default_small_sample",
    )
    output = output.drop(columns=[column for column in ["prior_rows"] if column in output.columns])
    return output
