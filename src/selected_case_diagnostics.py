"""Post-hoc selected-case interpretation utilities.

These helpers summarize already-generated diagnostic comparisons. They do not
participate in synthesis, scoring, or candidate selection.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


TRUE_ANNUAL_KWH_CANDIDATES = (
    "true_annual_kwh",
    "true_total_annual_kwh",
    "annual_kwh_true",
)
GENERATED_ANNUAL_KWH_CANDIDATES = (
    "generated_annual_kwh",
    "generated_total_annual_kwh",
    "annual_kwh_generated",
    "annual_kwh_gen",
)
TRUE_ANNUAL_PER_M2_CANDIDATES = (
    "true_E_std_annual_per_m2",
    "E_std_annual_per_m2_true",
    "true_annual_kwh_per_m2",
    "annual_kwh_per_m2_true",
    "E_std_annual_per_m2",
)
GENERATED_ANNUAL_PER_M2_CANDIDATES = (
    "generated_E_std_full_per_m2",
    "gen_E_std_annual_per_m2",
    "E_std_annual_per_m2_gen",
    "generated_annual_kwh_per_m2",
    "annual_kwh_per_m2_gen",
    "annual_kwh_per_m2_mean",
    "generated_E_std_annual_per_m2",
    "E_std_annual_per_m2_generated",
    "E_std_full_per_m2_generated",
    "E_std_full_per_m2",
    "E_proxy_annual_per_m2",
)
AREA_CANDIDATES = ("Area", "Area_true", "Area_gen", "true_Area", "generated_Area")


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def summarize_aggregate_annual_energy(
    comparison_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("case_id", "source_name", "selected_rank"),
) -> pd.DataFrame:
    """Summarize reference-versus-synthesized annual energy by selected groups.

    The input is a post-hoc comparison table. Annual totals are taken from
    direct kWh columns when available, otherwise they are computed from
    kWh/m2 and area. Missing requirements yield an empty table with a warning.
    """
    if not isinstance(comparison_df, pd.DataFrame) or comparison_df.empty:
        warnings.warn("Aggregate annual energy summary skipped: comparison data are unavailable.")
        return pd.DataFrame()

    work = comparison_df.copy()
    active_group_cols = [column for column in group_cols if column in work.columns]
    if not active_group_cols:
        warnings.warn("Aggregate annual energy summary skipped: requested group columns are unavailable.")
        return pd.DataFrame()

    true_kwh_col = _first_existing_column(work, TRUE_ANNUAL_KWH_CANDIDATES)
    generated_kwh_col = _first_existing_column(work, GENERATED_ANNUAL_KWH_CANDIDATES)
    true_per_m2_col = _first_existing_column(work, TRUE_ANNUAL_PER_M2_CANDIDATES)
    generated_per_m2_col = _first_existing_column(work, GENERATED_ANNUAL_PER_M2_CANDIDATES)
    area_col = _first_existing_column(work, AREA_CANDIDATES)

    if generated_per_m2_col == "E_proxy_annual_per_m2":
        warnings.warn(
            "Aggregate annual energy summary uses proxy synthesized energy "
            "because no full synthesized annual-energy column is available."
        )

    if true_kwh_col is None and (true_per_m2_col is None or area_col is None):
        warnings.warn("Aggregate annual energy summary skipped: reference energy or area is unavailable.")
        return pd.DataFrame()
    if generated_kwh_col is None and (generated_per_m2_col is None or area_col is None):
        warnings.warn("Aggregate annual energy summary skipped: synthesized energy or area is unavailable.")
        return pd.DataFrame()

    area = pd.to_numeric(work[area_col], errors="coerce") if area_col else pd.Series(np.nan, index=work.index)
    true_per_m2 = (
        pd.to_numeric(work[true_per_m2_col], errors="coerce")
        if true_per_m2_col
        else pd.Series(np.nan, index=work.index)
    )
    generated_per_m2 = (
        pd.to_numeric(work[generated_per_m2_col], errors="coerce")
        if generated_per_m2_col
        else pd.Series(np.nan, index=work.index)
    )
    work["_true_annual_kwh"] = (
        pd.to_numeric(work[true_kwh_col], errors="coerce") if true_kwh_col else true_per_m2 * area
    )
    work["_generated_annual_kwh"] = (
        pd.to_numeric(work[generated_kwh_col], errors="coerce")
        if generated_kwh_col
        else generated_per_m2 * area
    )
    work["_true_annual_kwh_per_m2"] = true_per_m2
    work["_generated_annual_kwh_per_m2"] = generated_per_m2
    work["_signed_error_kwh_per_m2"] = generated_per_m2 - true_per_m2
    work["_abs_error_kwh_per_m2"] = work["_signed_error_kwh_per_m2"].abs()

    rows: list[dict[str, object]] = []
    for keys, community in work.groupby(active_group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(active_group_cols, key_values))
        true_total = community["_true_annual_kwh"].sum(min_count=1)
        generated_total = community["_generated_annual_kwh"].sum(min_count=1)
        signed_error = generated_total - true_total
        signed_pct = (
            100.0 * signed_error / true_total
            if pd.notna(true_total) and abs(float(true_total)) > 1e-9
            else np.nan
        )
        row.update(
            {
                "n_buildings": int(len(community)),
                "true_total_annual_kwh": float(true_total) if pd.notna(true_total) else np.nan,
                "generated_total_annual_kwh": float(generated_total) if pd.notna(generated_total) else np.nan,
                "signed_error_kwh": float(signed_error) if pd.notna(signed_error) else np.nan,
                "signed_error_pct": float(signed_pct) if pd.notna(signed_pct) else np.nan,
                "absolute_error_kwh": abs(float(signed_error)) if pd.notna(signed_error) else np.nan,
                "absolute_error_pct": abs(float(signed_pct)) if pd.notna(signed_pct) else np.nan,
                "true_mean_annual_kwh_per_m2": float(community["_true_annual_kwh_per_m2"].mean()),
                "generated_mean_annual_kwh_per_m2": float(community["_generated_annual_kwh_per_m2"].mean()),
                "mean_signed_error_kwh_per_m2": float(community["_signed_error_kwh_per_m2"].mean()),
                "mean_abs_error_kwh_per_m2": float(community["_abs_error_kwh_per_m2"].mean()),
                "median_abs_error_kwh_per_m2": float(community["_abs_error_kwh_per_m2"].median()),
                "generated_energy_column": generated_kwh_col or generated_per_m2_col,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
