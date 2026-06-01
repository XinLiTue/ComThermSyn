"""
Density-aware hold-out split for validation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.feature_engineering import vintage_from_year


def select_holdout_buildings(
    model_df: pd.DataFrame,
    rc_fit_df: Optional[pd.DataFrame] = None,  # deprecated – ignored, columns taken from model_df
    vintage_col: str = "vintage",
    random_state: int = 42,
) -> Tuple[pd.Index, pd.Index, pd.DataFrame]:
    """
    Density-aware hold-out split.

    All quality columns (feasible, test_rollout_rmse_T, cv_R, cv_C, R, C, A, Qint)
    are read directly from model_df, which already contains them after
    load_merged_dataset(). The rc_fit_df parameter is kept for backwards
    compatibility but is no longer used.

    Returns
    -------
    train_idx : pd.Index
        Index labels of training buildings.
    test_idx : pd.Index
        Index labels of held-out test buildings.
    holdout_meta_df : pd.DataFrame
        Metadata for held-out buildings (R_fit, C_fit, tau_fit, quality_bin, …).
    """
    rng = np.random.default_rng(random_state)

    model_df = model_df.copy()
    model_df["user_id"] = model_df["user_id"].astype(str)

    if vintage_col not in model_df.columns:
        if "year" in model_df.columns:
            model_df[vintage_col] = model_df["year"].apply(vintage_from_year)
        else:
            raise ValueError(
                f"vintage_col '{vintage_col}' not found and 'year' column is missing"
            )

    # ── Step 1: quality binning (uses model_df columns directly) ────────────
    feasible = (
        model_df["feasible"].fillna(False).astype(bool)
        if "feasible" in model_df.columns
        else pd.Series(False, index=model_df.index)
    )
    rmse = pd.to_numeric(
        model_df.get("test_rollout_rmse_T", pd.Series(np.nan, index=model_df.index)),
        errors="coerce",
    )
    feasible_rmse = rmse[feasible]
    if len(feasible_rmse) >= 2:
        p50 = float(feasible_rmse.quantile(0.50))
        p75 = float(feasible_rmse.quantile(0.75))
    else:
        p50 = p75 = float("inf")

    quality = pd.Series("low", index=model_df.index, dtype=str)
    quality[feasible & (rmse < p50)] = "high"
    quality[feasible & (rmse >= p50) & (rmse < p75)] = "medium"
    model_df["quality_bin"] = quality

    # ── Step 2: per-vintage density-aware count ──────────────────────────────
    test_indices: list = []

    for _vintage_val, grp in model_df.groupby(vintage_col):
        n = len(grp)
        if n <= 3:
            continue
        elif n <= 6:
            n_holdout = 1
        else:
            n_holdout = max(2, math.floor(n * 0.2))

        candidates = grp[grp["quality_bin"].isin(["high", "medium"])].copy()
        if len(candidates) == 0:
            continue

        n_actual = min(n_holdout, len(candidates))

        # ── Step 3: soft representative sampling ────────────────────────────
        area_vals = pd.to_numeric(
            candidates.get("Area", pd.Series(0.0, index=candidates.index)),
            errors="coerce",
        ).fillna(0.0)
        score_vals = pd.to_numeric(
            candidates.get("label_score", pd.Series(0.0, index=candidates.index)),
            errors="coerce",
        ).fillna(0.0)

        area_std = float(area_vals.std()) if float(area_vals.std()) > 1e-9 else 1.0
        score_std = float(score_vals.std()) if float(score_vals.std()) > 1e-9 else 1.0
        area_z = (area_vals - area_vals.mean()) / area_std
        score_z = (score_vals - score_vals.mean()) / score_std

        group_mean = np.array([float(area_z.mean()), float(score_z.mean())])

        def _weighted_sample(pool: pd.DataFrame, n_pick: int) -> pd.DataFrame:
            pa = area_z.loc[pool.index]
            ps = score_z.loc[pool.index]
            d = np.sqrt((pa.values - group_mean[0]) ** 2 + (ps.values - group_mean[1]) ** 2)
            w = np.exp(-d)
            w = w / w.sum()
            idx = rng.choice(len(pool), size=n_pick, replace=False, p=w)
            return pool.iloc[idx]

        high_cands = candidates[candidates["quality_bin"] == "high"]
        if len(high_cands) >= n_actual:
            chosen = _weighted_sample(high_cands, n_actual)
        else:
            chosen = _weighted_sample(candidates, n_actual)

        test_indices.extend(chosen["user_id"].tolist())

    holdout_ids = set(test_indices)
    test_idx  = model_df.index[model_df["user_id"].isin(holdout_ids)]
    train_idx = model_df.index[~model_df["user_id"].isin(holdout_ids)]

    # ── Build holdout_meta_df from model_df columns ──────────────────────────
    if len(holdout_ids) > 0:
        test_rows = model_df.loc[model_df["user_id"].isin(holdout_ids)].copy()
        # Rename RCAQ columns to *_fit for downstream compatibility
        rename_map = {k: f"{k}_fit" for k in ["R", "C", "A", "Qint"] if k in test_rows.columns}
        test_rows = test_rows.rename(columns=rename_map)
        _meta_cols = [
            c for c in
            ["user_id", "R_fit", "C_fit", "A_fit", "Qint_fit",
             "test_rollout_rmse_T", "cv_R", "cv_C", "quality_bin"]
            if c in test_rows.columns
        ]
        meta = test_rows[_meta_cols].copy()
        r_fit = pd.to_numeric(meta.get("R_fit", pd.Series(np.nan, index=meta.index)), errors="coerce")
        c_fit = pd.to_numeric(meta.get("C_fit", pd.Series(np.nan, index=meta.index)), errors="coerce")
        meta["tau_fit"] = r_fit * c_fit
        holdout_meta_df = meta.reset_index(drop=True)
    else:
        holdout_meta_df = pd.DataFrame(columns=[
            "user_id", "R_fit", "C_fit", "A_fit", "Qint_fit",
            "tau_fit", "test_rollout_rmse_T", "cv_R", "cv_C", "quality_bin",
        ])

    assert len(test_idx) > 0, "holdout selection returned 0 test buildings"
    assert len(holdout_meta_df) > 0, "holdout_meta_df is empty"
    assert set(model_df.loc[train_idx, "user_id"]).isdisjoint(
        set(model_df.loc[test_idx, "user_id"])
    ), "train/test leakage"

    return train_idx, test_idx, holdout_meta_df
