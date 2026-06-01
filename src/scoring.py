"""
Candidate and population scoring functions (V3, V4, V5).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    LABEL_BOUNDS,
    LABEL_OFFICIAL_WEIGHT,
    LABEL_ORDER,
    LABEL_SCORE,
    VINTAGE_EXPECTED_LABEL_SCORE,
)
from src.feature_engineering import clean_energy_label


# ---------------------------------------------------------------------------
# Basic math helpers
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def summarize_series(values: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {k: np.nan for k in ["count", "mean", "std", "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]}
    return {
        "count": float(len(arr)),
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr, ddof=0)),
        "q01":   float(np.quantile(arr, 0.01)),
        "q05":   float(np.quantile(arr, 0.05)),
        "q10":   float(np.quantile(arr, 0.10)),
        "q25":   float(np.quantile(arr, 0.25)),
        "q50":   float(np.quantile(arr, 0.50)),
        "q75":   float(np.quantile(arr, 0.75)),
        "q90":   float(np.quantile(arr, 0.90)),
        "q95":   float(np.quantile(arr, 0.95)),
        "q99":   float(np.quantile(arr, 0.99)),
    }


def to_stats_dict(
    df_in: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
) -> Dict[Tuple, Dict[str, float]]:
    if len(df_in) == 0:
        return {}
    out: Dict[Tuple, Dict[str, float]] = {}
    for key, values in df_in.groupby(group_cols, dropna=False)[value_col]:
        key_tuple = key if isinstance(key, tuple) else (key,)
        out[key_tuple] = summarize_series(values)
    return out


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        return np.nan
    quantiles = np.linspace(0.0, 1.0, max(len(x), len(y)))
    xq = np.quantile(x, quantiles)
    yq = np.quantile(y, quantiles)
    return float(np.mean(np.abs(xq - yq)))


# ---------------------------------------------------------------------------
# V3/V4 single-building scoring penalties
# ---------------------------------------------------------------------------

def official_label_penalty(energy_per_m2: float, label: Any) -> float:
    clean = clean_energy_label(label)
    if clean is None:
        return 0.0
    lo, hi = LABEL_BOUNDS.get(clean, (0.0, 1e9))
    if lo <= energy_per_m2 <= hi:
        return 0.0
    margin = max(hi - lo, 10.0)
    if energy_per_m2 < lo:
        return float((lo - energy_per_m2) / margin)
    return float((energy_per_m2 - hi) / margin)


def empirical_label_penalty(energy_per_m2: float, label_stats: Dict[str, float]) -> float:
    q10 = label_stats.get("q10", np.nan)
    q90 = label_stats.get("q90", np.nan)
    mean = label_stats.get("mean", np.nan)
    if not (np.isfinite(q10) and np.isfinite(q90) and np.isfinite(mean)):
        return 0.0
    span = max(q90 - q10, 10.0)
    if q10 <= energy_per_m2 <= q90:
        return 0.0
    if energy_per_m2 < q10:
        return float((q10 - energy_per_m2) / span)
    return float((energy_per_m2 - q90) / span)


def response_penalty(
    candidate: Dict[str, float],
    year: float,
    screen_summary: Dict[str, float],
    model: Any,
) -> float:
    summer_share = float(screen_summary.get("summer_timestep_share", 0.0))
    return max(summer_share - 0.12, 0.0) / 0.12


def mahalanobis_penalty(residual_vec: np.ndarray, model: Any) -> float:
    """Penalise outlier residuals using model's inverse-covariance matrix."""
    try:
        cov_inv = model.cov_inv
        diff = np.asarray(residual_vec, dtype=float)
        dist = float(diff.T @ cov_inv @ diff)
        return min(dist / 10.0, 3.0)
    except Exception:
        return 0.0


def lookup_energy_stats(model: Any, year: float, label: Any) -> Dict[str, float]:
    """
    Look up empirical energy statistics from a ParamGeneratorModel.
    Priority: (vintage, label) → label → vintage → global.
    """
    from src.feature_engineering import vintage_from_year

    clean = clean_energy_label(label)
    vintage = vintage_from_year(year)

    try:
        vl_stats = model.energy_stats_vl
        if vl_stats is not None and clean is not None:
            row = vl_stats.loc[(vl_stats["vintage"] == vintage) & (vl_stats["EnergyLabel"] == clean)]
            if len(row) >= 1 and int(row.iloc[0]["n"]) >= 3:
                return {**row.iloc[0].to_dict(), "source": "vintage_label"}

        l_stats = model.energy_stats_l
        if l_stats is not None and clean is not None:
            row = l_stats.loc[l_stats["EnergyLabel"] == clean]
            if len(row) >= 1 and int(row.iloc[0]["n"]) >= 3:
                return {**row.iloc[0].to_dict(), "source": "label"}

        v_stats = model.energy_stats_v
        if v_stats is not None:
            row = v_stats.loc[v_stats["vintage"] == vintage]
            if len(row) >= 1 and int(row.iloc[0]["n"]) >= 3:
                return {**row.iloc[0].to_dict(), "source": "vintage"}
    except Exception:
        pass

    g = getattr(model, "energy_stats_g", None)
    if g is not None:
        return {**g, "source": "global"}
    return {"mean": 110.0, "q10": 80.0, "q90": 160.0, "n": 0, "source": "fallback"}


# ---------------------------------------------------------------------------
# V4 score_candidate (with per-label official weight)
# ---------------------------------------------------------------------------

def score_candidate(
    candidate: Dict[str, float],
    residual_vec: np.ndarray,
    year: float,
    area: float,
    label: Optional[str],
    model: Any,
    screen_summary: Dict[str, float],
    dynamic_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """V4 single-building score with per-label official-penalty weights."""
    energy_value = (
        float(dynamic_summary["annual_kwh_per_m2_mean"])
        if dynamic_summary is not None
        else float(screen_summary["annual_kwh_per_m2"])
    )
    label_stats = lookup_energy_stats(model=model, year=year, label=label)
    label_clean = clean_energy_label(label)
    off_w = LABEL_OFFICIAL_WEIGHT.get(label_clean, 1.0) if label_clean is not None else 1.0

    p_official = official_label_penalty(energy_value, label) if off_w > 0.0 else 0.0
    p_empirical = empirical_label_penalty(energy_value, label_stats)
    p_response = response_penalty(candidate=candidate, year=year, screen_summary=screen_summary, model=model)
    p_maha = mahalanobis_penalty(residual_vec, model)
    p_summer = max(float(screen_summary.get("summer_timestep_share", 0.0)) - 0.12, 0.0) / 0.12

    total = off_w * p_official + 0.8 * p_empirical + 0.7 * p_response + 0.35 * p_maha + 0.7 * p_summer
    return {
        "score":                float(total),
        "score_official":       float(p_official),
        "score_empirical":      float(p_empirical),
        "score_response":       float(p_response),
        "score_mahalanobis":    float(p_maha),
        "score_summer":         float(p_summer),
        "energy_stats_source":  label_stats.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# V4 community scoring
# ---------------------------------------------------------------------------

def label_ordering_score(
    population_df: pd.DataFrame,
    energy_col: str,
) -> Tuple[float, bool]:
    if "label_clean" not in population_df.columns or energy_col not in population_df.columns:
        return 0.0, True
    grouped = (
        population_df.dropna(subset=["label_clean"])
        .groupby("label_clean", dropna=False)[energy_col]
        .mean()
        .to_dict()
    )
    labels_present = [l for l in LABEL_ORDER if l in grouped]
    total_pairs, violations = 0, 0
    for i, left in enumerate(labels_present):
        for right in labels_present[i + 1:]:
            total_pairs += 1
            if grouped[left] > grouped[right] + 1e-9:
                violations += 1
    if total_pairs == 0:
        return 0.0, True
    return float(violations / total_pairs), bool(violations == 0)


def community_score_components(
    population_df: pd.DataFrame,
    community_conditions: pd.DataFrame,
    gen: Any,
    energy_col: str,
) -> Dict[str, float]:
    from src.thermal_synthesis import lookup_energy_stat_record
    from src.feature_engineering import prepare_building_features

    cond_df = prepare_building_features(community_conditions).reset_index(drop=True)
    pop_df = population_df.reset_index(drop=True).copy()

    expected_e = float(np.mean([
        float(lookup_energy_stat_record(gen=gen, vintage=str(r.vintage), label=r.label_clean)["mean"])
        for r in cond_df.itertuples(index=False)
    ]))
    energies = pop_df[energy_col].to_numpy(dtype=float)
    mean_e = float(np.mean(energies))
    std_scale = max(float(gen.train_energy_stats["std"]), 1.0)
    mean_error = abs(mean_e - expected_e) / std_scale

    coverage_hits: List[float] = []
    outlier_terms: List[float] = []
    for cond_row, e_val in zip(cond_df.itertuples(index=False), energies):
        stats = lookup_energy_stat_record(gen=gen, vintage=str(cond_row.vintage), label=cond_row.label_clean)
        q10, q90 = float(stats["q10"]), float(stats["q90"])
        q01, q99 = float(stats["q01"]), float(stats["q99"])
        span = max(q90 - q10, 10.0)
        coverage_hits.append(float(q10 <= e_val <= q90))
        if e_val < q01:
            outlier_terms.append((q01 - e_val) / span)
        elif e_val > q99:
            outlier_terms.append((e_val - q99) / span)
        else:
            outlier_terms.append(0.0)

    coverage_ratio = float(np.mean(coverage_hits)) if coverage_hits else 1.0
    coverage_penalty = 1.0 - coverage_ratio
    outlier_penalty = float(np.mean(outlier_terms)) if outlier_terms else 0.0

    label_penalty, label_ok = label_ordering_score(pop_df, energy_col=energy_col)

    reno_mask = pop_df["reno_signal"] > 1.0
    base_mask = pop_df["reno_signal"] <= 0.0
    if reno_mask.any() and base_mask.any():
        renovation_ok = float(pop_df.loc[reno_mask, "R"].mean()) > float(pop_df.loc[base_mask, "R"].mean())
    else:
        renovation_ok = True
    renovation_penalty = 0.0 if renovation_ok else 1.0

    total_score = (
        1.6 * mean_error
        + 1.2 * label_penalty
        + 0.8 * coverage_penalty
        + 1.0 * outlier_penalty
        + 0.6 * renovation_penalty
    )
    return {
        "score":                float(total_score),
        "expected_E":           expected_e,
        "mean_E":               mean_e,
        "mean_E_error_std":     float(mean_error),
        "coverage_ratio":       coverage_ratio,
        "coverage_penalty":     coverage_penalty,
        "outlier_penalty":      outlier_penalty,
        "label_ordering_score": label_penalty,
        "label_ordering_ok":    bool(label_ok),
        "renovation_R_ok":      bool(renovation_ok),
    }


def score_population_proxy(
    population_df: pd.DataFrame,
    community_conditions: pd.DataFrame,
    gen: Any,
) -> Dict[str, float]:
    return community_score_components(
        population_df=population_df,
        community_conditions=community_conditions,
        gen=gen,
        energy_col="E_proxy_annual_per_m2",
    )


def score_population_full(
    population_df: pd.DataFrame,
    community_conditions: pd.DataFrame,
    gen: Any,
) -> Dict[str, float]:
    return community_score_components(
        population_df=population_df,
        community_conditions=community_conditions,
        gen=gen,
        energy_col="E_std_full_per_m2",
    )


# ---------------------------------------------------------------------------
# V5 scoring
# ---------------------------------------------------------------------------

def expected_E_for_building_v5(
    gen: Any,
    vintage: str,
    label: Optional[str],
    label_score: float,
    vintage_expected_score: float,
    beta_label_E: float = -3.0,
    n_min: int = 3,
) -> Tuple[float, str]:
    label_clean = clean_energy_label(label)
    label_adj = beta_label_E * (label_score - vintage_expected_score)

    if label_clean is not None:
        stats = gen.energy_stats_by_vintage_label.get((vintage, label_clean))
        if stats is not None and stats["count"] >= n_min:
            return float(stats["mean"]), "vintage_label"

    stats_v = gen.energy_stats_by_vintage.get((vintage,))
    if stats_v is not None:
        return float(stats_v["mean"]) + label_adj, "vintage_beta"

    return float(gen.train_energy_stats["mean"]) + label_adj, "global_beta"


def label_ordering_metrics_v5(
    population_df: pd.DataFrame,
    energy_col: str,
) -> Dict[str, object]:
    if "label_clean" not in population_df.columns or energy_col not in population_df.columns:
        return {
            "label_ordering_score": 0.0,
            "label_ordering_ok": True,
            "label_violation_count": 0,
            "label_spearman_corr": np.nan,
            "n_unique_labels": 0,
        }

    grouped = (
        population_df.dropna(subset=["label_clean"])
        .groupby("label_clean", dropna=False)[energy_col]
        .mean()
        .to_dict()
    )
    labels_present = [lbl for lbl in LABEL_ORDER if lbl in grouped]
    n_unique = len(labels_present)

    total_pairs, violation_count = 0, 0
    for i, left in enumerate(labels_present):
        for right in labels_present[i + 1:]:
            total_pairs += 1
            if grouped[left] > grouped[right] + 1e-9:
                violation_count += 1

    ordering_score = violation_count / max(total_pairs, 1)

    spearman_corr = np.nan
    if n_unique >= 3:
        try:
            from scipy.stats import spearmanr
            scores_list = [LABEL_SCORE[lbl] for lbl in labels_present]
            energies_list = [grouped[lbl] for lbl in labels_present]
            if len(set(energies_list)) > 1:
                corr, _ = spearmanr(scores_list, energies_list)
                spearman_corr = float(corr)
        except Exception:
            pass

    return {
        "label_ordering_score":  float(ordering_score),
        "label_ordering_ok":     bool(violation_count == 0),
        "label_violation_count": int(violation_count),
        "label_spearman_corr":   spearman_corr,
        "n_unique_labels":       n_unique,
    }


def community_score_components_v5(
    population_df: pd.DataFrame,
    community_conditions: pd.DataFrame,
    gen: Any,
    energy_col: str,
    beta_label_E: float = -3.0,
) -> Dict[str, object]:
    from src.feature_engineering import prepare_building_features
    from src.thermal_synthesis import lookup_energy_stat_record

    cond_df = prepare_building_features(community_conditions).reset_index(drop=True)
    pop_df = population_df.reset_index(drop=True).copy()
    if len(cond_df) != len(pop_df):
        raise ValueError(
            f"community_conditions ({len(cond_df)} rows) and population_df "
            f"({len(pop_df)} rows) must have the same length."
        )

    exp_energies = []
    for cond_row in cond_df.itertuples(index=False):
        exp_e, _ = expected_E_for_building_v5(
            gen=gen,
            vintage=str(cond_row.vintage),
            label=cond_row.label_clean,
            label_score=float(cond_row.label_score),
            vintage_expected_score=float(cond_row.vintage_expected_label_score),
            beta_label_E=beta_label_E,
        )
        exp_energies.append(exp_e)

    expected_e = float(np.mean(exp_energies))
    energies = pop_df[energy_col].to_numpy(dtype=float)
    mean_e = float(np.mean(energies))
    std_scale = max(float(gen.train_energy_stats["std"]), 1.0)
    mean_error = abs(mean_e - expected_e) / std_scale

    coverage_hits: List[float] = []
    outlier_terms: List[float] = []
    for cond_row, e_val in zip(cond_df.itertuples(index=False), energies):
        stats = lookup_energy_stat_record(gen=gen, vintage=str(cond_row.vintage), label=cond_row.label_clean)
        q10, q90 = float(stats["q10"]), float(stats["q90"])
        q01, q99 = float(stats["q01"]), float(stats["q99"])
        span = max(q90 - q10, 10.0)
        coverage_hits.append(float(q10 <= e_val <= q90))
        if e_val < q01:
            outlier_terms.append((q01 - e_val) / span)
        elif e_val > q99:
            outlier_terms.append((e_val - q99) / span)
        else:
            outlier_terms.append(0.0)

    coverage_ratio = float(np.mean(coverage_hits)) if coverage_hits else 1.0
    coverage_penalty = 1.0 - coverage_ratio
    outlier_penalty = float(np.mean(outlier_terms)) if outlier_terms else 0.0

    lm = label_ordering_metrics_v5(pop_df, energy_col)
    label_penalty = lm["label_ordering_score"]

    reno_mask = pop_df["reno_signal"] > 1.0
    base_mask = pop_df["reno_signal"] <= 0.0
    if reno_mask.any() and base_mask.any():
        renovation_ok = float(pop_df.loc[reno_mask, "R"].mean()) > float(pop_df.loc[base_mask, "R"].mean())
    else:
        renovation_ok = True
    renovation_penalty = 0.0 if renovation_ok else 1.0

    total_score = (
        1.6 * mean_error
        + 1.2 * label_penalty
        + 0.8 * coverage_penalty
        + 1.0 * outlier_penalty
        + 0.6 * renovation_penalty
    )
    return {
        "score":                 float(total_score),
        "expected_E":            expected_e,
        "mean_E":                mean_e,
        "mean_E_error_std":      float(mean_error),
        "coverage_ratio":        coverage_ratio,
        "coverage_penalty":      coverage_penalty,
        "outlier_penalty":       outlier_penalty,
        "label_ordering_score":  lm["label_ordering_score"],
        "label_ordering_ok":     lm["label_ordering_ok"],
        "label_violation_count": lm["label_violation_count"],
        "label_spearman_corr":   lm["label_spearman_corr"],
        "n_unique_labels":       lm["n_unique_labels"],
        "renovation_R_ok":       bool(renovation_ok),
    }


def score_population_proxy_v5(
    population_df: pd.DataFrame,
    community_conditions: pd.DataFrame,
    gen: Any,
    beta_label_E: float = -3.0,
) -> Dict[str, object]:
    return community_score_components_v5(
        population_df, community_conditions, gen,
        energy_col="E_proxy_annual_per_m2", beta_label_E=beta_label_E,
    )


def score_population_full_v5(
    population_df: pd.DataFrame,
    community_conditions: pd.DataFrame,
    gen: Any,
    beta_label_E: float = -3.0,
) -> Dict[str, object]:
    return community_score_components_v5(
        population_df, community_conditions, gen,
        energy_col="E_std_full_per_m2", beta_label_E=beta_label_E,
    )


# Strategy-explicit API aliases. Formulas and weights remain defined above.
score_population_proxy_bootstrap = score_population_proxy
score_population_full_bootstrap = score_population_full
score_population_proxy_smoothed_residual = score_population_proxy_v5
score_population_full_smoothed_residual = score_population_full_v5
