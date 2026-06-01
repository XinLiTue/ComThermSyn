"""
Internal validation framework: sanity checks, PIT validation,
conditional distribution checks, monotonicity, and case validation.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    ENERGY_PRIORITY_COLS,
    LABEL_GROUP_ORDER,
    LABEL_SCORE,
    TAU_COARSE_GROUPS,
    VALIDATION_MODE_CONFIGS,
    CUSTOM_DAY_END_HOUR,
    CUSTOM_DAY_SETPOINT_C,
    CUSTOM_DAY_START_HOUR,
    CUSTOM_INITIAL_TEMP_C,
    CUSTOM_OTHER_SETPOINT_C,
    CUSTOM_SUMMER_MONTHS,
    get_validation_mode_config,
)
from src.feature_engineering import clean_energy_label, vintage_from_year
from src.scoring import wasserstein_1d
from src.utils import safe_numeric as _safe_numeric

try:
    from scipy.stats import chi2, kstest, spearmanr
except Exception:
    chi2 = None
    kstest = None
    spearmanr = None

try:
    from scipy.signal import welch
except Exception:
    welch = None

DEFAULT_CASE_VALIDATION_DATA_PATH = (
    Path("DACS-Data")
    / "JupyterCode"
    / "processed_v1"
    / "P2_2024_10_to_2025_03__train_strict_core_weather.parquet"
)


# ---------------------------------------------------------------------------
# Column / series helpers
# ---------------------------------------------------------------------------

def _first_valid_column(df: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _with_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "vintage" not in out.columns and "year" in out.columns:
        out["vintage"] = out["year"].map(vintage_from_year)
    if "C_per_area" not in out.columns and {"C", "Area"}.issubset(out.columns):
        area = _safe_numeric(out["Area"]).replace(0, np.nan)
        out["C_per_area"] = _safe_numeric(out["C"]) / area
    if "label_clean" not in out.columns and "EnergyLabel" in out.columns:
        out["label_clean"] = out["EnergyLabel"].map(clean_energy_label)
    if "label_score" not in out.columns:
        labels = out.get("label_clean", out.get("EnergyLabel"))
        if labels is not None:
            out["label_score"] = labels.map(
                lambda x: float(LABEL_SCORE.get(clean_energy_label(x), np.nan))
            )
    if "tau" not in out.columns and {"R", "C"}.issubset(out.columns):
        out["tau"] = _safe_numeric(out["R"]) * _safe_numeric(out["C"])
    return out


def _energy_series(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    for col in ENERGY_PRIORITY_COLS:
        if col in df.columns:
            vals = _safe_numeric(df[col])
            if vals.notna().any():
                return vals, col
    return pd.Series(np.nan, index=df.index, dtype=float), None


def _summer_share_series(df: pd.DataFrame) -> pd.Series:
    if "summer_kwh_share" in df.columns:
        return _safe_numeric(df["summer_kwh_share"])
    if "summer_heat_share_pct" in df.columns:
        return _safe_numeric(df["summer_heat_share_pct"]) / 100.0
    return pd.Series(np.nan, index=df.index, dtype=float)


def _coarse_vintage(vintage: str) -> str:
    return TAU_COARSE_GROUPS.get(str(vintage), "global")


def _label_group(label: Any) -> str:
    label = clean_energy_label(label)
    if label in {"A", "A+", "A++", "A+++", "A++++"}:
        return "A-family"
    if label == "B":
        return "B"
    if label == "C":
        return "C"
    return "D-G"


def _quantile_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    left = max(float(a_lo), float(b_lo))
    right = min(float(a_hi), float(b_hi))
    denom = max(float(max(a_hi, b_hi) - min(a_lo, b_lo)), 1e-9)
    return max(0.0, right - left) / denom


def _uniform_ks_pvalue(values: Any) -> float:
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) == 0:
        return np.nan
    if kstest is not None:
        return float(kstest(values, "uniform").pvalue)
    values = np.sort(values)
    n = len(values)
    cdf_emp = np.arange(1, n + 1) / n
    d = max(
        np.max(np.abs(cdf_emp - values)),
        np.max(np.abs((np.arange(n) / n) - values)),
    )
    return float(np.exp(-2.0 * n * d * d))


def _pit_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, part in detail_df.groupby("variable"):
        pits = part["pit"].dropna().to_numpy(dtype=float)
        if len(pits) == 0:
            continue
        rows.append({
            "variable": variable,
            "n_obs": int(len(pits)),
            "pit_mean": float(np.mean(pits)),
            "pit_std": float(np.std(pits, ddof=0)),
            "coverage_80": float(np.mean((pits >= 0.10) & (pits <= 0.90))),
            "coverage_90": float(np.mean((pits >= 0.05) & (pits <= 0.95))),
            "tail_rate": float(np.mean((pits < 0.05) | (pits > 0.95))),
            "ks_pvalue": _uniform_ks_pvalue(pits),
            "ideal_pit_mean": 0.5,
            "ideal_pit_std": float(np.sqrt(1.0 / 12.0)),
            "ideal_coverage_80": 0.8,
            "ideal_coverage_90": 0.9,
            "ideal_tail_rate": 0.1,
        })
    return pd.DataFrame(rows)


def get_validation_output_dir(base_dir: Optional[Path] = None) -> Path:
    out_dir = Path(base_dir or DATA_DIR) / "validation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_generation_kwargs(
    n_population_candidates: int,
    top_k_full: int,
    disable_label_penalty: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    beta_label = 0.0 if disable_label_penalty else -3.0
    top_k_full_eff = max(20, int(top_k_full))
    if n_population_candidates < top_k_full_eff:
        warnings.warn(
            f"n_population_candidates={n_population_candidates} is smaller than the "
            f"PIT target top_k_full={top_k_full_eff}; using {min(20, n_population_candidates)}."
        )
        top_k_full_eff = min(20, int(n_population_candidates))
    top_k_proxy_eff = min(max(6, top_k_full_eff), int(n_population_candidates))
    if top_k_full_eff < 10:
        warnings.warn(
            f"Only {top_k_full_eff} selected samples per held-out building are available "
            "after filtering; PIT resolution is coarse and should be interpreted cautiously."
        )
    return {
        "n_population_candidates": int(n_population_candidates),
        "top_k_proxy": int(top_k_proxy_eff),
        "top_k_full": int(top_k_full_eff),
        "random_state": int(random_state),
        "beta_label_E": float(beta_label),
    }


def _generate_with_final_pipeline(
    train_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_population_candidates: int,
    top_k_full: int,
    random_state: int = 42,
    disable_label_penalty: bool = False,
    use_v5: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    from src.thermal_synthesis import (
        fit_thermal_parameter_store as fit_group_generator,
        synthesize_community_bootstrap as generate_community_rcaq,
    )

    gen_cv = fit_group_generator(train_df)
    kwargs = _build_generation_kwargs(
        n_population_candidates=n_population_candidates,
        top_k_full=top_k_full,
        disable_label_penalty=disable_label_penalty,
        random_state=random_state,
    )

    if use_v5:
        try:
            from src.thermal_synthesis import (
                MIN_RESID_SAMPLES,
                compute_residual_statistics,
                synthesize_community_smoothed_residual,
            )
            resid_stats = compute_residual_statistics(gen_cv, min_samples=MIN_RESID_SAMPLES)
            return synthesize_community_smoothed_residual(
                community_conditions=condition_df.copy(),
                gen=gen_cv,
                weather_by_year=weather_by_year,
                r_resid_std_by_vintage=resid_stats[0],
                global_r_resid_std=resid_stats[1],
                resid_feat_stds_by_vint=resid_stats[2],
                global_resid_feat_stds=resid_stats[3],
                attach_profiles=False,
                **kwargs,
            )
        except ImportError:
            pass

    try:
        return generate_community_rcaq(
            community_conditions=condition_df.copy(),
            gen=gen_cv,
            weather_by_year=weather_by_year,
            attach_profiles=False,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("beta_label_E", None)
        return generate_community_rcaq(
            community_conditions=condition_df.copy(),
            gen=gen_cv,
            weather_by_year=weather_by_year,
            attach_profiles=False,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Layer 1 – Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(
    selected_df: pd.DataFrame,
    training_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    selected = _with_core_columns(selected_df)
    training = _with_core_columns(training_df)
    report = selected.copy()

    energy_selected, energy_col_selected = _energy_series(selected)
    energy_training, energy_col_training = _energy_series(training)
    summer_share = _summer_share_series(selected)

    qint_q01 = float(_safe_numeric(training["Qint"]).quantile(0.01))
    qint_q99 = float(_safe_numeric(training["Qint"]).quantile(0.99))
    e_threshold = (
        float(energy_training.quantile(0.99)) * 1.5
        if energy_col_training is not None
        else np.nan
    )

    peak_per_m2_train = pd.Series(np.nan, index=training.index, dtype=float)
    if {"peak_heat_kw", "Area"}.issubset(training.columns):
        peak_per_m2_train = _safe_numeric(training["peak_heat_kw"]) / _safe_numeric(training["Area"]).replace(0, np.nan)
    peak_q95 = float(peak_per_m2_train.dropna().quantile(0.95)) if peak_per_m2_train.notna().any() else np.nan

    tau_global_lo = float(training["tau"].quantile(0.05))
    tau_global_hi = float(training["tau"].quantile(0.95))

    tau_bounds_by_vintage: Dict[str, Tuple[float, float]] = {}
    for vintage, part in training.groupby("vintage"):
        if len(part) >= 8:
            tau_bounds_by_vintage[str(vintage)] = (
                float(part["tau"].quantile(0.05)),
                float(part["tau"].quantile(0.95)),
            )

    tau_bounds_by_coarse: Dict[str, Tuple[float, float]] = {}
    coarse_training = training.assign(vintage_coarse=training["vintage"].map(_coarse_vintage))
    for coarse_vintage, part in coarse_training.groupby("vintage_coarse"):
        if len(part) >= 8:
            tau_bounds_by_coarse[str(coarse_vintage)] = (
                float(part["tau"].quantile(0.05)),
                float(part["tau"].quantile(0.95)),
            )

    train_maha = training[["R", "C_per_area", "A"]].apply(_safe_numeric).replace([np.inf, -np.inf], np.nan).dropna()
    maha_inv_cov = None
    maha_center = None
    maha_cutoff = float(chi2.ppf(0.99, df=3)) if chi2 is not None else 11.344867
    if len(train_maha) >= 5:
        maha_center = train_maha.mean().to_numpy(dtype=float)
        cov = np.cov(train_maha.to_numpy(dtype=float).T)
        maha_inv_cov = np.linalg.pinv(cov)

    maha_distances = []
    peak_warnings = []

    for idx, row in report.iterrows():
        vintage = str(row.get("vintage"))
        tau_lo, tau_hi = tau_global_lo, tau_global_hi
        tau_source = "global"
        if vintage in tau_bounds_by_vintage:
            tau_lo, tau_hi = tau_bounds_by_vintage[vintage]
            tau_source = "vintage"
        else:
            coarse = _coarse_vintage(vintage)
            if coarse in tau_bounds_by_coarse:
                tau_lo, tau_hi = tau_bounds_by_coarse[coarse]
                tau_source = "coarse_vintage"

        r_ok = bool(pd.notna(row.get("R")) and float(row["R"]) > 0)
        c_ok = bool(pd.notna(row.get("C")) and float(row["C"]) > 0)
        a_ok = bool(pd.notna(row.get("A")) and float(row["A"]) > 0)
        qint_ok = bool(pd.notna(row.get("Qint")) and qint_q01 <= float(row["Qint"]) <= qint_q99)
        tau_val = float(row["tau"]) if pd.notna(row.get("tau")) else np.nan
        tau_ok = bool(pd.notna(tau_val) and tau_lo <= tau_val <= tau_hi)
        energy_val = float(energy_selected.loc[idx]) if pd.notna(energy_selected.loc[idx]) else np.nan
        e_ok = bool(pd.isna(e_threshold) or (pd.notna(energy_val) and energy_val <= e_threshold))
        summer_val = float(summer_share.loc[idx]) if pd.notna(summer_share.loc[idx]) else np.nan
        summer_ok = bool(pd.isna(summer_val) or summer_val < 0.001)

        peak_per_m2 = np.nan
        peak_warn = False
        if {"peak_heat_kw", "Area"}.issubset(report.columns) and pd.notna(row.get("Area")) and float(row["Area"]) > 0:
            peak_per_m2 = float(row["peak_heat_kw"]) / float(row["Area"])
            peak_warn = bool(peak_per_m2 > 0.12 and (pd.isna(peak_q95) or peak_per_m2 > peak_q95))
        peak_warnings.append(peak_warn)

        maha = np.nan
        maha_ok = True
        if maha_inv_cov is not None:
            vec = np.array([row.get("R", np.nan), row.get("C_per_area", np.nan), row.get("A", np.nan)], dtype=float)
            if np.all(np.isfinite(vec)):
                diff = vec - maha_center
                maha = float(diff.T @ maha_inv_cov @ diff)
                maha_ok = bool(maha <= maha_cutoff)
        maha_distances.append(maha)

        report.loc[idx, "check_R_positive"] = r_ok
        report.loc[idx, "check_C_positive"] = c_ok
        report.loc[idx, "check_A_positive"] = a_ok
        report.loc[idx, "check_Qint_q01_q99"] = qint_ok
        report.loc[idx, "check_tau_q05_q95"] = tau_ok
        report.loc[idx, "tau_check_source"] = tau_source
        report.loc[idx, "check_energy_below_q99x1p5"] = e_ok
        report.loc[idx, "check_summer_share_lt_0p001"] = summer_ok
        report.loc[idx, "check_mahalanobis_q99"] = maha_ok

    report["mahalanobis_distance"] = maha_distances
    report["mahalanobis_cutoff_q99"] = maha_cutoff
    report["energy_validation_col"] = energy_col_selected
    report["summer_share_fraction"] = summer_share
    report["peak_warning"] = peak_warnings

    hard_cols = [
        "check_R_positive", "check_C_positive", "check_A_positive",
        "check_Qint_q01_q99", "check_tau_q05_q95",
        "check_energy_below_q99x1p5", "check_summer_share_lt_0p001",
        "check_mahalanobis_q99",
    ]
    report["sanity_pass"] = report[hard_cols].fillna(False).all(axis=1)

    summary: Dict[str, Any] = {
        "n_rows": int(len(report)),
        "n_pass": int(report["sanity_pass"].sum()),
        "pass_rate": float(report["sanity_pass"].mean()) if len(report) else np.nan,
        "n_peak_warnings": int(np.sum(report["peak_warning"])),
        "energy_col_selected": energy_col_selected,
        "energy_col_training": energy_col_training,
    }
    fail_counts = {col: int((~report[col].fillna(False)).sum()) for col in hard_cols}
    summary.update(fail_counts)

    if any(v > 0 for v in fail_counts.values()) or summary["n_peak_warnings"] > 0:
        print("Sanity-check warnings:")
        print(summary)
    return report, summary


# ---------------------------------------------------------------------------
# Layer 2 – PIT / coverage / label-penalty / conditional validation
# ---------------------------------------------------------------------------

def run_pit_validation_v5(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: Optional[int] = None,
    group_size: int = 6,
    n_population_candidates: Optional[int] = None,
    top_k_full: int = 20,
    random_state: int = 42,
    disable_label_penalty: bool = True,
    validation_mode: str = "fast",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mode_cfg = get_validation_mode_config(validation_mode)
    n_repeats = mode_cfg["n_repeats"] if n_repeats is None else int(n_repeats)
    n_population_candidates = (
        mode_cfg["n_population_candidates"] if n_population_candidates is None else int(n_population_candidates)
    )

    work = _with_core_columns(df_in).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    detail_rows: List[Dict] = []

    for repeat in range(n_repeats):
        test_idx = np.sort(rng.choice(len(work), size=group_size, replace=False))
        train_df = work.drop(index=test_idx).reset_index(drop=True)
        test_df = work.iloc[test_idx].copy().reset_index(drop=True)
        test_df["building_id"] = [f"holdout_{repeat+1:02d}_{i+1:02d}" for i in range(len(test_df))]
        cond_df = test_df[["building_id", "year", "Area", "EnergyLabel"]].copy()

        selected_df, score_df, _ = _generate_with_final_pipeline(
            train_df=train_df,
            condition_df=cond_df,
            weather_by_year=weather_by_year,
            n_population_candidates=n_population_candidates,
            top_k_full=max(20, int(top_k_full)),
            random_state=random_state + repeat,
            disable_label_penalty=disable_label_penalty,
        )
        selected_df = _with_core_columns(selected_df)

        for _, true_row in test_df.iterrows():
            bldg_id = true_row["building_id"]
            gen_part = selected_df.loc[selected_df["building_id"] == bldg_id].copy()
            if len(gen_part) == 0:
                warnings.warn(f"No generated samples found for {bldg_id} in repeat {repeat + 1}.")
                continue
            n_samples = int(gen_part["population_id"].nunique()) if "population_id" in gen_part.columns else int(len(gen_part))
            if n_samples < 10:
                warnings.warn(
                    f"Only {n_samples} samples available for {bldg_id} in repeat {repeat + 1}; "
                    "PIT resolution is coarse."
                )

            comparisons = {
                "E_std": (
                    float(true_row["E_std_annual_per_m2"]),
                    _safe_numeric(gen_part[_first_valid_column(gen_part, ["E_std_full_per_m2", "E_proxy_annual_per_m2"])]),
                ),
                "R": (float(true_row["R"]), _safe_numeric(gen_part["R"])),
                "C_per_area": (float(true_row["C_per_area"]), _safe_numeric(gen_part["C_per_area"])),
            }

            for variable, (truth, gen_values) in comparisons.items():
                gen_values = pd.Series(gen_values).dropna()
                if len(gen_values) == 0 or not np.isfinite(truth):
                    continue
                pit = float((gen_values < truth).mean()) + 0.5 * float((gen_values == truth).mean())
                detail_rows.append({
                    "repeat": int(repeat + 1),
                    "building_id": bldg_id,
                    "variable": variable,
                    "pit": pit,
                    "truth_value": truth,
                    "n_generated_samples": int(len(gen_values)),
                    "disable_label_penalty": bool(disable_label_penalty),
                    "score_min": float(score_df["score_full"].min()) if "score_full" in score_df.columns and len(score_df) else np.nan,
                })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = _pit_summary(detail_df)
    return detail_df, summary_df


def run_label_penalty_effect_validation(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: Optional[int] = None,
    group_size: int = 6,
    n_population_candidates: Optional[int] = None,
    random_state: int = 42,
    beta_label_E_normal: float = -3.0,
    validation_mode: str = "fast",
) -> pd.DataFrame:
    mode_cfg = get_validation_mode_config(validation_mode)
    n_repeats = mode_cfg["n_repeats"] if n_repeats is None else int(n_repeats)
    n_population_candidates = (
        mode_cfg["n_population_candidates"] if n_population_candidates is None else int(n_population_candidates)
    )

    work = _with_core_columns(df_in).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    rows: List[Dict] = []

    for repeat in range(n_repeats):
        test_idx = np.sort(rng.choice(len(work), size=group_size, replace=False))
        train_df = work.drop(index=test_idx).reset_index(drop=True)
        test_df = work.iloc[test_idx].copy().reset_index(drop=True)
        test_df["building_id"] = [f"label_{repeat+1:02d}_{i+1:02d}" for i in range(len(test_df))]
        cond_df = test_df[["building_id", "year", "Area", "EnergyLabel"]].copy()

        no_label_df, _, _ = _generate_with_final_pipeline(
            train_df=train_df,
            condition_df=cond_df,
            weather_by_year=weather_by_year,
            n_population_candidates=n_population_candidates,
            top_k_full=20,
            random_state=random_state + repeat,
            disable_label_penalty=True,
        )
        normal_df, _, _ = _generate_with_final_pipeline(
            train_df=train_df,
            condition_df=cond_df,
            weather_by_year=weather_by_year,
            n_population_candidates=n_population_candidates,
            top_k_full=20,
            random_state=random_state + 10_000 + repeat,
            disable_label_penalty=False,
        )

        best_no_label = _with_core_columns(no_label_df).query("selected_rank == 1").reset_index(drop=True)
        best_normal = _with_core_columns(normal_df).query("selected_rank == 1").reset_index(drop=True)
        truth = _with_core_columns(test_df)

        rows.append({
            "repeat": int(repeat + 1),
            "wasserstein_E_no_label": float(wasserstein_1d(
                truth["E_std_annual_per_m2"].to_numpy(dtype=float),
                best_no_label["E_std_full_per_m2"].to_numpy(dtype=float),
            )),
            "wasserstein_E_normal": float(wasserstein_1d(
                truth["E_std_annual_per_m2"].to_numpy(dtype=float),
                best_normal["E_std_full_per_m2"].to_numpy(dtype=float),
            )),
            "wasserstein_R_no_label": float(wasserstein_1d(
                truth["R"].to_numpy(dtype=float), best_no_label["R"].to_numpy(dtype=float)
            )),
            "wasserstein_R_normal": float(wasserstein_1d(
                truth["R"].to_numpy(dtype=float), best_normal["R"].to_numpy(dtype=float)
            )),
            "beta_label_E_no_label": 0.0,
            "beta_label_E_normal": float(beta_label_E_normal),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["delta_wasserstein_E_normal_minus_no_label"] = out["wasserstein_E_normal"] - out["wasserstein_E_no_label"]
        out["delta_wasserstein_R_normal_minus_no_label"] = out["wasserstein_R_normal"] - out["wasserstein_R_no_label"]
    return out


def run_conditional_distribution_validation(
    train_df: pd.DataFrame,
    generated_df: pd.DataFrame,
) -> pd.DataFrame:
    train = _with_core_columns(train_df).copy()
    generated = _with_core_columns(generated_df).copy()
    if len(generated) == 0:
        return pd.DataFrame()

    area_bins = pd.qcut(_safe_numeric(train["Area"]), q=3, duplicates="drop")
    edges = np.unique(area_bins.cat.categories.left.tolist() + [area_bins.cat.categories[-1].right])
    if len(edges) < 2:
        edges = np.array([train["Area"].min(), train["Area"].max()])

    def assign_bins(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label_group"] = out["EnergyLabel"].map(_label_group)
        out["vintage_group"] = out["vintage"].astype(str)
        out["area_bin"] = pd.cut(_safe_numeric(out["Area"]), bins=edges, include_lowest=True, duplicates="drop").astype(str)
        return out

    train = assign_bins(train)
    generated = assign_bins(generated)

    energy_train, energy_col_train = _energy_series(train)
    energy_gen, energy_col_gen = _energy_series(generated)
    train["_energy_val"] = energy_train
    generated["_energy_val"] = energy_gen

    rows: List[Dict] = []
    group_cols = ["label_group", "vintage_group", "area_bin"]
    keys = sorted(
        set(map(tuple, train[group_cols].drop_duplicates().to_numpy()))
        & set(map(tuple, generated[group_cols].drop_duplicates().to_numpy()))
    )
    for key in keys:
        train_part = train.loc[(train[group_cols] == list(key)).all(axis=1)].copy()
        gen_part = generated.loc[(generated[group_cols] == list(key)).all(axis=1)].copy()
        if len(train_part) < 3 or len(gen_part) < 3:
            continue
        for variable, tvals, gvals in [
            ("E", train_part["_energy_val"], gen_part["_energy_val"]),
            ("R", train_part["R"], gen_part["R"]),
        ]:
            tvals = _safe_numeric(tvals).dropna()
            gvals = _safe_numeric(gvals).dropna()
            if len(tvals) < 3 or len(gvals) < 3:
                continue
            t25, t75 = np.quantile(tvals, [0.25, 0.75])
            g25, g75 = np.quantile(gvals, [0.25, 0.75])
            g10, g90 = np.quantile(gvals, [0.10, 0.90])
            rows.append({
                "label_group": key[0],
                "vintage_group": key[1],
                "area_bin": key[2],
                "variable": variable,
                "n_train": int(len(tvals)),
                "n_generated": int(len(gvals)),
                "wasserstein_distance": float(wasserstein_1d(tvals.to_numpy(dtype=float), gvals.to_numpy(dtype=float))),
                "q25_q75_overlap": float(_quantile_overlap(t25, t75, g25, g75)),
                "coverage_rate": float(np.mean((tvals >= g10) & (tvals <= g90))),
                "train_energy_col": energy_col_train,
                "generated_energy_col": energy_col_gen,
                "generated_pool_source": generated.attrs.get("pool_source", "unknown"),
            })
    return pd.DataFrame(rows)


def run_monotonicity_validation(df: pd.DataFrame) -> pd.DataFrame:
    work = _with_core_columns(df).copy()

    def _spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        pair = pd.DataFrame({"x": _safe_numeric(x), "y": _safe_numeric(y)}).dropna()
        if len(pair) < 3:
            return np.nan, np.nan
        if spearmanr is not None:
            corr, pvalue = spearmanr(pair["x"], pair["y"])
            return float(corr), float(pvalue)
        corr = pair["x"].rank().corr(pair["y"].rank())
        return float(corr), np.nan

    rows: List[Dict] = []
    for check_name, x, y, expected, note in [
        ("area_vs_C",       work["Area"],       work["C"],          "positive", "Expected positive."),
        ("area_vs_A",       work["Area"],       work["A"],          "positive", "Expected positive."),
        ("label_score_vs_E", work["label_score"], _energy_series(work)[0], "negative", "Weak evidence only; do not fail if insignificant."),
    ]:
        corr, pvalue = _spearman(x, y)
        rows.append({
            "check": check_name,
            "spearman_corr": corr,
            "pvalue": pvalue,
            "expected_direction": expected,
            "interpretation": note,
            "passes_directional_expectation": (
                np.nan if pd.isna(corr)
                else bool((corr >= 0 and expected == "positive") or (corr <= 0 and expected == "negative"))
            ),
        })

    vintage_mean_r = work.groupby("vintage", dropna=False)["R"].mean().sort_index().reset_index(name="mean_R")
    for _, row in vintage_mean_r.iterrows():
        rows.append({
            "check": "vintage_mean_R",
            "vintage": row["vintage"],
            "mean_R": float(row["mean_R"]),
            "interpretation": "Observed trend only; no directional failure imposed.",
        })
    old_mean = vintage_mean_r.loc[vintage_mean_r["vintage"].astype(str).isin(["pre1965", "1965-1974"]), "mean_R"].mean()
    newer_mean = vintage_mean_r.loc[~vintage_mean_r["vintage"].astype(str).isin(["pre1965", "1965-1974"]), "mean_R"].mean()
    rows.append({
        "check": "vintage_R_finding",
        "interpretation": (
            "Finding: pre1965/1965-1974 mean R exceeds newer mean R."
            if pd.notna(old_mean) and pd.notna(newer_mean) and old_mean > newer_mean
            else "No contradictory old>newer mean R finding observed in this pool."
        ),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 3 – Case validation (external anchor)
# ---------------------------------------------------------------------------

def load_measured_case_data(
    path: Path,
    use_ta_as_tamb: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path)
    out = df.copy()

    rename_map: Dict[str, str] = {}
    if "participant_id" in out.columns:
        rename_map["participant_id"] = "user_id"
    if "date" in out.columns:
        rename_map["date"] = "time"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "Tin" not in out.columns and "indoor_actual_C" in out.columns:
        out["Tin"] = _safe_numeric(out["indoor_actual_C"])
    if "Tamb" not in out.columns:
        tamb_col = "ta" if use_ta_as_tamb else "outdoor_C"
        if tamb_col in out.columns:
            out["Tamb"] = _safe_numeric(out[tamb_col])
        elif use_ta_as_tamb and "outdoor_C" in out.columns:
            out["Tamb"] = _safe_numeric(out["outdoor_C"])
    if "Q" not in out.columns:
        q_cols = [c for c in ["boiler_power_kW", "hp_power_kW"] if c in out.columns]
        if q_cols:
            out["Q"] = out.reindex(columns=q_cols).fillna(0).sum(axis=1)
    if "S" not in out.columns and "qg" in out.columns:
        out["S"] = _safe_numeric(out["qg"]) / 1000.0
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"])

    required = ["user_id", "time", "Tin", "Tamb", "S", "Q"]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    return out[required].copy()


def select_case_validation_buildings(model_df: pd.DataFrame) -> pd.DataFrame:
    work = _with_core_columns(model_df).copy()
    if "user_id" not in work.columns:
        warnings.warn("model_df has no user_id column; case validation cannot match measured data.")
        return pd.DataFrame()

    work["label_group"] = work["EnergyLabel"].map(_label_group)
    old_mask = work["vintage"].astype(str).isin(["pre1965", "1965-1974"])

    selectors = [
        ("Case A", old_mask & work["label_group"].isin(["C", "D-G"])),
        ("Case B", old_mask & (work["label_group"] == "A-family")),
        ("Case C", pd.Series(True, index=work.index)),
    ]

    used_users: set = set()
    area_median = float(_safe_numeric(work["Area"]).median())
    candidates: List[Dict] = []

    for case_name, mask in selectors:
        part = work.loc[mask].copy()
        if len(part) == 0:
            continue
        part = part.loc[~part["user_id"].isin(used_users)].copy()
        if len(part) == 0:
            continue
        if case_name == "Case C":
            part["area_distance"] = (_safe_numeric(part["Area"]) - area_median).abs()
            chosen = part.sort_values("area_distance").iloc[0]
        else:
            chosen = part.sort_values(["Area"]).iloc[0]
        used_users.add(chosen["user_id"])
        candidates.append({
            "case_name": case_name,
            "case_source": "identified_real_building",
            "user_id": chosen["user_id"],
            "R": float(chosen["R"]),
            "C": float(chosen["C"]),
            "A": float(chosen["A"]),
            "Qint": float(chosen["Qint"]),
            "Area": float(chosen["Area"]),
            "year": float(chosen["year"]),
            "EnergyLabel": chosen.get("EnergyLabel"),
            "vintage": chosen.get("vintage"),
        })
    return pd.DataFrame(candidates)


def _infer_dt_hours(index: Any) -> float:
    if len(index) < 2:
        return 1.0
    dt = (pd.DatetimeIndex(index)[1] - pd.DatetimeIndex(index)[0]).total_seconds() / 3600.0
    return float(dt) if dt > 0 else 1.0


def _autocorr_vector(values: Any, max_lag: int = 24) -> np.ndarray:
    x = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(x) < max_lag + 2:
        return np.full(max_lag, np.nan)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.full(max_lag, np.nan)
    return np.array([np.dot(x[:-lag], x[lag:]) / denom for lag in range(1, max_lag + 1)], dtype=float)


def _daily_cycle_peak(values: Any, dt_hours: float) -> float:
    x = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(x) < 16:
        return np.nan
    fs = 1.0 / max(dt_hours, 1e-9)
    target = 1.0 / 24.0
    if welch is not None:
        freqs, power = welch(x, fs=fs, nperseg=min(256, len(x)))
        return float(power[np.argmin(np.abs(freqs - target))])
    freqs = np.fft.rfftfreq(len(x), d=dt_hours)
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    return float(power[np.argmin(np.abs(freqs - target))])


def _dtw_distance(a: Any, b: Any) -> float:
    try:
        from fastdtw import fastdtw
        distance, _ = fastdtw(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        return float(distance)
    except Exception:
        return np.nan


def run_case_validation(
    case_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    simulate_heating_profile_custom_fn: Optional[Callable] = None,
    output_dir: Optional[Path] = None,
    initial_temp: float = CUSTOM_INITIAL_TEMP_C,
    day_start: int = CUSTOM_DAY_START_HOUR,
    day_end: int = CUSTOM_DAY_END_HOUR,
    day_setpoint: float = CUSTOM_DAY_SETPOINT_C,
    other_setpoint: float = CUSTOM_OTHER_SETPOINT_C,
    summer_months: tuple = CUSTOM_SUMMER_MONTHS,
) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if simulate_heating_profile_custom_fn is None:
        from src.simulation import simulate_heating_profile_custom
        simulate_heating_profile_custom_fn = simulate_heating_profile_custom

    if output_dir is None:
        output_dir = get_validation_output_dir()

    rows: List[Dict] = []
    for _, case_row in case_df.iterrows():
        user_id = case_row["user_id"]
        measured_part = measured_df.loc[measured_df["user_id"] == user_id].copy()
        if len(measured_part) == 0:
            rows.append({"case_name": case_row["case_name"], "user_id": user_id, "status": "skipped_no_measured_data"})
            continue

        measured_part = measured_part.sort_values("time").dropna(subset=["time", "Tamb", "S", "Q"])
        weather_input = (
            measured_part[["time", "Tamb", "S"]]
            .rename(columns={"time": "datetime"})
            .assign(ta=lambda x: _safe_numeric(x["Tamb"]), qg=lambda x: _safe_numeric(x["S"]) * 1000.0)
            .set_index("datetime")[["ta", "qg"]]
        )

        sim_profile = simulate_heating_profile_custom_fn(
            case_row.to_dict(),
            weather_input,
            summer_months=summer_months,
            initial_temp=initial_temp,
            day_start=day_start,
            day_end=day_end,
            day_setpoint=day_setpoint,
            other_setpoint=other_setpoint,
        )

        compare_df = (
            measured_part.set_index("time")[["Q", "Tin", "Tamb", "S"]]
            .join(sim_profile[["heat_kw", "indoor_temp_C"]], how="inner")
            .dropna()
        )
        if len(compare_df) == 0:
            rows.append({"case_name": case_row["case_name"], "user_id": user_id, "status": "skipped_no_time_overlap"})
            continue

        measured_q = _safe_numeric(compare_df["Q"]).to_numpy(dtype=float)
        sim_q = _safe_numeric(compare_df["heat_kw"]).to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((sim_q - measured_q) ** 2)))
        mean_measured = float(np.mean(measured_q)) if len(measured_q) else np.nan
        cv_rmse = float(100.0 * rmse / max(mean_measured, 1e-9))
        nmbe = float(100.0 * np.mean(sim_q - measured_q) / max(mean_measured, 1e-9))
        dt_hours = _infer_dt_hours(compare_df.index)
        measured_peak = float(np.quantile(measured_q, 0.95))
        simulated_peak = float(np.quantile(sim_q, 0.95))
        measured_lf = float(np.mean(measured_q) / max(np.max(measured_q), 1e-9))
        simulated_lf = float(np.mean(sim_q) / max(np.max(sim_q), 1e-9))
        acf_sim = _autocorr_vector(sim_q, max_lag=24)
        acf_meas = _autocorr_vector(measured_q, max_lag=24)
        acf_similarity = (
            float(1.0 - np.nanmean(np.abs(acf_sim - acf_meas)))
            if np.isfinite(acf_sim).any() and np.isfinite(acf_meas).any()
            else np.nan
        )
        psd_measured = _daily_cycle_peak(measured_q, dt_hours)
        psd_simulated = _daily_cycle_peak(sim_q, dt_hours)

        fig, ax = plt.subplots(figsize=(12, 4))
        compare_df[["Q", "heat_kw"]].rename(columns={"Q": "measured_Q", "heat_kw": "simulated_Q"}).plot(ax=ax, alpha=0.8)
        ax.set_title(f"{case_row['case_name']} measured vs simulated heat power")
        ax.set_ylabel("kW")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{case_row['case_name'].lower().replace(' ', '_')}_measured_vs_simulated.png", dpi=150)
        plt.close(fig)

        rows.append({
            "case_name": case_row["case_name"],
            "user_id": user_id,
            "status": "ok",
            "n_points": int(len(compare_df)),
            "cv_rmse_pct": cv_rmse,
            "nmbe_pct": nmbe,
            "cv_rmse_ref_lt_30": bool(cv_rmse < 30.0),
            "nmbe_ref_within_pm10": bool(abs(nmbe) <= 10.0),
            "tau_generated": float(case_row["R"] * case_row["C"]),
            "tau_estimated_from_measured": np.nan,
            "tau_ratio_generated_over_measured": np.nan,
            "peak_heat_kw": float(np.max(sim_q)),
            "simulated_q95_heat_kw": simulated_peak,
            "measured_q95_heat_kw": measured_peak,
            "simulated_load_factor": simulated_lf,
            "measured_load_factor": measured_lf,
            "acf_similarity_lag1_24": acf_similarity,
            "psd_daily_peak_simulated": psd_simulated,
            "psd_daily_peak_measured": psd_measured,
            "dtw_distance": _dtw_distance(sim_q, measured_q),
            "supplementary_note": "ACF, PSD, and DTW are supplementary structural-similarity evidence only.",
        })

    return pd.DataFrame(rows)
