"""
Unified thermal profile synthesis API.

The existing implementations are retained as synthesis strategies:

- V3: legacy parametric sampler.
- V4: bootstrap residual synthesis with hard/multiplicative renovation lift.
- V5: smoothed residual synthesis with soft/additive renovation adjustment.

These are strategy names for conditional thermal-parameter synthesis, rather
than separate generic generator concepts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_RENOVATION_LIFT,
    GROUP_FEATURES_CPA,
    GROUP_FEATURES_RAW_C,
    LABEL_SCORE,
    PROFILE_YEAR,
    RNG_SEED,
    VINTAGE_EXPECTED_LABEL_SCORE,
)
from src.feature_engineering import (
    clean_energy_label,
    prepare_building_features,
    vintage_from_year,
)
from src.scoring import (
    label_ordering_metrics_v5,
    label_ordering_score,
    mahalanobis_penalty,
    official_label_penalty,
    score_population_full_bootstrap,
    score_population_full_smoothed_residual,
    score_population_proxy_bootstrap,
    score_population_proxy_smoothed_residual,
    sigmoid,
    summarize_series,
    to_stats_dict,
    wasserstein_1d,
)
from src.simulation import (
    estimate_std_energy,
    estimate_std_energy_typical,
    load_or_build_typical_weather_scenarios,
    simulate_heating_profile,
    simulate_standard_year_summary,
)

MIN_RESID_SAMPLES: int = 8


# ---------------------------------------------------------------------------
# Shared store fitting and bootstrap residual data
# ---------------------------------------------------------------------------

@dataclass
class ThermalParameterStore:
    baseline_by_vintage: Dict[str, Dict[str, float]]
    global_baseline: Dict[str, float]
    residuals_by_vintage: Dict[str, pd.DataFrame]
    global_residuals: pd.DataFrame
    renovation_lift_by_vintage: Dict[str, float]
    global_renovation_lift: float
    train_energy_stats: Dict[str, float]
    energy_stats_by_vintage: Dict[Tuple, Dict[str, float]]
    energy_stats_by_label: Dict[Tuple, Dict[str, float]]
    energy_stats_by_vintage_label: Dict[Tuple, Dict[str, float]]
    r_clip_by_vintage: Dict[Tuple, Dict[str, float]]
    r_clip_by_vintage_label: Dict[Tuple, Dict[str, float]]
    qint_std: float
    capacity_mode: str
    train_df: pd.DataFrame


GroupGenerator = ThermalParameterStore


def fit_thermal_parameter_store(
    df_in: pd.DataFrame, capacity_mode: str = "per_area"
) -> ThermalParameterStore:
    work = prepare_building_features(df_in)
    if "E_std_annual_per_m2" not in work.columns:
        raise ValueError("fit_thermal_parameter_store requires E_std_annual_per_m2 in the training data.")
    if capacity_mode not in {"per_area", "raw"}:
        raise ValueError("capacity_mode must be either 'per_area' or 'raw'.")

    features = GROUP_FEATURES_CPA if capacity_mode == "per_area" else GROUP_FEATURES_RAW_C
    required = ["R", "A", "Qint", "E_std_annual_per_m2"] + (
        ["C_per_area"] if capacity_mode == "per_area" else ["C"]
    )
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise ValueError(f"fit_thermal_parameter_store missing required columns: {missing}")

    fit_df = work.dropna(subset=required).copy().reset_index(drop=True)
    baseline_frame = fit_df.groupby("vintage")[features].mean()
    baseline_by_vintage = {
        vintage: {feature: float(value) for feature, value in row.items()}
        for vintage, row in baseline_frame.iterrows()
    }
    global_baseline = {feature: float(fit_df[feature].mean()) for feature in features}

    residual_df = fit_df[["vintage"] + features].copy()
    for feature in features:
        residual_df[f"{feature}_resid"] = residual_df[feature] - residual_df["vintage"].map(
            lambda v: baseline_by_vintage.get(v, global_baseline)[feature]
        )
    residual_cols = [f"{feature}_resid" for feature in features]
    residuals_by_vintage = {
        vintage: sub[residual_cols].reset_index(drop=True)
        for vintage, sub in residual_df.groupby("vintage", dropna=False)
    }
    global_residuals = residual_df[residual_cols].reset_index(drop=True)

    renovated = fit_df["reno_signal"] > 1.0
    baseline_stock = fit_df["reno_signal"] <= 0.0
    global_lift = DEFAULT_RENOVATION_LIFT
    if renovated.sum() >= 3 and baseline_stock.sum() >= 3:
        denom = float(fit_df.loc[baseline_stock, "R"].median())
        numer = float(fit_df.loc[renovated, "R"].median())
        if np.isfinite(denom) and denom > 0 and np.isfinite(numer):
            global_lift = numer / denom
    global_lift = float(np.clip(global_lift, 1.1, 4.0))

    renovation_lift_by_vintage: Dict[str, float] = {}
    for vintage, sub in fit_df.groupby("vintage", dropna=False):
        reno_mask = sub["reno_signal"] > 1.0
        base_mask = sub["reno_signal"] <= 0.0
        lift = global_lift
        if reno_mask.sum() >= 2 and base_mask.sum() >= 2:
            denom = float(sub.loc[base_mask, "R"].median())
            numer = float(sub.loc[reno_mask, "R"].median())
            if np.isfinite(denom) and denom > 0 and np.isfinite(numer):
                lift = numer / denom
        renovation_lift_by_vintage[str(vintage)] = float(np.clip(lift, 1.1, 4.0))

    train_energy_stats = summarize_series(fit_df["E_std_annual_per_m2"])
    train_energy_stats["Qint_std"] = float(pd.to_numeric(fit_df["Qint"], errors="coerce").std(ddof=0))

    energy_stats_by_vintage = to_stats_dict(fit_df, ["vintage"], "E_std_annual_per_m2")
    energy_stats_by_label = to_stats_dict(fit_df, ["label_clean"], "E_std_annual_per_m2")
    energy_stats_by_vintage_label = to_stats_dict(fit_df, ["vintage", "label_clean"], "E_std_annual_per_m2")
    r_clip_by_vintage = to_stats_dict(fit_df, ["vintage"], "R")
    r_clip_by_vintage_label = to_stats_dict(fit_df, ["vintage", "label_clean"], "R")

    return ThermalParameterStore(
        baseline_by_vintage=baseline_by_vintage,
        global_baseline=global_baseline,
        residuals_by_vintage=residuals_by_vintage,
        global_residuals=global_residuals,
        renovation_lift_by_vintage=renovation_lift_by_vintage,
        global_renovation_lift=global_lift,
        train_energy_stats=train_energy_stats,
        energy_stats_by_vintage=energy_stats_by_vintage,
        energy_stats_by_label=energy_stats_by_label,
        energy_stats_by_vintage_label=energy_stats_by_vintage_label,
        r_clip_by_vintage=r_clip_by_vintage,
        r_clip_by_vintage_label=r_clip_by_vintage_label,
        qint_std=float(train_energy_stats["Qint_std"]),
        capacity_mode=capacity_mode,
        train_df=fit_df,
    )


fit_group_generator = fit_thermal_parameter_store


def lookup_baseline(gen: ThermalParameterStore, vintage: str) -> Dict[str, float]:
    return dict(gen.baseline_by_vintage.get(vintage, gen.global_baseline))


def lookup_energy_stat_record(
    gen: ThermalParameterStore, vintage: str, label: Optional[str]
) -> Dict[str, float]:
    label_clean = clean_energy_label(label)
    if label_clean is not None:
        key_vl = (vintage, label_clean)
        if key_vl in gen.energy_stats_by_vintage_label:
            return {**gen.energy_stats_by_vintage_label[key_vl], "source": "vintage_label"}
        key_l = (label_clean,)
        if key_l in gen.energy_stats_by_label:
            return {**gen.energy_stats_by_label[key_l], "source": "label"}
    key_v = (vintage,)
    if key_v in gen.energy_stats_by_vintage:
        return {**gen.energy_stats_by_vintage[key_v], "source": "vintage"}
    return {**gen.train_energy_stats, "source": "global"}


def lookup_r_clip_bounds(
    gen: ThermalParameterStore, vintage: str, label: Optional[str]
) -> Tuple[float, float]:
    label_clean = clean_energy_label(label)
    stats = None
    if label_clean is not None:
        stats = gen.r_clip_by_vintage_label.get((vintage, label_clean))
    if stats is None:
        stats = gen.r_clip_by_vintage.get((vintage,))
    if stats is None:
        stats = summarize_series(gen.train_df["R"])
    lo = max(float(stats["q01"]) * 0.75, 1e-4)
    hi = max(float(stats["q99"]) * 1.35, lo * 1.05)
    return float(lo), float(hi)


def sample_residual_row(
    gen: ThermalParameterStore, vintage: str, rng: np.random.Generator
) -> Dict[str, float]:
    residual_df = gen.residuals_by_vintage.get(vintage, pd.DataFrame())
    if len(residual_df) < 2:
        residual_df = gen.global_residuals
    sampled = residual_df.iloc[int(rng.integers(0, len(residual_df)))]
    return {str(col): float(sampled[col]) for col in residual_df.columns}


def sample_shared_scenario_factors(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "z_renovation_quality": float(rng.normal(0.0, 0.20)),
        "z_solar_exposure":     float(rng.normal(0.0, 0.12)),
        "z_occupancy":          float(rng.normal(0.0, 0.08)),
    }


sample_community_latent = sample_shared_scenario_factors


# ---------------------------------------------------------------------------
# V3 legacy parametric sampler
# ---------------------------------------------------------------------------

@dataclass
class ParametricSamplerModel:
    """Fitted V3 legacy parametric sampler model."""
    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    feature_cols: List[str]
    energy_stats_g: Dict[str, float]
    energy_stats_v: Optional[pd.DataFrame] = None
    energy_stats_l: Optional[pd.DataFrame] = None
    energy_stats_vl: Optional[pd.DataFrame] = None
    train_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    capacity_mode: str = "per_area"


ParamGeneratorModel = ParametricSamplerModel


def fit_parametric_sampler(
    df_in: pd.DataFrame,
    capacity_mode: str = "per_area",
) -> ParametricSamplerModel:
    work = prepare_building_features(df_in)
    features = GROUP_FEATURES_CPA if capacity_mode == "per_area" else ["R", "C", "A", "Qint"]
    required = [c for c in features if c in work.columns]
    fit_df = work.dropna(subset=required + ["E_std_annual_per_m2"]).copy().reset_index(drop=True)

    X = fit_df[required].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    cov = np.cov(X.T, ddof=1) if X.shape[1] > 1 else np.array([[float(np.var(X[:, 0], ddof=1))]])
    cov_inv = np.linalg.pinv(cov)

    energy_stats_g = dict(summarize_series(fit_df["E_std_annual_per_m2"]))
    energy_stats_v = (
        fit_df.groupby("vintage")["E_std_annual_per_m2"]
        .agg(["count", "mean", "std", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)])
        .rename(columns={"count": "n", "<lambda_0>": "q10", "<lambda_1>": "q90"})
        .reset_index()
    ) if "vintage" in fit_df.columns else None

    return ParametricSamplerModel(
        mean=mean,
        cov=cov,
        cov_inv=cov_inv,
        feature_cols=required,
        energy_stats_g=energy_stats_g,
        energy_stats_v=energy_stats_v,
        train_df=fit_df,
        capacity_mode=capacity_mode,
    )


fit_parametric_generator = fit_parametric_sampler


def sample_candidate_latents(
    model: ParametricSamplerModel,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    return rng.multivariate_normal(model.mean, model.cov, size=n)


def latent_to_params(
    latent: np.ndarray,
    area: float,
    model: ParametricSamplerModel,
) -> Dict[str, float]:
    cols = model.feature_cols
    result = {col: float(val) for col, val in zip(cols, latent)}
    result["R"] = max(result.get("R", 1e-4), 1e-4)
    if model.capacity_mode == "per_area":
        cpa = max(result.get("C_per_area", 1e-6), 1e-6)
        result["C_per_area"] = cpa
        result["C"] = cpa * max(area, 1.0)
    else:
        result["C"] = max(result.get("C", 1e-6), 1e-6)
        result["C_per_area"] = result["C"] / max(area, 1.0)
    result["A"] = max(result.get("A", 1e-6), 1e-6)
    result["Qint"] = max(result.get("Qint", 0.0), 0.0)
    return result


def sample_single_building_parametric(
    row: pd.Series,
    model: ParametricSamplerModel,
    weather_by_year: Dict[int, pd.DataFrame],
    n_candidates: int = 200,
    top_k: int = 1,
    rng: Optional[np.random.Generator] = None,
    profile_year: int = 2024,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    area = float(row.get("Area", 1.0))
    label = clean_energy_label(row.get("EnergyLabel"))
    year = float(row.get("year", 1990))
    latents = sample_candidate_latents(model, n_candidates, rng)
    scored = []
    for latent in latents:
        params = latent_to_params(latent, area, model)
        residual = latent - model.mean
        try:
            energy = estimate_std_energy(
                R=params["R"], C=params["C"], A=params["A"],
                Qint=params["Qint"], area=area,
                weather_by_year=weather_by_year, profile_year=profile_year,
            )
        except Exception:
            energy = np.nan
        penalty = official_label_penalty(energy, label) if label else 0.0
        maha = mahalanobis_penalty(residual, model)
        score = penalty + 0.35 * maha
        scored.append({
            **params,
            "Area": area,
            "year": year,
            "EnergyLabel": label,
            "E_proxy": energy,
            "score": score,
        })

    return pd.DataFrame(scored).sort_values("score").head(top_k).reset_index(drop=True)


generate_rcaq_for_building = sample_single_building_parametric


def synthesize_buildings_parametric_batch(
    conditions_df: pd.DataFrame,
    model: ParametricSamplerModel,
    weather_by_year: Dict[int, pd.DataFrame],
    n_candidates: int = 200,
    top_k: int = 1,
    random_state: int = RNG_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    frames = []
    for _, row in conditions_df.iterrows():
        result = sample_single_building_parametric(
            row, model, weather_by_year, n_candidates, top_k, rng
        )
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


generate_buildings_batch = synthesize_buildings_parametric_batch


def evaluate_generator_cv(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: int = 8,
    n_candidates: int = 200,
    random_state: int = RNG_SEED,
) -> pd.DataFrame:
    work = prepare_building_features(df_in).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    rows = []
    for repeat in range(n_repeats):
        test_idx = int(rng.integers(0, len(work)))
        train_df = work.drop(index=test_idx).reset_index(drop=True)
        test_row = work.iloc[test_idx]
        model = fit_parametric_sampler(train_df)
        result = sample_single_building_parametric(
            test_row, model, weather_by_year, n_candidates, top_k=1, rng=rng
        )
        if len(result) == 0:
            continue
        best = result.iloc[0]
        rows.append({
            "repeat": repeat + 1,
            "wasserstein_E": wasserstein_1d(
                np.array([float(test_row["E_std_annual_per_m2"])]),
                np.array([float(best.get("E_proxy", np.nan))]),
            ),
            "R_error": float(abs(best["R"] - float(test_row["R"]))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V4 bootstrap residual synthesis strategy
# ---------------------------------------------------------------------------

def sample_single_building_bootstrap(
    row: pd.Series,
    gen: ThermalParameterStore,
    z_community: Dict[str, float],
    rng: np.random.Generator,
    use_community_latent: bool = True,
    use_renovation_signal: bool = True,
) -> Dict[str, float]:
    year = float(row["year"])
    area = float(row["Area"])
    label = clean_energy_label(row.get("EnergyLabel"))
    vintage = vintage_from_year(year)

    baseline = lookup_baseline(gen, vintage)
    residual = sample_residual_row(gen=gen, vintage=vintage, rng=rng)
    feature_key = "C_per_area" if gen.capacity_mode == "per_area" else "C"

    R_raw = float(baseline["R"] + residual["R_resid"])
    C_base = float(baseline[feature_key] + residual[f"{feature_key}_resid"])
    A_raw = float(baseline["A"] + residual["A_resid"])
    Q_raw = float(baseline["Qint"] + residual["Qint_resid"])

    vintage_expected = float(VINTAGE_EXPECTED_LABEL_SCORE[vintage])
    label_score = float(LABEL_SCORE.get(label, vintage_expected))
    reno_signal = label_score - vintage_expected
    p_reno = sigmoid(1.2 * ((reno_signal if use_renovation_signal else 0.0) - 1.0))
    lift = float(gen.renovation_lift_by_vintage.get(vintage, gen.global_renovation_lift))

    z_rq = z_community["z_renovation_quality"] if use_community_latent else 0.0
    z_se = z_community["z_solar_exposure"] if use_community_latent else 0.0
    z_oc = z_community["z_occupancy"] if use_community_latent else 0.0

    R = R_raw * ((1.0 - p_reno) + p_reno * lift * float(np.exp(z_rq)))
    r_lo, r_hi = lookup_r_clip_bounds(gen=gen, vintage=vintage, label=label)
    R = float(np.clip(max(R, 1e-4), r_lo, r_hi))

    if gen.capacity_mode == "per_area":
        C_per_area = float(max(C_base, 1e-6))
        C = float(C_per_area * area)
    else:
        C = float(max(C_base, 1e-6))
        C_per_area = float(C / max(area, 1e-6))

    A = float(max(A_raw, 1e-6) * np.exp(z_se))
    Qint = float(max(Q_raw + z_oc * gen.qint_std * 0.1, 0.0))

    return {
        "year": year,
        "Area": area,
        "EnergyLabel": label,
        "vintage": vintage,
        "label_clean": label,
        "label_score": label_score,
        "reno_signal": reno_signal,
        "p_reno": p_reno,
        "R_raw": R_raw,
        "R": R,
        "C_per_area": C_per_area,
        "C": C,
        "A": A,
        "Qint": Qint,
    }


generate_single_building = sample_single_building_bootstrap


def estimate_conditional_expected_energy(
    community_conditions: pd.DataFrame, gen: ThermalParameterStore
) -> float:
    enriched = prepare_building_features(community_conditions)
    expected = [
        float(lookup_energy_stat_record(gen=gen, vintage=str(r.vintage), label=r.label_clean)["mean"])
        for r in enriched.itertuples(index=False)
    ]
    return float(np.mean(expected)) if expected else float(gen.train_energy_stats["mean"])


def synthesize_community_bootstrap(
    community_conditions: pd.DataFrame,
    gen: ThermalParameterStore,
    weather_by_year: Dict[int, pd.DataFrame],
    n_population_candidates: int = 60,
    top_k_proxy: int = 8,
    top_k_full: int = 3,
    profile_year: int = PROFILE_YEAR,
    attach_profiles: bool = False,
    random_state: Optional[int] = None,
    use_community_latent: bool = True,
    use_renovation_signal: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    inputs = prepare_building_features(community_conditions)
    rng = np.random.default_rng(RNG_SEED if random_state is None else random_state)

    candidate_buildings: Dict[str, pd.DataFrame] = {}
    proxy_rows = []
    for candidate_idx in range(n_population_candidates):
        population_id = f"p{candidate_idx + 1:03d}"
        z_community = (
            sample_shared_scenario_factors(rng=rng)
            if use_community_latent
            else {"z_renovation_quality": 0.0, "z_solar_exposure": 0.0, "z_occupancy": 0.0}
        )

        rows = []
        for building_idx, row in inputs.iterrows():
            generated = sample_single_building_bootstrap(
                row=row,
                gen=gen,
                z_community=z_community,
                rng=rng,
                use_community_latent=use_community_latent,
                use_renovation_signal=use_renovation_signal,
            )
            generated["building_id"] = str(row.get("building_id", f"b{building_idx + 1:02d}"))
            generated["population_id"] = population_id
            generated["E_proxy_annual_per_m2"] = estimate_std_energy(
                R=generated["R"],
                C=generated["C"],
                A=generated["A"],
                Qint=generated["Qint"],
                area=generated["Area"],
                weather_by_year=weather_by_year,
                profile_year=profile_year,
            )
            rows.append(generated)

        population_df = pd.DataFrame(rows)
        proxy_score = score_population_proxy_bootstrap(
            population_df=population_df, community_conditions=inputs, gen=gen
        )
        candidate_buildings[population_id] = population_df
        proxy_rows.append({
            "population_id": population_id,
            "stage": "proxy",
            **z_community,
            **proxy_score,
            "population_size": len(population_df),
        })

    proxy_df = pd.DataFrame(proxy_rows).sort_values("score").reset_index(drop=True)
    shortlisted = proxy_df.head(min(top_k_proxy, len(proxy_df)))["population_id"].tolist()

    profiles: Dict[str, pd.DataFrame] = {}
    full_rows = []
    selected_frames = []
    for population_id in shortlisted:
        population_df = candidate_buildings[population_id].copy()
        full_metrics = []
        for building in population_df.itertuples(index=False):
            summary = simulate_standard_year_summary(
                R=float(building.R),
                C=float(building.C),
                A=float(building.A),
                Qint=float(building.Qint),
                area=float(building.Area),
                weather_by_year=weather_by_year,
            )
            full_metrics.append(summary)

            if attach_profiles:
                profile_key = f"{population_id}_{building.building_id}"
                profiles[profile_key] = simulate_heating_profile(
                    R=float(building.R),
                    C=float(building.C),
                    A=float(building.A),
                    Qint=float(building.Qint),
                    area=float(building.Area),
                    weather_df=weather_by_year[profile_year],
                )

        full_metrics_df = pd.DataFrame(full_metrics)
        population_df["E_std_full_per_m2"] = full_metrics_df["annual_kwh_per_m2_mean"].to_numpy(dtype=float)
        population_df["E_std_full_kwh"] = full_metrics_df["annual_kwh_mean"].to_numpy(dtype=float)
        population_df["summer_kwh_share"] = full_metrics_df["summer_kwh_share_mean"].to_numpy(dtype=float)
        population_df["peak_heat_kw"] = full_metrics_df["peak_kw_mean"].to_numpy(dtype=float)

        full_score = score_population_full_bootstrap(
            population_df=population_df, community_conditions=inputs, gen=gen
        )
        full_rows.append({
            "population_id": population_id,
            "stage": "full",
            **full_score,
            "total_kwh": float(population_df["E_std_full_kwh"].sum()),
            "mean_R": float(population_df["R"].mean()),
            "mean_C_per_area": float(population_df["C_per_area"].mean()),
            "population_size": len(population_df),
        })
        candidate_buildings[population_id] = population_df

    full_df = pd.DataFrame(full_rows).sort_values("score").reset_index(drop=True)
    keep_ids = full_df.head(min(top_k_full, len(full_df)))["population_id"].tolist()
    for rank, population_id in enumerate(keep_ids, start=1):
        selected_frames.append(candidate_buildings[population_id].assign(selected_rank=rank))

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    score_df = proxy_df.merge(full_df, on="population_id", how="left", suffixes=("_proxy", "_full"))
    score_df = score_df.sort_values(["score_full", "score_proxy"], na_position="last").reset_index(drop=True)
    return selected_df, score_df, profiles


generate_community_rcaq = synthesize_community_bootstrap


def evaluate_group_generator_cv(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: int = 8,
    group_size: int = 8,
    n_population_candidates: int = 36,
    top_k_proxy: int = 6,
    top_k_full: int = 1,
    random_state: int = RNG_SEED,
    use_community_latent: bool = True,
    use_renovation_signal: bool = True,
    capacity_mode: str = "per_area",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_df = prepare_building_features(df_in)
    if len(model_df) <= group_size:
        raise ValueError("group_size must be smaller than the number of modelling rows.")

    rng = np.random.default_rng(random_state)
    repeat_rows = []

    for repeat in range(n_repeats):
        test_idx = np.sort(rng.choice(len(model_df), size=group_size, replace=False))
        train_df = model_df.drop(index=test_idx).reset_index(drop=True)
        test_df = model_df.iloc[test_idx].reset_index(drop=True)
        gen = fit_thermal_parameter_store(train_df, capacity_mode=capacity_mode)

        generated_df, _, _ = synthesize_community_bootstrap(
            community_conditions=test_df[["year", "Area", "EnergyLabel"]].copy(),
            gen=gen,
            weather_by_year=weather_by_year,
            n_population_candidates=n_population_candidates,
            top_k_proxy=top_k_proxy,
            top_k_full=top_k_full,
            attach_profiles=False,
            random_state=random_state + repeat,
            use_community_latent=use_community_latent,
            use_renovation_signal=use_renovation_signal,
        )
        best_df = generated_df[generated_df["selected_rank"] == 1].copy().reset_index(drop=True)

        true_e = test_df["E_std_annual_per_m2"].to_numpy(dtype=float)
        gen_e = best_df["E_std_full_per_m2"].to_numpy(dtype=float)
        true_r = test_df["R"].to_numpy(dtype=float)
        gen_r = best_df["R"].to_numpy(dtype=float)
        true_cpa = test_df["C_per_area"].to_numpy(dtype=float)
        gen_cpa = best_df["C_per_area"].to_numpy(dtype=float)
        true_total_kwh = float((test_df["E_std_annual_per_m2"] * test_df["Area"]).sum())
        gen_total_kwh = float(best_df["E_std_full_kwh"].sum())

        order_score, order_ok = label_ordering_score(best_df, energy_col="E_std_full_per_m2")
        reno_mask = best_df["reno_signal"] > 1.0
        base_mask = best_df["reno_signal"] <= 0.0
        renovation_ok = (
            float(best_df.loc[reno_mask, "R"].mean()) > float(best_df.loc[base_mask, "R"].mean())
            if reno_mask.any() and base_mask.any()
            else True
        )

        repeat_rows.append({
            "repeat": repeat + 1,
            "group_size": int(group_size),
            "abs_mean_E_error": float(abs(np.mean(gen_e) - np.mean(true_e))),
            "median_E_error": float(abs(np.median(gen_e) - np.median(true_e))),
            "total_kWh_error_pct": float(100.0 * abs(gen_total_kwh - true_total_kwh) / max(true_total_kwh, 1e-9)),
            "wasserstein_E": wasserstein_1d(true_e, gen_e),
            "wasserstein_R": wasserstein_1d(true_r, gen_r),
            "wasserstein_Cpa": wasserstein_1d(true_cpa, gen_cpa),
            "label_ordering_score": order_score,
            "label_ordering_ok": bool(order_ok),
            "renovation_R_ok": bool(renovation_ok),
        })

    repeat_df = pd.DataFrame(repeat_rows)
    summary_df = pd.DataFrame([{
        "abs_mean_E_error":    float(repeat_df["abs_mean_E_error"].mean()),
        "median_E_error":      float(repeat_df["median_E_error"].mean()),
        "total_kWh_error_pct": float(repeat_df["total_kWh_error_pct"].mean()),
        "wasserstein_E":       float(repeat_df["wasserstein_E"].mean()),
        "wasserstein_R":       float(repeat_df["wasserstein_R"].mean()),
        "wasserstein_Cpa":     float(repeat_df["wasserstein_Cpa"].mean()),
        "label_ordering_ok":   float(repeat_df["label_ordering_ok"].mean()),
        "renovation_R_ok":     float(repeat_df["renovation_R_ok"].mean()),
    }])
    return repeat_df, summary_df


def evaluate_group_generator_ablation(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: int = 6,
    group_size: int = 8,
    n_population_candidates: int = 28,
    top_k_proxy: int = 5,
    top_k_full: int = 1,
    random_state: int = RNG_SEED,
) -> pd.DataFrame:
    configs = [
        {"ablation": "full",                        "use_community_latent": True,  "use_renovation_signal": True,  "capacity_mode": "per_area"},
        {"ablation": "without_z_community",         "use_community_latent": False, "use_renovation_signal": True,  "capacity_mode": "per_area"},
        {"ablation": "without_renovation_signal",   "use_community_latent": True,  "use_renovation_signal": False, "capacity_mode": "per_area"},
        {"ablation": "raw_C_instead_of_C_per_area", "use_community_latent": True,  "use_renovation_signal": True,  "capacity_mode": "raw"},
    ]
    rows = []
    for idx, cfg in enumerate(configs):
        _, summary_df = evaluate_group_generator_cv(
            df_in=df_in,
            weather_by_year=weather_by_year,
            n_repeats=n_repeats,
            group_size=group_size,
            n_population_candidates=n_population_candidates,
            top_k_proxy=top_k_proxy,
            top_k_full=top_k_full,
            random_state=random_state + idx * 100,
            use_community_latent=cfg["use_community_latent"],
            use_renovation_signal=cfg["use_renovation_signal"],
            capacity_mode=cfg["capacity_mode"],
        )
        rows.append({"ablation": cfg["ablation"], **summary_df.iloc[0].to_dict()})

    out = pd.DataFrame(rows)
    baseline = out.loc[out["ablation"] == "full"].iloc[0]
    for metric in ["abs_mean_E_error", "median_E_error", "total_kWh_error_pct",
                   "wasserstein_E", "wasserstein_R", "wasserstein_Cpa"]:
        out[f"delta_{metric}_vs_full"] = out[metric] - float(baseline[metric])
    return out


# ---------------------------------------------------------------------------
# V5 smoothed residual synthesis strategy
# ---------------------------------------------------------------------------

def compute_residual_statistics(
    gen: ThermalParameterStore,
    min_samples: int = MIN_RESID_SAMPLES,
) -> Tuple[Dict[str, float], float, Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute per-vintage residual statistics for smoothed residual synthesis.
    """
    global_r_resid_std = float(gen.global_residuals["R_resid"].std(ddof=1))
    global_resid_feat_stds: Dict[str, float] = {
        col: float(gen.global_residuals[col].std(ddof=1))
        for col in gen.global_residuals.columns
    }

    r_resid_std_by_vintage: Dict[str, float] = {}
    resid_feat_stds_by_vint: Dict[str, Dict[str, float]] = {}

    for vintage, resid_df in gen.residuals_by_vintage.items():
        v = str(vintage)
        if len(resid_df) >= min_samples:
            r_resid_std_by_vintage[v] = float(resid_df["R_resid"].std(ddof=1))
            resid_feat_stds_by_vint[v] = {
                col: float(resid_df[col].std(ddof=1))
                for col in resid_df.columns
            }

    return r_resid_std_by_vintage, global_r_resid_std, resid_feat_stds_by_vint, global_resid_feat_stds


compute_resid_stats = compute_residual_statistics


def _smooth_residual(
    residual: Dict[str, float],
    feat_stds: Dict[str, float],
    scale: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    return {
        col: val + float(rng.normal(0.0, scale * max(feat_stds.get(col, 1e-9), 1e-9)))
        for col, val in residual.items()
    }


def sample_single_building_smoothed_residual(
    row: pd.Series,
    gen: ThermalParameterStore,
    r_resid_std_by_vintage: Dict[str, float],
    global_r_resid_std: float,
    resid_feat_stds_by_vint: Dict[str, Dict[str, float]],
    global_resid_feat_stds: Dict[str, float],
    z_community: Dict[str, float],
    rng: np.random.Generator,
    use_z_community: bool = False,
    alpha_reno: float = 0.25,
    residual_smoothing: bool = True,
    smoothing_scale: float = 0.02,
    use_hard_lift: bool = False,
) -> Dict[str, float]:
    """
    V5 synthesis of one building using soft additive renovation adjustment.
    """
    year = float(row["year"])
    area = float(row["Area"])
    label = clean_energy_label(row.get("EnergyLabel"))
    vintage = vintage_from_year(year)
    feature_key = "C_per_area" if gen.capacity_mode == "per_area" else "C"

    baseline = lookup_baseline(gen, vintage)

    raw_resid = sample_residual_row(gen=gen, vintage=vintage, rng=rng)
    if residual_smoothing:
        feat_stds = resid_feat_stds_by_vint.get(vintage, global_resid_feat_stds)
        residual = _smooth_residual(raw_resid, feat_stds, smoothing_scale, rng)
    else:
        residual = raw_resid

    R_resid = float(residual["R_resid"])
    C_resid = float(residual[f"{feature_key}_resid"])
    A_resid = float(residual["A_resid"])
    Q_resid = float(residual["Qint_resid"])

    R_base = float(baseline["R"])
    R_raw = R_base + R_resid

    vintage_expected = float(VINTAGE_EXPECTED_LABEL_SCORE[vintage])
    label_score_val = float(LABEL_SCORE.get(label, vintage_expected))
    reno_signal = label_score_val - vintage_expected

    if use_hard_lift:
        p_reno = sigmoid(1.2 * (reno_signal - 1.0))
        lift = float(gen.renovation_lift_by_vintage.get(vintage, gen.global_renovation_lift))
        R = R_raw * ((1.0 - p_reno) + p_reno * lift)
    else:
        r_std = r_resid_std_by_vintage.get(vintage, global_r_resid_std)
        reno_adjustment = float(np.tanh(reno_signal * 0.4)) * r_std * alpha_reno
        z_rq = float(z_community["z_renovation_quality"]) if use_z_community else 0.0
        R = R_raw + reno_adjustment + z_rq * r_std * 0.5

    r_lo, r_hi = lookup_r_clip_bounds(gen=gen, vintage=vintage, label=label)
    R = float(np.clip(max(R, 1e-4), r_lo, r_hi))

    if gen.capacity_mode == "per_area":
        C_per_area = float(max(float(baseline[feature_key]) + C_resid, 1e-6))
        C = float(C_per_area * area)
    else:
        C = float(max(float(baseline[feature_key]) + C_resid, 1e-6))
        C_per_area = float(C / max(area, 1e-6))

    A_raw = float(baseline["A"] + A_resid)
    z_se = float(z_community["z_solar_exposure"]) if use_z_community else 0.0
    A = float(max(A_raw, 1e-6) * float(np.exp(z_se)))

    Q_raw = float(baseline["Qint"] + Q_resid)
    z_oc = float(z_community["z_occupancy"]) if use_z_community else 0.0
    Qint = float(max(Q_raw + z_oc * gen.qint_std * 0.1, 0.0))

    return {
        "year":        year,
        "Area":        area,
        "EnergyLabel": label,
        "vintage":     vintage,
        "label_clean": label,
        "label_score": label_score_val,
        "reno_signal": reno_signal,
        "p_reno":      sigmoid(1.2 * (reno_signal - 1.0)),
        "R_raw":       R_raw,
        "R":           R,
        "C_per_area":  C_per_area,
        "C":           C,
        "A":           A,
        "Qint":        Qint,
    }


generate_single_building_v5 = sample_single_building_smoothed_residual


def synthesize_community_smoothed_residual(
    community_conditions: pd.DataFrame,
    gen: ThermalParameterStore,
    weather_by_year: Dict[int, pd.DataFrame],
    r_resid_std_by_vintage: Dict[str, float],
    global_r_resid_std: float,
    resid_feat_stds_by_vint: Dict[str, Dict[str, float]],
    global_resid_feat_stds: Dict[str, float],
    n_population_candidates: int = 60,
    top_k_proxy: int = 8,
    top_k_full: int = 3,
    profile_year: int = PROFILE_YEAR,
    attach_profiles: bool = False,
    random_state: Optional[int] = None,
    use_z_community: bool = False,
    alpha_reno: float = 0.25,
    residual_smoothing: bool = True,
    smoothing_scale: float = 0.02,
    use_hard_lift: bool = False,
    beta_label_E: float = -3.0,
    proxy_simulation_mode: str = "screen",
    typical_weather_scenarios: Optional[Dict[str, Any]] = None,
    typical_weather_cache_path: Optional[str] = None,
    typical_weather_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if proxy_simulation_mode not in {"screen", "typical_weather"}:
        raise ValueError("proxy_simulation_mode must be either 'screen' or 'typical_weather'.")
    typical_config = dict(typical_weather_config or {})
    typical_warmup_repeats = int(typical_config.pop("warmup_repeats", 2))
    if proxy_simulation_mode == "typical_weather" and typical_weather_scenarios is None:
        typical_weather_scenarios = load_or_build_typical_weather_scenarios(
            weather_by_year=weather_by_year,
            cache_path=typical_weather_cache_path,
            **typical_config,
        )
    inputs = prepare_building_features(community_conditions)
    rng = np.random.default_rng(RNG_SEED if random_state is None else random_state)

    candidate_buildings: Dict[str, pd.DataFrame] = {}
    proxy_rows: List[Dict] = []

    for cand_idx in range(n_population_candidates):
        pop_id = f"p{cand_idx + 1:03d}"
        z_community = (
            sample_shared_scenario_factors(rng=rng)
            if use_z_community
            else {"z_renovation_quality": 0.0, "z_solar_exposure": 0.0, "z_occupancy": 0.0}
        )

        rows = []
        for bldg_idx, row in inputs.iterrows():
            gen_bldg = sample_single_building_smoothed_residual(
                row=row,
                gen=gen,
                r_resid_std_by_vintage=r_resid_std_by_vintage,
                global_r_resid_std=global_r_resid_std,
                resid_feat_stds_by_vint=resid_feat_stds_by_vint,
                global_resid_feat_stds=global_resid_feat_stds,
                z_community=z_community,
                rng=rng,
                use_z_community=use_z_community,
                alpha_reno=alpha_reno,
                residual_smoothing=residual_smoothing,
                smoothing_scale=smoothing_scale,
                use_hard_lift=use_hard_lift,
            )
            gen_bldg["building_id"] = str(row.get("building_id", f"b{bldg_idx + 1:02d}"))
            gen_bldg["population_id"] = pop_id
            if proxy_simulation_mode == "typical_weather":
                gen_bldg["E_proxy_annual_per_m2"] = estimate_std_energy_typical(
                    R=gen_bldg["R"],
                    C=gen_bldg["C"],
                    A=gen_bldg["A"],
                    Qint=gen_bldg["Qint"],
                    area=gen_bldg["Area"],
                    typical_weather_scenarios=typical_weather_scenarios,
                    warmup_repeats=typical_warmup_repeats,
                )
            else:
                gen_bldg["E_proxy_annual_per_m2"] = estimate_std_energy(
                    R=gen_bldg["R"],
                    C=gen_bldg["C"],
                    A=gen_bldg["A"],
                    Qint=gen_bldg["Qint"],
                    area=gen_bldg["Area"],
                    weather_by_year=weather_by_year,
                    profile_year=profile_year,
                )
            rows.append(gen_bldg)

        pop_df = pd.DataFrame(rows)
        proxy_score = score_population_proxy_smoothed_residual(pop_df, inputs, gen, beta_label_E)
        candidate_buildings[pop_id] = pop_df
        proxy_rows.append({
            "population_id": pop_id,
            "stage": "proxy",
            **z_community,
            **proxy_score,
            "population_size": len(pop_df),
        })

    proxy_df = pd.DataFrame(proxy_rows).sort_values("score").reset_index(drop=True)
    shortlisted = proxy_df.head(min(top_k_proxy, len(proxy_df)))["population_id"].tolist()

    profiles: Dict[str, pd.DataFrame] = {}
    full_rows: List[Dict] = []
    selected_frames: List[pd.DataFrame] = []

    for pop_id in shortlisted:
        pop_df = candidate_buildings[pop_id].copy()
        full_metrics = []
        for bldg in pop_df.itertuples(index=False):
            summary = simulate_standard_year_summary(
                R=float(bldg.R),
                C=float(bldg.C),
                A=float(bldg.A),
                Qint=float(bldg.Qint),
                area=float(bldg.Area),
                weather_by_year=weather_by_year,
            )
            full_metrics.append(summary)
            if attach_profiles:
                profiles[f"{pop_id}_{bldg.building_id}"] = simulate_heating_profile(
                    R=float(bldg.R),
                    C=float(bldg.C),
                    A=float(bldg.A),
                    Qint=float(bldg.Qint),
                    area=float(bldg.Area),
                    weather_df=weather_by_year[profile_year],
                )

        full_df_m = pd.DataFrame(full_metrics)
        pop_df["E_std_full_per_m2"] = full_df_m["annual_kwh_per_m2_mean"].to_numpy(dtype=float)
        pop_df["E_std_full_kwh"] = full_df_m["annual_kwh_mean"].to_numpy(dtype=float)
        pop_df["summer_kwh_share"] = full_df_m["summer_kwh_share_mean"].to_numpy(dtype=float)
        pop_df["peak_heat_kw"] = full_df_m["peak_kw_mean"].to_numpy(dtype=float)

        full_score = score_population_full_smoothed_residual(pop_df, inputs, gen, beta_label_E)
        full_rows.append({
            "population_id": pop_id,
            "stage": "full",
            **full_score,
            "total_kwh": float(pop_df["E_std_full_kwh"].sum()),
            "mean_R": float(pop_df["R"].mean()),
            "mean_C_per_area": float(pop_df["C_per_area"].mean()),
            "population_size": len(pop_df),
        })
        candidate_buildings[pop_id] = pop_df

    full_df_scores = pd.DataFrame(full_rows).sort_values("score").reset_index(drop=True)
    keep_ids = full_df_scores.head(min(top_k_full, len(full_df_scores)))["population_id"].tolist()
    for rank, pop_id in enumerate(keep_ids, start=1):
        selected_frames.append(candidate_buildings[pop_id].assign(selected_rank=rank))

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    score_df = proxy_df.merge(full_df_scores, on="population_id", how="left", suffixes=("_proxy", "_full"))
    score_df = score_df.sort_values(["score_full", "score_proxy"], na_position="last").reset_index(drop=True)
    return selected_df, score_df, profiles


generate_community_rcaq_v5 = synthesize_community_smoothed_residual


def evaluate_group_generator_cv_v5(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: int = 6,
    group_size: int = 6,
    n_population_candidates: int = 36,
    top_k_proxy: int = 6,
    top_k_full: int = 1,
    random_state: int = RNG_SEED,
    use_z_community: bool = False,
    alpha_reno: float = 0.25,
    residual_smoothing: bool = True,
    smoothing_scale: float = 0.02,
    use_hard_lift: bool = False,
    beta_label_E: float = -3.0,
    capacity_mode: str = "per_area",
    min_resid_samples: int = MIN_RESID_SAMPLES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = prepare_building_features(df_in)
    if len(work) <= group_size:
        raise ValueError("group_size must be smaller than the number of modelling rows.")

    rng = np.random.default_rng(random_state)
    repeat_rows: List[Dict] = []

    for repeat in range(n_repeats):
        test_idx = np.sort(rng.choice(len(work), size=group_size, replace=False))
        train_df = work.drop(index=test_idx).reset_index(drop=True)
        test_df = work.iloc[test_idx].reset_index(drop=True)

        gen_cv = fit_thermal_parameter_store(train_df, capacity_mode=capacity_mode)
        r_std_vint_cv, global_r_std_cv, feat_stds_vint_cv, global_feat_stds_cv = compute_residual_statistics(
            gen_cv, min_samples=min_resid_samples
        )

        gen_df, _, _ = synthesize_community_smoothed_residual(
            community_conditions=test_df[["year", "Area", "EnergyLabel"]].copy(),
            gen=gen_cv,
            weather_by_year=weather_by_year,
            r_resid_std_by_vintage=r_std_vint_cv,
            global_r_resid_std=global_r_std_cv,
            resid_feat_stds_by_vint=feat_stds_vint_cv,
            global_resid_feat_stds=global_feat_stds_cv,
            n_population_candidates=n_population_candidates,
            top_k_proxy=top_k_proxy,
            top_k_full=top_k_full,
            attach_profiles=False,
            random_state=random_state + repeat,
            use_z_community=use_z_community,
            alpha_reno=alpha_reno,
            residual_smoothing=residual_smoothing,
            smoothing_scale=smoothing_scale,
            use_hard_lift=use_hard_lift,
            beta_label_E=beta_label_E,
        )
        best_df = gen_df[gen_df["selected_rank"] == 1].copy().reset_index(drop=True)

        true_e = test_df["E_std_annual_per_m2"].to_numpy(dtype=float)
        gen_e = best_df["E_std_full_per_m2"].to_numpy(dtype=float)
        true_r = test_df["R"].to_numpy(dtype=float)
        gen_r = best_df["R"].to_numpy(dtype=float)
        true_cpa = test_df["C_per_area"].to_numpy(dtype=float)
        gen_cpa = best_df["C_per_area"].to_numpy(dtype=float)
        true_total_kwh = float((test_df["E_std_annual_per_m2"] * test_df["Area"]).sum())
        gen_total_kwh = float(best_df["E_std_full_kwh"].sum())

        lm = label_ordering_metrics_v5(best_df, energy_col="E_std_full_per_m2")
        reno_mask = best_df["reno_signal"] > 1.0
        base_mask = best_df["reno_signal"] <= 0.0
        renovation_ok = (
            float(best_df.loc[reno_mask, "R"].mean()) > float(best_df.loc[base_mask, "R"].mean())
            if reno_mask.any() and base_mask.any()
            else True
        )

        repeat_rows.append({
            "repeat": repeat + 1,
            "group_size": int(group_size),
            "abs_mean_E_error": float(abs(np.mean(gen_e) - np.mean(true_e))),
            "median_E_error": float(abs(np.median(gen_e) - np.median(true_e))),
            "total_kWh_error_pct": float(100.0 * abs(gen_total_kwh - true_total_kwh) / max(true_total_kwh, 1e-9)),
            "wasserstein_E": wasserstein_1d(true_e, gen_e),
            "wasserstein_R": wasserstein_1d(true_r, gen_r),
            "wasserstein_Cpa": wasserstein_1d(true_cpa, gen_cpa),
            **lm,
            "renovation_R_ok": bool(renovation_ok),
        })

    repeat_df = pd.DataFrame(repeat_rows)
    agg_cols = [
        "abs_mean_E_error", "median_E_error", "total_kWh_error_pct",
        "wasserstein_E", "wasserstein_R", "wasserstein_Cpa",
        "label_ordering_ok", "label_violation_count", "renovation_R_ok",
    ]
    summary: Dict[str, float] = {col: float(repeat_df[col].mean()) for col in agg_cols}
    if "label_spearman_corr" in repeat_df.columns:
        valid_corr = repeat_df["label_spearman_corr"].dropna()
        summary["label_spearman_corr"] = float(valid_corr.mean()) if len(valid_corr) > 0 else np.nan
    summary_df = pd.DataFrame([summary])
    return repeat_df, summary_df


_ABLATION_CONFIGS_V5 = [
    ("hard_lift",             True,  0.25, False, "per_area"),
    ("soft_reno_adj",         False, 0.25, False, "per_area"),
    ("no_label_individual",   False, 0.0,  False, "per_area"),
    ("with_z",                False, 0.25, True,  "per_area"),
    ("raw_C",                 False, 0.25, False, "raw"),
]


def evaluate_group_generator_ablation_v5(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    n_repeats: int = 4,
    group_size: int = 6,
    n_population_candidates: int = 24,
    top_k_proxy: int = 5,
    top_k_full: int = 1,
    random_state: int = RNG_SEED,
    min_resid_samples: int = MIN_RESID_SAMPLES,
    beta_label_E: float = -3.0,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for idx, (name, hard_lift, alpha, use_z, cap_mode) in enumerate(_ABLATION_CONFIGS_V5):
        _, summary_df = evaluate_group_generator_cv_v5(
            df_in=df_in,
            weather_by_year=weather_by_year,
            n_repeats=n_repeats,
            group_size=group_size,
            n_population_candidates=n_population_candidates,
            top_k_proxy=top_k_proxy,
            top_k_full=top_k_full,
            random_state=random_state + idx * 100,
            use_z_community=use_z,
            alpha_reno=alpha,
            use_hard_lift=hard_lift,
            beta_label_E=beta_label_E,
            capacity_mode=cap_mode,
            min_resid_samples=min_resid_samples,
        )
        rows.append({"ablation": name, **summary_df.iloc[0].to_dict()})

    out = pd.DataFrame(rows)
    baseline = out.loc[out["ablation"] == "soft_reno_adj"].iloc[0]
    for metric in ["abs_mean_E_error", "total_kWh_error_pct",
                   "wasserstein_E", "wasserstein_R", "wasserstein_Cpa"]:
        out[f"delta_{metric}_vs_soft"] = out[metric] - float(baseline[metric])
    return out


# ---------------------------------------------------------------------------
# Unified strategy API
# ---------------------------------------------------------------------------

def synthesize_community_rcaq(
    community_conditions: pd.DataFrame,
    gen: Any,
    weather_by_year: Dict[int, pd.DataFrame],
    strategy: str = "smoothed_residual",
    *,
    r_resid_std_by_vintage: Optional[Dict[str, float]] = None,
    global_r_resid_std: Optional[float] = None,
    resid_feat_stds_by_vint: Optional[Dict[str, Dict[str, float]]] = None,
    global_resid_feat_stds: Optional[Dict[str, float]] = None,
    min_resid_samples: int = MIN_RESID_SAMPLES,
    n_population_candidates: int = 60,
    top_k_proxy: int = 8,
    top_k_full: int = 3,
    profile_year: int = PROFILE_YEAR,
    attach_profiles: bool = False,
    random_state: Optional[int] = None,
    use_community_latent: bool = True,
    use_renovation_signal: bool = True,
    use_z_community: bool = False,
    alpha_reno: float = 0.25,
    residual_smoothing: bool = True,
    smoothing_scale: float = 0.02,
    use_hard_lift: bool = False,
    beta_label_E: float = -3.0,
    n_candidates: int = 200,
    proxy_simulation_mode: str = "screen",
    typical_weather_scenarios: Optional[Dict[str, Any]] = None,
    typical_weather_cache_path: Optional[str] = None,
    typical_weather_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Synthesize a community using one existing sampling strategy."""
    if strategy == "bootstrap":
        return synthesize_community_bootstrap(
            community_conditions=community_conditions,
            gen=gen,
            weather_by_year=weather_by_year,
            n_population_candidates=n_population_candidates,
            top_k_proxy=top_k_proxy,
            top_k_full=top_k_full,
            profile_year=profile_year,
            attach_profiles=attach_profiles,
            random_state=random_state,
            use_community_latent=use_community_latent,
            use_renovation_signal=use_renovation_signal,
        )

    if strategy == "smoothed_residual":
        stats = (
            r_resid_std_by_vintage,
            global_r_resid_std,
            resid_feat_stds_by_vint,
            global_resid_feat_stds,
        )
        if any(value is None for value in stats):
            stats = compute_residual_statistics(gen, min_samples=min_resid_samples)
        return synthesize_community_smoothed_residual(
            community_conditions=community_conditions,
            gen=gen,
            weather_by_year=weather_by_year,
            r_resid_std_by_vintage=stats[0],
            global_r_resid_std=stats[1],
            resid_feat_stds_by_vint=stats[2],
            global_resid_feat_stds=stats[3],
            n_population_candidates=n_population_candidates,
            top_k_proxy=top_k_proxy,
            top_k_full=top_k_full,
            profile_year=profile_year,
            attach_profiles=attach_profiles,
            random_state=random_state,
            use_z_community=use_z_community,
            alpha_reno=alpha_reno,
            residual_smoothing=residual_smoothing,
            smoothing_scale=smoothing_scale,
            use_hard_lift=use_hard_lift,
            beta_label_E=beta_label_E,
            proxy_simulation_mode=proxy_simulation_mode,
            typical_weather_scenarios=typical_weather_scenarios,
            typical_weather_cache_path=typical_weather_cache_path,
            typical_weather_config=typical_weather_config,
        )

    if strategy == "parametric_legacy":
        selected_df = synthesize_buildings_parametric_batch(
            conditions_df=community_conditions,
            model=gen,
            weather_by_year=weather_by_year,
            n_candidates=n_candidates,
            top_k=top_k_full,
            random_state=RNG_SEED if random_state is None else random_state,
        )
        return selected_df, pd.DataFrame(), {}

    raise ValueError(
        "strategy must be one of 'bootstrap', 'smoothed_residual', or 'parametric_legacy'."
    )


def evaluate_case_size_scenarios(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    community_sizes: Tuple[int, ...] = (20, 50, 100),
    n_cases_per_size: int = 3,
    n_population_candidates: int = 36,
    top_k_proxy: int = 6,
    top_k_full: int = 1,
    random_state: int = RNG_SEED,
    use_community_latent: bool = True,
    use_renovation_signal: bool = True,
    capacity_mode: str = "per_area",
    strategy: str = "smoothed_residual",
    use_z_community: bool = False,
    alpha_reno: float = 0.25,
    residual_smoothing: bool = True,
    smoothing_scale: float = 0.02,
    use_hard_lift: bool = False,
    beta_label_E: float = -3.0,
    min_resid_samples: int = MIN_RESID_SAMPLES,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[int, int], Dict[str, pd.DataFrame]]]:
    """
    Stress-test community synthesis for either residual strategy.

    For sizes <= n_observed: leave-group-out holdout.
    For sizes > n_observed: bootstrap resampling of the full dataset.
    """
    if strategy not in {"bootstrap", "smoothed_residual"}:
        raise ValueError("case-size evaluation supports strategy='bootstrap' or 'smoothed_residual'.")

    work = prepare_building_features(df_in)
    rng = np.random.default_rng(random_state)
    case_rows: List[Dict[str, object]] = []
    case_artifacts: Dict[Tuple[int, int], Dict[str, pd.DataFrame]] = {}

    for community_size in community_sizes:
        for case_id in range(1, n_cases_per_size + 1):
            replace = community_size > len(work)
            sampled_idx = np.sort(rng.choice(len(work), size=community_size, replace=replace))

            if replace:
                train_df = work.copy().reset_index(drop=True)
                test_df = work.iloc[sampled_idx].copy().reset_index(drop=True)
            else:
                train_df = work.drop(index=sampled_idx).reset_index(drop=True)
                test_df = work.iloc[sampled_idx].copy().reset_index(drop=True)

            gen = fit_thermal_parameter_store(train_df, capacity_mode=capacity_mode)
            generated_df, score_df, _ = synthesize_community_rcaq(
                community_conditions=test_df[["year", "Area", "EnergyLabel"]].copy(),
                gen=gen,
                weather_by_year=weather_by_year,
                strategy=strategy,
                min_resid_samples=min_resid_samples,
                n_population_candidates=n_population_candidates,
                top_k_proxy=top_k_proxy,
                top_k_full=top_k_full,
                attach_profiles=False,
                random_state=random_state + community_size * 100 + case_id,
                use_community_latent=use_community_latent,
                use_renovation_signal=use_renovation_signal,
                use_z_community=use_z_community,
                alpha_reno=alpha_reno,
                residual_smoothing=residual_smoothing,
                smoothing_scale=smoothing_scale,
                use_hard_lift=use_hard_lift,
                beta_label_E=beta_label_E,
            )
            best_df = generated_df[generated_df["selected_rank"] == 1].copy().reset_index(drop=True)

            true_e = test_df["E_std_annual_per_m2"].to_numpy(dtype=float)
            gen_e = best_df["E_std_full_per_m2"].to_numpy(dtype=float)
            true_r = test_df["R"].to_numpy(dtype=float)
            gen_r = best_df["R"].to_numpy(dtype=float)
            true_c = test_df["C"].to_numpy(dtype=float)
            gen_c = best_df["C"].to_numpy(dtype=float)
            true_cpa = test_df["C_per_area"].to_numpy(dtype=float)
            gen_cpa = best_df["C_per_area"].to_numpy(dtype=float)
            true_a = test_df["A"].to_numpy(dtype=float)
            gen_a = best_df["A"].to_numpy(dtype=float)
            true_q = test_df["Qint"].to_numpy(dtype=float)
            gen_q = best_df["Qint"].to_numpy(dtype=float)
            true_total_kwh = float((test_df["E_std_annual_per_m2"] * test_df["Area"]).sum())
            gen_total_kwh = float(best_df["E_std_full_kwh"].sum())

            if strategy == "smoothed_residual":
                label_metrics = label_ordering_metrics_v5(best_df, energy_col="E_std_full_per_m2")
                order_score = float(label_metrics["label_ordering_score"])
                order_ok = bool(label_metrics["label_ordering_ok"])
            else:
                order_score, order_ok = label_ordering_score(best_df, energy_col="E_std_full_per_m2")
            reno_mask = best_df["reno_signal"] > 1.0
            base_mask = best_df["reno_signal"] <= 0.0
            renovation_ok = (
                float(best_df.loc[reno_mask, "R"].mean()) > float(best_df.loc[base_mask, "R"].mean())
                if reno_mask.any() and base_mask.any()
                else True
            )

            score_row = score_df.iloc[0] if len(score_df) > 0 else pd.Series(dtype=object)
            case_rows.append({
                "community_size": int(community_size),
                "case_id": int(case_id),
                "abs_mean_E_error": float(abs(np.mean(gen_e) - np.mean(true_e))),
                "median_E_error": float(abs(np.median(gen_e) - np.median(true_e))),
                "total_kWh_error_pct": float(100.0 * abs(gen_total_kwh - true_total_kwh) / max(true_total_kwh, 1e-9)),
                "wasserstein_E": wasserstein_1d(true_e, gen_e),
                "wasserstein_R": wasserstein_1d(true_r, gen_r),
                "wasserstein_C": wasserstein_1d(true_c, gen_c),
                "wasserstein_Cpa": wasserstein_1d(true_cpa, gen_cpa),
                "wasserstein_A": wasserstein_1d(true_a, gen_a),
                "wasserstein_Qint": wasserstein_1d(true_q, gen_q),
                "label_ordering_score": float(order_score),
                "label_ordering_ok": bool(order_ok),
                "renovation_R_ok": bool(renovation_ok),
                "score_full": float(score_row.get("score_full", np.nan)),
                "coverage_ratio_full": float(score_row.get("coverage_ratio_full", np.nan)),
                "label_target_penalty_full": float(score_row.get("label_target_penalty_full", np.nan)),
            })

            real_case_df = test_df.copy()
            real_case_df["community_size"] = int(community_size)
            real_case_df["case_id"] = int(case_id)
            real_case_df["source"] = "real"

            gen_case_df = best_df.copy()
            gen_case_df["community_size"] = int(community_size)
            gen_case_df["case_id"] = int(case_id)
            gen_case_df["source"] = "generated"

            score_case_df = score_df.copy()
            score_case_df["community_size"] = int(community_size)
            score_case_df["case_id"] = int(case_id)

            case_artifacts[(int(community_size), int(case_id))] = {
                "real_df": real_case_df,
                "generated_df": gen_case_df,
                "score_df": score_case_df,
            }

    case_detail_df = (
        pd.DataFrame(case_rows)
        .sort_values(["community_size", "case_id"])
        .reset_index(drop=True)
    )
    summary_cols = [
        "abs_mean_E_error", "median_E_error", "total_kWh_error_pct",
        "wasserstein_E", "wasserstein_R", "wasserstein_C", "wasserstein_Cpa",
        "wasserstein_A", "wasserstein_Qint",
        "label_ordering_score", "label_ordering_ok", "renovation_R_ok",
        "score_full", "coverage_ratio_full", "label_target_penalty_full",
    ]
    size_summary_df = (
        case_detail_df.groupby("community_size", dropna=False)[summary_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    return case_detail_df, size_summary_df, case_artifacts


def evaluate_case_size_scenarios_bootstrap(
    df_in: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    community_sizes: Tuple[int, ...] = (20, 50, 100),
    n_cases_per_size: int = 3,
    n_population_candidates: int = 36,
    top_k_proxy: int = 6,
    top_k_full: int = 1,
    random_state: int = RNG_SEED,
    use_community_latent: bool = True,
    use_renovation_signal: bool = True,
    capacity_mode: str = "per_area",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[int, int], Dict[str, pd.DataFrame]]]:
    """Compatibility entry point retaining the former V4 bootstrap default."""
    return evaluate_case_size_scenarios(
        df_in=df_in,
        weather_by_year=weather_by_year,
        community_sizes=community_sizes,
        n_cases_per_size=n_cases_per_size,
        n_population_candidates=n_population_candidates,
        top_k_proxy=top_k_proxy,
        top_k_full=top_k_full,
        random_state=random_state,
        use_community_latent=use_community_latent,
        use_renovation_signal=use_renovation_signal,
        capacity_mode=capacity_mode,
        strategy="bootstrap",
    )
