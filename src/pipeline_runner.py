"""
Top-level pipeline orchestration: load data, fit generators, run generation
and validation, export results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    BUILDING_DATA_PATH,
    CASE_SIZE_CONFIG,
    DATA_DIR,
    HOUSE_INFO_PATH,
    PROFILE_YEAR,
    RNG_SEED,
    STANDARD_YEARS,
    STREAMLIT_OUTPUT_DIR,
    VALIDATION_OUTPUT_DIR,
    WEATHER_PATH,
    get_active_case_size_config,
)
from src.data_pipeline import load_merged_dataset, load_weather_data, split_weather_by_year
from src.feature_engineering import add_simulated_energy_metrics, prepare_building_features
from src.thermal_synthesis import (
    GroupGenerator,
    evaluate_case_size_scenarios_bootstrap as evaluate_case_size_scenarios,
    fit_thermal_parameter_store as fit_group_generator,
    synthesize_community_bootstrap as generate_community_rcaq,
)
from src.simulation import simulate_community_heating_custom
from src.validation_internal import (
    run_conditional_distribution_validation,
    run_monotonicity_validation,
    run_pit_validation_v5,
    run_sanity_check,
    select_case_validation_buildings,
)


# ---------------------------------------------------------------------------
# Step 1: Data loading
# ---------------------------------------------------------------------------

def load_and_prepare_data(
    building_path: Optional[Path] = None,
    house_info_path: Optional[Path] = None,
    weather_path: Optional[Path] = None,
    standard_years: Optional[list] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Load and prepare all inputs.

    Returns
    -------
    model_df       : merged building data with forward metrics
    weather_df     : full weather time-series
    weather_by_year: {year: subset_df}
    """
    merged = load_merged_dataset(
        building_path=building_path or BUILDING_DATA_PATH,
        house_info_path=house_info_path or HOUSE_INFO_PATH,
    )
    weather_df = load_weather_data(weather_path=weather_path or WEATHER_PATH)
    weather_by_year = split_weather_by_year(weather_df, years=standard_years or STANDARD_YEARS)

    model_df = add_simulated_energy_metrics(merged, weather_by_year=weather_by_year, verbose=verbose)
    model_df = prepare_building_features(model_df)
    return model_df, weather_df, weather_by_year


# ---------------------------------------------------------------------------
# Step 2: Fit generators
# ---------------------------------------------------------------------------

def fit_generators(
    model_df: pd.DataFrame,
    capacity_mode: str = "per_area",
    use_v5: bool = True,
) -> Dict[str, Any]:
    """
    Fit GroupGenerator (V4) and optionally compute V5 residual stats.

    Returns dict with keys: "gen", "v5_resid_stats" (if use_v5).
    """
    gen = fit_group_generator(model_df, capacity_mode=capacity_mode)
    result: Dict[str, Any] = {"gen": gen}

    if use_v5:
        try:
            from src.thermal_synthesis import compute_residual_statistics, MIN_RESID_SAMPLES
            resid_stats = compute_residual_statistics(gen, min_samples=MIN_RESID_SAMPLES)
            result["v5_resid_stats"] = resid_stats
        except ImportError:
            pass

    return result


# ---------------------------------------------------------------------------
# Step 3: Community generation (case-size scenarios)
# ---------------------------------------------------------------------------

def run_generation_pipeline(
    model_df: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    analysis_mode: str = "fast",
    random_state: int = RNG_SEED,
    use_v5: bool = True,
    capacity_mode: str = "per_area",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run full case-size generation pipeline.

    Returns
    -------
    case_detail_df   : per-case metrics
    size_summary_df  : per-size mean metrics
    case_artifacts   : {(size, case_id): {"real_df", "generated_df", "score_df"}}
    """
    cfg = get_active_case_size_config(analysis_mode)
    community_sizes = cfg["community_sizes"]
    n_cases_per_size = cfg["n_cases_per_size"]
    n_population_candidates = cfg["n_population_candidates"]
    top_k_proxy = cfg["top_k_proxy"]
    top_k_full = cfg["top_k_full"]

    if use_v5:
        try:
            from src.generator_v5 import (
                compute_resid_stats,
                evaluate_case_size_scenarios as _eval_v5,
                MIN_RESID_SAMPLES,
                generate_community_rcaq_v5,
            )

            def _run(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
                gen_full = fit_group_generator(df_in, capacity_mode=capacity_mode)
                resid_stats = compute_resid_stats(gen_full, min_samples=MIN_RESID_SAMPLES)

                from src.generator_v4 import enrich_model_df, GroupGenerator
                from src.scoring import score_population_full_v5, score_population_proxy_v5
                from src.simulation import simulate_standard_year_summary
                from src.utils import save_df

                work = enrich_model_df(df_in)
                rng = np.random.default_rng(random_state)
                case_rows = []
                case_artifacts_local: Dict = {}

                from src.scoring import wasserstein_1d

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

                        gen_cv = fit_group_generator(train_df, capacity_mode=capacity_mode)
                        rs_cv = compute_resid_stats(gen_cv, min_samples=MIN_RESID_SAMPLES)
                        generated_df, score_df, _ = generate_community_rcaq_v5(
                            community_conditions=test_df[["year", "Area", "EnergyLabel"]].copy(),
                            gen=gen_cv,
                            weather_by_year=weather_by_year,
                            r_resid_std_by_vintage=rs_cv[0],
                            global_r_resid_std=rs_cv[1],
                            resid_feat_stds_by_vint=rs_cv[2],
                            global_resid_feat_stds=rs_cv[3],
                            n_population_candidates=n_population_candidates,
                            top_k_proxy=top_k_proxy,
                            top_k_full=top_k_full,
                            random_state=random_state + community_size * 100 + case_id,
                        )
                        best_df = generated_df[generated_df["selected_rank"] == 1].copy().reset_index(drop=True)

                        real_case_df = test_df.copy()
                        real_case_df["community_size"] = community_size
                        real_case_df["case_id"] = case_id
                        real_case_df["source"] = "real"
                        gen_case_df = best_df.copy()
                        gen_case_df["community_size"] = community_size
                        gen_case_df["case_id"] = case_id
                        gen_case_df["source"] = "generated"
                        score_case_df = score_df.copy()
                        score_case_df["community_size"] = community_size
                        score_case_df["case_id"] = case_id

                        case_artifacts_local[(int(community_size), int(case_id))] = {
                            "real_df": real_case_df,
                            "generated_df": gen_case_df,
                            "score_df": score_case_df,
                        }

                        true_e = test_df["E_std_annual_per_m2"].to_numpy(dtype=float)
                        gen_e = best_df["E_std_full_per_m2"].to_numpy(dtype=float)
                        case_rows.append({
                            "community_size": int(community_size),
                            "case_id": int(case_id),
                            "abs_mean_E_error": float(abs(np.mean(gen_e) - np.mean(true_e))),
                            "wasserstein_E": wasserstein_1d(true_e, gen_e),
                            "wasserstein_R": wasserstein_1d(test_df["R"].to_numpy(dtype=float), best_df["R"].to_numpy(dtype=float)),
                        })

                case_detail_df = pd.DataFrame(case_rows).sort_values(["community_size", "case_id"]).reset_index(drop=True)
                size_summary_df = (
                    case_detail_df.groupby("community_size")[["abs_mean_E_error", "wasserstein_E", "wasserstein_R"]]
                    .mean()
                    .reset_index()
                )
                return case_detail_df, size_summary_df, case_artifacts_local

            return _run(model_df)
        except ImportError:
            pass

    return evaluate_case_size_scenarios(
        df_in=model_df,
        weather_by_year=weather_by_year,
        community_sizes=tuple(cfg["community_sizes"]),
        n_cases_per_size=n_cases_per_size,
        n_population_candidates=n_population_candidates,
        top_k_proxy=top_k_proxy,
        top_k_full=top_k_full,
        random_state=random_state,
        capacity_mode=capacity_mode,
    )


# ---------------------------------------------------------------------------
# Step 4: Validation
# ---------------------------------------------------------------------------

def run_validation_pipeline(
    model_df: pd.DataFrame,
    weather_by_year: Dict[int, pd.DataFrame],
    case_size_artifacts: Optional[Dict] = None,
    analysis_mode: str = "fast",
    run_sanity: bool = True,
    run_sys_validation: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run all validation layers and return a dict of result DataFrames.
    """
    output_dir = Path(output_dir or VALIDATION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    # Layer 1 – sanity
    if run_sanity and case_size_artifacts:
        frames = []
        for payload in case_size_artifacts.values():
            gen_df = payload.get("generated_df")
            if gen_df is not None and len(gen_df):
                frames.append(gen_df.copy())
        if frames:
            all_gen = pd.concat(frames, ignore_index=True)
            sanity_report, sanity_summary = run_sanity_check(all_gen, model_df)
            results["sanity_report_df"] = sanity_report
            results["sanity_summary"] = sanity_summary

    # Layer 2 – systemic validation
    if run_sys_validation:
        results["monotonicity_report_df"] = run_monotonicity_validation(model_df)
        if case_size_artifacts:
            frames = []
            for payload in case_size_artifacts.values():
                gen_df = payload.get("generated_df")
                if gen_df is not None and len(gen_df):
                    frames.append(gen_df.copy())
            if frames:
                all_gen = pd.concat(frames, ignore_index=True)
                results["conditional_validation_df"] = run_conditional_distribution_validation(model_df, all_gen)

        pit_detail, pit_summary = run_pit_validation_v5(
            df_in=model_df,
            weather_by_year=weather_by_year,
            validation_mode=analysis_mode,
        )
        results["pit_validation_detail_df"] = pit_detail
        results["pit_validation_summary_df"] = pit_summary

    return results


# ---------------------------------------------------------------------------
# Step 5: Export
# ---------------------------------------------------------------------------

def run_export_pipeline(
    *,
    export_root: Optional[Path] = None,
    case_size_artifacts: Optional[Dict] = None,
    case_size_summary_df: Optional[pd.DataFrame] = None,
    case_size_detail_df: Optional[pd.DataFrame] = None,
    validation_results: Optional[Dict] = None,
    input_config: Optional[Dict] = None,
    analysis_mode: str = "fast",
) -> Dict[str, Any]:
    """Convenience wrapper around src.export_utils.run_export_pipeline."""
    from src.export_utils import run_export_pipeline as _run_export

    export_root = Path(export_root or STREAMLIT_OUTPUT_DIR)
    val = validation_results or {}
    return _run_export(
        export_root=export_root,
        case_size_artifacts=case_size_artifacts,
        case_size_summary_df=case_size_summary_df,
        case_size_detail_df=case_size_detail_df,
        sanity_report_df=val.get("sanity_report_df"),
        conditional_validation_df=val.get("conditional_validation_df"),
        monotonicity_report_df=val.get("monotonicity_report_df"),
        pit_validation_detail_df=val.get("pit_validation_detail_df"),
        pit_validation_summary_df=val.get("pit_validation_summary_df"),
        case_validation_summary_df=val.get("case_validation_summary_df"),
        input_config=input_config,
    )


# ---------------------------------------------------------------------------
# Community extraction and heating analysis helpers
# ---------------------------------------------------------------------------

def get_available_community_sizes(case_size_artifacts: Dict) -> List[int]:
    """Return sorted list of community sizes present in case_size_artifacts."""
    if not case_size_artifacts:
        return []
    return sorted({size for (size, _case_id) in case_size_artifacts.keys()})


def get_case_size_pools(
    case_artifacts: Dict,
    community_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate real and generated pools for a given community size."""
    real_frames: List[pd.DataFrame] = []
    gen_frames: List[pd.DataFrame] = []
    for (size, case_id), payload in case_artifacts.items():
        if size != community_size:
            continue
        real_df = payload.get("real_df")
        gen_df = payload.get("generated_df")
        if real_df is not None and len(real_df) > 0:
            real_frames.append(real_df.assign(community_size=size, case_id=case_id))
        if gen_df is not None and len(gen_df) > 0:
            gen_frames.append(gen_df.assign(community_size=size, case_id=case_id))

    available = get_available_community_sizes(case_artifacts)
    if not real_frames or not gen_frames:
        raise ValueError(
            f"No real/generated pools found for community_size={community_size}. "
            f"Available sizes: {available}."
        )
    return (
        pd.concat(real_frames, ignore_index=True),
        pd.concat(gen_frames, ignore_index=True),
    )


def extract_generated_communities(
    case_size_artifacts: Dict,
    *,
    community_sizes: Optional[List[int]] = None,
    max_cases_per_size: Optional[int] = None,
) -> List[Tuple[str, pd.DataFrame]]:
    """Return [(case_name, selected_df)] filtered by size/case limit."""
    allowed_sizes = None if community_sizes is None else set(community_sizes)
    counts_by_size: Dict[int, int] = {}
    cases: List[Tuple[str, pd.DataFrame]] = []

    for (community_size, case_id), payload in sorted(case_size_artifacts.items()):
        if allowed_sizes is not None and community_size not in allowed_sizes:
            continue
        counts_by_size.setdefault(community_size, 0)
        if max_cases_per_size is not None and counts_by_size[community_size] >= max_cases_per_size:
            continue
        selected_df = payload.get("generated_df")
        if selected_df is None or len(selected_df) == 0:
            continue
        counts_by_size[community_size] += 1
        case_name = f"size{community_size}_case{case_id}"
        cases.append((case_name, selected_df.copy().reset_index(drop=True)))

    if not cases:
        raise ValueError(
            "No generated communities were extracted. "
            "Check case_size_artifacts and the selected mode filters."
        )
    return cases


def _get_generated_building_row_local(
    case_size_artifacts: Dict,
    *,
    case_name: str,
    building_id: str,
) -> Dict[str, Any]:
    """Look up a single building row from case_size_artifacts by case_name + building_id."""
    for (community_size, case_id), payload in case_size_artifacts.items():
        if f"size{community_size}_case{case_id}" != case_name:
            continue
        gen_df = payload.get("generated_df")
        if gen_df is None or len(gen_df) == 0:
            break
        match = gen_df.loc[gen_df["building_id"].astype(str) == str(building_id)]
        if len(match):
            return match.iloc[0].to_dict()
        break
    raise ValueError(
        f"Could not find building_id={building_id!r} in case_name={case_name!r}. "
        f"Available sizes: {get_available_community_sizes(case_size_artifacts)}."
    )


def _build_sunny_cloudy_weather_from_weather_by_year(
    weather_by_year: Dict[int, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sunny_weather_df, cloudy_weather_df) from the first available year's winter months."""
    from src.simulation import ensure_datetime_index, infer_weather_columns

    if not weather_by_year:
        raise ValueError("weather_by_year is empty.")
    weather_year = sorted(weather_by_year.keys())[0]
    base_weather = ensure_datetime_index(weather_by_year[weather_year].copy())
    _temp_col, solar_col = infer_weather_columns(base_weather)

    winter_weather = base_weather.loc[base_weather.index.month.isin([12, 1, 2])].copy()
    if len(winter_weather) == 0:
        winter_weather = base_weather.copy()

    daily_solar = winter_weather[solar_col].resample("D").sum().dropna()
    if len(daily_solar) == 0:
        raise ValueError("No daily solar data available to construct sunny/cloudy slices.")

    sunny_day = daily_solar.idxmax()
    cloudy_day = daily_solar.idxmin()
    sunny_weather = winter_weather.loc[sunny_day.strftime("%Y-%m-%d")].copy()
    cloudy_weather = winter_weather.loc[cloudy_day.strftime("%Y-%m-%d")].copy()
    if not isinstance(sunny_weather, pd.DataFrame):
        sunny_weather = winter_weather.loc[[sunny_day]].copy()
    if not isinstance(cloudy_weather, pd.DataFrame):
        cloudy_weather = winter_weather.loc[[cloudy_day]].copy()
    return sunny_weather, cloudy_weather


def run_annual_heating_case_analysis(
    case_size_artifacts: Dict,
    weather_by_year: Dict[int, pd.DataFrame],
    *,
    weather_year: Optional[int] = None,
    community_sizes: Optional[List[int]] = None,
    max_cases_per_size: Optional[int] = None,
    extreme_quantiles: Tuple[float, float] = (0.05, 0.95),
    keep_all_community_profiles: bool = False,
    summer_months: tuple = (6, 7, 8),
    initial_temp: float = 19.0,
    day_start: int = 9,
    day_end: int = 17,
    day_setpoint: float = 18.0,
    other_setpoint: float = 20.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Simulate annual heating for all generated communities and label min/middle/max cases.

    Returns (annual_case_df, building_summaries, community_profiles).
    Auto-derives annual_heat_mwh from total_annual_mwh or total_annual_kwh if missing.
    """
    if weather_year is None:
        weather_year = sorted(weather_by_year.keys())[0]
    weather_df = weather_by_year[weather_year]

    selected_cases = extract_generated_communities(
        case_size_artifacts,
        community_sizes=community_sizes,
        max_cases_per_size=max_cases_per_size,
    )

    community_rows: List[Dict] = []
    building_summaries: Dict[str, pd.DataFrame] = {}
    community_profiles: Dict[str, pd.DataFrame] = {}

    for case_name, selected_df in selected_cases:
        community_summary, building_summary, community_profile = simulate_community_heating_custom(
            selected_df=selected_df,
            weather_df=weather_df,
            case_name=case_name,
            summer_months=summer_months,
            initial_temp=initial_temp,
            day_start=day_start,
            day_end=day_end,
            day_setpoint=day_setpoint,
            other_setpoint=other_setpoint,
        )
        community_summary["weather_year"] = weather_year
        community_rows.append(community_summary)
        building_summaries[case_name] = building_summary
        community_profiles[case_name] = community_profile

    annual_case_df = pd.DataFrame(community_rows)

    # Normalise column names — derive missing aliases
    if "annual_heat_mwh" not in annual_case_df.columns:
        if "total_annual_mwh" in annual_case_df.columns:
            annual_case_df["annual_heat_mwh"] = annual_case_df["total_annual_mwh"]
        elif "total_annual_kwh" in annual_case_df.columns:
            annual_case_df["annual_heat_mwh"] = annual_case_df["total_annual_kwh"] / 1000.0
        elif "annual_heat_kwh" in annual_case_df.columns:
            annual_case_df["annual_heat_mwh"] = annual_case_df["annual_heat_kwh"] / 1000.0
    if "annual_heat_kwh_per_m2" not in annual_case_df.columns and "mean_annual_kwh_per_m2" in annual_case_df.columns:
        annual_case_df["annual_heat_kwh_per_m2"] = annual_case_df["mean_annual_kwh_per_m2"]
    if "peak_heat_kw" not in annual_case_df.columns and "peak_community_kw" in annual_case_df.columns:
        annual_case_df["peak_heat_kw"] = annual_case_df["peak_community_kw"]

    q_low, q_high = float(annual_case_df["annual_heat_mwh"].quantile(extreme_quantiles[0])), float(
        annual_case_df["annual_heat_mwh"].quantile(extreme_quantiles[1])
    )
    annual_case_df["is_extreme"] = (
        annual_case_df["annual_heat_mwh"].lt(q_low) | annual_case_df["annual_heat_mwh"].gt(q_high)
    )

    non_extreme = annual_case_df.loc[~annual_case_df["is_extreme"]].copy()
    if len(non_extreme) < 3:
        non_extreme = annual_case_df.copy()

    min_idx = non_extreme["annual_heat_mwh"].idxmin()
    max_idx = non_extreme["annual_heat_mwh"].idxmax()
    mid_idx = (non_extreme["annual_heat_mwh"] - non_extreme["annual_heat_mwh"].median()).abs().idxmin()

    annual_case_df["display_role"] = ""
    annual_case_df.loc[min_idx, "display_role"] = "minimum"
    annual_case_df.loc[mid_idx, "display_role"] = "middle"
    annual_case_df.loc[max_idx, "display_role"] = "maximum"

    selected_names = annual_case_df.loc[annual_case_df["display_role"] != "", "case_name"].tolist()
    if not keep_all_community_profiles:
        community_profiles = {k: v for k, v in community_profiles.items() if k in selected_names}

    return annual_case_df, building_summaries, community_profiles
