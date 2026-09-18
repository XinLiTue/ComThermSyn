from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, norm, qmc

from .copula_runtime import (
    COMPONENTS,
    _candidate_summaries,
    _empirical_quantile,
    _marginal_lookup,
    _parameter_lookup,
    _stable_seed,
    generate_typical_community as generate_full_mc_community,
)


def _lhs_probability_design(
    community: pd.DataFrame,
    correlation: np.ndarray,
    *,
    candidate_count: int,
    model_version: str,
    run_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    building_count = len(community)
    independent = np.empty((candidate_count, building_count, len(COMPONENTS)), dtype=float)
    correlated = np.empty_like(independent)
    cholesky = np.linalg.cholesky(correlation)
    for building_index, building in enumerate(community.itertuples(index=False)):
        seed = _stable_seed(model_version, run_seed, "lhs", building.building_id) % (2**32)
        uniforms = qmc.LatinHypercube(d=len(COMPONENTS), scramble=True, seed=seed).random(
            candidate_count
        )
        normals = norm.ppf(np.clip(uniforms, 1e-12, 1.0 - 1e-12))
        independent[:, building_index, :] = normals
        correlated[:, building_index, :] = normals @ cholesky.T
    return independent, norm.cdf(correlated)


def _rank_lhs_designs(
    independent: np.ndarray,
    correlated_uniforms: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    candidate_count, building_count, component_count = independent.shape
    degrees_of_freedom = building_count * component_count
    d2 = np.sum(np.square(independent), axis=(1, 2))
    joint_deviation = np.abs(d2 - degrees_of_freedom) / np.sqrt(2.0 * degrees_of_freedom)

    soft_low, soft_high = map(float, config["lhs_soft_marginal_quantiles"])
    lower_excess = np.maximum(norm.ppf(soft_low) - norm.ppf(correlated_uniforms), 0.0)
    upper_excess = np.maximum(norm.ppf(correlated_uniforms) - norm.ppf(soft_high), 0.0)
    tail_penalty = np.mean(np.square(lower_excess + upper_excess), axis=(1, 2))
    maximum_tail = np.max(np.abs(norm.ppf(correlated_uniforms)), axis=(1, 2))

    hard_low, hard_high = map(float, config["lhs_hard_marginal_quantiles"])
    extreme_count = np.sum(
        (correlated_uniforms < hard_low) | (correlated_uniforms > hard_high), axis=(1, 2)
    )
    expected_extreme_probability = hard_low + (1.0 - hard_high)
    allowed_extremes = int(
        binom.ppf(0.95, degrees_of_freedom, expected_extreme_probability)
    )
    c_low, c_high = map(float, config["lhs_central_c_quantiles"])
    c_uniforms = correlated_uniforms[:, :, COMPONENTS.index("C")]
    c_lower_tail_count = np.sum(c_uniforms < c_low, axis=1)
    c_upper_tail_count = np.sum(c_uniforms > c_high, axis=1)
    c_tail_count = c_lower_tail_count + c_upper_tail_count
    allowed_c_tail_count = int(
        np.floor(building_count * (c_low + 1.0 - c_high) + 1e-12)
    )
    allowed_c_upper_tail_count = int(np.floor(building_count * (1.0 - c_high) + 1e-12))
    c_tail_gate = (c_tail_count <= allowed_c_tail_count) & (
        c_upper_tail_count <= allowed_c_upper_tail_count
    )
    typical_low, typical_high = map(float, config["lhs_joint_typical_quantiles"])
    joint_low = float(chi2.ppf(typical_low, degrees_of_freedom))
    joint_high = float(chi2.ppf(typical_high, degrees_of_freedom))
    joint_gate = (d2 >= joint_low) & (d2 <= joint_high) & (extreme_count <= allowed_extremes)
    typicality_pass = joint_gate & c_tail_gate

    c_tail_excess = np.maximum(c_tail_count - allowed_c_tail_count, 0) + np.maximum(
        c_upper_tail_count - allowed_c_upper_tail_count, 0
    )
    score = (
        joint_deviation
        + float(config["lhs_tail_weight"]) * tail_penalty
        + float(config["lhs_max_tail_weight"]) * np.square(
            np.maximum(maximum_tail - norm.ppf(0.99), 0.0)
        )
        + float(config["lhs_c_tail_excess_weight"]) * c_tail_excess / building_count
    )
    ranked = pd.DataFrame(
        {
            "lhs_candidate_index": np.arange(candidate_count, dtype=int),
            "joint_d2": d2,
            "joint_expected_d2": degrees_of_freedom,
            "joint_typical_low": joint_low,
            "joint_typical_high": joint_high,
            "joint_deviation_score": joint_deviation,
            "tail_penalty": tail_penalty,
            "maximum_absolute_latent": maximum_tail,
            "extreme_marginal_count": extreme_count,
            "allowed_extreme_marginal_count": allowed_extremes,
            "c_lower_tail_count": c_lower_tail_count,
            "c_upper_tail_count": c_upper_tail_count,
            "c_tail_count": c_tail_count,
            "allowed_c_tail_count": allowed_c_tail_count,
            "allowed_c_upper_tail_count": allowed_c_upper_tail_count,
            "c_tail_gate_pass": c_tail_gate,
            "joint_probability_gate_pass": joint_gate,
            "probability_typicality_pass": typicality_pass,
            "prescreen_score": score,
        }
    )
    return ranked.sort_values(
        ["probability_typicality_pass", "prescreen_score", "lhs_candidate_index"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _parameter_schedule(
    parameter_draws: pd.DataFrame,
    *,
    candidate_count: int,
    model_version: str,
    run_seed: int,
) -> np.ndarray:
    draw_ids = np.sort(parameter_draws["parameter_draw_id"].unique().astype(int))
    if candidate_count > len(draw_ids):
        raise ValueError("lhs candidate count cannot exceed the available parameter draws")
    seed = _stable_seed(model_version, run_seed, "parameter_draw_schedule")
    return np.random.default_rng(seed).permutation(draw_ids)[:candidate_count]


def _generate_lhs_rows(
    community: pd.DataFrame,
    artifacts: dict[str, Any],
    design_rows: pd.DataFrame,
    correlated_uniforms: np.ndarray,
    parameter_schedule: np.ndarray,
) -> pd.DataFrame:
    parameter_draws = artifacts["parameter_draws"]
    lookup = _parameter_lookup(parameter_draws)
    marginals = _marginal_lookup(artifacts["residual_marginals"])
    q_scales = artifacts["q_period_scales"].set_index("period")["scale"].to_dict()
    generated: list[dict[str, Any]] = []

    for design in design_rows.itertuples(index=False):
        candidate_index = int(design.lhs_candidate_index)
        draw_id = int(parameter_schedule[candidate_index])
        community_id = f"lhs_fast__c{candidate_index:03d}__b{draw_id:03d}"
        for building_index, building in enumerate(community.itertuples(index=False)):
            uniforms = correlated_uniforms[candidate_index, building_index]
            residual = {
                component: _empirical_quantile(marginals[component], uniforms[index])
                for index, component in enumerate(COMPONENTS)
            }
            residual["Q"] *= float(q_scales[str(building.period)])
            z: dict[str, float] = {}
            component_means: dict[str, float] = {}
            for component in COMPONENTS:
                params = lookup[(draw_id, component, str(building.period))]
                mean = params["pooled_mean"] + params["sparse_period_offset"]
                if component != "Q":
                    mean += params["coefficient"] * (
                        np.log(float(building.Area)) - params["predictor_center"]
                    )
                component_means[component] = float(mean)
                z[component] = float(mean + residual[component])

            G, C, A = (float(np.exp(z[name])) for name in ("G", "C", "A"))
            generated.append(
                {
                    "community_id": community_id,
                    "parameter_draw_id": draw_id,
                    "residual_realization_id": candidate_index,
                    "lhs_candidate_index": candidate_index,
                    "building_id": str(building.building_id),
                    "year": int(building.year),
                    "period": str(building.period),
                    "Area": float(building.Area),
                    "EnergyLabel": str(building.EnergyLabel),
                    "support_level": str(building.support_level),
                    "z_G": z["G"],
                    "z_C": z["C"],
                    "z_A": z["A"],
                    "z_Q": z["Q"],
                    "G": G,
                    "R": 1.0 / G,
                    "C": C,
                    "A": A,
                    "Qint": z["Q"],
                    "tau_hours": C / G,
                    "C_conditional_center": float(np.exp(component_means["C"])),
                    "C_residual_multiplier": float(np.exp(residual["C"])),
                    "residual_quantile_G": float(uniforms[0]),
                    "residual_quantile_C": float(uniforms[1]),
                    "residual_quantile_A": float(uniforms[2]),
                    "residual_quantile_Q": float(uniforms[3]),
                    "probability_typicality_pass": bool(design.probability_typicality_pass),
                    "prescreen_score": float(design.prescreen_score),
                    "energy_label_effect_used": False,
                }
            )
    rows = pd.DataFrame(generated)
    numeric = rows[["R", "C", "A", "Qint", "tau_hours"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (rows[["R", "C", "A", "tau_hours"]] > 0).all().all():
        raise ValueError("LHS sampling produced a non-finite or non-positive thermal parameter")
    return rows


def generate_lhs_fast_community(
    community: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    run_seed: int,
    runtime_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_version = artifacts["model_metadata"]["model_version"]
    candidate_count = int(runtime_config["lhs_max_candidates"])
    evaluated_count = int(runtime_config["lhs_evaluated_candidates"])
    independent, uniforms = _lhs_probability_design(
        community,
        np.asarray(artifacts["copula_correlation"], dtype=float),
        candidate_count=candidate_count,
        model_version=model_version,
        run_seed=run_seed,
    )
    ranked_designs = _rank_lhs_designs(independent, uniforms, runtime_config)
    acceptable = ranked_designs.loc[ranked_designs["probability_typicality_pass"]]
    acceptance_fallback_used = acceptable.empty
    if acceptable.empty:
        fallback_pool = ranked_designs.loc[ranked_designs["joint_probability_gate_pass"]]
        if fallback_pool.empty:
            fallback_pool = ranked_designs
        evaluated_designs = fallback_pool.head(evaluated_count).copy()
    else:
        evaluated_designs = acceptable.head(evaluated_count).copy()
    schedule = _parameter_schedule(
        artifacts["parameter_draws"],
        candidate_count=candidate_count,
        model_version=model_version,
        run_seed=run_seed,
    )
    rows = _generate_lhs_rows(community, artifacts, evaluated_designs, uniforms, schedule)
    summaries = _candidate_summaries(rows).merge(
        evaluated_designs,
        left_on="residual_realization_id",
        right_on="lhs_candidate_index",
        how="left",
        validate="one_to_one",
    )
    summaries = summaries.sort_values(
        ["prescreen_score", "community_id"], kind="mergesort"
    ).reset_index(drop=True)
    summaries["is_selected_typical"] = False
    summaries.loc[0, "is_selected_typical"] = True
    selected_id = str(summaries.loc[0, "community_id"])
    selected = rows.loc[rows["community_id"].eq(selected_id)].copy()
    selected["selected_realization_id"] = selected_id
    selected["model_version"] = model_version
    selected["selection_rule"] = "lhs_scale_stable_typicality_v2"
    metadata = {
        "sampling_mode": "lhs_fast",
        "planned_probability_designs": candidate_count,
        "fully_evaluated_communities": int(len(summaries)),
        "acceptable_probability_designs": int(len(acceptable)),
        "candidate_community_count": int(len(summaries)),
        "selected_community_id": selected_id,
        "acceptance_fallback_used": bool(acceptance_fallback_used),
        "selected_prescreen_score": float(summaries.loc[0, "prescreen_score"]),
        "selection_rule": "lhs_scale_stable_typicality_v2",
        "run_seed": int(run_seed),
    }
    return selected.reset_index(drop=True), summaries, metadata


def generate_deployment_community(
    community: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    run_seed: int,
    runtime_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sampling_mode = str(runtime_config.get("sampling_mode", "lhs_fast"))
    if sampling_mode == "full_mc":
        selected, summaries, metadata = generate_full_mc_community(
            community, artifacts, run_seed=run_seed
        )
        metadata["sampling_mode"] = "full_mc"
        metadata["planned_probability_designs"] = int(len(summaries))
        metadata["fully_evaluated_communities"] = int(len(summaries))
        metadata["acceptable_probability_designs"] = int(len(summaries))
        return selected, summaries, metadata
    if sampling_mode in {"e4_rank_lhs_v1", "e4_complete_row_v1"}:
        from .e4_runtime import generate_e4_community

        return generate_e4_community(
            community, artifacts, run_seed=run_seed, runtime_config=runtime_config,
            mode=sampling_mode,
        )
    if sampling_mode != "lhs_fast":
        raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
    return generate_lhs_fast_community(
        community,
        artifacts,
        run_seed=run_seed,
        runtime_config=runtime_config,
    )
