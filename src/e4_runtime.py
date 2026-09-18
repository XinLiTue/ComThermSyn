from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .copula_runtime import (
    COMPONENTS,
    _candidate_summaries,
    _marginal_lookup,
    _parameter_lookup,
    _stable_seed,
    select_typical_community,
)


E4_MODES = ("e4_rank_lhs_v1", "e4_complete_row_v1")
DIAGNOSTIC_MODES = ("independent_robust_medoid_200", "e2_complete_row_robust_medoid_200")
RANK_COLUMNS = ("rank_G", "rank_C", "rank_A", "rank_Q")


def _validate_template(template: pd.DataFrame) -> None:
    expected = ("template_row_index", *RANK_COLUMNS)
    if tuple(template.columns) != expected or len(template) != 50:
        raise ValueError("E4 rank template must have the frozen 50-row schema")
    target = np.arange(50)
    if any(not np.array_equal(np.sort(template[column].to_numpy(int)), target)
           for column in RANK_COLUMNS):
        raise ValueError("each E4 rank-template column must be a permutation of 0..49")


def _residual_design(mode: str, marginals: dict[str, np.ndarray], template: pd.DataFrame,
                     count: int, rng: np.random.Generator) -> np.ndarray:
    selected = template.iloc[rng.choice(len(template), count, replace=False)]
    ranks = selected[list(RANK_COLUMNS)].to_numpy(int)
    if mode == "independent_robust_medoid_200":
        return np.column_stack([
            rng.choice(marginals[name], count, replace=True) for name in COMPONENTS])
    if mode == "e2_complete_row_robust_medoid_200":
        ranks = template.iloc[rng.choice(len(template), count, replace=True)][list(RANK_COLUMNS)].to_numpy(int)
    if mode in {"e4_complete_row_v1", "e2_complete_row_robust_medoid_200"}:
        return np.column_stack([
            np.sort(marginals[name], kind="stable")[ranks[:, index]]
            for index, name in enumerate(COMPONENTS)
        ])
    result = np.empty((count, 4), float)
    for index, name in enumerate(COMPONENTS):
        uniforms = (rng.permutation(count) + rng.random(count)) / count
        values = np.quantile(marginals[name], uniforms, method="linear")
        order = np.argsort(ranks[:, index], kind="stable")
        result[order, index] = np.sort(values, kind="stable")
    return result


def generate_representative_community(community: pd.DataFrame, artifacts: dict[str, Any], *,
                                      run_seed: int, mode: str
                                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if mode not in {*E4_MODES, *DIAGNOSTIC_MODES}:
        raise ValueError(f"unsupported representative sampling mode: {mode}")
    count = len(community)
    if not 2 <= count <= 50:
        raise ValueError(f"{mode} requires 2 <= N <= 50; received N={count}")
    if "residual_rank_template" not in artifacts:
        raise ValueError(f"{mode} requires anonymous_residual_rank_template.csv")
    template = artifacts["residual_rank_template"]
    _validate_template(template)
    community = community.sort_values("building_id", kind="stable").reset_index(drop=True)
    marginals = _marginal_lookup(artifacts["residual_marginals"])
    lookup = _parameter_lookup(artifacts["parameter_draws"])
    q_scales = artifacts["q_period_scales"].set_index("period")["scale"].to_dict()
    model_version = artifacts["model_metadata"]["model_version"]
    draw_ids = np.sort(artifacts["parameter_draws"].parameter_draw_id.unique().astype(int))
    generated: list[dict[str, Any]] = []
    for draw_id in draw_ids:
        for design_seed in range(4):
            seed = _stable_seed(model_version, run_seed, mode, draw_id, design_seed)
            residuals = _residual_design(mode, marginals, template, count,
                                         np.random.default_rng(seed))
            community_id = f"{mode}__b{draw_id:03d}__d{design_seed:02d}"
            for position, building in enumerate(community.itertuples(index=False)):
                residual = residuals[position].copy()
                residual[3] *= float(q_scales[str(building.period)])
                z = []
                for index, component in enumerate(COMPONENTS):
                    params = lookup[(int(draw_id), component, str(building.period))]
                    mean = params["pooled_mean"] + params["sparse_period_offset"]
                    if component != "Q":
                        mean += params["coefficient"] * (
                            np.log(float(building.Area)) - params["predictor_center"])
                    z.append(float(mean + residual[index]))
                g, c, aperture = np.exp(z[:3])
                generated.append({
                    "community_id": community_id, "parameter_draw_id": int(draw_id),
                    "residual_realization_id": design_seed, "building_id": str(building.building_id),
                    "year": int(building.year), "period": str(building.period),
                    "Area": float(building.Area), "EnergyLabel": str(building.EnergyLabel),
                    "support_level": str(building.support_level), "z_G": z[0], "z_C": z[1],
                    "z_A": z[2], "z_Q": z[3], "G": g, "R": 1.0 / g, "C": c,
                    "A": aperture, "Qint": z[3], "tau_hours": c / g,
                    "energy_label_effect_used": False,
                })
    rows = pd.DataFrame(generated)
    summaries = _candidate_summaries(rows)
    selected_id, summaries = select_typical_community(summaries)
    selected = rows[rows.community_id.eq(selected_id)].copy()
    selected["selected_realization_id"] = selected_id
    selected["model_version"] = model_version
    selected["selection_rule"] = "whole_community_robust_medoid_v1"
    metadata = {"sampling_mode": mode, "candidate_community_count": 200,
                "selected_community_id": selected_id,
                "selection_rule": "whole_community_robust_medoid_v1", "run_seed": int(run_seed)}
    return selected.reset_index(drop=True), summaries, metadata


def generate_e4_community(community: pd.DataFrame, artifacts: dict[str, Any], *,
                          run_seed: int, runtime_config: dict[str, Any], mode: str
                          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if mode not in E4_MODES:
        raise ValueError(f"unsupported E4 sampling mode: {mode}")
    return generate_representative_community(
        community, artifacts, run_seed=run_seed, mode=mode)
