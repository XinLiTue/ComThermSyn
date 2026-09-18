from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


COMPONENTS = ("G", "C", "A", "Q")
SUPPORTED_PERIODS = (
    "1946 - 1964",
    "1965 - 1974",
    "1975 - 1991",
    "1992 - 2005",
)


def period_from_year(year: int) -> str:
    if 1946 <= year <= 1964:
        return "1946 - 1964"
    if 1965 <= year <= 1974:
        return "1965 - 1974"
    if 1975 <= year <= 1991:
        return "1975 - 1991"
    if 1992 <= year <= 2005:
        return "1992 - 2005"
    raise ValueError(f"construction year {year} is outside the supported range 1946-2005")


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _empirical_quantile(sorted_values: np.ndarray, probability: float) -> float:
    probability = float(np.clip(probability, 0.0, 1.0))
    return float(np.quantile(sorted_values, probability, method="linear"))


@dataclass(frozen=True)
class CandidateGeneration:
    rows: pd.DataFrame
    summaries: pd.DataFrame
    selected_community_id: str


def _parameter_lookup(parameter_draws: pd.DataFrame) -> dict[tuple[int, str, str], dict[str, float]]:
    lookup: dict[tuple[int, str, str], dict[str, float]] = {}
    for row in parameter_draws.itertuples(index=False):
        key = (int(row.parameter_draw_id), str(row.component), str(row.period))
        lookup[key] = {
            "pooled_mean": float(row.pooled_mean),
            "sparse_period_offset": float(row.sparse_period_offset),
            "coefficient": float(row.coefficient),
            "predictor_center": float(row.predictor_center),
        }
    return lookup


def _marginal_lookup(residual_marginals: pd.DataFrame) -> dict[str, np.ndarray]:
    pools: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        values = (
            residual_marginals.loc[residual_marginals["component"].eq(component)]
            .sort_values("order")["value"]
            .to_numpy(dtype=float)
        )
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid residual marginal for {component}")
        pools[component] = values
    return pools


def _candidate_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, float | int | str]] = []
    for community_id, group in rows.groupby("community_id", sort=True):
        summaries.append(
            {
                "community_id": community_id,
                "parameter_draw_id": int(group["parameter_draw_id"].iloc[0]),
                "residual_realization_id": int(group["residual_realization_id"].iloc[0]),
                "building_count": int(len(group)),
                "log_R_median": float(np.log(group["R"]).median()),
                "log_C_median": float(np.log(group["C"]).median()),
                "log_A_median": float(np.log(group["A"]).median()),
                "Qint_mean": float(group["Qint"].mean()),
                "log_tau_p10": float(np.quantile(np.log(group["tau_hours"]), 0.10)),
                "log_tau_p50": float(np.quantile(np.log(group["tau_hours"]), 0.50)),
                "log_tau_p90": float(np.quantile(np.log(group["tau_hours"]), 0.90)),
            }
        )
    return pd.DataFrame(summaries)


def select_typical_community(summaries: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    feature_columns = [
        "log_R_median",
        "log_C_median",
        "log_A_median",
        "Qint_mean",
        "log_tau_p10",
        "log_tau_p50",
        "log_tau_p90",
    ]
    features = summaries[feature_columns].to_numpy(dtype=float)
    center = np.median(features, axis=0)
    mad = np.median(np.abs(features - center), axis=0)
    fallback = np.std(features, axis=0, ddof=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, np.where(fallback > 1e-12, fallback, 1.0))
    distances = np.sum(np.square((features - center) / scale), axis=1)
    ranked = summaries.copy()
    ranked["robust_medoid_distance"] = distances
    ranked = ranked.sort_values(
        ["robust_medoid_distance", "community_id"], kind="mergesort"
    ).reset_index(drop=True)
    ranked["is_selected_typical"] = False
    ranked.loc[0, "is_selected_typical"] = True
    return str(ranked.loc[0, "community_id"]), ranked


def generate_candidate_communities(
    community: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    run_seed: int,
) -> CandidateGeneration:
    config = artifacts["runtime_config"]
    model_version = artifacts["model_metadata"]["model_version"]
    parameter_draws = artifacts["parameter_draws"]
    residual_marginals = artifacts["residual_marginals"]
    correlation = np.asarray(artifacts["copula_correlation"], dtype=float)
    q_scales = artifacts["q_period_scales"].set_index("period")["scale"].to_dict()

    lookup = _parameter_lookup(parameter_draws)
    marginals = _marginal_lookup(residual_marginals)
    cholesky = np.linalg.cholesky(correlation)
    draw_ids = sorted(int(value) for value in parameter_draws["parameter_draw_id"].unique())
    residual_realizations = int(config["residual_realizations_per_draw"])

    generated: list[dict[str, Any]] = []
    for draw_id in draw_ids:
        for residual_id in range(residual_realizations):
            community_id = f"shrunk_copula__b{draw_id:03d}__r{residual_id:03d}"
            for building in community.itertuples(index=False):
                seed = _stable_seed(
                    model_version,
                    int(run_seed),
                    draw_id,
                    residual_id,
                    building.building_id,
                )
                rng = np.random.default_rng(seed)
                latent = cholesky @ rng.standard_normal(len(COMPONENTS))
                uniforms = norm.cdf(latent)
                residual = {
                    component: _empirical_quantile(marginals[component], uniforms[index])
                    for index, component in enumerate(COMPONENTS)
                }
                residual["Q"] *= float(q_scales[str(building.period)])

                z: dict[str, float] = {}
                for component in COMPONENTS:
                    params = lookup[(draw_id, component, str(building.period))]
                    mean = params["pooled_mean"] + params["sparse_period_offset"]
                    if component != "Q":
                        mean += params["coefficient"] * (
                            np.log(float(building.Area)) - params["predictor_center"]
                        )
                    z[component] = float(mean + residual[component])

                G = float(np.exp(z["G"]))
                C = float(np.exp(z["C"]))
                A = float(np.exp(z["A"]))
                generated.append(
                    {
                        "community_id": community_id,
                        "parameter_draw_id": draw_id,
                        "residual_realization_id": residual_id,
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
                        "energy_label_effect_used": False,
                    }
                )

    rows = pd.DataFrame(generated)
    summaries = _candidate_summaries(rows)
    selected_id, ranked = select_typical_community(summaries)
    return CandidateGeneration(rows=rows, summaries=ranked, selected_community_id=selected_id)


def generate_typical_community(
    community: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    run_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = generate_candidate_communities(community, artifacts, run_seed=run_seed)
    selected = candidates.rows.loc[
        candidates.rows["community_id"].eq(candidates.selected_community_id)
    ].copy()
    selected["selected_realization_id"] = candidates.selected_community_id
    selected["model_version"] = artifacts["model_metadata"]["model_version"]
    selected["selection_rule"] = "whole_community_robust_medoid_v1"
    metadata = {
        "candidate_community_count": int(len(candidates.summaries)),
        "selected_community_id": candidates.selected_community_id,
        "selection_rule": "whole_community_robust_medoid_v1",
        "run_seed": int(run_seed),
    }
    return selected.reset_index(drop=True), candidates.summaries, metadata
