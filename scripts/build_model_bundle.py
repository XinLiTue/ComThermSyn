from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ("G", "C", "A", "Q")
SUPPORTED_PERIODS = (
    "1946 - 1964",
    "1965 - 1974",
    "1975 - 1991",
    "1992 - 2005",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_bundle(project_root: Path, deploy_root: Path) -> None:
    base = project_root / "analysis_outputs" / "community_rcaq_v1"
    module4 = base / "task099_module4_v1"
    module5 = base / "task099_module5_v1"
    freeze = base / "task102b_product_freeze_v1"
    model_root = deploy_root / "artifacts_public" / "model"
    model_root.mkdir(parents=True, exist_ok=True)

    source_draws = module5 / "parameter_draw_audit.csv"
    source_residuals = module4 / "innovation_residual_candidates.csv"
    source_scales = module5 / "residual_scale_model.csv"
    source_copula = module5 / "copula_matrices.csv"
    source_config = freeze / "product_candidate_config.json"

    draws = pd.read_csv(source_draws)
    draws = draws.loc[
        draws["period"].isin(SUPPORTED_PERIODS),
        [
            "parameter_draw_id",
            "component",
            "model",
            "period",
            "predictor_center",
            "coefficient",
            "pooled_mean",
            "sparse_period_offset",
        ],
    ].copy()
    numeric_columns = [
        "parameter_draw_id",
        "predictor_center",
        "coefficient",
        "pooled_mean",
        "sparse_period_offset",
    ]
    for column in numeric_columns:
        draws[column] = pd.to_numeric(draws[column], errors="coerce").fillna(0.0)
    draws["parameter_draw_id"] = draws["parameter_draw_id"].astype(int)
    expected_rows = 50 * len(COMPONENTS) * len(SUPPORTED_PERIODS)
    if len(draws) != expected_rows or draws["parameter_draw_id"].nunique() != 50:
        raise ValueError("unexpected frozen parameter-draw shape")
    draws.to_parquet(model_root / "parameter_draws.parquet", index=False)

    scales = pd.read_csv(source_scales)
    q_scales = scales.loc[
        scales["component"].eq("Q") & scales["period"].isin(SUPPORTED_PERIODS),
        ["period", "target_period_scale"],
    ].rename(columns={"target_period_scale": "scale"})
    q_scales["scale"] = pd.to_numeric(q_scales["scale"], errors="raise")
    if set(q_scales["period"]) != set(SUPPORTED_PERIODS):
        raise ValueError("missing Q period scale")
    q_scales.to_parquet(model_root / "q_period_scales.parquet", index=False)
    q_scale_lookup = q_scales.set_index("period")["scale"].to_dict()

    innovations = pd.read_csv(source_residuals)
    innovations = innovations.loc[innovations["q_model"].eq("Q0")].copy()
    if len(innovations) != 50 or innovations["building_id_public"].nunique() != 50:
        raise ValueError("the frozen residual pool must contain 50 Q0 rows")
    marginal_rows: list[dict[str, float | int | str]] = []
    for component in COMPONENTS:
        values = pd.to_numeric(innovations[f"{component}_innovation_used"], errors="raise")
        if component == "Q":
            values = pd.Series(
                [
                    value / float(q_scale_lookup[period])
                    for value, period in zip(values, innovations["period"], strict=True)
                ]
            )
        for order, value in enumerate(np.sort(values.to_numpy(dtype=float))):
            marginal_rows.append({"component": component, "order": order, "value": float(value)})
    pd.DataFrame(marginal_rows).to_parquet(
        model_root / "residual_marginals.parquet", index=False
    )

    copula = pd.read_csv(source_copula)
    copula = copula.loc[copula["matrix_type"].eq("shrunk_correlation")].copy()
    matrix = np.empty((4, 4), dtype=float)
    for row in copula.itertuples(index=False):
        matrix[COMPONENTS.index(row.row_component), COMPONENTS.index(row.column_component)] = float(
            row.value
        )
    if np.linalg.eigvalsh(matrix).min() <= 0:
        raise ValueError("frozen Copula matrix is not positive definite")
    _write_json(
        model_root / "copula_correlation.json",
        {
            "components": list(COMPONENTS),
            "matrix": matrix.tolist(),
            "method": "Ledoit-Wolf shrunk Gaussian Copula",
            "lambda_Sigma": float(copula["lambda_Sigma"].iloc[0]),
        },
    )

    frozen_config = json.loads(source_config.read_text(encoding="utf-8"))
    metadata = {
        "model_version": "task102b_copula_deploy_v1_20260830",
        "payload_type": "community_rcaq_typical_realization",
        "status": "frozen_development_candidate",
        "source_candidate_id": frozen_config["candidate_id"],
        "method": "M0a/Q0 conditional mean plus joint shrunk Gaussian Copula residuals",
        "target": "community-level RCAQ generation",
        "energy_label_role": "required pass-through; not predictive in this version",
        "evidence_boundaries": frozen_config["evidence_boundaries"],
    }
    _write_json(model_root / "model_metadata.json", metadata)
    _write_json(
        model_root / "runtime_config.json",
        {
            "parameter_draw_count": 50,
            "residual_realizations_per_draw": 2,
            "full_mc_candidate_community_count": 100,
            "sampling_mode": "lhs_fast",
            "sampler_version": "lhs_scale_stable_typicality_v2_20260909",
            "lhs_max_candidates": 16,
            "lhs_evaluated_candidates": 3,
            "lhs_joint_typical_quantiles": [0.05, 0.95],
            "lhs_soft_marginal_quantiles": [0.10, 0.90],
            "lhs_hard_marginal_quantiles": [0.005, 0.995],
            "lhs_central_c_quantiles": [0.10, 0.90],
            "lhs_tail_weight": 2.0,
            "lhs_max_tail_weight": 0.25,
            "lhs_c_tail_excess_weight": 4.0,
            "default_run_seed": 20260830,
            "selection_rule": "lhs_scale_stable_typicality_v2",
            "heating_proxy": "max((Tset-Ta)/R - A*qg/1000 - Qint, 0)",
            "typical_day_equivalent_days": 151,
        },
    )
    _write_json(
        model_root / "input_schema.json",
        {
            "required_columns": ["year", "Area", "EnergyLabel"],
            "optional_columns": ["building_id"],
            "allowed_energy_labels": ["A", "B", "C", "D", "E", "F", "G", "unknown", "0"],
            "year_range": [1946, 2005],
            "area_range": [76.0, 217.0],
            "max_buildings": 100,
            "energy_label_role": "required_pass_through_not_predictive",
            "period_support": {
                "1946 - 1964": {"n": 7, "level": "weak"},
                "1965 - 1974": {"n": 31, "level": "core"},
                "1975 - 1991": {"n": 11, "level": "adjacent"},
                "1992 - 2005": {"n": 1, "level": "nearly_unsupported"},
            },
        },
    )

    source_files = [source_draws, source_residuals, source_scales, source_copula, source_config]
    deployed_files = [
        "model/model_metadata.json",
        "model/runtime_config.json",
        "model/input_schema.json",
        "model/parameter_draws.parquet",
        "model/residual_marginals.parquet",
        "model/q_period_scales.parquet",
        "model/copula_correlation.json",
        "weather/typical_weather_scenarios.json",
        "weather/typical_day_weights.json",
    ]
    manifest = {
        "bundle_version": metadata["model_version"],
        "contains_raw_dalen_rows": False,
        "contains_validation_or_test_targets": False,
        "contains_private_identifiers": False,
        "source_hashes": [
            {
                "source": str(path.relative_to(project_root)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in source_files
        ],
        "files": [
            {"path": path, "sha256": _sha256(deploy_root / "artifacts_public" / path)}
            for path in deployed_files
        ],
    }
    _write_json(model_root / "model_manifest.json", manifest)


def main() -> None:
    deploy_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the sanitized online model bundle.")
    parser.add_argument("--project-root", type=Path, default=deploy_root.parent / "Project")
    parser.add_argument("--deploy-root", type=Path, default=deploy_root)
    args = parser.parse_args()
    build_bundle(args.project_root.resolve(), args.deploy_root.resolve())
    print(args.deploy_root.resolve() / "artifacts_public" / "model" / "model_manifest.json")


if __name__ == "__main__":
    main()
