"""Notebook-independent online demo deployment runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifact_io import remove_private_columns, save_json, validate_public_artifact_safety


PRIVATE_INPUT_TERMS = (
    "user_id",
    "participant_id",
    "source_user_id",
    "address",
    "postcode",
    "postal",
    "zip",
)


def default_input_schema() -> dict[str, Any]:
    return {
        "required_columns": ["year", "Area", "EnergyLabel"],
        "optional_columns": ["building_id"],
        "allowed_energy_labels": ["A", "B", "C", "D", "E", "F", "G", "unknown", "0"],
        "year_range": [1900, 2025],
        "area_range": [20, 400],
        "max_buildings": 100,
    }


def _private_columns(df: pd.DataFrame) -> list[str]:
    lower_terms = tuple(term.lower() for term in PRIVATE_INPUT_TERMS)
    return [
        column
        for column in df.columns
        if any(term in str(column).lower() for term in lower_terms)
    ]


def validate_community_input(
    input_df: pd.DataFrame,
    input_schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate public user-provided community input and return safe columns."""
    schema = input_schema or default_input_schema()
    if not isinstance(input_df, pd.DataFrame) or input_df.empty:
        raise ValueError("Community input is empty.")
    unsafe_columns = _private_columns(input_df)
    if unsafe_columns:
        raise ValueError(f"Community input contains private columns: {unsafe_columns}.")
    missing = [column for column in schema["required_columns"] if column not in input_df.columns]
    if missing:
        raise ValueError(f"Community input is missing required columns: {missing}.")
    max_buildings = int(schema.get("max_buildings", 100))
    if len(input_df) > max_buildings:
        raise ValueError(f"Community input has {len(input_df)} buildings; maximum is {max_buildings}.")

    cleaned = input_df.copy()
    if "building_id" not in cleaned.columns:
        cleaned["building_id"] = [f"building_{index + 1}" for index in range(len(cleaned))]
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["Area"] = pd.to_numeric(cleaned["Area"], errors="coerce")
    if cleaned["year"].isna().any():
        raise ValueError("Community input has non-numeric or missing year values.")
    if cleaned["Area"].isna().any():
        raise ValueError("Community input has non-numeric or missing Area values.")
    min_year, max_year = schema.get("year_range", [1900, 2025])
    min_area, max_area = schema.get("area_range", [20, 400])
    if not cleaned["year"].between(min_year, max_year).all():
        raise ValueError(f"Community input year must be between {min_year} and {max_year}.")
    if not cleaned["Area"].between(min_area, max_area).all():
        raise ValueError(f"Community input Area must be between {min_area} and {max_area}.")
    cleaned["EnergyLabel"] = cleaned["EnergyLabel"].astype(str).str.strip()
    allowed = {str(label).lower() for label in schema.get("allowed_energy_labels", [])}
    invalid = sorted(
        label for label in cleaned["EnergyLabel"].dropna().unique()
        if str(label).lower() not in allowed
    )
    if invalid:
        raise ValueError(f"Community input has unsupported EnergyLabel values: {invalid}.")
    return cleaned[["building_id", "year", "Area", "EnergyLabel"]].copy()


def _label_score(label: str) -> float:
    scores = {"A": 0.75, "B": 0.9, "C": 1.0, "D": 1.15, "E": 1.3, "F": 1.5, "G": 1.7}
    return scores.get(str(label).upper(), 1.2)


def _mock_building_parameters(cleaned: pd.DataFrame, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    work = cleaned.copy()
    age_factor = np.clip((2025 - work["year"].astype(float)) / 100.0, 0.0, 1.5)
    label_factor = work["EnergyLabel"].map(_label_score).astype(float)
    area = work["Area"].astype(float)
    noise = rng.normal(0.0, 0.03, size=(len(work), 4))
    work["R"] = np.maximum(0.6, (2.0 + 1.2 * age_factor) * label_factor * (1.0 + noise[:, 0]))
    work["C"] = np.maximum(0.8, (1.2 + 0.012 * area) * (1.0 + noise[:, 1]))
    work["A"] = np.maximum(0.5, (0.018 * area) * (1.0 + noise[:, 2]))
    work["Qint"] = np.maximum(0.1, (0.0045 * area) * (1.0 + noise[:, 3]))
    return work[["building_id", "year", "Area", "EnergyLabel", "R", "C", "A", "Qint"]]


def _mock_annual_energy(generated_df: pd.DataFrame) -> pd.Series:
    label_factor = generated_df["EnergyLabel"].map(_label_score).astype(float)
    age_factor = np.clip((2025 - generated_df["year"].astype(float)) / 100.0, 0.0, 1.5)
    return generated_df["Area"].astype(float) * (35.0 + 42.0 * label_factor + 25.0 * age_factor)


def _typical_day_labels(runtime_config: dict[str, Any]) -> list[str]:
    return list(runtime_config.get("output_typical_days") or ["Cold cloudy day", "Mild cloudy day", "Mild sunny day"])


def _mock_profile(
    generated_df: pd.DataFrame,
    runtime_config: dict[str, Any],
) -> pd.DataFrame:
    annual_energy = _mock_annual_energy(generated_df)
    community_scale = float(annual_energy.sum() / max(annual_energy.mean(), 1.0))
    rows: list[dict[str, Any]] = []
    for day_index, label in enumerate(_typical_day_labels(runtime_config), start=1):
        day_factor = 1.35 if "Cold" in label else 0.85 if "Mild" in label else 1.0
        daylight_factor = 0.78 if "sunny" in label.lower() else 1.0
        previous_heat = 0.0
        for timestep in range(96):
            hour = timestep / 4.0
            morning_peak = np.exp(-0.5 * ((hour - 7.0) / 2.2) ** 2)
            evening_peak = np.exp(-0.5 * ((hour - 19.0) / 2.8) ** 2)
            base = 0.8 + 1.6 * morning_peak + 1.2 * evening_peak
            heat_kw = float(community_scale * day_factor * daylight_factor * base)
            rows.append(
                {
                    "typical_day_id": f"day_{day_index}",
                    "display_day_label": label,
                    "timestep_index": timestep,
                    "hour": hour,
                    "synthesized_heat_kw": heat_kw,
                    "synthesized_ramp_kw": heat_kw - previous_heat,
                }
            )
            previous_heat = heat_kw
    return pd.DataFrame(rows)


def _artifact_runtime_mode(artifacts: dict[str, Any], runtime_config: dict[str, Any]) -> str:
    payload = artifacts.get("deployment_model_payload") or {}
    if payload.get("payload_type") == "public_sampler_v1":
        return "public_sampler_v1"
    if runtime_config.get("strategy") == "public_sampler_v1":
        return "public_sampler_v1"
    return "mock_runtime"


def _vintage_from_year(year: float) -> str:
    year = float(year)
    if year < 1945:
        return "pre_1945"
    if year < 1975:
        return "1945_1974"
    if year < 1992:
        return "1975_1991"
    if year < 2006:
        return "1992_2005"
    if year < 2015:
        return "2006_2014"
    return "2015_plus"


def _lookup_public_row(
    table: pd.DataFrame,
    *,
    vintage: str,
    label: str,
) -> pd.Series:
    if table.empty:
        raise ValueError("Public sampler artifact table is empty.")
    label = str(label).upper()
    candidates = [
        table[(table.get("vintage") == vintage) & (table.get("EnergyLabel").astype(str).str.upper() == label)]
        if "vintage" in table.columns and "EnergyLabel" in table.columns else pd.DataFrame(),
        table[table.get("vintage") == vintage] if "vintage" in table.columns else pd.DataFrame(),
        table[table.get("EnergyLabel").astype(str).str.upper() == label] if "EnergyLabel" in table.columns else pd.DataFrame(),
        table[table.get("fallback_level").astype(str).str.lower() == "global"] if "fallback_level" in table.columns else pd.DataFrame(),
        table,
    ]
    for candidate in candidates:
        if len(candidate):
            return candidate.iloc[0]
    return table.iloc[0]


def _public_sample_building_parameters(
    cleaned: pd.DataFrame,
    artifacts: dict[str, Any],
    runtime_config: dict[str, Any],
) -> pd.DataFrame:
    baselines = artifacts.get("public_sampling_baselines")
    residual_pool = artifacts.get("synthetic_residual_pool")
    if not isinstance(baselines, pd.DataFrame) or baselines.empty:
        raise ValueError("public_sampler_v1 artifacts are missing public_sampling_baselines.")
    if not isinstance(residual_pool, pd.DataFrame) or residual_pool.empty:
        raise ValueError("public_sampler_v1 artifacts are missing synthetic_residual_pool.")

    rng = np.random.default_rng(int(runtime_config.get("random_state", 42)))
    rows: list[dict[str, Any]] = []
    residual_groups = residual_pool["residual_group"].astype(str) if "residual_group" in residual_pool.columns else pd.Series(["global"] * len(residual_pool))
    for _, input_row in cleaned.iterrows():
        vintage = _vintage_from_year(input_row["year"])
        baseline = _lookup_public_row(baselines, vintage=vintage, label=input_row["EnergyLabel"])
        residual_group = str(baseline.get("residual_group", baseline.get("vintage", "global")))
        group_pool = residual_pool[residual_groups == residual_group]
        if group_pool.empty:
            group_pool = residual_pool
        residual = group_pool.iloc[int(rng.integers(0, len(group_pool)))]
        area = float(input_row["Area"])
        c_per_area = float(baseline.get("C_per_area_median", baseline.get("C_median", 1.8)))
        row = {
            "building_id": input_row["building_id"],
            "year": int(input_row["year"]),
            "Area": area,
            "EnergyLabel": str(input_row["EnergyLabel"]),
            "R": max(0.05, float(baseline.get("R_median", 2.5)) + float(residual.get("R_resid", 0.0))),
            "C": max(0.05, (c_per_area + float(residual.get("C_per_area_resid", 0.0))) * area),
            "A": max(0.01, float(baseline.get("A_median", 2.0)) + float(residual.get("A_resid", 0.0))),
            "Qint": max(0.01, float(baseline.get("Qint_median", 0.5)) + float(residual.get("Qint_resid", 0.0))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _scenario_rows_from_artifacts(artifacts: dict[str, Any], runtime_config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = artifacts.get("typical_weather_scenarios") or {}
    rows = scenarios.get("scenarios") or scenarios.get("scenario_table") or []
    if rows and isinstance(rows[0], dict) and "series" in rows[0]:
        return rows
    labels = _typical_day_labels(runtime_config)
    defaults: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        series = []
        for timestep in range(96):
            hour = timestep / 4.0
            is_cold = "cold" in label.lower()
            is_sunny = "sunny" in label.lower()
            temp_base = 1.0 if is_cold else 8.0
            solar_peak = 250.0 if is_sunny else 60.0
            series.append(
                {
                    "timestep_index": timestep,
                    "hour": hour,
                    "outdoor_temperature_C": temp_base + 3.0 * np.sin((hour - 7.0) / 24.0 * 2.0 * np.pi),
                    "solar_radiation_W_m2": max(0.0, solar_peak * np.sin((hour - 7.0) / 12.0 * np.pi)),
                    "setpoint_C": 20.0,
                }
            )
        defaults.append({"typical_day_id": f"day_{index}", "display_day_label": label, "weight": 1.0 / len(labels), "series": series})
    return defaults


def _weight_days(weight: float, weights_are_probabilities: bool = True) -> float:
    return float(weight) * 365.0 if weights_are_probabilities else float(weight)


def _public_typical_day_profile(
    generated_df: pd.DataFrame,
    artifacts: dict[str, Any],
    runtime_config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    scenario_rows = _scenario_rows_from_artifacts(artifacts, runtime_config)
    weights_meta = artifacts.get("typical_day_weights") or {}
    weight_map = weights_meta.get("weights_by_day", {}) if isinstance(weights_meta, dict) else {}
    weights_are_probabilities = bool(weights_meta.get("weights_are_probabilities", True)) if isinstance(weights_meta, dict) else True
    profile_rows: list[dict[str, Any]] = []
    annual_kwh = 0.0
    for scenario_index, scenario in enumerate(scenario_rows, start=1):
        typical_day_id = str(scenario.get("typical_day_id") or scenario.get("day_id") or f"day_{scenario_index}")
        label = str(scenario.get("display_day_label") or scenario.get("scenario_label") or typical_day_id)
        weight = float(scenario.get("weight", weight_map.get(typical_day_id, 1.0 / max(len(scenario_rows), 1))))
        series = scenario.get("series") or []
        previous_heat = 0.0
        day_kwh = 0.0
        for step_index, weather_row in enumerate(series):
            hour = float(weather_row.get("hour", step_index / 4.0))
            ta = float(weather_row.get("outdoor_temperature_C", weather_row.get("ta", 8.0)))
            qg = float(weather_row.get("solar_radiation_W_m2", weather_row.get("qg", 0.0)))
            setpoint = float(weather_row.get("setpoint_C", 20.0))
            heating_kw = 0.0
            for _, building in generated_df.iterrows():
                envelope_kw = max((setpoint - ta) / max(float(building["R"]), 0.05), 0.0)
                solar_offset_kw = max(float(building["A"]) * qg / 1000.0, 0.0)
                internal_offset_kw = max(float(building["Qint"]), 0.0)
                heating_kw += max(envelope_kw - 0.35 * solar_offset_kw - 0.25 * internal_offset_kw, 0.0)
            ramp_kw = heating_kw - previous_heat
            profile_rows.append(
                {
                    "typical_day_id": typical_day_id,
                    "display_day_label": label,
                    "timestep_index": int(weather_row.get("timestep_index", step_index)),
                    "hour": hour,
                    "synthesized_heat_kw": float(heating_kw),
                    "synthesized_ramp_kw": float(ramp_kw),
                    "typical_day_weight": weight,
                }
            )
            day_kwh += heating_kw * 0.25
            previous_heat = heating_kw
        annual_kwh += day_kwh * _weight_days(weight, weights_are_probabilities)
    return pd.DataFrame(profile_rows), float(annual_kwh)


def _public_expected_annual_energy(
    generated_df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> float | None:
    baseline = artifacts.get("expected_energy_baseline")
    if not isinstance(baseline, pd.DataFrame) or baseline.empty:
        return None
    total = 0.0
    for _, building in generated_df.iterrows():
        vintage = _vintage_from_year(building["year"])
        row = _lookup_public_row(baseline, vintage=vintage, label=building["EnergyLabel"])
        energy_per_m2 = float(
            row.get(
                "median_E_std_annual_per_m2",
                row.get("E_std_annual_per_m2_median", row.get("annual_kwh_per_m2_mean", np.nan)),
            )
        )
        if not np.isfinite(energy_per_m2):
            return None
        total += energy_per_m2 * float(building["Area"])
    return float(total)


def _run_public_sampler_v1(
    cleaned_input: pd.DataFrame,
    artifacts: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    generated = _public_sample_building_parameters(cleaned_input, artifacts, runtime_config)
    profile, typical_annual_kwh = _public_typical_day_profile(generated, artifacts, runtime_config)
    expected_annual_kwh = _public_expected_annual_energy(generated, artifacts)
    total_kwh = expected_annual_kwh if expected_annual_kwh is not None else typical_annual_kwh
    peak_kw = float(profile["synthesized_heat_kw"].max()) if len(profile) else 0.0
    max_ramp = float(profile["synthesized_ramp_kw"].abs().max()) if len(profile) else 0.0
    annual_summary = pd.DataFrame(
        [
            {"metric": "total_annual_heating_energy", "value": total_kwh, "unit": "kWh"},
            {"metric": "typical_day_proxy_annual_heating_energy", "value": typical_annual_kwh, "unit": "kWh"},
            {"metric": "mean_annual_heating_energy_per_building", "value": total_kwh / len(generated), "unit": "kWh/building"},
            {"metric": "n_buildings", "value": float(len(generated)), "unit": "count"},
        ]
    )
    dynamic_metrics = pd.DataFrame(
        [
            {
                "metric": "peak_heat_demand",
                "value": peak_kw,
                "unit": "kW",
                "display_label": "Peak heat demand",
                "description": "Maximum synthesized community heat demand across public typical days.",
            },
            {
                "metric": "max_ramp_per_timestep",
                "value": max_ramp,
                "unit": "kW/timestep",
                "display_label": "Maximum ramp",
                "description": "Maximum absolute change in synthesized heat demand between adjacent timesteps.",
            },
            {
                "metric": "n_typical_days",
                "value": float(profile["typical_day_id"].nunique()) if len(profile) else 0.0,
                "unit": "count",
                "display_label": "Typical days",
                "description": "Number of public typical-day profiles produced for display.",
            },
        ]
    )
    return {
        "cleaned_input_df": cleaned_input,
        "generated_building_parameters_df": generated,
        "community_annual_energy_summary_df": annual_summary,
        "typical_day_community_profile_df": profile,
        "community_dynamic_metrics_df": dynamic_metrics,
        "warnings": [
            "Public sampler runtime: online approximation using public-safe aggregate artifacts, not research validation."
        ],
        "runtime_metadata": {
            "mock_runtime": False,
            "payload_type": "public_sampler_v1",
            "synthesis_mode": "public_deployment_runtime",
            "private_data_included": False,
            "contains_user_ids": False,
            "contains_truth_tables": False,
            "random_state": int(runtime_config.get("random_state", 42)),
            "artifact_root": artifacts.get("artifact_root"),
            "artifact_version": artifacts.get("model_metadata", {}).get("artifact_version", "public_sampler_v1"),
        },
    }


def run_deployment_synthesis(
    community_inputs: pd.DataFrame,
    artifacts: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deployment synthesis from mock or ``public_sampler_v1`` artifacts."""
    config = dict(artifacts.get("runtime_config", {}))
    if runtime_config:
        config.update(runtime_config)
    schema = artifacts.get("input_schema") or default_input_schema()
    cleaned_input = validate_community_input(community_inputs, schema)
    runtime_mode = _artifact_runtime_mode(artifacts, config)
    if runtime_mode == "public_sampler_v1":
        return _run_public_sampler_v1(cleaned_input, artifacts, config)

    random_state = int(config.get("random_state", 42))
    generated = _mock_building_parameters(cleaned_input, random_state=random_state)
    annual_kwh = _mock_annual_energy(generated)
    profile = _mock_profile(generated, config)
    total_kwh = float(annual_kwh.sum())
    peak_kw = float(profile["synthesized_heat_kw"].max()) if len(profile) else 0.0
    max_ramp = float(profile["synthesized_ramp_kw"].abs().max()) if len(profile) else 0.0
    annual_summary = pd.DataFrame(
        [
            {"metric": "total_annual_heating_energy", "value": total_kwh, "unit": "kWh"},
            {"metric": "mean_annual_heating_energy_per_building", "value": total_kwh / len(generated), "unit": "kWh/building"},
            {"metric": "n_buildings", "value": float(len(generated)), "unit": "count"},
        ]
    )
    dynamic_metrics = pd.DataFrame(
        [
            {
                "metric": "peak_heat_demand",
                "value": peak_kw,
                "unit": "kW",
                "display_label": "Peak heat demand",
                "description": "Maximum synthesized community heat demand across mock typical days.",
            },
            {
                "metric": "max_ramp_per_timestep",
                "value": max_ramp,
                "unit": "kW/timestep",
                "display_label": "Maximum ramp",
                "description": "Maximum absolute change in synthesized heat demand between adjacent timesteps.",
            },
            {
                "metric": "n_typical_days",
                "value": float(profile["typical_day_id"].nunique()),
                "unit": "count",
                "display_label": "Typical days",
                "description": "Number of mock typical-day profiles produced for display.",
            },
        ]
    )
    warnings_list = [
        "Mock runtime: outputs are for deployment skeleton validation only, not research validation or final synthesis."
    ]
    return {
        "cleaned_input_df": cleaned_input,
        "generated_building_parameters_df": generated,
        "community_annual_energy_summary_df": annual_summary,
        "typical_day_community_profile_df": profile,
        "community_dynamic_metrics_df": dynamic_metrics,
        "warnings": warnings_list,
        "runtime_metadata": {
            "mock_runtime": True,
            "payload_type": "mock_runtime",
            "synthesis_mode": "mock_deployment_runtime",
            "random_state": random_state,
            "artifact_root": artifacts.get("artifact_root"),
            "artifact_version": artifacts.get("model_metadata", {}).get("artifact_version", "mock_step1"),
        },
    }


def create_runtime_manifest(
    *,
    result: dict[str, Any],
    output_files: dict[str, str],
    runtime_config_used: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = result.get("runtime_metadata", {})
    default_prefix = "deployment_mock" if bool(metadata.get("mock_runtime", True)) else "deployment_public"
    run_id = run_id or f"{default_prefix}_{created_at.replace(':', '').replace('-', '')[:15]}"
    generated = result["generated_building_parameters_df"]
    cleaned = result["cleaned_input_df"]
    return {
        "run_id": run_id,
        "created_at": created_at,
        "artifact_version": metadata.get("artifact_version", "mock_step1"),
        "mock_runtime": bool(metadata.get("mock_runtime", True)),
        "payload_type": metadata.get("payload_type", "mock_runtime"),
        "private_data_included": bool(metadata.get("private_data_included", False)),
        "contains_user_ids": bool(metadata.get("contains_user_ids", False)),
        "contains_truth_tables": bool(metadata.get("contains_truth_tables", False)),
        "synthesis_mode": metadata.get("synthesis_mode", "mock_deployment_runtime"),
        "n_input_buildings": int(len(cleaned)),
        "n_generated_buildings": int(len(generated)),
        "runtime_config_used": runtime_config_used,
        "output_files": output_files,
        "warnings": result.get("warnings", []),
        "privacy_note": (
            "This runtime output is generated from public-safe deployment artifacts and user-provided inputs. "
            "It does not include private training user identifiers or truth tables."
        ),
    }


def _save_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_df = remove_private_columns(df)
    validate_public_artifact_safety(safe_df)
    safe_df.to_csv(path, index=False)


def build_streamlit_output_package(result: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Save Streamlit-ready result tables and manifest."""
    output_root = Path(output_root)
    output_files = {
        "inputs/community_input.csv": output_root / "inputs" / "community_input.csv",
        "results/generated_building_parameters.csv": output_root / "results" / "generated_building_parameters.csv",
        "results/community_annual_energy_summary.csv": output_root / "results" / "community_annual_energy_summary.csv",
        "results/typical_day_community_profile.csv": output_root / "results" / "typical_day_community_profile.csv",
        "results/community_dynamic_metrics.csv": output_root / "results" / "community_dynamic_metrics.csv",
        "plot_data/community_annual_energy.csv": output_root / "plot_data" / "community_annual_energy.csv",
        "plot_data/typical_day_profiles.csv": output_root / "plot_data" / "typical_day_profiles.csv",
        "plot_data/community_dynamic_metrics.csv": output_root / "plot_data" / "community_dynamic_metrics.csv",
    }
    _save_df(output_files["inputs/community_input.csv"], result["cleaned_input_df"])
    _save_df(output_files["results/generated_building_parameters.csv"], result["generated_building_parameters_df"])
    _save_df(output_files["results/community_annual_energy_summary.csv"], result["community_annual_energy_summary_df"])
    _save_df(output_files["results/typical_day_community_profile.csv"], result["typical_day_community_profile_df"])
    _save_df(output_files["results/community_dynamic_metrics.csv"], result["community_dynamic_metrics_df"])
    _save_df(output_files["plot_data/community_annual_energy.csv"], result["community_annual_energy_summary_df"])
    _save_df(output_files["plot_data/typical_day_profiles.csv"], result["typical_day_community_profile_df"])
    _save_df(output_files["plot_data/community_dynamic_metrics.csv"], result["community_dynamic_metrics_df"])

    plot_manifest = {
        "plot_data_files": {
            key: str(path.relative_to(output_root)).replace("\\", "/")
            for key, path in output_files.items()
            if key.startswith("plot_data/")
        }
    }
    save_json(output_root / "plot_data" / "plot_data_manifest.json", plot_manifest)
    output_files["plot_data/plot_data_manifest.json"] = output_root / "plot_data" / "plot_data_manifest.json"
    output_files["manifest.json"] = output_root / "manifest.json"

    runtime_config_used = result.get("runtime_metadata", {})
    manifest = create_runtime_manifest(
        result=result,
        runtime_config_used=runtime_config_used,
        output_files={
            key: str(path.relative_to(output_root)).replace("\\", "/")
            for key, path in output_files.items()
        },
    )
    validate_public_artifact_safety(manifest)
    save_json(output_root / "manifest.json", manifest)
    manifest["output_root"] = str(output_root)
    return manifest
