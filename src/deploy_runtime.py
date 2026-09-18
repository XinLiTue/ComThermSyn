from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .copula_runtime import period_from_year
from .lhs_runtime import generate_deployment_community


PRIVATE_INPUT_TOKENS = ("address", "postcode", "user_id", "participant", "target", "truth")


def validate_community_input(
    community_input: pd.DataFrame,
    *,
    input_schema: dict[str, Any],
) -> pd.DataFrame:
    frame = community_input.copy()
    aliases = {
        "construction_year": "year",
        "floor_area_m2": "Area",
        "energy_label": "EnergyLabel",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    lowered = [str(column).lower() for column in frame.columns]
    unsafe = [column for column in lowered if any(token in column for token in PRIVATE_INPUT_TOKENS)]
    if unsafe:
        raise ValueError(f"private or target-like input columns are not accepted: {unsafe}")

    required = list(input_schema["required_columns"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required input columns: {missing}")
    if frame.empty:
        raise ValueError("community input must contain at least one building")
    if len(frame) > int(input_schema["max_buildings"]):
        raise ValueError(f"community input exceeds {input_schema['max_buildings']} buildings")

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    if frame["year"].isna().any() or not np.allclose(frame["year"], np.round(frame["year"])):
        raise ValueError("year must contain finite integer construction years")
    frame["year"] = frame["year"].astype(int)
    year_min, year_max = map(int, input_schema["year_range"])
    if not frame["year"].between(year_min, year_max).all():
        raise ValueError(f"year must be within the calibrated online range {year_min}-{year_max}")

    frame["Area"] = pd.to_numeric(frame["Area"], errors="coerce")
    area_min, area_max = map(float, input_schema["area_range"])
    if frame["Area"].isna().any() or not np.isfinite(frame["Area"]).all():
        raise ValueError("Area must contain finite numeric floor areas")
    if not frame["Area"].between(area_min, area_max).all():
        raise ValueError(f"Area must be within the calibrated online range {area_min:g}-{area_max:g} m2")

    frame["EnergyLabel"] = frame["EnergyLabel"].astype("string").str.strip()
    if frame["EnergyLabel"].isna().any() or frame["EnergyLabel"].eq("").any():
        raise ValueError("EnergyLabel is required even though it is pass-through in this model")
    allowed = {str(value).upper() for value in input_schema["allowed_energy_labels"]}
    normalized = frame["EnergyLabel"].str.upper()
    if not normalized.isin(allowed).all():
        bad = sorted(frame.loc[~normalized.isin(allowed), "EnergyLabel"].unique().tolist())
        raise ValueError(f"unsupported EnergyLabel values: {bad}")
    frame["EnergyLabel"] = normalized.replace({"UNKNOWN": "unknown"})

    if "building_id" not in frame.columns:
        frame["building_id"] = [f"building_{index + 1:03d}" for index in range(len(frame))]
    frame["building_id"] = frame["building_id"].astype("string").str.strip()
    missing_ids = frame["building_id"].isna() | frame["building_id"].eq("")
    frame.loc[missing_ids, "building_id"] = [
        f"building_{index + 1:03d}" for index in frame.index[missing_ids]
    ]
    if frame["building_id"].duplicated().any():
        raise ValueError("building_id values must be unique")
    if not frame["building_id"].map(lambda value: bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))).all():
        raise ValueError("building_id may only contain letters, numbers, dots, underscores, and hyphens")

    frame["period"] = frame["year"].map(period_from_year)
    support = input_schema["period_support"]
    frame["support_level"] = frame["period"].map(lambda period: support[period]["level"])
    return frame[["building_id", "year", "Area", "EnergyLabel", "period", "support_level"]].reset_index(
        drop=True
    )


def _typical_day_profiles(
    buildings: pd.DataFrame,
    weather_payload: dict[str, Any],
    runtime_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile_rows: list[dict[str, Any]] = []
    annual_energy = 0.0
    equivalent_days = float(runtime_config["typical_day_equivalent_days"])
    community_series: list[pd.DataFrame] = []

    for scenario in weather_payload["scenarios"]:
        series = pd.DataFrame(scenario["series"])
        if len(series) < 2:
            raise ValueError("weather scenario requires at least two timesteps")
        timestep_hours = float(np.median(np.diff(series["hour"].to_numpy(dtype=float))))
        day_weight = float(scenario["weight"])
        for building in buildings.itertuples(index=False):
            envelope_kw = (series["setpoint_C"] - series["outdoor_temperature_C"]) / float(
                building.R
            )
            solar_kw = float(building.A) * series["solar_radiation_W_m2"] / 1000.0
            heat_kw = np.maximum(envelope_kw - solar_kw - float(building.Qint), 0.0)
            for weather_row, demand in zip(series.itertuples(index=False), heat_kw, strict=True):
                profile_rows.append(
                    {
                        "typical_day_id": scenario["typical_day_id"],
                        "display_day_label": scenario["display_day_label"],
                        "weight": day_weight,
                        "timestep_index": int(weather_row.timestep_index),
                        "hour": float(weather_row.hour),
                        "outdoor_temperature_C": float(weather_row.outdoor_temperature_C),
                        "solar_radiation_W_m2": float(weather_row.solar_radiation_W_m2),
                        "setpoint_C": float(weather_row.setpoint_C),
                        "building_id": building.building_id,
                        "heat_kw": float(demand),
                    }
                )
            annual_energy += float(np.sum(heat_kw) * timestep_hours * day_weight * equivalent_days)

        scenario_profile = pd.DataFrame(profile_rows)
        scenario_profile = scenario_profile.loc[
            scenario_profile["typical_day_id"].eq(scenario["typical_day_id"])
        ]
        community = scenario_profile.groupby(
            ["typical_day_id", "timestep_index"], as_index=False
        )["heat_kw"].sum()
        community["weight"] = day_weight
        community_series.append(community)

    profiles = pd.DataFrame(profile_rows)
    combined = pd.concat(community_series, ignore_index=True)
    peak_kw = float(combined["heat_kw"].max())
    ramps = combined.groupby("typical_day_id")["heat_kw"].diff().abs()
    max_ramp = float(ramps.max())
    annual = pd.DataFrame(
        [
            {
                "metric": "total_annual_heating_energy",
                "value": annual_energy,
                "unit": "kWh",
                "interpretation": "typical-weather winter heating-demand proxy; not measured energy",
            },
            {
                "metric": "typical_day_proxy_annual_heating_energy",
                "value": annual_energy,
                "unit": "kWh",
                "interpretation": "weighted typical days multiplied by equivalent winter days",
            },
        ]
    )
    dynamic = pd.DataFrame(
        [
            {"metric": "peak_heat_demand", "value": peak_kw, "unit": "kW"},
            {"metric": "max_ramp_per_timestep", "value": max_ramp, "unit": "kW/step"},
            {
                "metric": "n_typical_days",
                "value": len(weather_payload["scenarios"]),
                "unit": "count",
            },
        ]
    )
    return profiles, annual, dynamic


def run_deployment_synthesis(
    community_input: pd.DataFrame,
    *,
    artifacts: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = runtime_config or artifacts["runtime_config"]
    cleaned = validate_community_input(community_input, input_schema=artifacts["input_schema"])
    selected, candidate_summaries, selection = generate_deployment_community(
        cleaned,
        artifacts,
        run_seed=int(config["default_run_seed"]),
        runtime_config=config,
    )
    profiles, annual, dynamic = _typical_day_profiles(
        selected, artifacts["weather_scenarios"], config
    )
    support_counts = cleaned["support_level"].value_counts().to_dict()
    warnings = [
        "Development candidate only: no address-level, population, privacy, or measured-energy claim.",
        "EnergyLabel is preserved in output but does not alter RCAQ in this frozen model.",
        "Annual energy and peak values are typical-weather heating-demand proxies.",
    ]
    if selection.get("acceptance_fallback_used"):
        warnings.append(
            "No LHS design passed every preferred typicality gate; the closest joint-valid design was used."
        )
    if support_counts.get("nearly_unsupported", 0):
        warnings.append("At least one 1992-2005 building uses nearly unsupported calibration evidence (n=1).")
    return {
        "cleaned_input_df": cleaned,
        "generated_building_parameters_df": selected,
        "community_annual_energy_summary_df": annual,
        "typical_day_community_profile_df": profiles,
        "community_dynamic_metrics_df": dynamic,
        "candidate_community_summary_df": candidate_summaries,
        "warnings": warnings,
        "runtime_metadata": {
            **selection,
            "model_version": artifacts["model_metadata"]["model_version"],
            "payload_type": artifacts["model_metadata"]["payload_type"],
            "synthesis_mode": f"standalone_shrunk_copula_{selection['sampling_mode']}",
            "support_counts": support_counts,
            "energy_label_effect_used": False,
        },
    }


def build_streamlit_output_package(result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    inputs_dir = run_dir / "inputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    result["cleaned_input_df"].to_csv(inputs_dir / "community_input_validated.csv", index=False)
    result["generated_building_parameters_df"].to_csv(
        results_dir / "generated_building_parameters.csv", index=False
    )
    result["community_annual_energy_summary_df"].to_csv(
        results_dir / "community_annual_energy_summary.csv", index=False
    )
    result["typical_day_community_profile_df"].to_csv(
        results_dir / "typical_day_community_profile.csv", index=False
    )
    result["community_dynamic_metrics_df"].to_csv(
        results_dir / "community_dynamic_metrics.csv", index=False
    )
    result["candidate_community_summary_df"].to_csv(
        results_dir / "candidate_community_summary.csv", index=False
    )
    metadata = result["runtime_metadata"]
    manifest = {
        **metadata,
        "n_input_buildings": int(len(result["cleaned_input_df"])),
        "n_generated_buildings": int(len(result["generated_building_parameters_df"])),
        "warnings": result["warnings"],
        "files": [
            "inputs/community_input_validated.csv",
            "results/generated_building_parameters.csv",
            "results/community_annual_energy_summary.csv",
            "results/typical_day_community_profile.csv",
            "results/community_dynamic_metrics.csv",
            "results/candidate_community_summary.csv",
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
