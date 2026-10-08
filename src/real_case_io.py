from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = {
    "case",
    "building_year_distribution",
    "floor_area_distribution_m2",
    "monthly_results",
    "annual_results",
    "comparison",
    "method",
    "provenance",
}


def load_real_case_result(path: str | Path) -> dict[str, Any]:
    """Load and validate a public, aggregate real-community result bundle."""
    result_path = Path(path)
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    missing = REQUIRED_SECTIONS.difference(result)
    if missing:
        raise ValueError(f"Real-case result is missing sections: {sorted(missing)}")

    case = result["case"]
    unit_count = int(case["modelled_residential_units"])
    building_count = int(case["bag_buildings"])
    if unit_count <= 0 or building_count <= 0 or building_count > unit_count:
        raise ValueError("Real-case residential-unit and BAG-building counts are invalid.")

    for distribution_name in (
        "building_year_distribution",
        "floor_area_distribution_m2",
    ):
        distribution = result[distribution_name]
        if not distribution or any("class" not in row or "number" not in row for row in distribution):
            raise ValueError(f"{distribution_name} must contain class and number fields.")
        if sum(int(row["number"]) for row in distribution) != unit_count:
            raise ValueError(f"{distribution_name} does not sum to the residential-unit count.")

    monthly_results = result["monthly_results"]
    if len(monthly_results) != 12:
        raise ValueError("Real-case result must contain exactly 12 monthly rows.")
    if [int(row["month_number"]) for row in monthly_results] != list(range(1, 13)):
        raise ValueError("Real-case monthly rows must be ordered from January to December.")

    required_monthly_fields = {
        "month",
        "heating_energy_mwh_th",
        "hp_electricity_mwh_e",
        "mean_outdoor_temperature_c",
        "min_outdoor_temperature_c",
        "max_outdoor_temperature_c",
    }
    for row in monthly_results:
        if not required_monthly_fields.issubset(row):
            raise ValueError("A real-case monthly row is incomplete.")
        if min(
            float(row["heating_energy_mwh_th"]),
            float(row["hp_electricity_mwh_e"]),
        ) < 0:
            raise ValueError("Monthly heating and electricity must be non-negative.")
        if not (
            float(row["min_outdoor_temperature_c"])
            <= float(row["mean_outdoor_temperature_c"])
            <= float(row["max_outdoor_temperature_c"])
        ):
            raise ValueError("Monthly outdoor-temperature bounds are inconsistent.")

    annual = result["annual_results"]
    annual_heat = float(annual["heating_energy_mwh_th"])
    annual_electricity = float(annual["hp_electricity_mwh_e"])
    if annual_heat <= 0 or annual_electricity <= 0:
        raise ValueError("Annual heating and electricity must be positive.")
    for monthly_field, annual_value in (
        ("heating_energy_mwh_th", annual_heat),
        ("hp_electricity_mwh_e", annual_electricity),
    ):
        if abs(sum(float(row[monthly_field]) for row in monthly_results) - annual_value) > 0.01:
            raise ValueError("Monthly and annual real-case energy totals disagree.")
    if abs(annual_heat / annual_electricity - float(annual["seasonal_performance_factor"])) > 0.001:
        raise ValueError("The annual performance factor disagrees with the energy totals.")

    comparison = result["comparison"]
    required_comparison_fields = {
        "reference_url",
        "reference_name",
        "reference_weather_year",
        "reference_residential_units",
        "reference_annual_heating_mwh_th",
        "sycotherm_normalized_heating_mwh_th",
        "normalized_difference_percent",
    }
    if not required_comparison_fields.issubset(comparison):
        raise ValueError("Real-case comparison metadata is incomplete.")
    reference_units = int(comparison["reference_residential_units"])
    reference_heat = float(comparison["reference_annual_heating_mwh_th"])
    normalized_heat = float(comparison["sycotherm_normalized_heating_mwh_th"])
    if reference_units <= 0 or reference_heat <= 0:
        raise ValueError("The TNO comparison inputs must be positive.")
    if abs(normalized_heat - annual_heat * reference_units / unit_count) > 0.001:
        raise ValueError("The normalized SyCoTherm heating total is inconsistent.")
    expected_difference = 100 * abs(normalized_heat - reference_heat) / reference_heat
    if abs(expected_difference - float(comparison["normalized_difference_percent"])) > 0.005:
        raise ValueError("The normalized heating difference is inconsistent.")

    return result
