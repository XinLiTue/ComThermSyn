from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from src.real_case_io import load_real_case_result


class RealCaseResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = load_real_case_result(
            DEPLOY_ROOT / "real_cases" / "ballum_real_case_result.json"
        )

    def test_scope_counts_and_distributions_are_consistent(self) -> None:
        case = self.result["case"]
        self.assertEqual(case["modelled_residential_units"], 212)
        self.assertEqual(case["bag_buildings"], 192)
        self.assertEqual(
            sum(row["number"] for row in self.result["building_year_distribution"]),
            212,
        )
        self.assertEqual(
            sum(row["number"] for row in self.result["floor_area_distribution_m2"]),
            212,
        )

    def test_monthly_energy_matches_annual_totals_with_rounding_tolerance(self) -> None:
        monthly = self.result["monthly_results"]
        annual = self.result["annual_results"]
        for row in monthly[5:8]:
            self.assertEqual(row["heating_energy_mwh_th"], 0.0)
            self.assertEqual(row["hp_electricity_mwh_e"], 0.0)
        self.assertAlmostEqual(
            sum(row["heating_energy_mwh_th"] for row in monthly),
            annual["heating_energy_mwh_th"],
            delta=0.01,
        )
        self.assertAlmostEqual(
            sum(row["hp_electricity_mwh_e"] for row in monthly),
            annual["hp_electricity_mwh_e"],
            delta=0.01,
        )
        self.assertAlmostEqual(
            annual["heating_energy_mwh_th"] / annual["hp_electricity_mwh_e"],
            annual["seasonal_performance_factor"],
            delta=0.001,
        )

    def test_comparison_uses_matching_residential_unit_counts(self) -> None:
        comparison = self.result["comparison"]
        self.assertEqual(
            comparison["reference_url"],
            "https://www.warmteprofielengenerator.nl/results/6045",
        )
        self.assertEqual(comparison["reference_weather_year"], 2019)
        self.assertEqual(comparison["reference_residential_units"], 177)
        self.assertAlmostEqual(
            comparison["reference_annual_heating_mwh_th"], 2180.089, places=3
        )
        sycotherm_scaled = (
            self.result["annual_results"]["heating_energy_mwh_th"]
            * comparison["reference_residential_units"]
            / self.result["case"]["modelled_residential_units"]
        )
        self.assertAlmostEqual(
            comparison["sycotherm_normalized_heating_mwh_th"],
            sycotherm_scaled,
            delta=0.001,
        )
        expected_difference = (
            100
            * abs(sycotherm_scaled - comparison["reference_annual_heating_mwh_th"])
            / comparison["reference_annual_heating_mwh_th"]
        )
        self.assertAlmostEqual(
            comparison["normalized_difference_percent"],
            expected_difference,
            delta=0.005,
        )

    def test_cop_method_and_research_reference_are_explicit(self) -> None:
        method = self.result["method"]
        self.assertIn("20.3595", method["cop_formula"])
        self.assertIn("40", method["cop_formula"])
        self.assertEqual(method["electricity_formula"], "P_HP = Q_heat / COP")
        self.assertEqual(
            method["research_url"],
            "https://ieeexplore.ieee.org/abstract/document/11169153/",
        )


if __name__ == "__main__":
    unittest.main()
