from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOY_ROOT))

from src.artifact_io import load_deployment_artifacts
from src.copula_runtime import generate_candidate_communities
from src.deploy_runtime import _typical_day_profiles, run_deployment_synthesis, validate_community_input
from src.lhs_runtime import generate_deployment_community


def sample_roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": ["a", "b", "c"],
            "year": [1958, 1970, 1985],
            "Area": [102.0, 118.0, 96.0],
            "EnergyLabel": ["A", "C", "unknown"],
        }
    )


class OnlineRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = load_deployment_artifacts(DEPLOY_ROOT / "artifacts_public")
        cls.e4_artifacts = load_deployment_artifacts(
            DEPLOY_ROOT / "artifacts_candidates" / "task107_dual_e4_v1"
        )

    def test_public_bundle_is_sanitized_and_frozen(self) -> None:
        manifest = self.artifacts["model_manifest"]
        self.assertFalse(manifest["contains_raw_dalen_rows"])
        self.assertFalse(manifest["contains_validation_or_test_targets"])
        self.assertFalse(manifest["contains_private_identifiers"])
        self.assertEqual(self.artifacts["parameter_draws"]["parameter_draw_id"].nunique(), 50)
        self.assertEqual(set(self.artifacts["residual_marginals"]["component"]), {"G", "C", "A", "Q"})
        self.assertEqual(len(self.artifacts["residual_marginals"]), 200)

    def test_runtime_is_deterministic_and_selects_one_whole_community(self) -> None:
        first = run_deployment_synthesis(sample_roster(), artifacts=self.artifacts)
        second = run_deployment_synthesis(sample_roster(), artifacts=self.artifacts)
        pd.testing.assert_frame_equal(
            first["generated_building_parameters_df"], second["generated_building_parameters_df"]
        )
        selected = first["generated_building_parameters_df"]
        self.assertEqual(len(selected), len(sample_roster()))
        self.assertEqual(selected["selected_realization_id"].nunique(), 1)
        evaluated_count = len(first["candidate_community_summary_df"])
        self.assertGreaterEqual(evaluated_count, 1)
        self.assertLessEqual(evaluated_count, 3)
        self.assertEqual(int(first["candidate_community_summary_df"]["is_selected_typical"].sum()), 1)
        self.assertTrue((selected[["R", "C", "A", "tau_hours"]].to_numpy() > 0).all())
        self.assertEqual(first["runtime_metadata"]["sampling_mode"], "lhs_fast")
        self.assertEqual(first["runtime_metadata"]["planned_probability_designs"], 16)
        self.assertFalse(first["runtime_metadata"]["acceptance_fallback_used"])
        summary = first["candidate_community_summary_df"].loc[
            first["candidate_community_summary_df"]["is_selected_typical"]
        ].iloc[0]
        self.assertLessEqual(summary["c_tail_count"], summary["allowed_c_tail_count"])
        self.assertLessEqual(
            summary["c_upper_tail_count"], summary["allowed_c_upper_tail_count"]
        )

    def test_full_mc_remains_available_as_reference(self) -> None:
        config = dict(self.artifacts["runtime_config"])
        config["sampling_mode"] = "full_mc"
        artifacts = dict(self.artifacts)
        artifacts["runtime_config"] = config
        result = run_deployment_synthesis(sample_roster(), artifacts=artifacts)
        self.assertEqual(result["runtime_metadata"]["sampling_mode"], "full_mc")
        self.assertEqual(len(result["candidate_community_summary_df"]), 100)

    def test_optional_e4_modes_are_deterministic_and_bounded(self) -> None:
        artifacts = self.e4_artifacts
        cleaned = validate_community_input(sample_roster(), input_schema=artifacts["input_schema"])
        for mode in ("e4_rank_lhs_v1", "e4_complete_row_v1"):
            with self.subTest(mode=mode):
                config = {**artifacts["runtime_config"], "sampling_mode": mode}
                first = generate_deployment_community(
                    cleaned, artifacts, run_seed=17, runtime_config=config)
                second = generate_deployment_community(
                    cleaned, artifacts, run_seed=17, runtime_config=config)
                pd.testing.assert_frame_equal(first[0], second[0])
                self.assertEqual(len(first[1]), 200)
                oversized = pd.concat([cleaned] * 17, ignore_index=True).iloc[:51].copy()
                oversized["building_id"] = [f"x{i}" for i in range(51)]
                with self.assertRaisesRegex(ValueError, mode):
                    generate_deployment_community(
                        oversized, artifacts, run_seed=17, runtime_config=config)

    def test_six_building_roster_does_not_fail_probability_gate(self) -> None:
        roster = pd.DataFrame(
            {
                "building_id": ["building_1", "building_2", "building_3", "4", "5", "6"],
                "year": [1985, 2001, 1971, 1980, 1990, 1972],
                "Area": [95.0, 120.0, 82.0, 100.0, 100.0, 90.0],
                "EnergyLabel": ["C", "B", "A", "B", "A", "B"],
            }
        )
        result = run_deployment_synthesis(roster, artifacts=self.artifacts)
        self.assertEqual(len(result["generated_building_parameters_df"]), 6)
        self.assertFalse(result["runtime_metadata"]["acceptance_fallback_used"])
        selected_summary = result["candidate_community_summary_df"].loc[
            result["candidate_community_summary_df"]["is_selected_typical"]
        ].iloc[0]
        self.assertTrue(selected_summary["probability_typicality_pass"])

    def test_energy_label_is_pass_through_not_predictive(self) -> None:
        roster = sample_roster()
        changed = roster.copy()
        changed["EnergyLabel"] = ["G", "F", "B"]
        original = run_deployment_synthesis(roster, artifacts=self.artifacts)
        updated = run_deployment_synthesis(changed, artifacts=self.artifacts)
        columns = ["R", "C", "A", "Qint", "tau_hours"]
        np.testing.assert_allclose(
            original["generated_building_parameters_df"][columns],
            updated["generated_building_parameters_df"][columns],
        )
        self.assertEqual(updated["generated_building_parameters_df"]["EnergyLabel"].tolist(), ["G", "F", "B"])

    def test_hierarchical_seed_keeps_existing_buildings_stable(self) -> None:
        roster = sample_roster()
        cleaned = validate_community_input(roster, input_schema=self.artifacts["input_schema"])
        expanded = pd.concat(
            [
                roster,
                pd.DataFrame(
                    {"building_id": ["d"], "year": [1966], "Area": [88.0], "EnergyLabel": ["D"]}
                ),
            ],
            ignore_index=True,
        )
        expanded = validate_community_input(expanded, input_schema=self.artifacts["input_schema"])
        first = generate_candidate_communities(cleaned, self.artifacts, run_seed=11).rows
        second = generate_candidate_communities(expanded, self.artifacts, run_seed=11).rows
        key = ["community_id", "building_id"]
        columns = key + ["R", "C", "A", "Qint"]
        left = first[columns].sort_values(key).reset_index(drop=True)
        right = second.loc[second["building_id"].isin(roster["building_id"]), columns]
        right = right.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_calibration_support_is_enforced(self) -> None:
        for column, value in [("year", 1945), ("year", 2006), ("Area", 75.0), ("Area", 218.0)]:
            with self.subTest(column=column, value=value):
                bad = sample_roster()
                bad.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_community_input(bad, input_schema=self.artifacts["input_schema"])

    def test_heating_proxy_uses_signed_qint_without_extra_gain_factors(self) -> None:
        buildings = pd.DataFrame({"building_id": ["x"], "R": [2.0], "A": [4.0], "Qint": [-1.0]})
        weather = {
            "scenarios": [
                {
                    "typical_day_id": "fixture",
                    "display_day_label": "fixture",
                    "weight": 1.0,
                    "series": [
                        {
                            "timestep_index": index,
                            "hour": float(index),
                            "outdoor_temperature_C": 10.0,
                            "solar_radiation_W_m2": 100.0,
                            "setpoint_C": 20.0,
                        }
                        for index in range(2)
                    ],
                }
            ]
        }
        profile, _, _ = _typical_day_profiles(buildings, weather, {"typical_day_equivalent_days": 1})
        expected = (20.0 - 10.0) / 2.0 - 4.0 * 100.0 / 1000.0 - (-1.0)
        np.testing.assert_allclose(profile["heat_kw"], expected)


if __name__ == "__main__":
    unittest.main()
