"""
Compatibility exports for the V4 bootstrap residual synthesis strategy.

New code should import synthesis names from ``src.thermal_synthesis``.
"""
from src.feature_engineering import prepare_building_features as enrich_model_df
from src.thermal_synthesis import (
    GroupGenerator,
    ThermalParameterStore,
    estimate_conditional_expected_energy,
    evaluate_case_size_scenarios_bootstrap,
    evaluate_group_generator_ablation,
    evaluate_group_generator_cv,
    fit_group_generator,
    fit_thermal_parameter_store,
    generate_community_rcaq,
    generate_single_building,
    lookup_baseline,
    lookup_energy_stat_record,
    lookup_r_clip_bounds,
    sample_community_latent,
    sample_residual_row,
    sample_shared_scenario_factors,
    sample_single_building_bootstrap,
    synthesize_community_bootstrap,
)

# Retain the original V4 function behavior and default strategy.
evaluate_case_size_scenarios = evaluate_case_size_scenarios_bootstrap

__all__ = [
    "ThermalParameterStore",
    "GroupGenerator",
    "fit_thermal_parameter_store",
    "fit_group_generator",
    "lookup_baseline",
    "lookup_energy_stat_record",
    "lookup_r_clip_bounds",
    "sample_residual_row",
    "sample_shared_scenario_factors",
    "sample_community_latent",
    "sample_single_building_bootstrap",
    "generate_single_building",
    "synthesize_community_bootstrap",
    "generate_community_rcaq",
    "estimate_conditional_expected_energy",
    "evaluate_group_generator_cv",
    "evaluate_group_generator_ablation",
    "evaluate_case_size_scenarios",
    "evaluate_case_size_scenarios_bootstrap",
    "enrich_model_df",
]
