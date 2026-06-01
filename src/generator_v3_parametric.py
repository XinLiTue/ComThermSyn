"""
Compatibility exports for the legacy V3 parametric synthesis strategy.

New code should import the clearer synthesis names from ``src.thermal_synthesis``.
"""
from src.thermal_synthesis import (
    ParametricSamplerModel,
    ParamGeneratorModel,
    evaluate_generator_cv,
    fit_parametric_sampler,
    fit_parametric_generator,
    generate_buildings_batch,
    generate_rcaq_for_building,
    latent_to_params,
    sample_candidate_latents,
    sample_single_building_parametric,
    synthesize_buildings_parametric_batch,
)

__all__ = [
    "ParametricSamplerModel",
    "ParamGeneratorModel",
    "fit_parametric_sampler",
    "fit_parametric_generator",
    "sample_candidate_latents",
    "latent_to_params",
    "sample_single_building_parametric",
    "generate_rcaq_for_building",
    "synthesize_buildings_parametric_batch",
    "generate_buildings_batch",
    "evaluate_generator_cv",
]
