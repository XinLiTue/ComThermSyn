"""
Compatibility exports for the V5 smoothed residual synthesis strategy.

New code should import synthesis names from ``src.thermal_synthesis``.
"""
from src.thermal_synthesis import (
    MIN_RESID_SAMPLES,
    compute_resid_stats,
    compute_residual_statistics,
    evaluate_group_generator_ablation_v5,
    evaluate_group_generator_cv_v5,
    generate_community_rcaq_v5,
    generate_single_building_v5,
    sample_single_building_smoothed_residual,
    synthesize_community_smoothed_residual,
)

__all__ = [
    "MIN_RESID_SAMPLES",
    "compute_residual_statistics",
    "compute_resid_stats",
    "sample_single_building_smoothed_residual",
    "generate_single_building_v5",
    "synthesize_community_smoothed_residual",
    "generate_community_rcaq_v5",
    "evaluate_group_generator_cv_v5",
    "evaluate_group_generator_ablation_v5",
]
