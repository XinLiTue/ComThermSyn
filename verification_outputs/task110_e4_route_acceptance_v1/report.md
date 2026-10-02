# Task110 E4 Route Acceptance

## Acceptance result

All four current public routes execute through the same public runtime entry point. The public default is `e4_rank_lhs_v1` and was not changed. No production runtime or sampler code was modified.

## Task109 tradeoff

Task109 remains the authoritative comparative evidence. Rankings are metric-dependent; there is no universal winner. E4 is a representative-community designer, not an iid uncertainty ensemble.

```csv
sampling_route,mean_macro_standardized_absolute_error,mean_seven_coordinate_summary_distance,mean_calendar_year_energy_absolute_error_pct,mean_weighted_profile_nrmse_pct,task109_end_to_end_seconds_per_shard
e4_complete_row_v1,1.107,0.25657,6.3183,8.3168,4.5812
e4_rank_lhs_v1,1.0966,0.24956,4.8385,6.9541,4.6165
full_mc,1.0609,0.25566,8.4066,10.262,2.5133
lhs_fast,1.0716,0.31842,7.0199,9.0675,0.19434
```

## Same-machine 15-building smoke

The smoke uses a fixed public roster, fixed seed and the same direct runtime entry point. It is an interface/runtime check, not new model evidence and not a replacement for Task109 timing.

```csv
sampling_mode,wall_time_seconds,candidate_count,row_count,deterministic_same_input,order_invariant,finite_pass,positive_rca_tau_pass,qint_support_pass
e4_rank_lhs_v1,0.44603,200,15,True,True,True,True,True
e4_complete_row_v1,0.38409,200,15,True,True,True,True,True
lhs_fast,0.073642,3,15,True,True,True,True,True
full_mc,0.43944,100,15,True,True,True,True,True
```

## Artifact and interface audit

- Protected artifact hashes pass: `True` (10 files).
- Privacy flags: `{"anonymous_rank_only": true, "contains_private_identifiers": false, "contains_raw_dalen_rows": false, "contains_validation_or_test_targets": false}`.
- Anonymous rank template: `[50, 5]`, permutation check `True`.
- Available modes: `['e4_rank_lhs_v1', 'e4_complete_row_v1', 'lhs_fast', 'full_mc']`.
- Public default: `e4_rank_lhs_v1`; E4 limit: `50`.
- Streamlit/CLI/direct-runtime consistency checks: `{"cli_all_modes": true, "cli_artifacts_public": true, "cli_default_e4_rank_lhs": true, "runtime_reports_selected_mode": true, "streamlit_all_modes": true, "streamlit_artifacts_public": true, "streamlit_default_e4_rank_lhs": true}`.

## Claim boundary

This package confirms current deploy integrity and interface consistency only. It does not change route mathematics, candidate budgets, selectors, artifacts or the public default. No external, blind, measured-energy, address-level, privacy or production-accuracy claim is authorized.
