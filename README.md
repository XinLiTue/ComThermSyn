# SyCoTherm Standalone Deploy

This directory is an independent online version of the community RCAQ generator. It keeps the
existing Streamlit shell, but its model artifacts, inference code, CLI, and output path all live
under `deploy/`. Runtime code never reads `Project/`.

## Frozen model and deployment routes

The model core is the Task102B development candidate. The default deployment bundle additionally
contains the anonymous Task107 residual-rank template used by the E4 representative routes:

- conditional means: `M0a` for `log(G)`, `log(C)`, and `log(A)`; `Q0` for `Qint`;
- partial pooling: `n0 = 5`;
- dependence: 4-dimensional Ledoit-Wolf shrunk Gaussian Copula;
- default representative community: `e4_rank_lhs_v1`, which combines Latin-hypercube marginal
  coverage with the anonymous empirical residual-rank dependence template;
- empirical representative alternative: `e4_complete_row_v1`, which preserves complete empirical
  residual-rank combinations;
- fast comparison route: 16 Latin-hypercube residual designs are probability-screened and only up
  to the best 3 acceptable complete communities are evaluated as `lhs_fast`;
- reference uncertainty: the original 50 bootstrap parameter draws x 2 residual realizations =
  100 complete communities remains available as `full_mc`;
- single result: one complete sampled community is selected, never an element-wise median;
- `EnergyLabel`: required and returned, but not predictive in this frozen version.

The public bundle contains only sanitized parameter draws, empirical marginal values, Q period
scales, a 4x4 correlation matrix, schemas, and typical-weather profiles. It contains no raw Dalen
rows, private identifiers, or validation/final-test targets.

## Input contract

CSV columns:

```text
building_id,year,Area,EnergyLabel
```

The strict development support is `1946 <= year <= 2005` and `76 <= Area <= 217 m2`, with at most
100 buildings. The 1992-2005 period has only one calibration building and is marked nearly
unsupported.

## Run

The deploy directory can be copied by itself to another computer. It does not require the
Project directory, raw Dalen/RVO data, training outputs, or project logs at runtime.

From inside the copied deploy directory, start the web application with:

    uv run streamlit run app.py

The local page is normally available at http://localhost:8501. The included pyproject.toml lets
uv create/synchronize the deploy-specific environment automatically. On Windows, run_web.ps1 is
the equivalent one-command launcher.

From the repository root:

```powershell
uv run python deploy\scripts\run_deployment_case.py `
  deploy\sample_inputs\example_community_inputs.csv `
  deploy\streamlit_demo_outputs\jobs\example
```

The default is `e4_rank_lhs_v1`. The available overrides are:

```text
--sampling-mode e4_complete_row_v1
--sampling-mode lhs_fast
--sampling-mode full_mc
```

The E4 routes accept 2-50 buildings. `lhs_fast` and `full_mc` remain available for larger inputs
within the public input schema. The fast output also reports the conditional C center, sampled C multiplier, and residual
quantile for each building so unusually large values can be traced directly.
For C, the representative community may contain no more than the expected 20% of samples outside
the central 10%-90% interval and no more than the expected 10% in the upper tail. Counts scale with
the roster size, while the other variables use joint chi-square typicality and soft tail penalties.
If none of the 16 designs passes every preferred gate, the closest joint-valid design is returned
with an explicit warning instead of failing. These rules select a central deliverable and do not
truncate the underlying model or the `full_mc` mode.

Start the retained web shell:

```powershell
uv run streamlit run deploy\app.py
```

Rebuilding the sanitized model bundle is a maintenance operation that requires Project. It is not
part of standalone inference. Rebuild only after the frozen Project artifacts change:

```powershell
uv run python deploy\scripts\build_model_bundle.py
```

Run focused tests:

```powershell
uv run python -m unittest discover -s deploy\tests -v
```

## Output interpretation

`generated_building_parameters.csv` is one generated community with `R`, `C`, `A`, `Qint`, and
`tau_hours`. The chosen `selected_realization_id` is identical across all rows because the output is
a genuine joint realization.

The heating proxy is:

```text
Qheat = max((Tset - Ta) / R - A * qg / 1000 - Qint, 0)
```

Typical-day annual and peak outputs are simulation proxies, not measured-energy predictions. This
release remains a frozen development candidate for community studies; it does not support
address-level accuracy, population representativeness, formal privacy, or production claims.
