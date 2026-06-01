# ComThermSyn

A Community Thermal Parameter Synthesizer for Energy System Optimization.

## Overview

ComThermSyn is a public online demo for synthesizing community-level thermal parameters and heating profiles from simple building or community inputs. It supports fast public-demo runtime using prebuilt public-safe deployment artifacts.

## What It Does

- Accepts building/community inputs such as construction year, floor area, and energy label.
- Synthesizes thermal parameters such as R, C, A, and Qint.
- Generates typical-day community heating profiles.
- Reports community annual heating energy, peak demand, and dynamic ramp indicators.
- Produces Streamlit-ready output tables for visualization and download.

## Repository Structure

- `app.py` - Streamlit demo interface.
- `src/` - deployment runtime and artifact loading utilities.
- `scripts/` - command-line runtime utilities and tests.
- `artifacts_public/` - public-safe prebuilt deployment artifacts.
- `sample_inputs/` - example input CSVs.
- `docs/` - deployment notes and project rules.

## Input Format

The app/runtime expects a CSV with columns:

```text
building_id,year,Area,EnergyLabel
```

Example:

```csv
building_id,year,Area,EnergyLabel
demo_1,1968,95,D
demo_2,1987,120,C
demo_3,2003,145,A
demo_4,1996,130,B
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

Or run the CLI test:

```bash
python scripts/run_deployment_case.py --artifact-root artifacts_public --input sample_inputs/example_community_inputs.csv --output streamlit_demo_outputs_test
```

## Outputs

The runtime produces:

- `generated_building_parameters.csv`
- `community_annual_energy_summary.csv`
- `typical_day_community_profile.csv`
- `community_dynamic_metrics.csv`
- `manifest.json`

## Privacy and Data Note

This public demo does not include raw private training data, participant identifiers, truth tables, holdout split files, or private fitted-building records. The runtime uses public-safe deployment artifacts generated offline.

## Limitations

- The `public_sampler_v1` runtime is an online deployment approximation.
- It is intended for academic demonstration and community-level analysis, not individual building reconstruction.
- Typical-day annual energy estimates may differ from full-year research validation.
- Generated outputs should be interpreted as synthetic community-level profiles.

## Author / Contact

Xin Li  
Eindhoven University of Technology (TU/e)  
Electrical Energy Systems group  
Contact email: to be added
