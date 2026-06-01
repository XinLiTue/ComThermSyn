from pathlib import Path
import json
import pandas as pd

from src.artifact_io import load_deployment_artifacts, validate_public_artifact_safety
from src.deploy_runtime import (
    validate_community_input,
    run_deployment_synthesis,
    build_streamlit_output_package,
)


def main():
    # ------------------------------------------------------------------
    # 1. Paths
    # ------------------------------------------------------------------
    artifact_root = Path("artifacts_public")
    input_csv = Path("sample_inputs/my_test_10.csv")
    output_root = Path("streamlit_demo_outputs_test")

    print("=" * 80)
    print("Online runtime test")
    print("=" * 80)
    print(f"Artifact root: {artifact_root.resolve()}")
    print(f"Input CSV:     {input_csv.resolve()}")
    print(f"Output root:   {output_root.resolve()}")

    # ------------------------------------------------------------------
    # 2. Basic path checks
    # ------------------------------------------------------------------
    if not artifact_root.exists():
        raise FileNotFoundError(f"Artifact root not found: {artifact_root}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    required_artifact_files = [
        artifact_root / "model" / "deployment_model_payload.pkl",
        artifact_root / "model" / "runtime_config.json",
        artifact_root / "model" / "input_schema.json",
        artifact_root / "model" / "model_metadata.json",
        artifact_root / "model" / "public_sampling_baselines.parquet",
        artifact_root / "model" / "synthetic_residual_pool.parquet",
        artifact_root / "model" / "expected_energy_baseline_public.parquet",
        artifact_root / "weather" / "typical_weather_scenarios.json",
        artifact_root / "weather" / "typical_day_weights.json",
    ]

    missing = [p for p in required_artifact_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required deployment artifact files:\n"
            + "\n".join(str(p) for p in missing)
        )

    print("\n[OK] Required artifact files exist.")

    # ------------------------------------------------------------------
    # 3. Load artifacts
    # ------------------------------------------------------------------
    artifacts = load_deployment_artifacts(artifact_root)

    print("\nLoaded artifacts:")
    for key in sorted(artifacts.keys()):
        value = artifacts[key]
        if value is None:
            continue
        print(f"  - {key}: {type(value).__name__}")

    # Optional safety check on loaded artifact dictionary.
    # This should raise if obvious private identifiers are present.
    validate_public_artifact_safety(artifacts)
    print("\n[OK] Public artifact safety validation passed.")

    # ------------------------------------------------------------------
    # 4. Load and validate user input
    # ------------------------------------------------------------------
    input_df = pd.read_csv(input_csv)
    print("\nInput preview:")
    print(input_df.head())

    cleaned_input_df = validate_community_input(
        input_df,
        input_schema=artifacts.get("input_schema"),
    )

    print("\nValidated input:")
    print(cleaned_input_df.head())
    print(f"Input buildings: {len(cleaned_input_df)}")

    # ------------------------------------------------------------------
    # 5. Run online deployment synthesis
    # ------------------------------------------------------------------
    result = run_deployment_synthesis(
        cleaned_input_df,
        artifacts=artifacts,
        runtime_config=artifacts.get("runtime_config"),
    )

    generated_df = result["generated_building_parameters_df"]
    annual_df = result["community_annual_energy_summary_df"]
    profile_df = result["typical_day_community_profile_df"]
    dynamic_df = result["community_dynamic_metrics_df"]
    warnings_list = result.get("warnings", [])
    runtime_metadata = result.get("runtime_metadata", {})

    print("\nRuntime metadata:")
    print(json.dumps(runtime_metadata, indent=2, ensure_ascii=False))

    print("\nGenerated building parameters preview:")
    print(generated_df.head())

    print("\nAnnual energy summary:")
    print(annual_df)

    print("\nDynamic metrics:")
    print(dynamic_df)

    print("\nTypical-day profile preview:")
    print(profile_df.head())

    if warnings_list:
        print("\nWarnings:")
        for warning in warnings_list:
            print(f"  - {warning}")

    # ------------------------------------------------------------------
    # 6. Save Streamlit-ready output package
    # ------------------------------------------------------------------
    output_files = build_streamlit_output_package(result, output_root)

    print("\nSaved output files:")
    if isinstance(output_files, dict):
        for key, value in output_files.items():
            print(f"  - {key}: {value}")
    else:
        print(output_files)

    # ------------------------------------------------------------------
    # 7. Check expected output files
    # ------------------------------------------------------------------
    expected_outputs = [
        output_root / "manifest.json",
        output_root / "inputs" / "community_input.csv",
        output_root / "results" / "generated_building_parameters.csv",
        output_root / "results" / "community_annual_energy_summary.csv",
        output_root / "results" / "typical_day_community_profile.csv",
        output_root / "results" / "community_dynamic_metrics.csv",
        output_root / "plot_data" / "community_annual_energy.csv",
        output_root / "plot_data" / "typical_day_profiles.csv",
        output_root / "plot_data" / "community_dynamic_metrics.csv",
        output_root / "plot_data" / "plot_data_manifest.json",
    ]

    missing_outputs = [p for p in expected_outputs if not p.exists()]
    if missing_outputs:
        raise FileNotFoundError(
            "Missing expected output files:\n"
            + "\n".join(str(p) for p in missing_outputs)
        )

    print("\n[OK] All expected Streamlit-ready output files exist.")

    # ------------------------------------------------------------------
    # 8. Read manifest and print final concise summary
    # ------------------------------------------------------------------
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("\nManifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Try to extract common summary values.
    total_annual_kwh = None
    peak_kw = None

    for _, row in annual_df.iterrows():
        metric = str(row.get("metric", "")).lower()
        if "annual" in metric and "kwh" in metric:
            total_annual_kwh = row.get("value")

    for _, row in dynamic_df.iterrows():
        metric = str(row.get("metric", "")).lower()
        if "peak" in metric:
            peak_kw = row.get("value")
            break

    print("\n" + "=" * 80)
    print("ONLINE RUNTIME TEST PASSED")
    print("=" * 80)
    print(f"Runtime mode:        {runtime_metadata.get('runtime_mode')}")
    print(f"Mock runtime:        {runtime_metadata.get('mock_runtime')}")
    print(f"Input buildings:     {len(cleaned_input_df)}")
    print(f"Generated buildings: {len(generated_df)}")
    if total_annual_kwh is not None:
        print(f"Annual heating:      {total_annual_kwh} kWh")
    if peak_kw is not None:
        print(f"Peak heat demand:    {peak_kw} kW")
    print(f"Output folder:       {output_root.resolve()}")


if __name__ == "__main__":
    main()