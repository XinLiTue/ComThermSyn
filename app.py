from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.artifact_io import load_deployment_artifacts, validate_public_artifact_safety
from src.deploy_runtime import (
    build_streamlit_output_package,
    run_deployment_synthesis,
    validate_community_input,
)


st.set_page_config(
    page_title="ComThermSyn",
    layout="wide",
)


DEFAULT_ARTIFACT_ROOT = Path("artifacts_public")
DEFAULT_OUTPUT_ROOT = Path("streamlit_demo_outputs") / "jobs"
INTRO_DIR = Path("Intro")
INTRO_SLIDE_COUNT = 6
INTRO_SCENES = {
    1: "Scene 1 - Modeling challenge",
    2: "Scene 2 - Data confusion",
    3: "Scene 3 - ComThermSyn processing",
    4: "Scene 4 - Parameter synthesis",
    5: "Scene 5 - Profile generation",
    6: "Scene 6 - Energy-system insights",
}
DEFAULT_SCHEMA = {
    "required_columns": ["year", "Area", "EnergyLabel"],
    "optional_columns": ["building_id"],
    "allowed_energy_labels": ["A", "B", "C", "D", "E", "F", "G", "unknown", "0"],
    "year_range": [1900, 2025],
    "area_range": [20, 400],
    "max_buildings": 100,
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.info(f"Could not read JSON: {path} ({exc})")
        return None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.info(f"Could not read CSV: {path} ({exc})")
        return None


@st.cache_resource(show_spinner=False)
def cached_artifacts(artifact_root: str) -> dict[str, Any]:
    artifacts = load_deployment_artifacts(Path(artifact_root))
    validate_public_artifact_safety(artifacts)
    return artifacts


def safe_run_id(raw_run_id: str) -> str:
    run_id = raw_run_id.strip()
    if not run_id:
        raise ValueError("Enter a run_id.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may only contain letters, numbers, underscores, and hyphens.")
    return run_id


def default_input_table(schema: dict[str, Any]) -> pd.DataFrame:
    allowed_labels = schema.get("allowed_energy_labels") or ["A", "B", "C"]
    return pd.DataFrame(
        [
            {
                "building_id": "building_1",
                "year": 1985,
                "Area": 95.0,
                "EnergyLabel": allowed_labels[2] if len(allowed_labels) > 2 else allowed_labels[0],
            },
            {
                "building_id": "building_2",
                "year": 2001,
                "Area": 120.0,
                "EnergyLabel": allowed_labels[1] if len(allowed_labels) > 1 else allowed_labels[0],
            },
            {"building_id": "building_3", "year": 2018, "Area": 82.0, "EnergyLabel": allowed_labels[0]},
        ]
    )


def schema_columns(schema: dict[str, Any]) -> list[str]:
    columns = list(schema.get("optional_columns") or [])
    for column in schema.get("required_columns") or []:
        if column not in columns:
            columns.append(column)
    return columns


def df_download(df: pd.DataFrame | None, label: str, filename: str) -> None:
    if df is None:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def intro_slide_paths() -> tuple[list[tuple[int, Path]], list[Path]]:
    expected_paths = [(index, INTRO_DIR / f"{index}.png") for index in range(1, INTRO_SLIDE_COUNT + 1)]
    available = [(index, path) for index, path in expected_paths if path.exists()]
    missing = [path for _, path in expected_paths if not path.exists()]
    return available, missing


def display_intro_slideshow() -> None:
    if not INTRO_DIR.exists():
        st.info("Intro slideshow folder is not available.")
        return

    available, missing = intro_slide_paths()
    if not available:
        st.info("Intro slideshow images are not available.")
        return

    slide_numbers = [index for index, _ in available]
    current_slide = st.session_state.get("intro_slide_number", slide_numbers[0])
    if current_slide not in slide_numbers:
        current_slide = slide_numbers[0]
        st.session_state["intro_slide_number"] = current_slide

    _, carousel, _ = st.columns([1, 2, 1])
    with carousel:
        controls = st.columns([1, 2, 1])
        with controls[0]:
            if st.button("Previous", disabled=current_slide == slide_numbers[0]):
                current_position = slide_numbers.index(current_slide)
                st.session_state["intro_slide_number"] = slide_numbers[current_position - 1]
                st.rerun()
        with controls[1]:
            selected_slide = st.selectbox(
                "What ComThermSyn helps with",
                options=slide_numbers,
                index=slide_numbers.index(current_slide),
                format_func=lambda slide_number: INTRO_SCENES[slide_number],
            )
            st.session_state["intro_slide_number"] = selected_slide
        with controls[2]:
            if st.button("Next", disabled=current_slide == slide_numbers[-1]):
                current_position = slide_numbers.index(current_slide)
                st.session_state["intro_slide_number"] = slide_numbers[current_position + 1]
                st.rerun()

        selected_path = dict(available)[st.session_state["intro_slide_number"]]
        selected_title = INTRO_SCENES[st.session_state["intro_slide_number"]]
        st.image(
            str(selected_path),
            caption=selected_title,
            use_container_width=True,
        )

    if missing:
        missing_names = ", ".join(path.name for path in missing)
        st.info(f"Some intro slides are missing: {missing_names}.")


def metric_lookup(df: pd.DataFrame | None, metric: str) -> tuple[Any, str]:
    if df is None or "metric" not in df.columns:
        return None, ""
    rows = df[df["metric"].astype(str) == metric]
    if rows.empty:
        return None, ""
    row = rows.iloc[0]
    return row.get("value"), str(row.get("unit", ""))


def first_existing_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    columns_by_lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns_by_lower:
            return columns_by_lower[candidate.lower()]
    return None


def display_metric(label: str, value: Any, unit: str = "") -> None:
    if value is None:
        st.metric(label, "-")
        return
    try:
        number = float(value)
        formatted = f"{number:,.1f}"
    except Exception:
        formatted = str(value)
    st.metric(label, f"{formatted} {unit}".strip())


def result_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "synthetic": run_dir / "results" / "generated_building_parameters.csv",
        "annual": run_dir / "results" / "community_annual_energy_summary.csv",
        "profile": run_dir / "results" / "typical_day_community_profile.csv",
        "dynamic": run_dir / "results" / "community_dynamic_metrics.csv",
    }


def load_result_tables(run_dir: Path) -> dict[str, pd.DataFrame | None]:
    paths = result_paths(run_dir)
    return {name: load_csv(path) for name, path in paths.items()}


def scenario_column(profile_df: pd.DataFrame | None) -> str | None:
    return first_existing_column(
        profile_df,
        ["display_day_label", "weather_day", "scenario", "scenario_label", "weather_kind", "typical_day_id"],
    )


def time_column(profile_df: pd.DataFrame | None) -> str | None:
    return first_existing_column(profile_df, ["time", "datetime", "timestamp", "hour", "timestep", "timestep_index"])


def heat_column(profile_df: pd.DataFrame | None) -> str | None:
    return first_existing_column(
        profile_df,
        [
            "heat_kw",
            "heating_power_kw",
            "p_heat_kw",
            "community_heat_kw",
            "thermal_power_kw",
            "synthesized_heat_kw",
        ],
    )


def widget_key_from_path(run_dir: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(run_dir))


def filter_profile(
    profile_df: pd.DataFrame | None,
    key_prefix: str,
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    if profile_df is None or profile_df.empty:
        st.info("typical_day_community_profile.csv not available.")
        return None, None, None

    scenario_col = scenario_column(profile_df)
    building_col = first_existing_column(profile_df, ["building_id", "building", "building_name"])
    time_col = time_column(profile_df)
    demand_col = heat_column(profile_df)
    if time_col is None or demand_col is None:
        st.info("Could not detect time or heat-demand columns in the typical-day profile.")
        return None, None, None

    work = profile_df.copy()
    control_cols = st.columns(2)
    with control_cols[0]:
        if scenario_col is not None:
            scenarios = sorted(work[scenario_col].dropna().astype(str).unique().tolist())
            selected_scenario = st.selectbox(
                "Weather day/scenario",
                ["All available scenarios"] + scenarios,
                key=f"{key_prefix}_scenario",
            )
            if selected_scenario != "All available scenarios":
                work = work[work[scenario_col].astype(str) == selected_scenario].copy()
        else:
            st.info("Community-level profile only: no scenario column found.")

    with control_cols[1]:
        if building_col is not None:
            buildings = sorted(work[building_col].dropna().astype(str).unique().tolist())
            selected_scope = st.selectbox(
                "Building scope",
                ["Community total", "All buildings"] + buildings,
                key=f"{key_prefix}_building_scope",
            )
            if selected_scope == "Community total":
                group_cols = [time_col]
                if scenario_col is not None:
                    group_cols.insert(0, scenario_col)
                work = work.groupby(group_cols, as_index=False)[demand_col].sum()
            elif selected_scope != "All buildings":
                work = work[work[building_col].astype(str) == selected_scope].copy()
        else:
            st.selectbox(
                "Building scope",
                ["Community total"],
                disabled=True,
                key=f"{key_prefix}_building_scope",
            )

    return work, time_col, demand_col


def infer_timestep_hours(profile_df: pd.DataFrame, time_col: str) -> float | None:
    values = profile_df[time_col]
    if pd.api.types.is_numeric_dtype(values):
        ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values()
        diffs = ordered.diff().dropna()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            step = float(diffs.median())
            return step if step <= 6 else None
        return None

    parsed = pd.to_datetime(values, errors="coerce").dropna().sort_values()
    diffs = parsed.diff().dropna().dt.total_seconds() / 3600.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    return float(diffs.median())


def plot_typical_day_profile(profile_df: pd.DataFrame | None, key_prefix: str) -> None:
    filtered_df, time_col, demand_col = filter_profile(profile_df, key_prefix)
    if filtered_df is None or time_col is None or demand_col is None:
        return

    color_col = "display_day_label" if "display_day_label" in profile_df.columns else None
    if color_col not in filtered_df.columns:
        color_col = scenario_column(filtered_df)
    if color_col is None:
        color_col = first_existing_column(filtered_df, ["building_id", "building", "building_name"])

    fig = px.line(
        filtered_df,
        x=time_col,
        y=demand_col,
        color=color_col,
        title="Typical-Day Heat Demand Profile",
        labels={time_col: "Time", demand_col: "Heat demand (kW)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Heat pump electricity estimate", expanded=False):
        use_hp = st.checkbox(
            "Use heat pump conversion",
            value=False,
            key=f"{key_prefix}_use_hp",
        )
        cop = st.number_input(
            "COP",
            min_value=0.1,
            value=3.0,
            step=0.1,
            format="%g",
            key=f"{key_prefix}_cop",
        )
        if use_hp:
            hp_df = filtered_df.copy()
            hp_df["hp_electricity_kw"] = pd.to_numeric(hp_df[demand_col], errors="coerce") / cop
            hp_fig = px.line(
                hp_df,
                x=time_col,
                y="hp_electricity_kw",
                color=color_col if color_col in hp_df.columns else None,
                title="Heat Pump Electricity Profile",
                labels={time_col: "Time", "hp_electricity_kw": "Electricity demand (kW)"},
            )
            st.plotly_chart(hp_fig, use_container_width=True)

            timestep_hours = infer_timestep_hours(hp_df, time_col)
            if timestep_hours is None:
                st.info("Could not infer timestep duration, so only the kW profile is shown.")
            else:
                electricity_kwh = float(hp_df["hp_electricity_kw"].dropna().sum() * timestep_hours)
                display_metric("Selected-profile HP electricity consumption", electricity_kwh, "kWh")


def display_results(run_dir: Path, key_prefix: str | None = None) -> None:
    key_prefix = key_prefix or f"results_{widget_key_from_path(run_dir)}"
    tables = load_result_tables(run_dir)
    synthetic_df = tables["synthetic"]
    annual_df = tables["annual"]
    profile_df = tables["profile"]
    dynamic_df = tables["dynamic"]
    manifest = load_json(run_dir / "manifest.json") or {}

    if all(df is None for df in tables.values()):
        st.warning(f"No runtime outputs found in `{run_dir}`.")
        return

    st.subheader("Runtime summary")
    col_a, col_b = st.columns(2)
    col_a.metric("payload_type", manifest.get("payload_type", "-"))
    col_b.metric("synthesis_mode", manifest.get("synthesis_mode", "-"))

    total_annual, total_unit = metric_lookup(annual_df, "total_annual_heating_energy")
    peak_heat, peak_unit = metric_lookup(dynamic_df, "peak_heat_demand")

    st.subheader("Simulation summary")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        display_metric("Total annual heating energy", total_annual, total_unit)
    with metric_cols[1]:
        display_metric("Peak heat demand", peak_heat, peak_unit)
    with metric_cols[2]:
        display_metric("Input buildings", manifest.get("n_input_buildings"))
    with metric_cols[3]:
        display_metric("Synthetic buildings", manifest.get("n_generated_buildings"))

    plot_typical_day_profile(profile_df, key_prefix)

    st.subheader("Synthetic R, C, A, Qint")
    if synthetic_df is None:
        st.info("generated_building_parameters.csv not available.")
    else:
        preferred = [
            column
            for column in ["building_id", "year", "Area", "EnergyLabel", "R", "C", "A", "Qint"]
            if column in synthetic_df.columns
        ]
        st.dataframe(synthetic_df[preferred] if preferred else synthetic_df, use_container_width=True)

    with st.expander("Advanced diagnostics", expanded=False):
        proxy_value, proxy_unit = metric_lookup(annual_df, "typical_day_proxy_annual_heating_energy")
        if proxy_value is not None:
            display_metric("typical_day_proxy_annual_heating_energy", proxy_value, proxy_unit)
        if annual_df is not None:
            st.dataframe(annual_df, use_container_width=True)
        if dynamic_df is not None:
            st.dataframe(dynamic_df, use_container_width=True)

    st.subheader("Downloads")
    download_cols = st.columns(4)
    filenames = {
        "synthetic": "generated_building_parameters.csv",
        "annual": "community_annual_energy_summary.csv",
        "profile": "typical_day_community_profile.csv",
        "dynamic": "community_dynamic_metrics.csv",
    }
    labels = {
        "synthetic": "Synthetic thermal parameters",
        "annual": "Annual summary",
        "profile": "Typical-day profile",
        "dynamic": "Dynamic metrics",
    }
    for column, key in zip(download_cols, ["synthetic", "annual", "profile", "dynamic"]):
        with column:
            df_download(tables[key], labels[key], filenames[key])


logo_path = Path("logo.svg") if Path("logo.svg").exists() else Path("logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=165)
st.title("ComThermSyn")
st.caption("A Community Thermal Parameter Synthesizer for Energy System Optimization")

st.sidebar.header("ComThermSyn")
st.sidebar.subheader("Runtime Status")
artifact_root_input = str(DEFAULT_ARTIFACT_ROOT)
output_root_input = str(DEFAULT_OUTPUT_ROOT)
with st.sidebar.expander("Advanced settings", expanded=False):
    artifact_root_input = st.text_input("Artifact root", artifact_root_input)
    output_root_input = st.text_input("Output jobs folder", output_root_input)
artifact_root = Path(artifact_root_input)
output_root = Path(output_root_input)

try:
    artifacts = cached_artifacts(str(artifact_root))
    input_schema = artifacts.get("input_schema") or DEFAULT_SCHEMA
    st.sidebar.success("Artifact root detected")
    st.sidebar.success("Public runtime ready")
except Exception as exc:
    artifacts = None
    input_schema = DEFAULT_SCHEMA
    st.sidebar.error("Artifact root missing or incomplete")
    st.sidebar.warning("Public runtime not ready")
    with st.sidebar.expander("Runtime error", expanded=False):
        st.write(str(exc))
st.sidebar.info(f"Output folder: `{output_root}`")

tab_about, tab_submit, tab_load = st.tabs(["About", "Submit Case", "Load Results"])

with tab_about:
    st.subheader("ComThermSyn online demo")
    st.write(
        "ComThermSyn: A Community Thermal Parameter Synthesizer for Energy System Optimization "
        "synthesizes public-demo thermal parameters and community heating profiles from simple "
        "building/community inputs."
    )
    st.subheader("What ComThermSyn helps with")
    display_intro_slideshow()
    st.subheader("How ComThermSyn works")
    st.markdown(
        """
1. Submit a case with building or community inputs.
2. Wait for the online runtime to synthesize thermal parameters and profiles.
3. Load results using the same Run ID.
4. Download synthetic thermal parameters and profiles if needed.
        """
    )
    st.info(
        "This demo uses public-safe deployment artifacts and does not require private raw data."
    )

with tab_submit:
    st.subheader("Submit Case")
    run_id_input = st.text_input("Run ID", value="CPN8_001")
    overwrite = st.checkbox("overwrite existing run", value=False)

    uploaded_csv = st.file_uploader("Upload community CSV", type=["csv"])
    if uploaded_csv is not None:
        try:
            input_df = pd.read_csv(uploaded_csv)
        except Exception as exc:
            input_df = default_input_table(input_schema)
            st.error(f"Could not read uploaded CSV: {exc}")
    else:
        input_df = default_input_table(input_schema)

    columns = schema_columns(input_schema)
    for column in columns:
        if column not in input_df.columns:
            input_df[column] = None
    st.markdown("**Community input table**")
    editor_df = st.data_editor(
        input_df[columns],
        num_rows="dynamic",
        use_container_width=True,
        key="community_input_editor",
    )

    run_dir: Path | None = None
    try:
        run_id = safe_run_id(run_id_input)
        run_dir = output_root / run_id
        if run_dir.exists():
            st.warning(f"Run folder already exists: `{run_dir}`")
    except ValueError as exc:
        st.warning(str(exc))

    if st.button("Submit and Run Synthesis", type="primary", disabled=artifacts is None):
        try:
            run_id = safe_run_id(run_id_input)
            run_dir = output_root / run_id
            if run_dir.exists() and not overwrite:
                st.warning("Choose a new run_id or check overwrite existing run before submitting.")
            else:
                with st.spinner("Validating input and running public synthesis..."):
                    validated_input = validate_community_input(
                        editor_df,
                        input_schema=input_schema,
                    )
                    input_path = run_dir / "inputs" / "community_input.csv"
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    validated_input.to_csv(input_path, index=False)

                    result = run_deployment_synthesis(
                        validated_input,
                        artifacts=artifacts,
                        runtime_config=artifacts.get("runtime_config"),
                    )
                    manifest = build_streamlit_output_package(result, run_dir)

                st.success(f"Run complete: `{run_dir}`")
                st.caption(
                    f"payload_type={manifest.get('payload_type', '-')}, "
                    f"synthesis_mode={manifest.get('synthesis_mode', '-')}"
                )
                display_results(run_dir, key_prefix="submitted_result")
        except Exception as exc:
            st.error(f"Run failed: {exc}")

with tab_load:
    st.subheader("Load Results")
    load_run_id_input = st.text_input("Run ID to load", value="CPN8_001")
    if st.button("Load Results"):
        try:
            load_run_id = safe_run_id(load_run_id_input)
            loaded_run_dir = output_root / load_run_id
            st.session_state["loaded_run_id"] = load_run_id
            st.session_state["loaded_run_dir"] = loaded_run_dir
        except ValueError as exc:
            st.warning(str(exc))

    if "loaded_run_dir" in st.session_state:
        loaded_run_id = st.session_state.get("loaded_run_id", "")
        loaded_run_dir = Path(st.session_state["loaded_run_dir"])
        status_cols = st.columns([3, 1])
        with status_cols[0]:
            st.info(f"Loaded Run ID: `{loaded_run_id}`")
        with status_cols[1]:
            if st.button("Clear loaded result"):
                st.session_state.pop("loaded_run_id", None)
                st.session_state.pop("loaded_run_dir", None)
                st.rerun()
        display_results(loaded_run_dir, key_prefix="loaded_result")
