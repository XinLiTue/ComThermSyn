from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.artifact_io import load_deployment_artifacts, validate_public_artifact_safety
from src.deploy_runtime import (
    build_streamlit_output_package,
    fill_optional_energy_labels,
    run_deployment_synthesis,
    simulate_hourly_rcaq_control,
    validate_community_input,
)
from src.real_case_io import load_real_case_result


# Presentation names are separate from the frozen runtime sampling identifiers.
SAMPLING_METHODS = {
    "lhs_fast": (
        "Fast candidate-LHS",
        "Stratifies the residual probability range for each house across candidate "
        "communities and returns the most typical community. This is the web default "
        "and supports communities of up to 100 buildings.",
    ),
    "e4_rank_lhs_v1": (
        "Community rank-LHS",
        "Stratifies each residual distribution across the houses within one community "
        "while retaining the pilot rank dependence.",
    ),
    "e4_complete_row_v1": (
        "Empirical residuals",
        "Uses complete pilot residual vectors without interval stratification or "
        "cross-variable recombination and returns the generated community nearest "
        "the robust centre.",
    ),
    "full_mc": (
        "Copula Monte Carlo",
        "Draws joint residual vectors from the fitted Copula and returns the generated "
        "community nearest the robust centre.",
    ),
}


st.set_page_config(
    page_title="SyCoTherm",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .sycotherm-hero {
        margin: 0.35rem 0 1.8rem 0;
    }
    .sycotherm-hero h1 {
        font-size: clamp(2.25rem, 4vw, 3.25rem);
        line-height: 1.08;
        letter-spacing: -0.025em;
        margin: 0;
    }
    .sycotherm-hero p {
        font-size: clamp(1.05rem, 2vw, 1.35rem);
        line-height: 1.5;
        margin: 0.75rem 0 0 0;
        color: rgba(49, 51, 63, 0.72);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        flex-wrap: wrap;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        padding-bottom: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        min-height: 3.9rem;
        padding: 0.8rem 1.5rem;
        border-radius: 0.65rem 0.65rem 0 0;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .example-formula {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.8rem;
        padding: 1rem 1.25rem;
        margin: 0.7rem 0 1.2rem 0;
        background: rgba(128, 128, 128, 0.06);
        font-size: 1.05rem;
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 1.2rem; }
        .stTabs [data-baseweb="tab"] {
            min-height: 3.25rem;
            padding: 0.6rem 0.85rem;
        }
        .stTabs [data-baseweb="tab"] p { font-size: 1.1rem; }
    }
    /* One type family and a restrained palette across the product. */
    .stApp, .stApp input, .stApp textarea, .stApp button,
    .stApp [data-testid="stMarkdownContainer"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }
    .stApp { background: #ffffff; color: #203b49; }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }
    .block-container { max-width: 1240px; padding-top: 3rem; }
    .stApp h1, .stApp h2, .stApp h3 { color: #173c4d; font-weight: 600; }
    .stApp h3 { font-size: 1.5rem; line-height: 1.35; }
    .brand-header {
        display: flex; align-items: center; gap: 3rem;
        padding: 0 0 2rem; margin-bottom: 1rem;
    }
    .brand-logo { width: 370px; max-width: 42%; height: auto; flex-shrink: 0; }
    .brand-copy { max-width: 620px; }
    .brand-eyebrow { color: #9b6334; font-size: 0.8rem; letter-spacing: 0.15em;
        text-transform: uppercase; font-weight: 600; }
    .brand-copy h1 { font-size: clamp(1.8rem, 3vw, 2.65rem); line-height: 1.2;
        letter-spacing: -0.025em; padding: 0.5rem 0; margin: 0; }
    .brand-copy p { font-size: 1.05rem; line-height: 1.65; color: #576e79; }
    :is(.stTabs, [data-testid="stTabs"]) [role="tablist"] {
        gap: 0.5rem; padding: 0.45rem;
        background: #f2f6f7 !important;
        border: 1px solid #e0e8eb;
        border-radius: 12px;
    }
    :is(.stTabs, [data-testid="stTabs"]) [role="tab"] {
        height: auto; min-height: 3.3rem;
        padding: 0.7rem 1.35rem;
        border-radius: 8px;
        background: #f2f6f7 !important;
        color: #203b49 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    :is(.stTabs, [data-testid="stTabs"]) [role="tab"] * {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
    }
    :is(.stTabs, [data-testid="stTabs"]) [role="tab"][aria-selected="true"] {
        background: #173c4d !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 2rem; }
    .workflow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
        margin: 1rem 0 1.6rem; }
    .workflow article { border: 1px solid #e0e8eb; border-radius: 12px; padding: 1.4rem; }
    .workflow span { font-size: 0.85rem; color: #9b6334; font-weight: 600; }
    .workflow h4 { margin: 0.7rem 0 0.4rem; color: #173c4d; font-size: 1.1rem; }
    .workflow p { color: #576e79; font-size: 0.95rem; line-height: 1.6; margin: 0; }
    .model-source {
        margin: 0.6rem 0 0.4rem;
        padding: 0.7rem 0.95rem;
        border-left: 3px solid #b7743c;
        background: #f5f8f8;
        color: #294b5b;
        font-size: 1.08rem;
        line-height: 1.5;
    }
    .model-source a { color: #145b77; font-weight: 600; }
    @media (max-width: 700px) {
        .brand-header { flex-direction: column; gap: 0.5rem; align-items: flex-start; }
        .brand-logo { width: 320px; max-width: 100%; }
        .workflow { grid-template-columns: 1fr; }
        :is(.stTabs, [data-testid="stTabs"]) [role="tab"] {
            padding: 0.6rem 0.75rem;
            font-size: 1.12rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = BASE_DIR / "artifacts_public"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "streamlit_demo_outputs" / "jobs"
INTRO_DIR = BASE_DIR / "Intro"
BALLUM_RESULT_PATH = BASE_DIR / "real_cases" / "ballum_real_case_result.json"
INTRO_SLIDE_COUNT = 6
INTRO_SCENES = {
    1: "Scene 1 - Modeling challenge",
    2: "Scene 2 - Data confusion",
    3: "Scene 3 - SyCoTherm processing",
    4: "Scene 4 - Parameter synthesis",
    5: "Scene 5 - Profile generation",
    6: "Scene 6 - Energy-system insights",
}
DEFAULT_SCHEMA = {
    "required_columns": ["year", "Area", "EnergyLabel"],
    "optional_columns": ["building_id"],
    "allowed_energy_labels": ["A", "B", "C", "D", "E", "F", "G", "unknown", "0"],
    "year_range": [1946, 2005],
    "area_range": [76, 217],
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
            {"building_id": "building_3", "year": 1971, "Area": 82.0, "EnergyLabel": allowed_labels[0]},
        ]
    )


def schema_columns(schema: dict[str, Any]) -> list[str]:
    columns = list(schema.get("optional_columns") or [])
    for column in schema.get("required_columns") or []:
        if column not in columns:
            columns.append(column)
    return columns


def df_download(df: pd.DataFrame | None, label: str, filename: str, *, key: str) -> None:
    if df is None:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
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
                "What SyCoTherm helps with",
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


def render_intro_video(video_path: str, max_width: int = 760) -> None:
    video_file = Path(video_path)
    if not video_file.exists():
        st.warning(f"Intro video not found: {video_path}")
        return

    encoded_video = base64.b64encode(video_file.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin: 1.0rem 0 1.4rem 0;">
            <video
                autoplay
                loop
                muted
                playsinline
                style="
                    width: 100%;
                    max-width: {max_width}px;
                    border-radius: 14px;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                "
            >
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def hourly_rcaq_example(
    resistance: float,
    capacitance: float,
    solar_aperture: float,
    internal_gains: float,
) -> pd.DataFrame:
    outdoor_temperature = [
        4.0, 3.5, 3.0, 2.5, 2.0, 2.0, 3.0, 5.0,
        7.0, 9.0, 11.0, 12.0, 13.0, 13.0, 12.0, 11.0,
        9.0, 8.0, 7.0, 6.0, 5.0, 5.0, 4.5, 4.0,
    ]
    solar_radiation = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
        0.08, 0.16, 0.26, 0.35, 0.42, 0.45, 0.40, 0.30,
        0.18, 0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    target_temperature = [
        17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 20.0, 20.0,
        20.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
        18.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 17.0,
    ]
    return simulate_hourly_rcaq_control(
        resistance_c_per_kw=resistance,
        capacitance_kwh_per_c=capacitance,
        solar_aperture_m2=solar_aperture,
        internal_gains_kw=internal_gains,
        outdoor_temperature_c=outdoor_temperature,
        solar_radiation_kw_m2=solar_radiation,
        target_temperature_c=target_temperature,
        initial_indoor_temperature_c=18.0,
        time_step_hours=1.0,
    )


def render_basic_use_case() -> None:
    st.subheader("24-hour RCAQ heating-control example")
    st.write(
        "When indoor temperature changes, C represents the building's ability to store or "
        "release heat. Raising the indoor temperature requires heat to be stored in the "
        "building mass; when the temperature falls, part of that stored heat is released. "
        "This example uses 24 one-hour control points to show that dynamic effect."
    )
    st.markdown(
        """
        <div class="example-formula">
            <strong>Hourly heating demand</strong><br>
            Q<sub>heat</sub> = max(
            C × (T<sub>next</sub> − T<sub>indoor</sub>) / Δt
            + (T<sub>indoor</sub> − T<sub>outdoor</sub>) / R
            − A × q<sub>solar</sub> − Q<sub>int</sub>, 0)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Example building parameters")
    parameter_cols = st.columns(4)
    with parameter_cols[0]:
        resistance = st.number_input(
            "R (degC/kW)", min_value=0.01, value=2.0, step=0.1, key="example_r"
        )
    with parameter_cols[1]:
        capacitance = st.number_input(
            "C (kWh/degC)", min_value=0.01, value=5.0, step=0.5, key="example_c"
        )
    with parameter_cols[2]:
        solar_aperture = st.number_input(
            "A (m2)", min_value=0.0, value=4.0, step=0.5, key="example_a"
        )
    with parameter_cols[3]:
        internal_gains = st.number_input(
            "Qint (kW)", value=0.6, step=0.1, key="example_qint"
        )

    example_df = hourly_rcaq_example(
        float(resistance),
        float(capacitance),
        float(solar_aperture),
        float(internal_gains),
    )
    st.info(
        "At each hour, the controller chooses the minimum non-negative heating power needed "
        "to reach the next target temperature. **Envelope heat loss = (Tin − Tout) / R**. "
        "**A is the effective solar aperture area** used to convert incident solar radiation "
        "into useful solar heat gain: **Solar gain = A × qsolar**, where qsolar is in kW/m2."
    )

    total_heating_kwh = float(example_df["Heating power (kW)"].sum())
    peak_heating_kw = float(example_df["Heating power (kW)"].max())
    peak_solar_gain_kw = float(example_df["Solar gain (kW)"].max())
    metric_cols = st.columns(3)
    metric_cols[0].metric("24-hour heating energy", f"{total_heating_kwh:.1f} kWh")
    metric_cols[1].metric("Peak hourly heating", f"{peak_heating_kw:.1f} kW")
    metric_cols[2].metric("Peak solar gain", f"{peak_solar_gain_kw:.2f} kW")

    temperature_plot = example_df[
        ["Hour", "Indoor end (degC)", "Target end (degC)", "Outdoor (degC)"]
    ].melt(id_vars="Hour", var_name="Temperature series", value_name="Temperature (degC)")
    temperature_figure = px.line(
        temperature_plot,
        x="Hour",
        y="Temperature (degC)",
        color="Temperature series",
        markers=True,
        title="Hourly indoor temperature, target, and outdoor temperature",
    )
    st.plotly_chart(temperature_figure, use_container_width=True)

    cop = st.number_input(
        "Heat pump COP",
        min_value=0.1,
        value=3.0,
        step=0.1,
        format="%.1f",
        key="example_cop",
        help="Constant illustrative COP. Electricity power equals heating power divided by COP.",
    )
    example_df["Electricity power (kW)"] = example_df["Heating power (kW)"] / cop
    heating_figure = make_subplots(specs=[[{"secondary_y": True}]])
    heating_figure.add_trace(
        go.Bar(
            x=example_df["Hour"],
            y=example_df["Heating power (kW)"],
            name="Heating power",
            marker_color="#173c4d",
            opacity=0.85,
        ),
        secondary_y=False,
    )
    heating_figure.add_trace(
        go.Scatter(
            x=example_df["Hour"],
            y=example_df["Electricity power (kW)"],
            name="Electricity power",
            mode="lines+markers",
            line={"color": "#b7743c", "width": 3},
            marker={"color": "#b7743c", "size": 6},
        ),
        secondary_y=True,
    )
    heating_figure.update_layout(
        title="Hourly heating-control points",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#203b49"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 55, "r": 60, "t": 90, "b": 50},
        hovermode="x unified",
    )
    heating_figure.update_xaxes(title_text="Hour", dtick=2)
    heating_figure.update_yaxes(
        title_text="Heating power (kW)",
        gridcolor="#e2e9ec",
        secondary_y=False,
    )
    heating_figure.update_yaxes(
        title_text="Electricity power (kW)",
        showgrid=False,
        secondary_y=True,
    )
    st.plotly_chart(heating_figure, use_container_width=True)

    st.markdown("#### Hourly calculation table")
    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column, format="%.2f")
            for column in example_df.columns
        },
    )
    st.info(
        "This is an educational deterministic 1R1C control example with one-hour steps and no "
        "active cooling. It illustrates how RCAQ parameters enter a dynamic calculation; it is "
        "not a measured-energy result or a full building-control optimization."
    )


def render_ballum_real_case() -> None:
    try:
        result = load_real_case_result(BALLUM_RESULT_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error("The Ballum aggregate result is currently unavailable.")
        with st.expander("Setup details", expanded=False):
            st.write(str(exc))
        return

    case = result["case"]
    comparison = result["comparison"]
    method = result["method"]
    monthly = pd.DataFrame(result["monthly_results"])

    st.subheader("Real case: Ballum, Ameland")
    st.markdown(
        "Ballum is a village on Ameland, an island in the northern Netherlands. "
        "SyCoTherm models its community space-heating demand using public building "
        f"information and {case['weather_year']} hourly weather observations from "
        f"{case['weather_station']}. "
        "TNO's [Warmteprofielengenerator]"
        f"({comparison['reference_url']}) has also modelled Ballum, "
        "providing an independent example for comparison."
    )
    # st.caption(
    #     "Both are modelled scenarios with different inputs and assumptions; "
    #     "this is not a measured-energy validation."
    # )

    st.markdown("#### Monthly heating energy and outdoor temperature")
    energy_temperature_figure = make_subplots(specs=[[{"secondary_y": True}]])
    energy_temperature_figure.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["heating_energy_mwh_th"],
            name="Heating energy",
            marker_color="#173c4d",
            opacity=0.88,
            hovertemplate="%{x}<br>%{y:.1f} MWhₜₕ<extra></extra>",
        ),
        secondary_y=False,
    )
    energy_temperature_figure.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["min_outdoor_temperature_c"],
            name="Monthly temperature range",
            showlegend=False,
            mode="lines",
            line={"color": "rgba(183, 116, 60, 0.18)", "width": 1},
            hovertemplate="%{x}<br>Minimum %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=True,
    )
    energy_temperature_figure.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["max_outdoor_temperature_c"],
            name="Min–max temperature",
            mode="lines",
            line={"color": "rgba(183, 116, 60, 0.18)", "width": 1},
            fill="tonexty",
            fillcolor="rgba(183, 116, 60, 0.14)",
            hovertemplate="%{x}<br>Maximum %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=True,
    )
    energy_temperature_figure.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["mean_outdoor_temperature_c"],
            name="Mean outdoor temperature",
            mode="lines+markers",
            line={"color": "#b7743c", "width": 3},
            marker={"color": "#b7743c", "size": 7},
            hovertemplate="%{x}<br>Mean %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=True,
    )
    energy_temperature_figure.update_layout(
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#203b49"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "y": 1.16, "x": 0},
        margin={"l": 55, "r": 60, "t": 80, "b": 45},
        hovermode="x unified",
    )
    energy_temperature_figure.update_xaxes(title_text="Month")
    energy_temperature_figure.update_yaxes(
        title_text="Heating energy (MWhₜₕ)", gridcolor="#e2e9ec", secondary_y=False
    )
    energy_temperature_figure.update_yaxes(
        title_text="Outdoor temperature (°C)", showgrid=False, secondary_y=True
    )
    st.plotly_chart(energy_temperature_figure, use_container_width=True)
    st.caption("Heating is switched off in June–August in this example.")

    st.markdown("#### Monthly electrified heating energy")
    electricity_figure = go.Figure(
        go.Bar(
            x=monthly["month"],
            y=monthly["hp_electricity_mwh_e"],
            name="Heat-pump electricity",
            marker_color="#b7743c",
            hovertemplate="%{x}<br>%{y:.1f} MWhₑ<extra></extra>",
        )
    )
    electricity_figure.update_layout(
        xaxis_title="Month",
        yaxis_title="Heat-pump electricity (MWhₑ)",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#203b49"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 55, "r": 30, "t": 35, "b": 45},
    )
    electricity_figure.update_yaxes(gridcolor="#e2e9ec")
    st.plotly_chart(electricity_figure, use_container_width=True)
    with st.expander("Monthly result table", expanded=False):
        monthly_table = monthly[
            [
                "month",
                "heating_energy_mwh_th",
                "hp_electricity_mwh_e",
                "mean_outdoor_temperature_c",
                "min_outdoor_temperature_c",
                "max_outdoor_temperature_c",
            ]
        ].rename(
            columns={
                "month": "Month",
                "heating_energy_mwh_th": "Heating energy (MWhₜₕ)",
                "hp_electricity_mwh_e": "HP electricity (MWhₑ)",
                "mean_outdoor_temperature_c": "Mean outdoor temperature (°C)",
                "min_outdoor_temperature_c": "Minimum outdoor temperature (°C)",
                "max_outdoor_temperature_c": "Maximum outdoor temperature (°C)",
            }
        )
        st.dataframe(monthly_table, use_container_width=True, hide_index=True)

    st.caption(
        "Electricity uses P_HP = Q_heat / COP, with COP = 20.3595 − 3.2061 × "
        "log₂(1 + 40 − T_out). For related modelling details, see "
        f"[the paper]({method['research_url']})."
    )

    st.markdown("#### Conclusion")
    st.markdown(
        "After normalising annual space-heating energy to the same number of "
        "residential units, SyCoTherm and TNO differ by only "
        f"**{comparison['normalized_difference_percent']:.2f}% relative to TNO**."
    )
    # st.caption(
    #     "This is a scale-adjusted comparison, not a like-for-like validation: "
    #     f"the weather years differ ({case['weather_year']} versus "
    #     f"{comparison['reference_weather_year']}), as do the building selections "
    #     "and model assumptions."
    # )


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

    sampling_mode = str(manifest.get("sampling_mode", ""))
    if sampling_mode in SAMPLING_METHODS:
        st.markdown(f"**Sampling method:** {SAMPLING_METHODS[sampling_mode][0]}")

    total_annual, total_unit = metric_lookup(annual_df, "total_annual_heating_energy")
    peak_heat, peak_unit = metric_lookup(dynamic_df, "peak_heat_demand")

    st.subheader("Simulation summary")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        display_metric("Typical-weather winter heating proxy", total_annual, total_unit)
    with metric_cols[1]:
        display_metric("Peak heating-demand proxy", peak_heat, peak_unit)
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
            for column in [
                "building_id",
                "year",
                "period",
                "Area",
                "EnergyLabel",
                "support_level",
                "R",
                "C",
                "A",
                "Qint",
                "C_conditional_center",
                "C_residual_multiplier",
                "residual_quantile_C",
            ]
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
            df_download(
                tables[key],
                labels[key],
                filenames[key],
                key=f"{key_prefix}_download_{key}",
            )


logo_svg_path = BASE_DIR / "logo.svg"
logo_path = logo_svg_path if logo_svg_path.exists() else BASE_DIR / "logo.jpg"
if logo_path.exists():
    logo_mime = "image/svg+xml" if logo_path.suffix == ".svg" else "image/jpeg"
    logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    logo_html = f'<img class="brand-logo" src="data:{logo_mime};base64,{logo_data}" alt="SyCoTherm logo">'
else:
    logo_html = '<div class="brand-logo">SyCoTherm</div>'
st.markdown(
    f"""
    <div class="brand-header">
        {logo_html}
        <div class="brand-copy">
            <span class="brand-eyebrow">Community energy modelling</span>
            <h1>Thermal parameters.<br>Community insights.</h1>
            <p>Synthesize house thermal parameters and explore heating demand
            for community-level energy system studies.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

artifact_root = DEFAULT_ARTIFACT_ROOT
output_root = DEFAULT_OUTPUT_ROOT

try:
    artifacts = cached_artifacts(str(artifact_root))
    input_schema = artifacts.get("input_schema") or DEFAULT_SCHEMA
    runtime_config = artifacts.get("runtime_config") or {}
except Exception as exc:
    artifacts = None
    input_schema = DEFAULT_SCHEMA
    st.error("The model is currently unavailable. Please try again later.")
    with st.expander("Setup details", expanded=False):
        st.write(str(exc))

tab_about, tab_submit, tab_load, tab_example, tab_real_case, tab_contact = st.tabs(
    ["About", "Submit Case", "Load Results", "Basic Example", "Real Case", "Contact"]
)

with tab_about:
    st.subheader("From building inputs to community heating demand")
    st.write("Explore how SyCoTherm connects simple building information with thermal modelling.")
    # Previous intro slideshow kept for rollback.
    # display_intro_slideshow()
    render_intro_video(str(INTRO_DIR / "IntroVideo_muted.mp4"), max_width=1040)
    st.subheader("Your workflow")
    st.markdown(
        """
<div class="workflow">
    <article><span>01 / INPUT</span><h4>Describe your community</h4>
    <p>Enter construction years and floor areas, or upload your building roster.</p></article>
    <article><span>02 / GENERATE</span><h4>Create a case</h4>
    <p>Submit your inputs to synthesize a complete set of community thermal parameters.</p></article>
    <article><span>03 / EXPLORE</span><h4>Put results to work</h4>
    <p>Open Load Results with your Run ID to explore profiles and download your results.</p></article>
</div>
        """
        , unsafe_allow_html=True
    )
    st.caption("Research demonstration for community-level studies. Generated heating profiles are simulation proxies, not measured energy use.")
    with st.expander("Model details and scope", expanded=False):
        if artifacts is not None:
            st.write("Supports 1–100 buildings, construction years 1946–2005, and floor areas 76–217 m². Community rank-LHS and Empirical residuals selections use Copula Monte Carlo automatically outside the 2–50 building range.")
        st.write("Frozen development model with internal retrospective evidence. EnergyLabel is optional and does not affect RCAQ generation. Outputs do not establish address-level accuracy.")
        for method_name, method_description in SAMPLING_METHODS.values():
            st.markdown(f"**{method_name}** — {method_description}")
    st.markdown(
        """
<div class="model-source">
    Measured-data source: DACS-HW projetct. See the
    <a href="https://kennisdelen.rvo.nl/groups/view/1434df6b-d9c4-4d87-8c73-36834399abeb/kennis-en-dataplatform-individuele-technieken-kite/page/view/6b6b0b0c-5d15-4b8d-9f02-2bc84a886715/dacs-hw" target="_blank" rel="noopener noreferrer">RVO/KITE DACS-HW page</a>.
</div>
        """,
        unsafe_allow_html=True,
    )

with tab_submit:
    st.subheader("Submit Case")
    run_id_input = st.text_input("Run ID", value="CPN8_001")
    overwrite = st.checkbox("overwrite existing run", value=False)
    selected_sampling_mode = st.selectbox(
        "Sampling mode",
        list(SAMPLING_METHODS),
        format_func=lambda mode: SAMPLING_METHODS[mode][0]
        + (" (default)" if mode == "lhs_fast" else ""),
        index=0,
    )
    st.write(SAMPLING_METHODS[selected_sampling_mode][1])
    st.caption(
        "Submit 1–100 buildings, with construction years 1946–2005 and floor areas "
        "76–217 m². For Community rank-LHS and Empirical residuals selections, "
        "cases outside 2–50 buildings automatically use Copula Monte Carlo."
    )

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
    st.caption(
        "EnergyLabel is optional in this web form. Blank or missing values are saved as "
        "`unknown` and do not affect RCAQ generation in the current frozen model."
    )
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
                with st.spinner("Generating and selecting a complete community realization..."):
                    submission_df = fill_optional_energy_labels(editor_df)
                    validated_input = validate_community_input(
                        submission_df,
                        input_schema=input_schema,
                    )
                    input_path = run_dir / "inputs" / "community_input.csv"
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    validated_input.to_csv(input_path, index=False)
                    selected_runtime_config = dict(artifacts.get("runtime_config") or {})
                    effective_sampling_mode = selected_sampling_mode
                    if selected_sampling_mode in {"e4_rank_lhs_v1", "e4_complete_row_v1"} and not 2 <= len(validated_input) <= 50:
                        effective_sampling_mode = "full_mc"
                        st.info(
                            f"This case contains {len(validated_input)} buildings. "
                            "Using Copula Monte Carlo because Community rank-LHS "
                            "and Empirical residuals support 2–50 buildings."
                        )
                    selected_runtime_config["sampling_mode"] = effective_sampling_mode

                    result = run_deployment_synthesis(
                        validated_input,
                        artifacts=artifacts,
                        runtime_config=selected_runtime_config,
                    )
                    build_streamlit_output_package(result, run_dir)

                st.success(f"Run complete: `{run_dir}`")
                st.session_state["last_completed_run_id"] = run_id
                st.info(
                    f"Next step: open **Load Results** and load Run ID **{run_id}** to view "
                    "charts, synthesized RCAQ parameters, and downloads."
                )
        except Exception as exc:
            st.error(f"Run failed: {exc}")

with tab_load:
    st.subheader("Load Results")
    load_run_id_input = st.text_input(
        "Run ID to load",
        value=st.session_state.get("last_completed_run_id", "CPN8_001"),
    )
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

with tab_example:
    render_basic_use_case()

with tab_real_case:
    render_ballum_real_case()

with tab_contact:
    st.subheader("Contact")
    st.write(
        "SyCoTherm is developed as an academic online demonstration for "
        "community-level thermal parameter synthesis and energy-system analysis."
    )
    st.markdown(
        """
**Author**  
Xin Li  
Eindhoven University of Technology (TU/e)  
Electrical Energy Systems group


**Contact**  
Contact email: x.li7@tue.nl  

        """
    )
    # st.info(
    #     "This public demo uses public-safe deployment artifacts and user-provided inputs. "
    #     "It does not expose private training user identifiers or truth tables."
    # )
