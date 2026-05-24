from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Community Thermal Profile Generation Demo",
    layout="wide",
)


DEFAULT_RUN_DIR = Path("streamlit_demo_outputs")


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


def maybe_df(path: Path, label: str) -> pd.DataFrame | None:
    df = load_csv(path)
    if df is None:
        st.info(f"{label} not available: `{path}`")
    return df


def filter_df(
    df: pd.DataFrame | None,
    *,
    community_size: Any = None,
    case_name: Any = None,
    weather_kind: Any = None,
) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if community_size is not None and "community_size" in out.columns:
        out = out[out["community_size"].astype(str) == str(community_size)]
    if case_name is not None and "case_name" in out.columns:
        out = out[out["case_name"].astype(str) == str(case_name)]
    if weather_kind is not None and "weather_kind" in out.columns:
        out = out[out["weather_kind"].astype(str) == str(weather_kind)]
    return out


def detect_first_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def numeric_plot_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    candidates = [
        "R",
        "C",
        "A",
        "Q",
        "Qint",
        "E_std",
        "E_std_full_per_m2",
        "annual_heating_kwh",
        "annual_heating_mwh",
        "E_std_annual_per_m2",
        "E_proxy_annual_per_m2",
        "annual_heat_kwh",
        "annual_heat_mwh",
        "annual_heat_kwh_per_m2",
        "peak_heat_kw",
    ]
    out = []
    for col in candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def df_download(df: pd.DataFrame | None, label: str, filename: str) -> None:
    if df is None:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def compact_building_input_view(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    preferred = [
        "community_size",
        "case_id",
        "case_name",
        "building_id",
        "construction_year",
        "floor_area",
        "energy_label",
    ]
    cols = [col for col in preferred if col in df.columns]
    return df[cols].copy() if cols else df.copy()


def representative_buildings(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or "annual_heat_kwh" not in df.columns:
        return None
    work = df.copy().sort_values("annual_heat_kwh").reset_index(drop=True)
    if len(work) == 0:
        return None
    idx_min = 0
    idx_max = len(work) - 1
    idx_mid = int((work["annual_heat_kwh"] - work["annual_heat_kwh"].median()).abs().idxmin())
    rows = [
        {"role": "minimum", **work.iloc[idx_min].to_dict()},
        {"role": "middle", **work.iloc[idx_mid].to_dict()},
        {"role": "maximum", **work.iloc[idx_max].to_dict()},
    ]
    return pd.DataFrame(rows)


def show_metric_row(items: list[tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        display_value = "-" if value is None else value
        col.metric(label, display_value)


def collect_options(
    dfs: list[pd.DataFrame | None],
    column: str,
) -> list[Any]:
    values: set[Any] = set()
    for df in dfs:
        if df is not None and column in df.columns:
            for val in df[column].dropna().tolist():
                values.add(val)
    return sorted(values, key=lambda x: str(x))


def plot_distribution(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        st.info(f"Column `{column}` not available.")
        return
    fig = px.histogram(
        df,
        x=column,
        color="case_name" if "case_name" in df.columns else None,
        marginal="box",
        nbins=30,
        title=f"Distribution of {column}",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_real_vs_generated_rca(
    real_df: pd.DataFrame | None,
    generated_df: pd.DataFrame | None,
) -> None:
    if real_df is None or generated_df is None or len(real_df) == 0 or len(generated_df) == 0:
        st.info("Need both real input rows and generated rows to draw the RCA scatter.")
        return

    real_cols = {"R", "C", "A"}
    gen_cols = {"R", "C", "A"}
    if not real_cols.issubset(real_df.columns) or not gen_cols.issubset(generated_df.columns):
        st.info("R/C/A columns not available for the real-vs-generated RCA scatter.")
        return

    real_plot = real_df.copy()
    real_plot["source_group"] = "real_input_pool"
    gen_plot = generated_df.copy()
    gen_plot["source_group"] = "generated"
    plot_df = pd.concat([real_plot, gen_plot], ignore_index=True, sort=False)

    fig = px.scatter(
        plot_df,
        x="R",
        y="C",
        color="source_group",
        size="A" if "A" in plot_df.columns else None,
        hover_data=[c for c in ["case_name", "building_id", "A", "Qint", "EnergyLabel", "energy_label"] if c in plot_df.columns],
        title="Real vs Generated RCA Distribution",
        opacity=0.75,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_energy_profile(df: pd.DataFrame | None) -> None:
    if df is None or len(df) == 0:
        st.info("No 24h profile data available for the current selection.")
        return

    time_col = detect_first_column(df, ["time", "datetime", "timestamp"])
    y_col = detect_first_column(
        df,
        ["heat_kw", "heating_power_kw", "p_heat_kw", "power_kw", "hp_power_kw"],
    )
    if time_col is None or y_col is None:
        st.info("Could not detect time or heating-power columns in selected_24h_profiles.csv.")
        return

    plot_df = df.copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[time_col])

    fig = px.line(
        plot_df,
        x=time_col,
        y=y_col,
        color="role" if "role" in plot_df.columns else None,
        facet_col="weather_kind" if "weather_kind" in plot_df.columns and plot_df["weather_kind"].nunique() > 1 else None,
        title="Selected 24h Heating Power Profiles",
    )
    fig.update_layout(legend_title_text="role")
    st.plotly_chart(fig, use_container_width=True)


def plot_pit_histogram(df: pd.DataFrame | None) -> None:
    if df is None or "pit" not in df.columns:
        st.info("No PIT samples available.")
        return
    fig = px.histogram(
        df,
        x="pit",
        color="variable" if "variable" in df.columns else None,
        nbins=20,
        barmode="overlay",
        title="PIT Histogram",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_status_counts(df: pd.DataFrame | None, title: str) -> None:
    if df is None or len(df) == 0:
        return
    for col in ["status", "check_status", "sanity_pass"]:
        if col in df.columns:
            counts = df[col].astype(str).value_counts(dropna=False).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", title=title)
            st.plotly_chart(fig, use_container_width=True)
            return


st.title("Community Thermal Profile Generation Demo")

st.sidebar.header("Demo Controls")
run_dir_input = st.sidebar.text_input("Demo run folder", str(DEFAULT_RUN_DIR))
run_dir = Path(run_dir_input)

input_config = load_json(run_dir / "input_config.json") or {}
manifest = load_json(run_dir / "manifest.json") or {}

building_inputs_df = load_csv(run_dir / "inputs" / "building_inputs.csv")
generated_rcaq_df = load_csv(run_dir / "results" / "generated_rcaq.csv")
candidate_scores_df = load_csv(run_dir / "results" / "candidate_scores.csv")
case_size_summary_df = load_csv(run_dir / "results" / "case_size_summary.csv")
case_size_detail_df = load_csv(run_dir / "results" / "case_size_detail.csv")
annual_building_df = load_csv(run_dir / "results" / "annual_building_summary.csv")
annual_community_df = load_csv(run_dir / "results" / "annual_community_summary.csv")
profiles_df = load_csv(run_dir / "results" / "selected_24h_profiles.csv")

validation_summary_df = load_csv(run_dir / "validation" / "validation_summary.csv")
sanity_check_df = load_csv(run_dir / "validation" / "sanity_check.csv")
conditional_validation_df = load_csv(run_dir / "validation" / "conditional_validation.csv")
monotonicity_df = load_csv(run_dir / "validation" / "monotonicity_validation.csv")
pit_validation_df = load_csv(run_dir / "validation" / "pit_validation.csv")
case_validation_df = load_csv(run_dir / "validation" / "case_validation.csv")

all_dfs = [
    building_inputs_df,
    generated_rcaq_df,
    candidate_scores_df,
    case_size_summary_df,
    case_size_detail_df,
    annual_building_df,
    annual_community_df,
    profiles_df,
    sanity_check_df,
    conditional_validation_df,
    monotonicity_df,
    pit_validation_df,
    case_validation_df,
]

community_size_options = collect_options(all_dfs, "community_size")
case_name_options = collect_options(all_dfs, "case_name")
weather_kind_options = collect_options(all_dfs, "weather_kind")

selected_community_size = st.sidebar.selectbox(
    "community_size",
    options=[None] + community_size_options,
    format_func=lambda x: "All" if x is None else str(x),
)
selected_case_name = st.sidebar.selectbox(
    "case_name",
    options=[None] + case_name_options,
    format_func=lambda x: "All" if x is None else str(x),
)
selected_weather_kind = st.sidebar.selectbox(
    "weather_kind",
    options=[None] + weather_kind_options,
    format_func=lambda x: "All" if x is None else str(x),
)

tab_input, tab_results, tab_profiles, tab_validation = st.tabs(
    ["Input Setup", "Generation Results", "Energy Profiles", "Validation"]
)

with tab_input:
    st.subheader("Run Settings")
    show_metric_row(
        [
            ("analysis_mode", input_config.get("analysis_mode")),
            ("community_size", input_config.get("community_size_or_building_number")),
            ("n_cases_per_size", input_config.get("n_cases_per_size")),
            ("candidate_number", input_config.get("candidate_number")),
        ]
    )
    show_metric_row(
        [
            ("top_k_proxy", input_config.get("top_k_proxy")),
            ("top_k_full", input_config.get("top_k_full")),
            ("weather_year", input_config.get("weather_year")),
            ("sunny_day", (input_config.get("selected_weather_days") or {}).get("sunny")),
        ]
    )
    show_metric_row(
        [
            ("cloudy_day", (input_config.get("selected_weather_days") or {}).get("cloudy")),
            ("run_sanity_check", input_config.get("run_sanity_check")),
            ("run_sys_validation", input_config.get("run_sys_validation")),
            ("run_case_validation", input_config.get("run_case_validation")),
        ]
    )

    with st.expander("Important Hyperparameters", expanded=False):
        st.json(input_config.get("important_hyperparameters", {}))

    st.subheader("Building Inputs")
    building_inputs_filtered = filter_df(
        building_inputs_df,
        community_size=selected_community_size,
        case_name=selected_case_name,
    )
    if building_inputs_filtered is None:
        st.info("building_inputs.csv not available.")
    else:
        st.dataframe(compact_building_input_view(building_inputs_filtered), use_container_width=True)
        df_download(building_inputs_filtered, "Download building_inputs.csv", "building_inputs.csv")

with tab_results:
    st.subheader("Generated Building Profiles")
    real_inputs_filtered = filter_df(
        building_inputs_df,
        community_size=selected_community_size,
        case_name=selected_case_name,
    )
    generated_filtered = filter_df(
        generated_rcaq_df,
        community_size=selected_community_size,
        case_name=selected_case_name,
    )
    if generated_filtered is None:
        st.info("generated_rcaq.csv not available.")
    else:
        st.dataframe(generated_filtered, use_container_width=True)
        df_download(generated_filtered, "Download generated_rcaq.csv", "generated_rcaq.csv")

        plot_cols = numeric_plot_columns(generated_filtered)
        if plot_cols:
            selected_plot_col = st.selectbox("Distribution column", plot_cols, key="dist_col")
            plot_distribution(generated_filtered, selected_plot_col)
        else:
            st.info("No numeric distribution columns detected for generated_rcaq.csv.")

    st.subheader("Real vs Generated RCA Scatter")
    plot_real_vs_generated_rca(real_inputs_filtered, generated_filtered)

    st.subheader("Community-Level Results")
    col1, col2 = st.columns(2)
    with col1:
        if case_size_summary_df is not None:
            summary_filtered = filter_df(case_size_summary_df, community_size=selected_community_size)
            st.dataframe(summary_filtered, use_container_width=True)
            df_download(summary_filtered, "Download case_size_summary.csv", "case_size_summary.csv")
        else:
            st.info("case_size_summary.csv not available.")
    with col2:
        if case_size_detail_df is not None:
            detail_filtered = filter_df(
                case_size_detail_df,
                community_size=selected_community_size,
                case_name=selected_case_name,
            )
            st.dataframe(detail_filtered, use_container_width=True)
            df_download(detail_filtered, "Download case_size_detail.csv", "case_size_detail.csv")
        else:
            st.info("case_size_detail.csv not available.")

    st.subheader("Annual Energy Summaries")
    col3, col4 = st.columns(2)
    with col3:
        annual_building_filtered = filter_df(
            annual_building_df,
            community_size=selected_community_size,
            case_name=selected_case_name,
        )
        if annual_building_filtered is not None:
            st.dataframe(annual_building_filtered, use_container_width=True)
            df_download(
                annual_building_filtered,
                "Download annual_building_summary.csv",
                "annual_building_summary.csv",
            )
        else:
            st.info("annual_building_summary.csv not available.")
    with col4:
        annual_community_filtered = filter_df(
            annual_community_df,
            community_size=selected_community_size,
            case_name=selected_case_name,
        )
        if annual_community_filtered is not None:
            st.dataframe(annual_community_filtered, use_container_width=True)
            df_download(
                annual_community_filtered,
                "Download annual_community_summary.csv",
                "annual_community_summary.csv",
            )
        else:
            st.info("annual_community_summary.csv not available.")

    with st.expander("Candidate Scores", expanded=False):
        candidate_filtered = filter_df(
            candidate_scores_df,
            community_size=selected_community_size,
            case_name=selected_case_name,
        )
        if candidate_filtered is not None:
            st.dataframe(candidate_filtered, use_container_width=True)
            df_download(candidate_filtered, "Download candidate_scores.csv", "candidate_scores.csv")
        else:
            st.info("candidate_scores.csv not available.")

with tab_profiles:
    st.subheader("24h Heating Power Profiles")
    annual_building_filtered = filter_df(
        annual_building_df,
        community_size=selected_community_size,
        case_name=selected_case_name,
    )
    representative_df = representative_buildings(annual_building_filtered)

    selected_role = None
    if representative_df is not None and len(representative_df):
        st.markdown("**Selected exported buildings for profile display**")
        st.dataframe(
            representative_df[
                [col for col in ["role", "building_id", "Area", "EnergyLabel", "annual_heat_kwh", "annual_heat_kwh_per_m2", "peak_heat_kw"] if col in representative_df.columns]
            ],
            use_container_width=True,
        )
        role_options = representative_df["role"].astype(str).tolist()
        selected_role = st.selectbox(
            "Select exported result",
            options=["All selected roles"] + role_options,
            index=0,
        )
    else:
        st.info("Representative minimum/middle/maximum building summary is not available for the current selection.")

    profiles_filtered = filter_df(
        profiles_df,
        community_size=selected_community_size,
        case_name=selected_case_name,
        weather_kind=selected_weather_kind,
    )
    if (
        profiles_filtered is not None
        and selected_role is not None
        and selected_role != "All selected roles"
        and "role" in profiles_filtered.columns
    ):
        profiles_filtered = profiles_filtered[
            profiles_filtered["role"].astype(str) == str(selected_role)
        ].copy()

    plot_energy_profile(profiles_filtered)
    if profiles_filtered is not None:
        st.dataframe(profiles_filtered, use_container_width=True)
        df_download(
            profiles_filtered,
            "Download selected_24h_profiles.csv",
            "selected_24h_profiles.csv",
        )

with tab_validation:
    st.subheader("Validation Summary")
    if validation_summary_df is not None:
        st.dataframe(validation_summary_df, use_container_width=True)
        df_download(
            validation_summary_df,
            "Download validation_summary.csv",
            "validation_summary.csv",
        )
    else:
        st.info("validation_summary.csv not available.")

    st.subheader("Validation Charts")
    plot_pit_histogram(pit_validation_df)
    plot_status_counts(sanity_check_df, "Sanity Check Status Counts")
    plot_status_counts(case_validation_df, "Case Validation Status Counts")

    validation_tables = [
        ("Sanity Check", sanity_check_df, "sanity_check.csv"),
        ("Conditional Validation", conditional_validation_df, "conditional_validation.csv"),
        ("Monotonicity Validation", monotonicity_df, "monotonicity_validation.csv"),
        ("PIT Validation", pit_validation_df, "pit_validation.csv"),
        ("Case Validation", case_validation_df, "case_validation.csv"),
    ]
    for title, df, filename in validation_tables:
        with st.expander(title, expanded=False):
            if df is None:
                st.info(f"{filename} not available.")
            else:
                st.dataframe(df, use_container_width=True)
                df_download(df, f"Download {filename}", filename)

with st.sidebar.expander("Manifest", expanded=False):
    if manifest:
        st.json(
            {
                "export_root": manifest.get("export_root"),
                "saved_files": len(manifest.get("saved_files", [])),
                "missing_items": len(manifest.get("missing_items", [])),
            }
        )
    else:
        st.info("manifest.json not available.")
