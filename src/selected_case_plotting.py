"""Selected-case community diagnostic plots.

This module contains presentation-only plotting utilities for post-hoc
selected-case diagnostics. It does not participate in synthesis, scoring, or
validation calculations.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


mpl.rcParams["svg.fonttype"] = "none"

PLOT_COLORS = {
    "reference": "#1F4E79",
    "generated": "#D95F02",
    "train": "#4C78A8",
    "holdout": "#7A5195",
    "neutral": "#6B7280",
    "grid": "#E5E7EB",
    "error": "#B23A48",
}

DEFAULT_CSV_PATHS = {
    "annual": "validation_outputs/selected_case_annual_energy_aggregate.csv",
    "profiles": "validation_outputs/selected_case_typical_day_community_profiles.csv",
    "metrics": "validation_outputs/selected_case_typical_day_community_metrics.csv",
    "ramp": "validation_outputs/selected_case_community_ramp_metrics.csv",
    "truth_comparison": "validation_outputs/selected_case_truth_comparison.csv",
    "generated": "validation_outputs/selected_case_generated_outputs.csv",
}

TimingCallback = Callable[[str, float, str], None]


def apply_academic_business_plot_style() -> None:
    """Apply restrained, SVG-friendly style for selected-case figures."""
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "font.size": 11,
            "axes.edgecolor": PLOT_COLORS["neutral"],
        }
    )


def format_source_label(source_name: object) -> str:
    """Return an audience-facing dataset label."""
    value = str(source_name).strip().lower()
    return {"holdout": "Test set", "train": "Training set"}.get(
        value, str(source_name).replace("_", " ").title()
    )


def format_community_title_source(source_name: object) -> str:
    """Return an audience-facing community label."""
    value = str(source_name).strip().lower()
    return {"holdout": "Test community", "train": "Training community"}.get(
        value, f"{str(source_name).replace('_', ' ').title()} community"
    )


def format_curve_label(kind: object) -> str:
    """Return a clear legend label without implying measured demand."""
    value = str(kind).strip().lower()
    if value in {"reference", "truth", "true", "fitted"}:
        return "Reference from fitted RCAQ"
    if value in {"generated", "synthesized"}:
        return "Synthesized community"
    return str(kind).replace("_", " ").title()


def format_scenario_label(
    value: object,
    row: Optional[pd.Series] = None,
    index: Optional[int] = None,
) -> str:
    """Return a reader-friendly representative-day label."""
    known_labels = {
        "cold_dark": "Cold winter day",
        "cold_sunny": "Cold sunny winter day",
        "mild_dark": "Mild cloudy day",
        "mild_sunny": "Mild sunny day",
        "extreme_cold": "Extreme cold day",
    }
    if row is not None and "scenario_label" in row.index and pd.notna(row["scenario_label"]):
        scenario_label = str(row["scenario_label"]).strip()
        scenario_key = scenario_label.lower()
        if scenario_key in known_labels:
            return known_labels[scenario_key]
        if scenario_label and not scenario_key.startswith("scenario_"):
            return scenario_label.replace("_", " ").title()
    value_key = str(value).strip().lower()
    if value_key in known_labels:
        return known_labels[value_key]
    if row is not None:
        for column in ("date", "datetime", "representative_date"):
            if column in row.index:
                timestamp = pd.to_datetime(row[column], errors="coerce")
                if pd.notna(timestamp):
                    seasons = {
                        12: "Winter representative day", 1: "Winter representative day", 2: "Winter representative day",
                        3: "Spring representative day", 4: "Spring representative day", 5: "Spring representative day",
                        6: "Summer representative day", 7: "Summer representative day", 8: "Summer representative day",
                        9: "Autumn representative day", 10: "Autumn representative day", 11: "Autumn representative day",
                    }
                    return seasons[int(timestamp.month)]
    return f"Representative day {(index if index is not None else 0) + 1}"


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next(
        (column for column in candidates if isinstance(df, pd.DataFrame) and column in df.columns),
        None,
    )


def _safe_pct(numerator: float, denominator: float) -> float:
    if pd.notna(denominator) and abs(float(denominator)) > 1e-9:
        return 100.0 * numerator / denominator
    return np.nan


def _load_plot_frame(
    supplied_df: Optional[pd.DataFrame],
    csv_path: Optional[str],
    label: str,
) -> pd.DataFrame:
    if isinstance(supplied_df, pd.DataFrame) and len(supplied_df):
        return supplied_df.copy()
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return pd.read_csv(path)
    warnings.warn(
        f"Plot input unavailable for {label}: no non-empty DataFrame or readable CSV was provided."
    )
    return pd.DataFrame()


def load_selected_case_plot_data(
    *,
    annual_df: Optional[pd.DataFrame] = None,
    profile_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    ramp_df: Optional[pd.DataFrame] = None,
    truth_comparison_df: Optional[pd.DataFrame] = None,
    generated_df: Optional[pd.DataFrame] = None,
    csv_paths: Optional[Mapping[str, str]] = None,
) -> dict[str, pd.DataFrame]:
    """Load selected-case plot sources from supplied frames or CSV fallbacks."""
    paths = dict(DEFAULT_CSV_PATHS)
    if csv_paths:
        paths.update(csv_paths)
    return {
        "annual": _load_plot_frame(annual_df, paths.get("annual"), "annual diagnostics"),
        "profiles": _load_plot_frame(profile_df, paths.get("profiles"), "typical-day profiles"),
        "metrics": _load_plot_frame(metrics_df, paths.get("metrics"), "dynamic metrics"),
        "ramp": _load_plot_frame(ramp_df, paths.get("ramp"), "ramp metrics"),
        "truth_comparison": _load_plot_frame(
            truth_comparison_df, paths.get("truth_comparison"), "truth comparison"
        ),
        "generated": _load_plot_frame(generated_df, paths.get("generated"), "generated outputs"),
    }


def select_best_selected_case_rank(
    annual_df: pd.DataFrame,
    source_name: str = "holdout",
    rank_mode: str = "best_annual_energy",
) -> Optional[dict[str, object]]:
    """Return the selected case/rank descriptor for a source."""
    if not isinstance(annual_df, pd.DataFrame) or len(annual_df) == 0:
        return None
    work = annual_df.copy()
    if "source_name" in work.columns:
        work = work.loc[work["source_name"].astype(str) == str(source_name)]
    if len(work) == 0:
        return None
    error_col = _first_existing_col(work, ["absolute_error_pct", "signed_error_pct"])
    if rank_mode == "best_annual_energy" and error_col:
        errors = pd.to_numeric(work[error_col], errors="coerce").abs()
        selected = work.loc[errors.idxmin()] if errors.notna().any() else work.iloc[0]
    else:
        selected = work.iloc[0]
    return {
        key: selected[key]
        for key in ("case_id", "source_name", "selected_rank")
        if key in selected.index
    }


def _filter_selected_case_rows(
    df: pd.DataFrame, selection: Optional[Mapping[str, object]]
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or not selection:
        return pd.DataFrame()
    work = df.copy()
    for column, value in selection.items():
        if column in work.columns:
            work = work.loc[work[column].astype(str) == str(value)]
    return work


def save_current_figure(
    fig,
    figure_dir: str = "validation_outputs/figures",
    basename: str = "selected_case_figure.svg",
    formats: Sequence[str] = ("svg",),
    dpi: int = 300,
) -> list[str]:
    """Save a figure in the permitted SVG format and return saved paths."""
    figure_path = Path(figure_dir)
    figure_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for output_format in formats:
        if str(output_format).lower() != "svg":
            warnings.warn(
                f"Skipping unsupported plot output format for this task: {output_format!r}."
            )
            continue
        filename = basename if basename.lower().endswith(".svg") else f"{basename}.svg"
        target = figure_path / filename
        fig.savefig(target, format="svg", dpi=dpi, bbox_inches="tight")
        saved_paths.append(str(target))
        print("Saved SVG figure:", target)
    return saved_paths


def _infer_profile_columns(profile_df: pd.DataFrame) -> dict[str, Optional[str]]:
    return {
        "time": _first_existing_col(profile_df, ["datetime", "time", "timestep_index"]),
        "day": _first_existing_col(profile_df, ["typical_day_id", "scenario_id", "day_id"]),
        "reference_heat": _first_existing_col(
            profile_df,
            [
                "true_community_heat_kw",
                "reference_community_heat_kw",
                "fitted_rcaq_reference_heat_kw",
                "truth_heat_kw",
            ],
        ),
        "generated_heat": _first_existing_col(
            profile_df, ["generated_community_heat_kw", "generated_heat_kw"]
        ),
        "reference_ramp": _first_existing_col(
            profile_df,
            ["true_community_ramp_kw", "reference_community_ramp_kw", "truth_ramp_kw"],
        ),
        "generated_ramp": _first_existing_col(
            profile_df, ["generated_community_ramp_kw", "generated_ramp_kw"]
        ),
    }


def _mean_numeric(df: pd.DataFrame, column: str, absolute: bool = False) -> float:
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.abs().mean() if absolute else values.mean())


def _mean_peak_absolute_error_pct(df: pd.DataFrame) -> float:
    if not isinstance(df, pd.DataFrame) or not {"peak_kw_error", "true_peak_kw"}.issubset(df.columns):
        return np.nan
    peak_error = pd.to_numeric(df["peak_kw_error"], errors="coerce").abs()
    true_peak = pd.to_numeric(df["true_peak_kw"], errors="coerce").replace(0, np.nan)
    return float((100.0 * peak_error / true_peak).mean())


def _add_available_metric(
    summary_values: list[tuple[str, float, str]],
    *,
    label: str,
    value: float,
    unit: str,
    missing_message: str,
) -> None:
    if pd.notna(value):
        summary_values.append((label, value, unit))
    else:
        warnings.warn(missing_message)


def _record_timing(
    callback: Optional[TimingCallback],
    section: str,
    start_time: float,
    note: str,
) -> None:
    if callback is not None:
        callback(section, time.perf_counter() - start_time, note)


def _finish_figure(
    fig,
    *,
    basename: str,
    figure_dir: str,
    formats: Sequence[str],
    dpi: int,
    save_figures: bool,
    show_figures: bool,
) -> list[str]:
    paths = (
        save_current_figure(fig, figure_dir, basename, formats=formats, dpi=dpi)
        if save_figures
        else []
    )
    if show_figures:
        plt.show()
    plt.close(fig)
    return paths


def plot_holdout_annual_energy_total(
    annual_df: pd.DataFrame,
    selection: Optional[Mapping[str, object]],
    *,
    figure_dir: str,
    formats: Sequence[str],
    dpi: int,
    save_figures: bool,
    show_figures: bool,
) -> dict[str, object]:
    annual_choice = _filter_selected_case_rows(annual_df, selection)
    reference_col = _first_existing_col(
        annual_choice, ["true_total_annual_kwh", "reference_total_annual_kwh"]
    )
    generated_col = _first_existing_col(annual_choice, ["generated_total_annual_kwh"])
    if not (len(annual_choice) and reference_col and generated_col):
        warnings.warn("Test community annual energy figure skipped: required aggregate columns are unavailable.")
        return {"figure": "holdout_annual_energy_total.svg", "status": "skipped"}
    row = annual_choice.iloc[0]
    source_label = format_community_title_source(row.get("source_name", "holdout"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    values = [float(row[reference_col]), float(row[generated_col])]
    bars = ax.bar(
        [format_curve_label("reference"), format_curve_label("generated")],
        values,
        color=[PLOT_COLORS["reference"], PLOT_COLORS["generated"]],
        width=0.56,
    )
    ax.set_title(f"{source_label.title()} Annual Heating Energy")
    ax.set_ylabel("Annual heating energy [kWh]")
    ax.grid(axis="y", color=PLOT_COLORS["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=11)
    signed = row.get("signed_error_pct", np.nan)
    absolute = row.get("absolute_error_pct", np.nan)
    n_buildings = row.get("n_buildings", np.nan)
    n_label = f"{int(n_buildings)}" if pd.notna(n_buildings) else "n/a"
    ax.text(
        0.98,
        0.96,
        f"Signed error: {signed:.1f}%\nAbsolute error: {absolute:.1f}%\nBuildings: {n_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=PLOT_COLORS["neutral"],
    )
    paths = _finish_figure(
        fig,
        basename="holdout_annual_energy_total.svg",
        figure_dir=figure_dir,
        formats=formats,
        dpi=dpi,
        save_figures=save_figures,
        show_figures=show_figures,
    )
    return {"figure": "holdout_annual_energy_total.svg", "status": "created", "paths": paths}


def _profile_plot_inputs(
    profile_df: pd.DataFrame,
    selection: Optional[Mapping[str, object]],
    max_typical_days: int,
) -> tuple[pd.DataFrame, dict[str, Optional[str]], list[object]]:
    profile_choice = _filter_selected_case_rows(profile_df, selection)
    columns = _infer_profile_columns(profile_choice)
    days = (
        list(profile_choice[columns["day"]].dropna().drop_duplicates().iloc[:max_typical_days])
        if columns["day"]
        else []
    )
    return profile_choice, columns, days


def _profile_hour_axis(day_df: pd.DataFrame, time_column: Optional[str]) -> pd.Series:
    if time_column is None:
        return pd.Series(np.arange(len(day_df)), index=day_df.index)
    timestamp_values = pd.to_datetime(day_df[time_column], errors="coerce")
    if timestamp_values.notna().any():
        return timestamp_values.dt.hour + timestamp_values.dt.minute / 60.0
    return pd.to_numeric(day_df[time_column], errors="coerce")


def _infer_ramp_unit(profile_df: pd.DataFrame) -> str:
    time_column = _infer_profile_columns(profile_df)["time"]
    if time_column is not None:
        timestamps = pd.to_datetime(profile_df[time_column], errors="coerce").dropna().sort_values()
        deltas = timestamps.drop_duplicates().diff().dropna().dt.total_seconds()
        if len(deltas) and np.isclose(float(deltas.median()), 15.0 * 60.0):
            return "kW/15 min"
    return "kW/timestep"


def plot_holdout_typical_day_heat_profile(
    profile_df: pd.DataFrame,
    selection: Optional[Mapping[str, object]],
    *,
    max_typical_days: int,
    figure_dir: str,
    formats: Sequence[str],
    dpi: int,
    save_figures: bool,
    show_figures: bool,
) -> dict[str, object]:
    choice, columns, days = _profile_plot_inputs(profile_df, selection, max_typical_days)
    if not (len(choice) and days and columns["reference_heat"] and columns["generated_heat"]):
        warnings.warn("Test community typical-day heat profile figure skipped: required profile columns are unavailable.")
        return {"figure": "holdout_typical_day_heat_profile.svg", "status": "skipped"}
    fig, axes = plt.subplots(1, len(days), figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for index, (ax, day) in enumerate(zip(axes, days)):
        day_df = choice.loc[choice[columns["day"]] == day].copy()
        hour = _profile_hour_axis(day_df, columns["time"])
        ax.plot(hour, day_df[columns["reference_heat"]], color=PLOT_COLORS["reference"], linewidth=2.4, label=format_curve_label("reference"))
        ax.plot(hour, day_df[columns["generated_heat"]], color=PLOT_COLORS["generated"], linewidth=2.2, linestyle="--", label=format_curve_label("generated"))
        weight = day_df["typical_day_weight"].iloc[0] if "typical_day_weight" in day_df.columns else np.nan
        weight_label = f" (w={weight:.3f})" if pd.notna(weight) else ""
        ax.set_title(f"{format_scenario_label(day, day_df.iloc[0], index)}{weight_label}", fontsize=13)
        ax.set_xlabel("Hour of day")
        ax.grid(axis="y", color=PLOT_COLORS["grid"], linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Community heat demand [kW]")
    fig.suptitle("Test Community Typical-Day Heat Demand", fontsize=16)
    fig.legend(
        handles=[
            Line2D([0], [0], color=PLOT_COLORS["reference"], linewidth=2.4, label=format_curve_label("reference")),
            Line2D([0], [0], color=PLOT_COLORS["generated"], linewidth=2.2, linestyle="--", label=format_curve_label("generated")),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.90),
    )
    paths = _finish_figure(
        fig,
        basename="holdout_typical_day_heat_profile.svg",
        figure_dir=figure_dir,
        formats=formats,
        dpi=dpi,
        save_figures=save_figures,
        show_figures=show_figures,
    )
    return {"figure": "holdout_typical_day_heat_profile.svg", "status": "created", "paths": paths}


def plot_holdout_typical_day_ramp_profile(
    profile_df: pd.DataFrame,
    selection: Optional[Mapping[str, object]],
    *,
    max_typical_days: int,
    figure_dir: str,
    formats: Sequence[str],
    dpi: int,
    save_figures: bool,
    show_figures: bool,
    ramp_unit: str,
) -> dict[str, object]:
    choice, columns, days = _profile_plot_inputs(profile_df, selection, max_typical_days)
    if not (len(choice) and days and columns["reference_ramp"] and columns["generated_ramp"]):
        warnings.warn("Test community ramp profile figure skipped: required ramp columns are unavailable.")
        return {"figure": "holdout_typical_day_ramp_profile.svg", "status": "skipped"}
    fig, axes = plt.subplots(1, len(days), figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for index, (ax, day) in enumerate(zip(axes, days)):
        day_df = choice.loc[choice[columns["day"]] == day].copy()
        hour = _profile_hour_axis(day_df, columns["time"])
        ax.plot(hour, day_df[columns["reference_ramp"]], color=PLOT_COLORS["reference"], linewidth=2.1, label=format_curve_label("reference"))
        ax.plot(hour, day_df[columns["generated_ramp"]], color=PLOT_COLORS["generated"], linewidth=2.0, linestyle="--", label=format_curve_label("generated"))
        ax.axhline(0, color=PLOT_COLORS["grid"], linewidth=1.0)
        ax.set_title(format_scenario_label(day, day_df.iloc[0], index), fontsize=13)
        ax.set_xlabel("Hour of day")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"Community ramp [{ramp_unit}]")
    fig.suptitle("Test Community Typical-Day Ramp", fontsize=16)
    fig.legend(
        handles=[
            Line2D([0], [0], color=PLOT_COLORS["reference"], linewidth=2.2, label=format_curve_label("reference")),
            Line2D([0], [0], color=PLOT_COLORS["generated"], linewidth=2.1, linestyle="--", label=format_curve_label("generated")),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.90),
    )
    paths = _finish_figure(
        fig,
        basename="holdout_typical_day_ramp_profile.svg",
        figure_dir=figure_dir,
        formats=formats,
        dpi=dpi,
        save_figures=save_figures,
        show_figures=show_figures,
    )
    return {"figure": "holdout_typical_day_ramp_profile.svg", "status": "created", "paths": paths}


def plot_holdout_dynamic_metric_summary(
    annual_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selection: Optional[Mapping[str, object]],
    *,
    ramp_df: Optional[pd.DataFrame],
    ramp_unit: str,
    figure_dir: str,
    formats: Sequence[str],
    dpi: int,
    save_figures: bool,
    show_figures: bool,
) -> dict[str, object]:
    annual_choice = _filter_selected_case_rows(annual_df, selection)
    metrics_choice = _filter_selected_case_rows(metrics_df, selection)
    ramp_choice = _filter_selected_case_rows(ramp_df, selection) if isinstance(ramp_df, pd.DataFrame) else pd.DataFrame()
    if not len(annual_choice):
        warnings.warn("Test community diagnostic summary skipped: annual metrics are unavailable.")
        return {"figure": "holdout_dynamic_metric_summary.svg", "status": "skipped"}
    row = annual_choice.iloc[0]
    annual_error_column = _first_existing_col(annual_choice, ["absolute_error_pct", "signed_error_pct"])
    annual_error_label = "Annual energy error" if annual_error_column == "absolute_error_pct" else "Annual signed energy error"
    ramp_column = _first_existing_col(
        ramp_choice,
        ["weighted_ramp_rmse", "ramp_rmse_kw_per_step", "ramp_rmse", "generated_ramp_rmse_residual_mean"],
    )
    metrics_column = _first_existing_col(
        metrics_choice,
        ["weighted_ramp_rmse", "ramp_rmse_kw_per_step", "ramp_rmse", "generated_ramp_rmse_residual_mean"],
    )
    dynamic_source, dynamic_column = (
        (ramp_choice, ramp_column) if ramp_column else (metrics_choice, metrics_column)
    )
    summary_values: list[tuple[str, float, str]] = []
    _add_available_metric(
        summary_values,
        label=annual_error_label,
        value=row.get(annual_error_column, np.nan) if annual_error_column else np.nan,
        unit="%",
        missing_message="Test community summary: annual energy error metric is unavailable.",
    )
    daily_absolute_column = _first_existing_col(
        metrics_choice,
        ["absolute_day_kwh_error_pct", "abs_day_kwh_error_pct", "day_kwh_absolute_error_pct"],
    )
    daily_signed_column = _first_existing_col(
        metrics_choice, ["signed_day_kwh_error_pct", "day_kwh_error_pct"]
    )
    daily_column = daily_absolute_column or daily_signed_column
    _add_available_metric(
        summary_values,
        label="Typical-day energy error" if daily_absolute_column else "Typical-day signed energy error",
        value=_mean_numeric(metrics_choice, daily_column) if daily_column else np.nan,
        unit="%",
        missing_message="Test community summary: typical-day energy error metric is unavailable.",
    )
    peak_pct_column = _first_existing_col(
        metrics_choice,
        ["peak_absolute_error_pct", "abs_peak_kw_error_pct", "peak_kw_error_pct"],
    )
    peak_value = (
        _mean_numeric(metrics_choice, peak_pct_column)
        if peak_pct_column
        else _mean_peak_absolute_error_pct(metrics_choice)
    )
    _add_available_metric(
        summary_values,
        label="Peak error",
        value=peak_value,
        unit="%",
        missing_message="Test community summary: peak error metric is unavailable.",
    )
    _add_available_metric(
        summary_values,
        label="Dynamic error",
        value=_mean_numeric(dynamic_source, dynamic_column) if dynamic_column else np.nan,
        unit=ramp_unit,
        missing_message="Test community summary: ramp dynamic error metric is unavailable.",
    )
    profile_rmse_column = _first_existing_col(
        metrics_choice, ["profile_rmse_kw", "community_profile_rmse_kw", "profile_rmse"]
    )
    _add_available_metric(
        summary_values,
        label="Profile RMSE",
        value=_mean_numeric(metrics_choice, profile_rmse_column) if profile_rmse_column else np.nan,
        unit="kW",
        missing_message="Test community summary: profile RMSE metric is unavailable.",
    )
    correlation_column = _first_existing_col(
        metrics_choice, ["correlation", "profile_correlation", "community_profile_correlation"]
    )
    _add_available_metric(
        summary_values,
        label="Profile correlation",
        value=_mean_numeric(metrics_choice, correlation_column) if correlation_column else np.nan,
        unit="-",
        missing_message="Test community summary: profile correlation metric is unavailable.",
    )
    if not summary_values:
        warnings.warn("Test community diagnostic summary skipped: no displayable metrics are available.")
        return {"figure": "holdout_dynamic_metric_summary.svg", "status": "skipped"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axis("off")
    ax.set_title("Test Community Diagnostics", loc="left")
    for index, (label, value, unit) in enumerate(summary_values):
        y_pos = 0.83 - index * 0.105
        display_value = f"{value:.2f} {unit}" if pd.notna(value) else "n/a"
        ax.text(0.03, y_pos, label, fontsize=12, color=PLOT_COLORS["neutral"], transform=ax.transAxes)
        ax.text(0.96, y_pos, display_value, fontsize=14, fontweight="bold", color=PLOT_COLORS["reference"], ha="right", transform=ax.transAxes)
        if index < len(summary_values) - 1:
            ax.plot([0.03, 0.97], [y_pos - 0.04, y_pos - 0.04], color=PLOT_COLORS["grid"], transform=ax.transAxes)
    timestep_text = "15-min community ramp" if ramp_unit == "kW/15 min" else "community ramp between consecutive timesteps"
    fig.text(
        0.04,
        0.06,
        "Annual energy error (%) compares total annual heating energy between the synthesized community and the "
        "reference community from fitted RCAQ. Typical-day energy error (%) compares daily community heating energy.\n"
        f"Dynamic error [{ramp_unit}] is the RMSE of the {timestep_text}. Profile RMSE and correlation describe "
        "similarity of the community heat-demand curves.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=PLOT_COLORS["neutral"],
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    paths = _finish_figure(
        fig,
        basename="holdout_dynamic_metric_summary.svg",
        figure_dir=figure_dir,
        formats=formats,
        dpi=dpi,
        save_figures=save_figures,
        show_figures=show_figures,
    )
    return {"figure": "holdout_dynamic_metric_summary.svg", "status": "created", "paths": paths}


def plot_train_vs_holdout_community_summary(
    annual_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    rank_mode: str,
    ramp_unit: str,
    figure_dir: str,
    formats: Sequence[str],
    dpi: int,
    save_figures: bool,
    show_figures: bool,
) -> dict[str, object]:
    comparison_rows = []
    for source in ("train", "holdout"):
        selection = select_best_selected_case_rank(annual_df, source, rank_mode)
        annual_source = _filter_selected_case_rows(annual_df, selection)
        metrics_source = _filter_selected_case_rows(metrics_df, selection)
        if len(annual_source):
            annual_row = annual_source.iloc[0]
            comparison_rows.append(
                {
                    "source_name": source,
                    "annual_abs_error_pct": annual_row.get("absolute_error_pct", np.nan),
                    "typical_day_abs_error_pct": _mean_numeric(metrics_source, "signed_day_kwh_error_pct", absolute=True),
                    "profile_rmse_kw": _mean_numeric(metrics_source, "profile_rmse_kw"),
                    "ramp_rmse_kw_per_step": _mean_numeric(metrics_source, "ramp_rmse_kw_per_step"),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)
    if len(comparison_df) < 2:
        warnings.warn("Training-versus-test community summary skipped: both source summaries are unavailable.")
        return {"figure": "train_vs_holdout_community_summary.svg", "status": "skipped"}
    fig, (ax_pct, ax_kw) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = [PLOT_COLORS.get(source, PLOT_COLORS["neutral"]) for source in comparison_df["source_name"]]
    x_pos = np.arange(len(comparison_df))
    width = 0.34
    ax_pct.bar(x_pos - width / 2, comparison_df["annual_abs_error_pct"], width=width, color=colors, alpha=0.95, label="Annual energy")
    ax_pct.bar(x_pos + width / 2, comparison_df["typical_day_abs_error_pct"], width=width, color=colors, alpha=0.5, label="Typical-day energy")
    ax_pct.set_ylabel("Absolute error [%]")
    ax_pct.set_xticks(x_pos, [format_source_label(source) for source in comparison_df["source_name"]])
    ax_pct.set_title("Energy error")
    ax_pct.legend()
    ax_kw.bar(x_pos - width / 2, comparison_df["profile_rmse_kw"], width=width, color=colors, alpha=0.95, label="Profile RMSE")
    ax_kw.bar(x_pos + width / 2, comparison_df["ramp_rmse_kw_per_step"], width=width, color=colors, alpha=0.5, label="Ramp RMSE")
    ax_kw.set_ylabel(f"Error [kW; ramp in {ramp_unit}]")
    ax_kw.set_xticks(x_pos, [format_source_label(source) for source in comparison_df["source_name"]])
    ax_kw.set_title("Dynamic error")
    ax_kw.legend()
    for ax in (ax_pct, ax_kw):
        ax.grid(axis="y", color=PLOT_COLORS["grid"], linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Training Set vs Test Set Community Diagnostics", fontsize=16)
    paths = _finish_figure(
        fig,
        basename="train_vs_holdout_community_summary.svg",
        figure_dir=figure_dir,
        formats=formats,
        dpi=dpi,
        save_figures=save_figures,
        show_figures=show_figures,
    )
    return {"figure": "train_vs_holdout_community_summary.svg", "status": "created", "paths": paths}


def generate_selected_case_community_plots(
    *,
    annual_df: Optional[pd.DataFrame] = None,
    profile_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    ramp_df: Optional[pd.DataFrame] = None,
    truth_comparison_df: Optional[pd.DataFrame] = None,
    generated_df: Optional[pd.DataFrame] = None,
    figure_dir: str = "validation_outputs/figures",
    formats: Sequence[str] = ("svg",),
    source_name: str = "holdout",
    rank_mode: str = "best_annual_energy",
    max_typical_days: int = 3,
    dpi: int = 300,
    save_figures: bool = True,
    show_figures: bool = True,
    include_train_vs_test_summary: bool = False,
    csv_paths: Optional[Mapping[str, str]] = None,
    timing_callback: Optional[TimingCallback] = None,
) -> list[dict[str, object]]:
    """Create available selected-case community diagnostic SVG figures."""
    apply_academic_business_plot_style()
    start_time = time.perf_counter()
    data = load_selected_case_plot_data(
        annual_df=annual_df,
        profile_df=profile_df,
        metrics_df=metrics_df,
        ramp_df=ramp_df,
        truth_comparison_df=truth_comparison_df,
        generated_df=generated_df,
        csv_paths=csv_paths,
    )
    selection = select_best_selected_case_rank(data["annual"], source_name, rank_mode)
    if selection is None:
        warnings.warn("No selected annual-energy row available for the requested plot source; test-community figures will be skipped.")
    _record_timing(timing_callback, "selected_case_plot_load_sec", start_time, "load selected-case plot inputs")
    ramp_unit = _infer_ramp_unit(data["profiles"])

    outputs: list[dict[str, object]] = []
    figure_calls = [
        (
            "selected_case_plot_annual_energy_sec",
            "plot test community annual energy",
            lambda: plot_holdout_annual_energy_total(data["annual"], selection, figure_dir=figure_dir, formats=formats, dpi=dpi, save_figures=save_figures, show_figures=show_figures),
        ),
        (
            "selected_case_plot_typical_profile_sec",
            "plot test community typical-day heat profiles",
            lambda: plot_holdout_typical_day_heat_profile(data["profiles"], selection, max_typical_days=max_typical_days, figure_dir=figure_dir, formats=formats, dpi=dpi, save_figures=save_figures, show_figures=show_figures),
        ),
        (
            "selected_case_plot_ramp_profile_sec",
            "plot test community typical-day ramp profiles",
            lambda: plot_holdout_typical_day_ramp_profile(data["profiles"], selection, max_typical_days=max_typical_days, figure_dir=figure_dir, formats=formats, dpi=dpi, save_figures=save_figures, show_figures=show_figures, ramp_unit=ramp_unit),
        ),
        (
            "selected_case_plot_metric_summary_sec",
            "plot test community dynamic metric summary",
            lambda: plot_holdout_dynamic_metric_summary(data["annual"], data["metrics"], selection, ramp_df=data["ramp"], ramp_unit=ramp_unit, figure_dir=figure_dir, formats=formats, dpi=dpi, save_figures=save_figures, show_figures=show_figures),
        ),
    ]
    if include_train_vs_test_summary:
        figure_calls.append(
            (
                "selected_case_plot_train_holdout_summary_sec",
                "plot training set versus test set community summary",
                lambda: plot_train_vs_holdout_community_summary(data["annual"], data["metrics"], rank_mode=rank_mode, ramp_unit=ramp_unit, figure_dir=figure_dir, formats=formats, dpi=dpi, save_figures=save_figures, show_figures=show_figures),
            )
        )
    for section, note, plot_call in figure_calls:
        start_time = time.perf_counter()
        outputs.append(plot_call())
        _record_timing(timing_callback, section, start_time, note)
    return outputs
