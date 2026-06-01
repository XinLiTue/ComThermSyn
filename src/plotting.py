"""
Visualization helpers: RC parameter distributions, validation plots, profile charts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import LABEL_ORDER


# ---------------------------------------------------------------------------
# RC parameter comparison
# ---------------------------------------------------------------------------

def plot_real_vs_generated_rca(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    gen_name: str = "Generated",
    title_suffix: str = "",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot of R vs C coloured by A for real (grey) vs generated buildings."""
    fig = plt.figure(figsize=(8.5, 6.5))
    plt.scatter(
        real_df["R"], real_df["C"],
        s=25, alpha=0.25, label="Real holdout data", color="tab:gray",
    )
    sc = plt.scatter(
        generated_df["R"], generated_df["C"],
        c=generated_df["A"], s=60, cmap="viridis", alpha=0.9, label=gen_name,
    )
    plt.xlabel("R")
    plt.ylabel("C")
    plt.title(f"Real vs {gen_name} RCA distribution{title_suffix}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    cbar = plt.colorbar(sc)
    cbar.set_label("A")
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_real_vs_generated_rca_histograms(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Real vs Generated RC Parameters",
) -> plt.Figure:
    """Side-by-side histograms of R, C_per_area, A for real vs generated."""
    params = [c for c in ["R", "C_per_area", "A", "Qint"] if c in real_df.columns and c in generated_df.columns]
    if not params:
        raise ValueError("No common RC columns found in both DataFrames.")

    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, params):
        r_vals = pd.to_numeric(real_df[col], errors="coerce").dropna()
        g_vals = pd.to_numeric(generated_df[col], errors="coerce").dropna()
        lo = float(min(r_vals.quantile(0.01), g_vals.quantile(0.01)))
        hi = float(max(r_vals.quantile(0.99), g_vals.quantile(0.99)))
        bins = np.linspace(lo, hi, 30)
        ax.hist(r_vals, bins=bins, alpha=0.6, label="real", color="steelblue", density=True)
        ax.hist(g_vals, bins=bins, alpha=0.6, label="generated", color="tomato", density=True)
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_energy_by_label(
    df: pd.DataFrame,
    energy_col: str = "E_std_annual_per_m2",
    output_path: Optional[Path] = None,
    title: str = "Energy by Label",
) -> plt.Figure:
    """Boxplot of energy distribution per energy label."""
    if energy_col not in df.columns or "label_clean" not in df.columns:
        raise ValueError(f"DataFrame must have '{energy_col}' and 'label_clean' columns.")

    labels_present = [l for l in LABEL_ORDER if l in df["label_clean"].values]
    data = [pd.to_numeric(df.loc[df["label_clean"] == lbl, energy_col], errors="coerce").dropna().to_numpy() for lbl in labels_present]

    fig, ax = plt.subplots(figsize=(max(6, len(labels_present)), 5))
    ax.boxplot(data, labels=labels_present, patch_artist=True)
    ax.set_xlabel("Energy label")
    ax.set_ylabel(energy_col)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_case_size_wasserstein(
    case_detail_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Line plot of Wasserstein distances by community size."""
    if metrics is None:
        metrics = [c for c in ["wasserstein_E", "wasserstein_R", "wasserstein_Cpa"] if c in case_detail_df.columns]
    if not metrics:
        raise ValueError("No Wasserstein metric columns found.")

    sizes = sorted(case_detail_df["community_size"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metrics:
        means = [case_detail_df.loc[case_detail_df["community_size"] == s, metric].mean() for s in sizes]
        ax.plot(sizes, means, marker="o", label=metric)
    ax.set_xlabel("Community size")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title("Distribution quality vs. community size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_heating_profiles(
    profiles: Dict[str, pd.DataFrame],
    heat_col: str = "heat_kw",
    output_path: Optional[Path] = None,
    title: str = "Annual Heating Profiles",
    max_curves: int = 10,
) -> plt.Figure:
    """Overlay multiple heating profiles on a single axis."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, prof_df) in enumerate(profiles.items()):
        if i >= max_curves:
            break
        if heat_col in prof_df.columns:
            ax.plot(prof_df[heat_col].to_numpy(), alpha=0.5, linewidth=0.8, label=label)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Heat power (kW)")
    ax.set_title(title)
    if len(profiles) <= 8:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pit_histogram(
    detail_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of PIT values per variable; ideal is uniform [0,1]."""
    variables = detail_df["variable"].unique() if "variable" in detail_df.columns else []
    n = max(len(variables), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        pits = detail_df.loc[detail_df["variable"] == var, "pit"].dropna()
        ax.hist(pits, bins=20, range=(0, 1), density=True, color="steelblue", alpha=0.75)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="ideal uniform")
        ax.set_title(f"PIT – {var}")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("PIT validation (ideal: flat)", y=1.02)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
